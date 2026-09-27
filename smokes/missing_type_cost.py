"""What does a collaborator missing from the prompt actually cost?

For every teacher run in MAVerIT4/inference_records, each javac error is attributed to the
project type whose API it got wrong: the owner of a missing method, the class whose
constructor was misapplied, the exception left uncaught, a hallucinated type. Then, per
(run, class under test, collaborator) pair the model engaged with -- used in its final suite
or got an error about -- compare:

    P(at least one API error about T) when T WAS in the prompt  (v1 retrieval, as run)
    P(at least one API error about T) when T was NOT

The class under test itself is shown as a reference: its full source is always in the prompt.
Run from the harness root after smokes/retrieval_audit.py (it reads v1's retrieved sets).
"""
import collections
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smokes import retrieval_audit as A  # noqa: E402
from utils import parser  # noqa: E402

ERR = re.compile(r"\[ERROR\] \S+?\.java:\[\d+,\d+\] (?:error: )?(.*)")
SYMBOL = re.compile(r"symbol:\s+(class|variable|method|constructor)\s+([\w$]+)")
LOCATION = re.compile(r"location:\s+(?:class|interface|variable \w+ of type)\s+([\w.$<>]+)")
PATTERNS = [  # (regex, group holding the type) -- the type whose API was misused
    (re.compile(r"constructor \w+ in (?:class|enum) ([\w.$]+) cannot be applied"), 1),
    (re.compile(r"no suitable constructor found for ([\w$]+)\("), 1),
    (re.compile(r"method \w+ in (?:class|interface|enum) ([\w.$<>]+) cannot be applied"), 1),
    (re.compile(r"\w+(?:\([^)]*\))? has private access in ([\w.$]+)"), 1),
    (re.compile(r"\w+(?:\([^)]*\))? is not public in ([\w.$]+)"), 1),
    (re.compile(r"\w+(?:\([^)]*\))? has protected access in ([\w.$]+)"), 1),
    (re.compile(r"unreported exception ([\w.$]+)"), 1),
    (re.compile(r"incompatible types: \S+ cannot be converted to ([\w.$<>]+)"), 1),
    (re.compile(r"([\w.$]+) is abstract; cannot be instantiated"), 1),
]


def attributed_types(feedback: str) -> set[str]:
    """Fully-qualified or simple names of the types each distinct error is about."""
    out = set()
    lines = feedback.splitlines()
    for i, line in enumerate(lines):
        m = ERR.search(line)
        if not m:
            continue
        msg = m.group(1)
        if msg.startswith("cannot find symbol"):
            ctx = "\n".join(lines[i + 1:i + 3])
            sym, loc = SYMBOL.search(ctx), LOCATION.search(ctx)
            if sym and sym.group(1) in ("class", "variable") and sym.group(2)[:1].isupper():
                out.add(sym.group(2))                    # a type that does not exist as named
            elif loc and not loc.group(1).endswith("Test"):
                out.add(loc.group(1).split("<")[0])     # a member missing on this type
            continue
        for pattern, group in PATTERNS:
            p = pattern.search(msg)
            if p:
                out.add(p.group(group).split("<")[0])
                break
    return out


def main() -> int:
    rows = [json.loads(l) for l in open("smokes/retrieval-audit/rows.jsonl", encoding="utf-8")]
    v1 = {(r["project"], r["cut"]): set(r["retrieved"]) for r in rows if r["retriever"] == "v1"}
    stats = collections.defaultdict(lambda: collections.Counter())
    per_model = collections.defaultdict(lambda: collections.Counter())

    for project, root in A.PROJECTS.items():
        if project == "joda-time":                      # no suites: nothing to call "engaged"
            continue
        proj = A.Project(root)
        simple_index = collections.defaultdict(list)
        for key in proj.by_rel:
            simple_index[key.rsplit("/", 1)[-1][:-5]].append(key)

        def to_key(name: str) -> str | None:
            name = name.split("$")[0]
            if "." in name:
                key = proj.by_fqn.get(name)
                if key is None:                          # Outer.Inner
                    key = proj.by_fqn.get(name.rsplit(".", 1)[0])
                return key
            hits = simple_index.get(name, [])
            return hits[0] if len(hits) == 1 else None

        for run_dir in glob.glob(str(A.RECORDS / project / "*")):
            log = Path(run_dir) / "log.jsonl"
            if not log.exists():
                continue
            model = Path(run_dir).name.split("_")[0].split("-")[0]
            errors = collections.defaultdict(set)       # cut -> types with an API error
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("compiler_success"):
                    continue
                path = (r.get("test_file_path") or "").replace("\\", "/")
                if "/src/test/java/" not in path:
                    continue
                cut = path.split("/src/test/java/", 1)[1][:-len("Test.java")] + ".java"
                for t in attributed_types(r.get("compiler_feedback") or ""):
                    key = to_key(t)
                    if key:
                        errors[cut].add(key)
            used = collections.defaultdict(set)
            for suite_path in glob.glob(str(Path(run_dir) / "test" / "**" / "*Test.java"), recursive=True):
                p = suite_path.replace("\\", "/")
                cut = p.split("/test/java/", 1)[-1][:-len("Test.java")] + ".java"
                text = Path(suite_path).read_text(encoding="utf-8", errors="replace")
                used[cut] = A.resolve_all(proj, A._refs(parser.parse(text.encode()).root_node,
                                                        text.encode()), text, "")
            for cut in set(errors) | set(used):
                retrieved = v1.get((project, cut))
                if retrieved is None:
                    continue
                for t in errors[cut] | used[cut]:
                    group = "class under test" if t == cut else \
                        "in prompt" if t in retrieved else "NOT in prompt"
                    stats[group]["pairs"] += 1
                    stats[group]["errored"] += t in errors[cut]
                    stats[group]["errored, never used"] += t in errors[cut] and t not in used[cut]
                    per_model[(model, group)]["pairs"] += 1
                    per_model[(model, group)]["errored"] += t in errors[cut]

    def line(label, c):
        n = c["pairs"]
        return (f"  {label:<34}{n:>7}{c['errored']:>9}{100 * c['errored'] / max(n, 1):>9.1f}%"
                f"{c['errored, never used']:>14}")

    print("(run, class, collaborator) pairs the model engaged with, and how often it got the API wrong")
    print(f"  {'':<34}{'pairs':>7}{'errored':>9}{'rate':>10}{'never fixed':>14}")
    for g in ("class under test", "in prompt", "NOT in prompt"):
        print(line(g, stats[g]))
    print("\nby model")
    for model in sorted({m for m, _ in per_model}):
        cells = []
        for g in ("in prompt", "NOT in prompt"):
            c = per_model[(model, g)]
            cells.append(f"{g}: {c['errored']:>4}/{c['pairs']:<5} {100 * c['errored'] / max(c['pairs'], 1):5.1f}%")
        print(f"  {model:<10}  " + "   ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
