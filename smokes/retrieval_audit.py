"""Audit of collaborator retrieval: what get_relevant_source_files hands the writer, against
what the class under test (CUT) actually needs.

The retriever runs HERE, on every concrete class of every project, exactly as the nodes call
it. The logged prompts in MAVerIT4/inference_records serve two purposes: they confirm this
smoke run reproduces what the harness really retrieved, and their saved test suites say
which project types a test that COMPILED actually used.

Ground truth, per CUT, resolved to project files (JDK types never count):
  api      types in the CUT's non-private surface: supertypes, and parameter / return /
           throws / field types of non-private members. A test cannot call the API
           without them.
  used     types referenced by the logged test suites for the CUT (they compiled, so their
           API use was right). Only for classes that have suites.
  needed   api | used
  internal referenced in the CUT's code but in neither of the above: private helpers.

Every retrieved file is classified as needed / internal / unreferenced, and unreferenced
files are explained: a name match inside a comment or a longer identifier, an
implementor/extender pulled in behind a matched interface or abstract class, and so on.

Run from the harness root:  python smokes/retrieval_audit.py
"""
import collections
import contextlib
import glob
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Extractor import Extractor  # noqa: E402
from utils import (KEEP_COMMENT_MARKER, CLASS_RE, extract_methods, get_relevant_source_files,  # noqa: E402
                   is_concrete_class, parser, walk)

PROJECTS = {  # project -> the checkout the logs were generated from (same main sources)
    "commons-cli": r"C:\Users\akosc\IdeaProjects\commons-cli2",
    "commons-codec": r"C:\Users\akosc\IdeaProjects\commons-codec2",
    "commons-csv": r"C:\Users\akosc\IdeaProjects\commons-csv2",
    "jackson-core": r"C:\Users\akosc\IdeaProjects\jackson-core",
    "joda-time": r"C:\Users\akosc\IdeaProjects\joda-time2",
}
from smokes import retrieval_v2, retrieval_v3  # noqa: E402
from smokes.retrieval_v2 import get_relevant_source_files_v2  # noqa: E402
from smokes.retrieval_v3 import get_relevant_source_files_v3  # noqa: E402

RETRIEVERS = {"v1": get_relevant_source_files, "v2": get_relevant_source_files_v2,
              "v3": get_relevant_source_files_v3}
REASONS = {"v2": retrieval_v2.LAST_REASONS, "v3": retrieval_v3.LAST_REASONS}
RECORDS = Path(r"C:\Users\akosc\Desktop\MAVerIT4\inference_records")
OUT = Path("smokes/retrieval-audit")


# ---------------------------------------------------------------- source model
def rel(path: str) -> str:
    """Path from the package root, the key everything is joined on."""
    p = path.replace("\\", "/")
    i = p.find("src/main/java/")
    return p[i + len("src/main/java/"):] if i >= 0 else p


def package_of(code: str) -> str:
    m = re.search(r"^\s*package\s+([\w.]+)\s*;", code, re.M)
    return m.group(1) if m else ""


def imports_of(code: str) -> tuple[dict[str, str], list[str]]:
    single, wild = {}, []
    for m in re.finditer(r"^\s*import\s+(static\s+)?([\w.]+)(\.\*)?\s*;", code, re.M):
        if m.group(1):
            continue
        if m.group(3):
            wild.append(m.group(2))
        else:
            single[m.group(2).rsplit(".", 1)[-1]] = m.group(2)
    return single, wild


class Project:
    def __init__(self, root: str):
        with contextlib.redirect_stdout(io.StringIO()):
            self.files = Extractor(root).extract_all()
        self.by_rel = {rel(f.file_path): f for f in self.files}
        self.by_fqn = {}
        for key in self.by_rel:
            self.by_fqn[key[:-5].replace("/", ".")] = key

    def resolve(self, simple: str, code: str) -> str | None:
        """Simple type name -> project file, the way javac would, or None (JDK / nested)."""
        single, wild = imports_of(code)
        if simple in single:
            return self.by_fqn.get(single[simple])
        pkg = package_of(code)
        for candidate in [pkg] + wild:
            key = self.by_fqn.get(f"{candidate}.{simple}" if candidate else simple)
            if key:
                return key
        return None


# ---------------------------------------------------------------- reference extraction
def _refs(node, source: bytes, skip_bodies=False) -> set[str]:
    """Type names referenced under `node`: type identifiers, plus Upper.member() / Upper.FIELD
    static access and Upper::method references."""
    names = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in ("import_declaration", "package_declaration", "line_comment", "block_comment"):
            continue
        if skip_bodies and n.type in ("block", "constructor_body") and n is not node:
            continue
        if n.type == "type_identifier":
            names.add(source[n.start_byte:n.end_byte].decode())
        elif n.type in ("method_invocation", "field_access"):
            obj = n.child_by_field_name("object")
            if obj is not None and obj.type == "identifier":
                text = source[obj.start_byte:obj.end_byte].decode()
                if text[:1].isupper():
                    names.add(text)
        elif n.type == "method_reference" and n.children and n.children[0].type == "identifier":
            text = source[n.children[0].start_byte:n.children[0].end_byte].decode()
            if text[:1].isupper():
                names.add(text)
        stack.extend(n.children)
    return names


def _is_private(decl, source: bytes) -> bool:
    for child in decl.children:
        if child.type == "modifiers":
            return "private" in source[child.start_byte:child.end_byte].decode().split()
    return False


def api_refs(code: str) -> set[str]:
    """Type names in the non-private surface of the top-level type and its non-private
    nested types: supertypes, member signatures, non-private field types. Bodies skipped."""
    source = code.encode()
    tree = parser.parse(source)
    names = set()
    type_decls = {"class_declaration", "interface_declaration", "enum_declaration",
                  "record_declaration"}

    def visit_type(decl):
        for field in ("superclass", "interfaces"):
            sub = decl.child_by_field_name(field)
            if sub is not None:
                names.update(_refs(sub, source))
        for child in decl.children:
            if child.type == "super_interfaces" or child.type == "extends_interfaces":
                names.update(_refs(child, source))
        body = decl.child_by_field_name("body")
        if body is None:
            return
        for member in body.children:
            if member.type in type_decls:
                if not _is_private(member, source):
                    visit_type(member)
            elif member.type in ("method_declaration", "constructor_declaration",
                                 "field_declaration", "constant_declaration"):
                if not _is_private(member, source):
                    names.update(_refs(member, source, skip_bodies=True))

    for top in tree.root_node.children:
        if top.type in type_decls:
            visit_type(top)
    return names


def code_tokens(code: str) -> set[str]:
    """Every identifier in actual code (not comments, strings or imports)."""
    source = code.encode()
    tree = parser.parse(source)
    out = set()
    for n in walk(tree.root_node):
        if n.type in ("identifier", "type_identifier"):
            p = n.parent
            while p is not None and p.type not in ("import_declaration", "package_declaration"):
                p = p.parent
            if p is None:
                out.add(source[n.start_byte:n.end_byte].decode())
    return out


def resolve_all(project: Project, names: set[str], code: str, exclude: str) -> set[str]:
    out = {project.resolve(n, code) for n in names}
    return {k for k in out if k and k != exclude}


# ---------------------------------------------------------------- logs
def logged_retrievals(project_name: str) -> dict[str, set[str]]:
    """CUT key -> retrieved keys, from every run's first initial_test_write prompt."""
    found = {}
    for pairs in glob.glob(str(RECORDS / project_name / "*" / "prompt-response-pairs.jsonl")):
        with open(pairs, encoding="utf-8") as f:
            for line in f:
                if '"initial_test_write_node"' not in line[:200]:
                    continue
                try:
                    prompt = json.loads(line)["prompt"]
                except (json.JSONDecodeError, KeyError):
                    continue
                prompt = prompt if isinstance(prompt, str) else json.dumps(prompt)
                cut_at = prompt.find("## Class Under Test")
                src_at = prompt.find("## Relevant Source Files")
                if cut_at < 0 or src_at < 0:
                    continue
                cut_code = prompt[cut_at:src_at]
                m = CLASS_RE.search(cut_code.split("```", 2)[1] if "```" in cut_code else cut_code)
                pkg = package_of(cut_code)
                if not m:
                    continue
                cut_key = (pkg.replace(".", "/") + "/" if pkg else "") + m.group(1) + ".java"
                paths = re.findall(r"Path: `([^`]+)`", prompt[src_at:])
                found.setdefault(cut_key, {rel(p.replace("\\\\", "\\")) for p in paths})
    return found


def logged_suites(project_name: str) -> dict[str, list[str]]:
    """CUT key -> texts of every logged suite written for it."""
    suites = collections.defaultdict(list)
    for path in glob.glob(str(RECORDS / project_name / "*" / "test" / "**" / "*Test.java"),
                          recursive=True):
        p = path.replace("\\", "/")
        key = p.split("/test/java/", 1)[-1] if "/test/java/" in p else p.split("/test/", 1)[-1]
        cut_key = key[:-len("Test.java")] + ".java"
        suites[cut_key].append(Path(path).read_text(encoding="utf-8", errors="replace"))
    return suites


# ---------------------------------------------------------------- member-level check
def missing_members(retrieved_text: str, collab_key: str, suite: str) -> set[str]:
    """Constructors and static methods of a collaborator a compiling suite called, which
    the retrieved (possibly compacted) text of that collaborator does not declare."""
    simple = collab_key.rsplit("/", 1)[-1][:-5]
    used = set()
    if re.search(rf"\bnew\s+{simple}\s*[(<]", suite):
        used.add(f"{simple}(...)")
    used |= {f"{simple}.{m}()" for m in re.findall(rf"\b{simple}\.([a-z]\w*)\s*\(", suite)}
    # Compacted text is bare member snippets joined by "...": a constructor only parses
    # as one inside a class body, so wrap it in one named after the collaborator.
    body = re.sub(r"^\s*package\s+[\w.]+\s*;", "", retrieved_text, flags=re.M)
    wrapped = f"class {simple} {{\n{body.replace(chr(10) + '...' + chr(10), chr(10))}\n}}"
    declared = {m["name"] for m in extract_methods(wrapped)}
    return {u for u in used
            if (u.split("(")[0] if not u.startswith(simple + "(") else simple) not in declared
            and u.split(".", 1)[-1].split("(")[0] not in declared}


# ---------------------------------------------------------------- main
def prompt_chars(text: str) -> int:
    """Size as the prompt carries it: the harness strips comment lines before sending, and
    the Extractor's doubled newlines are not content."""
    markers = ("/**", "/*", "*", "*/", "//")
    return sum(len(line) + 1 for line in text.splitlines()
               if line.strip() and (KEEP_COMMENT_MARKER in line or not line.strip().startswith(markers)))


def audit_project(name: str, root: str) -> list[dict]:
    project = Project(root)
    logged = logged_retrievals(name)
    suites = logged_suites(name)
    cuts = [f for f in project.files if is_concrete_class(f)]
    rows = []
    for cut in cuts:
        for retriever_name, retriever in RETRIEVERS.items():
            rows.append(_audit_class(name, project, cut, retriever_name, retriever, logged, suites))
    return rows


def _audit_class(name, project, cut, retriever_name, retriever, logged, suites) -> dict:
    if True:
        cut_key = rel(cut.file_path)
        state = {"all_files": project.files, "all_test_files": [cut], "current_class_index": 0}
        with contextlib.redirect_stdout(io.StringIO()):
            retrieved = retriever(state)
        reasons = dict(REASONS.get(retriever_name, {}))
        got = {rel(f.file_path): f for f in retrieved}

        code = cut.file_content
        api = resolve_all(project, api_refs(code), code, cut_key)
        static = resolve_all(project, _refs(parser.parse(code.encode()).root_node, code.encode()),
                             code, cut_key)
        used = set()
        member_gaps = []
        for suite in suites.get(cut_key, []):
            used |= resolve_all(project, _refs(parser.parse(suite.encode()).root_node,
                                               suite.encode()), suite, cut_key)
        needed = api | used
        tokens = code_tokens(code)

        classified = {}
        for key, f in got.items():
            simple = key.rsplit("/", 1)[-1][:-5]
            if key == cut_key:
                # Behind a matched supertype, "the first class that extends/implements it"
                # is often the CUT itself; the path check only guards the name match.
                cls = "unref: the class under test itself"
            elif key in needed:
                cls = "needed"
            elif key in static:
                cls = "internal"
            elif simple not in code:
                cls = "unref: not named by the CUT (implementer, producer, 2nd hop)"
            elif simple in tokens:
                cls = "unref: same name, different type"
            elif re.search(rf"\b{re.escape(simple)}\b", code):
                cls = "unref: comment or string only"
            else:
                cls = "unref: substring of a longer name"
            classified[key] = {"class": cls, "chars": prompt_chars(f.file_content),
                               "reason": reasons.get(f.file_path, "name match")}

        for key in used & set(got):
            for suite in suites.get(cut_key, []):
                gaps = missing_members(got[key].file_content, key, suite)
                if gaps:
                    member_gaps.append({"collaborator": key, "missing": sorted(gaps)})
                    break

        return {
            "retriever": retriever_name,
            "project": name,
            "cut": cut_key,
            "cut_chars": prompt_chars(code),
            "retrieved": classified,
            "api": sorted(api),
            "used": sorted(used),
            "has_suites": cut_key in suites,
            "missing_api": sorted(api - set(got)),
            "missing_used": sorted(used - set(got)),
            "internal_not_retrieved": sorted(static - needed - set(got)),
            "member_gaps": member_gaps,
            "log_match": None if cut_key not in logged else (logged[cut_key] == set(got)),
            "log_diff": None if cut_key not in logged else {
                "only_logged": sorted(logged[cut_key] - set(got)),
                "only_smoke": sorted(set(got) - logged[cut_key])},
        }


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "    -"


def report(rows: list[dict]) -> None:
    by_project = collections.defaultdict(list)
    for r in rows:
        by_project[r["project"]].append(r)

    print("\n== Smoke run vs logged prompts (same retrieval?) ==")
    for name, rs in by_project.items():
        checked = [r for r in rs if r["log_match"] is not None]
        print(f"  {name:<14} {sum(r['log_match'] for r in checked)}/{len(checked)} classes identical")

    print("\n== Recall: are the needed collaborators there? ==")
    print(f"  {'project':<14}{'classes':>8}{'api refs':>9}{'api recall':>11}"
          f"{'w/ suites':>10}{'used refs':>10}{'used recall':>12}{'classes missing any':>21}")
    for name, rs in list(by_project.items()) + [("ALL", rows)]:
        api = sum(len(r["api"]) for r in rs)
        api_miss = sum(len(r["missing_api"]) for r in rs)
        su = [r for r in rs if r["has_suites"]]
        used = sum(len(r["used"]) for r in su)
        used_miss = sum(len(r["missing_used"]) for r in su)
        any_miss = sum(bool(r["missing_api"] or r["missing_used"]) for r in rs)
        print(f"  {name:<14}{len(rs):>8}{api:>9}{pct(api - api_miss, api):>11}{len(su):>10}"
              f"{used:>10}{pct(used - used_miss, used):>12}{any_miss:>13} ({pct(any_miss, len(rs)).strip()})")

    print("\n== Precision: what is retrieved, and is it needed? ==")
    classes = collections.Counter()
    chars = collections.Counter()
    for r in rows:
        for v in r["retrieved"].values():
            classes[v["class"]] += 1
            chars[v["class"]] += v["chars"]
    total_files, total_chars = sum(classes.values()), sum(chars.values())
    print(f"  {'category':<62}{'files':>7}{'share':>8}{'chars':>11}{'share':>8}")
    for cls, n in classes.most_common():
        print(f"  {cls:<62}{n:>7}{pct(n, total_files):>8}{chars[cls]:>11,}{pct(chars[cls], total_chars):>8}")
    print(f"  {'TOTAL':<62}{total_files:>7}{'':>8}{total_chars:>11,}")
    cut_chars = sum(r["cut_chars"] for r in rows)
    print(f"  retrieved context is {total_chars / max(1, cut_chars):.2f}x the size of the classes under test "
          f"({total_chars:,} vs {cut_chars:,} chars)")

    per_class = sorted(len(r["retrieved"]) for r in rows)
    print(f"  files per class: median {per_class[len(per_class) // 2]}, max {per_class[-1]}, "
          f"zero for {sum(1 for n in per_class if n == 0)} of {len(per_class)} classes")

    print("\n== Why each file was retrieved, and how often it was needed (classes with suites) ==")
    by_reason = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        if not r["has_suites"]:
            continue
        for v in r["retrieved"].values():
            b = by_reason[v["reason"]]
            b[0] += 1
            b[1] += v["class"] == "needed"
            b[2] += v["chars"]
            b[3] += v["chars"] if v["class"] == "needed" else 0
    print(f"  {'reason':<22}{'files':>7}{'needed':>8}{'hit rate':>9}{'chars':>11}{'useful chars':>13}")
    for reason, (n, u, c, cu) in sorted(by_reason.items(), key=lambda x: -x[1][2]):
        print(f"  {reason:<22}{n:>7}{u:>8}{pct(u, n):>9}{c:>11,}{pct(cu, c):>13}")

    print("\n== Retrieved, but the members a compiling test called were compacted away ==")
    gaps = [(r["cut"], g) for r in rows for g in r["member_gaps"]]
    used_pairs = sum(len(set(r["used"]) & set(r["retrieved"])) for r in rows)
    print(f"  {len(gaps)} of {used_pairs} retrieved used-collaborators miss a called constructor/static method")
    for cut, g in gaps[:8]:
        print(f"    {cut} -> {g['collaborator']}: {', '.join(g['missing'][:4])}")

    print("\n== Most-missed collaborators (needed, not retrieved) ==")
    missed = collections.Counter(k for r in rows for k in set(r["missing_api"]) | set(r["missing_used"]))
    for key, n in missed.most_common(10):
        print(f"  {n:>4}x  {key}")

    print("\n== Noisiest classes (unreferenced chars) ==")
    noisy = sorted(rows, key=lambda r: -sum(v["chars"] for v in r["retrieved"].values()
                                              if v["class"].startswith("unref")))
    for r in noisy[:8]:
        unref = {k: v for k, v in r["retrieved"].items() if v["class"].startswith("unref")}
        print(f"  {r['cut']:<55} {sum(v['chars'] for v in unref.values()):>7,} chars in {len(unref)} files "
              f"(cut {r['cut_chars']:,})")


def summary(rows: list[dict]) -> dict:
    files = [v for r in rows for v in r["retrieved"].values()]
    su = [r for r in rows if r["has_suites"]]
    used = sum(len(r["used"]) for r in su)
    per_class = sorted(sum(v["chars"] for v in r["retrieved"].values()) for r in rows)
    used_pairs = sum(len(set(r["used"]) & set(r["retrieved"])) for r in rows)
    return {
        "api recall": pct(sum(len(r["api"]) - len(r["missing_api"]) for r in rows),
                          sum(len(r["api"]) for r in rows)),
        "used recall": pct(used - sum(len(r["missing_used"]) for r in su), used),
        "classes missing a needed type": f"{sum(bool(r['missing_api'] or r['missing_used']) for r in rows)}",
        "called member cut away": f"{sum(len(r['member_gaps']) for r in rows)} of {used_pairs}",
        "files retrieved": f"{len(files):,}",
        "  needed": pct(sum(v["class"] == "needed" for v in files), len(files)),
        "  internal": pct(sum(v["class"] == "internal" for v in files), len(files)),
        "  unreferenced": pct(sum(v["class"].startswith("unref") for v in files), len(files)),
        "chars retrieved": f"{sum(v['chars'] for v in files):,}",
        "  in unreferenced files": pct(sum(v["chars"] for v in files if v["class"].startswith("unref")),
                                       sum(v["chars"] for v in files)),
        "context / CUT size": f"{sum(v['chars'] for v in files) / sum(r['cut_chars'] for r in rows):.2f}x",
        "median chars per class": f"{per_class[len(per_class) // 2]:,}",
        "p90 chars per class": f"{per_class[int(len(per_class) * 0.9)]:,}",
        "max chars per class": f"{per_class[-1]:,}",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, root in PROJECTS.items():
        print(f"auditing {name} ...", flush=True)
        rows += audit_project(name, root)
    (OUT / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    for retriever_name in RETRIEVERS:
        print(f"\n\n################ {retriever_name} ################")
        report([r for r in rows if r["retriever"] == retriever_name])

    print("\n\n################ v1 vs v2 ################")
    s = {name: summary([r for r in rows if r["retriever"] == name]) for name in RETRIEVERS}
    names = list(RETRIEVERS)
    print(f"  {'':<32}" + "".join(f"{n:>16}" for n in names))
    for key in s[names[0]]:
        print(f"  {key:<32}" + "".join(f"{s[n][key].strip():>16}" for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
