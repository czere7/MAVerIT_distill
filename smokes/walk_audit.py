"""What the collaborator walk finds, per pattern, and what it would cost to include it.

Every candidate is rendered with its pattern's view and measured as the prompt would carry
it. Ground truth is the retrieval audit's: `needed` = the CUT's non-private surface plus
the types the logged suites used (the latter only for classes with suites).

Reports:
  1. per pattern: candidates, how many were needed, and their size
  2. what each pattern finds that no other pattern does
  3. the recall-vs-size curve when patterns are added best-first -- the input for choosing
     a budget
Run from the harness root:  python smokes/walk_audit.py
"""
import collections
import contextlib
import io
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re  # noqa: E402

import collaborator_render as R  # noqa: E402
import collaborators  # noqa: E402
from collaborator_walk import PATTERNS, walk  # noqa: E402
from smokes import retrieval_audit as A  # noqa: E402
from smokes import retrieval_v2 as v2  # noqa: E402
from smokes import retrieval_v3 as v3  # noqa: E402
from utils import SourceCodeFileData, is_concrete_class, parser  # noqa: E402

INDEX_NAMES = {"commons-cli": "commons-cli", "commons-codec": "commons-codec",
               "commons-csv": "commons-csv", "jackson-core": "jackson-core", "joda-time": "joda-time"}
OUT = Path("smokes/walk-audit")


def fqn_of(key: str) -> str:
    return key[:-5].replace("/", ".")


def render(cand, f: SourceCodeFileData, idx2, cut_ids: set[str], called: set[str]) -> str:
    view = cand.view
    if view == "api":
        return v2._view(f, idx2, "api", cut_ids).file_content
    if view == "internal":
        return v2._view(f, idx2, "internal", cut_ids | called).file_content
    if view == "ctor":
        return v2._view(f, idx2, "hint", cut_ids).file_content
    if view == "producer":
        entries: dict[str, set[str]] = {}
        for decl in cand.entries:
            entries.setdefault(decl.split("(")[0].split()[-1], set()).add(decl)
        return v3._producer_view(f, idx2, entries).file_content
    # "inherited" / "signatures": the whole non-private surface, no bodies
    root, src = idx2.parse(f)
    parts = [v2._render_type(d, src, set(), lambda m, n: True) for d in v2._top_types(root)]
    return "\n\n".join(parts)


def audit() -> list[dict]:
    rows = []
    for project, root in A.PROJECTS.items():
        print(f"walking {project} ...", flush=True)
        proj = A.Project(root)
        index = collaborators.load(Path(root), INDEX_NAMES[project])
        suites = A.logged_suites(project)
        idx2 = v2._Index.of(proj.files)
        by_fqn = {fqn_of(k): f for k, f in proj.by_rel.items()}
        for cut in [f for f in proj.files if is_concrete_class(f)]:
            key = A.rel(cut.file_path)
            cut_fqn = fqn_of(key)
            code = cut.file_content
            api = {fqn_of(k) for k in A.resolve_all(proj, A.api_refs(code), code, key)}
            used = set()
            for s in suites.get(key, []):
                used |= {fqn_of(k) for k in A.resolve_all(
                    proj, A._refs(parser.parse(s.encode()).root_node, s.encode()), s, key)}
            root_node, src = idx2.parse(cut)
            cut_ids = v2._identifiers(root_node, src)
            calls = index.get(cut_fqn, {}).get("calls", {})
            test_pkg = index.get(cut_fqn, {}).get("package", "")
            supers = R.supertype_views(index, by_fqn, cut_fqn, test_pkg, cut_ids) if cut_fqn in index else {}
            producer_lines = R.producer_recipes(index, cut_fqn, test_pkg) if cut_fqn in index else []
            suite_text = "\n".join(suites.get(key, []))
            cands = []
            for c in walk(index, cut_fqn):
                f = by_fqn.get(c.fqn)
                if f is None:
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    text = render(c, f, idx2, cut_ids, set(calls.get(c.fqn, [])))
                    middle = R.render_candidate(c, f, index, cut_fqn, test_pkg, cut_ids,
                                                supers, producer_lines)
                cands.append({"fqn": c.fqn, "pattern": c.pattern, "view": c.view, "hops": c.hops,
                              "all_patterns": sorted({p for p, _ in c.patterns}),
                              "chars": A.prompt_chars(text),
                              "full_chars": A.prompt_chars(f.file_content),
                              "middle_chars": A.prompt_chars(middle),
                              "members_missing": {
                                  "views": members_missing(text, c.fqn, suite_text),
                                  "middle": members_missing(middle, c.fqn, suite_text)}
                              if c.fqn in used else None,
                              "needed": c.fqn in api or c.fqn in used, "used": c.fqn in used})
            tagged = R.tag_private_members(code)
            rows.append({"project": project, "cut": cut_fqn, "cut_chars": A.prompt_chars(code),
                         "tag_overhead": A.prompt_chars(tagged) - A.prompt_chars(code),
                         "has_suites": key in suites, "api": sorted(api), "used": sorted(used),
                         "candidates": cands})
    return rows


def members_missing(text: str, fqn: str, suite: str) -> int:
    """Constructors and static methods of `fqn` the suites called that `text` does not show."""
    simple = fqn.rsplit(".", 1)[-1]
    wanted = set()
    if re.search(rf"\bnew\s+{simple}\s*[(<]", suite):
        wanted.add(simple)
    wanted |= set(re.findall(rf"\b{simple}\.([a-z]\w*)\s*\(", suite))
    return sum(1 for m in wanted if not re.search(rf"\b{re.escape(m)}\s*\(", text))


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def report(rows: list[dict]) -> None:
    su = [r for r in rows if r["has_suites"]]
    total_used = sum(len(r["used"]) for r in su)
    total_needed = sum(len(set(r["api"]) | set(r["used"])) for r in su)

    print("\n== 1. Per pattern (a type counts under its highest-priority pattern) ==")
    print("   hit rate and useful chars: classes with suites; chars: all 249 classes")
    print(f"  {'pattern':<18}{'view':<11}{'cands':>7}{'needed':>8}{'hit rate':>9}"
          f"{'chars':>12}{'useful':>8}{'median/class':>14}")
    stat = collections.defaultdict(lambda: [0, 0, 0, 0])
    size_all = collections.defaultdict(int)
    per_class = collections.defaultdict(list)
    for r in rows:
        by_p = collections.Counter()
        for c in r["candidates"]:
            size_all[c["pattern"]] += c["chars"]
            by_p[c["pattern"]] += c["chars"]
            if r["has_suites"]:
                s = stat[c["pattern"]]
                s[0] += 1
                s[1] += c["needed"]
                s[2] += c["chars"]
                s[3] += c["chars"] if c["needed"] else 0
        for p in PATTERNS:
            per_class[p].append(by_p[p])
    for p in PATTERNS:
        n, u, ch, cu = stat[p]
        print(f"  {p:<18}{PATTERNS[p]:<11}{n:>7}{u:>8}{pct(u, n):>9}{size_all[p]:>12,}"
              f"{pct(cu, ch):>8}{statistics.median(per_class[p]):>14,.0f}")

    print("\n== 2. Needed types found by ONE pattern only (what dropping it would lose) ==")
    only = collections.Counter()
    for r in su:
        for c in r["candidates"]:
            if c["needed"] and len(c["all_patterns"]) == 1:
                only[c["all_patterns"][0]] += 1
    for p in PATTERNS:
        print(f"  {p:<18}{only[p]:>5}")

    print("\n== 3. Adding patterns best-first (by hit rate): recall vs size, views vs whole files ==")
    print("   views: each type rendered for its pattern; files: the whole source file (comment lines stripped)")
    order = sorted(PATTERNS, key=lambda p: -(stat[p][1] / stat[p][0] if stat[p][0] else 0))
    print(f"  {'':<20}{'used':>8}{'------------- views -------------':>36}"
          f"{'------------- files -------------':>36}{'------------ middle -------------':>36}")
    print(f"  {'+ pattern':<20}{'recall':>8}" + f"{'total':>12}{'median':>8}{'p90':>8}{'max':>8}" * 3)
    chosen = set()
    for p in order:
        chosen.add(p)
        hit_used = 0
        views, files, middle = [], [], []
        for r in rows:
            picked = [c for c in r["candidates"] if c["pattern"] in chosen]
            views.append(sum(c["chars"] for c in picked))
            files.append(sum(c["full_chars"] for c in picked))
            middle.append(sum(c["middle_chars"] for c in picked))
            if r["has_suites"]:
                hit_used += len(set(r["used"]) & {c["fqn"] for c in picked})
        cells = ""
        for s in (sorted(views), sorted(files), sorted(middle)):
            cells += (f"{sum(s):>12,}{statistics.median(s):>8,.0f}{s[int(len(s) * 0.9)]:>8,}"
                      f"{s[-1]:>8,}")
        print(f"  {'+ ' + p:<20}{pct(hit_used, total_used):>8}{cells}")
    print("\n== 4. Constructors / static methods the suites called, missing from what was rendered ==")
    for mode in ("views", "middle"):
        miss = sum(c["members_missing"][mode] for r in rows for c in r["candidates"] if c["members_missing"])
        types = sum(1 for r in rows for c in r["candidates"] if c["members_missing"] and c["members_missing"][mode])
        print(f"  {mode:<8} {miss} members missing, across {types} used collaborators")
    print(f"  private-member tags on the CUT: {sum(r['tag_overhead'] for r in rows):,} chars in total, "
          f"median {statistics.median(sorted(r['tag_overhead'] for r in rows)):,.0f} per class")

    cut = sorted(r["cut_chars"] for r in rows)
    print(f"\n  for scale: the classes under test themselves are median {statistics.median(cut):,.0f}, "
          f"p90 {cut[int(len(cut) * 0.9)]:,}, max {cut[-1]:,} chars")
    print("  v3 retriever, same measure: used recall 82.1%, total 4,210,960 chars, median 7,267/class")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = audit()
    (OUT / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
