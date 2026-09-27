"""Does the mutation writer only APPEND tests, or does it also rewrite existing ones?

For every mutation_test_writer_node call in the DeepSeek runs (MAVerIT4/inference_records),
the suite it was given (the prompt's "## Current Test Class" section) is diffed against the
suite it returned, per method, ignoring whitespace and comments:

  added      a test method whose name is new
  removed    a test method that disappeared
  modified   same name, different code -- with the change in assert count
  unchanged

plus where the added tests landed (all after the last kept test = appended), what changed
outside test methods (imports, fields, helpers), and how much of the response was new text.
Run from the harness root:  python smokes/mutation_writer_diff.py [node_name]
"""
import collections
import glob
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import RECORDS  # noqa: E402
from utils import parser, strip_markdown_code_fence  # noqa: E402

TEST_ANN = re.compile(r"@\s*(?:[\w.]+\.)?(?:Test|ParameterizedTest|RepeatedTest|TestFactory)\b")
ASSERT = re.compile(r"\b(?:assert\w*|fail|verify)\s*\(")


def normalize(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    return re.sub(r"\s+", "", code)


def members(code: str):
    """(tests, other): tests is an ordered list of (name, source); other is the normalized
    text of everything that is not a test method (imports, fields, helpers, setup)."""
    src = code.encode("utf-8")
    root = parser.parse(src).root_node
    tests, spans = [], []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "method_declaration":
            text = src[n.start_byte:n.end_byte].decode("utf-8")
            name_node = n.child_by_field_name("name")
            name = src[name_node.start_byte:name_node.end_byte].decode() if name_node else ""
            if TEST_ANN.search(text) or name.startswith("test"):
                tests.append((n.start_byte, name, text))
                spans.append((n.start_byte, n.end_byte))
                continue
        stack.extend(n.children)
    tests.sort()
    other = src
    for s, e in sorted(spans, reverse=True):
        other = other[:s] + other[e:]
    return [(name, text) for _, name, text in tests], normalize(other.decode("utf-8"))


def extract_current_suite(prompt: str) -> str | None:
    start = prompt.find("## Current Test Class")
    end = prompt.find("## Class Under Test", start)
    if start < 0 or end < 0:
        return None
    section = prompt[start:end]
    m = re.search(r"```(?:java)?\n(.*?)```", section, re.S)
    return m.group(1) if m else None


def main() -> int:
    node = sys.argv[1] if len(sys.argv) > 1 else "mutation_test_writer_node"
    per_call = []
    for pairs in sorted(glob.glob(str(RECORDS / "*" / "ds4*" / "prompt-response-pairs.jsonl"))):
        project = Path(pairs).parent.parent.name
        for line in open(pairs, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d["node_name"] != node:
                continue
            prompt = d["prompt"] if isinstance(d["prompt"], str) else json.dumps(d["prompt"])
            before = extract_current_suite(prompt)
            after = strip_markdown_code_fence(d["response"]["data"].get("content") or "")
            if not before or not after.strip():
                per_call.append({"project": project, "status": "unparseable" if not before else "empty"})
                continue
            tb, ob = members(before)
            ta, oa = members(after)
            before_map = {n: normalize(t) for n, t in tb}
            after_map = {n: normalize(t) for n, t in ta}
            added = [n for n, _ in ta if n not in before_map]
            removed = [n for n in before_map if n not in after_map]
            kept = [n for n, _ in ta if n in before_map]
            modified = [n for n in kept if before_map[n] != after_map[n]]
            assert_delta = [len(ASSERT.findall(dict(ta)[n])) - len(ASSERT.findall(dict(tb)[n]))
                            for n in modified]
            order = [n for n, _ in ta]
            last_kept = max((order.index(n) for n in kept), default=-1)
            interleaved = [n for n in added if order.index(n) < last_kept]
            added_chars = sum(len(t) for n, t in ta if n in added)
            per_call.append({
                "project": project, "status": "ok",
                "tests_before": len(tb), "added": len(added), "removed": len(removed),
                "modified": len(modified), "unchanged": len(kept) - len(modified),
                "assert_delta": assert_delta, "interleaved": len(interleaved),
                "other_changed": ob != oa, "out_chars": len(after), "added_chars": added_chars,
                "modified_names": modified[:5], "removed_names": removed[:5]})

    ok = [c for c in per_call if c["status"] == "ok"]
    print(f"{node}: {len(per_call)} calls, {len(ok)} diffable "
          f"({sum(c['status'] == 'empty' for c in per_call)} empty, "
          f"{sum(c['status'] == 'unparseable' for c in per_call)} without a current suite)\n")

    def share(pred):
        n = sum(1 for c in ok if pred(c))
        return f"{n:>4} ({100 * n / len(ok):4.1f}%)"

    print("  calls that ...")
    print(f"    only added tests (nothing else touched)          {share(lambda c: c['added'] and not c['modified'] and not c['removed'] and not c['other_changed'])}")
    print(f"    added tests                                      {share(lambda c: c['added'])}")
    print(f"    modified at least one existing test              {share(lambda c: c['modified'])}")
    print(f"    removed at least one existing test               {share(lambda c: c['removed'])}")
    print(f"    changed imports / fields / helpers               {share(lambda c: c['other_changed'])}")
    print(f"    placed a new test between existing ones          {share(lambda c: c['interleaved'])}")
    print(f"    changed nothing at all                           {share(lambda c: not c['added'] and not c['modified'] and not c['removed'] and not c['other_changed'])}")

    tot = collections.Counter()
    for c in ok:
        for k in ("tests_before", "added", "removed", "modified", "unchanged"):
            tot[k] += c[k]
    print(f"\n  test methods: {tot['tests_before']} given; {tot['added']} added, {tot['modified']} modified, "
          f"{tot['removed']} removed, {tot['unchanged']} returned unchanged")
    deltas = [d for c in ok for d in c["assert_delta"]]
    if deltas:
        print(f"  modified tests' assert count: {sum(d > 0 for d in deltas)} gained asserts, "
              f"{sum(d == 0 for d in deltas)} same count, {sum(d < 0 for d in deltas)} lost asserts "
              f"(median change {statistics.median(deltas):+})")
    new_share = [c["added_chars"] / c["out_chars"] for c in ok if c["out_chars"]]
    print(f"  share of each response that is new tests: median {100 * statistics.median(new_share):.1f}%, "
          f"mean {100 * statistics.mean(new_share):.1f}%")

    print("\n  by project:")
    for project in sorted({c["project"] for c in ok}):
        cs = [c for c in ok if c["project"] == project]
        print(f"    {project:<15} calls {len(cs):>3}   modified-existing {sum(1 for c in cs if c['modified']):>3}   "
              f"removed {sum(1 for c in cs if c['removed']):>3}   interleaved {sum(1 for c in cs if c['interleaved']):>3}")

    print("\n  examples of modified tests:")
    for c in [c for c in ok if c["modified"]][:8]:
        print(f"    {c['project']:<15} {c['modified_names']}  assert delta {c['assert_delta'][:5]}")
    Path("smokes/mutation-writer-diff.jsonl").write_text("\n".join(json.dumps(c) for c in per_call), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
