"""Paired report for a retrieval ablation (v1 vs v4): outcome, process, cost.

Per run, from its log.jsonl (one row per compile) and prompt-response-pairs.jsonl:
  outcome   ended green?, best green mutation score, its line coverage and test count
  process   compiles, compiles that failed, the javac errors by kind, red (runtime) rounds
  cost      model calls, input/output tokens, wall-clock minutes, no-thinking retries

Only classes where BOTH arms finished count in the paired totals.
    python smokes/ablation_report.py [tag]        (default: deepseek)
"""
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ERR = re.compile(r"\[ERROR\] \S+?\.java:\[(\d+),\d+\] (?:error: )?(.*)")


def error_kind(msg: str) -> str:
    if msg.startswith("cannot find symbol"):
        return "missing symbol"
    if "private access" in msg or "protected access" in msg or "is not public" in msg:
        return "access violation"
    if msg.startswith("unreported exception"):
        return "uncaught checked exception"
    if "cannot be applied" in msg or "no suitable" in msg or msg.startswith("incompatible types"):
        return "wrong signature/type"
    return "other"


def summarize(run_dir: Path) -> dict | None:
    result = run_dir / "result.json"
    if not result.exists():
        return None
    res = json.loads(result.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (run_dir / "log.jsonl").exists() else []
    calls = [json.loads(l) for l in (run_dir / "prompt-response-pairs.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (run_dir / "prompt-response-pairs.jsonl").exists() else []
    greens = [r for r in rows if r.get("suite_green")]
    num = lambda v: v if isinstance(v, (int, float)) else None
    best = max(greens, key=lambda r: num(r.get("mutation")) or -1, default=None)
    errors = collections.Counter()
    for r in rows:
        if not r.get("compiler_success"):
            seen = {(m.group(1), m.group(2)) for m in ERR.finditer(r.get("compiler_feedback") or "")}
            for _, msg in seen:
                errors[error_kind(msg)] += 1
    usage = [(c["response"]["data"].get("usage_metadata") or {}) for c in calls]
    return {
        "run_id": res["run_id"], "arm": res["arm"], "class": res["class"].rsplit(".", 1)[-1],
        "status": res["status"], "minutes": res["minutes"],
        "ok": res["status"] == "finished",
        "ended_green": bool(rows) and bool(rows[-1].get("suite_green")) or bool(greens),
        "best_mutation": num(best.get("mutation")) if best else None,
        "line_coverage": num(best.get("line_coverage")) if best else None,
        "tests": best.get("tests_run") if best else None,
        "compiles": len(rows),
        "compile_failures": sum(1 for r in rows if not r.get("compiler_success")),
        "red_rounds": sum(1 for r in rows if r.get("compiler_success") and not r.get("suite_green")),
        "errors": dict(errors),
        "first_compile_ok": bool(rows) and bool(rows[0].get("compiler_success")),
        "first_compile_green": bool(rows) and bool(rows[0].get("suite_green")),
        "calls": len(calls),
        "retries": sum(1 for c in calls if c["node_name"].endswith(":no_thinking_retry")),
        "tokens_in": sum(u.get("input_tokens", 0) for u in usage),
        "tokens_out": sum(u.get("output_tokens", 0) for u in usage),
    }


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    A, B = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("v1", "v4")
    base = Path("ablation") / tag
    runs = [s for s in (summarize(d) for d in sorted(base.glob("abl-*")) if d.is_dir()) if s]
    by_class = collections.defaultdict(dict)
    for r in runs:
        by_class[r["class"]][r["arm"]] = r
    pairs = {c: arms for c, arms in by_class.items() if arms.get(A, {}).get("ok") and arms.get(B, {}).get("ok")}
    dropped = sorted(set(by_class) - set(pairs))

    def f(v, spec=""):
        return "-" if v is None else format(v, spec)

    print(f"{len(pairs)} complete pairs; dropped (an arm did not finish): {dropped}\n")
    print(f"{'class':<28}{'arm':<5}{'green':>6}{'mut':>7}{'line':>7}{'tests':>6}{'compiles':>9}"
          f"{'failed':>7}{'red':>5}{'1st ok':>7}{'min':>6}{'tok in':>9}{'tok out':>9}  errors")
    for c in sorted(pairs):
        for arm in (A, B):
            r = pairs[c][arm]
            print(f"{c:<28}{arm:<5}{'yes' if r['ended_green'] else 'NO':>6}{f(r['best_mutation'], '.1f'):>7}"
                  f"{f(r['line_coverage'], '.1f'):>7}{f(r['tests']):>6}{r['compiles']:>9}{r['compile_failures']:>7}"
                  f"{r['red_rounds']:>5}{'yes' if r['first_compile_ok'] else 'no':>7}{r['minutes']:>6}"
                  f"{r['tokens_in']:>9,}{r['tokens_out']:>9,}  {r['errors'] or ''}")

    if not pairs:
        return 0
    print("\n== paired totals ==")
    for arm in (A, B):
        rs = [pairs[c][arm] for c in pairs]
        muts = [r["best_mutation"] for r in rs if r["best_mutation"] is not None]
        errs = collections.Counter()
        for r in rs:
            errs.update(r["errors"])
        print(f"  {arm}: green {sum(r['ended_green'] for r in rs)}/{len(rs)}   "
              f"first compile ok {sum(r['first_compile_ok'] for r in rs)}/{len(rs)}   "
              f"mean best mutation {statistics.mean(muts):.1f}   "
              f"compiles {sum(r['compiles'] for r in rs)} (failed {sum(r['compile_failures'] for r in rs)}, "
              f"red {sum(r['red_rounds'] for r in rs)})   minutes {sum(r['minutes'] for r in rs):.1f}   "
              f"tokens in {sum(r['tokens_in'] for r in rs):,} out {sum(r['tokens_out'] for r in rs):,}   "
              f"retries {sum(r['retries'] for r in rs)}")
        print(f"      javac errors by kind: {dict(errs.most_common())}")
    wins = collections.Counter()
    for c in pairs:
        a, b = pairs[c][A]["best_mutation"], pairs[c][B]["best_mutation"]
        if a is None or b is None:
            continue
        wins[f"{B} higher" if b > a + 0.5 else f"{A} higher" if a > b + 0.5 else "tie (within 0.5)"] += 1
    print(f"  best mutation, per class: {dict(wins)}")
    Path(base / "report.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
