"""Thinking on vs off, same four rollouts: what the harness ended with, and what it cost.

"best green" is the highest mutation score of any green compile, which is what the harness
keeps; "last" is where the run stopped, which differs when a run crashed after a good suite.
"""
import json
import sys
from pathlib import Path

CLASSES = ["CharSequenceUtils", "PercentCodec", "URLCodec", "Metaphone"]
ARMS = {"on": "qwen38-{}", "off": "qwen38-nothink-{}"}


def find(run_id: str) -> Path | None:
    for base in (Path("logs"), Path(".")):
        if (base / run_id / "log.jsonl").exists():
            return base / run_id
    return None


def summarize(run: Path) -> dict:
    rows = [json.loads(l) for l in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    calls = [json.loads(l) for l in
             (run / "prompt-response-pairs.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    greens = [r for r in rows if r.get("suite_green")]
    best = max(greens, key=lambda r: r.get("mutation") or 0, default=None)
    gen = secs = empty = 0
    for c in calls:
        d = c["response"]["data"]
        md = d.get("response_metadata") or {}
        gen += md.get("eval_count") or 0
        secs += (md.get("total_duration") or 0) / 1e9
        empty += not (d.get("content") or "").strip()
    return {
        "compiles": len(rows),
        "green": len(greens),
        "best_mut": best.get("mutation") if best else None,
        "best_line": best.get("line_coverage") if best else None,
        "best_tests": best.get("tests_run") if best else None,
        "last_mut": rows[-1].get("mutation") if rows else None,
        "calls": len(calls),
        "empty": empty,
        "gen_tokens": gen,
        "model_min": secs / 60,
    }


def fmt(v, spec=""):
    if v is None:
        return "-"
    return format(v, spec) if isinstance(v, (int, float)) else str(v)


def main() -> int:
    print(f"{'class':<18}{'arm':<5}{'best mut':>9}{'line':>7}{'tests':>6}{'last mut':>9}"
          f"{'compiles':>9}{'green':>6}{'calls':>6}{'empty':>6}{'gen tok':>9}{'min':>6}")
    totals = {arm: {"gen_tokens": 0, "model_min": 0.0} for arm in ARMS}
    for name in CLASSES:
        for arm, pattern in ARMS.items():
            run = find(pattern.format(name))
            if run is None:
                print(f"{name:<18}{arm:<5}(no run)")
                continue
            s = summarize(run)
            totals[arm]["gen_tokens"] += s["gen_tokens"]
            totals[arm]["model_min"] += s["model_min"]
            print(f"{name:<18}{arm:<5}{fmt(s['best_mut'], '.1f'):>9}{fmt(s['best_line'], '.1f'):>7}"
                  f"{fmt(s['best_tests']):>6}{fmt(s['last_mut'], '.1f'):>9}{s['compiles']:>9}"
                  f"{s['green']:>6}{s['calls']:>6}{s['empty']:>6}{s['gen_tokens']:>9,}"
                  f"{s['model_min']:>6.1f}")
    for arm, t in totals.items():
        print(f"TOTAL thinking {arm:<4} generated {t['gen_tokens']:,} tokens, "
              f"model time {t['model_min']:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
