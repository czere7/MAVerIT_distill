"""Qwen3.8 as teacher: outcomes, and how close each call came to the 64k context.

PROMPT TOKENS ARE ESTIMATED, NOT READ. Ollama caches the shared prompt prefix, so on every
call after the first `prompt_eval_count` counts only newly evaluated tokens -- a low number
there is caching, not truncation. The estimate uses chars-per-token calibrated on the
first, uncached call of each run. What can actually overflow is prompt + generated, and a
generation that hits the ceiling ends with done_reason == "length".
"""
import json
import sys
from pathlib import Path

CONTEXT = 65536
RUNS = ["CharSequenceUtils", "PercentCodec", "URLCodec", "Metaphone"]


def prompt_text(p) -> str:
    return p if isinstance(p, str) else json.dumps(p)


def main() -> int:
    grand = {"calls": 0, "in": 0, "out": 0, "thinking": 0, "answer": 0, "seconds": 0.0}
    for name in RUNS:
        run = Path(f"qwen38-{name}")
        pairs = run / "prompt-response-pairs.jsonl"
        log = run / "log.jsonl"
        if not pairs.exists():
            print(f"\n=== {name}: no calls logged ===")
            continue
        calls = [json.loads(l) for l in pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()
                if l.strip()] if log.exists() else []

        first_md = calls[0]["response"]["data"].get("response_metadata") or {}
        cpt = (len(prompt_text(calls[0]["prompt"])) / first_md["prompt_eval_count"]
               if first_md.get("prompt_eval_count") else 4.27)

        print(f"\n=== {name} ===  ({len(calls)} calls, {cpt:.2f} chars/token)")
        print(f"  {'node':<26}{'prompt~':>8}{'out':>7}{'think%':>8}{'peak':>8}"
              f"{'headroom':>10}{'tok/s':>7}  done")
        for c in calls:
            d = c["response"]["data"]
            md = d.get("response_metadata") or {}
            est_in = int(len(prompt_text(c["prompt"])) / cpt)
            out = md.get("eval_count") or 0
            think = len((d.get("additional_kwargs") or {}).get("reasoning_content", ""))
            answer = len(d.get("content") or "")
            peak = est_in + out
            secs = (md.get("total_duration") or 0) / 1e9
            speed = out / (md["eval_duration"] / 1e9) if md.get("eval_duration") else 0
            flag = "  <-- CUT OFF" if md.get("done_reason") == "length" else ""
            print(f"  {c['node_name']:<26}{est_in:>8}{out:>7}"
                  f"{100 * think / max(1, think + answer):>7.0f}%{peak:>8}"
                  f"{CONTEXT - peak:>10}{speed:>7.1f}  {md.get('done_reason')}{flag}")
            grand["calls"] += 1
            grand["in"] += est_in
            grand["out"] += out
            grand["thinking"] += think
            grand["answer"] += answer
            grand["seconds"] += secs

        muts = [r.get("mutation") for r in rows if isinstance(r.get("mutation"), (int, float))]
        greens = [r for r in rows if r.get("suite_green")]
        line = [r.get("line_coverage") for r in rows
                if isinstance(r.get("line_coverage"), (int, float))]
        final_suite = list((run / "final_suite").glob("*.java"))
        n_tests = (final_suite[0].read_text(encoding="utf-8").count("@Test")
                   if final_suite else 0)
        print(f"  compiles logged {len(rows)}   green {len(greens)}   "
              f"final mutation {muts[-1] if muts else '-'}%   "
              f"final line cov {line[-1] if line else '-'}%   final @Test {n_tests}")

    print("\n=== TOTALS ===")
    print(f"  calls {grand['calls']}   est. prompt tokens {grand['in']:,}   "
          f"generated tokens {grand['out']:,}")
    share = 100 * grand["thinking"] / max(1, grand["thinking"] + grand["answer"])
    print(f"  thinking share of generated text {share:.0f}%   "
          f"model time {grand['seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
