"""The distillation loop, end to end. Run from the repository root with the harness venv:

    python -m training.loop prompts                  build the v5 prompt sets, token-counted
    python -m training.loop cold-start               harness (teacher mode) on every training
                                                     class -> SFT set -> cold-start adapter
    python -m training.loop round 1                  sample, score, route, retrain
    python -m training.loop eval --adapter DIR --tag NAME     held-out measurement only

Every step is resumable: its output is checked before it runs, so a crash or a WSL restart
costs the step in flight and nothing else. Outputs go to logs/loop/ (gitignored); adapters
and merged models to MODELS_ROOT, because a merged model is 4.5 GB.

ONE ROUND:
  1. merge the adapter under test and serve it
  2. 8 rollouts per training prompt, tau 1.2, top_k 40
  3. score: tier 3 on every rollout, tier 4 where mutation can exist (the suite compiled,
     or its salvaged subset ran green)
  4. route, per class -- THE FALL-THROUGH RULE, T = 0 and frozen:
       a rollout kills >= 1 mutant  -> self-served: its verified salvaged subset is the target
       otherwise                    -> the harness takes the rollout with the most tests; its
                                       final suite is a target, and so is the model's own best
                                       verifiable subset, if any
     Harness usage -- fall-through classes over classes -- is the paper's measurement.
  5. dataset = every round's targets so far (per class: the self/model target with the most
     tests, the latest harness target); examples over cutoff_len are dropped, not truncated
  6. train from the COLD-START adapter, never the previous round's: STaR re-fits from base
     on the accumulated set, and continuing would compound the previous round's drift.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training import generate, llamafactory, prompts as prompt_set, stats, targets, vllm_server
from training.harness_run import most_tests, run_harness
from training.score_rollouts import kills_a_mutant, score_rollouts
from training.scoring import java, pertest
from training.tokens import TRAIN_PYTHON, count

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "logs" / "loop"
MODELS = Path(os.environ.get("MODELS_ROOT", Path.home() / "distil-models"))
TEACHER_PROVIDER = os.environ.get("TEACHER_PROVIDER", "deepseek")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "deepseek-v4-flash")
SCORE_WORKERS = 6
TIER4_WORKERS = 4


# ------------------------------------------------------------------ small helpers
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def simple(key: str) -> str:
    return key.rsplit("/", 1)[-1]


def n_tests(source: str) -> int:
    return len(pertest.test_spans(source))


def dataset_rows(target_rows: list[dict], prompts: dict[str, dict], scratch: Path) -> tuple[list[dict], int]:
    """Fence, count exactly, drop what does not fit cutoff_len. Returns (rows, n_dropped)."""
    rows = [{"class_key": t["class_key"], "prompt": prompts[t["class_key"]]["prompt"],
             "output": targets.fence(t["source"])} for t in target_rows]
    counted = count(rows, scratch)
    kept = [{"instruction": r["prompt"], "output": r["output"], "_tokens": r["prompt_tokens"] + r["output_tokens"]}
            for r in counted if r["prompt_tokens"] + r["output_tokens"] <= llamafactory.CUTOFF_LEN]
    return kept, len(counted) - len(kept)


# ------------------------------------------------------------------ prompts
def cmd_prompts(_args) -> None:
    for name, projects in (("train", prompt_set.TRAINING), ("heldout", prompt_set.HELD_OUT)):
        rows = prompt_set.build(projects, LOOP / "scratch")
        prompt_set.save(rows, LOOP / f"prompts_{name}.jsonl")
        kept = [r for r in rows if not r["excluded"]]
        print(f"{name}: {len(rows)} classes, {len(rows) - len(kept)} over {prompt_set.PROMPT_LIMIT:,} "
              f"tokens excluded, median {statistics.median(r['prompt_tokens'] for r in kept):,.0f}")


def load_prompts(name: str) -> dict[str, dict]:
    path = LOOP / f"prompts_{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `python -m training.loop prompts` first")
    return {r["class_key"]: r for r in prompt_set.load(path)}


# ------------------------------------------------------------------ cold start
def cmd_cold_start(args) -> None:
    prompts = load_prompts("train")
    out = LOOP / "cold-start"
    runs = out / "runs.jsonl"
    done = {r["class_key"] for r in read_jsonl(runs)}
    by_project = defaultdict(list)
    for key, p in sorted(prompts.items()):
        if key not in done:
            by_project[p["project"]].append(key)
    print(f"cold start: {len(prompts)} classes, {len(done)} done, teacher "
          f"{TEACHER_PROVIDER}/{TEACHER_MODEL}", flush=True)

    # One worker PER PROJECT: runs on the same project would share its src/test.
    def one_project(project: str) -> None:
        for key in by_project[project]:
            result = run_harness(project, key, f"cold-{project}-{simple(key)}",
                                 provider=TEACHER_PROVIDER, model=TEACHER_MODEL)
            with open(runs, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(result) + "\n")
            suite = result["suite"]
            print(f"  {project:20} {simple(key)[:30]:30} {result['status']:8} {result['minutes']:5.1f} min  "
                  + (f"{suite['n_tests']} tests, mutation {suite['mutation_score_pct']}" if suite else "no green suite"),
                  flush=True)

    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(args.parallel, len(by_project) or 1)) as ex:
        list(ex.map(one_project, sorted(by_project)))

    results = [r for r in read_jsonl(runs) if r["suite"] and r["class_key"] in prompts]
    rows, dropped = dataset_rows([{"class_key": r["class_key"], "source": r["suite"]["source"]}
                                  for r in results], prompts, LOOP / "scratch")
    targets.write_dataset(rows, out / "data", "cold_start")
    # The per-class rollout cap is 2 x the teacher's suite + 64 tokens, the convention every
    # earlier round used, so "terminated" keeps meaning the same thing.
    refs = count([{"class_key": r["class_key"], "prompt": "", "output": targets.fence(r["suite"]["source"])}
                  for r in results], LOOP / "scratch")
    write_jsonl(out / "references.jsonl", [{"class_key": r["class_key"], "tokens": r["output_tokens"]} for r in refs])
    print(f"cold-start set: {len(rows)} targets from {len(results)} green suites "
          f"({dropped} over {llamafactory.CUTOFF_LEN:,} tokens dropped)")

    adapter = MODELS / "cold-start"
    if not (adapter / "adapter_config.json").exists():
        cfg = llamafactory.render("cold_start", out / "data", "cold_start", adapter)
        llamafactory.train(cfg, out / "train.yaml", out / "train.log")
    print(f"cold-start adapter: {adapter}")


# ------------------------------------------------------------------ sampling and scoring
def merged_model(adapter: Path, tag: str) -> Path:
    merged = MODELS / "merged" / tag
    if not (merged / "config.json").exists():
        subprocess.run([str(TRAIN_PYTHON), "-m", "training.merge", "--adapter", str(adapter),
                        "--out", str(merged)], cwd=ROOT, check=True,
                       env={**os.environ, "HF_HOME": os.environ.get("HF_HOME", "/mnt/c/Users/akosc/.cache/huggingface")})
    return merged


def caps(prompts: dict[str, dict]) -> list[dict]:
    refs = {r["class_key"]: r["tokens"] for r in read_jsonl(LOOP / "cold-start" / "references.jsonl")}
    fallback = int(statistics.median(refs.values())) if refs else 4096
    rows = []
    for key, p in sorted(prompts.items()):
        cap = 2 * refs.get(key, fallback) + 64
        rows.append({**p, "max_tokens": min(cap, vllm_server.MAX_MODEL_LEN - p["prompt_tokens"] - 16)})
    return rows


def sample_and_score(adapter: Path, tag: str, prompts: dict[str, dict], out: Path) -> dict:
    """Rollouts and their scores for one adapter. Returns (class_key, rollout) -> record,
    tier 4 wherever it ran."""
    gens_path = out / "generations.jsonl"
    rows = caps(prompts)
    if len({g["class_key"] for g in read_jsonl(gens_path)}) < len(rows):
        merged = merged_model(adapter, tag)
        vllm_server.start(merged, tag, out / "vllm.log")
        try:
            failed = generate.generate(rows, gens_path, vllm_server.URL, tag)
        finally:
            print(f"vLLM stopped, VRAM in use: {vllm_server.stop()}", flush=True)
        if failed:
            raise SystemExit(f"{failed} prompts failed to sample; rerun to resume")
    gens = read_jsonl(gens_path)

    score_rollouts(gens, out / "scores.jsonl", max_tier=3, workers=SCORE_WORKERS)
    t3 = {(s["class_key"], s["rollout"]): s for s in read_jsonl(out / "scores.jsonl")}
    # Tier 4 only where mutation can exist: PIT needs a green suite, and a rollout that
    # neither compiled nor salvaged to green has nothing to run it on.
    t4_gens = [g for g in gens if (s := t3.get((g["class_key"], g["rollout"])))
               and (s["compiled"] or s.get("salvaged_is_green"))]
    score_rollouts(t4_gens, out / "scores_t4.jsonl", max_tier=4, workers=TIER4_WORKERS)
    t4 = {(s["class_key"], s["rollout"]): s for s in read_jsonl(out / "scores_t4.jsonl")}
    return {**t3, **t4}


# ------------------------------------------------------------------ one round
def cmd_round(args) -> None:
    r = args.round
    prompts = load_prompts("train")
    out = LOOP / f"round-{r}"
    out.mkdir(parents=True, exist_ok=True)
    adapter = MODELS / ("cold-start" if r == 1 else f"round-{r - 1}")
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter}")
    scored = sample_and_score(adapter, "cold-start" if r == 1 else f"round-{r - 1}", prompts, out)
    gens = read_jsonl(out / "generations.jsonl")
    scores = list(scored.values())

    # ---- route
    by_class = defaultdict(list)
    for s in scores:
        by_class[s["class_key"]].append(s)
    killers = {k for k, ss in by_class.items() if any(kills_a_mutant(s) for s in ss)}
    self_scores = [s for s in scores if s["class_key"] in killers and kills_a_mutant(s)]
    report = targets.select(prompts, gens, self_scores, workers=SCORE_WORKERS)
    self_targets = {row["class_key"]: row["selected"] for row in report if row["selected"]}
    fall = sorted(k for k in prompts if k not in self_targets)
    model_best = {row["class_key"]: row["selected"] for row in
                  targets.select(prompts, gens, [s for s in scores if s["class_key"] in fall],
                                 workers=SCORE_WORKERS) if row["selected"]}

    rescues_path = out / "rescues.jsonl"
    rescued = {x["class_key"]: x for x in read_jsonl(rescues_path)}
    gens_by_class = defaultdict(list)
    for g in gens:
        gens_by_class[g["class_key"]].append(g)
    for key in fall:
        if key in rescued or not gens_by_class[key]:
            continue
        start_from = most_tests(gens_by_class[key])
        result = run_harness(prompts[key]["project"], key, f"round{r}-{prompts[key]['project']}-{simple(key)}",
                             provider=TEACHER_PROVIDER, model=TEACHER_MODEL, rollout_text=start_from["text"])
        result["from_rollout"] = start_from["rollout"]
        rescued[key] = result
        with rescues_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")
        print(f"  rescue {simple(key)[:30]:30} {result['status']:8} {result['minutes']:5.1f} min  "
              + (f"{result['suite']['n_tests']} tests" if result["suite"] else "no green suite"), flush=True)

    round_targets = (
        [{"class_key": k, "kind": "self", "round": r, "source": t["source"]} for k, t in self_targets.items()]
        + [{"class_key": k, "kind": "model_best", "round": r, "source": t["source"]} for k, t in model_best.items()]
        + [{"class_key": k, "kind": "harness", "round": r, "source": x["suite"]["source"]}
           for k, x in rescued.items() if x["suite"]])
    write_jsonl(out / "targets.jsonl", round_targets)

    # ---- accumulate every round so far
    own, harness = {}, {}
    for rr in range(1, r + 1):
        for t in read_jsonl(LOOP / f"round-{rr}" / "targets.jsonl"):
            if t["kind"] == "harness":
                harness[t["class_key"]] = t                         # latest wins
            elif t["class_key"] not in own or n_tests(t["source"]) >= n_tests(own[t["class_key"]]["source"]):
                own[t["class_key"]] = t                             # more tests wins
    rows, dropped = dataset_rows(list(own.values()) + list(harness.values()), prompts, LOOP / "scratch")
    targets.write_dataset(rows, out / "data", f"round_{r}")

    summary = {"round": r, "adapter_sampled": str(adapter), **stats.rollout_stats(scores),
               "classes": len(prompts), "self_served": len(self_targets), "fall_through": len(fall),
               "harness_usage_pct": round(100 * len(fall) / len(prompts), 1),
               "rescued_green": sum(1 for x in rescued.values() if x["suite"]),
               "targets_this_round": len(round_targets), "dataset_rows": len(rows),
               "dataset_dropped_over_cutoff": dropped,
               "median_target_tests": statistics.median(n_tests(t["source"]) for t in round_targets)
               if round_targets else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    new_adapter = MODELS / f"round-{r}"
    if not (new_adapter / "adapter_config.json").exists():
        cfg = llamafactory.render("rft_round", out / "data", f"round_{r}", new_adapter,
                                  adapter=MODELS / "cold-start")
        llamafactory.train(cfg, out / "train.yaml", out / "train.log")
    print(f"round {r} adapter: {new_adapter}")


# ------------------------------------------------------------------ held-out
def cmd_eval(args) -> None:
    """Measurement only. Nothing here may reach a dataset: no targets, no rescue."""
    prompts = load_prompts("heldout")
    out = LOOP / f"eval-{args.tag}"
    scored = sample_and_score(Path(args.adapter), args.tag, prompts, out)
    summary = {"tag": args.tag, "adapter": args.adapter, **stats.rollout_stats(list(scored.values())),
               "prompts_any_killer": len({s["class_key"] for s in scored.values() if kills_a_mutant(s)})}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m training.loop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts").set_defaults(fn=cmd_prompts)
    cs = sub.add_parser("cold-start")
    cs.add_argument("--parallel", type=int, default=4, help="projects run concurrently (one class each)")
    cs.set_defaults(fn=cmd_cold_start)
    rd = sub.add_parser("round")
    rd.add_argument("round", type=int)
    rd.set_defaults(fn=cmd_round)
    ev = sub.add_parser("eval")
    ev.add_argument("--adapter", required=True)
    ev.add_argument("--tag", required=True)
    ev.set_defaults(fn=cmd_eval)
    args = ap.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
