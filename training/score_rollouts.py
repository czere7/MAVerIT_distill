"""Score every sampled rollout with the verifier. Ported from middleband_score.py.

LENIENT FENCE. A rollout that hits its token cap never emits the closing ```, so its first
line keeps ```java and it cannot parse for a reason that has nothing to do with the Java;
326 of 928 rollouts in the first middle-band run carried a live fence into tier 0.

EVERY FIELD THE SCORE CARRIES IS WRITTEN. The old recorder wrote a fixed field list that was
never extended, and three passes of 928 rollouts landed with no line coverage and no
green-subset numbers -- the husk rate a round existed to measure could not be computed after
the fact. The record here is built from the ScoreResult itself, so a new field cannot be
silently dropped.

PARALLEL ACROSS THE POOL: one Maven per slot; serial scoring left ~10 of 12 cores idle. NOT
SAFE TO WIDEN AT TIER 4, where run_pitest passes -Dthreads=12 and N parallel PIT runs
oversubscribe the machine N-fold.

Resumable on (class_key, rollout).
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training.pools import class_path, pool_for, project_dir
from training.quality import quality, quality_v2
from training.scoring import score


def _rounded(fn, result) -> float | None:
    """None for an INFRA_ERROR: quality() refuses to score one, and a 0.0 here would make
    our own failure look like the model's."""
    try:
        return round(fn(result), 6)
    except ValueError:
        return None


def record(row: dict, result, seconds: float) -> dict:
    fields = dataclasses.asdict(result)
    fields["status"] = result.status.value
    fields["compiler_output"] = result.compiler_output[:8000]
    return {
        "class_key": row["class_key"], "project": row["project"], "rollout": row["rollout"],
        "finish_reason": row["finish_reason"], "chars": row["chars"], **fields,
        "usable_for_training": result.usable_for_training, "is_green": result.is_green,
        "pass_rate": result.pass_rate, "compiled_fraction": result.compiled_fraction,
        "effective_coverage_pct": result.effective_coverage_pct,
        "salvaged_is_green": result.salvaged_is_green,
        "quality_binary": _rounded(lambda r: quality(r, graded_compile=False), result),
        "quality_graded": _rounded(lambda r: quality(r, graded_compile=True), result),
        "quality_v2": _rounded(quality_v2, result),
        "score_seconds": seconds, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def kills_a_mutant(rec: dict) -> bool:
    """The fall-through rule's test: does anything this rollout could become a target from
    kill at least one mutant -- the suite as generated, its green subset, or its salvaged
    subset. Needs tier 4; below it, mutation is not measured and this is False."""
    return any((rec.get(k) or 0.0) > 0.0 for k in (
        "mutation_score_pct", "green_subset_mutation_score_pct", "salvaged_mutation_score_pct"))


def score_rollouts(generations: list[dict], output: Path, max_tier: int, workers: int = 6) -> Counter:
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["class_key"], r["rollout"]) for r in map(json.loads, output.open(encoding="utf-8"))} \
        if output.exists() else set()
    todo = [g for g in generations if (g["class_key"], g["rollout"]) not in done]
    print(f"rollouts {len(generations)}, already scored {len(done)}, to score {len(todo)}, "
          f"tier {max_tier}, {workers} workers", flush=True)
    if max_tier >= 4 and workers > 2:
        print(f"NOTE: tier 4 runs PIT with -Dthreads=12; {workers} workers oversubscribe "
              f"12 cores", flush=True)

    tally: Counter = Counter()
    lock = threading.Lock()
    started = time.monotonic()
    seen = 0
    with output.open("a", encoding="utf-8") as handle:
        for project in sorted({g["project"] for g in todo}):
            pool = pool_for(project, workers)

            def run_one(row, project=project, pool=pool):
                case_started = time.monotonic()
                with pool.acquire() as slot:
                    result = score(row["text"], class_path(project, row["class_key"]), slot,
                                   project_dir(project), max_tier=max_tier,
                                   terminated=row["finish_reason"] == "stop", lenient_fence=True)
                return row, result, round(time.monotonic() - case_started, 1)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for row, result, seconds in executor.map(
                        run_one, [g for g in todo if g["project"] == project]):
                    # One append per result: interleaved writes would corrupt the jsonl,
                    # and the resume key is read back out of it.
                    with lock:
                        seen += 1
                        tally[result.status.value] += 1
                        handle.write(json.dumps(record(row, result, seconds)) + "\n")
                        handle.flush()
                        rate = (time.monotonic() - started) / seen
                        flag = ("GREEN" if result.is_green else
                                "compiled" if result.compiled else result.status.value)
                        print(f"  [{seen:4d}/{len(todo)}] "
                              f"{row['class_key'].split('/')[-1][:24]:<24} #{row['rollout']} "
                              f"{flag:<14} {seconds:.0f}s "
                              f"(eta {rate * (len(todo) - seen) / 60:.0f}m)", flush=True)
    return tally
