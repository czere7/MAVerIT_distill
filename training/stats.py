"""What a round reports: the rates the previous rounds were judged on, plus harness usage.

Compile and green describe the suite AS GENERATED (salvage never feeds a headline).
Coverage and mutation are medians over compiling rollouts. Spread -- within-prompt stdev
of graded quality and distinct values per prompt -- is how a STaR loop's collapse would
show, since a group scoring alike carries no signal. Median @Test is a diagnostic for the
deletion ratchet, not a target.
"""

from __future__ import annotations

import statistics
from collections import defaultdict


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 2) if values else None


def rollout_stats(scores: list[dict]) -> dict:
    usable = [s for s in scores if s.get("usable_for_training")]
    n = len(usable)
    compiled = [s for s in usable if s["compiled"]]
    by_prompt = defaultdict(list)
    for s in usable:
        by_prompt[s["class_key"]].append(s.get("quality_graded") or 0.0)
    spreads = [statistics.pstdev(v) for v in by_prompt.values() if len(v) > 1]
    return {
        "rollouts": len(scores), "usable": n,
        "compiled_pct": round(100 * len(compiled) / n, 1) if n else None,
        "green_pct": round(100 * sum(s["is_green"] for s in usable) / n, 1) if n else None,
        "terminated_pct": round(100 * sum(s["terminated"] for s in usable) / n, 1) if n else None,
        "prompts": len(by_prompt),
        "prompts_any_compile": len({s["class_key"] for s in compiled}),
        "prompts_any_green": len({s["class_key"] for s in usable if s["is_green"]}),
        "median_tests": _median([s["n_tests"] for s in usable]),
        "median_line_coverage_compiling": _median([s.get("line_coverage_pct") for s in compiled]),
        "median_mutation_green": _median([s.get("mutation_score_pct") for s in compiled if s["is_green"]]),
        "mean_quality_graded": round(statistics.mean(q for v in by_prompt.values() for q in v), 4)
        if by_prompt else None,
        "within_prompt_stdev": round(statistics.mean(spreads), 4) if spreads else None,
        "distinct_values_per_prompt": round(statistics.mean(len(set(v)) for v in by_prompt.values()), 2)
        if by_prompt else None,
    }
