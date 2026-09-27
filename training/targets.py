"""Turn scored rollouts into training targets: the model's own output, filtered by the
verifier. Ported from ~/maverit-preference/scripts/build_rft.py.

THE TARGET IS THE SALVAGED SUBSET, not the raw rollout. A rollout that did not compile still
has tests that do; drop the broken ones and what remains is a strict subset of what the
model wrote -- nothing is put in its mouth -- and it terminates by construction, which is
the defect four SFT runs could not fix.

EVERY TARGET IS COMPILE-VERIFIED, AND SALVAGE ITERATES. The stored compiler output is cut
to its last 4000 characters, so one-shot attribution from it marks broken tests clean
(measured: 48% of candidates then failed verification). Instead each candidate is compiled,
the FRESH output is attributed, the blamed tests are dropped, and it is compiled again,
until it passes or nothing is left. Each round removes at least one test, so it terminates;
most converge in one or two.

TWO GUARDS, both at target-build time:

  min_fraction   THE DELETION RATCHET. Targets are made by removing tests, so a subset that
                 kept 1 of 27 says "you wrote 27; the answer was one". That compounds round
                 on round (median @Test written went 15 -> 5 -> 5 -> 1). A candidate keeping
                 less than this share is rejected and the prompt falls to its next-best
                 rollout.
  coverage gate  the verified subset runs under JaCoCo and is rejected if it touches
                 nothing. LINE coverage, not branch: a branch veto rejected 18 rollouts that
                 had executed up to 45.8% of their class, and a branchless class reports
                 100% branch coverage for free.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training.pools import pool_for
from training.scoring import java, maven, pertest, reports
from training.scoring.workspace import install_test_class, remove_test_class

TRIES_PER_PROMPT = 3
MAX_SALVAGE_ROUNDS = 4
FENCE_OPEN, FENCE_CLOSE = "```java\n", "\n```"


def measure_coverage(module: Path, cut_in_slot: Path):
    """Run the verified subset under JaCoCo and read the class under test back out.

    The exec file is deleted first. The agent appends by default and pool slots are reused
    across candidates, so without this a husk inherits the coverage of whatever ran in the
    same slot before it -- which would make the veto silently stop firing.
    """
    target = module / "target"
    for stale in (target / "jacoco.exec", target / "site" / "jacoco" / "jacoco.xml"):
        try:
            stale.unlink()
        except OSError:
            pass
    try:
        cut_fqn = java.get_java_fully_qualified_name(cut_in_slot.read_text(encoding="utf-8"))
    except (OSError, RuntimeError):
        return None
    jac = maven.run_jacoco(module)
    if jac["timed_out"]:
        return None                    # no evidence is not evidence of no coverage
    found = reports.find_jacoco_xml_reports(module)
    if not found:
        return None
    return reports.calculate_coverage_for_class(found, cut_fqn)


def iterative_salvage(source: str, pool, key: str, test_file: str,
                      coverage_gate: bool = True) -> tuple[str, dict]:
    """Compile, drop what javac blames, repeat. Returns ("", info) if nothing survives.

    Attribution reads the output of the compile that just ran, so it is never truncated.
    """
    n_start = len(pertest.test_spans(source))
    dropped: list[str] = []
    info = {"n_tests": n_start, "n_broken": 0, "n_skeleton": 0, "dropped": dropped,
            "rounds": 0, "compiles": 0, "line_coverage_pct": None,
            "branch_coverage_pct": None, "vetoed": False}
    if not n_start:
        return "", info

    current = source
    for round_no in range(1, MAX_SALVAGE_ROUNDS + 1):
        try:
            pkg = java.get_java_package_name(current)
            cls = java.get_java_entity_name(current)
        except RuntimeError:
            return "", info
        cov = None
        with pool.acquire() as slot:
            cut_in_slot = slot / "src" / "main" / "java" / f"{key}.java"
            module = maven.get_maven_module_directory(cut_in_slot, slot)
            path = install_test_class(module, current, pkg, cls)
            res = maven.run_test_compile(module)
            if res["ok"] and coverage_gate:
                cov = measure_coverage(module, cut_in_slot)
            remove_test_class(path)
        info["compiles"] += 1
        info["rounds"] = round_no
        if res["ok"]:
            info["n_broken"] = len(dropped)
            if cov is not None:
                info["line_coverage_pct"] = cov.line_pct
                info["branch_coverage_pct"] = cov.branch_pct
                if cov.line_pct is not None and cov.line_pct <= 0.0:
                    info["vetoed"] = True
                    return "", info
            return current, info

        marks = pertest.attribute(current, res["combined_result"], test_file)
        if marks.skeleton_errors:
            info["n_skeleton"] = len(marks.skeleton_errors)
            return "", info            # dropping tests cannot rescue a bad import
        if not marks.broken or not marks.n_tests:
            return "", info            # nothing attributable left to remove
        dropped.extend(sorted(marks.broken))
        current = pertest.drop_tests(current, marks.broken)
        if not pertest.test_spans(current):
            return "", info            # every test was blamed
    return "", info


def select(prompts: dict[str, dict], generations: list[dict], scores: list[dict], *,
           score_field: str = "quality_graded", min_quality: float = 0.30,
           min_fraction: float = 0.35, coverage_gate: bool = True,
           workers: int = 6) -> list[dict]:
    """The best verifiable target per prompt, or none. One report row per ranked prompt.

    `prompts` maps class_key -> {"project", "prompt", ...}. A row's `selected` carries the
    target source when a candidate verified. Rows come back in (project, class_key) order,
    whatever order the workers finished in, so a rebuilt set is byte-comparable.
    """
    gens = {(g["class_key"], g["rollout"]): g for g in generations}
    ranked: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        q = s.get(score_field)
        if q is not None and q >= min_quality and s["class_key"] in prompts:
            ranked[s["class_key"]].append(s)
    for key in ranked:
        ranked[key].sort(key=lambda r: -(r.get(score_field) or 0.0))

    results: dict[str, dict] = {}
    guard = threading.Lock()

    def salvage_one(key: str) -> None:
        project = prompts[key]["project"]
        pool = pool_for(project, workers)
        simple = key.split("/")[-1]
        chosen, attempts = None, []
        for cand in ranked[key][:TRIES_PER_PROMPT]:
            gen = gens[(key, cand["rollout"])]
            subset, info = iterative_salvage(java.strip_fence_lenient(gen["text"]), pool, key,
                                             f"{simple}Test.java", coverage_gate=coverage_gate)
            kept_frac = ((info["n_tests"] - info["n_broken"]) / info["n_tests"]
                         if info["n_tests"] else 0.0)
            if subset.strip() and kept_frac < min_fraction:
                attempts.append({"rollout": cand["rollout"],
                                 "why": f"kept only {kept_frac:.0%} of its tests",
                                 "rounds": info["rounds"]})
                continue
            if subset.strip():
                chosen = {"rollout": cand["rollout"], "quality": cand.get(score_field) or 0.0,
                          "score_field": score_field, "source": subset,
                          "kept_fraction": round(kept_frac, 3),
                          "n_kept_tests": info["n_tests"] - info["n_broken"], **info}
                attempts.append({"rollout": cand["rollout"], "why": "VERIFIED",
                                 "rounds": info["rounds"]})
                break
            attempts.append({"rollout": cand["rollout"],
                             "why": ("compiles but covers 0% of the class" if info["vetoed"]
                                     else "skeleton fault" if info["n_skeleton"]
                                     else "nothing salvageable"),
                             "rounds": info["rounds"],
                             "line_coverage_pct": info["line_coverage_pct"]})
        with guard:
            results[key] = {"class_key": key, "project": project, "attempts": attempts,
                            "selected": chosen}
            print(f"  [{len(results):>3}/{len(ranked)}] {simple[:28]:<28} "
                  + (f"kept {chosen['n_kept_tests']}/{chosen['n_tests']} tests"
                     if chosen else "NO USABLE CANDIDATE"), flush=True)

    # One project at a time, so each pool's slots are the only Maven builds running.
    keys_by_project: dict[str, list[str]] = defaultdict(list)
    for key in ranked:
        keys_by_project[prompts[key]["project"]].append(key)
    for project in sorted(keys_by_project):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(salvage_one, sorted(keys_by_project[project])))
    return [results[k] for p in sorted(keys_by_project) for k in sorted(keys_by_project[p])]


def fence(source: str) -> str:
    """Targets are fenced: a bare `}` made "stop here" one rare token against millions, and
    the fence turns it into a ramp. The harness's strip_markdown_code_fence unwraps it."""
    return FENCE_OPEN + source.rstrip() + FENCE_CLOSE


def write_dataset(rows: list[dict], out_dir: Path, name: str) -> Path:
    """LLaMA-Factory alpaca format plus dataset_info.json. `rows` carry instruction/output;
    keys starting with "_" are bookkeeping and are not written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"instruction": row["instruction"], "input": "",
                                 "output": row["output"]}) + "\n")
    (out_dir / "dataset_info.json").write_text(json.dumps({
        name: {"file_name": path.name,
               "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
    }, indent=2), encoding="utf-8")
    return path
