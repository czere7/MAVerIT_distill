"""Running the harness on one class, for the cold start and for the fall-through.

COLD START: teacher mode, `main.py --class X`. DeepSeek writes the first draft and the graph
repairs and mutation-hardens it; the final green suite is the SFT target.

FALL-THROUGH: when no rollout of a class kills a mutant, the harness takes over.

The rollout with the MOST tests goes into the harness in rollout mode (`main.py --class X
--rollout FILE`): the teacher writes nothing first; DeepSeek repairs and extends what the
model wrote, through the same graph that produced the cold start. One harness, two modes --
if the cold start and the fall-through used different pipelines, the paper could not say
which harness was compiled into the weights.

Why the most tests: the mutation writer only appends and replaces tests, so the rollout
with the most material gives it the most to keep, and it is the rollout least like the
husks the deletion ratchet produces.

The harness runs IN the project directory, one class at a time: two runs on the same
project would write into the same src/test and read each other's reports. Before a run
src/test/java is emptied; after it, the harness's final suite is read from its log and the
test file removed again, so the project is left as it was found.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from training.pools import project_dir
from training.scoring import java, pertest

HARNESS = Path(__file__).resolve().parents[1]
WALL_CLOCK_MIN = 60


def clear_tests(project: Path) -> None:
    test_java = project / "src" / "test" / "java"
    for path in list(test_java.rglob("*.java")) if test_java.exists() else []:
        path.unlink()


def most_tests(generations: list[dict]) -> dict:
    """The rollout the harness starts from: most @Test methods, ties to the lower index."""
    return max(generations, key=lambda g: (len(pertest.test_spans(
        java.strip_fence_lenient(g["text"]))), -g["rollout"]))


def final_suite(run_dir: Path) -> dict | None:
    """The harness's surviving suite: the last green restore point in its log.

    compiler_node stores last_compilable_test_class only when the suite is green, and
    faulty_test_cleanup restores or trims to green, so the last row carrying one is the
    suite the class ended with. None when nothing was ever green.
    """
    log = run_dir / "log.jsonl"
    if not log.exists():
        return None
    rows = [json.loads(l) for l in log.open(encoding="utf-8") if l.strip()]
    suite, mutation = None, None
    for row in rows:
        if row.get("last_compilable_test_class"):
            suite = row["last_compilable_test_class"]
        if row.get("suite_green") and row.get("mutation") not in (None, "None"):
            mutation = float(row["mutation"])
    if not suite:
        return None
    return {"source": suite, "mutation_score_pct": mutation,
            "n_tests": len(pertest.test_spans(suite)), "log_rows": len(rows)}


def run_harness(project: str, class_key: str, run_id: str, *, provider: str, model: str,
                rollout_text: str | None = None, reasoning_effort: str | None = None) -> dict:
    """One harness run: teacher mode, or rollout mode when `rollout_text` is given.
    Returns the outcome, with the surviving suite or None."""
    project_path = project_dir(project)
    cmd = [sys.executable, "-u", "main.py", "--project", str(project_path),
           "--class", f"src/main/java/{class_key}.java",
           "--provider", provider, "--model", model, "--run-id", run_id, "--max-restarts", "1"]
    if rollout_text is not None:
        inputs = HARNESS / "logs" / "rollout-inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        rollout_file = inputs / f"{run_id}.java"
        rollout_file.write_text(java.strip_fence_lenient(rollout_text), encoding="utf-8")
        cmd += ["--rollout", str(rollout_file)]
    if reasoning_effort:
        cmd += ["--reasoning-effort", reasoning_effort]

    clear_tests(project_path)
    started = time.monotonic()
    run_dir = HARNESS / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as out:
        try:
            proc = subprocess.run(cmd, cwd=HARNESS, stdout=out, stderr=subprocess.STDOUT,
                                  timeout=WALL_CLOCK_MIN * 60)
            status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
    clear_tests(project_path)
    suite = final_suite(run_dir)
    return {"class_key": class_key, "project": project, "run_id": run_id, "status": status,
            "minutes": round((time.monotonic() - started) / 60, 1), "suite": suite}
