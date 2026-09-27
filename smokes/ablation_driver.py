"""Retrieval ablation: v1 (the harness as it was) against v4 (collaborator index), Qwen3.8.

Each class in smokes/ablation_classes.json runs once per arm, from scratch (the teacher writes
the first draft), as its own `main.py --class` process -- strictly one at a time, one GPU.
Arms alternate per class (v1,v4 / v4,v1 / ...) so drift in the machine or the server spreads
over both. Before every run the project's src/test is emptied; after it, the final suite is
copied into the run's directory. A run past WALL_CLOCK_MIN is killed and recorded as such.

Resumable: a run whose directory already has result.json is skipped.

    python smokes/ablation_driver.py [--provider deepseek --model deepseek-v4-flash --tag deepseek]

Results go to ablation/<tag>/, so runs with different teachers never mix.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
PROJECTS = Path(r"C:\Users\akosc\IdeaProjects")
PY = r"C:\Users\akosc\Desktop\PyCharmMiscProject\PyCharmMiscProject\venv\Scripts\python.exe"
ABLATION = HARNESS / "ablation"
WALL_CLOCK_MIN = 60


def clear_tests(project: Path, keep_into: Path | None) -> None:
    test_dir = project / "src" / "test"
    for java in list(test_dir.rglob("*.java")) if test_dir.exists() else []:
        if keep_into is not None:
            keep_into.mkdir(parents=True, exist_ok=True)
            shutil.copy2(java, keep_into / java.name)
        java.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="qwen3.8:latest")
    ap.add_argument("--tag", default="qwen")
    ap.add_argument("--reasoning-effort", default="high", choices=("off", "low", "medium", "high"))
    ap.add_argument("--arms", default="v1,v4", help="retrievers to run, e.g. v1,v4,v5; their order rotates per class")
    ap.add_argument("--classes", default="smokes/ablation_classes.json", help="JSON list of {project, class}")
    args = ap.parse_args()
    OUT = ABLATION / args.tag
    classes = json.loads((HARNESS / args.classes).read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    log = (OUT / "driver.log").open("a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    say(f"start: {len(classes)} classes x {len(arms)} arms {arms}, {args.provider}/{args.model}, reasoning {args.reasoning_effort}")
    for i, entry in enumerate(classes):
        project = PROJECTS / entry["project"]
        simple = entry["class"].rsplit(".", 1)[-1]
        for arm in arms[i % len(arms):] + arms[:i % len(arms)]:
            run_id = f"abl-{args.tag}-{arm}-{simple}"
            run_dir = HARNESS / run_id
            final = OUT / run_id
            if (final / "result.json").exists():
                say(f"{run_id}: done already, skipping")
                continue
            if run_dir.exists():
                shutil.rmtree(run_dir)
            clear_tests(project, None)
            say(f"{run_id}: starting ({entry['project']})")
            started = time.monotonic()
            with (OUT / f"{run_id}.log").open("w", encoding="utf-8") as out:
                proc = subprocess.Popen(
                    [PY, "-u", "main.py", "--project", str(project), "--class", entry["class"],
                     "--provider", args.provider, "--model", args.model, "--retriever", arm,
                     "--reasoning-effort", args.reasoning_effort,
                     "--run-id", run_id, "--max-restarts", "1"],
                    cwd=HARNESS, stdout=out, stderr=subprocess.STDOUT)
                try:
                    code = proc.wait(timeout=WALL_CLOCK_MIN * 60)
                    status = "finished" if code == 0 else f"exit {code}"
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    status = f"killed after {WALL_CLOCK_MIN} min"
            minutes = (time.monotonic() - started) / 60
            final.mkdir(parents=True, exist_ok=True)
            if run_dir.exists():
                for item in run_dir.iterdir():
                    dest = final / item.name
                    if dest.exists():
                        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                    shutil.move(str(item), str(dest))
                run_dir.rmdir()
            clear_tests(project, final / "final_suite")
            (final / "result.json").write_text(json.dumps({
                "run_id": run_id, "arm": arm, "project": entry["project"], "class": entry["class"],
                "provider": args.provider, "model": args.model, "reasoning_effort": args.reasoning_effort,
                "status": status, "minutes": round(minutes, 1)}, indent=1), encoding="utf-8")
            say(f"{run_id}: {status} after {minutes:.1f} min")
    say("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
