"""Live check of the edit tools with Qwen3.8, through the real graph nodes.

Takes the last failing suite of the aborted ablation run (JsonParserDelegate, jackson-core-clean),
then: repair (replace) -> compile -> if green, mutation writer (append_test / replace_test)
-> compile. Prints what the model did, what it cost, and whether it compiled.
Run from the harness root:  python smokes/live_edit_tools.py
"""
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import config, initial_metric_state, is_concrete_class  # noqa: E402

PROJECT = Path(r"C:\Users\akosc\IdeaProjects\jackson-core-clean")
CUT = "JsonParserDelegate.java"
SOURCE = Path("ablation/_aborted-2026-09-27/abl-v1-JsonParserDelegate/log.jsonl")
RUN_ID = "live-edit-tools"

config.update(WORKING_DIRECTORY=str(PROJECT), PROVIDER="ollama", MODEL="qwen3.8:latest",
              RETRIEVER="v4")

from Extractor import Extractor  # noqa: E402
from nodes.compiler_node import compiler_node  # noqa: E402
from nodes.mutation_test_writer_node import mutation_test_writer_node  # noqa: E402
from nodes.test_repair_node import test_repair_node  # noqa: E402


def calls_made(node: str) -> list[str]:
    out = []
    for line in (Path(RUN_ID) / "prompt-response-pairs.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["node_name"].startswith(node):
            r = d["response"]["data"]
            md = r.get("response_metadata") or {}
            tc = r.get("tool_calls") or []
            out.append(f"{d['node_name']}: {md.get('eval_count')} tokens out, "
                       f"{(md.get('total_duration') or 0) / 6e10:.1f} min, tool calls: "
                       + ("; ".join(f"{c['name']}({', '.join(f'{k}={str(v)[:60]!r}' for k, v in c['args'].items())})" for c in tc) or "none"))
    return out


def main() -> int:
    if Path(RUN_ID).exists():
        shutil.rmtree(RUN_ID)
    with contextlib.redirect_stdout(io.StringIO()):
        files = Extractor(str(PROJECT)).extract_all()
    tests = [f for f in files if is_concrete_class(f)]
    cut = next(f for f in tests if f.file_path.endswith("\\" + CUT))
    failing = [json.loads(l) for l in SOURCE.read_text(encoding="utf-8").splitlines()][-1]
    state = {**initial_metric_state(), "all_files": files, "all_test_files": [cut],
             "current_class_index": 0, "run_id": RUN_ID, "test_class": failing["test_class"],
             "compiler_feedback": failing["compiler_feedback"], "compiler_success": False,
             "repair_attempts": 0, "mutation_iterations": 0, "input_tokens": 0,
             "output_tokens": 0, "total_tokens": 0, "last_compilable_test_class": ""}

    # The logged feedback came from a project missing JUnit 4 and Mockito; compile first so
    # the repair works from the suite's real errors on the fixed project.
    state.update(compiler_node(state))
    print(f"== initial compile: compiled={state['compiler_success']} green={state.get('suite_green')} "
          f"tests={state.get('tests_run')} failed={state.get('tests_failed')}")
    for round_ in range(1, 4):
        if state.get("suite_green"):
            break
        t = time.time()
        state.update(test_repair_node(state))
        print(f"\n== repair round {round_}: {time.time() - t:.0f}s")
        t = time.time()
        state.update(compiler_node(state))
        print(f"== compile after repair ({time.time() - t:.0f}s): compiled={state['compiler_success']} "
              f"green={state.get('suite_green')} tests={state.get('tests_run')} "
              f"failed={state.get('tests_failed')} mutation={state.get('mutation_current')}")
    for c in calls_made("test_repair_node"):
        print("  ", c)
    if not state.get("suite_green"):
        print(state["compiler_feedback"][-1500:])
        return 1

    t = time.time()
    before = state["test_class"].count("@Test")
    state.update(mutation_test_writer_node(state))
    print(f"\n== mutation writer: {time.time() - t:.0f}s, @Test {before} -> {state['test_class'].count('@Test')}")
    for c in calls_made("mutation_test_writer_node"):
        print("  ", c)
    prev = state.get("mutation_current")
    t = time.time()
    state.update(compiler_node(state))
    print(f"== compile after mutation writer ({time.time() - t:.0f}s): compiled={state['compiler_success']} "
          f"green={state.get('suite_green')} tests={state.get('tests_run')} "
          f"mutation {prev} -> {state.get('mutation_current')}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        for java in (PROJECT / "src" / "test").rglob("*.java"):
            java.unlink()
