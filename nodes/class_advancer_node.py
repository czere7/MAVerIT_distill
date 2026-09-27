import json
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from retrieval import retriever
from utils import advance_to_next_class, persist_checkpoint, config, run_dir, write_to_log_file
import os

if TYPE_CHECKING:
    from AgentState import AgentState


def class_advancer_node(agent_state: "AgentState") -> dict:
    next_index = agent_state["current_class_index"] + 1
    total = len(agent_state.get("all_test_files", []))
    runtime = round((time() - agent_state.get("current_test_start_time", 0)) * 1000)
    runtime_total = agent_state.get("runtime", 0.0) + runtime
    _write_metadata(agent_state.get("run_id"), runtime_total)
    if next_index == total:
        if os.path.exists(config.get("CLASS_INDEX_TMP_FILE", "tmp.txt")):
            Path(config.get("CLASS_INDEX_TMP_FILE", "tmp.txt")).unlink()
    else:
        persist_checkpoint(agent_state, next_index)
    print(f"[class_advancer_node] Advancing to next class: index {next_index + 1}/{total}, run_id: {agent_state['run_id']}.")
    # write_to_log_file(f"[class_advancer_node] Advancing to next class: index {next_index + 1}/{total}.\n---\n", str(agent_state.get("run_id")))
    return advance_to_next_class(agent_state, runtime_total)

def _write_metadata(run_id, runtime):
    metadata_path = run_dir(run_id) / "metadata.json"
    if not metadata_path.exists():
        metadata = {
            "model": config.get("MODEL", ""),
            "provider": config.get("PROVIDER", ""),
            "retriever": retriever(),
            "reasoning_effort": config.get("REASONING_EFFORT") or "high",
            "working_directory": config.get("WORKING_DIRECTORY", ""),
            "max_repair_attempts": config.get("MAX_REPAIR_ATTEMPTS", ""),
            "total_runtime_ms": runtime,
            "run_id": run_id,
            "thresholds": {
                "coverage": config.get("COVERAGE_IMPROVEMENT_THRESHOLD", ""),
                "mutation": config.get("MUTATION_IMPROVEMENT_THRESHOLD", ""),
            },
            "max_iterations": {
                "coverage": config.get("MAX_COVERAGE_ITERATIONS", ""),
                "mutation": config.get("MAX_MUTATION_ITERATIONS", ""),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
