from typing import TYPE_CHECKING

from Extractor import Extractor
from retrieval import prepare
from utils import get_working_directory, initial_metric_state, is_concrete_class, retrieve_checkpoint

if TYPE_CHECKING:
    from AgentState import AgentState


def file_retriever_node(agent_state: "AgentState") -> dict:
    project_dir = get_working_directory()
    print(f"[file_retriever_node] Reading Java source files from: {project_dir}")
    extracted_files = Extractor(str(project_dir)).extract_all()
    test_target_files = [
        source_file
        for source_file in extracted_files
        if is_concrete_class(source_file)
    ]

    if not test_target_files:
        raise RuntimeError(f"No concrete production Java classes found under WORKING_DIRECTORY: {project_dir}")

    # ONE CLASS, when the caller asked for one. The list is narrowed rather than the index
    # moved, so class_advancer sees total == 1 and the router ends the run after it -- no
    # node has to know it is in single-class mode. `all_files` stays whole: the repair and
    # mutation writers draw collaborator context from it.
    target = agent_state.get("target_class_path")
    if target:
        test_target_files = [f for f in test_target_files if f.file_path == target]
        if not test_target_files:
            raise RuntimeError(f"Target class is not a concrete class under the project: {target}")

    print(
        f"[file_retriever_node] Loaded {len(extracted_files)} Java context files; "
        f"{len(test_target_files)} concrete classes will be tested."
    )

    # Once per run, not per class: v4 needs the project compiled and its collaborator index.
    print(f"[file_retriever_node] {prepare(project_dir)}")

    checkpoint = retrieve_checkpoint()
    return {
        "all_files": extracted_files,
        "all_test_files": test_target_files,
        "current_class_index": 0 if target else checkpoint.get("class_index"),
        "run_id": agent_state.get("run_id") or checkpoint.get("run_id"),
        "input_tokens": checkpoint.get("input_tokens"),
        "output_tokens": checkpoint.get("output_tokens"),
        "total_tokens": checkpoint.get("total_tokens"),
        "runtime": checkpoint.get("runtime"),
        "assert_less_test_amount": checkpoint.get("assert_less_test_amount", 0),
        **initial_metric_state(),
    }
