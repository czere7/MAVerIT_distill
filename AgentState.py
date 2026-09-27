from typing import TypedDict, Sequence, Annotated, List

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from utils import SourceCodeFileData


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    run_id: str
    # --- per-run inputs, set by the CLI in the initial state ------------------------
    target_class_path: str          # run exactly this class, then end
    rollout_source: str             # initial suite supplied by the caller, not the teacher
    runtime: float
    current_test_start_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    current_class_index: int
    all_files: List[SourceCodeFileData]
    all_test_files: List[SourceCodeFileData]
    test_class: str
    compiler_success: bool          # the test class COMPILED -- no longer "and passed"
    suite_green: bool               # compiled AND every test passed; what PIT requires
    tests_run: int
    tests_passed: int
    tests_failed: int
    failing_tests: List[str]        # names, so feedback can say which ones broke
    jacoco_ok: bool                 # False when the jacoco goal itself failed
    compiler_feedback: str
    test_file_path: str
    last_compilable_test_class: str
    last_compilable_test_file_path: str
    coverage_previous: float
    coverage_current: float
    coverage_delta: float
    coverage_improved_significantly: bool
    coverage_feedback: str
    mutation_previous: float
    mutation_current: float
    mutation_delta: float
    mutation_improved_significantly: bool
    mutation_feedback: str
    active_validation_phase: str
    repair_attempts: int
    coverage_iterations: int
    mutation_iterations: int
    assert_less_test_amount: int
