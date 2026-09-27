from typing import TYPE_CHECKING

from edit_tools import MUTATION_TOOLS, SUITE_SLOT, apply_mutation_calls, run_edit_turns
from retrieval import prompt_context
from utils import (
    extract_response_content,
    format_source_files_for_prompt,
    format_test_examples_for_prompt,
    get_current_class_under_test,
    get_int_config,
    get_nearby_test_examples,
    get_relevant_source_files,
    load_prompt,
    strip_markdown_code_fence, write_log, strip_comments_for_long_prompt,
)

if TYPE_CHECKING:
    from AgentState import AgentState


def mutation_test_writer_node(agent_state: "AgentState") -> dict[str, str]:
    print(
        f"[mutation_test_writer_node] Adding mutation-focused tests "
        f"(iteration {agent_state.get('mutation_iterations', 0) + 1})."
    )
    prompt = _build_prompt(agent_state)
    # The model edits by tool calls (append_test / replace_test) instead of regenerating the
    # suite: in the DeepSeek logs 99.2% of existing tests came back unchanged anyway.
    updated_test_class, input_tokens, output_tokens, how = run_edit_turns(
        prompt, agent_state.get("test_class", "").strip(), MUTATION_TOOLS, apply_mutation_calls,
        agent_state.get("run_id", ""), "mutation_test_writer_node")

    iterations = agent_state.get("mutation_iterations", 0) + 1
    if not updated_test_class:
        # No usable edit after every turn (or an empty answer even without thinking). This
        # node only runs on a green suite, so hand that suite back unchanged with the
        # iteration budget spent: it recompiles green and routing advances the class with it.
        print(f"[mutation_test_writer_node] No usable edit ({how}). Keeping the current "
              "green suite and ending mutation iterations for this class.")
        updated_test_class = agent_state.get("test_class", "")
        iterations = get_int_config("MAX_MUTATION_ITERATIONS")

    # write_log(f"[mutation_test_writer_node] Produced updated test class with {len(updated_test_class.splitlines())} line(s).", response.usage_metadata, agent_state)

    return {
        "test_class": updated_test_class,
        "active_validation_phase": "mutation",
        "repair_attempts": 0,
        "mutation_iterations": iterations,
        "input_tokens": input_tokens + agent_state.get('input_tokens'),
        "output_tokens": output_tokens + agent_state.get('output_tokens'),
        "total_tokens": input_tokens + output_tokens + agent_state.get('total_tokens', 0),
    }


def _build_prompt(agent_state: "AgentState") -> str:
    test_class = agent_state.get("test_class", "").strip()
    mutation_feedback = agent_state.get("mutation_feedback", "").strip()

    if not test_class:
        raise RuntimeError("Cannot improve mutation score because AgentState['test_class'] is empty.")
    if not mutation_feedback:
        raise RuntimeError("Cannot improve mutation score because AgentState['mutation_feedback'] is empty.")

    class_under_test = get_current_class_under_test(agent_state)
    class_under_test_text, relevant_source_files = prompt_context(agent_state)
    nearby_test_examples = get_nearby_test_examples(agent_state)
    prompt_template = load_prompt("mutation_test_writer_prompt.md")

    # The suite goes in AFTER comment stripping: the stripper cuts lines at "//", string
    # literals included, and the model must see the suite exactly as its edits will apply.
    return strip_comments_for_long_prompt(prompt_template.format(
        class_path=class_under_test.file_path,
        mutation_feedback=mutation_feedback,
        test_class=SUITE_SLOT,
        class_under_test=class_under_test_text,
        relevant_source_files=relevant_source_files,
        # nearby_test_examples=format_test_examples_for_prompt(nearby_test_examples),
    )).replace(SUITE_SLOT, test_class)
