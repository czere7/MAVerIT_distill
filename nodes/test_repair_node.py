from typing import TYPE_CHECKING

from edit_tools import REPAIR_TOOLS, SUITE_SLOT, apply_repair_calls, numbered, run_edit_turns
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


def test_repair_node(agent_state: "AgentState") -> dict[str, str]:
    failure = ("failing tests" if agent_state.get("compiler_success")
               else "compile failure")
    prompt = _build_prompt(agent_state)
    max_attempts = get_int_config("MAX_REPAIR_ATTEMPTS")
    attempts = agent_state.get("repair_attempts", 0)
    input_tokens = output_tokens = 0
    while True:
        attempts += 1
        print(f"[test_repair_node] Repairing test after {failure} (attempt {attempts}).")
        # The model edits lines of the numbered suite (replace) instead of regenerating it:
        # the typical failure is one bad line, and a full rewrite cost ~25k tokens per round.
        repaired_test_class, used_in, used_out, how = run_edit_turns(
            prompt, agent_state.get("test_class", "").strip(), REPAIR_TOOLS, apply_repair_calls,
            agent_state.get("run_id", ""), "test_repair_node")
        input_tokens += used_in
        output_tokens += used_out
        if repaired_test_class:
            break
        if how == "no_edit":
            # Nothing usable in any turn (an empty answer even without thinking, or prose).
            # Spend the repair budget instead of aborting: the unchanged suite fails again
            # and routing sends it to cleanup, which restores the last green suite if any.
            print("[test_repair_node] No edit in any turn. Giving up on repair so cleanup "
                  "can restore the last green suite.")
            attempts = max_attempts
        if attempts >= max_attempts:
            if how == "rejected":
                print("[test_repair_node] Every edit was rejected and the repair budget is spent.")
            repaired_test_class = agent_state.get("test_class", "")
            break
        # Every turn tried an edit and was rejected (almost always a syntax break). That costs
        # this attempt, not the budget: the next one starts from a fresh conversation, without
        # a compile in between, since the suite has not changed.
        print("[test_repair_node] Every edit was rejected; starting the next attempt fresh.")

    # write_log(f"[test_repair_node] Produced repaired test class with {len(repaired_test_class.splitlines())} line(s).", response.usage_metadata, agent_state)

    return {
        "test_class": repaired_test_class,
        "repair_attempts": attempts,
        "input_tokens": input_tokens + agent_state.get('input_tokens'),
        "output_tokens": output_tokens + agent_state.get('output_tokens'),
        "total_tokens": input_tokens + output_tokens + agent_state.get('total_tokens', 0),
    }


def _build_prompt(agent_state: "AgentState") -> str:
    test_class = agent_state.get("test_class", "").strip()
    compiler_feedback = agent_state.get("compiler_feedback", "").strip()

    if not test_class:
        raise RuntimeError("Cannot repair tests because AgentState['test_class'] is empty.")
    if not compiler_feedback:
        raise RuntimeError("Cannot repair tests because AgentState['compiler_feedback'] is empty.")

    class_under_test = get_current_class_under_test(agent_state)
    class_under_test_text, relevant_source_files = prompt_context(agent_state)
    nearby_test_examples = get_nearby_test_examples(agent_state)
    prompt_template = load_prompt("test_repair_prompt.md")

    # The numbered suite goes in AFTER comment stripping: the stripper cuts lines at "//",
    # string literals included, and the numbers must match the file javac reports on.
    return strip_comments_for_long_prompt(prompt_template.format(
        class_path=class_under_test.file_path,
        compiler_feedback=compiler_feedback,
        test_class=SUITE_SLOT,
        last_compilable_test_class=_format_last_compilable_test_class(agent_state),
        class_under_test=class_under_test_text,
        relevant_source_files=relevant_source_files,
        # nearby_test_examples=format_test_examples_for_prompt(nearby_test_examples),
    )).replace(SUITE_SLOT, numbered(test_class))


def _format_last_compilable_test_class(agent_state: "AgentState") -> str:
    last_compilable_test_class = agent_state.get("last_compilable_test_class", "").strip()
    if not last_compilable_test_class:
        return "No previous compilable version is available."

    return f"```java\n{last_compilable_test_class}\n```"
