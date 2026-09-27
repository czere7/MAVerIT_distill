from typing import Any, Mapping

from utils import get_int_config


def route_after_compile(agent_state: Mapping[str, Any]) -> str:
    """The graph's only real decision. Four outcomes, in cost order.

    Coverage is gone from the routing entirely: killing a mutant requires executing the
    line AND asserting on the difference, so mutation-driven writing does the coverage work
    aimed at lines that matter, while coverage-driven writing adds execution without
    assertion -- which is how a suite that runs the class and catches nothing gets built.
    """
    # 1. Does not compile. test_repair is aimed exactly at this.
    if not agent_state.get("compiler_success", False):
        if agent_state.get("repair_attempts", 0) >= get_int_config("MAX_REPAIR_ATTEMPTS"):
            return "faulty_test_cleanup_node"
        return "test_repair_node"

    # 2. Compiles, but a test fails. A DIFFERENT failure, separable only since
    #    -Dmaven.test.failure.ignore=true stopped the build aborting on a failing
    #    assertion. It still goes to test_repair, which is where this graph has always
    #    sent it -- but the feedback now names the failing methods, and the log records
    #    which of the two cases it was, so how often the wrong tool is used is measurable.
    if not agent_state.get("suite_green", False):
        if agent_state.get("repair_attempts", 0) >= get_int_config("MAX_REPAIR_ATTEMPTS"):
            return "faulty_test_cleanup_node"
        return "test_repair_node"

    # 3. Green. PIT has already run in the compile node, so the score is in state.
    if agent_state.get("mutation_current", 0.0) >= 100.0:
        return "class_advancer_node"

    iterations = agent_state.get("mutation_iterations", 0)
    if iterations >= get_int_config("MAX_MUTATION_ITERATIONS"):
        return "class_advancer_node"

    # The first pass always earns a writer turn; after that it has to keep paying for
    # itself, or the loop spends its budget on a score that stopped moving.
    if iterations == 0 or agent_state.get("mutation_improved_significantly", False):
        return "mutation_test_writer_node"

    return "class_advancer_node"


def route_after_class_advance(agent_state: Mapping[str, Any]) -> str:
    # "end" is the KEY main.py maps to END, not langgraph's sentinel. Returning
    # "__end__" here compiles fine and fails at runtime with an unmapped branch.
    if agent_state["current_class_index"] >= len(agent_state["all_test_files"]):
        return "end"
    return "initial_test_write_node"
