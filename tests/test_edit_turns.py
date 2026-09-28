"""run_edit_turns against a stub model: the conversation it sends back must answer every tool
call id, or DeepSeek rejects it with a 400 and the class crashes (cold start, Base64)."""

from langchain_core.messages import AIMessage, ToolMessage

import edit_tools

SUITE = "package p;\n\npublic class FooTest {\n}\n"
NEW_TEST = "    @org.junit.Test\n    public void t() { }\n"


class StubModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self):
        return self

    def invoke(self, messages, run_id, node_name, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def test_invalid_json_arguments_are_answered_and_the_next_turn_applies(monkeypatch):
    bad = AIMessage(content="", invalid_tool_calls=[{
        "name": "append_test", "args": '{"tests": "unterminated', "id": "call_bad",
        "error": "Expecting value", "type": "invalid_tool_call"}])
    good = AIMessage(content="", tool_calls=[{
        "name": "append_test", "args": {"imports": [], "tests": NEW_TEST}, "id": "call_ok"}])
    stub = StubModel([bad, good])
    monkeypatch.setattr(edit_tools, "ModelWrapper", stub)

    suite, _, _, how = edit_tools.run_edit_turns("prompt", SUITE, edit_tools.MUTATION_TOOLS,
                                                 edit_tools.apply_mutation_calls, "r", "node")
    assert how == "tools" and "public void t()" in suite
    second_request = stub.seen[1]
    answered = {m.tool_call_id for m in second_request if isinstance(m, ToolMessage)}
    assert answered == {"call_bad"}


def test_only_invalid_calls_in_every_turn_counts_as_rejected_not_no_edit(monkeypatch):
    bad = lambda i: AIMessage(content="", invalid_tool_calls=[{  # noqa: E731
        "name": "append_test", "args": "{", "id": f"call_{i}", "error": "x",
        "type": "invalid_tool_call"}])
    monkeypatch.setattr(edit_tools, "ModelWrapper", StubModel([bad(i) for i in range(edit_tools.MAX_TURNS)]))
    suite, _, _, how = edit_tools.run_edit_turns("prompt", SUITE, edit_tools.MUTATION_TOOLS,
                                                 edit_tools.apply_mutation_calls, "r", "node")
    assert suite is None and how == "rejected"
