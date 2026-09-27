"""Edit tools: the model changes the test suite by operations, not by regenerating it.

The DeepSeek logs showed why (smokes/mutation_writer_diff.py): per mutation-writer call, the
median response was ~10% new tests and ~90% the existing suite copied back; 99.2% of the tests
it was given came back unchanged. Repair was worse -- a whole 24k-char suite rewritten to fix
one `cannot find symbol`.

  mutation writer   append_test(imports, tests)       new members go before the class's last "}"
                    replace_test(test_name, new_test)  swap one method, annotations included
  repair            replace(range, content)            swap lines of the numbered suite

The tools are pure functions over the suite text; nothing here touches files or the model.
A TURN (all the calls in one reply) is atomic: either every call applies or none does and the
model gets the errors back. Line ranges in one reply all refer to the numbering it was shown,
and are applied bottom-up so no edit shifts another. A turn that introduces a syntax error
the suite did not have is rejected like any other failed call.
"""
import re
from typing import Callable

from langchain_core.messages import HumanMessage, ToolMessage

from models import ModelWrapper
from utils import extract_response_content, parser, strip_markdown_code_fence

MAX_TURNS = 3
# Stands in for the suite while a prompt goes through the comment stripper; the suite is
# substituted afterwards. Contains nothing the stripper touches.
SUITE_SLOT = "@@CURRENT_TEST_SUITE@@"


class ToolError(Exception):
    pass


# ============================================================== the tools
def append_test(suite: str, imports: list[str] | None, tests: str) -> str:
    """Imports after the last existing import; tests in place of the last '}'."""
    if not tests or not tests.strip():
        raise ToolError("append_test: `tests` is empty")
    end = suite.rfind("}")
    if end < 0:
        raise ToolError("append_test: the suite has no closing '}'")
    out = suite[:end].rstrip() + "\n\n" + tests.strip("\n") + "\n}" + suite[end + 1:]
    return add_imports(out, imports or [])


def add_imports(suite: str, imports: list[str]) -> str:
    existing = set(re.findall(r"^\s*import\s+(?:static\s+)?[\w.*]+\s*;", suite, re.M))
    existing = {re.sub(r"\s+", " ", i.strip()) for i in existing}
    new = []
    for imp in imports:
        body = imp.strip().removeprefix("import").strip().rstrip(";").strip()
        if not body:
            continue
        line = f"import {body};"
        if re.sub(r"\s+", " ", line) not in existing and line not in new:
            new.append(line)
    if not new:
        return suite
    block = "\n".join(new)
    last = None
    for last in re.finditer(r"^\s*import\s+[^;]+;[ \t]*$", suite, re.M):
        pass
    if last is not None:
        return suite[:last.end()] + "\n" + block + suite[last.end():]
    pkg = re.search(r"^\s*package\s+[\w.]+\s*;[ \t]*$", suite, re.M)
    if pkg is not None:
        return suite[:pkg.end()] + "\n\n" + block + suite[pkg.end():]
    return block + "\n\n" + suite


def replace_test(suite: str, test_name: str, new_test: str) -> str:
    """The one method named `test_name` (annotations included) becomes `new_test`."""
    if not new_test or not new_test.strip():
        raise ToolError("replace_test: `new_test` is empty")
    src = suite.encode("utf-8")
    root = parser.parse(src).root_node
    hits, names, stack = [], [], [root]
    while stack:
        n = stack.pop()
        if n.type == "method_declaration":
            name_node = n.child_by_field_name("name")
            name = src[name_node.start_byte:name_node.end_byte].decode() if name_node else ""
            names.append(name)
            if name == test_name:
                hits.append(n)
        stack.extend(n.children)
    if not hits:
        known = ", ".join(sorted(set(names))[:40])
        raise ToolError(f"replace_test: no method named {test_name!r}. Methods in the suite: {known}")
    if len(hits) > 1:
        raise ToolError(f"replace_test: {test_name!r} is overloaded ({len(hits)} methods); "
                        "use append_test for a new test instead")
    node = hits[0]
    return (src[:node.start_byte] + new_test.strip().encode("utf-8") + src[node.end_byte:]).decode("utf-8")


_NUMBERED = re.compile(r"^\s*\d+\s*\|\s?")


def numbered(suite: str) -> str:
    """The suite as the repair agent sees it: '  12 | code'."""
    lines = suite.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, start=1))


def parse_range(spec: str, n_lines: int) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*(?:[-:]\s*(\d+)\s*)?", str(spec))
    if not m:
        raise ToolError(f"replace: range {spec!r} is not 'N' or 'START-END'")
    start, end = int(m.group(1)), int(m.group(2) or m.group(1))
    if start < 1 or end < start or end > n_lines:
        raise ToolError(f"replace: range {spec!r} is outside lines 1-{n_lines} or reversed")
    return start, end


def replace_lines(suite: str, edits: list[tuple[str, str]]) -> str:
    """All edits refer to the ORIGINAL numbering; applied bottom-up; overlaps rejected."""
    lines = suite.split("\n")
    parsed = []
    for spec, content in edits:
        start, end = parse_range(spec, len(lines))
        body = content.split("\n") if content else []
        # A model that copies the numbered view back includes the prefixes: drop them.
        if body and all(_NUMBERED.match(b) for b in body if b.strip()):
            body = [_NUMBERED.sub("", b, count=1) for b in body]
        parsed.append((start, end, body))
    parsed.sort()
    for (s1, e1, _), (s2, e2, _) in zip(parsed, parsed[1:]):
        if s2 <= e1:
            raise ToolError(f"replace: ranges {s1}-{e1} and {s2}-{e2} overlap; "
                            "all ranges in one reply refer to the numbering shown")
    for start, end, body in reversed(parsed):
        lines[start - 1:end] = body
    return "\n".join(lines)


# ============================================================== schemas
def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


MUTATION_TOOLS = [
    _fn("append_test",
        "Add new test methods to the end of the test class. `tests` holds one or more complete "
        "members (test methods with their annotations; new helper methods or fields too). "
        "`imports` lists what they need that the class does not import yet.",
        {"imports": {"type": "array", "items": {"type": "string"},
                     "description": "e.g. \"java.util.List\" or \"static org.junit.Assert.assertEquals\""},
         "tests": {"type": "string", "description": "Java source of the new members"}},
        ["tests"]),
    _fn("replace_test",
        "Replace one existing test method, found by name, with a new version -- e.g. to "
        "strengthen its assertions. Give the complete method, annotations included.",
        {"test_name": {"type": "string"}, "new_test": {"type": "string"}},
        ["test_name", "new_test"]),
]

REPAIR_TOOLS = [
    _fn("replace",
        "Replace lines of the current test class. `range` is 'N' or 'START-END' (inclusive) "
        "in the numbering shown. `content` replaces those lines: several lines are fine, an "
        "empty string deletes them, and to insert keep the original line in `content`. Do not "
        "include the line-number prefixes. All ranges in one reply refer to the numbering shown "
        "and must not overlap.",
        {"range": {"type": "string"}, "content": {"type": "string"}},
        ["range", "content"]),
]


# ============================================================== applying a turn
def _has_syntax_error(code: str) -> bool:
    return parser.parse(code.encode("utf-8")).root_node.has_error


def _first_syntax_error(code: str) -> int | None:
    """1-based line of the first ERROR or MISSING node, or None if the code parses."""
    root = parser.parse(code.encode("utf-8")).root_node
    if not root.has_error:
        return None
    first, stack = None, [root]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or n.is_missing:
            line = n.start_point[0] + 1
            first = line if first is None else min(first, line)
            continue
        stack.extend(c for c in n.children if c.has_error or c.is_missing)
    return first


_MAX_SHOWN = 40


def _syntax_feedback(suite: str, new_suite: str, calls: list[dict]) -> str:
    """Why a turn broke the class, concretely enough to fix: 212 of 213 gpt-oss rejections were
    a real syntax break (typically a range that took a method's closing '}' whose content left
    it out), and the bare 'no longer parses' rarely got them fixed on the next turn."""
    parts = ["the edited class no longer parses as Java; nothing was applied."]
    line = _first_syntax_error(new_suite)
    if line is not None:
        edited = new_suite.split("\n")
        lo, hi = max(1, line - 2), min(len(edited), line + 2)
        view = "\n".join(f"{i} | {edited[i - 1]}" for i in range(lo, hi + 1))
        parts.append(f"After your edit the parser first fails near line {line} of the EDITED "
                     f"class (a missing brace may only be noticed later):\n{view}")
    lines = suite.split("\n")
    for call in calls:
        if call.get("name") != "replace":
            continue
        args = call.get("args") or {}
        try:
            start, end = parse_range(args.get("range", ""), len(lines))
        except ToolError:
            continue
        original = lines[start - 1:end]
        last = min(end, start + _MAX_SHOWN - 1)
        shown = "\n".join(f"{i} | {lines[i - 1]}" for i in range(start, last + 1))
        if end > last:
            shown += f"\n... ({end - last} more lines)"
        note = f"Your replace of {start}-{end} removed these lines:\n{shown}"
        content = args.get("content", "") or ""
        was = ("\n".join(original).count("{"), "\n".join(original).count("}"))
        now = (content.count("{"), content.count("}"))
        if was != now:
            note += (f"\nThey contain {was[0]} '{{' and {was[1]} '}}'; your content has "
                     f"{now[0]} '{{' and {now[1]} '}}'.")
        parts.append(note)
    parts.append("Resend the edit; ranges still refer to the numbering shown in the prompt.")
    return "\n\n".join(parts)


def apply_mutation_calls(suite: str, calls: list[dict]) -> str:
    out = suite
    for call in calls:
        args = call.get("args") or {}
        if call["name"] == "append_test":
            imports = args.get("imports") or []
            if isinstance(imports, str):
                imports = [i for i in re.split(r"[\n,]", imports) if i.strip()]
            out = append_test(out, imports, args.get("tests", ""))
        elif call["name"] == "replace_test":
            out = replace_test(out, args.get("test_name", ""), args.get("new_test", ""))
        else:
            raise ToolError(f"unknown tool {call['name']!r}; use append_test or replace_test")
    return out


def apply_repair_calls(suite: str, calls: list[dict]) -> str:
    edits = []
    for call in calls:
        if call["name"] != "replace":
            raise ToolError(f"unknown tool {call['name']!r}; use replace")
        args = call.get("args") or {}
        edits.append((args.get("range", ""), args.get("content", "")))
    return replace_lines(suite, edits)


def run_edit_turns(prompt: str, suite: str, tools: list[dict],
                   apply: Callable[[str, list[dict]], str], run_id: str, node_name: str):
    """Up to MAX_TURNS replies; returns (new_suite or None, input_tokens, output_tokens, how).
    `how` is "tools", "rewrite" (no tool calls, but a whole class came back), "rejected" (it
    called tools, and every turn was rejected) or "no_edit" (no tool call and no class in any
    turn). Callers treat the last two differently: a model that keeps trying earns another
    round, one that returns nothing does not."""
    messages = [HumanMessage(prompt)]
    tokens_in = tokens_out = 0
    attempted = False
    for turn in range(MAX_TURNS):
        response = ModelWrapper().invoke(messages, run_id, node_name, tools=tools)
        usage = response.usage_metadata or {}
        tokens_in += usage.get("input_tokens", 0)
        tokens_out += usage.get("output_tokens", 0)
        calls = response.tool_calls or []
        if calls:
            attempted = True
            try:
                new_suite = apply(suite, calls)
                if _has_syntax_error(new_suite) and not _has_syntax_error(suite):
                    raise ToolError(_syntax_feedback(suite, new_suite, calls))
                print(f"[{node_name}] applied {len(calls)} tool call(s): "
                      f"{', '.join(c['name'] for c in calls)}")
                return new_suite, tokens_in, tokens_out, "tools"
            except ToolError as error:
                print(f"[{node_name}] tool call rejected (turn {turn + 1}): {error}")
                messages.append(response)
                for call in calls:
                    messages.append(ToolMessage(f"NOT APPLIED: {error}", tool_call_id=call.get("id") or call["name"]))
                continue
        text = strip_markdown_code_fence(extract_response_content(response))
        if re.search(r"\bclass\s+\w+", text) and "@Test" in text:
            print(f"[{node_name}] no tool calls; accepting the returned class as a rewrite")
            return text, tokens_in, tokens_out, "rewrite"
        messages.append(response)
        messages.append(HumanMessage("Make your change by calling the tools; do not answer in prose."))
    return None, tokens_in, tokens_out, ("rejected" if attempted else "no_edit")
