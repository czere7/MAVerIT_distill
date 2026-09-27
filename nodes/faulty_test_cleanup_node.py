from bisect import bisect_right
import re
from pathlib import Path
from typing import TYPE_CHECKING

from nodes.compiler_node import _score_mutation, write_compiler_log
from utils import ensure_file, get_working_directory, node_text, parser, run_maven, walk, \
    test_fqn_or_none, get_java_fully_qualified_name, get_maven_module_directory, \
    get_current_class_under_test, find_surefire_reports, parse_surefire

if TYPE_CHECKING:
    from AgentState import AgentState

MAVEN_JAVA_ERROR_RE = re.compile(r"^\[ERROR]\s+(.+?\.java):\[(\d+),(\d+)]", re.MULTILINE)
TEST_ANNOTATION_RE = re.compile(r"@\s*(?:[\w.]+\.)?(?:Test|ParameterizedTest|RepeatedTest|TestFactory|TestTemplate)\b")


def faulty_test_cleanup_node(agent_state: "AgentState") -> dict:
    last_compilable_test_class = agent_state.get("last_compilable_test_class", "").strip()
    last_compilable_test_file_path = agent_state.get("last_compilable_test_file_path", "")
    if last_compilable_test_class and last_compilable_test_file_path:
        print(
            "[faulty_test_cleanup_node] Repair attempts exhausted. "
            f"Restoring last green test file: {last_compilable_test_file_path}"
        )
        last_compilable_path = ensure_file(last_compilable_test_file_path)
        last_compilable_path.write_text(last_compilable_test_class + "\n", encoding="utf-8")
        return {
            "test_class": "",
            "compiler_success": False,
            "compiler_feedback": "",
            "test_file_path": "",
            "repair_attempts": 0,
        }

    test_file_path = agent_state.get("test_file_path", "")
    trimmed_test_class = _trim_failing_test_methods(agent_state, test_file_path)
    if trimmed_test_class is not None:
        maven_result = run_maven(str(get_working_directory()),
                                 test_fqn_or_none(trimmed_test_class))
        # GREEN, not merely compiling: with failure.ignore on, "ok" only means it built,
        # and a trimmed suite whose remaining tests fail is no better than the one before.
        tests_run, tests_failed = _surefire_counts(agent_state, trimmed_test_class)
        if maven_result["ok"] and tests_run > 0 and tests_failed == 0:
            print(f"[faulty_test_cleanup_node] Trimmed test class is green: {tests_run} tests pass.")
            _score_and_log(agent_state, trimmed_test_class, test_file_path, tests_run, maven_result)
            return {
                "test_class": trimmed_test_class,
                "compiler_success": False,
                "compiler_feedback": "",
                "test_file_path": test_file_path,
                "repair_attempts": 0,
            }

        test_path = Path(test_file_path)
        if test_path.is_file():
            test_path.unlink()
        state = ("does not compile" if not maven_result["ok"]
                 else f"is not green ({tests_failed} of {tests_run} tests fail)")
        print(
            f"[faulty_test_cleanup_node] Trimmed test file {state}. "
            f"Deleted faulty test file: {test_file_path}"
        )
        return {
            "test_class": "",
            "compiler_success": False,
            "compiler_feedback": "",
            "test_file_path": "",
            "repair_attempts": 0,
        }

    if test_file_path:
        test_path = Path(test_file_path)
        if test_path.is_file():
            test_path.unlink()
            print(
                "[faulty_test_cleanup_node] Repair attempts exhausted and no green snapshot exists. "
                f"Deleted faulty test file: {test_file_path}"
            )
        else:
            print(
                "[faulty_test_cleanup_node] Repair attempts exhausted and no green snapshot exists. "
                f"No test file found to delete at: {test_file_path}"
            )
    else:
        print("[faulty_test_cleanup_node] Repair attempts exhausted and no test file path was available.")

    return {
        "test_class": "",
        "compiler_success": False,
        "compiler_feedback": "",
        "test_file_path": "",
        "repair_attempts": 0,
    }


def _trim_failing_test_methods(agent_state: "AgentState", test_file_path: str) -> str | None:
    if not test_file_path:
        print("[faulty_test_cleanup_node] No test file path was available for selective cleanup.")
        return None

    test_path = Path(test_file_path)
    if not test_path.is_file():
        print(f"[faulty_test_cleanup_node] No test file found for selective cleanup at: {test_file_path}")
        return None

    test_class = test_path.read_text(encoding="utf-8")
    failing_tests = agent_state.get("failing_tests") or []
    if agent_state.get("compiler_success") and failing_tests:
        # The suite COMPILES and some tests fail at runtime. There are no javac line
        # numbers to go on, but surefire already named the failing methods.
        removable_methods = _find_test_methods_by_name(test_class, set(failing_tests))
        if removable_methods is None:
            print(
                "[faulty_test_cleanup_node] A failing test could not be matched to a method "
                f"in the file ({', '.join(sorted(failing_tests))}). "
                "Falling back to deleting the faulty test file."
            )
            return None
        trimmed_test_class = _remove_source_ranges(test_class, removable_methods).rstrip() + "\n"
        test_path.write_text(trimmed_test_class, encoding="utf-8")
        print(
            "[faulty_test_cleanup_node] Repair attempts exhausted and no green snapshot exists. "
            f"Removed {len(removable_methods)} failing test method(s) "
            f"({', '.join(sorted(failing_tests))}) from: {test_file_path}"
        )
        return trimmed_test_class.rstrip("\n")

    failing_lines = _extract_failing_lines(agent_state.get("compiler_feedback", ""), test_path)
    if failing_lines is None:
        print(
            "[faulty_test_cleanup_node] Maven compiler errors included files other than the current test file. "
            "Falling back to deleting the faulty test file."
        )
        return None
    if not failing_lines:
        print("[faulty_test_cleanup_node] Could not find Maven compiler error line numbers for selective cleanup.")
        return None

    removable_methods = _find_removable_test_methods(test_class, failing_lines)
    if removable_methods is None:
        print(
            "[faulty_test_cleanup_node] At least one compiler error was outside a parsed test method. "
            "Falling back to deleting the faulty test file."
        )
        return None
    if not removable_methods:
        print("[faulty_test_cleanup_node] Compiler errors did not map to any removable test methods.")
        return None

    trimmed_test_class = _remove_source_ranges(test_class, removable_methods).rstrip() + "\n"
    test_path.write_text(trimmed_test_class, encoding="utf-8")
    print(
        "[faulty_test_cleanup_node] Repair attempts exhausted and no green snapshot exists. "
        f"Removed {len(removable_methods)} failing test method(s) from: {test_file_path}"
    )
    return trimmed_test_class.rstrip("\n")


def _score_and_log(agent_state: "AgentState", test_class: str, test_file_path: str,
                   tests_run: int, maven_result: dict) -> None:
    """PIT and coverage for a suite trimmed to green, logged like any compile.

    Cleanup routes straight to the class advancer, so without this the suite a class
    ends with would be the one suite in log.jsonl with no scores. The score is absolute:
    previous is pinned to 0, since a delta against the red suite means nothing.
    """
    state_update = {
        "compiler_success": True,
        "suite_green": True,
        "tests_run": tests_run,
        "tests_passed": tests_run,
        "tests_failed": 0,
        "failing_tests": [],
        "jacoco_ok": maven_result.get("jacoco_ok", True),
        "compiler_feedback": "",
        "test_file_path": test_file_path,
    }
    scored_state = {**agent_state, "test_class": test_class, "mutation_current": 0.0}
    state_update.update(_score_mutation(scored_state, test_class))
    state_update["mutation_delta"] = None
    write_compiler_log(state_update, scored_state, origin="faulty_test_cleanup_node")


def _surefire_counts(agent_state: "AgentState", test_class: str) -> tuple[int, int]:
    """(run, failed) for the generated class only, as the compiler node counts them."""
    fqn = get_java_fully_qualified_name(test_class)
    module_dir = get_maven_module_directory(
        get_current_class_under_test(agent_state).file_path, get_working_directory())
    tests_run, _, tests_failed = parse_surefire(find_surefire_reports(module_dir, fqn))
    return tests_run, tests_failed


def _find_test_methods_by_name(test_class: str, names: set[str]) -> list[tuple[int, int]] | None:
    """Source ranges of the named test methods, or None if any name has no method.

    None rather than a partial list: a name surefire reports but the file does not declare
    (an inherited test, a class-level initializationError) means the failure lives
    somewhere trimming cannot reach, and removing only the rest cannot make it green.
    """
    source = test_class.encode("utf-8")
    tree = parser.parse(source)
    line_start_bytes = _line_start_bytes(test_class)
    ranges, found = [], set()
    for node in walk(tree.root_node):
        if node.type != "method_declaration":
            continue
        name_node = node.child_by_field_name("name")
        method_name = node_text(source, name_node) if name_node is not None else ""
        if method_name in names and _is_test_method(node_text(source, node), method_name):
            found.add(method_name)
            ranges.append(_expand_method_range_to_leading_test_context(test_class, node, line_start_bytes))
    if found != names:
        return None
    return sorted(set(ranges), reverse=True)


def _extract_failing_lines(compiler_feedback: str, test_path: Path) -> set[int] | None:
    normalized_test_path = _normalize_path_for_maven(test_path)
    failing_lines = set()
    saw_java_error = False

    for match in MAVEN_JAVA_ERROR_RE.finditer(compiler_feedback):
        saw_java_error = True
        error_path = _normalize_path_for_maven(match.group(1))
        if (
            error_path != normalized_test_path
            and not error_path.endswith(f"/{normalized_test_path}")
            and not normalized_test_path.endswith(f"/{error_path}")
        ):
            return None
        failing_lines.add(int(match.group(2)))

    if not saw_java_error:
        return set()
    return failing_lines


def _normalize_path_for_maven(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    if re.match(r"^/[A-Za-z]:/", normalized):
        normalized = normalized[1:]
    return normalized.lower()


def _find_removable_test_methods(test_class: str, failing_lines: set[int]) -> list[tuple[int, int]] | None:
    source = test_class.encode("utf-8")
    tree = parser.parse(source)
    test_methods = []
    helper_declarations = []
    line_start_bytes = _line_start_bytes(test_class)

    for node in walk(tree.root_node):
        if node.type == "method_declaration":
            method_source = node_text(source, node)
            name_node = node.child_by_field_name("name")
            method_name = node_text(source, name_node) if name_node is not None else ""
            if _is_test_method(method_source, method_name):
                test_methods.append(node)
            else:
                helper_declarations.append(node)
        elif _is_nested_type_declaration(node):
            helper_declarations.append(node)

    removable = []
    for failing_line in failing_lines:
        method = _find_method_containing_line(test_methods, failing_line, line_start_bytes)
        declaration = method or _find_smallest_declaration_containing_line(
            helper_declarations,
            failing_line,
            line_start_bytes,
        )
        if declaration is None:
            return None
        removable.append(_expand_method_range_to_leading_test_context(test_class, declaration, line_start_bytes))

    return sorted(set(removable), reverse=True)


def _is_test_method(method_source: str, method_name: str) -> bool:
    return bool(TEST_ANNOTATION_RE.search(method_source)) or method_name.startswith("test")


def _is_nested_type_declaration(node) -> bool:
    if node.type not in {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}:
        return False
    return node.parent is not None and node.parent.type == "class_body"


def _find_method_containing_line(methods, line: int, line_start_bytes: list[int]):
    if line <= 0 or line > len(line_start_bytes):
        return None

    line_start_byte = line_start_bytes[line - 1]
    for method in methods:
        if method.start_byte <= line_start_byte <= method.end_byte:
            return method
    return None


def _find_smallest_declaration_containing_line(declarations, line: int, line_start_bytes: list[int]):
    containing = [
        declaration
        for declaration in declarations
        if _find_method_containing_line([declaration], line, line_start_bytes) is not None
    ]
    if not containing:
        return None
    return min(containing, key=lambda declaration: declaration.end_byte - declaration.start_byte)


def _expand_method_range_to_leading_test_context(
    test_class: str,
    method,
    line_start_bytes: list[int],
) -> tuple[int, int]:
    lines = test_class.splitlines(keepends=True)
    start_line = _line_index_for_byte(line_start_bytes, method.start_byte)
    end_line = _line_index_for_byte(line_start_bytes, max(method.end_byte - 1, method.start_byte))

    while start_line > 0 and not lines[start_line - 1].strip():
        start_line -= 1

    while start_line > 0:
        previous = lines[start_line - 1].strip()
        if (
            previous.startswith("@")
            or previous.startswith("*")
            or previous.startswith("/*")
            or previous.startswith("//")
        ):
            start_line -= 1
            continue
        break

    start_byte = line_start_bytes[start_line]
    if end_line + 1 < len(line_start_bytes):
        end_byte = line_start_bytes[end_line + 1]
    else:
        end_byte = len(test_class.encode("utf-8"))
    return start_byte, end_byte


def _remove_source_ranges(source: str, ranges: list[tuple[int, int]]) -> str:
    source_bytes = source.encode("utf-8")
    for start_byte, end_byte in ranges:
        source_bytes = source_bytes[:start_byte] + source_bytes[end_byte:]
    return source_bytes.decode("utf-8")


def _line_start_bytes(source: str) -> list[int]:
    starts = []
    current = 0
    for line in source.splitlines(keepends=True):
        starts.append(current)
        current += len(line.encode("utf-8"))
    if not starts:
        starts.append(0)
    return starts


def _line_index_for_byte(line_start_bytes: list[int], byte_offset: int) -> int:
    return max(0, bisect_right(line_start_bytes, byte_offset) - 1)
