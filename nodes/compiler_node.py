import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from utils import ensure_file, get_current_class_under_test, get_java_entity_name, get_working_directory, run_maven, \
    count_test_without_assert, write_to_log_file, get_maven_module_directory, run_pitest, \
    get_java_fully_qualified_name, calculate_mutation_score, find_pitest_xml_reports, \
    find_surefire_reports, parse_surefire, parse_surefire_failures, calculate_line_coverage_for_class, \
    get_float_config, summarize_pitest_mutations, summarize_uncovered_lines, test_fqn_or_none, \
    calculate_branch_coverage_for_class, find_jacoco_xml_reports

if TYPE_CHECKING:
    from AgentState import AgentState


def compiler_node(agent_state: "AgentState") -> dict:
    test_class = agent_state.get("test_class", "").strip()
    if not test_class:
        raise RuntimeError("Cannot compile tests because AgentState['test_class'] is empty.")

    project_dir = get_working_directory()
    class_under_test = get_current_class_under_test(agent_state)
    source_file_path = Path(class_under_test.file_path)
    print(f"[compiler_node] Writing and compiling generated test for: {source_file_path}")
    try:
        test_file_path = _write_test_class(project_dir, source_file_path, test_class)

        print(f"[compiler_node] Test file written to: {test_file_path}")
        maven_result = run_maven(str(project_dir), test_fqn_or_none(test_class))
    except RuntimeError as e:
        if "does not contain a class, interface, enum, or record declaration" in str(e):
            maven_result = {
                "ok": False,
                "combined_result": f"Generated test code is invalid: {e}"
            }
            test_file_path = []
        else:
            raise

    compiler_feedback = _format_compiler_feedback(
        ok=maven_result["ok"],
        test_file_path=test_file_path,
        combined_result=maven_result["combined_result"],
    )

    # The build no longer aborts on a failing assertion, so the return code alone cannot
    # say whether the suite is green. Surefire can, and it can also name the failures.
    tests_run = tests_passed = tests_failed = 0
    failing_tests: set[str] = set()
    if maven_result["ok"]:
        module_dir = get_maven_module_directory(
            class_under_test.file_path, get_working_directory())
        # SCOPED TO THE GENERATED CLASS. The project's own tests run too, and counting
        # them here would make their failures ours.
        surefire = find_surefire_reports(
            module_dir, get_java_fully_qualified_name(test_class))
        tests_run, tests_passed, tests_failed = parse_surefire(surefire)
        if tests_failed:
            failing_tests = parse_surefire_failures(surefire)
            # Re-format now that "green" is known -- this is what triggers compaction.
            compiler_feedback = _format_compiler_feedback(
                ok=True, test_file_path=test_file_path,
                combined_result=maven_result["combined_result"], green=False)
            compiler_feedback = (
                f"{compiler_feedback}\n\n"
                f"The class COMPILES. {tests_failed} of {tests_run} tests fail:\n  "
                + "\n  ".join(sorted(failing_tests))
                + "\n\nThese are assertion or runtime failures, not compilation errors."
            )

    # COMPILED, BUT NOTHING RAN. Either the class has no runnable @Test methods, or the
    # -Dtest pattern matched nothing -- which failIfNoSpecifiedTests=false lets through
    # silently. Left unsaid, the feedback reads "compilation passed" and the suite goes to
    # repair with nothing visibly wrong.
    if maven_result["ok"] and tests_run == 0:
        compiler_feedback = (
            f"{compiler_feedback}\n\n"
            "The class COMPILES, but NO TESTS RAN. The suite needs at least one runnable "
            "public @Test method (JUnit 4: org.junit.Test, in a public class)."
        )

    suite_green = bool(maven_result["ok"]) and tests_run > 0 and tests_failed == 0

    state_update = {
        "compiler_success": maven_result["ok"],
        "suite_green": suite_green,
        "tests_run": tests_run,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "failing_tests": sorted(failing_tests),
        "jacoco_ok": maven_result.get("jacoco_ok", True),
        "compiler_feedback": compiler_feedback,
        "test_file_path": str(test_file_path),
        "messages": [
            {
                "role": "user",
                "content": compiler_feedback,
            }
        ],
    }
    if maven_result["ok"]:
        # The restore point is the last GREEN suite, despite the key's name. A red suite
        # compiles too, and storing it let cleanup hand back a suite with failing tests
        # in place of an earlier green one (URLCodec: 103 tests, 10 failing, over 98.2%).
        if suite_green:
            state_update["last_compilable_test_class"] = test_class
            state_update["last_compilable_test_file_path"] = str(test_file_path)
            print("[compiler_node] Compiled and green. Stored as the restore point.")
        else:
            print(f"[compiler_node] COMPILES but {tests_failed}/{tests_run} tests fail: "
                  f"{', '.join(sorted(failing_tests)) or 'no tests ran'}")
    else:
        print("[compiler_node] Does not compile. Routing will attempt repair if attempts remain.")
        print(compiler_feedback)
    if not state_update["jacoco_ok"]:
        print("[compiler_node] WARNING: the jacoco goal failed -- coverage is unavailable "
              "for this project. Check that its pom configures jacoco-maven-plugin.")

    # THE MUTATION VALIDATOR, MERGED. It used to be its own node, which meant the PIT
    # result reached the router only by way of a separate hop. The router now needs it
    # here: "mutation satisfied" is a branch out of this node.
    if suite_green:
        state_update.update(_score_mutation(agent_state, test_class))

    write_compiler_log(state_update, agent_state)

    return state_update


def _score_mutation(agent_state: "AgentState", test_class: str) -> dict:
    """PIT on a green suite, returning the delta state the router reads.

    Ported verbatim from mutation_validator_node, minus the node wrapper. A PIT failure
    is NOT a mutation score of zero -- it carries the previous score forward, so a
    transient PIT problem cannot read as the suite having got worse.
    """
    project_dir = get_working_directory()
    threshold = get_float_config("MUTATION_IMPROVEMENT_THRESHOLD")
    previous_score = float(agent_state.get("mutation_current", 0.0))
    class_under_test = get_current_class_under_test(agent_state)
    module_dir = get_maven_module_directory(class_under_test.file_path, project_dir)
    target_classes = get_java_fully_qualified_name(class_under_test.file_content)
    target_tests = get_java_fully_qualified_name(test_class)

    print(f"[compiler_node] Running PIT for {target_classes} with tests {target_tests}")
    pitest_result = run_pitest(str(module_dir), target_classes, target_tests)

    if not pitest_result["ok"]:
        print("[compiler_node] PIT run failed; carrying the previous score forward.")
        return {
            "mutation_previous": previous_score,
            "mutation_current": previous_score,
            "mutation_delta": 0.0,
            "mutation_improved_significantly": False,
            "mutation_feedback": _format_mutation_feedback(
                ok=False, previous_score=previous_score, current_score=previous_score,
                mutation_delta=0.0, threshold=threshold, metric_summary="",
                command_output=pitest_result["combined_result"]),
        }

    report_paths = find_pitest_xml_reports(module_dir)
    if not report_paths:
        raise RuntimeError(
            f"No PIT mutations.xml reports found under Maven module directory: {module_dir}")

    current_score = calculate_mutation_score(report_paths)
    metric_summary = summarize_pitest_mutations(report_paths, target_classes)

    # WHAT PIT CANNOT SEE. Its NO_COVERAGE status only appears where a mutant exists, so a
    # line it cannot mutate is invisible to the writer no matter how untested it is.
    # JaCoCo counts every line, and we have already run it.
    uncovered = summarize_uncovered_lines(
        find_jacoco_xml_reports(module_dir), target_classes)
    if uncovered:
        metric_summary = f"{metric_summary}\n\n{uncovered}"
    mutation_delta = round(current_score - previous_score, 2)

    print(f"[compiler_node] Mutation score: {current_score:.2f}% "
          f"(previous {previous_score:.2f}%, delta {mutation_delta:.2f}%).")

    return {
        "mutation_previous": previous_score,
        "mutation_current": current_score,
        "mutation_delta": mutation_delta,
        "mutation_improved_significantly": mutation_delta >= threshold,
        "mutation_feedback": _format_mutation_feedback(
            ok=True, previous_score=previous_score, current_score=current_score,
            mutation_delta=mutation_delta, threshold=threshold,
            metric_summary=metric_summary,
            command_output=pitest_result["combined_result"]),
    }


def _format_mutation_feedback(
    ok: bool,
    previous_score: float,
    current_score: float,
    mutation_delta: float,
    threshold: float,
    metric_summary: str,
    command_output: str,
) -> str:
    if not ok:
        return (
            "Mutation validator result: PIT mutation report generation failed.\n"
            f"Mutation score previous: {previous_score:.2f}%\n"
            f"Mutation score current: {current_score:.2f}%\n"
            f"Mutation score delta: {mutation_delta:.2f}%\n"
            f"Significant improvement threshold: {threshold:.2f}%\n\n"
            "Maven output:\n"
            f"{command_output.strip()}"
        ).strip()

    return (
        "Mutation validator result: PIT mutation report generation succeeded.\n"
        f"Mutation score previous: {previous_score:.2f}%\n"
        f"Mutation score current: {current_score:.2f}%\n"
        f"Mutation score delta: {mutation_delta:.2f}%\n"
        f"Significant improvement threshold: {threshold:.2f}%\n\n"
        "Mutation details for the class under test:\n"
        f"{metric_summary.strip()}"
    ).strip()

def write_compiler_log(state_update: dict, agent_state: TypedDict, origin: str = "compiler_node"):
    content = {
        "time_stamp": round(time.time() * 1000),
        # Which node scored this suite: cleanup also logs the suites it trims to green.
        "origin": origin,
        "current_assert_less_test_amount": count_test_without_assert(agent_state.get("test_class")),
        "input_tokens": agent_state.get("input_tokens"),
        "output_tokens": agent_state.get("output_tokens"),
        "total_tokens": agent_state.get("total_tokens"),
        "current_class_index": agent_state.get("current_class_index"),
        "test_class": agent_state.get("test_class"),
        "compiler_success": state_update.get("compiler_success"),
        "compiler_feedback": state_update.get("compiler_feedback"),
        "test_file_path": state_update.get("test_file_path"),
        "last_compilable_test_class": state_update.get("last_compilable_test_class", agent_state.get("last_compilable_test_class")),
        "last_compilable_test_file_path": state_update.get("last_compilable_test_file_path", agent_state.get("last_compilable_test_file_path")),
        "active_validation_phase": agent_state.get("active_validation_phase"),
        "repair_attempts": agent_state.get("repair_attempts"),
        "coverage_iterations": agent_state.get("coverage_iterations"),
        "mutation_iterations": agent_state.get("mutation_iterations"),
        "mutation": "None",
        "coverage": "None",
        "line_coverage": "None",
        "suite_green": state_update.get("suite_green"),
        "tests_run": state_update.get("tests_run"),
        "tests_passed": state_update.get("tests_passed"),
        "tests_failed": state_update.get("tests_failed"),
        "failing_tests": state_update.get("failing_tests"),
    }
    if not state_update.get('compiler_success'):
        write_to_log_file(json.dumps(content), agent_state.get("run_id"))
        return
    # common
    class_under_test = get_current_class_under_test(agent_state)
    module_dir = get_maven_module_directory(class_under_test.file_path, get_working_directory())

    # COVERAGE IS MEASURED WHENEVER IT COMPILED, red or green. That is what
    # -Dmaven.test.failure.ignore=true bought: JaCoCo instruments whatever actually ran,
    # so a failing assertion no longer costs us the coverage data.
    # run_jacoco(str(module_dir)) # run_maven already calls jacoco:report
    target_class = get_java_fully_qualified_name(class_under_test.file_content)
    if state_update.get("jacoco_ok", True):
        jacoco_reports = find_jacoco_xml_reports(module_dir)
        content['coverage'] = calculate_branch_coverage_for_class(jacoco_reports, target_class)
        content['line_coverage'] = calculate_line_coverage_for_class(jacoco_reports, target_class)

    # PIT already ran in the node, gated on a green suite -- it cannot tell an introduced
    # fault from a pre-existing failure. The logger reads that result rather than paying
    # for a second run.
    if state_update.get("mutation_current") is not None:
        content['mutation'] = state_update.get("mutation_current")
        content['mutation_delta'] = state_update.get("mutation_delta")
    content['overall_assert_less_test_amount'] = (
            int(agent_state.get("assert_less_test_amount", 0)) +
            count_test_without_assert(agent_state.get("test_class")))
    write_to_log_file(json.dumps(content), agent_state.get("run_id"))


def _write_test_class(project_dir: Path, source_file_path: Path, test_class: str) -> Path:
    relative_source_path = _relative_source_path(project_dir, source_file_path)
    relative_test_path = _source_path_to_test_path(relative_source_path, test_class)
    test_file_path = project_dir / relative_test_path

    ensure_file(test_file_path)
    test_file_path.write_text(test_class + "\n", encoding="utf-8")
    return test_file_path


def _relative_source_path(project_dir: Path, source_file_path: Path) -> Path:
    project_dir = project_dir.resolve()
    source_file_path = source_file_path.resolve()

    try:
        return source_file_path.relative_to(project_dir)
    except ValueError as error:
        raise RuntimeError(f"Source file is not inside WORKING_DIRECTORY: {source_file_path}") from error


def _source_path_to_test_path(relative_source_path: Path, test_class: str) -> Path:
    parts = list(relative_source_path.parts)
    try:
        src_index = parts.index("src")
        main_index = src_index + 1
    except ValueError as error:
        raise RuntimeError(f"Source file path does not contain a src/main segment: {relative_source_path}") from error

    if main_index >= len(parts) or parts[main_index] != "main":
        raise RuntimeError(f"Source file path does not contain a src/main segment: {relative_source_path}")

    parts[main_index] = "test"
    test_class_name = get_java_entity_name(test_class)
    parts[-1] = f"{test_class_name}.java"
    return Path(*parts)


def _format_compiler_feedback(ok: bool, test_file_path: Path, combined_result: str,
                              green: bool = True) -> str:
    """Compaction keys on GREEN, not on `ok`.

    `ok` now means "compiled", so a suite with failing tests takes the ok=True path. Left
    uncompacted that shipped the whole build log -- 13,792 characters of "Scanning for
    projects...", the enforcer plugin and the reactor summary -- on every repair attempt,
    and claimed the compilation "passed" while three tests failed.
    """
    if not ok:
        status = "failed"
    elif green:
        status = "passed"
    else:
        status = "succeeded, but tests fail"
    output = combined_result.strip() if (ok and green) else _compact_maven_failure(combined_result)
    return (
        f"Compiler node result: Maven test compilation {status}.\n"
        f"Generated test file: {test_file_path}\n\n"
        "Relevant Maven output:\n"
        f"{output}"
    ).strip()


def _compact_maven_failure(combined_result: str) -> str:
    lines = combined_result.splitlines()
    build_failure_index = _find_first_line_index(lines, ("BUILD FAILURE",))
    if build_failure_index is None:
        compacted = "\n".join(lines[-120:])
    else:
        start_index = max(0, build_failure_index - 120)
        compacted = "\n".join(lines[start_index : build_failure_index + 1])

    if '[ERROR]' in compacted:
        error_index = compacted.find("[ERROR]")
        compacted = compacted[error_index:]
    if '[INFO] Results:' in compacted:
        error_index = compacted.find("[INFO] Results:")
        compacted = compacted[error_index:]

    max_chars = 12000
    if len(compacted) > max_chars:
        return "... truncated Maven failure output\n" + compacted[-max_chars:].lstrip()

    return compacted.strip()


def _find_first_line_index(lines: list[str], markers: tuple[str, ...]) -> int | None:
    for index, line in enumerate(lines):
        if any(marker in line for marker in markers):
            return index
    return None
