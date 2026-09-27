"""Tiered scoring of one generated test class.

Each tier is a gate; failing tier n short-circuits and returns what was measured so far.
The ordering is by cost, which spans four orders of magnitude between tier 0 and tier 4,
and most rejected candidates die at tier 0 or 1.

    tier 0   parse, @Test count, termination      pure Python, ~0 ms
    tier 1   compiles                             mvn test-compile, seconds
    tier 2   tests pass                           mvn test, seconds to minutes
    tier 3   branch coverage                      mvn jacoco:report, seconds
    tier 4   mutation score                       PIT with -Dthreads=12, minutes

Deliberately absent: any repair, any retry, any LLM call. Scoring must measure what the
model produced. Running the harness's repair loop first would credit the model for the
repair node's work and train it to emit conveniently-repairable output.
"""

from __future__ import annotations

import time
from pathlib import Path

from training.scoring import java, maven, pertest, reports
from training.scoring.result import ScoreResult, Status
from training.scoring.workspace import (
    WorkspaceError,
    clear_reports,
    install_test_class,
    remove_test_class,
)


def score(
    test_source: str,
    class_under_test: Path,
    slot: Path,
    project_dir: Path,
    max_tier: int = 4,
    terminated: bool | None = None,
    lenient_fence: bool = False,
) -> ScoreResult:
    """Measure one generated test class inside an already-acquired workspace slot.

    Args:
        test_source: raw model output; a ```java fence is stripped internally.
        class_under_test: path to the production class, relative to `project_dir`.
        slot: an exclusive project copy from WorkspacePool.acquire().
        project_dir: the pool's source project, used to relativise `class_under_test`.
        max_tier: highest tier to attempt.
        terminated: the generation's finish_reason if known. Authoritative when given;
            otherwise a text heuristic is used, which cannot see a length cap.
        lenient_fence: strip a fence whose closing ``` is missing.

            DEFAULT FALSE, and it must stay false for anything comparable to an earlier
            number. The strict stripper is byte-identical to MAVerIT's, and MAVerIT is
            right to give up on an unpaired fence: an answer that never closed is a
            failed generation, and recovering it would flatter the model.

            It is wrong for a MEASUREMENT of the model's syntax, which is what the
            middle-band run is. A rollout that hits its token cap mid-file never emits
            the closing ```, so line 1 keeps its ```java and the file cannot parse for a
            reason that has nothing to do with the Java. 326 of 928 rollouts in the first
            middle-band run carried a live fence into tier 0.
    """
    timings: dict[int, float] = {}
    source = (java.strip_fence_lenient(test_source) if lenient_fence
              else java.strip_markdown_code_fence(test_source))

    # ---- tier 0 -------------------------------------------------------------------
    started = time.monotonic()
    n_tests = java.count_tests(source)
    n_assertless = java.count_test_without_assert(source)
    balanced = java.is_balanced(source)
    ended = java.looks_terminated(test_source) if terminated is None else terminated

    try:
        test_fqn = java.get_java_fully_qualified_name(source)
        test_package = java.get_java_package_name(source)
        test_name = java.get_java_entity_name(source)
        parses = balanced
    except RuntimeError:
        test_fqn = test_package = test_name = ""
        parses = False
    timings[0] = time.monotonic() - started

    # The class under test's own FQN, read from the slot rather than the original, so a
    # multi-module layout resolves against the copy Maven will actually build.
    cut_in_slot = slot / class_under_test.relative_to(project_dir)
    try:
        cut_fqn = java.get_java_fully_qualified_name(cut_in_slot.read_text(encoding="utf-8"))
    except (OSError, RuntimeError) as error:
        return ScoreResult(
            class_under_test=str(class_under_test),
            status=Status.INFRA_ERROR,
            tier_reached=0,
            compiler_output=f"cannot read class under test in slot: {error}",
            wall_time_s=timings,
        )

    base = dict(
        class_under_test=cut_fqn,
        n_tests=n_tests,
        n_assertless_tests=n_assertless,
        parses=parses,
        terminated=bool(ended),
        wall_time_s=timings,
    )

    if not parses:
        return ScoreResult(status=Status.UNPARSEABLE, tier_reached=0, **base)
    if max_tier < 1:
        return ScoreResult(status=Status.OK, tier_reached=0, **base)

    # ---- tiers 1-4 ----------------------------------------------------------------
    try:
        module_dir = maven.get_maven_module_directory(cut_in_slot, slot)
    except RuntimeError as error:
        return ScoreResult(status=Status.INFRA_ERROR, tier_reached=0,
                           compiler_output=str(error), **base)

    installed: Path | None = None
    try:
        # Slots are reused, and we do not run `mvn clean`. Surefire reports accumulate and
        # jacoco.exec APPENDS, so without this the previous class's tests and coverage are
        # attributed to this one -- always upwards, which is the direction that looks like
        # success.
        clear_reports(module_dir)
        installed = install_test_class(module_dir, source, test_package, test_name)

        # tier 1 -- compile
        started = time.monotonic()
        compile_result = maven.run_test_compile(module_dir)
        timings[1] = time.monotonic() - started
        if compile_result["timed_out"]:
            return ScoreResult(status=Status.TIMEOUT, tier_reached=1,
                               compiler_output=_tail(compile_result), **base)
        if not compile_result["ok"]:
            # Attribute the failure to individual @Test methods before the output is
            # tailed -- _tail() keeps the END, and the errors are at the START, so
            # doing this later would attribute a truncated error list.
            marks = pertest.attribute(source, compile_result["combined_result"],
                                      f"{test_name}.java")
            # Then measure the subset that survives dropping the broken tests. The
            # fraction alone cannot rank a partial suite against a trivial compiling
            # one, because coverage and mutation are structurally unmeasured for
            # anything that fails tier 1 -- zero because we never ran it, not zero
            # because it covers nothing.
            salvage: dict = {}
            if (max_tier >= 2 and marks.available and not marks.skeleton_errors
                    and marks.n_broken > 0 and marks.n_clean > 0):
                salvage = _salvage_and_run(
                    marks, source, module_dir, cut_fqn, test_fqn,
                    test_package, test_name, max_tier)
            return ScoreResult(status=Status.COMPILE_FAIL, tier_reached=1,
                               compiler_output=_tail(compile_result),
                               **salvage,
                               n_tests_clean=marks.n_clean,
                               n_tests_broken=marks.n_broken,
                               n_skeleton_errors=len(marks.skeleton_errors),
                               n_compile_errors=marks.n_errors,
                               errors_truncated=marks.truncated,
                               attribution_available=marks.available,
                               **base)
        if max_tier < 2:
            return ScoreResult(status=Status.OK, tier_reached=1, compiled=True, **base)

        # tier 2 -- run. failure.ignore keeps a red suite alive so tier 3 can still
        # measure coverage; pass/fail come from surefire, not the exit code.
        started = time.monotonic()
        test_result = maven.run_tests(module_dir)
        timings[2] = time.monotonic() - started
        if test_result["timed_out"]:
            return ScoreResult(status=Status.TIMEOUT, tier_reached=2, compiled=True,
                               compiler_output=_tail(test_result), **base)

        run, passed, failed = reports.parse_surefire(reports.find_surefire_reports(module_dir))
        green = failed == 0 and run > 0
        counts = dict(compiled=True, tests_run=run, tests_passed=passed, tests_failed=failed)

        if max_tier < 3:
            return ScoreResult(status=Status.OK if green else Status.TEST_FAIL,
                               tier_reached=2, **counts, **base)

        # tier 3 -- coverage. Measurable on a red suite: JaCoCo instruments whatever ran.
        started = time.monotonic()
        jacoco_result = maven.run_jacoco(module_dir)
        timings[3] = time.monotonic() - started
        coverage = None
        cov_extra: dict = {}
        if not jacoco_result["timed_out"]:
            cov = reports.calculate_coverage_for_class(
                reports.find_jacoco_xml_reports(module_dir), cut_fqn)
            coverage = cov.branch_pct
            cov_extra = {"line_coverage_pct": cov.line_pct, "n_branches": cov.n_branches}

        if max_tier < 4 or not green:
            # SIMULATED GREEN. A red suite blocks PIT, so mutation -- the metric this
            # project exists to optimise -- is unmeasurable for ~96% of rollouts and any
            # weight on it is mostly a constant. Drop the failing tests and the rest is a
            # green suite PIT will accept. Removal only, and it never touches `is_green`.
            simulated: dict = {}
            if max_tier >= 4 and not green and passed > 0:
                simulated = _simulate_green(
                    source, module_dir, cut_fqn, test_fqn, test_package, test_name)
            if simulated or not green:
                return ScoreResult(status=Status.OK if green else Status.TEST_FAIL,
                                   tier_reached=3, branch_coverage_pct=coverage,
                                   **cov_extra, **simulated, **counts, **base)
            # PIT needs green: with a failing test it cannot tell an introduced fault from
            # a pre-existing one. mutation stays None -- NOT MEASURED, not zero.
            return ScoreResult(status=Status.OK if green else Status.TEST_FAIL,
                               tier_reached=3, branch_coverage_pct=coverage,
                               **cov_extra, **counts, **base)

        # tier 4 -- mutation
        started = time.monotonic()
        pit_result = maven.run_pitest(module_dir, cut_fqn, test_fqn)
        timings[4] = time.monotonic() - started
        mutation = None
        if not pit_result["timed_out"] and pit_result["ok"]:
            mutation = reports.calculate_mutation_score(
                reports.find_pitest_xml_reports(module_dir))

        return ScoreResult(status=Status.OK, tier_reached=4,
                           branch_coverage_pct=coverage, mutation_score_pct=mutation,
                           **cov_extra, **counts, **base)

    except WorkspaceError as error:
        return ScoreResult(status=Status.INFRA_ERROR, tier_reached=0,
                           compiler_output=str(error), **base)
    except maven.MavenNotFoundError as error:
        return ScoreResult(status=Status.INFRA_ERROR, tier_reached=0,
                           compiler_output=str(error), **base)
    finally:
        # Always remove it: a leftover class from the previous scoring would compile into
        # the next run and inflate its coverage.
        if installed is not None:
            remove_test_class(installed)


def _salvage_and_run(marks, source, module_dir, cut_fqn, test_fqn,
                     test_package, test_name, max_tier) -> dict:
    """Drop the broken @Test methods, then score what is left through tiers 2-4.

    Returns only `salvaged_*` keys, so a caller cannot accidentally promote any of it into
    `compiled` or `tests_run`.

    Nothing is added to the source -- `drop_tests` only removes -- so the subset is
    strictly what the model wrote. The suite as generated is still COMPILE_FAIL and still
    counts as a failure in every reported rate.
    """
    started = time.monotonic()
    out: dict = {"salvaged": False}
    try:
        subset = pertest.drop_tests(source, marks.broken)
        # Reports accumulate and jacoco.exec APPENDS, so without this the FAILED suite's
        # leftovers would be attributed to the salvaged one -- upwards, as always.
        clear_reports(module_dir)
        install_test_class(module_dir, subset, test_package, test_name)

        compiled = maven.run_test_compile(module_dir)
        if compiled["timed_out"] or not compiled["ok"]:
            # Attribution over-credits about 1 predicted-clean test in 15, so the subset
            # failing to compile is expected sometimes rather than a bug.
            return out
        out["salvaged"] = True
        if max_tier < 2:
            return out

        tests = maven.run_tests(module_dir)
        if tests["timed_out"]:
            return out
        run, passed, failed = reports.parse_surefire(
            reports.find_surefire_reports(module_dir))
        out.update(salvaged_tests_run=run, salvaged_tests_passed=passed,
                   salvaged_tests_failed=failed)
        if max_tier < 3:
            return out

        jacoco = maven.run_jacoco(module_dir)
        if not jacoco["timed_out"]:
            # Both counters, so the salvaged subset gets the same line-coverage fallback
            # on a branchless class that the full suite does. Without it a salvaged husk
            # collects a free 100% and the v2 veto cannot fire on it.
            cov = reports.calculate_coverage_for_class(
                reports.find_jacoco_xml_reports(module_dir), cut_fqn)
            out["salvaged_branch_coverage_pct"] = cov.branch_pct
            out["salvaged_line_coverage_pct"] = cov.line_pct
            out["salvaged_n_branches"] = cov.n_branches
        if max_tier < 4 or failed != 0 or run == 0:
            return out

        pit = maven.run_pitest(module_dir, cut_fqn, test_fqn)
        if not pit["timed_out"] and pit["ok"]:
            out["salvaged_mutation_score_pct"] = reports.calculate_mutation_score(
                reports.find_pitest_xml_reports(module_dir))
        return out
    except (WorkspaceError, maven.MavenNotFoundError, RuntimeError, OSError):
        # Salvage is a bonus measurement. It must never turn a usable COMPILE_FAIL into an
        # INFRA_ERROR, which would drop the rollout from the dataset entirely.
        return out
    finally:
        out["salvage_seconds"] = round(time.monotonic() - started, 1)


def _simulate_green(source, module_dir, cut_fqn, test_fqn, test_package, test_name) -> dict:
    """Drop the FAILING tests, then measure what the passing ones assert.

    Returns only `green_subset_*` keys, so nothing here can be promoted into `is_green` or
    `mutation_score_pct` by accident -- those stay the suite as generated.

    Coverage is recomputed ON THE SUBSET. Crediting the full suite's coverage alongside the
    subset's mutation would count tests that were removed, which is the mistake the salvage
    path already avoids.
    """
    started = time.monotonic()
    out: dict = {}
    try:
        failing = reports.parse_surefire_failures(reports.find_surefire_reports(module_dir))
        if not failing:
            return out
        subset = pertest.drop_tests(source, failing)
        if not pertest.test_spans(subset):
            return out                     # every test failed; nothing to measure

        clear_reports(module_dir)
        install_test_class(module_dir, subset, test_package, test_name)
        compiled = maven.run_test_compile(module_dir)
        if compiled["timed_out"] or not compiled["ok"]:
            return out                     # a dropped test took a helper with it
        tests = maven.run_tests(module_dir)
        if tests["timed_out"]:
            return out
        run, passed, failed = reports.parse_surefire(
            reports.find_surefire_reports(module_dir))
        if failed or run == 0:
            return out                     # still red: PIT would refuse it anyway
        out["n_tests_dropped_for_green"] = len(failing)

        jacoco = maven.run_jacoco(module_dir)
        if not jacoco["timed_out"]:
            cov = reports.calculate_coverage_for_class(
                reports.find_jacoco_xml_reports(module_dir), cut_fqn)
            out["green_subset_branch_coverage_pct"] = cov.branch_pct
            out["green_subset_line_coverage_pct"] = cov.line_pct
            out["green_subset_n_branches"] = cov.n_branches

        pit = maven.run_pitest(module_dir, cut_fqn, test_fqn)
        if not pit["timed_out"] and pit["ok"]:
            out["green_subset_mutation_score_pct"] = reports.calculate_mutation_score(
                reports.find_pitest_xml_reports(module_dir))
        return out
    except (WorkspaceError, maven.MavenNotFoundError, RuntimeError, OSError):
        # A bonus measurement must never turn a usable TEST_FAIL into an INFRA_ERROR,
        # which would drop the rollout from the dataset entirely.
        return out
    finally:
        out["green_subset_seconds"] = round(time.monotonic() - started, 1)


def _tail(result: dict, limit: int = 4000) -> str:
    """Keep the end of Maven's output -- that is where the errors are."""
    return result["combined_result"][-limit:]
