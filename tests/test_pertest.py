"""Per-@Test attribution: the span finder, the javac parser, and the grading rules."""

import pytest

from training.quality import quality
from training.scoring import pertest
from training.scoring.result import ScoreResult, Status

pytest.importorskip("tree_sitter_java")

SUITE = """package org.example;

import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class WidgetTest {

    private final Widget widget = new Widget();

    @Test
    public void firstIsFine() {
        assertEquals(1, widget.size());
    }

    @Test
    public void secondIsBroken() {
        assertEquals(2, widget.nonExistentMethod());
    }

    private int helper() {
        return 3;
    }
}
"""


def _error(line, message, path="/slot/src/test/java/org/example/WidgetTest.java"):
    return f"[ERROR] {path}:[{line},9] {message}"


class TestSpans:
    def test_finds_only_annotated_methods(self):
        spans = pertest.test_spans(SUITE)
        assert [s.name for s in spans] == ["firstIsFine", "secondIsBroken"]

    def test_span_covers_the_annotation_line(self):
        # @Test(expected = X.class) reports on the annotation, above the declaration.
        first = pertest.test_spans(SUITE)[0]
        assert SUITE.splitlines()[first.start_line - 1].strip() == "@Test"

    def test_spans_do_not_overlap_and_are_ordered(self):
        spans = pertest.test_spans(SUITE)
        assert spans[0].end_line < spans[1].start_line

    def test_recovers_tests_from_a_file_that_does_not_parse(self):
        broken = SUITE.replace("return 3;", "return 3 ###;")
        assert "firstIsFine" in {s.name for s in pertest.test_spans(broken)}


class TestJavacParsing:
    def test_deduplicates_mavens_doubled_report(self):
        out = "\n".join([_error(12, "cannot find symbol"), _error(12, "cannot find symbol")])
        errors, _ = pertest.parse_javac_errors(out, "WidgetTest.java")
        assert errors == [(12, "cannot find symbol")]

    def test_ignores_errors_in_other_files(self):
        out = _error(12, "boom", path="/slot/src/main/java/org/example/Widget.java")
        errors, _ = pertest.parse_javac_errors(out, "WidgetTest.java")
        assert errors == []

    def test_flags_the_javac_error_cap(self):
        out = _error(12, "boom") + "\n[INFO] 100 errors"
        _, truncated = pertest.parse_javac_errors(out, "WidgetTest.java")
        assert truncated is True


class TestAttribution:
    def test_error_inside_a_test_breaks_only_that_test(self):
        line = pertest.test_spans(SUITE)[1].start_line + 2
        marks = pertest.attribute(SUITE, _error(line, "cannot find symbol"), "WidgetTest.java")
        assert marks.broken == {"secondIsBroken"}
        assert marks.n_clean == 1
        assert marks.compiled_fraction == 0.5

    def test_error_in_the_skeleton_zeroes_the_whole_suite(self):
        marks = pertest.attribute(SUITE, _error(3, "package does not exist"), "WidgetTest.java")
        assert marks.skeleton_errors
        assert marks.n_clean == 0
        assert marks.compiled_fraction == 0.0

    def test_error_in_a_private_helper_counts_as_skeleton(self):
        # A helper is not inside any @Test span, and a test calling it carries no error of
        # its own -- so this must fail the suite rather than look like 2 of 2 clean.
        helper_line = SUITE.splitlines().index("        return 3;") + 1
        marks = pertest.attribute(SUITE, _error(helper_line, "boom"), "WidgetTest.java")
        assert marks.skeleton_errors
        assert marks.compiled_fraction == 0.0

    def test_clean_output_leaves_every_test_intact(self):
        marks = pertest.attribute(SUITE, "", "WidgetTest.java")
        assert marks.compiled_fraction == 1.0
        assert marks.n_errors == 0

    def test_a_suite_with_no_tests_scores_zero_not_one(self):
        empty = "package org.example;\npublic class WidgetTest {\n}\n"
        assert pertest.attribute(empty, "", "WidgetTest.java").compiled_fraction == 0.0

    def test_unavailable_parser_yields_zero_not_a_crash(self):
        marks = pertest.Attribution(available=False)
        assert marks.compiled_fraction == 0.0


class TestQualityIntegration:
    def _result(self, **kw):
        base = dict(class_under_test="X", status=Status.COMPILE_FAIL, tier_reached=1)
        return ScoreResult(**{**base, **kw})

    def test_binary_term_is_unchanged_by_default(self):
        r = self._result(n_tests_clean=6, n_tests_broken=2)
        assert quality(r) == 0.0

    def test_graded_term_ranks_partial_suites(self):
        good = self._result(n_tests_clean=6, n_tests_broken=2)
        bad = self._result(n_tests_clean=1, n_tests_broken=7)
        assert quality(good, graded_compile=True) > quality(bad, graded_compile=True) > 0.0

    def test_graded_partial_never_beats_a_compiling_suite(self):
        partial = self._result(n_tests_clean=99, n_tests_broken=1)
        compiling = ScoreResult(class_under_test="X", status=Status.TEST_FAIL, tier_reached=2,
                                compiled=True, tests_run=1, tests_passed=0, tests_failed=1)
        assert quality(partial, graded_compile=True) < quality(compiling, graded_compile=True)

    def test_skeleton_fault_scores_zero_even_when_graded(self):
        r = self._result(n_tests_clean=0, n_tests_broken=0, n_skeleton_errors=3)
        assert quality(r, graded_compile=True) == 0.0

    def test_infra_error_still_refuses_to_be_scored(self):
        r = ScoreResult(class_under_test="X", status=Status.INFRA_ERROR, tier_reached=0)
        with pytest.raises(ValueError):
            quality(r, graded_compile=True)

class TestGradedInvariants:
    """A failure may be ranked against other failures. It may never tie a success."""

    def _failed(self, **kw):
        base = dict(class_under_test="X", status=Status.COMPILE_FAIL, tier_reached=1)
        return ScoreResult(**{**base, **kw})

    def _compiled(self, **kw):
        base = dict(class_under_test="X", status=Status.TEST_FAIL, tier_reached=2,
                    compiled=True, tests_run=1, tests_passed=0, tests_failed=1)
        return ScoreResult(**{**base, **kw})

    def test_unexplained_compile_failure_scores_zero(self):
        # javac refused the class but nothing was attributed to it. Every test looks
        # clean, which used to score a full 0.4 -- the same as a suite that compiled.
        r = self._failed(n_tests_clean=8, n_tests_broken=0, n_compile_errors=0)
        assert r.compiled_fraction == 0.0
        assert quality(r, graded_compile=True) == 0.0

    def test_no_failure_can_reach_the_compiling_floor(self):
        floor = quality(self._compiled(), graded_compile=True)
        worst_case = self._failed(n_tests_clean=999, n_tests_broken=1, n_compile_errors=1)
        assert quality(worst_case, graded_compile=True) < floor

    def test_partial_credit_is_capped_by_the_compilation_weight(self):
        r = self._failed(n_tests_clean=7, n_tests_broken=1, n_compile_errors=1)
        assert 0.0 < quality(r, graded_compile=True) < 0.4

    def test_more_clean_tests_scores_higher(self):
        better = self._failed(n_tests_clean=7, n_tests_broken=1, n_compile_errors=1)
        worse = self._failed(n_tests_clean=2, n_tests_broken=6, n_compile_errors=6)
        assert quality(better, graded_compile=True) > quality(worse, graded_compile=True)
