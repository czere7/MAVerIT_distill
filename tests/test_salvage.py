"""Salvage: the subset is scored, the headline is not touched, and broken tests only hurt."""

import pytest

from training.quality import quality
from training.scoring import pertest
from training.scoring.result import ScoreResult, Status

pytest.importorskip("tree_sitter_java")

SUITE = """package org.example;

import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class WidgetTest {

    @Test
    public void alpha() {
        assertEquals(1, new Widget().size());
    }

    @Test
    public void beta() {
        assertEquals(2, new Widget().nope());
    }

    @Test
    public void gamma() {
        assertEquals(3, new Widget().depth());
    }
}
"""


class TestDropTests:
    def test_removes_only_the_named_methods(self):
        out = pertest.drop_tests(SUITE, {"beta"})
        assert "beta" not in out
        assert "alpha" in out and "gamma" in out

    def test_result_is_a_strict_subset_of_the_input(self):
        # Removal, never repair: every kept line must have come from the original.
        out = pertest.drop_tests(SUITE, {"beta"})
        original = set(SUITE.splitlines())
        assert all(line in original for line in out.splitlines())

    def test_the_subset_still_parses(self):
        out = pertest.drop_tests(SUITE, {"beta"})
        assert {s.name for s in pertest.test_spans(out)} == {"alpha", "gamma"}

    def test_dropping_nothing_is_a_no_op(self):
        assert pertest.drop_tests(SUITE, set()) == SUITE


class TestHeadlineIsolation:
    """Salvage must never reach a reported rate."""

    def _salvaged(self, **kw):
        base = dict(class_under_test="X", status=Status.COMPILE_FAIL, tier_reached=1,
                    n_tests_clean=2, n_tests_broken=1, n_compile_errors=1,
                    salvaged=True, salvaged_tests_run=2, salvaged_tests_passed=2,
                    salvaged_tests_failed=0, salvaged_branch_coverage_pct=100.0)
        return ScoreResult(**{**base, **kw})

    def test_compiled_stays_false(self):
        assert self._salvaged().compiled is False

    def test_is_green_stays_false_even_when_the_subset_is_green(self):
        r = self._salvaged()
        assert r.salvaged_is_green is True
        assert r.is_green is False

    def test_headline_test_counts_stay_zero(self):
        r = self._salvaged()
        assert (r.tests_run, r.tests_passed, r.tests_failed) == (0, 0, 0)
        assert r.pass_rate == 0.0

    def test_headline_coverage_stays_unmeasured(self):
        r = self._salvaged()
        assert r.branch_coverage_pct is None
        assert r.salvaged_branch_coverage_pct == 100.0

    def test_binary_quality_ignores_salvage_entirely(self):
        assert quality(self._salvaged()) == 0.0


class TestGradedUsesTheSubset:
    def _r(self, **kw):
        base = dict(class_under_test="X", status=Status.COMPILE_FAIL, tier_reached=1,
                    n_compile_errors=1)
        return ScoreResult(**{**base, **kw})

    def _partial(self, coverage, mutation=None):
        """4 tests, 2 broken; the 2 survivors pass."""
        return self._r(n_tests_clean=2, n_tests_broken=2, salvaged=True,
                       salvaged_tests_run=2, salvaged_tests_passed=2,
                       salvaged_tests_failed=0,
                       salvaged_branch_coverage_pct=coverage,
                       salvaged_mutation_score_pct=mutation)

    def _trivial(self, coverage):
        return ScoreResult(class_under_test="X", status=Status.OK, tier_reached=3,
                           compiled=True, tests_run=2, tests_passed=2, tests_failed=0,
                           branch_coverage_pct=coverage)

    def test_coverage_alone_does_not_beat_compiling(self):
        """The crossover is steeper than it looks, and this pins the number.

        Losing half the suite costs 0.4 x 0.5 = 0.2 of the compilation term. Coverage can
        return at most 0.2 in total, so perfect coverage against a trivial suite's 20%
        earns back only 0.16 -- not enough on its own. That is the compilation weight
        doing what spec O7 asks of it, not a defect, but it means partial credit does NOT
        automatically rank a half-broken suite above a weak compiling one.
        """
        assert quality(self._partial(100.0), graded_compile=True) \
            < quality(self._trivial(20.0), graded_compile=True)

    def test_coverage_plus_mutation_does_beat_it(self):
        # With PIT run on the green subset, the partial suite clears the bar.
        assert quality(self._partial(100.0, mutation=60.0), graded_compile=True) \
            > quality(self._trivial(20.0), graded_compile=True)

    def test_the_break_even_is_the_lost_compilation_credit(self):
        """partial > trivial iff (cov+mut)_partial - (cov+mut)_trivial > 0.4*(1-fraction)/0.2."""
        lost = 0.4 * (1 - 0.5)
        gain = 0.2 * ((1.00 + 0.60) - (0.20 + 0.00))
        assert gain > lost
        assert quality(self._partial(100.0, 60.0), graded_compile=True) \
            - quality(self._trivial(20.0), graded_compile=True) == pytest.approx(gain - lost)

    def test_coverage_of_the_subset_raises_the_score(self):
        low = self._r(n_tests_clean=2, n_tests_broken=2, salvaged=True,
                      salvaged_tests_run=2, salvaged_tests_passed=2,
                      salvaged_branch_coverage_pct=10.0)
        high = self._r(n_tests_clean=2, n_tests_broken=2, salvaged=True,
                       salvaged_tests_run=2, salvaged_tests_passed=2,
                       salvaged_branch_coverage_pct=90.0)
        assert quality(high, graded_compile=True) > quality(low, graded_compile=True)

    def test_a_subset_that_would_not_compile_scores_only_the_fraction(self):
        r = self._r(n_tests_clean=2, n_tests_broken=2, salvaged=False)
        assert quality(r, graded_compile=True) == pytest.approx(0.2)

    def test_mutation_needs_a_green_subset(self):
        red = self._r(n_tests_clean=2, n_tests_broken=2, salvaged=True,
                      salvaged_tests_run=2, salvaged_tests_passed=1,
                      salvaged_tests_failed=1, salvaged_mutation_score_pct=80.0)
        assert red.salvaged_is_green is False
        # the mutation term contributes nothing; only fraction and pass_rate do
        assert quality(red, graded_compile=True) == pytest.approx(0.2 + 0.2 * 0.5)


class TestNoShotgunIncentive:
    """Adding a broken test must always lower the score, never raise it."""

    def _r(self, clean, broken, **kw):
        base = dict(class_under_test="X", status=Status.COMPILE_FAIL, tier_reached=1,
                    n_tests_clean=clean, n_tests_broken=broken, n_compile_errors=1,
                    salvaged=True, salvaged_tests_run=clean, salvaged_tests_passed=clean,
                    salvaged_tests_failed=0, salvaged_branch_coverage_pct=60.0)
        return ScoreResult(**{**base, **kw})

    def test_each_extra_broken_test_strictly_lowers_the_score(self):
        scores = [quality(self._r(5, b), graded_compile=True) for b in range(1, 8)]
        assert scores == sorted(scores, reverse=True)
        assert len(set(scores)) == len(scores)

    def test_writing_the_same_good_tests_cleanly_always_wins(self):
        # 5 good tests inside a 30-test spray, against the same 5 written on their own.
        spray = self._r(5, 25)
        clean = ScoreResult(class_under_test="X", status=Status.OK, tier_reached=3,
                            compiled=True, tests_run=5, tests_passed=5, tests_failed=0,
                            branch_coverage_pct=60.0)
        assert quality(clean, graded_compile=True) > quality(spray, graded_compile=True)

    def test_a_broken_test_cannot_add_coverage(self):
        # The subset drops it before anything runs, so the coverage term is identical.
        few = self._r(5, 1)
        many = self._r(5, 20)
        assert few.salvaged_branch_coverage_pct == many.salvaged_branch_coverage_pct
        assert quality(few, graded_compile=True) > quality(many, graded_compile=True)
