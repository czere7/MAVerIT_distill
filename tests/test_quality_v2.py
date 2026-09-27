"""quality_v2: the veto, the missing floor, the branchless fallback, the pass fraction."""

import pytest

from training.quality import quality, quality_v2, SALVAGE_CEILING
from training.scoring.reports import Coverage
from training.scoring.result import ScoreResult, Status


def compiling(**kw):
    base = dict(class_under_test="X", status=Status.OK, tier_reached=4, compiled=True,
                tests_run=5, tests_passed=5, tests_failed=0,
                branch_coverage_pct=60.0, line_coverage_pct=60.0, n_branches=20)
    return ScoreResult(**{**base, **kw})


def failing(**kw):
    base = dict(class_under_test="X", status=Status.COMPILE_FAIL, tier_reached=1,
                n_compile_errors=1)
    return ScoreResult(**{**base, **kw})


class TestCoverageFallback:
    def test_branchless_falls_back_to_line(self):
        assert Coverage(branch_pct=100.0, line_pct=20.0, n_branches=0, n_lines=10
                        ).effective_pct == 20.0

    def test_branchy_uses_branch(self):
        assert Coverage(branch_pct=45.0, line_pct=90.0, n_branches=20, n_lines=10
                        ).effective_pct == 45.0

    def test_result_exposes_the_same_rule(self):
        r = compiling(branch_coverage_pct=100.0, line_coverage_pct=15.0, n_branches=0)
        assert r.branch_coverage_pct == 100.0
        assert r.effective_coverage_pct == 15.0


class TestTheVeto:
    def test_compiling_with_no_coverage_scores_zero(self):
        r = compiling(branch_coverage_pct=0.0, line_coverage_pct=0.0)
        assert quality(r, graded_compile=True) > 0.5     # v1 paid for this
        assert quality_v2(r) == 0.0                      # v2 does not

    def test_a_branchless_husk_cannot_hide_behind_free_branch_coverage(self):
        # branch says 100% because there are no branches; line says nothing ran
        r = compiling(branch_coverage_pct=100.0, line_coverage_pct=0.0, n_branches=0)
        assert quality_v2(r) == 0.0

    def test_one_covered_branch_is_enough_to_pass_the_veto(self):
        r = compiling(branch_coverage_pct=5.0, line_coverage_pct=5.0)
        assert quality_v2(r) > 0.0

    def test_zero_branches_but_lines_executed_is_NOT_vetoed(self):
        """The line rule spares a suite that ran code without taking a branch.

        Vetoing on branch was measured and was too harsh: 18 of the 32 rollouts it
        rejected had executed code, up to 45.8% of their class's lines. Nysiis runs 26.5%
        of its lines at zero branch coverage, and a branch veto scored that identically to
        a suite that ran nothing at all.
        """
        r = compiling(branch_coverage_pct=0.0, line_coverage_pct=26.5, n_branches=71)
        assert quality_v2(r) > 0.0

    def test_line_gates_but_branch_credits(self):
        """A deliberate asymmetry: clearing the gate does not itself earn anything.

        Line coverage answers "did this suite touch the class"; branch coverage answers
        "how much of it". A branchy class exercised only on straight-line code clears the
        veto and still collects no coverage credit.
        """
        r = compiling(branch_coverage_pct=0.0, line_coverage_pct=40.0, n_branches=20,
                      tests_run=4, tests_passed=4)
        assert quality_v2(r) == pytest.approx(0.4)


class TestNoFloorForCompiling:
    def test_compiling_alone_earns_nothing(self):
        # runs nothing, covers nothing
        r = compiling(tests_run=0, tests_passed=0, branch_coverage_pct=0.0,
                      line_coverage_pct=0.0)
        assert quality_v2(r) == 0.0

    def test_v1_paid_a_floor_and_v2_does_not(self):
        r = compiling(branch_coverage_pct=0.0, line_coverage_pct=0.0)
        assert quality(r, graded_compile=True) >= 0.4
        assert quality_v2(r) == 0.0


class TestPassFractionIsOneVariable:
    def test_mutation_is_discounted_by_the_pass_fraction(self):
        half = compiling(status=Status.TEST_FAIL, tests_run=6, tests_passed=3,
                         tests_failed=3, green_subset_mutation_score_pct=100.0)
        full = compiling(mutation_score_pct=100.0)
        assert quality_v2(half) < quality_v2(full)

    def test_all_passing_beats_better_assertions_on_half_a_suite(self):
        # a deliberate consequence: red tests are defects
        strong_half = compiling(status=Status.TEST_FAIL, tests_run=6, tests_passed=3,
                                tests_failed=3, green_subset_mutation_score_pct=100.0)
        weak_full = compiling(mutation_score_pct=50.0)
        assert quality_v2(weak_full) > quality_v2(strong_half)

    def test_unmeasurable_mutation_costs_only_its_own_term(self):
        with_mut = compiling(mutation_score_pct=80.0)
        without = compiling(mutation_score_pct=None)
        assert quality_v2(with_mut) - quality_v2(without) == pytest.approx(0.4 * 0.8)

    def test_score_is_bounded_by_one(self):
        best = compiling(branch_coverage_pct=100.0, line_coverage_pct=100.0,
                         mutation_score_pct=100.0)
        assert quality_v2(best) == pytest.approx(1.0)


class TestSalvageBandStaysBelow:
    def _salvaged(self, **kw):
        base = dict(n_tests_clean=6, n_tests_broken=2, salvaged=True,
                    salvaged_tests_run=6, salvaged_tests_passed=6,
                    salvaged_branch_coverage_pct=80.0, salvaged_line_coverage_pct=80.0,
                    salvaged_n_branches=10)
        return failing(**{**base, **kw})

    def test_a_good_salvage_still_loses_to_the_weakest_compiling_suite(self):
        floor = quality_v2(compiling(branch_coverage_pct=5.0, line_coverage_pct=5.0,
                                     tests_run=1, tests_passed=0, tests_failed=1,
                                     status=Status.TEST_FAIL))
        assert quality_v2(self._salvaged()) < max(floor, SALVAGE_CEILING)

    def test_the_veto_applies_to_salvage_too(self):
        husk = self._salvaged(salvaged_branch_coverage_pct=100.0,
                              salvaged_line_coverage_pct=0.0, salvaged_n_branches=0)
        assert quality_v2(husk) == 0.0

    def test_skeleton_faults_still_score_zero(self):
        assert quality_v2(self._salvaged(n_skeleton_errors=3)) == 0.0

    def test_more_surviving_tests_scores_higher(self):
        better = self._salvaged(n_tests_clean=7, n_tests_broken=1)
        worse = self._salvaged(n_tests_clean=2, n_tests_broken=6)
        assert quality_v2(better) > quality_v2(worse)


class TestV1IsUntouched:
    def test_v1_still_pays_the_compile_floor(self):
        r = compiling(branch_coverage_pct=0.0, line_coverage_pct=0.0)
        assert quality(r) >= 0.4

    def test_infra_error_refused_by_both(self):
        r = ScoreResult(class_under_test="X", status=Status.INFRA_ERROR, tier_reached=0)
        with pytest.raises(ValueError):
            quality(r)
        with pytest.raises(ValueError):
            quality_v2(r)
