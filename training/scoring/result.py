"""Result types for the scoring harness.

The one design rule here: `INFRA_ERROR` must never be confusable with a bad test class.
Scoring a Maven-not-found or a worktree collision as quality zero would put pairs into the
preference dataset that teach nothing, and the damage would be invisible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    """Why scoring stopped. Ordered from best to worst outcome."""

    OK = "OK"                        # reached the requested tier
    TEST_FAIL = "TEST_FAIL"          # compiled, some tests failed -> PIT unavailable
    COMPILE_FAIL = "COMPILE_FAIL"    # did not compile
    UNPARSEABLE = "UNPARSEABLE"      # tier 0: no class declaration, unbalanced delimiters
    TIMEOUT = "TIMEOUT"              # a Maven tier exceeded its timeout
    INFRA_ERROR = "INFRA_ERROR"      # our problem, not the model's -- EXCLUDE from datasets

    @property
    def is_model_fault(self) -> bool:
        """True when the outcome says something about the generated class.

        TIMEOUT is excluded: a slow build is not evidence of a bad test suite.
        """
        return self in {Status.OK, Status.TEST_FAIL, Status.COMPILE_FAIL, Status.UNPARSEABLE}


@dataclass(frozen=True)
class ScoreResult:
    """One generated test class, measured.

    Coverage and mutation are PERCENTAGES in [0, 100] -- MAVerIT's convention, and the
    scale of report.csv. `None` means NOT MEASURED, which is a different fact from zero:
    mutation is None whenever the suite is red, because PIT cannot distinguish an
    introduced fault from a pre-existing failure.
    """

    class_under_test: str
    status: Status
    tier_reached: int

    # tier 0 -- free
    n_tests: int = 0
    n_assertless_tests: int = 0
    parses: bool = False
    terminated: bool = False

    # tiers 1-2
    compiled: bool = False
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0

    # tiers 3-4
    branch_coverage_pct: float | None = None
    mutation_score_pct: float | None = None

    # Line coverage and the branch COUNT, recorded together because the percentage alone
    # is ambiguous: branch coverage reads 100.0 both for a fully covered class and for one
    # with no branches at all. `n_branches` separates them.
    line_coverage_pct: float | None = None
    n_branches: int | None = None

    # SIMULATED GREEN. PIT refuses a red suite -- it cannot tell an introduced fault from a
    # pre-existing failure -- so mutation is unmeasurable for ~96% of rollouts, which makes
    # any weight on it mostly a constant. Dropping the FAILING tests turns a red suite into
    # a green one whose mutation score is measurable; the score is then discounted by the
    # share of tests that survived, so a suite that had to throw away half of itself keeps
    # half the credit.
    #
    # Like salvage, this is removal only, and like salvage it never feeds a headline:
    # `is_green` and `mutation_score_pct` stay the suite AS GENERATED.
    green_subset_mutation_score_pct: float | None = None
    green_subset_branch_coverage_pct: float | None = None
    green_subset_line_coverage_pct: float | None = None
    green_subset_n_branches: int | None = None
    n_tests_dropped_for_green: int = 0
    green_subset_seconds: float = 0.0

    # per-@Test attribution (tier 1). See training/scoring/pertest.py for what these
    # do and do not prove -- `n_tests_clean` is a proxy for "compiles alone", not a
    # proof, and it is deliberately zeroed when the skeleton is broken.
    n_tests_clean: int = 0
    n_tests_broken: int = 0
    n_skeleton_errors: int = 0
    n_compile_errors: int = 0
    errors_truncated: bool = False
    attribution_available: bool = True

    # SALVAGE (tier 1 fallback). The suite did not compile; these describe the
    # subset that survives once the broken @Test methods are DROPPED -- no repair,
    # no additions, a strict subset of what the model wrote.
    #
    # THESE NEVER FEED A HEADLINE. `compiled`, `is_green`, `tests_run` and the
    # reported compile/green rates describe the suite AS GENERATED, because a suite
    # that does not compile is worth nothing to a user whatever its survivors cover.
    # Salvage exists to rank failures against each other for training signal, and
    # `quality(graded_compile=True)` is the only thing that reads it.
    salvaged: bool = False
    salvaged_tests_run: int = 0
    salvaged_tests_passed: int = 0
    salvaged_tests_failed: int = 0
    salvaged_branch_coverage_pct: float | None = None
    salvaged_line_coverage_pct: float | None = None
    salvaged_n_branches: int | None = None
    salvaged_mutation_score_pct: float | None = None
    salvage_seconds: float = 0.0

    # diagnostics
    compiler_output: str = ""
    wall_time_s: dict[int, float] = field(default_factory=dict)

    @property
    def compiled_fraction(self) -> float:
        """Share of @Test methods that compile, in [0, 1]. 1.0 when the suite compiled.

        The graded counterpart to `compiled`. A suite that compiles scores 1.0 whatever
        the attribution said; a suite that does not scores the share of its tests with
        no javac error attributed, and 0.0 if the fault is in the skeleton, since
        dropping tests cannot rescue a bad import.

        INVARIANT: a suite that did not compile scores strictly below 1.0, so under
        `quality(graded_compile=True)` it can never reach the 0.4 floor of a suite that
        did. Partial credit ranks failures against each other; it must not let a failure
        tie a success.
        """
        if self.compiled:
            return 1.0
        if not self.attribution_available or self.n_skeleton_errors:
            return 0.0
        if not self.n_compile_errors and not self.n_tests_broken:
            # javac refused the class and NOTHING was attributed to it -- the failure is
            # real but unexplained (an error in another file, a javac message with no line
            # number, a format the parser does not recognise). Every test then looks clean
            # and the suite scores a full 0.4, TYING a suite that genuinely compiled. A
            # failure we cannot account for is scored as total failure.
            #
            # Both conditions, not just the error count: a broken test is itself proof that
            # an error was attributed, so requiring only  would zero any
            # result whose error count was never populated.
            return 0.0
        total = self.n_tests_clean + self.n_tests_broken
        return self.n_tests_clean / total if total else 0.0

    @staticmethod
    def _effective(branch_pct, line_pct, n_branches) -> float:
        """Branch coverage, or line coverage when the class has no branches.

        A branchless class is not fully tested because it has no branches; it is untested
        until something runs its lines. Without this, 43% of compiling rollouts collect a
        free 100% and a coverage veto can never fire on them.
        """
        if branch_pct is None and line_pct is None:
            return 0.0
        if n_branches:
            return branch_pct or 0.0
        return line_pct if line_pct is not None else (branch_pct or 0.0)

    @property
    def effective_coverage_pct(self) -> float:
        return self._effective(self.branch_coverage_pct, self.line_coverage_pct,
                               self.n_branches)

    @property
    def salvaged_effective_coverage_pct(self) -> float:
        return self._effective(self.salvaged_branch_coverage_pct,
                               self.salvaged_line_coverage_pct, self.salvaged_n_branches)

    @property
    def green_subset_effective_coverage_pct(self) -> float:
        return self._effective(self.green_subset_branch_coverage_pct,
                               self.green_subset_line_coverage_pct,
                               self.green_subset_n_branches)

    @property
    def salvaged_pass_rate(self) -> float:
        return (self.salvaged_tests_passed / self.salvaged_tests_run
                if self.salvaged_tests_run else 0.0)

    @property
    def salvaged_is_green(self) -> bool:
        """The SALVAGED subset ran and nothing failed. Not a headline; see `is_green`."""
        return (self.salvaged and self.salvaged_tests_run > 0
                and self.salvaged_tests_failed == 0)

    @property
    def is_green(self) -> bool:
        """A suite PIT can work with: it compiled, ran something, and nothing failed."""
        return self.compiled and self.tests_run > 0 and self.tests_failed == 0

    @property
    def pass_rate(self) -> float:
        return self.tests_passed / self.tests_run if self.tests_run else 0.0

    @property
    def usable_for_training(self) -> bool:
        """Whether this result may enter a preference dataset at all."""
        return self.status.is_model_fault

    def summary(self) -> str:
        cov = "--" if self.branch_coverage_pct is None else f"{self.branch_coverage_pct:.1f}%"
        mut = "--" if self.mutation_score_pct is None else f"{self.mutation_score_pct:.1f}%"
        return (
            f"{self.status.value:<12} T{self.tier_reached} "
            f"tests={self.n_tests:<3} "
            f"pass={self.tests_passed}/{self.tests_run:<3} "
            f"cov={cov:<7} mut={mut:<7} {self.class_under_test}"
        )
