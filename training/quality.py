"""The rewards that rank a model's rollouts. Ported from mpref/pairs/quality.py.

    quality = 0.4 · compiled
            + 0.2 · pass_rate
            + 0.2 · branch_coverage
            + 0.2 · mutation_score      (0 when the suite is red -- PIT needs green)

Rejection-sampling fine-tuning ranks candidates with  and
uses  only as a veto: v2 cannot bootstrap (no non-compiling rollout clears its
salvage ceiling), and as a ranker it halved eligibility while barely moving suite length.
The preference-pair machinery that used to live here (raw_delta, DeltaNormaliser) was not
ported; the preference phase is closed.

The obvious worry -- that forty good tests with one failure lose the mutation term while
one trivial passing test keeps it -- does not materialise, because PIT's denominator is
fixed by the class under test, not by the suite. A trivial suite kills ~5% of the mutants
that exist, not 100% of a tiny set.

 means NOT MEASURED. It is collapsed to 0 only here, inside
quality(), and never in ScoreResult -- a red suite genuinely cannot be trusted, but the
distinction has to survive into the data.
"""

from __future__ import annotations

from training.scoring.result import ScoreResult

W_COMPILED = 0.4
W_PASS_RATE = 0.2
W_COVERAGE = 0.2
W_MUTATION = 0.2

# v2: the whole non-compiling band is scaled below this, and the least a compiling
# suite can score is W2_COVERAGE * (one covered branch), so 0.15 leaves clear air.
SALVAGE_CEILING = 0.15


def quality(result: ScoreResult, graded_compile: bool = False) -> float:
    """Scalar quality in [0, 1]. Higher is better.

    Args:
        graded_compile: replace the binary compilation term with the share of @Test
            methods that compile.

            DEFAULT FALSE. Every dataset built so far, and both delta-bounds files,
            were fitted against the binary term; flipping the default would silently
            change delta for pairs already in training runs.

            It exists because the binary term cannot rank failures, and failure is
            almost the entire distribution: `quality_rejected` is 0.000 for all 93
            pairs of the ODPO dataset, and 102 of 116 middle-band prompts produce no
            compiling rollout in eight tries. Measured on a 51-rollout pilot, grading
            per test moves 26% of rollouts off zero.
    """
    if not result.usable_for_training:
        raise ValueError(
            f"{result.status.value} is an infrastructure outcome and has no quality; "
            "such results must be excluded from datasets, not scored as zero")

    if not result.compiled:
        if not graded_compile:
            return 0.0
        # The other three terms come from the SALVAGED subset -- the same tests minus
        # the broken ones -- because for a failing suite they are otherwise zero for
        # want of a run rather than for want of coverage. A suite whose two working
        # tests cover every branch is worth more than a trivial suite that compiles,
        # and only a measured coverage term can say so.
        #
        # Adding a broken test always LOWERS this: it cuts compiled_fraction and is
        # dropped before the subset runs, so it cannot raise pass, coverage or
        # mutation. There is no incentive to write tests and hope.
        coverage = (result.salvaged_branch_coverage_pct or 0.0) / 100.0
        mutation = ((result.salvaged_mutation_score_pct or 0.0) / 100.0
                    if result.salvaged_is_green else 0.0)
        return (
            W_COMPILED * result.compiled_fraction
            + W_PASS_RATE * result.salvaged_pass_rate
            + W_COVERAGE * coverage
            + W_MUTATION * mutation
        )

    coverage = (result.branch_coverage_pct or 0.0) / 100.0
    mutation = (result.mutation_score_pct or 0.0) / 100.0 if result.is_green else 0.0

    return (
        W_COMPILED
        + W_PASS_RATE * result.pass_rate
        + W_COVERAGE * coverage
        + W_MUTATION * mutation
    )


# ---------------------------------------------------------------------------------------
# v2. Kept beside v1 rather than replacing it: every RFT round so far ranked on v1, and the
# two must stay comparable.

W2_COVERAGE = 0.2
W2_PASS = 0.4
W2_MUTATION = 0.4


def quality_v2(result: ScoreResult, graded_compile: bool = True) -> float:
    """Scalar quality in [0, 1] under the revised formula.

        does not compile              -> the salvage path, strictly below any compiling suite
        compiles but covers NOTHING   -> 0
        otherwise                     -> 0.2*coverage + 0.4*p + 0.4*p*mutation

    WHAT CHANGED, AND WHY.

    NO FLOOR FOR COMPILING. v1 paid 0.4 for compilation alone, which made a two-test suite
    that compiles beat a twenty-test suite that does not, whatever either covered. That is
    the local optimum rejection-sampling found and went straight to: after one round the
    model's median suite fell from 15 tests to 5, and on a 62-branch class those five tests
    covered 0% of it. The reward asked for that.

    COVERAGE IS A VETO, NOT A CREDIT. A suite that runs and touches nothing is useless by
    definition, so it scores zero rather than a reduced amount.

    THE VETO READS LINE COVERAGE, and the graded term reads branch. Vetoing on branch was
    measured and was too harsh: of 32 compiling rollouts it rejected, 18 had DONE work --
    median 23.6% line coverage, up to 45.8%. Nysiis executes 26.5% of its class while
    taking zero branches, and a branch veto scores that identically to a suite that ran
    nothing. It also pushed 14 of 80 prompts to all-rollouts-zero, which is the exact
    degeneracy the graded reward exists to remove: a group scoring alike gives a
    group-relative objective no gradient.

    Line coverage is the honest test of "did this suite touch the class at all". Branch
    coverage stays where it belongs, in the 0.2 term, where missing branches costs
    proportionally rather than everything. Measured: line-based rejects 14 of 122, and
    those 14 genuinely executed nothing.

    A class with no branches still falls back to line coverage in the TERM, via
    `effective_coverage_pct` -- branch reads 100.0 there by construction.

    ONE VARIABLE FOR THE PASS FRACTION. `p` is `tests_passed / tests_run`. It appears in
    both remaining terms because it does two jobs: a failing test is a defect, and the
    mutation score is measured on the suite with the failing tests REMOVED, so it must be
    discounted by how much of the suite survived. Writing it as two terms with one variable
    keeps that visible; the product form is 0.4*p*(1 + mutation).

    A consequence worth stating, because it is a choice rather than an accident: at equal
    coverage a suite with p=1.0 and mutation 0.5 scores above one with p=0.5 and mutation
    1.0 (0.6 against 0.4 before the coverage term). Fully-passing beats
    better-assertions-on-half-a-suite. Red tests are defects, so this is intended.
    """
    if not result.usable_for_training:
        raise ValueError(
            f"{result.status.value} is an infrastructure outcome and has no quality; "
            "such results must be excluded from datasets, not scored as zero")

    if not result.compiled:
        if not graded_compile:
            return 0.0
        # The salvage path survives, with the SAME veto applied to it: a subset that
        # compiles and covers nothing is worth no more than one that does not compile.
        # It is scaled to stay strictly below the lowest score any compiling suite can
        # earn, so partial credit ranks failures against each other and never against a
        # success.
        if result.n_skeleton_errors or not result.salvaged:
            return 0.0
        # Veto on line coverage; credit on branch. See the docstring.
        line = result.salvaged_line_coverage_pct
        if line is not None and line <= 0.0:
            return 0.0
        cov = result.salvaged_effective_coverage_pct / 100.0
        p = result.salvaged_pass_rate
        inner = W2_COVERAGE * cov + W2_PASS * p
        return round(SALVAGE_CEILING * inner * result.compiled_fraction, 6)

    # THE VETO. `None` means coverage was never measured -- tier 2 or below, or a jacoco
    # timeout -- which is not evidence the suite ran nothing, so it does not veto. v2 is
    # only meaningful from tier 3 up.
    line = result.line_coverage_pct
    if line is not None and line <= 0.0:
        return 0.0

    cov = result.effective_coverage_pct / 100.0
    p = result.pass_rate
    # PIT measured on the subset with failing tests dropped; None when it could not run.
    mutation = (result.mutation_score_pct if result.is_green
                else result.green_subset_mutation_score_pct)
    mutation = (mutation or 0.0) / 100.0

    return round(
        W2_COVERAGE * cov
        + W2_PASS * p
        + W2_MUTATION * p * mutation,
        6)
