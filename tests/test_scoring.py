"""Tier-0 and parsing tests. No Maven, no GPU, no network -- these run anywhere.

The Maven-dependent acceptance criteria (A1 in particular, reproducing the teacher's
published 89.44 / 89.63) live in test_acceptance.py behind the `maven` marker, because
they need a real project checkout and take minutes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from training.scoring import java, reports
from training.scoring.result import ScoreResult, Status

FIXTURES = Path(__file__).parent / "fixtures"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# --- fence stripping ----------------------------------------------------------------

def test_strips_java_fence():
    """Trailing newline is preserved: the chunk between fences ends in one, and dropping
    the `java` tag line leaves it. Asserted exactly so a future 'tidy-up' that strips it
    shows up as a deliberate divergence from MAVerIT rather than a silent one."""
    fenced = "```java\npackage a.b;\nclass CTest {}\n```"
    assert java.strip_markdown_code_fence(fenced) == "package a.b;\nclass CTest {}\n"


def test_unpaired_fence_returned_untouched():
    """The harness gives up on an odd fence count; scoring must agree, or we would
    score something the harness would reject."""
    text = "```java\nclass A {}"
    assert java.strip_markdown_code_fence(text) == text


def test_unfenced_passes_through():
    text = "package a;\nclass BTest {}"
    assert java.strip_markdown_code_fence(text) == text


# --- name extraction ----------------------------------------------------------------

def test_fully_qualified_name():
    src = read("GoodTest.java")
    assert java.get_java_fully_qualified_name(src) == "org.example.GoodTest"


def test_fqn_without_package():
    assert java.get_java_fully_qualified_name("public class Solo {}") == "Solo"


def test_missing_class_declaration_raises():
    with pytest.raises(RuntimeError):
        java.get_java_entity_name("int x = 1;")


# --- tier 0 counts ------------------------------------------------------------------

def test_counts_tests():
    assert java.count_tests(read("GoodTest.java")) == 3


def test_counts_assertless_tests():
    """GoodTest has one test method with no assertion and no expected= annotation."""
    assert java.count_test_without_assert(read("GoodTest.java")) == 1


def test_assertless_counter_is_comment_blind():
    """Known limitation of the ported counter, recorded rather than fixed.

    count_test_without_assert splits on the bare string "@Test", so the annotation name
    appearing in a comment inflates the count. count_tests does not share the flaw -- its
    regex anchors to the start of a line.

    Left as-is deliberately: the port must agree with MAVerIT exactly, because the paper
    compares numbers produced by both. Fixing it here would make the two disagree on real
    test classes that mention the annotation in Javadoc.
    """
    src = "// mentions @Test in a comment\nclass T {\n    @Test\n    void f() { }\n}"
    assert java.count_test_without_assert(src) == 2   # the comment counts -- the flaw
    assert java.count_tests(src) == 1                 # unaffected


def test_balance_detects_unbalanced_parens():
    """Braces alone are not enough -- this is the check that was missed once."""
    assert java.is_balanced("class A { void f() { g(1; } }") is False
    assert java.is_balanced(read("GoodTest.java")) is True


def test_unterminated_class_is_unbalanced():
    assert java.is_balanced(read("Runaway.java")) is False


def test_looks_terminated():
    assert java.looks_terminated("class A {}\n") is True
    assert java.looks_terminated(read("Runaway.java")) is False


# --- status semantics ---------------------------------------------------------------

def test_infra_error_is_not_the_models_fault():
    """The single most important invariant: infrastructure failures must never enter a
    preference dataset as quality-zero samples."""
    assert Status.INFRA_ERROR.is_model_fault is False
    assert Status.TIMEOUT.is_model_fault is False
    assert Status.COMPILE_FAIL.is_model_fault is True
    assert Status.OK.is_model_fault is True


def test_infra_error_excluded_from_training():
    result = ScoreResult(class_under_test="a.B", status=Status.INFRA_ERROR, tier_reached=0)
    assert result.usable_for_training is False


def test_red_suite_is_not_green():
    red = ScoreResult(class_under_test="a.B", status=Status.TEST_FAIL, tier_reached=2,
                      compiled=True, tests_run=10, tests_passed=9, tests_failed=1)
    assert red.is_green is False
    assert red.pass_rate == pytest.approx(0.9)


def test_mutation_none_means_not_measured():
    """None is 'PIT could not run', which is a different fact from 'scored zero'."""
    red = ScoreResult(class_under_test="a.B", status=Status.TEST_FAIL, tier_reached=3,
                      compiled=True, tests_run=2, tests_passed=1, tests_failed=1,
                      branch_coverage_pct=42.0)
    assert red.mutation_score_pct is None
    assert red.branch_coverage_pct == 42.0


# --- report parsing -----------------------------------------------------------------

def test_mutation_score_excludes_non_viable(tmp_path: Path):
    """NON_VIABLE / MEMORY_ERROR / RUN_ERROR leave the denominator, so the engine's own
    problems do not count against the test suite."""
    xml = """<mutations>
      <mutation status='KILLED'/>
      <mutation status='SURVIVED'/>
      <mutation status='NON_VIABLE'/>
      <mutation status='MEMORY_ERROR'/>
    </mutations>"""
    path = tmp_path / "mutations.xml"
    path.write_text(xml, encoding="utf-8")
    assert reports.calculate_mutation_score([path]) == 50.0


def test_mutation_score_with_no_mutants_is_zero(tmp_path: Path):
    path = tmp_path / "mutations.xml"
    path.write_text("<mutations></mutations>", encoding="utf-8")
    assert reports.calculate_mutation_score([path]) == 0.0


def test_branch_coverage_scoped_to_class(tmp_path: Path):
    """Coverage of unrelated classes must not count -- a suite gets credit only for the
    class it was asked to test."""
    xml = """<report>
      <package name='org/example'>
        <class name='org/example/Target'>
          <counter type='BRANCH' missed='2' covered='8'/>
        </class>
        <class name='org/example/Other'>
          <counter type='BRANCH' missed='90' covered='0'/>
        </class>
      </package>
    </report>"""
    path = tmp_path / "jacoco.xml"
    path.write_text(xml, encoding="utf-8")
    assert reports.calculate_branch_coverage_for_class([path], "org.example.Target") == 80.0


def test_branch_coverage_of_branchless_class_is_100(tmp_path: Path):
    """Quirk inherited from MAVerIT, asserted so it stays a decision rather than a
    surprise: zero of zero branches missed reads as 100%."""
    xml = """<report><package name='p'><class name='p/T'/></package></report>"""
    path = tmp_path / "jacoco.xml"
    path.write_text(xml, encoding="utf-8")
    assert reports.calculate_branch_coverage_for_class([path], "p.T") == 100.0


def test_surefire_counts_errors_as_failures_and_drops_skipped(tmp_path: Path):
    xml = "<testsuite tests='10' failures='1' errors='2' skipped='3'/>"
    path = tmp_path / "TEST-a.xml"
    path.write_text(xml, encoding="utf-8")
    run, passed, failed = reports.parse_surefire([path])
    assert (run, passed, failed) == (7, 4, 3)


def test_pitest_report_discovery_keeps_only_latest(tmp_path: Path):
    """PIT writes a timestamped directory per run; a stale report from scoring a
    different class must not be mixed in."""
    old = tmp_path / "target" / "pit-reports" / "old" / "mutations.xml"
    new = tmp_path / "target" / "pit-reports" / "new" / "mutations.xml"
    for path in (old, new):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<mutations/>", encoding="utf-8")
    import os
    os.utime(old, (1_000_000, 1_000_000))

    found = reports.find_pitest_xml_reports(tmp_path)
    assert found == [new]
