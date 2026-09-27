"""JaCoCo and PIT XML report discovery and parsing. Ported from MAVerIT utils.py.

UNITS: every function here returns a PERCENTAGE in [0, 100], not a fraction. That is
MAVerIT's convention and the numbers in report.csv (89.44 branch / 89.63 mutation) are on
that scale, so keeping it means the acceptance test A1 compares like with like. The unit is
in the field names downstream (`branch_coverage_pct`) so it cannot be silently confused.

Also ported: surefire result parsing, which MAVerIT does not need because its graph only
asks whether the build succeeded.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


# --- discovery ----------------------------------------------------------------------

def find_jacoco_xml_reports(project_dir: str | Path) -> list[Path]:
    """Ported from MAVerIT utils.py:324."""
    return sorted(Path(project_dir).rglob("target/site/jacoco/jacoco.xml"))


def find_pitest_xml_reports(project_dir: str | Path) -> list[Path]:
    """Ported from MAVerIT utils.py:328.

    PIT writes a timestamped directory per run, so old runs accumulate. Only reports
    sharing the newest mtime are returned -- without this, a stale report from a previous
    scoring of a different class would be silently mixed in.
    """
    reports = sorted(Path(project_dir).rglob("target/pit-reports/**/mutations.xml"))
    if not reports:
        return []

    latest_timestamp = max(report.stat().st_mtime for report in reports)
    return [report for report in reports if report.stat().st_mtime == latest_timestamp]


def find_surefire_reports(project_dir: str | Path) -> list[Path]:
    return sorted(Path(project_dir).rglob("target/surefire-reports/TEST-*.xml"))


# --- parsing ------------------------------------------------------------------------

def _find_jacoco_class(root: ET.Element, jacoco_class_name: str) -> ET.Element | None:
    """Ported from MAVerIT utils.py:457."""
    for class_element in root.iter("class"):
        if class_element.attrib.get("name") == jacoco_class_name:
            return class_element
    return None


def calculate_branch_coverage(jacoco_report_paths: list[Path]) -> float:
    """Project-wide branch coverage. Ported from MAVerIT utils.py:337.

    This is the quantity report.csv publishes as `overall_branch_coverage` -- every
    counter in the report pooled, with ALL test classes present at once. It is NOT the
    mean of per-class percentages, and the two differ substantially: pooling weights a
    class by how many branches it has, while a mean weights every class equally and lets a
    branchless class contribute a free 100%.

    Used for validating against the teacher's published numbers, and for comparison tables
    that must line up with them. The per-class variant below is what scores a single
    generated class as a training reward.
    """
    covered = 0
    missed = 0

    for report_path in jacoco_report_paths:
        root = ET.parse(report_path).getroot()
        for counter in root.findall("counter"):
            if counter.attrib.get("type") == "BRANCH":
                covered += int(counter.attrib.get("covered", "0"))
                missed += int(counter.attrib.get("missed", "0"))

    total = covered + missed
    if total == 0:
        return 100.0

    return round((covered / total) * 100, 2)


def calculate_branch_coverage_for_class(
    jacoco_report_paths: list[Path],
    target_class_name: str,
) -> float:
    """Ported from MAVerIT utils.py:354. Returns a percentage in [0, 100].

    Scoped to the class under test, not the whole module: a suite that happens to exercise
    unrelated classes should get no credit for it.

    Quirk preserved from MAVerIT: a class with no branches at all returns 100.0, since
    zero of zero branches are missed. Defensible, but it means coverage is a weak signal
    for trivial classes -- mutation score carries the weight there.
    """
    covered = 0
    missed = 0
    jacoco_class_name = target_class_name.replace(".", "/")

    for report_path in jacoco_report_paths:
        root = ET.parse(report_path).getroot()
        class_element = _find_jacoco_class(root, jacoco_class_name)
        if class_element is None:
            continue

        for counter in class_element.findall("counter"):
            if counter.attrib.get("type") == "BRANCH":
                covered += int(counter.attrib.get("covered", "0"))
                missed += int(counter.attrib.get("missed", "0"))

    total = covered + missed
    if total == 0:
        return 100.0

    return round((covered / total) * 100, 2)


def calculate_mutation_score(pitest_report_paths: list[Path]) -> float:
    """Ported from MAVerIT utils.py:405. Returns a percentage in [0, 100].

    NON_VIABLE / MEMORY_ERROR / RUN_ERROR mutants are excluded from the denominator --
    those failed for reasons unrelated to test quality, and counting them would penalise
    a suite for the mutation engine's problems.

    The denominator is otherwise fixed by the production class, which is what stops a
    trivial suite from scoring well: it can only kill the mutants that exist, and there
    are as many of them as the class under test has behaviour.
    """
    killed = 0
    total = 0
    excluded_statuses = {"NON_VIABLE", "MEMORY_ERROR", "RUN_ERROR"}

    for report_path in pitest_report_paths:
        root = ET.parse(report_path).getroot()
        for mutation in root.findall("mutation"):
            status = mutation.attrib.get("status", "").upper()
            if status in excluded_statuses:
                continue
            if status == "KILLED":
                killed += 1
            total += 1

    if total == 0:
        return 0.0

    return round((killed / total) * 100, 2)


@dataclass(frozen=True)
class Coverage:
    """Branch AND line coverage for one class, with the counts behind them.

    The counts matter. `calculate_branch_coverage_for_class` returns 100.0 both for a fully
    covered class and for one with NO BRANCHES AT ALL -- 43% of the compiling rollouts we
    have measured are the second case -- and the percentage alone cannot tell them apart.
    `n_branches` can, which is what makes a line-coverage fallback possible.
    """

    branch_pct: float
    line_pct: float
    n_branches: int
    n_lines: int

    @property
    def effective_pct(self) -> float:
        """Branch coverage, or line coverage when the class has no branches to cover.

        A branchless class is not 100% tested because it has no branches; it is untested
        until something executes its lines. Measured: branch and line agree closely where
        both are defined (r = 0.992 over MAVerIT's 26 runs), and branch correlates very
        slightly better with mutation (0.994 vs 0.989), so branch stays primary.
        """
        return self.branch_pct if self.n_branches else self.line_pct


def calculate_coverage_for_class(
    jacoco_report_paths: list[Path],
    target_class_name: str,
) -> Coverage:
    """Both counters for the class under test, in one parse."""
    counts = {"BRANCH": [0, 0], "LINE": [0, 0]}
    jacoco_class_name = target_class_name.replace(".", "/")

    for report_path in jacoco_report_paths:
        root = ET.parse(report_path).getroot()
        class_element = _find_jacoco_class(root, jacoco_class_name)
        if class_element is None:
            continue
        for counter in class_element.findall("counter"):
            kind = counter.attrib.get("type")
            if kind in counts:
                counts[kind][0] += int(counter.attrib.get("covered", "0"))
                counts[kind][1] += int(counter.attrib.get("missed", "0"))

    def pct(covered, missed):
        total = covered + missed
        return 100.0 if total == 0 else round(100.0 * covered / total, 2)

    bc, bm = counts["BRANCH"]
    lc, lm = counts["LINE"]
    return Coverage(branch_pct=pct(bc, bm), line_pct=pct(lc, lm),
                    n_branches=bc + bm, n_lines=lc + lm)


def parse_surefire_failures(surefire_report_paths: list[Path]) -> set[str]:
    """Method names of the tests that FAILED or ERRORED.

    `parse_surefire` returns counts, which is enough to score a suite and not enough to
    repair one. Dropping the failing tests turns a red suite into a green one, and PIT --
    which refuses a red suite because it cannot tell an introduced fault from a
    pre-existing failure -- can then measure what the passing tests actually assert.
    """
    failing: set[str] = set()
    for report_path in surefire_report_paths:
        try:
            root = ET.parse(report_path).getroot()
        except ET.ParseError:
            # Same reasoning as parse_surefire: a truncated report is evidence the JVM
            # died, not a bug here. Unknown failures for one class, not for the rest.
            continue
        for case in root.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                name = case.attrib.get("name")
                if name:
                    failing.add(name.split("[")[0])   # strip a parameterised suffix
    return failing


def parse_surefire(surefire_report_paths: list[Path]) -> tuple[int, int, int]:
    """(tests_run, tests_passed, tests_failed) across the given reports.

    New here -- MAVerIT only asks whether the build succeeded. We need the counts because
    `-Dmaven.test.failure.ignore=true` makes a red suite exit 0, and because the pass rate
    is a term in the quality score.

    Errors and failures are both counted as failures; skipped tests are removed from the
    run total, since a skipped test neither passed nor failed.
    """
    run = failed = skipped = 0

    for report_path in surefire_report_paths:
        try:
            root = ET.parse(report_path).getroot()
        except ET.ParseError as error:
            # A TRUNCATED report is not a parse bug, it is evidence: surefire streams XML
            # as tests run, so a report that stops mid-tag means the JVM died or Maven was
            # killed by TIMEOUT_TEST partway through. Observed on 2026-09-08 with a file
            # ending at a bare "<error", which took down a multi-hour dataset build.
            #
            # Skip the file rather than raising. One unreadable class means that class's
            # counts are unknown; it does not mean the other classes' counts are, and it
            # certainly does not justify losing the whole run.
            print(f"[reports] skipping unreadable surefire report "
                  f"{report_path.name}: {error}", flush=True)
            continue
        run += int(root.attrib.get("tests", "0"))
        failed += int(root.attrib.get("failures", "0"))
        failed += int(root.attrib.get("errors", "0"))
        skipped += int(root.attrib.get("skipped", "0"))

    run -= skipped
    passed = max(run - failed, 0)
    return run, passed, failed
