"""Verify -Dtest scoping against the real project, without an LLM call.

A batch leaves each class's generated suite in src/test. This plants one such leftover --
a suite that exercises StringUtils broadly -- then measures a small probe's coverage of
StringUtils both ways:

  unscoped   the leftover runs too, and JaCoCo credits the probe with its coverage
  scoped     only the probe runs; coverage must equal the clean baseline

It also checks the failure mode the scoping flag introduces: a -Dtest pattern that matches
nothing must still build (failIfNoSpecifiedTests=false), and must run zero tests -- which
is the case compiler_node now reports explicitly instead of calling it a pass.

Run from the MAVerIT_distil directory.
"""
from __future__ import annotations

from pathlib import Path

from paths import PROJECTS  # noqa: E402
from utils import (calculate_branch_coverage_for_class,
                   calculate_line_coverage_for_class, find_jacoco_xml_reports,
                   find_surefire_reports, get_maven_module_directory, parse_surefire,
                   run_maven)

PROJECT = PROJECTS / "commons-codec-clean"
CUT = PROJECT / "src/main/java/org/apache/commons/codec/binary/StringUtils.java"
PKG_DIR = PROJECT / "src/test/java/org/apache/commons/codec/binary"
PROBE = PKG_DIR / "HarnessProbeTest.java"
LEFTOVER = PKG_DIR / "LeftoverFromEarlierClassTest.java"
TARGET = "org.apache.commons.codec.binary.StringUtils"
PROBE_FQN = "org.apache.commons.codec.binary.HarnessProbeTest"

PROBE_SRC = """package org.apache.commons.codec.binary;

import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class HarnessProbeTest {
    @Test
    public void passesOne() {
        assertEquals("abc", StringUtils.newStringUtf8("abc".getBytes()));
    }

    @Test
    public void passesTwo() {
        assertEquals(null, StringUtils.newStringUtf8(null));
    }
}
"""

# Stands in for a suite some earlier class in the batch left behind.
LEFTOVER_SRC = """package org.apache.commons.codec.binary;

import static org.junit.Assert.assertNotNull;
import org.junit.Test;

public class LeftoverFromEarlierClassTest {
    @Test
    public void touchesMuchOfStringUtils() {
        assertNotNull(StringUtils.getBytesUtf8("x"));
        assertNotNull(StringUtils.getBytesIso8859_1("x"));
        assertNotNull(StringUtils.getBytesUsAscii("x"));
        assertNotNull(StringUtils.getBytesUtf16("x"));
        assertNotNull(StringUtils.getBytesUtf16Be("x"));
        assertNotNull(StringUtils.getBytesUtf16Le("x"));
        assertNotNull(StringUtils.newStringIso8859_1("x".getBytes()));
        assertNotNull(StringUtils.newStringUsAscii("x".getBytes()));
        assertNotNull(StringUtils.newStringUtf16("x".getBytes()));
        assertNotNull(StringUtils.newStringUtf16Be("x".getBytes()));
        assertNotNull(StringUtils.newStringUtf16Le("x".getBytes()));
    }
}
"""


def measure(label: str, test_fqn: str | None) -> dict:
    result = run_maven(str(PROJECT), test_fqn)
    module = get_maven_module_directory(str(CUT), PROJECT)
    run, passed, failed = parse_surefire(find_surefire_reports(module))   # ALL reports
    reports = find_jacoco_xml_reports(module)
    line = calculate_line_coverage_for_class(reports, TARGET) if result["ok"] else None
    branch = calculate_branch_coverage_for_class(reports, TARGET) if result["ok"] else None
    print(f"  {label:<34} ok={result['ok']!s:<5} tests_run={run:<3} "
          f"line={line!s:<6} branch={branch}")
    return {"ok": result["ok"], "run": run, "line": line}


def main() -> int:
    for path in (PROBE, LEFTOVER):
        if path.exists():
            print(f"refusing to overwrite an existing file: {path}")
            return 1
    try:
        PROBE.write_text(PROBE_SRC, encoding="utf-8")
        print("probe alone:")
        baseline = measure("scoped, no leftover (baseline)", PROBE_FQN)

        LEFTOVER.write_text(LEFTOVER_SRC, encoding="utf-8")
        print("probe + a leftover suite from an 'earlier class':")
        unscoped = measure("UNSCOPED (the old behaviour)", None)
        scoped = measure("scoped to the probe", PROBE_FQN)
        nomatch = measure("scoped to a class that isn't there", "org.example.NoSuchTest")
    finally:
        for path in (PROBE, LEFTOVER):
            if path.exists():
                path.unlink()

    print("\n=== CLAIMS ===")
    checks = [
        ("the leftover inflates UNSCOPED coverage",
         unscoped["line"] is not None and unscoped["line"] > baseline["line"]),
        ("unscoped ran the leftover's test too", unscoped["run"] == 3),
        ("SCOPED coverage equals the clean baseline", scoped["line"] == baseline["line"]),
        ("scoped ran only the probe's 2 tests", scoped["run"] == 2),
        ("a non-matching scope still builds", nomatch["ok"]),
        ("...and runs zero tests (the case now reported)", nomatch["run"] == 0),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    remaining = list((PROJECT / "src/test").rglob("*.java"))
    print(f"\n  test files left in the project: {len(remaining)}")
    return 0 if ok and not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
