"""Verify the three changes against the real project, without an LLM call.

Three claims to check, each of which would otherwise only surface mid-run:

  1. A suite that COMPILES BUT FAILS no longer reports a compile failure, and coverage is
     still measured for it. Before -Dmaven.test.failure.ignore=true, Maven aborted at
     `test` and jacoco:report never ran.
  2. A suite that does not COMPILE still reports ok=False.
  3. parse_surefire / parse_surefire_failures read the real reports and name the right
     failing methods.

Run from the MAVerIT_distil directory so `.env` resolves.
"""
from __future__ import annotations

# The installed langchain_core 0.3.41 exports UsageMetadata from .messages.ai, while
# utils.py imports it from .messages. Shimmed HERE rather than edited in utils.py, because
# which import is right depends on the interpreter the harness actually runs under -- and
# that is not this one.
import langchain_core.messages as _lcm
if not hasattr(_lcm, "UsageMetadata"):
    from langchain_core.messages.ai import UsageMetadata as _UM
    _lcm.UsageMetadata = _UM

import shutil
import sys
from pathlib import Path

from utils import (calculate_branch_coverage_for_class,
                   calculate_line_coverage_for_class, find_jacoco_xml_reports,
                   find_surefire_reports, get_maven_module_directory,
                   parse_surefire, parse_surefire_failures, run_maven)

PROJECT = Path(r"C:\Users\akosc\IdeaProjects\commons-codec-clean")
CUT = PROJECT / "src/main/java/org/apache/commons/codec/binary/StringUtils.java"
TEST_DIR = PROJECT / "src/test/java/org/apache/commons/codec/binary"
TEST_FILE = TEST_DIR / "HarnessProbeTest.java"

GREEN = """package org.apache.commons.codec.binary;

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

RED = """package org.apache.commons.codec.binary;

import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class HarnessProbeTest {
    @Test
    public void passesOne() {
        assertEquals("abc", StringUtils.newStringUtf8("abc".getBytes()));
    }

    @Test
    public void failsDeliberately() {
        assertEquals("this will not match", StringUtils.newStringUtf8("abc".getBytes()));
    }
}
"""

BROKEN = """package org.apache.commons.codec.binary;

import org.junit.Test;

public class HarnessProbeTest {
    @Test
    public void doesNotCompile() {
        StringUtils.thisMethodDoesNotExist("abc");
    }
}
"""


def attempt(label: str, source: str) -> dict:
    TEST_FILE.write_text(source, encoding="utf-8")
    result = run_maven(str(PROJECT))
    module = get_maven_module_directory(str(CUT), PROJECT)
    # exercise the scoping the node now uses, rather than filtering by hand
    surefire = find_surefire_reports(
        module, "org.apache.commons.codec.binary.HarnessProbeTest")
    run = passed = failed = 0
    failing: set[str] = set()
    if result["ok"]:
        run, passed, failed = parse_surefire(surefire)
        failing = parse_surefire_failures(surefire)
    cov = line = None
    if result.get("jacoco_ok"):
        reports = find_jacoco_xml_reports(module)
        target = "org.apache.commons.codec.binary.StringUtils"
        cov = calculate_branch_coverage_for_class(reports, target)
        line = calculate_line_coverage_for_class(reports, target)
    print(f"\n--- {label} ---")
    print(f"  ok (compiled)   {result['ok']}")
    print(f"  jacoco_ok       {result.get('jacoco_ok')}")
    print(f"  tests           run={run} passed={passed} failed={failed}")
    print(f"  failing names   {sorted(failing) or '-'}")
    print(f"  branch cov      {cov}")
    print(f"  line cov        {line}")
    return {"ok": result["ok"], "run": run, "failed": failed,
            "failing": failing, "line": line}


def main() -> int:
    if not CUT.exists():
        print(f"class under test missing: {CUT}")
        return 1
    backup = None
    if TEST_FILE.exists():
        backup = TEST_FILE.read_text(encoding="utf-8")
    try:
        g = attempt("GREEN suite", GREEN)
        r = attempt("RED suite (compiles, one assertion fails)", RED)
        b = attempt("BROKEN suite (does not compile)", BROKEN)
    finally:
        if backup is not None:
            TEST_FILE.write_text(backup, encoding="utf-8")
        elif TEST_FILE.exists():
            TEST_FILE.unlink()

    print("\n=== CLAIMS ===")
    checks = [
        ("green compiles and is green", g["ok"] and g["failed"] == 0 and g["run"] == 2),
        ("RED still reports ok=True (the flag works)", r["ok"]),
        ("RED is detected as red via surefire", r["failed"] == 1),
        ("RED names the failing method", "failsDeliberately" in r["failing"]),
        ("RED still yields coverage", r["line"] is not None and r["line"] > 0),
        ("BROKEN reports ok=False", not b["ok"]),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
