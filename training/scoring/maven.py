"""Maven invocation. Ported from MAVerIT utils.py, with three deliberate deviations.

1. A COMPILE-ONLY entry point. MAVerIT's `run_maven` runs `mvn clean test`; as a cheap
   gate we need `mvn test-compile`, which is seconds rather than minutes and rejects most
   bad candidates before anything expensive runs.

2. `-Dmaven.test.failure.ignore=true` on the test and coverage tiers. Without it a single
   failing test aborts the build and we lose the JaCoCo report too -- but coverage is
   perfectly measurable on a red suite, since JaCoCo instruments whatever executed. Only
   PIT genuinely requires green.

3. PER-TIER TIMEOUTS. MAVerIT used one 120 s timeout for `clean test` and none at all for
   PIT. Mutation testing on a large class legitimately exceeds 120 s.

`-Dthreads=12` on PIT is preserved from MAVerIT -- it is what makes tier 4 affordable
enough to run on every pair rather than a subsample.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from training import config


class MavenNotFoundError(RuntimeError):
    """Maven is absent from PATH. An infrastructure failure, never a bad test class."""


def _mvn() -> str:
    """Ported from MAVerIT utils.py:216."""
    mvn_path = shutil.which("mvn")
    if mvn_path is None:
        raise MavenNotFoundError("Maven executable 'mvn' was not found on PATH.")
    return mvn_path


def _run(args: list[str], project_dir: str | Path, timeout: float) -> dict:
    """Shared shape: MAVerIT's {"ok", "combined_result"} plus a timeout flag.

    `timed_out` is separate from `ok` on purpose. A timeout is not evidence that the test
    class is bad -- it may be an infrastructure problem or a pathological project -- and
    scoring it as a quality-zero sample would poison the preference data.
    """
    command = [_mvn(), *args]
    if config.MVN_OFFLINE:
        command.insert(1, "-o")

    try:
        result = subprocess.run(
            command,
            cwd=str(project_dir),
            text=True,
            capture_output=True,
            shell=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "ok": False,
            "timed_out": True,
            "combined_result": (
                f"Maven timed out after {timeout} seconds.\n"
                f"{error.stdout or ''}\n{error.stderr or ''}"
            ),
        }

    return {
        "ok": result.returncode == 0,
        "timed_out": False,
        "combined_result": result.stdout + "\n" + result.stderr,
    }


def run_test_compile(project_dir: str | Path) -> dict:
    """Tier 1. Does it compile? Seconds, and rejects most bad candidates."""
    return _run(["-q", "test-compile"], project_dir, config.TIMEOUT_COMPILE)


def run_tests(project_dir: str | Path) -> dict:
    """Tier 2. Do the tests pass?

    `failure.ignore` means `ok` reflects whether Maven RAN, not whether tests passed --
    pass/fail counts come from the surefire reports. That separation is what lets a red
    suite still reach tier 3.
    """
    return _run(
        ["-q", "test", "-Dmaven.test.failure.ignore=true"],
        project_dir,
        config.TIMEOUT_TEST,
    )


def run_jacoco(project_dir: str | Path) -> dict:
    """Tier 3. Ported from MAVerIT utils.py:288, minus `clean`.

    MAVerIT runs `clean test jacoco:report`, which re-executes the whole suite that
    run_tests just ran. Acceptable once per harness iteration; wasteful for a reward
    function called thousands of times.
    """
    return _run(
        ["-q", "test", "jacoco:report", "-Dmaven.test.failure.ignore=true"],
        project_dir,
        config.TIMEOUT_JACOCO,
    )


def run_pitest(project_dir: str | Path, target_classes: str, target_tests: str) -> dict:
    """Tier 4. Ported from MAVerIT utils.py:302.

    Argument list preserved exactly, including `-Dthreads=12` and the XML output format
    the report parser depends on. No `clean`: tier 3 already built and ran everything.
    """
    return _run(
        [
            "test-compile",
            "org.pitest:pitest-maven:mutationCoverage",
            f"-DtargetClasses={target_classes}",
            f"-DtargetTests={target_tests}",
            "-Dthreads=12",
            "-DoutputFormats=XML,HTML",
        ],
        project_dir,
        config.TIMEOUT_PITEST,
    )


def get_maven_module_directory(
    source_file_path: str | Path,
    project_dir: str | Path | None = None,
) -> Path:
    """Ported from MAVerIT utils.py:237.

    Walks up from the source file to the nearest pom.xml, so multi-module projects run
    Maven in the right module rather than at the reactor root.
    """
    project_path = Path(project_dir) if project_dir is not None else config.get_working_directory()
    project_path = project_path.resolve()
    current_path = Path(source_file_path).resolve()
    if current_path.is_file():
        current_path = current_path.parent

    try:
        current_path.relative_to(project_path)
    except ValueError as error:
        raise RuntimeError(
            f"Source file is not inside the project directory: {source_file_path}") from error

    while current_path != project_path.parent:
        if (current_path / "pom.xml").is_file():
            return current_path
        if current_path == project_path:
            break
        current_path = current_path.parent

    raise RuntimeError(f"No Maven module pom.xml found for source file: {source_file_path}")
