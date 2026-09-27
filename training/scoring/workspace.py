"""A pool of pre-warmed copies of the project under test.

Why not write into the real checkout: `mvn clean` deletes target/, and two scorers running
at once would corrupt each other's reports. Why a pool rather than a fresh copy per call
(spec O1): dependency resolution does not always succeed, and a persistent slot can be
repaired by hand, whereas a disposable one re-hits the same failure every time.

Sizing is UNMEASURED. The first thing worth timing is a cold versus warm
`mvn test-compile`: if the gap is large, more slots pay for themselves immediately; if it
is small, they only cost disk.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from training import config
from training.scoring.maven import MavenNotFoundError, _mvn


class WorkspaceError(RuntimeError):
    """Pool problem -- always an INFRA_ERROR, never a bad test class."""


class WorkspacePool:
    """Hands out exclusive project copies.

    Not thread-safe to *create*, but `acquire()` is safe to call from several threads once
    `prepare()` has returned.
    """

    def __init__(self, source: Path | None = None, root: Path | None = None, size: int | None = None):
        self.source = (source or config.get_working_directory()).resolve()
        self.root = (root or config.get_pool_root()).resolve()
        self.size = size or config.POOL_SIZE
        self._free: queue.Queue[Path] = queue.Queue()

    def slot_paths(self) -> list[Path]:
        return [self.root / f"slot-{i}" for i in range(self.size)]

    def prepare(self, force: bool = False, warm: bool = True) -> None:
        """Create the copies and, if `warm`, populate each one's dependency cache.

        Warming matters more than it looks: it resolves dependencies once per slot while
        nothing else is running, so that scoring can then run offline (`mvn -o`) without a
        first-ever download racing across slots.
        """
        self.root.mkdir(parents=True, exist_ok=True)

        for slot in self.slot_paths():
            if slot.exists() and not force:
                continue
            if slot.exists():
                shutil.rmtree(slot)
            # copy2 preserves mtimes, which keeps Maven's incremental build honest.
            shutil.copytree(self.source, slot, symlinks=True, copy_function=shutil.copy2)

        # A scorer that dies between install_test_class and remove_test_class -- a crash, a
        # killed process -- leaves its generated class in the slot, and every later scoring
        # there compiles it too. The projects carry no tests of their own, so any .java under
        # src/test/java is such a leftover. Measured: one machine crash left six.
        for slot in self.slot_paths():
            for stray in (slot / "src" / "test" / "java").rglob("*.java"):
                stray.unlink()

        if warm:
            for slot in self.slot_paths():
                self._warm(slot)

        self._free = queue.Queue()
        for slot in self.slot_paths():
            self._free.put(slot)

    def _warm(self, slot: Path) -> None:
        """Resolve dependencies and build once, online, so later runs can go offline."""
        try:
            mvn = _mvn()
        except MavenNotFoundError as error:
            raise WorkspaceError(str(error)) from error

        started = time.monotonic()
        result = subprocess.run(
            [mvn, "-q", "test-compile"],
            cwd=str(slot), text=True, capture_output=True, timeout=config.TIMEOUT_COMPILE * 4,
        )
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise WorkspaceError(
                f"warming {slot} failed after {elapsed:.0f}s -- fix this slot by hand "
                f"(that is why we keep a pool):\n{result.stdout}\n{result.stderr}"
            )

    def attach(self) -> None:
        """Populate the free queue from slots that already exist on disk."""
        existing = [s for s in self.slot_paths() if s.exists()]
        if not existing:
            raise WorkspaceError(
                f"no prepared slots under {self.root}; call prepare() first")
        self._free = queue.Queue()
        for slot in existing:
            self._free.put(slot)

    @contextmanager
    def acquire(self, timeout: float = 3600.0):
        """Yield an exclusive slot, returning it to the pool afterwards."""
        try:
            slot = self._free.get(timeout=timeout)
        except queue.Empty as error:
            raise WorkspaceError(f"no free workspace within {timeout}s") from error
        try:
            yield slot
        finally:
            self._free.put(slot)


def install_test_class(slot: Path, source: str, package: str, class_name: str) -> Path:
    """Write a generated test class into a slot's test source tree.

    Returns the path written. The caller is responsible for removing it -- see
    `remove_test_class` -- so that consecutive scorings in the same slot do not
    accumulate test classes and inflate each other's coverage.
    """
    test_root = slot / "src" / "test" / "java"
    if not test_root.exists():
        raise WorkspaceError(f"{slot} has no src/test/java")

    package_dir = test_root / Path(*package.split(".")) if package else test_root
    package_dir.mkdir(parents=True, exist_ok=True)

    path = package_dir / f"{class_name}.java"
    path.write_text(source, encoding="utf-8")
    return path


def remove_test_class(path: Path) -> None:
    path.unlink(missing_ok=True)


# Report artefacts that survive between runs in a reused slot. We deliberately do not run
# `mvn clean` -- it would re-download and rebuild everything -- so these must be removed by
# hand before each scoring.
_STALE_REPORTS = (
    "target/surefire-reports",   # accumulates one TEST-*.xml per class ever run here
    "target/jacoco.exec",        # JaCoCo APPENDS by default: coverage from prior classes
    "target/site/jacoco",
    "target/pit-reports",
)


def clear_reports(module_dir: Path) -> None:
    """Delete build reports left by a previous scoring in this slot.

    Without this, a slot that has already scored five classes reports their tests and
    their coverage as though they belonged to the sixth. It inflates every number, and it
    inflates them upwards, which is the direction that looks like success.
    """
    for relative in _STALE_REPORTS:
        target = module_dir / relative
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink(missing_ok=True)
