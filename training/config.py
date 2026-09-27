"""Configuration for the training side, read from the harness's .env.

Ported from mpref/config.py. Same pattern as utils.config -- `dotenv_values(".env")`, so run
from the repository root -- with the process environment layered on top, and validated
eagerly: a missing WORKING_DIRECTORY should fail at import, not three minutes into a
scoring run.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

_ENV = {**dotenv_values(".env"), **os.environ}


def _get(key: str, default: str | None = None) -> str:
    value = _ENV.get(key) or default
    if value is None:
        raise RuntimeError(f"{key} is not set in .env")
    return str(value)


def _get_int(key: str, default: int) -> int:
    raw = _ENV.get(key)
    return int(raw) if raw else default


def _get_bool(key: str, default: bool) -> bool:
    raw = _ENV.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_working_directory() -> Path:
    """Root of the Maven project under test. Same checks as utils.get_working_directory."""
    project_dir = Path(_get("WORKING_DIRECTORY"))
    if not project_dir.is_absolute():
        project_dir = Path.cwd() / project_dir
    if not project_dir.exists():
        raise RuntimeError(f"WORKING_DIRECTORY does not exist: {project_dir}")
    return project_dir


# The clean project copies the harness runs on. The verifier scores in POOLS of copies of
# them (one Maven build per slot), never in the projects themselves.
PROJECTS_ROOT = Path(_get("PROJECTS_ROOT", str(Path.home() / "harness-projects")))


def get_pool_root() -> Path:
    return Path(_get("POOL_ROOT", str(Path.home() / "harness-pools")))


POOL_SIZE = _get_int("POOL_SIZE", 6)
MVN_OFFLINE = _get_bool("MVN_OFFLINE", True)

# Per-tier, because mutation testing legitimately runs far longer than a compile and a
# single shared timeout would either kill PIT or let a hung compile sit for 20 minutes.
TIMEOUT_COMPILE = _get_int("TIMEOUT_COMPILE", 180)
TIMEOUT_TEST = _get_int("TIMEOUT_TEST", 300)
TIMEOUT_JACOCO = _get_int("TIMEOUT_JACOCO", 300)
TIMEOUT_PITEST = _get_int("TIMEOUT_PITEST", 1200)
