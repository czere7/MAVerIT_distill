"""One warmed workspace pool per project, shared by scoring and target building.

A pool is a set of copies of ~/harness-projects/<project> under POOL_ROOT/<project>, one
Maven build per slot, so several rollouts can be measured at once without their reports
colliding. `prepare()` skips slots that already exist, so asking twice is cheap.
"""

from __future__ import annotations

import threading
from pathlib import Path

from training import config
from training.scoring import WorkspacePool

_POOLS: dict[tuple[str, int], WorkspacePool] = {}
_LOCK = threading.Lock()


def project_dir(project: str) -> Path:
    return config.PROJECTS_ROOT / project


def class_path(project: str, class_key: str) -> Path:
    """The production class a rollout tests. `class_key` is its path under src/main/java,
    without `.java` -- e.g. org/apache/commons/codec/net/URLCodec."""
    return project_dir(project) / "src" / "main" / "java" / f"{class_key}.java"


def pool_for(project: str, size: int | None = None) -> WorkspacePool:
    size = size or config.POOL_SIZE
    with _LOCK:
        if (project, size) not in _POOLS:
            pool = WorkspacePool(source=project_dir(project),
                                 root=config.get_pool_root() / project, size=size)
            pool.prepare()
            _POOLS[(project, size)] = pool
        return _POOLS[(project, size)]
