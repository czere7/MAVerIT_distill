"""Objective measurement of a generated JUnit 4 test class. Ported from mpref.scoring.

No repair, no loops, no LLM calls -- this measures what the model produced, which is what
makes it usable as a training reward rather than a description of what the harness could
fix the output into.
"""

from training.scoring.pertest import (Attribution, TestSpan, attribute, drop_tests,
                                      test_spans)
from training.scoring.result import ScoreResult, Status
from training.scoring.score import score
from training.scoring.workspace import WorkspaceError, WorkspacePool

__all__ = ["ScoreResult", "Status", "score", "WorkspacePool", "WorkspaceError",
           "Attribution", "TestSpan", "attribute", "drop_tests", "test_spans"]
