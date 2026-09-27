"""Where the smokes find the Maven projects and the old MAVerIT inference records.

Both come from .env (PROJECTS_ROOT, INFERENCE_RECORDS) so no script hardcodes a machine.
The defaults are the WSL layout the harness moved to on 2026-09-27; the records stay on the
Windows side, read-only, because MAVerIT4 is not ours to copy or change.
"""
import sys
from pathlib import Path

# Some smokes import this before (or instead of) putting the harness on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import config  # noqa: E402

PROJECTS = Path(config.get("PROJECTS_ROOT") or Path.home() / "harness-projects")
RECORDS = Path(config.get("INFERENCE_RECORDS")
               or "/mnt/c/Users/akosc/Desktop/MAVerIT4/inference_records")
