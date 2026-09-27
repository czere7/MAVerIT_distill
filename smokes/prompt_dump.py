"""Dump the initial-writer prompt of every ablation class, per retriever, as JSON on stdout.

Run it twice under different PYTHONHASHSEED values and compare: identical output means the
prompts are deterministic. Used by the determinism check; also handy for diffing retrievers.
    python smokes/prompt_dump.py v5,v6 > out.json
"""
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import PROJECTS  # noqa: E402
from Extractor import Extractor  # noqa: E402
from utils import config, is_concrete_class  # noqa: E402
import retrieval  # noqa: E402
from nodes.initial_test_write_node import _build_prompt  # noqa: E402


def main() -> int:
    arms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["v5"]
    classes = json.loads(Path("smokes/ablation_classes.json").read_text(encoding="utf-8"))
    out = {}
    files_cache = {}
    for entry in classes:
        project = str(PROJECTS / entry["project"])
        config["WORKING_DIRECTORY"] = project
        if project not in files_cache:
            with contextlib.redirect_stdout(io.StringIO()):
                files_cache[project] = Extractor(project).extract_all()
        files = files_cache[project]
        simple = entry["class"].rsplit(".", 1)[-1]
        cut = next(f for f in files if is_concrete_class(f) and Path(f.file_path).stem == simple
                   and entry["class"].replace(".", "/") in f.file_path.replace("\\", "/"))
        for arm in arms:
            config["RETRIEVER"] = arm
            retrieval._INDEX_CACHE.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                out[f"{arm}:{simple}"] = _build_prompt(
                    {"all_files": files, "all_test_files": [cut], "current_class_index": 0})
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
