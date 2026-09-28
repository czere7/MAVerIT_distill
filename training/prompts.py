"""The prompt set: every class the loop trains or evaluates on, with the prompt the harness
itself would send (retriever, then comment stripping -- initial_test_write_node._build_prompt).

One builder for both uses means the student is trained on exactly the prompt the harness
writes, and the fall-through harness run sees the same context the model saw.

PROMPT LIMIT 20,480 TOKENS. cutoff_len is 24,576 for prompt AND target, and LLaMA-Factory
truncates the END, which is the target's closing fence and EOS. 20,480 leaves 4k for the
target; measured on v5, it excludes 16 of 221 training classes and none of the held-out
set. The exact prompt + target check still runs when a dataset is written.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

PROMPT_LIMIT = 20_480
# Split BY PROJECT: a class-level split would let train and eval share collaborators, idioms
# and helpers, and measure recall. commons-cli, held out through the preference and RFT
# phases, trains now; the held-out projects are new to the loop and differ in style from
# training (jsoup is not Apache code; joda-money is joda-time's author family, a known,
# partial leak; petclinic-rest is a Spring Boot backend, the one application in the set).
# joda-money is pinned to v1.0.7, the last release that builds on JDK 17.
HELD_OUT = ("jsoup-clean", "joda-money-clean", "petclinic-rest-clean")
TRAINING = ("commons-cli-clean", "commons-codec-clean", "commons-csv-clean",
            "jackson-core-clean", "joda-time-clean")


def class_key(project_dir: Path, file_path: str) -> str:
    rel = Path(file_path).relative_to(project_dir / "src" / "main" / "java")
    return rel.with_suffix("").as_posix()


def build(projects: tuple[str, ...], scratch: Path) -> list[dict]:
    """Prompt rows for every concrete class of `projects`, token-counted, over-limit ones
    marked `excluded` rather than dropped so the report can say what was left out."""
    from Extractor import Extractor
    from nodes.initial_test_write_node import _build_prompt
    import retrieval
    from training import config as tconfig
    from training.tokens import count
    from utils import config, is_concrete_class

    config["RETRIEVER"] = "v5"          # the only retriever the loop is wired for
    rows = []
    for name in projects:
        project = tconfig.PROJECTS_ROOT / name
        config["WORKING_DIRECTORY"] = str(project)     # the retriever resolves against it
        retrieval._INDEX_CACHE.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            files = Extractor(str(project)).extract_all()
        targets = [f for f in files if is_concrete_class(f)]
        main_java = project / "src" / "main" / "java"
        for i, cut in enumerate(targets):
            # Only src/main/java: class keys, the collaborator index (target/classes) and the
            # harness all address classes there. Multi-release variants (jsoup's
            # src/main/java11, two HTTP client classes) compile into META-INF/versions.
            if not Path(cut.file_path).is_relative_to(main_java):
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                prompt = _build_prompt({"all_files": files, "all_test_files": targets,
                                        "current_class_index": i})
            rows.append({"class_key": class_key(project, cut.file_path), "project": name,
                         "prompt": prompt})
    rows = count(rows, scratch)
    for r in rows:
        r["excluded"] = r["prompt_tokens"] > PROMPT_LIMIT
    return rows


def save(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def load(path: Path, include_excluded: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    return rows if include_excluded else [r for r in rows if not r["excluded"]]
