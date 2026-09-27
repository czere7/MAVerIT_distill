"""The two prompt fields that depend on retrieval: the class under test and its context.

RETRIEVER (from .env, or --retriever) picks the implementation:
  v1  utils.get_relevant_source_files: class names substring-matched against the CUT's text.
      The CUT is shown as-is. This is the harness as it was; its prompts are unchanged.
  v4  the collaborator index (collaborators.py), walked (collaborator_walk.py) and rendered
      (collaborator_render.py): what a test can use, recipes for what it must build or
      obtain, and the CUT with its private declarations tagged. Needs target/classes.
  v5  v4 plus two fixes found by the gpt-oss (low) ablation: every collaborator view starts
      with its package line (and recipes name the full type), so the teacher can write the
      import; and an enum's recipes are its constants, not valueOf(String).
  v6  v5 plus superclass chains followed through NESTED classes (index binary_ancestors /
      type_supers): supertype views and inherited API for classes like JsonFactoryBuilder,
      whose superclass DecorableTSFactory$DecorableTSFBuilder hid TSFBuilder from v5.
"""
from pathlib import Path
from typing import Any, Mapping

import collaborator_render as R
import collaborators
from collaborator_walk import walk
from utils import (SourceCodeFileData, config, format_source_files_for_prompt,
                   get_current_class_under_test, get_relevant_source_files, get_working_directory,
                   parser)

RETRIEVERS = ("v1", "v4", "v5", "v6")
# v5 since the gpt-oss (low) ablation: green 10/12 vs v1 7/12, mean mutation 43.2 vs 22.6,
# about half the tokens. v1 stays selectable for comparison.
DEFAULT_RETRIEVER = "v5"

V4_NOTE = (
    "Only members that a test in this package can use are shown. Declarations in the class "
    "under test marked PRIVATE are not callable from the test: reach that code through the "
    "class's non-private methods. \"How to build\" and \"How to obtain\" lines show how to "
    "construct or get an instance of a type."
)
V5_NOTE = V4_NOTE + (" Members listed as INHERITED in the class under test are not callable "
                     "from the test either. Each collaborator is shown with its package: import "
                     "the types you use.")

_INDEX_CACHE: dict[str, dict] = {}


def retriever() -> str:
    name = (config.get("RETRIEVER") or DEFAULT_RETRIEVER).strip()
    if name not in RETRIEVERS:
        raise RuntimeError(f"RETRIEVER must be one of {', '.join(RETRIEVERS)}, got {name!r}")
    return name


def prepare(project_dir: Path) -> str:
    """Called once by file_retriever_node. For v4/v5: compile if needed, build or load the index."""
    if retriever() == "v1":
        return "retriever v1"
    if not (project_dir / "target" / "classes").is_dir():
        collaborators.compile_project(project_dir)
    index = collaborators.load(project_dir)
    _INDEX_CACHE[str(project_dir)] = index
    return f"retriever {retriever()}, collaborator index with {len(index)} classes"


def _index(project_dir: Path) -> dict:
    key = str(project_dir)
    if key not in _INDEX_CACHE:
        prepare(project_dir)
    return _INDEX_CACHE[key]


def _fqn(f: SourceCodeFileData) -> str:
    import re
    m = re.search(r"^\s*package\s+([\w.]+)\s*;", f.file_content, re.M)
    return (m.group(1) + "." if m else "") + Path(f.file_path).stem


def prompt_context(agent_state: Mapping[str, Any]) -> tuple[str, str]:
    """(class_under_test, relevant_source_files) as the prompt templates take them."""
    cut = get_current_class_under_test(agent_state)
    if retriever() == "v1":
        return cut.file_content, format_source_files_for_prompt(get_relevant_source_files(agent_state))

    index = _index(get_working_directory())
    by_fqn = {_fqn(f): f for f in agent_state["all_files"]}
    cut_fqn = _fqn(cut)
    if cut_fqn not in index:                      # not compiled into the index: fall back
        return cut.file_content, format_source_files_for_prompt(get_relevant_source_files(agent_state))
    test_pkg = index[cut_fqn]["package"]
    src = cut.file_content.encode("utf-8")
    cut_ids = R.identifiers(parser.parse(src).root_node, src)
    v5 = retriever() in ("v5", "v6")              # v6 = v5 + nested superclass chains
    v6 = retriever() == "v6"
    supers = R.supertype_views(index, by_fqn, cut_fqn, test_pkg, cut_ids, nested=v6)
    producer_lines = R.producer_recipes(index, cut_fqn, test_pkg)

    sections, builds = [], []
    for cand in walk(index, cut_fqn, nested_ancestors=v6):
        f = by_fqn.get(cand.fqn)
        if f is None or cand.pattern == "producers":
            continue
        text = R.render_candidate(cand, f, index, cut_fqn, test_pkg, cut_ids, supers, producer_lines,
                                  enum_constants=v5, nested=v6)
        if not text:
            continue
        # The Extractor doubles every newline; in a collaborator view blank lines carry nothing.
        text = "\n".join(line for line in text.splitlines() if line.strip())
        if cand.pattern in R.BUILD_ROLES:
            named = f" ({cand.fqn})" if v5 else ""
            builds.append(f"How to build {R.article(cand.fqn.rsplit('.', 1)[-1])}{named}:\n"
                          + "\n".join(f"    {line}" for line in text.splitlines()))
        else:
            if v5 and index.get(cand.fqn, {}).get("package"):
                text = f"package {index[cand.fqn]['package']};\n{text}"
            sections.append(f"### {cand.fqn}\n\nPath: `{f.file_path}`\n\n```java\n{text}\n```")
    if producer_lines:
        simple = cut_fqn.rsplit(".", 1)[-1]
        via = (" via " + ", ".join(p["class"] for p in index[cut_fqn]["constructed_by"])) if v5 else ""
        builds.insert(0, f"How to obtain {R.article(simple)} (other than its own constructors){via}:\n"
                      + "\n".join(f"    {line}" for line in producer_lines))
    if builds:
        sections.append("### How to build or obtain other types\n\n```\n" + "\n\n".join(builds) + "\n```")
    note = V5_NOTE if v5 else V4_NOTE
    context = "\n\n".join([note] + sections) if sections else note
    cut_text = R.tag_private_members(cut.file_content)
    if v5:
        cut_text = R.tag_inherited_members(cut_text, R.inaccessible_inherited(index, cut_fqn, cut_ids))
    return cut_text, context
