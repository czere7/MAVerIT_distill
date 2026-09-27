import argparse
import contextlib
import io
import json
import sys
import traceback
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import END, START, StateGraph

from AgentState import AgentState
from Extractor import Extractor
from conditions import route_after_class_advance, route_after_compile
from models import REASONING_EFFORTS
from retrieval import RETRIEVERS
from nodes.class_advancer_node import class_advancer_node
from nodes.compiler_node import compiler_node
from nodes.faulty_test_cleanup_node import faulty_test_cleanup_node
from nodes.file_retriever_node import file_retriever_node
from nodes.initial_test_write_node import initial_test_write_node
from nodes.mutation_test_writer_node import mutation_test_writer_node
from nodes.test_repair_node import test_repair_node
from utils import config, get_working_directory, is_concrete_class


def build_graph():
    """One harness, and the graph it is.

    Four nodes left this graph relative to MAVerIT, and each departure is a measurement
    rather than a preference:

      dependency_resolver   per-project setup masquerading as per-class work, and
                            unreliable. Dependencies are prepared once, outside the loop.
      coverage_validator    killing a mutant needs the line EXECUTED and the difference
      coverage_test_writer  ASSERTED. Mutation-driven writing therefore does the coverage
                            work, aimed at lines that matter; coverage-driven writing adds
                            execution without assertion, which is precisely how a suite
                            that runs the class and catches nothing gets built. 17 of the
                            46 classes the model cannot serve are exactly that shape.
      mutation_validator    MERGED into compiler_node rather than deleted. The router asks
                            "is mutation satisfied?", and it can only ask that if PIT's
                            result is in state -- so PIT runs where the compile runs.

    What is left is one decision point. compiler_node compiles, runs, measures coverage
    and, when the suite is green, mutation; route_after_compile then picks one of four
    destinations. Everything else is a straight edge back into it.
    """
    graph = StateGraph(AgentState)

    graph.add_node("file_retriever_node", file_retriever_node)
    graph.add_node("initial_test_write_node", initial_test_write_node)
    graph.add_node("compiler_node", compiler_node)
    graph.add_node("test_repair_node", test_repair_node)
    graph.add_node("mutation_test_writer_node", mutation_test_writer_node)
    graph.add_node("faulty_test_cleanup_node", faulty_test_cleanup_node)
    graph.add_node("class_advancer_node", class_advancer_node)

    graph.add_edge(START, "file_retriever_node")
    graph.add_edge("file_retriever_node", "initial_test_write_node")
    graph.add_edge("initial_test_write_node", "compiler_node")

    # Both writers hand their output straight back to the compile node. Nothing between
    # them measures anything any more, so there is nothing for an edge to stop at.
    graph.add_edge("test_repair_node", "compiler_node")
    graph.add_edge("mutation_test_writer_node", "compiler_node")

    graph.add_edge("faulty_test_cleanup_node", "class_advancer_node")

    graph.add_conditional_edges(
        "compiler_node",
        route_after_compile,
        {
            "test_repair_node": "test_repair_node",
            "mutation_test_writer_node": "mutation_test_writer_node",
            "faulty_test_cleanup_node": "faulty_test_cleanup_node",
            "class_advancer_node": "class_advancer_node",
        },
    )
    graph.add_conditional_edges(
        "class_advancer_node",
        route_after_class_advance,
        {
            "initial_test_write_node": "initial_test_write_node",
            "end": END,
        },
    )

    return graph.compile()


app = build_graph()


# ================================================================================ CLI

EXAMPLES = """
modes:
  batch         every concrete class in the project, resumable from the checkpoint file
  single        --class: one class, then stop. Never reads or deletes the batch checkpoint
  rollout       --class + --rollout: the first draft comes from a file instead of the
                teacher, and the harness repairs and extends it. Same graph as the other
                two; only the source of the initial suite differs.

examples:
  python main.py --list-classes
  python main.py                                   start or resume a batch run
  python main.py --fresh                           discard the checkpoint, start from class 0
  python main.py --class URLCodec                  one class, by simple name
  python main.py --class org.apache.commons.codec.net.URLCodec
  python main.py --class URLCodec --rollout rollout.java

Run from the harness directory: .env, prompts/ and the per-run log directories all
resolve against the working directory.
"""


class CliError(Exception):
    """A problem with how the harness was invoked, reported before anything is spent."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Generate, repair and mutation-harden JUnit tests for a Maven project.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    where = parser.add_argument_group("what to run on")
    where.add_argument("--project", metavar="DIR",
                       help="Maven project to test; overrides WORKING_DIRECTORY")
    where.add_argument("--class", dest="target_class", metavar="NAME",
                       help="run one class only: simple name, fully-qualified name, or a "
                            "path suffix such as net/URLCodec")
    where.add_argument("--rollout", metavar="FILE", type=Path,
                       help="use this test suite as the first draft instead of calling the "
                            "teacher; requires --class")
    where.add_argument("--list-classes", action="store_true",
                       help="print the classes the harness would test, then exit")

    model = parser.add_argument_group("model")
    model.add_argument("--model", help="overrides MODEL")
    model.add_argument("--provider", help="overrides PROVIDER (openai, deepseek, ollama)")
    model.add_argument("--reasoning-effort", choices=REASONING_EFFORTS,
                       help="overrides REASONING_EFFORT. ollama/openai take the levels; "
                            "deepseek only honours off vs on")
    model.add_argument("--retriever", choices=RETRIEVERS,
                       help="overrides RETRIEVER (default v5 = collaborator index, package lines, "
                            "enum constants; v6 = v5 + nested superclass chains; v4 = v5 without "
                            "packages/enums; v1 = substring-matched files)")

    run = parser.add_argument_group("run control")
    run.add_argument("--run-id", metavar="ID",
                     help="name of the log directory; defaults to a generated id")
    run.add_argument("--fresh", action="store_true",
                     help="discard the batch checkpoint and start from the first class")
    run.add_argument("--max-restarts", type=int, default=3, metavar="N",
                     help="consecutive failures tolerated without progress before giving "
                          "up (default: 3)")

    args = parser.parse_args(argv)
    if args.rollout and not args.target_class:
        parser.error("--rollout needs --class: a rollout is one class's test suite")
    if args.fresh and args.target_class:
        parser.error("--fresh resets the batch checkpoint, which a --class run never touches")
    if args.max_restarts < 0:
        parser.error("--max-restarts cannot be negative")
    return args


def _apply_overrides(args: argparse.Namespace) -> None:
    # `config` is one dict, imported by reference by every module and read at call time,
    # so writing to it here reaches every node without threading arguments through them.
    for key, value in (("WORKING_DIRECTORY", args.project),
                       ("MODEL", args.model),
                       ("PROVIDER", args.provider),
                       ("REASONING_EFFORT", args.reasoning_effort),
                       ("RETRIEVER", args.retriever)):
        if value:
            config[key] = str(value)


def _target_classes():
    """The classes to test, in exactly the order file_retriever_node will see them."""
    try:
        project_dir = get_working_directory()
    except RuntimeError as error:
        raise CliError(f"{error}. Check --project, or WORKING_DIRECTORY in .env.") from error
    with contextlib.redirect_stdout(io.StringIO()):     # Extractor narrates every file
        files = Extractor(str(project_dir)).extract_all()
    return project_dir, [f for f in files if is_concrete_class(f)]


def _resolve_class(query: str, targets) -> str:
    """One class from a simple name, an FQN or a path suffix -- or a clear error.

    All three reduce to one rule: normalise the query to a slash-separated path without
    the extension, and match it against the tail of each class's path.
    """
    needle = query.replace("\\", "/").removesuffix(".java").replace(".", "/").strip("/")
    hits = [t for t in targets
            if (path := t.file_path.replace("\\", "/").removesuffix(".java")) == needle
            or path.endswith("/" + needle)]
    if len(hits) == 1:
        return hits[0].file_path
    if not hits:
        raise CliError(f"No concrete class matches '{query}'. "
                       f"Use --list-classes to see what the harness can test.")
    listed = "\n  ".join(h.file_path for h in hits)
    raise CliError(f"'{query}' is ambiguous -- {len(hits)} classes match:\n  {listed}\n"
                   f"Use a fully-qualified name or a longer path suffix.")


def _read_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="UTF-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _list_classes() -> int:
    project_dir, targets = _target_classes()
    print(f"{len(targets)} concrete classes in {project_dir}")
    for index, target in enumerate(targets):
        try:
            shown = Path(target.file_path).relative_to(project_dir)
        except ValueError:
            shown = Path(target.file_path)
        print(f"  {index:>4}  {shown}")
    return 0


def _prepare(args: argparse.Namespace) -> tuple[dict, Path]:
    """Validate everything and build the initial state -- before a single token is spent.

    Returns the state for app.invoke and the checkpoint path progress is measured by.
    """
    state: dict = {}

    if args.target_class:
        _, targets = _target_classes()
        target_path = _resolve_class(args.target_class, targets)
        state["target_class_path"] = target_path
        run_id = args.run_id or f"{Path(target_path).stem}-{uuid.uuid4().hex[:8]}"

        # ISOLATED CHECKPOINT. class_advancer deletes the checkpoint file when the last
        # class finishes, and in single-class mode the one class IS the last -- so pointed
        # at the batch checkpoint, a one-class run would silently erase a batch run's
        # resume point. Each single run gets its own file instead.
        run_dir = Path.cwd() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config["CLASS_INDEX_TMP_FILE"] = str(run_dir / "checkpoint.json")

        if args.rollout:
            if not args.rollout.is_file():
                raise CliError(f"Rollout file not found: {args.rollout}")
            source = args.rollout.read_text(encoding="utf-8")
            if not source.strip():
                raise CliError(f"Rollout file is empty: {args.rollout}")
            state["rollout_source"] = source
    else:
        _target_classes()                               # fail now on a bad project path
        checkpoint_path = Path(config.get("CLASS_INDEX_TMP_FILE", "tmp.txt"))
        existing = _read_checkpoint(checkpoint_path)
        if existing and args.fresh:
            print(f"[main] --fresh: discarding checkpoint at class {existing.get('class_index')} "
                  f"of run {existing.get('run_id')}")
            checkpoint_path.unlink()
            existing = None
        if existing:
            if args.run_id and args.run_id != existing.get("run_id"):
                raise CliError(
                    f"A checkpoint exists for run '{existing.get('run_id')}'. Resuming under "
                    f"'{args.run_id}' would split one run's logs across two directories. "
                    f"Drop --run-id to resume, or pass --fresh to start over.")
            run_id = existing.get("run_id")
            print(f"[main] resuming run {run_id} at class {existing.get('class_index')}")
        else:
            # Generated HERE rather than by the checkpoint reader, which mints a new id on
            # every call until the first checkpoint is written -- so a failure during the
            # first class used to restart under a different id and scatter the logs.
            run_id = args.run_id or str(uuid.uuid4())

    state["run_id"] = run_id
    return state, Path(config.get("CLASS_INDEX_TMP_FILE", "tmp.txt"))


def _run(state: dict, checkpoint_path: Path, max_restarts: int) -> int:
    """Invoke the graph, restarting after a failure -- but not forever.

    The restart is how a batch survives a transient failure: the checkpoint records the
    last finished class and file_retriever resumes from it. But the loop this replaces
    retried unconditionally, so a DETERMINISTIC failure retried until killed, re-calling
    the teacher each time. Failures are now counted only while no class completes; any
    progress resets the count.
    """
    failures = 0
    last_index = (_read_checkpoint(checkpoint_path) or {}).get("class_index")
    while True:
        try:
            app.invoke(state)
            return 0
        except Exception as error:      # KeyboardInterrupt is not an Exception: Ctrl+C stops
            index = (_read_checkpoint(checkpoint_path) or {}).get("class_index")
            if index != last_index:
                failures, last_index = 0, index
            failures += 1
            print(f"\n[main] run failed: {type(error).__name__}: {error}", file=sys.stderr)
            traceback.print_exc()
            if failures > max_restarts:
                print(f"[main] giving up after {failures} consecutive failures with no class "
                      f"completed (--max-restarts {max_restarts}).", file=sys.stderr)
                return 1
            print(f"[main] restarting ({failures}/{max_restarts}) from the last checkpoint",
                  file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    # image = Image.open(BytesIO(app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)))
    # image.show()
    args = parse_args(argv)
    _apply_overrides(args)
    try:
        if args.list_classes:
            return _list_classes()
        state, checkpoint_path = _prepare(args)
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    mode = ("rollout" if "rollout_source" in state
            else "single" if "target_class_path" in state else "batch")
    print(f"[main] mode={mode}  run_id={state['run_id']}")
    print(f"[main] project={config.get('WORKING_DIRECTORY')}  "
          f"model={config.get('PROVIDER')}/{config.get('MODEL')}")
    if "target_class_path" in state:
        print(f"[main] class={state['target_class_path']}")
    return _run(state, checkpoint_path, args.max_restarts)


if __name__ == "__main__":
    sys.exit(main())
