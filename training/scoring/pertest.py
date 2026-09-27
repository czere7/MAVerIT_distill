"""Per-@Test grading: turn one javac run into a fraction, not a bit.

WHY. `quality()` weights compilation at 0.4 and treats it as binary, so a suite where
seven of eight tests are fine scores exactly what a suite of pure garbage scores. Measured
on the middle-band rollouts, that is not a corner case -- it is the whole distribution.
`quality_rejected` is 0.000 for all 93 pairs in the ODPO dataset, and 102 of 116 prompts
produce no compiling rollout in eight attempts. A binary gate cannot rank things that all
fail, and both the DPO offset and any group-relative RL advantage need that ranking.

ONE COMPILE, NOT N. javac reports every error with a line number, so a single
`mvn test-compile` can be attributed to individual methods by asking tree-sitter which
`@Test` encloses each reported line. Compiling each test separately would cost one Maven
invocation per test -- at 8 to 30 tests per suite, that is the difference between seconds
and minutes, on the hot path of every pair build.

WHAT IT IS NOT, AND HOW FAR OFF IT IS. "No error attributed to this test" is a PROXY for
"this test compiles on its own", not a proof. MEASURED against ground truth -- 76 tests
over 19 rollouts, each @Test compiled alone in its own suite and compared with what one
whole-suite compile predicted:

    predicted clean  & compiles alone    28
    predicted clean  & FAILS alone        2     <- inflates the reward
    predicted broken & compiles alone     0     <- would deflate it; did not occur
    predicted broken & fails alone       46

    agreement 97.4%,  precision of "clean" 93.3%

The error is one-sided: nothing was called broken that actually compiled, so the proxy
never punishes a good test. It over-credits about 1 predicted-clean test in 15.

Both misses came from javac not reporting every error in the whole-suite run. javac works
in phases, and severe errors in other methods can stop it before it attributes the lines
of a later one -- ColognePhonetic #0 and StreamConstraintsException #7 each had a genuine
error INSIDE the test's own span that the suite-level compile never printed. So the bias
is worst exactly where the suite is worst, which is also where the graded reward matters
least. Two further known sources, same direction:

  - javac stops at 100 errors by default, so a suite past the cap has tests that are
    broken but unreported. `Attribution.truncated` says when the cap was hit.
  - a test calling a broken private helper carries no error of its own; the error sits in
    the helper. Helpers live outside every @Test span, so they land in `skeleton_errors`,
    which zeroes the whole suite -- that direction is safe.

Measured in ~/maverit-preference with scripts/validate_attribution.py.

SKELETON FAULTS ARE FATAL, DELIBERATELY. An error outside every @Test method -- a bad
import, a field of a type that does not exist, a broken @Before -- stops the class from
compiling no matter which tests are dropped. Scoring such a suite as "6 of 8 tests clean"
would credit tests that can never run. Measured on the 51-rollout pilot: skeleton faults
hit 36% of COMPILE_FAIL and 68% of salvaged-UNPARSEABLE rollouts, so this is most of what
the grader sees, not an edge case.

tree-sitter is an OPTIONAL dependency. It is present in the harness venv (~/distil-env) and
absent from `~/maverit-ft` (GPU work). Scoring must not break in the venv that lacks it, so
the import is lazy and `Attribution.available` reports whether attribution actually ran.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# `[ERROR] /abs/path/FooTest.java:[57,24] cannot find symbol`
_JAVAC_ERROR = re.compile(r"^\[ERROR\]\s+(\S+\.java):\[(\d+),(\d+)\]\s+(.*)$")
# javac's own tally, printed when it gives up: `[INFO] 100 errors`
_ERROR_CAP = re.compile(r"^\[[A-Z]+\]\s*(\d+)\s+errors?\s*$")
JAVAC_DEFAULT_MAXERRS = 100


@dataclass(frozen=True)
class TestSpan:
    """One @Test method, 1-indexed and inclusive, annotations included.

    The annotation matters: `@Test(expected = X.class)` with a bad X reports on the
    annotation line, which is above the method declaration.
    """

    name: str
    start_line: int
    end_line: int

    def contains(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line


@dataclass(frozen=True)
class Attribution:
    """Which @Test methods javac complained about, and what it complained about elsewhere."""

    spans: tuple[TestSpan, ...] = ()
    broken: frozenset[str] = frozenset()
    skeleton_errors: tuple[tuple[int, str], ...] = ()
    n_errors: int = 0
    truncated: bool = False
    available: bool = True

    @property
    def n_tests(self) -> int:
        return len(self.spans)

    @property
    def n_broken(self) -> int:
        return len(self.broken)

    @property
    def n_clean(self) -> int:
        """@Test methods with no error attributed. Zero when the skeleton is broken."""
        if self.skeleton_errors:
            return 0
        return self.n_tests - self.n_broken

    @property
    def compiled_fraction(self) -> float:
        """Share of the suite that survives, in [0, 1].

        Zero when the skeleton is broken, when nothing could be attributed, or when there
        are no tests at all -- an empty class compiles, and rewarding that would teach the
        model to write nothing.
        """
        if not self.available or self.skeleton_errors or not self.n_tests:
            return 0.0
        return self.n_clean / self.n_tests


@lru_cache(maxsize=1)
def _parser():
    """None when tree-sitter is not installed in this venv."""
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_java
    except ImportError:
        return None
    return Parser(Language(tree_sitter_java.language()))


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def test_spans(source: str) -> tuple[TestSpan, ...]:
    """Every @Test method in `source`, in document order.

    Works on a file that does not parse: tree-sitter recovers around ERROR nodes, so the
    intact methods of a broken suite are still found. That is the whole reason for using a
    parser here rather than a regex over `@Test`.
    """
    parser = _parser()
    if parser is None:
        return ()
    blob = source.encode("utf-8")
    spans: list[TestSpan] = []
    for node in _walk(parser.parse(blob).root_node):
        if node.type != "method_declaration":
            continue
        start = node.start_point[0] + 1
        body = blob[node.start_byte:node.end_byte].decode("utf-8", "replace")
        annotated = "@Test" in body
        # Annotations are a  child in tree-sitter-java, but some grammars emit
        # them as a preceding sibling. Accept that shape too -- and ONLY that shape. An
        # earlier version accepted any preceding sibling containing "@Test", which meant a
        # private helper following an annotated test inherited its annotation and was
        # graded as a test, with a span swallowing the method above it.
        ANNOTATION_NODES = {"modifiers", "annotation", "marker_annotation",
                            "annotation_argument_list"}
        parent = node.parent
        if parent is not None:
            kids = parent.children
            idx = kids.index(node)
            if idx > 0 and kids[idx - 1].type in ANNOTATION_NODES:
                prev = kids[idx - 1]
                text = blob[prev.start_byte:prev.end_byte].decode("utf-8", "replace")
                if "@Test" in text:
                    annotated = True
                    start = prev.start_point[0] + 1
        if not annotated:
            continue
        name_node = node.child_by_field_name("name")
        name = (blob[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace")
                if name_node is not None else f"<line {start}>")
        spans.append(TestSpan(name=name, start_line=start, end_line=node.end_point[0] + 1))
    spans.sort(key=lambda s: s.start_line)
    return tuple(spans)


def drop_tests(source: str, names: frozenset[str] | set[str]) -> str:
    """`source` with the named @Test methods removed, everything else byte-identical.

    Removal, never repair. The result is a strict subset of what the model wrote, so a
    suite scored this way is credited only for lines it actually produced -- which is the
    line `score()` must not cross if the reward is to mean anything.
    """
    if not names:
        return source
    lines = source.splitlines()
    drop: set[int] = set()
    for span in test_spans(source):
        if span.name in names:
            drop.update(range(span.start_line - 1, span.end_line))
    kept = [l for i, l in enumerate(lines) if i not in drop]
    return "\n".join(kept) + "\n"


def parse_javac_errors(compiler_output: str, test_file_name: str) -> tuple[
        list[tuple[int, str]], bool]:
    """[(line, message)] for one file, deduplicated, plus whether javac hit its error cap.

    Maven prints each compilation error TWICE -- once in its own COMPILATION ERROR block
    and again under "Failed to execute goal" -- so an undeduplicated count is double, and a
    per-test count would be too.
    """
    errors: list[tuple[int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    truncated = False
    for raw in compiler_output.splitlines():
        line = raw.strip()
        cap = _ERROR_CAP.match(line)
        if cap and int(cap.group(1)) >= JAVAC_DEFAULT_MAXERRS:
            truncated = True
            continue
        m = _JAVAC_ERROR.match(line)
        if not m:
            continue
        path, row, col, message = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if not path.endswith(test_file_name):
            continue                      # an error in the project, not in our test class
        key = (row, col, message)
        if key in seen:
            continue
        seen.add(key)
        errors.append((row, message))
    return errors, truncated


def attribute(source: str, compiler_output: str, test_file_name: str) -> Attribution:
    """Attribute each javac error to the @Test method enclosing its line."""
    if _parser() is None:
        return Attribution(available=False)

    spans = test_spans(source)
    errors, truncated = parse_javac_errors(compiler_output, test_file_name)

    broken: set[str] = set()
    skeleton: list[tuple[int, str]] = []
    for line, message in errors:
        owner = next((s.name for s in spans if s.contains(line)), None)
        if owner is None:
            skeleton.append((line, message))
        else:
            broken.add(owner)

    return Attribution(
        spans=spans,
        broken=frozenset(broken),
        skeleton_errors=tuple(skeleton),
        n_errors=len(errors),
        truncated=truncated,
    )
