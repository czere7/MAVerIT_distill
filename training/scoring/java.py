"""Java source inspection -- no build tools, no I/O, no GPU. Tier 0 lives here.

PORTED FROM MAVerIT (utils.py). The regexes and `count_test_without_assert` are copied
verbatim so that a class scored here and a class scored by the harness agree exactly;
divergence between the two would silently invalidate every comparison in the paper.

New here: `count_tests`, `is_balanced` and `looks_terminated`, which MAVerIT has no need
for because its graph never asks "did the model stop".
"""

from __future__ import annotations

import re

# --- verbatim from MAVerIT utils.py:21 and :26 --------------------------------------
PACKAGE_RE = re.compile(
    r'^\s*package\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*;',
    re.MULTILINE,
)

CLASS_RE = re.compile(r'''
    ^[ \t]*
    (?:@(?!interface\b)[A-Za-z_$][\w$]*(?:\s*\([^)]*\))?\s*)*
    (?:
        (?:public|protected|private|abstract|final|static|strictfp|sealed|non-sealed)
        \s+
    )*
    (?:@interface|class|interface|enum|record)
    \s+
    ([A-Za-z_$][\w$]*)
    \b
''', re.MULTILINE | re.VERBOSE)

TEST_RE = re.compile(r"^\s*@Test\b", re.M)


def get_java_entity_name(java_code: str) -> str:
    """Ported from MAVerIT utils.py:528."""
    class_match = CLASS_RE.search(java_code)
    if class_match is None:
        raise RuntimeError(
            "Java code does not contain a class, interface, enum, or record declaration.")
    return class_match.group(1)


def get_java_package_name(java_code: str) -> str:
    """Ported from MAVerIT utils.py:535."""
    package_match = PACKAGE_RE.search(java_code)
    if package_match is None:
        return ""
    return package_match.group(1)


def get_java_fully_qualified_name(java_code: str) -> str:
    """Ported from MAVerIT utils.py:542. Used for PIT's targetClasses/targetTests."""
    entity_name = get_java_entity_name(java_code)
    package_name = get_java_package_name(java_code)
    if not package_name:
        return entity_name
    return f"{package_name}.{entity_name}"


def strip_markdown_code_fence(response: str) -> str:
    """Ported from MAVerIT utils.py:652.

    Kept byte-identical because the training targets are fenced: the model emits
    ```java ... ``` and this is what turns that back into compilable Java. An unpaired
    fence count returns the response untouched -- the harness gives up too, and scoring
    something the harness would not accept would inflate our numbers.
    """
    if '```' in response:
        response_chunks = response.split('```')

        if len(response_chunks) % 2 == 0:
            return response  # unpaired chunks; cannot tell which part is code

        code_chunks = [item for i, item in enumerate(response_chunks) if i % 2 == 1]
        candidate_code_chunk = max(code_chunks, key=len)

        lines = candidate_code_chunk.split('\n')
        if ('package' not in lines[0] and 'import' not in lines[0]
                and 'class' not in lines[0]):
            lines = lines[1:]
        return '\n'.join(lines)

    return response


def count_test_without_assert(src: str) -> int:
    """Ported verbatim from MAVerIT utils.py:124.

    Counts @Test methods containing no JUnit assertion and no `expected = ...Exception`
    annotation. A high count means the suite is going through the motions -- it executes
    code (so it scores on coverage) while asserting nothing (so it kills few mutants).
    """
    junit_asserts = [
        'assertAll', 'assertArrayEquals', 'assertDoesNotThrow',
        'assertEquals', 'assertFalse',
        'assertInstanceOf', 'assertIterableEquals', 'assertLinesMatch',
        'assertNotNull', 'assertNotSame', 'assertNotEquals',
        'assertNull', 'assertSame', 'assertThat',
        'assertThrows', 'assertThrowsExactly', 'assertTimeout',
        'assertTimeoutPreemptively', 'assertTrue', 'fail']

    assertion_pattern = re.compile(
        rf"\b(?:{'|'.join(map(re.escape, junit_asserts))})\s*\("
    )
    annotated_assertion_pattern = re.compile(
        r"\s*\([^)]*\bexpected\s*=.*Exception\.class\b[^)]*\)",
        re.DOTALL,
    )

    return sum(
        (not assertion_pattern.search(test_body)
         and not annotated_assertion_pattern.search(test_body))
        for test_body in src.split("@Test")[1:]
    )


# --- new: tier-0 checks MAVerIT has no equivalent for --------------------------------

def count_tests(src: str) -> int:
    """Number of @Test methods. The quantity runs 1-4 could not learn to choose."""
    return len(TEST_RE.findall(src))


def is_balanced(src: str) -> bool:
    """Braces AND parentheses both balanced.

    Checking braces alone once led to seven generated files being called "structurally
    sound" when two had unbalanced parentheses and could not compile.
    """
    # Literal-aware. A naive count treats a brace inside a string as structure, and a
    # Java test asserting on CLI output is full of them. Measured as LATENT on the
    # current held-out set -- 0 of 23 disagree -- but a false "unbalanced" costs a
    # whole retry in repair.py, which is the most expensive branch there is.
    clean = "\n".join(_strip_literals(line) for line in src.splitlines())
    return clean.count("{") == clean.count("}") and clean.count("(") == clean.count(")")


def looks_terminated(src: str) -> bool:
    """Heuristic: does the text end like a finished class?

    Only a heuristic -- the authoritative signal is the generation's finish_reason, which
    the caller passes in when it has one. This exists for scoring text read off disk,
    where that signal is gone.
    """
    stripped = src.rstrip()
    return stripped.endswith("}") or stripped.endswith("```")


def _strip_literals(line: str) -> str:
    """Blank out string/char literals and line comments so brace counting is honest."""
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])*'", "''", line)
    return re.sub(r"//.*$", "", line)


def strip_fence_lenient(response: str) -> str:
    """Like `strip_markdown_code_fence`, but recovers when the fence count is UNPAIRED.

    WHY A SECOND FUNCTION rather than fixing the first: the ported one is byte-identical
    to MAVerIT on purpose, and MAVerIT's give-up-on-unpaired behaviour is correct THERE --
    an unparseable answer is a failed generation and scoring it leniently would inflate
    the numbers. `score()` still uses it, so the measured compile rates are untouched.

    It is wrong HERE. A runaway is truncated mid-file, so its opening ```java never gets a
    closing ``` -- `split('```')` yields two chunks, an even count, and the input comes
    back untouched with the fence still on line 1. In this pipeline the runaway is the
    INPUT TO REPAIR, not a final answer, and a fence on line 1 is a syntax error that no
    amount of fixing the tail of the file can reach. Measured: all 20 held-out runaways
    carried a live ```java into the repairer.
    """
    stripped = strip_markdown_code_fence(response)
    lines = stripped.splitlines()
    while lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].lstrip().startswith("```"):
        lines = lines[:-1]
    return ("\n".join(lines) + "\n") if lines else ""


def truncate_to_last_complete_member(src: str) -> str:
    """Drop the partial tail of a truncated class and close it.

    A runaway stops mid-method. Everything up to the last member that closed cleanly is
    intact Java; the remainder is a fragment. Keeping the intact part and adding the
    class's closing brace is the minimal repair, and it needs no model.
    """
    lines = src.splitlines()
    depth = 0
    last_member_end = None
    for i, line in enumerate(lines):
        for char in _strip_literals(line):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 1:          # a member of the class body just closed
                    last_member_end = i
    if last_member_end is None:
        return src
    return "\n".join(lines[:last_member_end + 1]) + "\n}\n"


def deduplicate_tests(src: str) -> tuple[str, int]:
    """Remove repeated @Test methods, keeping the first of each name.

    Returns (source, number_removed).

    A model that fails to terminate often loops, emitting the same method again and again
    -- CommandLine's runaway repeated one name FIFTEEN times -- and javac rejects the class
    with "is already defined". For that failure mode this IS the repair, and it is exactly
    the behaviour the preference training is meant to remove.

    Expects complete methods, so run it AFTER `truncate_to_last_complete_member`.
    """
    lines = src.splitlines()
    blocks: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        if "@Test" not in lines[i]:
            i += 1
            continue

        name, sig = None, None
        for j in range(i, min(i + 6, len(lines))):
            match = re.search(r"\bvoid\s+(\w+)\s*\(", lines[j])
            if match:
                name, sig = match.group(1), j
                break
        if name is None:
            i += 1
            continue

        depth, end, started = 0, None, False
        for j in range(sig, len(lines)):
            for char in _strip_literals(lines[j]):
                if char == "{":
                    depth += 1
                    started = True
                elif char == "}":
                    depth -= 1
            if started and depth <= 0:
                end = j
                break
        if end is None:
            break
        blocks.append((i, end, name))
        i = end + 1

    seen: set[str] = set()
    drop: set[int] = set()
    removed = 0
    for start, end, name in blocks:
        if name in seen:
            drop.update(range(start, end + 1))
            removed += 1
        else:
            seen.add(name)

    if not removed:
        return src, 0
    kept = [line for n, line in enumerate(lines) if n not in drop]
    return "\n".join(kept) + "\n", removed
