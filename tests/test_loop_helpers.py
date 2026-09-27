"""The small decisions the loop routes on: which rollout the harness starts from, what counts
as killing a mutant, and which suite a harness run ended with."""

import json

from training.harness_run import final_suite, most_tests
from training.score_rollouts import kills_a_mutant
from training.targets import fence

SUITE = """package p;

import org.junit.Test;

public class FooTest {
%s
}
"""


def suite_of(n: int) -> str:
    return "\n".join(f"    @Test\n    public void t{i}() {{ }}" for i in range(n))


def test_most_tests_picks_the_longest_rollout_and_breaks_ties_to_the_lower_index():
    gens = [{"rollout": 0, "text": SUITE % suite_of(2)},
            {"rollout": 1, "text": "```java\n" + SUITE % suite_of(5) + "```"},
            {"rollout": 2, "text": SUITE % suite_of(5)}]
    assert most_tests(gens)["rollout"] == 1


def test_kills_a_mutant_reads_every_subset_a_target_could_come_from():
    assert not kills_a_mutant({"mutation_score_pct": None})
    assert not kills_a_mutant({"mutation_score_pct": 0.0, "green_subset_mutation_score_pct": 0.0})
    assert kills_a_mutant({"mutation_score_pct": 12.5})
    assert kills_a_mutant({"green_subset_mutation_score_pct": 3.0})
    assert kills_a_mutant({"salvaged_mutation_score_pct": 1.0})


def test_final_suite_is_the_last_green_restore_point(tmp_path):
    rows = [
        {"last_compilable_test_class": None, "suite_green": False, "mutation": "None"},
        {"last_compilable_test_class": SUITE % suite_of(2), "suite_green": True, "mutation": 40.0},
        {"last_compilable_test_class": SUITE % suite_of(3), "suite_green": True, "mutation": 55.0},
        {"last_compilable_test_class": SUITE % suite_of(3), "suite_green": False, "mutation": "None"},
    ]
    (tmp_path / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    suite = final_suite(tmp_path)
    assert suite["n_tests"] == 3
    assert suite["mutation_score_pct"] == 55.0


def test_final_suite_is_none_when_nothing_was_ever_green(tmp_path):
    (tmp_path / "log.jsonl").write_text(json.dumps({"last_compilable_test_class": None}) + "\n")
    assert final_suite(tmp_path) is None
    assert final_suite(tmp_path / "missing") is None


def test_fence_is_what_the_harness_unwraps():
    from utils import strip_markdown_code_fence
    src = SUITE % suite_of(1)
    assert strip_markdown_code_fence(fence(src)).strip() == src.strip()
