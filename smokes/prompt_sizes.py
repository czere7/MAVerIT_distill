"""How long is the initial-writer prompt, in the STUDENT's tokens, for every class the harness
would test -- and which classes does the training budget have to exclude?

Two stages, because they need two environments:

  dump    (harness venv)   build the initial_test_write prompt of every concrete class of
                           every project, exactly as the harness does (_build_prompt: the
                           retriever, then comment stripping), for each retriever asked for
  count   (training venv)  tokenize each prompt with the student's tokenizer, wrapped in the
                           qwen3_5_nothink user turn that LLaMA-Factory trains on

    ~/distil-env/bin/python smokes/prompt_sizes.py dump v1,v5
    ~/maverit-ft/bin/python smokes/prompt_sizes.py count

EXACT COUNTS, NEVER AN ESTIMATE. A chars/token proxy once under-counted by 7.6% at the median
and 3x at worst, and six "safe" examples would have been truncated mid-Java. Here every prompt
goes through the real tokenizer.

THE BUDGET IS PROMPT + TARGET. cutoff_len (24,576) bounds the whole training sequence, and
LLaMA-Factory truncates the END -- the target's closing fence and EOS. So the report gives each
prompt's HEADROOM, what is left for the target, and not only whether the prompt fits.
"""
import argparse
import contextlib
import io
import json
import statistics
import sys
from pathlib import Path

PROJECTS = ["commons-codec-clean", "commons-cli-clean", "commons-csv-clean",
            "jackson-core-clean", "joda-time-clean"]
HELD_OUT = {"commons-cli-clean"}
OUT = Path(__file__).resolve().parents[1] / "logs" / "prompt-sizes"
TOKENIZER = "Qwen/Qwen3.5-2B"
CUTOFF = 24_576
# The qwen3_5_nothink user turn, byte for byte (see the distillation log): the prompt tokens
# a training example actually spends. The assistant side adds the target plus <|im_end|>\n.
USER_TURN = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def dump(arms: list[str]) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import PROJECTS as ROOT
    from Extractor import Extractor
    from utils import config, is_concrete_class
    import retrieval
    from nodes.initial_test_write_node import _build_prompt

    OUT.mkdir(parents=True, exist_ok=True)
    rows = 0
    with (OUT / "prompts.jsonl").open("w", encoding="utf-8") as fh:
        for name in PROJECTS:
            project = ROOT / name
            config["WORKING_DIRECTORY"] = str(project)
            with contextlib.redirect_stdout(io.StringIO()):
                files = Extractor(str(project)).extract_all()
            targets = [f for f in files if is_concrete_class(f)]
            for arm in arms:
                config["RETRIEVER"] = arm
                retrieval._INDEX_CACHE.clear()
                for i, cut in enumerate(targets):
                    with contextlib.redirect_stdout(io.StringIO()):
                        prompt = _build_prompt({"all_files": files, "all_test_files": targets,
                                                "current_class_index": i})
                    rel = Path(cut.file_path).relative_to(project).as_posix()
                    fh.write(json.dumps({"project": name, "class": rel, "arm": arm,
                                         "chars": len(prompt), "prompt": prompt}) + "\n")
                    rows += 1
            print(f"{name}: {len(targets)} classes x {len(arms)} arms")
    print(f"wrote {rows} prompts to {OUT / 'prompts.jsonl'}")


def pct(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def count() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rows = [json.loads(line) for line in (OUT / "prompts.jsonl").open(encoding="utf-8")]
    for r in rows:
        r["tokens"] = len(tok(USER_TURN.format(r.pop("prompt")), add_special_tokens=False)["input_ids"])
        r["headroom"] = CUTOFF - r["tokens"]
    with (OUT / "sizes.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    arms = sorted({r["arm"] for r in rows})
    print(f"tokenizer {TOKENIZER}, cutoff {CUTOFF:,} (prompt + target), user turn included\n")
    print(f"{'project':22} {'arm':4} {'n':>4} {'median':>7} {'p90':>7} {'max':>7} "
          f"{'>cutoff':>8} {'<4k left':>9} {'chars/tok':>9}")
    for name in PROJECTS + ["ALL (training)", "ALL"]:
        for arm in arms:
            sel = [r for r in rows if r["arm"] == arm and (
                r["project"] == name or name == "ALL"
                or (name == "ALL (training)" and r["project"] not in HELD_OUT))]
            if not sel:
                continue
            t = [r["tokens"] for r in sel]
            over = sum(x > CUTOFF for x in t)
            tight = sum(CUTOFF - 4096 < x <= CUTOFF for x in t)
            cpt = sum(r["chars"] for r in sel) / sum(t)
            label = name + (" (held out)" if name in HELD_OUT else "")
            print(f"{label:22} {arm:4} {len(t):>4} {int(statistics.median(t)):>7,} {pct(t, .9):>7,} "
                  f"{max(t):>7,} {over:>8} {tight:>9} {cpt:>9.2f}")
        if name == PROJECTS[-1]:
            print()
    for arm in arms:
        over = sorted((r for r in rows if r["arm"] == arm and r["tokens"] > CUTOFF),
                      key=lambda r: -r["tokens"])
        print(f"\n{arm}: {len(over)} prompts over the cutoff" + (":" if over else ""))
        for r in over:
            print(f"  {r['tokens']:>7,}  {r['project']}  {r['class']}")
    print(f"\nper-class rows in {OUT / 'sizes.jsonl'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dump", "count"])
    ap.add_argument("arms", nargs="?", default="v5", help="dump only: retrievers, comma-separated")
    args = ap.parse_args()
    dump(args.arms.split(",")) if args.stage == "dump" else count()
    return 0


if __name__ == "__main__":
    sys.exit(main())
