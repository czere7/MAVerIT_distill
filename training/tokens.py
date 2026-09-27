"""Exact token counts in the student's tokenizer. Runs in the TRAINING venv:

    ~/maverit-ft/bin/python -m training.tokens IN.jsonl OUT.jsonl

The harness venv has no transformers, so callers write rows to a file and run this as a
subprocess (see count()). Each row gets `prompt_tokens` -- the prompt inside the
qwen3_5_nothink user turn -- and, if it has an `output`, `output_tokens` -- the target plus
the <|im_end|> the model must learn to emit. Together they are what a training example
spends against cutoff_len.

EXACT, NEVER ESTIMATED. A chars/token proxy once under-counted by 7.6% at the median and 3x
at worst, and six "safe" examples would have been truncated mid-Java.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from training.chat_template import ASSISTANT_FMT, USER_FMT

TOKENIZER = "Qwen/Qwen3.5-2B"
TRAIN_PYTHON = Path(os.environ.get("TRAIN_PYTHON", Path.home() / "maverit-ft" / "bin" / "python"))
HF_HOME = os.environ.get("HF_HOME", "/mnt/c/Users/akosc/.cache/huggingface")


def count(rows: list[dict], scratch: Path) -> list[dict]:
    """Count from the harness venv by shelling out to the training venv."""
    scratch.mkdir(parents=True, exist_ok=True)
    src, dst = scratch / "tokens_in.jsonl", scratch / "tokens_out.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    subprocess.run([str(TRAIN_PYTHON), "-m", "training.tokens", str(src), str(dst)],
                   check=True, env={**os.environ, "HF_HOME": HF_HOME},
                   cwd=Path(__file__).resolve().parents[1])
    return [json.loads(l) for l in dst.open(encoding="utf-8")]


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            row["prompt_tokens"] = len(tok(USER_FMT.format(content=row["prompt"]),
                                           add_special_tokens=False)["input_ids"])
            if "output" in row:
                row["output_tokens"] = len(tok(ASSISTANT_FMT.format(content=row["output"]),
                                               add_special_tokens=False)["input_ids"])
            fout.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
