"""SFT through LLaMA-Factory: render a config, run `llamafactory.cli train` in the training venv.

LLaMA-Factory stays the trainer for every supervised step. It is what produced ck60 and all
four RFT rounds, and switching trainers would put two changes into the next round's numbers.
It gives us, already verified: the qwen3_5_nothink template byte for byte with the prompt
masked out of the loss, Liger's fused cross-entropy, loading through
AutoModelForImageTextToText (the class PEFT needs for Qwen3.5's module paths), and
continuing an existing adapter.

Two recipes, copied from the configs that produced the results:

  cold_start   run3_fenced_r16.yaml: a new rank-16 adapter on the base model, lr 1e-4
  rft_round    sel_v1.yaml (round 4): continue the cold-start adapter, lr 2e-5, 3 epochs.
               Every round restarts from the COLD-START adapter on the accumulated set --
               STaR re-fits from base; continuing the previous round compounds its drift.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

TRAIN_PYTHON = Path(os.environ.get("TRAIN_PYTHON", Path.home() / "maverit-ft" / "bin" / "python"))
BASE_MODEL = "Qwen/Qwen3.5-2B"
CUTOFF_LEN = 24_576

COMMON = {
    "model_name_or_path": BASE_MODEL,
    "trust_remote_code": True,
    # MANDATORY at a 248,320-token vocabulary: logits at 24k context would be a 24 GB fp32
    # tensor. Liger's fused linear cross-entropy never builds it.
    "enable_liger_kernel": True,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target": "all",                  # LLaMA-Factory's spelling; "all-linear" fails
    "template": "qwen3_5_nothink",         # targets are fenced Java; no <think> block
    # Over-long examples are DROPPED when the dataset is written, never truncated here:
    # LLaMA-Factory cuts the END, which is the closing fence and EOS.
    "cutoff_len": CUTOFF_LEN,
    "overwrite_cache": True,
    "preprocessing_num_workers": 4,
    "logging_steps": 1,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "report_to": "none",
    "per_device_train_batch_size": 1,      # at 24k context one sequence is the batch
    "gradient_accumulation_steps": 4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "bf16": True,
    "gradient_checkpointing": True,
    "flash_attn": "sdpa",
    # MUST stay off: Liger gates its fused path on self.training, so evaluation
    # materialises the full logits. Run 1 died at exactly the first eval step.
    "eval_strategy": "no",
}

RECIPES = {
    "cold_start": {"learning_rate": 1.0e-4, "num_train_epochs": 2.0, "save_steps": 20},
    "rft_round": {"learning_rate": 2.0e-5, "num_train_epochs": 3.0, "save_steps": 15,
                  "create_new_adapter": False},
}


def render(recipe: str, dataset_dir: Path, dataset: str, output_dir: Path,
           adapter: Path | None = None) -> dict:
    cfg = {**COMMON, **RECIPES[recipe], "dataset_dir": str(dataset_dir), "dataset": dataset,
           "output_dir": str(output_dir)}
    if recipe == "rft_round":
        if adapter is None:
            raise ValueError("an RFT round continues the cold-start adapter; pass `adapter`")
        cfg["adapter_name_or_path"] = str(adapter)
    return cfg


def train(cfg: dict, config_path: Path, log_path: Path) -> None:
    """Write the config next to the run and train. Raises on a failed run, with the tail."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    env = {**os.environ,
           # Sequence lengths vary 1k-24k; the default allocator fragments badly enough to
           # OOM mid-run, and on WDDM a spill to host RAM is silent. This makes it an OOM.
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "HF_HOME": os.environ.get("HF_HOME", "/mnt/c/Users/akosc/.cache/huggingface")}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([str(TRAIN_PYTHON), "-m", "llamafactory.cli", "train",
                               str(config_path)], stdout=log, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"training failed (exit {proc.returncode}):\n{tail}")
