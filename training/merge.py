"""Merge a LoRA adapter into standalone weights vLLM can serve. Runs in the TRAINING venv:

    ~/maverit-ft/bin/python -m training.merge --adapter DIR --out DIR

Ported from ~/merge_ckpt.py.

MUST LOAD WITH AutoModelForImageTextToText -- the class LLaMA-Factory trains Qwen3.5 under.
AutoModelForCausalLM gives module paths without `.language_model.`, PEFT matches no saved
key, loads nothing, leaves lora_B at its zero init (an exact identity) and raises NOTHING:
the "merged" model is the base model wearing a fine-tuned name. Hence the sum|lora_B|
assertion before merging.

THE PROCESSOR IS SAVED TOO. Qwen3.5 is multimodal and vLLM refuses a model directory with no
image processor, even for text-only serving. The old merge saved only the tokenizer and the
processor configs were copied in by hand afterwards; forgetting that once cost two eval arms
a full night.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def merge(adapter: Path, out: Path, base: str = "Qwen/Qwen3.5-2B") -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    model = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa")
    peft_model = PeftModel.from_pretrained(model, str(adapter))
    loaded = sum(float(module.lora_B[name].weight.detach().float().abs().sum())
                 for _, module in peft_model.named_modules()
                 if getattr(module, "lora_B", None) is not None
                 for name in module.lora_B)
    if loaded == 0.0:
        raise RuntimeError("adapter loaded NOTHING -- every lora_B is at its zero init. Module "
                           "paths do not match the saved keys; check the model class.")
    print(f"adapter loaded: sum|lora_B| = {loaded:.1f}")

    merged = peft_model.merge_and_unload()
    if out.exists():
        shutil.rmtree(out)
    merged.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(str(out))
    AutoProcessor.from_pretrained(base).save_pretrained(str(out))
    # transformers 5 writes the processor as one processor_config.json; the image processor's
    # own preprocessor_config.json comes from the base model, as in every merge that served.
    shutil.copy(hf_hub_download(base, "preprocessor_config.json"), out / "preprocessor_config.json")
    for required in ("preprocessor_config.json", "tokenizer_config.json", "config.json"):
        if not (out / required).exists():
            raise RuntimeError(f"merged model is missing {required}; vLLM will refuse it")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9
    print(f"wrote {out} ({size:.1f} GB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base", default="Qwen/Qwen3.5-2B")
    args = parser.parse_args()
    merge(args.adapter, args.out, args.base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
