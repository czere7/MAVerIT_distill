"""Sample N rollouts per prompt from a vLLM server. Ported from middleband_generate.py.

One request per prompt with `n` completions: vLLM shares the prefill across them, which is
most of the cost -- eight separate requests would pay it eight times.

RAW COMPLETIONS, NOT CHAT. The prompt goes through chat_template.format_prompt and the
/v1/completions endpoint, because the chat endpoint would apply the server's own template
on top -- the tokenizer's template injects a <think> block training never saw.

TOP_K IS NOT OPTIONAL at tau 1.2. With nucleus sampling alone, 73% of rollouts failed to
parse, and 81% of those carried CJK, Cyrillic or Arabic characters inside identifiers: at
that temperature a 0.95 nucleus admits thousands of candidates from Qwen's multilingual
tail. A top_k cut excludes it; nucleus sampling cannot.

Resumable: rows are appended as they arrive and prompts already sampled are skipped,
because WSL2 has restarted unprompted more than once.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from training.chat_template import EOS_TOKEN, format_prompt

N = 8
TEMPERATURE = 1.2
TOP_P = 0.95
TOP_K = 40
SEED = 1234


def sample(url: str, model: str, instruction: str, max_tokens: int, n: int = N,
           temperature: float = TEMPERATURE, top_p: float = TOP_P, top_k: int = TOP_K,
           seed: int = SEED, timeout: int = 3600) -> dict:
    body = json.dumps({
        "model": model, "prompt": format_prompt(instruction), "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p, "top_k": top_k, "n": n, "seed": seed,
        "stop": [EOS_TOKEN],
    }).encode("utf-8")
    request = urllib.request.Request(f"{url.rstrip('/')}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def generate(prompts: list[dict], output: Path, url: str, model: str, n: int = N,
             temperature: float = TEMPERATURE, top_p: float = TOP_P, top_k: int = TOP_K,
             seed: int = SEED) -> int:
    """Append n rollouts per prompt to `output`; returns how many prompts failed.

    Each prompt row needs class_key, project, prompt and max_tokens -- the per-prompt cap,
    which the prompt set derives from the reference suite's length.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["class_key"] for l in output.open(encoding="utf-8")} \
        if output.exists() else set()
    todo = [p for p in prompts if p["class_key"] not in done]
    print(f"prompts {len(prompts)}, already sampled {len(done)}, to sample {len(todo)}; "
          f"n={n} tau={temperature} top_p={top_p} top_k={top_k} model={model}", flush=True)

    failed = 0
    started = time.monotonic()
    with output.open("a", encoding="utf-8") as handle:
        for i, row in enumerate(todo, start=1):
            case_started = time.monotonic()
            try:
                payload = sample(url, model, row["prompt"], row["max_tokens"], n,
                                 temperature, top_p, top_k, seed)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                failed += 1
                print(f"[{i:3d}/{len(todo)}] ERR  {row['class_key']}: {error}", flush=True)
                continue
            elapsed = time.monotonic() - case_started
            choices = payload.get("choices", [])
            for k, choice in enumerate(choices):
                handle.write(json.dumps({
                    "class_key": row["class_key"], "project": row["project"], "rollout": k,
                    "text": choice.get("text", ""),
                    "finish_reason": choice.get("finish_reason", ""),
                    "chars": len(choice.get("text", "")), "max_tokens": row["max_tokens"],
                    "temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed,
                    "request_seconds": round(elapsed, 1),
                    "sampled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }) + "\n")
            handle.flush()
            stops = sum(1 for c in choices if c.get("finish_reason") == "stop")
            rate = (time.monotonic() - started) / i
            print(f"[{i:3d}/{len(todo)}] {row['class_key'].split('/')[-1][:28]:<28} "
                  f"{len(choices)} rollouts, {stops} terminated, {elapsed:.0f}s "
                  f"(eta {rate * (len(todo) - i) / 60:.0f}m)", flush=True)
    return failed
