"""Smoke test: does REASONING_EFFORT change how long Qwen3.8 thinks?

Replays one logged repair prompt through the harness's own ModelWrapper at each level and
records thinking and answer length. gpt-oss:20b is the control: langchain-ollama documents
the levels as gpt-oss only, so if the control's thinking scales and Qwen's does not, Qwen is
treating every level as plain "on".

Run from the harness root:  python smokes/smoke_reasoning_effort.py [--reps N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import REASONING_EFFORTS, ModelWrapper  # noqa: E402
from utils import config  # noqa: E402

SOURCE = Path("logs/qwen38-Metaphone/prompt-response-pairs.jsonl")
CALL = 2                                   # a test_repair_node call: 24.5k chars in
RUN_ID = "smokes/reasoning-effort"
OUT = Path(RUN_ID) / "results.jsonl"


def logged_prompt():
    line = SOURCE.read_text(encoding="utf-8").splitlines()[CALL]
    call = json.loads(line)
    assert call["node_name"] == "test_repair_node", call["node_name"]
    return call["prompt"]


def one_call(model, effort, prompt):
    config["PROVIDER"], config["MODEL"], config["REASONING_EFFORT"] = "ollama", model, effort
    wrapper = ModelWrapper()
    wrapper._set_model()                   # a singleton: rebuild with the new settings
    start = time.time()
    response = wrapper.invoke(prompt, RUN_ID, f"smoke:{model}:{effort}")
    md = response.response_metadata or {}
    return {
        "model": model,
        "effort": effort,
        "seconds": round(time.time() - start, 1),
        "eval_count": md.get("eval_count"),
        "done_reason": md.get("done_reason"),
        "thinking_chars": len(response.additional_kwargs.get("reasoning_content", "")),
        "answer_chars": len(response.content or ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2, help="calls per Qwen level")
    args = parser.parse_args()

    prompt = logged_prompt()
    plan = [("gpt-oss:20b", e) for e in REASONING_EFFORTS]
    plan += [("qwen3.8:latest", e) for _ in range(args.reps) for e in REASONING_EFFORTS]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"{'model':<16}{'effort':<8}{'tokens':>8}{'think':>8}{'answer':>8}{'secs':>7}  done")
    for model, effort in plan:
        try:
            row = one_call(model, effort, prompt)
        except Exception as error:         # an unsupported level should show, not abort
            row = {"model": model, "effort": effort, "error": repr(error)[:300]}
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if "error" in row:
            print(f"{model:<16}{effort:<8}ERROR {row['error']}")
        else:
            print(f"{model:<16}{effort:<8}{row['eval_count']:>8}{row['thinking_chars']:>8}"
                  f"{row['answer_chars']:>8}{row['seconds']:>7}  {row['done_reason']}"
                  + ("   <-- EMPTY ANSWER" if not row["answer_chars"] else ""), flush=True)


if __name__ == "__main__":
    main()
