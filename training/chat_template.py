"""Qwen3.5 chat-template slots, applied by hand.

DO NOT replace this with `tokenizer.apply_chat_template`. That injects
`<think>\\n\\n</think>\\n\\n`, which training never saw -- every number measured through it
would be measured on an input the model was not trained on, and the checkpoint ranking can
invert as a result.

Copied verbatim from LLaMA-Factory's `qwen3_5_nothink` template (data/template.py):

    format_user      = "<|im_start|>user\\n{{content}}<|im_end|>\\n<|im_start|>assistant\\n"
    format_assistant = "{{content}}<|im_end|>\\n"

`replace_eos=True` makes <|im_end|> the EOS token. There is no <think> block -- that is what
separates qwen3_5_nothink from qwen3_5, which is a ReasoningTemplate.

The same slots appear in eval_checkpoints.py and termination_sweep.py from the SFT phase;
this is the canonical copy.
"""

from __future__ import annotations

USER_FMT = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
ASSISTANT_FMT = "{content}<|im_end|>\n"

EOS_TOKEN = "<|im_end|>"


def format_prompt(instruction: str) -> str:
    """Prompt text ready for a raw completions call.

    Used with /v1/completions rather than /v1/chat/completions on purpose: the chat
    endpoint would apply the server's own template on top, reintroducing exactly the
    mismatch this module exists to avoid.
    """
    return USER_FMT.format(content=instruction)


def format_target(output: str) -> str:
    """Training-target form, terminated by the EOS the model must learn to emit."""
    return ASSISTANT_FMT.format(content=output)
