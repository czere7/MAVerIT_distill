import json
from pathlib import Path
from time import time
from typing import Sequence, Any

import openai
from langchain_core.messages import message_to_dict
from langchain_core.prompt_values import PromptValue
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from utils import config, ensure_file, run_dir

# The levels both Ollama's `think` and OpenAI's `reasoning_effort` accept, plus "off".
# Only gpt-oss honours the levels on Ollama; Qwen3.8 treats every level as plain "on".
REASONING_EFFORTS = ("off", "low", "medium", "high")


def reasoning_effort() -> str:
    effort = config.get("REASONING_EFFORT") or "high"
    if effort not in REASONING_EFFORTS:
        raise RuntimeError(f"REASONING_EFFORT must be one of {', '.join(REASONING_EFFORTS)}, "
                           f"got {effort!r}")
    return effort


def _is_empty(response) -> bool:
    """No answer at all: no text and no tool calls (a tool-calling reply has no text)."""
    if getattr(response, "tool_calls", None):
        return False
    content = response.content
    if isinstance(content, list):
        content = "".join(p if isinstance(p, str) else str(p.get("text", "")) for p in content)
    return not str(content).strip()


class ModelWrapper:
    instance = None
    model = None
    time_out_seconds = 60

    def __init__(self):
        self.generation_model_name = config.get("MODEL")
        if not self.generation_model_name:
            raise RuntimeError("MODEL is not set in .env")
        self._set_model()

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def invoke(self, prompt_input: PromptValue | str | Sequence[Any], run_id: str, node_name: str,
               tools: list[dict] | None = None):
        """`tools`: OpenAI-style function schemas. The model may answer with tool calls; the
        caller executes them (see edit_tools.py) -- nothing is executed here."""
        time_stamp = round(time() * 1000)
        path = run_dir(run_id) / "prompt-response-pairs.jsonl"
        ensure_file(path)
        while self.time_out_seconds < int(config.get("MAX_LLM_TIMEOUT", 60)):
            try:
                model = self.model.bind_tools(tools) if tools else self.model
                response = model.invoke(prompt_input)
                self._log_call(path, time_stamp, node_name, prompt_input, response)
                if _is_empty(response) and self._can_retry_without_thinking():
                    # A thinking stall: the model stopped inside its thinking and never
                    # answered (done_reason "stop", not "length"). Thinking off cannot
                    # stall that way, so retry the same prompt once without it.
                    usage = response.usage_metadata or {}
                    finish = (response.response_metadata or {}).get("finish_reason") or \
                        (response.response_metadata or {}).get("done_reason")
                    print(f"[{node_name}] Empty answer ({finish}) after "
                          f"{usage.get('output_tokens', '?')} output tokens. "
                          f"Retrying once with thinking off.")
                    stalled = response
                    retry_model = self._no_thinking_model()
                    if tools:
                        retry_model = retry_model.bind_tools(tools)
                    response = retry_model.invoke(prompt_input)
                    self._log_call(path, time_stamp, f"{node_name}:no_thinking_retry",
                                   prompt_input, response)
                    # Nodes add usage_metadata to the run's totals; the stall was paid for.
                    if stalled.usage_metadata and response.usage_metadata:
                        for key in ("input_tokens", "output_tokens", "total_tokens"):
                            response.usage_metadata[key] += stalled.usage_metadata[key]
                return response
            except openai.APIConnectionError:
                self._handle_timeout()
        raise TimeoutError("Timeout exceeded maximum allowed value")

    @staticmethod
    def _log_call(path, time_stamp, node_name, prompt_input, response):
        if isinstance(prompt_input, (list, tuple)):              # a multi-turn tool conversation
            prompt_input = [m if isinstance(m, (str, dict)) else message_to_dict(m) for m in prompt_input]
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({
                "time_stamp": time_stamp,
                "node_name": node_name,
                "prompt": prompt_input,
                "response": message_to_dict(response),
            })+"\n")

    def _can_retry_without_thinking(self) -> bool:
        return config.get("PROVIDER") in ("ollama", "deepseek") and reasoning_effort() != "off"

    def _no_thinking_model(self):
        if config.get("PROVIDER") == "deepseek":
            return self._deepseek(thinking=False)
        return ChatOllama(model=self.generation_model_name, reasoning=False)

    def _deepseek(self, thinking: bool) -> ChatOpenAI:
        # DeepSeek ignores `enable_thinking` and `reasoning_effort`; only this switch works
        # (probed on deepseek-v4-flash: reasoning 25 tokens -> none).
        return ChatOpenAI(
            model=self.generation_model_name,
            api_key=config['DEEPSEEK_API_KEY'],
            base_url="https://api.deepseek.com",
            timeout=self.time_out_seconds,
            **({} if thinking else {"extra_body": {"thinking": {"type": "disabled"}}}),
        )

    def _set_model(self):
        if config.get("PROVIDER") == "ollama":
            self.model = ChatOllama(
                model=self.generation_model_name,
                # A level still puts the thinking in reasoning_content, as True did.
                reasoning=False if reasoning_effort() == "off" else reasoning_effort(),
            )
        elif config.get("PROVIDER") == "deepseek":
            if not config['DEEPSEEK_API_KEY']:
                raise RuntimeError("DEEPSEEK_API_KEY is not set in .env")
            # DeepSeek has no effort levels that work: only on/off (see _deepseek).
            self.model = self._deepseek(thinking=reasoning_effort() != "off")
        elif config.get("PROVIDER") == "openai":
            if not config['OPENAI_API_KEY']:
                raise RuntimeError("OPENAI_API_KEY is not set in .env")
            self.model = ChatOpenAI(
                model=self.generation_model_name,
                api_key=config['OPENAI_API_KEY'],
                organization="org-ENWuCLTuWsptJcXQVIR6Exwa",
                base_url="https://api.openai.com/v1",
                reasoning_effort="none" if reasoning_effort() == "off" else reasoning_effort(),
                timeout=self.time_out_seconds,
            )
        elif config.get("PROVIDER") == "lite_llm":
            if not config['LITE_LLM_API_KEY']:
                raise RuntimeError("LITE_LLM_API_KEY is not set in .env")
            self.model = ChatOpenAI(
                model=self.generation_model_name,
                api_key=config['LITE_LLM_API_KEY'],
                base_url=config['LITE_LLM_BASE_URL'],
                timeout=self.time_out_seconds,
            )
        else:
            raise NotImplementedError(
                "The specified model provider is not supported: {}".format(config.get("PROVIDER")))

    def _handle_timeout(self):
        print(f"[TIMEOUT]: timeout value of {self.time_out_seconds} seconds has been exceeded.")
        self.time_out_seconds += 60
        self._set_model()
