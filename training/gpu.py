"""Free the one GPU before training or serving.

With an Ollama teacher (smoke runs), the harness and the student share the 3090: Ollama on
the Windows host keeps a model resident for five minutes after its last call, and a vLLM
start at 0.85 utilisation or a 24k-token training step on top of a 13 GB gpt-oss would spill
into host RAM over WDDM -- silently slow, not an OOM. So every GPU step first asks Ollama to
unload whatever it holds. With DeepSeek as the teacher there is nothing to unload and this
is a no-op.
"""

from __future__ import annotations

import json
import os
import urllib.request

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def unload_ollama() -> list[str]:
    """Unload every model Ollama has resident; returns their names. Silent if unreachable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=3) as response:
            loaded = [m["name"] for m in json.loads(response.read()).get("models", [])]
    except OSError:
        return []
    for name in loaded:
        body = json.dumps({"model": name, "keep_alive": 0}).encode("utf-8")
        request = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60):
            pass
    if loaded:
        print(f"unloaded from Ollama: {', '.join(loaded)}", flush=True)
    return loaded
