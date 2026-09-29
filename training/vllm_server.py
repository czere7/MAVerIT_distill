"""Start and stop the vLLM server that serves a merged checkpoint for rollout sampling.

Environment, every line learned the hard way (see ~/setup_vllm.sh):
  - CUDA_HOME and PATH must point at the venv's own CUDA 13 toolkit, NOT /usr/bin/nvcc
    (11.5): flashinfer JIT-compiles kernels and CCCL hard-errors on CUDA < 12.
  - the venv's bin must be on PATH so flashinfer finds ninja.
  - VLLM_USE_FLASHINFER_SAMPLER=0 avoids a JIT build that hits a compiler/header skew.

STOPPING IS THE DANGEROUS PART. vLLM renames its engine subprocess to VLLM::EngineCore, so
killing the server by name misses it; the engine survives holding 7-20 GB of VRAM and the
next launch stacks on top of it, silently. So stop() kills the process listening on the
port, then every process still holding the GPU, and reports what VRAM is left.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from training.gpu import unload_ollama

VLLM_ENV = Path(os.environ.get("VLLM_ENV", Path.home() / "vllm-env"))
PORT = int(os.environ.get("VLLM_PORT", "8002"))
URL = f"http://127.0.0.1:{PORT}"
MAX_MODEL_LEN = 32_768
GPU_MEMORY_UTILIZATION = 0.85


def _env() -> dict:
    site = subprocess.run([str(VLLM_ENV / "bin" / "python"), "-c",
                           "import site; print(site.getsitepackages()[0])"],
                          capture_output=True, text=True, check=True).stdout.strip()
    cuda_home = Path(site) / "nvidia" / "cu13"
    return {**os.environ, "CUDA_HOME": str(cuda_home),
            "PATH": f"{cuda_home / 'bin'}:{VLLM_ENV / 'bin'}:{os.environ['PATH']}",
            "HF_HOME": os.environ.get("HF_HOME", "/mnt/c/Users/akosc/.cache/huggingface"),
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/v1/models", timeout=3):
            return True
    except OSError:
        return False


def served_models() -> list[str]:
    import json
    with urllib.request.urlopen(f"{URL}/v1/models", timeout=5) as response:
        return [m["id"] for m in json.loads(response.read())["data"]]


def start(model_dir: Path, name: str, log_path: Path, wait_s: int = 900, attempts: int = 2) -> None:
    """Serve `model_dir` as `name`; returns once /v1/models answers with that name.

    A server process that EXITS during startup is caught at once, not after `wait_s`: the
    first cold-start eval died two seconds in on a native heap abort ("corrupted
    double-linked list") and the old loop waited fifteen minutes for a port nothing would
    ever open. Such a crash gets one more attempt before the step fails.
    """
    for attempt in range(1, attempts + 1):
        stop()
        unload_ollama()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([str(VLLM_ENV / "bin" / "vllm"), "serve", str(model_dir),
                                     "--port", str(PORT), "--host", "127.0.0.1",
                                     "--max-model-len", str(MAX_MODEL_LEN),
                                     "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
                                     "--served-model-name", name],
                                    stdout=log, stderr=subprocess.STDOUT, env=_env(),
                                    start_new_session=True)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(10)
            if healthy():
                # A leftover server answers happily with the wrong weights; check the name.
                if name not in served_models():
                    raise RuntimeError(f"port {PORT} serves {served_models()}, not {name}")
                return
        if proc.poll() is None:
            raise RuntimeError(f"vLLM did not come up in {wait_s}s; see {log_path}")
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
        print(f"vLLM exited during startup (code {proc.returncode}, attempt {attempt}/{attempts}): "
              f"{' | '.join(tail)}", flush=True)
    raise RuntimeError(f"vLLM exited during startup {attempts} times; see {log_path}")


def _gpu_pids() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if p.strip().isdigit()]


def _port_pid() -> int | None:
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{PORT} " in line:
            m = re.search(r"pid=(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def stop() -> str:
    pid = _port_pid()
    if pid is not None:
        subprocess.run(["kill", str(pid)])
        time.sleep(15)
    for gpu_pid in _gpu_pids():          # EngineCore survives a plain kill of the server
        subprocess.run(["kill", "-9", str(gpu_pid)])
    time.sleep(5)
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
