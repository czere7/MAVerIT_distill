# Environments

Three virtualenvs, because their pins conflict: training needs torch 2.11+cu128, vLLM 0.28
needs torch 2.13+cu132, and the harness and verifier need no torch at all. The loop driver
runs in the harness venv and shells out to the other two.

| venv | role | built from |
|---|---|---|
| `~/distil-env` | harness, verifier, loop driver (Python 3.10) | `requirements.txt` at the repo root |
| `~/maverit-ft` | LLaMA-Factory training, merge, token counts (Python 3.12) | `train.lock.txt` |
| `~/vllm-env` | serving the student for rollouts (Python 3.12) | `vllm.lock.txt` |

Machine: RTX 3090, NVIDIA driver 591.86 (Windows host), WSL2 Ubuntu 22.04, JDK 17
(jackson-core targets 17), Maven 3.9.15. Build all three with `uv`
(`~/.local/bin/uv pip install --python <venv>/bin/python -r <file>`).

## Traps, each of which has cost a run

**Training venv (`~/maverit-ft`)**

- `pip install -e .` on LLaMA-Factory silently replaces a CUDA torch with a CPU build.
  Install torch from the cu128 index *after* LLaMA-Factory and check
  `torch.__version__` has no `+cpu`. A CPU torch does not fail: it trains on the CPU at
  zero steps per 18 minutes.
- LLaMA-Factory is an editable install of `~/LLaMA-Factory`, clean upstream at commit
  `a61cfa6`.
- `torchvision` must be the matching cu128 build, or `import fla` fails through a missing
  `torchvision::nms` operator.
- `flash-linear-attention` **and** `causal-conv1d` are both required: transformers gates the
  GatedDeltaNet fast path on both, and with only one, memory looks right while throughput
  is 2-8x too slow. causal-conv1d is the prebuilt `cu12torch2.10` wheel in the lock file.
- `sudo apt install python3.12-dev`: Triton JIT-compiles a C shim at runtime and fails
  mid-training without `Python.h`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for every run (llamafactory.py sets
  it). Without it variable-length batches fragment the allocator, and on WDDM an overflow
  spills silently to host RAM at ~200 W instead of raising OOM.
- Liger's fused cross-entropy is mandatory (248,320-token vocabulary) and eval during
  training must stay off (Liger materialises full logits outside `self.training`).

**vLLM venv (`~/vllm-env`)**

- `CUDA_HOME` and `PATH` must point at the venv's own `site-packages/nvidia/cu13`, not
  `/usr/bin/nvcc` (CUDA 11.5): flashinfer JIT-compiles and CCCL refuses CUDA < 12.
  `vllm_server.py` sets this.
- `VLLM_USE_FLASHINFER_SAMPLER=0` avoids a JIT build that hits a compiler/header skew.
- `vllm==0.28.0` must stay pinned; an unpinned dependency lets the resolver backtrack to
  an ancient release and build it from source.
- The editable `vllm-gguf-plugin` in the lock file is for serving Qwen3.8-27B GGUF and is
  not needed for the student; it is not part of this repository.
- Stop vLLM by port and then by GPU PID (`vllm_server.stop`): the engine process renames
  itself `VLLM::EngineCore`, survives a kill by name, and keeps its VRAM.

**Everything**

- `HF_HOME=/mnt/c/Users/akosc/.cache/huggingface` reuses the Windows download of
  `Qwen/Qwen3.5-2B`.
- A merged model must be saved with its processor (`merge.py` does); vLLM refuses a Qwen3.5
  directory without an image processor even for text.
