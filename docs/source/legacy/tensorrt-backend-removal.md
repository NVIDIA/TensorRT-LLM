# Migration Guide: TensorRT Backend Removed

```{note}
**Breaking change.** The TensorRT engine backend has been **removed** in this
release. PyTorch is now the sole execution backend for TensorRT LLM. AutoDeploy
(built on the PyTorch backend) remains available.
```

This guide explains what was removed, the errors you will encounter if you rely
on the old TensorRT path, and how to migrate to the PyTorch backend.

## What Was Removed

The following public APIs, command-line tools, and packaging behaviors are no
longer available:

- **`LLM(backend="tensorrt")`** — passing `backend="tensorrt"` to the `LLM`
  constructor now raises a `ValueError`. PyTorch is the only supported backend.
- **`TrtLlmArgs`** — the TensorRT-specific arguments class has been removed.
  Use `TorchLlmArgs` (the default) instead.
- **`tensorrt_llm._tensorrt_engine.LLM`** — the TensorRT engine `LLM` entry
  point has been removed.
- **CLI build/engine tooling** — `trtllm-build`, `trtllm-refit`, and
  `trtllm-prune` have been removed. The PyTorch backend loads HuggingFace
  checkpoints directly and does not require a separate engine-build step.
- **`--backend tensorrt`** — the `tensorrt` choice has been removed from CLI
  tools (for example `trtllm-serve`, `trtllm-bench`, `trtllm-eval`). Omit the
  flag (PyTorch is the default) or pass `--backend pytorch`.
- **`tensorrt` pip dependency** — the `tensorrt` Python package is no longer a
  dependency of TensorRT LLM and is not installed.
- **Checkpoint conversion scripts** — the per-model `convert_checkpoint.py`
  scripts and the associated TensorRT example directories have been removed.

## Breaking-Change Contract

| Removed surface | Previous behavior | New behavior |
| --- | --- | --- |
| `LLM(backend="tensorrt")` | Built and ran a TensorRT engine | Raises `ValueError` |
| `TrtLlmArgs` | TensorRT argument schema | Removed; use `TorchLlmArgs` |
| `tensorrt_llm._tensorrt_engine.LLM` | TensorRT engine entry point | Removed |
| `trtllm-build` / `trtllm-refit` / `trtllm-prune` | Engine build / refit / prune CLIs | Removed |
| `--backend tensorrt` | Selected the TensorRT backend | Removed choice; use `pytorch` |
| `tensorrt` pip dependency | Installed with TensorRT LLM | Dropped |
| `convert_checkpoint.py` | Converted HF weights to TRT checkpoints | Removed |

## Examples and Documentation

The per-model TensorRT example directories (the `convert_checkpoint.py` /
`trtllm-build` workflow under `examples/models/`) have been removed. Examples
and docs that cover the **PyTorch backend** are retained:

- PyTorch model usage now lives with the LLM Python API examples under
  [`examples/llm-api/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llm-api)
  (for example `quickstart_advanced.py` and `quickstart_multimodal.py`).
- Per-model READMEs that documented PyTorch-backend usage (for example
  Qwen3/Qwen3-Next, Gemma, Phi-4-multimodal, Nemotron-NAS) have been
  trimmed to keep only their PyTorch sections. Models that already have a
  dedicated deployment walkthrough (for example Llama-3.3-70B) point to the
  [deployment guides](../deployment-guide/) instead.
- Model-specific deployment walkthroughs remain under
  [deployment guides](../deployment-guide/).
- The legacy TensorRT documentation tree under `docs/source/legacy/` has been
  removed. Content that still applies to the PyTorch backend was relocated —
  for example the
  [Low-Precision AllReduce](../features/lowprecision-allreduce.md) guide now
  lives under Features.

## How to Migrate

### Python API

Before:

```python
from tensorrt_llm import LLM

llm = LLM(model="<hf_model>", backend="tensorrt")
```

After (PyTorch is the default backend — no `backend` argument needed):

```python
from tensorrt_llm import LLM

llm = LLM(model="<hf_model>")
```

The PyTorch backend loads HuggingFace checkpoints directly. There is no separate
checkpoint-conversion or engine-build step, so `convert_checkpoint.py` and
`trtllm-build` are no longer part of the workflow.

### Command Line

Before:

```bash
trtllm-build --checkpoint_dir <ckpt> --output_dir <engine>
trtllm-serve <engine> --backend tensorrt
```

After:

```bash
trtllm-serve <hf_model>
```

PyTorch is the default backend, so `--backend` may be omitted. To be explicit,
pass `--backend pytorch`.

## Where to Go Next

- [Quick Start Guide](../quick-start-guide.md)
- [LLM API Reference](../llm-api/index.md)
- [Supported Models](../models/supported-models.md)
- [PyTorch Backend Architecture](../torch/arch_overview.md)
