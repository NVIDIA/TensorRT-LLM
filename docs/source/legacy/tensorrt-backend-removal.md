# Migration Guide: TensorRT Backend Removed

```{note}
**Breaking change.** The TensorRT engine backend has been removed. PyTorch is now
the sole execution backend for TensorRT LLM (AutoDeploy, built on the PyTorch
backend, remains available).
```

## What changed

| Removed | Replacement / new behavior |
| --- | --- |
| `LLM(backend="tensorrt")` | Raises `ValueError` — PyTorch is the only backend; omit `backend` |
| `TrtLlmArgs` | Use `TorchLlmArgs` (the default) |
| `tensorrt_llm._tensorrt_engine.LLM` | Use `tensorrt_llm.LLM` |
| `trtllm-build` / `trtllm-refit` / `trtllm-prune` | No engine-build step — HuggingFace checkpoints load directly |
| Per-model `convert_checkpoint.py` | Not needed — no checkpoint conversion |
| `--backend tensorrt` (CLI) | Omit, or pass `--backend pytorch` |
| `tensorrt` pip dependency | Dropped (no longer installed) |

## How to migrate

Python API — PyTorch is the default, so drop the `backend` argument:

```python
from tensorrt_llm import LLM

llm = LLM(model="<hf_model>")
```

Command line — `--backend pytorch` is the default, so no engine is needed:

```bash
trtllm-serve <hf_model>
```

There is no separate checkpoint-conversion or engine-build step.

## Examples

PyTorch usage lives with the [LLM API examples](../examples/index.rst) (for example
`quickstart_advanced.py`, `quickstart_multimodal.py`) and the model-specific
[deployment guides](../deployment-guide/). The per-model `convert_checkpoint.py` /
`trtllm-build` example directories have been removed.

## Retained reference docs

The legacy tree under `docs/source/legacy/` is kept for **cross-reference only**.
Pages that are purely about the TensorRT engine mechanism carry a caution banner;
the rest document concepts still relevant to the PyTorch backend.

- **Features & concepts:** [attention](advanced/gpt-attention.md),
  [KV cache management](advanced/kv-cache-management.md) /
  [reuse](advanced/kv-cache-reuse.md),
  [speculative decoding](advanced/speculative-decoding.md),
  [expert parallelism](advanced/expert-parallelism.md),
  [disaggregated serving](advanced/disaggregated-service.md),
  [low-precision allreduce](advanced/lowprecision-pcie-allreduce.md),
  [executor API](advanced/executor.md).
- **Performance & tuning:** [analysis](performance/perf-analysis.md),
  [benchmarking](performance/perf-benchmarking.md),
  [tuning guide](performance/performance-tuning-guide/index.rst).
- **Architecture:** [adding a model](architecture/add-model.md),
  [checkpoint format](architecture/checkpoint.md),
  [conversion workflow](architecture/workflow.md),
  [weights loader](architecture/model-weights-loader.md).
- **Reference:** [precision](reference/precision.md),
  [memory](reference/memory.md),
  [multimodal support matrix](reference/multimodal-feature-support-matrix.md),
  [troubleshooting](reference/troubleshooting.md),
  [key features](key-features.md), [PyTorch overview](torch.md).
- **Dev environment:** [build a Docker image](dev-on-cloud/build-image-to-dockerhub.md),
  [develop on Runpod](dev-on-cloud/dev-on-runpod.md).

## Where to go next

- [Quick Start Guide](../quick-start-guide.md)
- [LLM API Reference](../llm-api/index.md)
- [Supported Models](../models/supported-models.md)
- [PyTorch Backend Architecture](../torch/arch_overview.md)
