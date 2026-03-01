# TensorRT-LLM Examples

## Quick Start

TensorRT-LLM uses the **PyTorch backend** by default. The fastest way to get started:

```bash
# Serve a model with OpenAI-compatible API
trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Or use a pre-quantized model for better performance
trtllm-serve "nvidia/Llama-3.1-8B-Instruct-FP8"
```

For the Python API:

```python
from tensorrt_llm import LLM

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
output = llm.generate(["What is TensorRT-LLM?"])
print(output[0].outputs[0].text)
```

Full documentation: https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html

## Examples Directory

| Directory | Description |
|---|---|
| [`llm-api/`](llm-api/) | Python LLM API examples (offline inference, quantization, speculative decoding) |
| [`apps/`](apps/) | Application examples (chat, FastAPI server) |
| [`configs/`](configs/) | Pre-tuned serving configurations for popular models |
| [`serve/`](serve/) | `trtllm-serve` deployment guides and examples |
| [`quantization/`](quantization/) | Quantization workflows with NVIDIA Model Optimizer |

## Pre-Tuned Model Configurations

The [`configs/database/`](configs/database/) directory contains optimized serving
configurations for popular models across different GPUs and concurrency levels.
These are validated Pareto-optimal configurations from benchmark data.

Use them with `trtllm-serve`:

```bash
trtllm-serve "deepseek-ai/DeepSeek-R1-0528" --config configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc64.yaml
```

See [`configs/database/lookup.yaml`](configs/database/lookup.yaml) for the full catalog.

## Legacy Engine-Build Workflow

> **⚠️ Legacy:** The `convert_checkpoint.py` → `trtllm-build` → `run.py`
> workflow is legacy and may not receive new features.
> For new projects, use `trtllm-serve` or the LLM API as shown above.

The [`models/`](models/) directory contains per-model scripts for the legacy
TensorRT engine-build workflow. These scripts convert Hugging Face checkpoints
to TensorRT engines for deployment. While still functional for supported models,
this workflow is no longer the recommended path and may not support newly added
models.

If you are following a tutorial or guide that references `convert_checkpoint.py`
or `trtllm-build`, please refer to the
[Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
for the current recommended workflow.
