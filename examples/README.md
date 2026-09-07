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
| [`kv_cache_compression/`](kv_cache_compression/) | KV cache compression examples, including NVFP4 cold-page quantization and TriAttention |
| [`apps/`](apps/) | Application examples (chat, FastAPI server) |
| [`configs/`](configs/) | Pre-tuned serving configurations — [curated](configs/curated/) quick-starts and a [comprehensive database](configs/database/) |
| [`auto_deploy/`](auto_deploy/) | AutoDeploy (beta) development examples, cookbooks, and model registry |
| [`serve/`](serve/) | `trtllm-serve` deployment guides and examples |
| [`quantization/`](quantization/) | Quantization workflows with NVIDIA Model Optimizer |

## Pre-Tuned Model Configurations

The [`configs/`](configs/) directory contains recommended `trtllm-serve` configurations.
Start with the hand-picked [curated configs](configs/curated/) or browse the full
[database](configs/database/) for specific GPU / ISL / OSL / concurrency combinations.

```bash
trtllm-serve "deepseek-ai/DeepSeek-R1-0528" \
  --config configs/curated/deepseek-r1-throughput.yaml
```

For model-specific walkthroughs and an interactive recipe selector, see the
[Model Recipes](https://nvidia.github.io/TensorRT-LLM/deployment-guide/index.html)
deployment guide.

## AutoDeploy (Beta)

The [AutoDeploy](https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/auto-deploy.html)
backend automatically translates HuggingFace models into optimized inference graphs.
It is accessed through the same `trtllm-serve`, `trtllm-bench`, and LLM API entry
points as the default PyTorch backend.

See [`auto_deploy/`](auto_deploy/) for development examples, Jupyter cookbooks,
and a registry of 90+ validated models.

## Model-Specific Guides

The [`models/`](models/) directory contains model-specific guides for serving
Hugging Face checkpoints with `trtllm-serve` or the LLM API.

The legacy `convert_checkpoint.py` → `trtllm-build` → `run.py` TensorRT engine
workflow has been removed. See the
[TensorRT backend removal guide](https://nvidia.github.io/TensorRT-LLM/legacy/tensorrt-backend-removal.html)
and the [Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
for the current recommended path.
