<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

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

## Migrating from the Engine-Build Workflow

The legacy TensorRT execution backend and its `convert_checkpoint.py` →
`trtllm-build` → `run.py` workflow have been removed. Use `trtllm-serve` or the
LLM API as shown above. See the
[Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
for the current workflow and the
[legacy documentation](../docs/source/legacy/)
for historical engine-build instructions.
