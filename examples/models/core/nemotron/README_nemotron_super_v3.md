# Nemotron Super V3 model

## Table of Contents

- [Overview](#overview)
- [Supported Hardware](#supported-hardware)
- [Usage](#usage)
  - [Online serving example](#online-serving-example)
    - [DGX Spark](#dgx-spark)
    - [SSM Stochastic Rounding with MTP](#ssm-stochastic-rounding-with-mtp)
  - [Offline inference example](#offline-inference-example)
- [Notes](#notes)

## Overview

The Nemotron Super V3 model uses a hybrid Mamba-Transformer MoE architecture with 120B total
parameters and only 12B active parameters per token, delivering efficient high-throughput inference.
It supports long context lengths and is optimized for complex, multi-document, and long-duration
applications.

This document outlines the procedures for executing Nemotron Super V3 using TensorRT LLM. The
implementation supports both single and multi-GPU configurations via the PyTorch backend.
Additionally, ModelOpt was employed to derive NVFP4 checkpoints from the source checkpoint.
The model repositories are:
* [Base BF16 repository](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16)
* [BF16 repository](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
* [NVFP4 repository](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)

All models are available under the [nvidia/nvidia-nemotron-v3](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) collection on Hugging Face.

Nemotron Super V3 supports the following features:
* BF16, NVFP4 model formats.
* Single and multi-GPU inference.
* Mixture-of-Experts (MoE) with expert parallelism.
* Hybrid SSM (Mamba) + Attention architecture.
* MTP (Multi-Token Prediction) speculative decoding.

## Supported Hardware
- **NVIDIA Blackwell**: B200, GB200, DGX Spark
- **NVIDIA Hopper**: H100, H200,


# Usage

## Online serving example

We can follow the configuration file from [nemotron-3-super-throughput.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/nemotron-3-super-throughput.yaml).

For the server:

```sh
# Example configuration:
cat > nemotron_super_v3.yaml<<EOF
max_batch_size: 512
max_num_tokens: 2048
tensor_parallel_size: 4
moe_expert_parallel_size: 4
trust_remote_code: true
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  max_batch_size: 256
kv_cache_config:
  free_gpu_memory_fraction: 0.8
  enable_block_reuse: false
num_postprocess_workers: 4
EOF

# Launch trtllm-serve with NVFP4 model (recommended).
trtllm-serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
--host 0.0.0.0 \
--port 8000 \
--reasoning_parser nano-v3 \
--tool_parser qwen3_coder \
--config nemotron_super_v3.yaml
```

For the client:

```sh
# Simple query example from client.
curl -X 'POST'   'http://localhost:8000/v1/chat/completions'   \
-H 'accept: application/json'   \
-H 'Content-Type: application/json'   \
-d '{
    "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "messages": [
      {
        "role":"user",
        "content": [
          {
            "type": "text",
            "text": "What is the capital of France?"
          }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95
  }' | jq
```

### DGX Spark

DGX Spark has a single Blackwell GPU with 128GB of unified memory. Note that DGX Spark is
currently functional support only; performance optimizations are ongoing. Use the NVFP4 checkpoint
with the following configuration:

```sh
cat > ./extra-llm-api-config.yml << EOF
kv_cache_config:
  enable_block_reuse: false
cuda_graph_config:
  max_batch_size: 32
  enable_padding: true
moe_config:
  backend: CUTLASS
EOF

trtllm-serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
--host 0.0.0.0 \
--port 8000 \
--reasoning_parser nano-v3 \
--tool_parser qwen3_coder \
--extra_llm_api_options ./extra-llm-api-config.yml
```

### SSM Stochastic Rounding with MTP

For long-context or high-throughput scenarios, enabling SSM stochastic rounding can improve output
quality by reducing numerical drift in the Mamba SSM state accumulation. This configuration also
enables MTP speculative decoding and chunked prefill for optimal performance.

```sh
cat > nemotron_super_v3_mtp.yaml << EOF
trust_remote_code: true
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.8
  mamba_ssm_cache_dtype: float16
  mamba_ssm_stochastic_rounding: true
  mamba_ssm_philox_rounds: 5
speculative_config:
  decoding_type: MTP
  max_draft_len: 5
  allow_advanced_sampling: true
cuda_graph_config:
  max_batch_size: 64
  enable_padding: true
moe_config:
  backend: TRTLLM
stream_interval: 10
enable_chunked_prefill: true
enable_attention_dp: true
num_postprocess_workers: 4
EOF

trtllm-serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
--host 0.0.0.0 \
--port 8000 \
--reasoning_parser nano-v3 \
--tool_parser qwen3_coder \
--config nemotron_super_v3_mtp.yaml
```

Key options:
* `mamba_ssm_stochastic_rounding`: Enables stochastic rounding for SSM state updates, improving numerical stability for long sequences.
* `mamba_ssm_philox_rounds`: Number of Philox RNG rounds for stochastic rounding.
* `mamba_ssm_cache_dtype`: Sets the data type for the Mamba SSM cache.
* `speculative_config`: Enables MTP with next-token prediction layers and advanced sampling.
* `enable_chunked_prefill`: Enables chunked prefill for better memory efficiency.

## Offline inference example

Using the `quickstart_advanced.py` script with MTP (Multi-Token Prediction) speculative decoding:

```sh
python3 examples/llm-api/quickstart_advanced.py \
    --model_dir nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --disable_kv_cache_reuse \
    --max_batch_size=128 \
    --moe_backend=TRTLLM \
    --spec_decode_algo=MTP \
    --spec_decode_max_draft_len=3 \
    --use_one_model \
    --tp_size=8 \
    --moe_ep_size 8 \
    --apply_chat_template
```

Key options:
* `--spec_decode_algo=MTP --spec_decode_max_draft_len=3`: Enables MTP speculative decoding with 3 draft tokens for faster generation.
* `--tp_size=8 --moe_ep_size 8`: Uses 8-way tensor parallelism and expert parallelism.
* `--moe_backend=TRTLLM`: Uses the optimized TensorRT LLM MoE backend.
* `--apply_chat_template`: Applies the chat template for the model.
* `--disable_kv_cache_reuse`: Required for hybrid SSM models.


# Notes

* prefix-cache is not supported for Nemotron Super V3 yet, so please set `enable_block_reuse: false` when launching a server.
* For detailed deployment instructions, see the [deployment guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/deployment-guide/deployment-guide-for-nemotron-3-super-on-trtllm.md).
