# Nemotron Nano V3 model

## Overview

The Nemotron Nano V3 model uses a hybrid Mamba-Transformer MoE architecture and supports a 1M
token context length. This enables developers to build reliable, high-throughput agents across
complex, multi-document, and long-duration applications.

This document outlines the procedures for executing Nemotron Nano V3 using TensorRT LLM. The
implementation supports both single and multi-GPU configurations via the AutoDeploy backend.
Additionally, ModelOpt was employed to derive FP8 and NVFP4 checkpoints from the source checkpoint.
The model repositories are:
* [BF16 repository](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
* [FP8 repository](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)

Nemotron Nano V3 supports the following features:
* BF16, FP8 with KV cache FP8, NVFP4 model formats.
* Single and multi-GPU inference.
* Support 1M token context with long context/generation sequences.

# Usage

## Online serving example

We can follow the configuration file from [nano_v3.yaml](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/auto_deploy/nano_v3.yaml).

For the server:

```sh
# Example configuration:
cat > nano_v3.yaml<<EOF
runtime: trtllm
compile_backend: torch-cudagraph
max_batch_size: 384
max_seq_len: 65536
enable_chunked_prefill: true
attn_backend: flashinfer
model_factory: AutoModelForCausalLM
skip_loading_weights: false
cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 320, 384]
kv_cache_config:
  free_gpu_memory_fraction: 0.88
  # tunable mamba cache dtype
  # --> use float32 for accuracy and default (auto) for speed
  # mamba_ssm_cache_dtype: float32
transforms:
  detect_sharding:
    allreduce_strategy: 'SYMM_MEM'
    sharding_dims: ['ep', 'bmm']
    manual_config:
      head_dim: 128
      tp_plan:
        # mamba SSM layer
        "in_proj": "mamba"
        "out_proj": "rowwise"
        # attention layer
        "q_proj": "colwise"
        "k_proj": "colwise"
        "v_proj": "colwise"
        "o_proj": "rowwise"
        # NOTE: consider not sharding shared experts and/or
        # latent projections at all, keeping them replicated.
        # To do so, comment out the corresponding entries.
        # moe layer: SHARED experts
        "up_proj": "colwise"
        "down_proj": "rowwise"
        # MoLE: latent projections: simple shard
        "fc1_latent_proj": "gather"
        "fc2_latent_proj": "gather"
  multi_stream_moe:
    stage: compile
    enabled: true
  gather_logits_before_lm_head:
    enabled: true
  fuse_mamba_a_log:
    stage: post_load_fusion
    enabled: true
EOF

# Launch trtllm-server.
TRTLLM_ENABLE_PDL=1 trtllm-serve <model_path> \
--host 0.0.0.0 \
--port 8000 \
--backend _autodeploy \
--trust_remote_code \
--config nano_v3.yaml

# OR you can launch trtllm-server to support reasoning content parsing.
TRTLLM_ENABLE_PDL=1 trtllm-serve <model_path> \
--host 0.0.0.0 \
--port 8000 \
--backend _autodeploy \
--trust_remote_code \
--reasoning_parser nano-v3 \
--config nano_v3.yaml

# OR you can launch trtllm-server to support tool-calling.
TRTLLM_ENABLE_PDL=1 trtllm-serve <model_path> \
--host 0.0.0.0 \
--port 8000 \
--backend _autodeploy \
--trust_remote_code \
--reasoning_parser nano-v3 \
--tool_parser qwen3_coder \
--config nano_v3.yaml
```

For the client:

```sh
# Simple query example from client.
curl -X 'POST'   'http://0.0.0.0:8000/v1/chat/completions'   \
-H 'accept: application/json'   \
-H 'Content-Type: application/json'   \
-d '{
    "model": "nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16",
    "messages": [
      {
        "role":"user",
        "content": [
          {
            "type": "text",
            "text": "Hello, my name is"
          }
        ]
      }
    ],
    "max_tokens": 128,
    "temperature": 0
  }' | jq

# Simple query example (with reasoning disabled)
curl -X 'POST'   'http://0.0.0.0:8000/v1/chat/completions'   \
-H 'accept: application/json'   \
-H 'Content-Type: application/json'   \
-d '{
    "model": "nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16",
    "messages": [
      {
        "role":"user",
        "content": [
          {
            "type": "text",
            "text": "Hello, my name is"
          }
        ]
      }
    ],
    "max_tokens": 128,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | jq
```

## Offline inference example

```sh
python examples/auto_deploy/build_and_run_ad.py --model <model_path> --args.compile_backend torch-cudagraph
```

**More verbose offline inference example**:

Use a yaml:

```sh
cat > nano_v3_offline.yaml<<EOF
model:
  nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3
args:
  compile_backend: torch-cudagraph
  enable_chunked_prefill: true
  kv_cache_config:
    # disable kv_cache reuse since not supported for hybrid/ssm models
    enable_block_reuse: false
EOF

python examples/auto_deploy/build_and_run_ad.py --yaml-extra nano_v3_offline.yaml
```

The CLI can also be used to override certain config values:

```sh
python examples/auto_deploy/build_and_run_ad.py \
  --model nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3 \
  --args.compile_backend torch-cudagraph \
  --args.enable_chunked_prefill true \
  --args.kv_cache_config.enable_block_reuse false
```

# Notes

* More examples can be found in [trtllm_cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/trtllm_cookbook.ipynb).
* prefix-cache is not supported for Nano v3 yet, so please set `enable_block_reuse: false` when launching a server.
* mamba-cache-dtype should be set to float32 to support better long sequences when launching a server.
