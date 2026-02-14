#!/bin/bash
# ============================================================
# GLM5 NVFP4 serving with TRT-LLM
# ============================================================
#
# VERIFIED WORKING: 2026-02-14 on 8xB200 (SM100), TRT-LLM 1.3.0rc3/rc4
#
# DIRECTIONS: To serve GLM5 NVFP4 in TRT-LLM, you need the following
# changes on top of the base container (nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3):
#
# A. MODEL CONFIG (GLM-5-nvfp4-v1/config.json)
#    TRT-LLM doesn't have native GLM5 support, so remap to DeepSeek V3.2:
#    - "model_type": "glm_moe_dsa" -> "deepseek_v32"
#    - "architectures": ["GlmMoeDsaForCausalLM"] -> ["DeepseekV32ForCausalLM"]
#    - Add top-level "rope_theta": 1000000 (GLM5 uses 1M, not DSV3 default 10K)
#    - Add "rope_scaling": {"type": "none", "factor": 1.0}
#      (prevents DSV3 code path from incorrectly forcing yarn scaling)
#    - Back up original: cp config.json config.json.orig
#
#    WHY V3.2 NOT V3: DeepseekV32Attention has a built-in DSA indexer that
#    routes context attention through absorption mode (576/512 FMHA kernels).
#    DeepseekV3Attention uses separate Q/K/V with headSize=256 for context,
#    which falls back to unfused MHA that doesn't support MLA (gibberish).
#
# B. TOKENIZER CONFIG (GLM-5-nvfp4-v1/tokenizer_config.json)
#    GLM5's custom tokenizer class doesn't load via standard HF auto:
#    - "tokenizer_class": "TokenizersBackend" -> "PreTrainedTokenizerFast"
#      (tokenizer.json is standard format, PreTrainedTokenizerFast loads it fine)
#    - "extra_special_tokens": [...] -> "additional_special_tokens": [...]
#      (HF expects "additional_special_tokens" as key name; the GLM5 field name
#       "extra_special_tokens" causes 'list has no attribute keys' error)
#
# C. SERVE CONFIG
#    - sparse_attention_config with algorithm=dsa + GLM5 indexer params
#    - kv_cache_config with dtype=fp8 for FP8 KV cache
#    - --trust_remote_code (needed for tokenizer)
#    - --host 0.0.0.0 (expose outside container)
#
# ============================================================

# --- Step 1: Allocate node ---
salloc -N 1 --gres=gpu:8 --time=04:00:00 --partition=b200@500/None@cr+mp/8gpu-224cpu-2048gb
ssh umb-b200-236

# --- Step 2: Start Docker (mount is read-only, build in /tmp) ---
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/scratch.asteiner:/workspace/scratch \
    -it nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3 bash

# ============================================================
# Everything below runs INSIDE the Docker container
# ============================================================

# --- Step 3: Serve GLM5 ---
cat > /tmp/glm5_config.yaml <<'YAML'
sparse_attention_config:
  algorithm: dsa
  index_n_heads: 32
  index_head_dim: 128
  index_topk: 2048
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.90
enable_chunked_prefill: true
YAML

trtllm-serve serve \
    /workspace/scratch/GLM-5-nvfp4-v1 \
    --tp_size 8 \
    --max_batch_size 64 \
    --max_num_tokens 8192 \
    --trust_remote_code \
    --host 0.0.0.0 \
    --extra_llm_api_options /tmp/glm5_config.yaml

# --- Test (from host node, not inside container) ---
curl http://172.17.0.2:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-5-nvfp4-v1","prompt":"The capital of France is","max_tokens":50}'
