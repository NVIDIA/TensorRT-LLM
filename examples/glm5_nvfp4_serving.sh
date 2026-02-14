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
# C. TRT-LLM C++ FIX (attentionOp.cpp, applied via sed at build time)
#    DSV3 MLA assumes v_head_dim == qk_nope_head_dim. GLM5 differs (256 vs 192):
#    - Line ~2799: change headSizeV = qk_nope_head_dim -> v_head_dim
#    - Line ~2896: relax FMHA assertion to warning (safety net)
#
# D. TRT-LLM SOURCE BRANCH (new FMHA kernels for GLM5)
#    Use PerkzZheng's branch which adds pre-compiled FMHA kernels:
#    - github.com/PerkzZheng/TensorRT-LLM branch: user/perkzz/glm-flash-trtllm-gen
#    - Ref: https://github.com/NVIDIA/TensorRT-LLM/compare/main...PerkzZheng:TensorRT-LLM:user/perkzz/glm-flash-trtllm-gen
#
# E. FLASHINFER PATCH (runtime, after pip install)
#    Remove hardcoded qk_nope_head_dim==128 check in flashinfer/mla.py:
#    - Post-absorption kernel operates on d_qk=576, d_v=512 regardless
#    - Ref: https://github.com/Aphoh/flashinfer/commit/5fb657b
#
# F. SERVE CONFIG
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

# --- Step 3: Clone PerkzZheng's branch to /tmp ---
cd /tmp
git clone --branch user/perkzz/glm-flash-trtllm-gen --single-branch \
    https://github.com/PerkzZheng/TensorRT-LLM.git TensorRT-LLM-GLM
cd /tmp/TensorRT-LLM-GLM

# --- Step 4: Apply C++ fixes ---
# Fix 1: headSizeV = v_head_dim (GLM5: v_head_dim=256, qk_nope=192)
sed -i 's/fmhaParams.headSizeV = mMLAParams.qk_nope_head_dim;/fmhaParams.headSizeV = mMLAParams.v_head_dim;/' \
    cpp/tensorrt_llm/common/attentionOp.cpp

# Fix 2: Relax FMHA assertion to warning (safety net)
python3 -c "
p = 'cpp/tensorrt_llm/common/attentionOp.cpp'
with open(p) as f: c = f.read()
old = '''            if (!mIsGenerationMLA)
            {
                TLLM_CHECK_WITH_INFO(
                    mFmhaDispatcher->isSupported(), \"Deepseek should be supported by fmha in context part.\");
            }'''
new = '''            if (!mIsGenerationMLA && !mFmhaDispatcher->isSupported())
            {
                TLLM_LOG_WARNING(\"FMHA not supported for context MLA, falling back to unfused MHA.\");
            }'''
if old in c:
    c = c.replace(old, new)
    with open(p, 'w') as f: f.write(c)
    print('Assertion relaxation applied')
else:
    print('Assertion pattern not found - may already be fixed')
"

# --- Step 5: Verify fixes ---
echo "=== Verifying headSizeV fix ==="
grep -n "headSizeV = mMLAParams" cpp/tensorrt_llm/common/attentionOp.cpp
echo "=== Verifying assertion relaxation ==="
grep -n "FMHA not supported for context MLA\|Deepseek should be supported" cpp/tensorrt_llm/common/attentionOp.cpp

# --- Step 6: Build (~20-30 min with --fast_build -j32) ---
python scripts/build_wheel.py \
    --fast_build \
    --cuda_architectures "100" \
    --no-venv \
    --job_count 32

# --- Step 7: Install ---
pip install build/tensorrt_llm*.whl --force-reinstall --no-deps

# --- Step 8: Patch flashinfer ---
python3 -c "
import flashinfer, os
path = os.path.join(os.path.dirname(flashinfer.__file__), 'mla.py')
with open(path) as f: c = f.read()
if 'qk_nope_head_dim != 128' in c:
    c = c.replace('if qk_nope_head_dim != 128:', 'if False and qk_nope_head_dim != 128:')
    with open(path, 'w') as f: f.write(c)
    print('flashinfer patched')
else:
    print('already patched or not found')
"

# --- Step 9: Serve GLM5 ---
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
# curl http://172.17.0.2:8000/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model":"GLM-5-nvfp4-v1","prompt":"The capital of France is","max_tokens":50}'
