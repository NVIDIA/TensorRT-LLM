# Serving GLM-5 NVFP4 with TensorRT-LLM

Verified on **8×B200 (SM100)** with TensorRT-LLM **1.3.0rc3** — Feb 2026.

**No source build required.** Works with the stock container — only config changes.

GLM-5 uses Multi-Latent Attention (MLA) with `v_head_dim=256` and `qk_nope_head_dim=192`.
TRT-LLM does not have a native GLM-5 model class, but **DeepSeek V3.2** (`deepseek_v32`)
is architecturally compatible and includes a built-in DSA indexer that routes context
attention through absorption mode (576/512 FMHA kernels).

> **Why V3.2 and not V3?** `DeepseekV3Attention` uses separate Q/K/V with headSize=256 for
> context, which falls back to unfused MHA — producing garbage output. `DeepseekV32Attention`
> has a built-in DSA indexer that routes context through absorption mode (576/512 kernels),
> which works correctly.

**Checkpoint:** [warnold-nv/GLM-5-nvfp4-v1](https://huggingface.co/warnold-nv/GLM-5-nvfp4-v1).  
*Not an official NVIDIA checkpoint — for experimentation purposes only. Model Opt performed by Will Arnold.*

---

## Prerequisites

| Item | Details |
|------|---------|
| Hardware | 8× B200 GPUs (SM100) |
| Container | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3` |
| Model | [GLM-5 NVFP4 checkpoint](https://huggingface.co/warnold-nv/GLM-5-nvfp4-v1) (e.g. `GLM-5-nvfp4-v1/`) |

---

## Step 1 — Prepare the model config

Edit `GLM-5-nvfp4-v1/config.json` (back up original first):

```bash
cd /path/to/GLM-5-nvfp4-v1
cp config.json config.json.orig
```

Make these changes:

```jsonc
{
  // Change these two:
  "architectures": ["DeepseekV32ForCausalLM"],   // was GlmMoeDsaForCausalLM
  "model_type": "deepseek_v32",                   // was glm_moe_dsa

  // Add these (GLM5 uses rope_theta=1M, no yarn scaling):
  "rope_theta": 1000000,
  "rope_scaling": {"type": "none", "factor": 1.0},

  // Everything else stays the same
}
```

## Step 2 — Fix the tokenizer config

> **Note:** These changes may not be required — needs confirmation. Try without them first.

Edit `GLM-5-nvfp4-v1/tokenizer_config.json`:

```jsonc
{
  // Change:
  "tokenizer_class": "PreTrainedTokenizerFast",   // was TokenizersBackend

  // Rename key:
  "additional_special_tokens": [...]               // was extra_special_tokens
}
```

## Step 3 — Start Docker

```bash
# Allocate GPUs (adjust partition for your cluster)
salloc -N 1 --gres=gpu:8 --time=04:00:00

# SSH to the node
ssh <node-name>

# Start TRT-LLM container
docker run --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/your/models:/workspace/scratch \
    -it nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3 bash
```

> **Everything below runs inside the Docker container.**

## Step 4 — Create the serve config

```bash
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
```

## Step 5 — Serve

```bash
trtllm-serve serve \
    /workspace/scratch/GLM-5-nvfp4-v1 \
    --tp_size 8 \
    --max_batch_size 64 \
    --max_num_tokens 8192 \
    --trust_remote_code \
    --host 0.0.0.0 \
    --extra_llm_api_options /tmp/glm5_config.yaml
```

Wait for CUDA graph warmup to finish (counts down from batch\_size=64 to 1, \~2 min).

## Step 6 — Test

From the host node (outside the container), find the container IP and send a request:

```bash
# Find container IP
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container-id>

# Test completion
curl http://<container-ip>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-5-nvfp4-v1","prompt":"The capital of France is","max_tokens":50}'
```

Expected: coherent English text, not repetitive gibberish.

---

## Summary of all changes

| File | Change | Why |
|------|--------|-----|
| `config.json` | `model_type` → `deepseek_v32`, `architectures` → `DeepseekV32ForCausalLM` | V3.2 has DSA indexer for absorption-mode context attention |
| `config.json` | Add `rope_theta: 1000000`, `rope_scaling: {"type":"none"}` | GLM5 uses 1M theta with no yarn scaling |
| `tokenizer_config.json` | `tokenizer_class` → `PreTrainedTokenizerFast`, rename `extra_special_tokens` | HF auto-tokenizer compatibility |
| Serve YAML | `sparse_attention_config` with DSA params | Enables the indexer for sparse attention |
