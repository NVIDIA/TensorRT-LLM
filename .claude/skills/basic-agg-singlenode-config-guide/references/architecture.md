# Architecture Classes and Constants

## How to Classify a Model

| Check | Class |
|---|---|
| Has MoE experts AND Multi-head Latent Attention (MLA) | **MoE_MLA** |
| Has MoE experts AND Grouped Query Attention (GQA) | **MoE_GQA** |
| No MoE experts, uses GQA or MHA | **Dense_GQA** |

## Known Model Mappings

| Model | Class |
|---|---|
| DeepSeek-R1 / DeepSeek-V3 / DeepSeek-V3.2-Exp | MoE_MLA |
| Kimi-K2 | MoE_MLA |
| Qwen3-235B-A22B | MoE_GQA |
| Llama-3.3-70B / Llama-4-Scout | Dense_GQA |
| GPT-OSS-120B | Dense_GQA |
| Nemotron-3-Super-120B | Dense_GQA |

## Pattern-Based Detection (Unknown Models)

- `deepseek` or `kimi` in name → likely MoE_MLA
- `qwen` + large param count suggesting MoE (e.g., 235B-A22B) → MoE_GQA
- `llama`, `nemotron`, `falcon`, `gpt` → likely Dense_GQA
- `moe` or `mixture` in name → MoE (check model config for MLA vs GQA attention)
- If uncertain, default to **MoE_MLA** (most conservative — lowest KV memory overhead)

## Constants (Set and Forget)

Identical across all basic aggregate (IFB, PyTorch, non-speculative) benchmark-optimal configs:

| Parameter | Value | Notes |
|---|---|---|
| `backend` | `pytorch` | PyTorch executor |
| `cuda_graph_config.enable_padding` | `true` | Always on for throughput |
| `kv_cache_config.dtype` | `fp8` | FP8 KV cache |
| `trust_remote_code` | `true` | Required for most HF models |
| `stream_interval` | `10` | Token streaming interval |
| `moe_expert_parallel_size` | = `tensor_parallel_size` | EP = TP for MoE models |
| `cuda_graph_config.max_batch_size` | = target concurrency (cap 512 MoE, 1024 Dense) | Matches concurrency; over-provisioning wastes graph memory |
| `max_seq_len` | >= ISL + OSL + 68 | Total token budget; +68 empirical margin |
