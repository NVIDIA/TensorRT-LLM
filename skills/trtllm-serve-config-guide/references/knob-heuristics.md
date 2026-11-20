<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Source-Backed Tuning Notes

Read an exact or nearby checked-in config and the model's deployment guide **before** using these notes. These are not universal thresholds.

## Commonly Tuned Fields

| Field | Guidance |
|---|---|
| `max_batch_size` | Scheduler ceiling, not a memory reservation and NOT proportional to concurrency — actual batch size adapts at runtime. Copy from the nearest checked-in source config; do not invent a value from concurrency. Prefer keeping the source value unless OOM occurs. MoE models generally cap lower than dense. |
| `max_num_tokens` | Scheduler token budget. When chunked prefill is **disabled** (default): must exceed ISL plus chat template overhead; sweet spot is ISL to 2× ISL. When chunked prefill is **enabled**: acts as the chunk size — see `enable_chunked_prefill` section below. General default is 8192. Tune together with `max_batch_size`. |
| `max_seq_len` | Global hard cap on total tokens per request (prompt + output). Set to `ISL + OSL + chat_template_overhead`. Chat templates and benchmarking preambles add tokens beyond raw ISL — overhead varies by model (checked-in configs show 20–200 tokens). Setting too tight rejects or truncates requests; setting too loose wastes KV cache per request. Copy from nearest checked-in config when available. |
| `enable_attention_dp` | High-throughput knob. MoE+GQA models benefit at lower concurrency thresholds than MoE+MLA or Dense+GQA. Memory overhead: small for MLA (compressed attention), substantial for GQA (full replication). Can trigger OOM when combined with aggressive KV cache fraction. Follow the exact model guide/config. |
| `kv_cache_config.free_gpu_memory_fraction` | OOM lever. MLA models (compressed KV) tolerate higher fractions; GQA models need more headroom. Lower when ADP enabled to account for replicated attention overhead. Large MoE models with ADP may need notably conservative fractions. Guides often adjust `max_batch_size` or `max_seq_len` first. |
| `moe_expert_parallel_size` / `moe_config.backend` | MoE only. Copy both from checked-in source — EP does not necessarily equal TP. If no backend source exists, mark as unverified; benchmark CUTLASS vs TRTLLM. |
| `cuda_graph_config.max_batch_size` / `batch_sizes` | Caps which decode batch sizes get CUDA graphs captured; batches above this fall back to eager execution (no error, just slower). **Default to `max_batch_size`** (safe, covers all batch sizes). Only lower when memory is tight — e.g., DeepSeek-R1 conc=1 uses `cuda_graph_config.max_batch_size: 1` with server `max_batch_size: 512` to avoid wasting graph memory on unreachable sizes. Also capped by `max_num_tokens / (1 + max_total_draft_tokens)` at runtime. |

## KV Cache Estimation

Use these formulas to sanity-check whether a concurrency target fits in GPU memory. Read the required values from the model's HuggingFace config (`config.json`).

**Per-token KV cache size:**

- **GQA (standard grouped-query attention):**
  `kv_per_token = 2 × num_attention_layers × (num_key_value_heads / TP) × head_dim × dtype_bytes`
  When `enable_attention_dp` is enabled, KV cache is fully replicated per rank (not TP-sharded); use divisor 1 instead of TP.
- **MLA (multi-latent attention, e.g. DeepSeek-V2/V3):**
  `kv_per_token = num_attention_layers × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes`

Where `dtype_bytes` is 2 for BF16/FP16, 1 for FP8/INT8.

**Approximate max concurrent requests (upper bound):**

```
max_requests ≈ floor((GPU_HBM × 0.90 − model_weights_bytes / TP) / (kv_per_token × (ISL + OSL)))
```

The 0.90 factor reserves ~10% of HBM for CUDA context, driver, and runtime overhead. Result is per-GPU.

**HF config fields to read:** `num_attention_layers` (equals `num_hidden_layers` for standard transformers; differs for hybrid models like Nemotron-H), `num_key_value_heads`, `head_dim` (or `hidden_size / num_attention_heads`), `kv_lora_rank`, `qk_rope_head_dim`.

**Caveats:** This estimate ignores activation memory, CUDA graph workspace, MoE expert workspace, and attention data parallelism (ADP) overhead. Always prefer checked-in config values over formula-derived estimates. Mark any formula-derived number as unverified.

## Chunked Prefill

Chunked prefill (`enable_chunked_prefill: true`) splits long prefill sequences into chunks so that decode batches sharing the same iteration are not starved. It is **disabled by default** and should be treated as an advanced latency optimization, not a default recommendation. See the `max_num_tokens` table entry above for how it changes token budget semantics.

**MLA models (DeepSeek-V2/V3/R1, Kimi-K2):**
- Chunked prefill IS supported for MLA — dedicated CUDA kernels exist with multi-round attention and softmax merging.
- **Hardware constraint:** only available on SM90 (Hopper) and SM100/SM103/SM120 (Blackwell+). The runtime automatically disables it with a warning on older GPUs.
- **Trade-off:** *"primarily designed to reduce TPOT [...] will also decrease overall throughput."*
- **Recommendation:** do not enable by default for MLA models. Consider it only for latency-sensitive workloads on Hopper or Blackwell GPUs where TPOT reduction outweighs the throughput cost.

**Non-MLA models (GQA):** more broadly supported across GPU generations. Still disabled by default; enable when long prefill sequences cause decode latency spikes.
