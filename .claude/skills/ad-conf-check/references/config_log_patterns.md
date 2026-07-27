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

# AutoDeploy Config-to-Log Pattern Reference

This file maps each YAML config parameter to the log patterns that confirm or deny its application.

> **Source of truth for default config values**: `tensorrt_llm/_torch/auto_deploy/config/default.yaml`
> That file defines all available transforms and their default settings. The patterns below document how each parameter's application (or failure) appears in server logs.

## Verification Sources

Each config parameter can be verified through one or more sources:

| Source | Tag | Description |
|--------|-----|-------------|
| Server log | `[log]` | Explicit log messages printed during startup/transforms |
| Graph dump | `[graph]` | FX graph snapshots from `AD_DUMP_GRAPHS_DIR` (op types, weight shapes, collective ops) |
| Nsys trace | `[nsys]` | Nsight Systems profile (kernel launches, stream concurrency, graph capture/replay) |

Parameters tagged with multiple sources can be verified by any of them. When a parameter has **no log evidence**, alternative sources are noted.

## How Transforms Are Logged

Every transform logs with prefix: `[stage=<stage>, transform=<name>]`

Summary lines after each transform:
- **Applied**: `[SUMMARY] matches=N | time: ...`
- **Skipped**: `[SUMMARY] skipped | time: ...`
- **Disabled**: `[SUMMARY] disabled`

Transform apply logs are prefixed with `[APPLY]`.

---

## Top-Level Config Parameters

### `runtime` `[log]`
- **Values**: `trtllm`, `demollm`
- **No direct log**. Infer from overall server startup.

### `compile_backend` `[log]` `[nsys]`
- **Values**: `torch-simple`, `torch-cudagraph`, `torch-compile`, `torch-opt`
- `torch-simple`: No-op backend (no compilation). No specific log pattern — absence of compile/graph-capture logs confirms it.
- `torch-cudagraph`: Look for `"Warm up with max_batch_size="` or `"Capturing graph for batch size:"`
- `torch-compile`: Look for `"Torch Dynamo cache size limit"`
- `torch-opt`: Look for `"Setting Torch Dynamo recompile limit"`
- **Nsys**: `torch-cudagraph` → CUDA graph capture/replay visible in trace; `torch-compile` → compiled kernels visible; `torch-simple` → no compilation or graph capture events

### `attn_backend` `[log]` `[graph]`
- **Values**: `trtllm`, `flashinfer`, `triton`, etc.
- **No explicit log for selection**. Check for attention-related ops in transform logs (e.g., `insert_cached_attention` transform).
- **Graph**: attention op types in post-transform graph confirm which backend was inserted

### `max_seq_len`, `max_num_tokens`, `max_batch_size` `[log]`
- **No direct confirmation log**. These are consumed by other transforms (e.g., piecewise bucket generation).
- `max_num_tokens` appears in: `"Auto-generated piecewise_num_tokens from max_num_tokens=N"`
- `max_batch_size` may be capped: `"max_batch_size (N) exceeds max_num_tokens (M). Capping max_batch_size to M."`
- `max_seq_len` is resolved via model factory when not explicitly set by the user.

### `cuda_graph_config.batch_sizes` `[log]` `[nsys]`
- **Note**: Replaces the deprecated top-level `cuda_graph_batch_sizes` shortcut. Now nested under `cuda_graph_config`.
- **Confirm**: `"Using cuda_graph_batch_sizes: [...]"` (logged after syncing to compile_model transform)
- **Confirm per-size**: `"Capturing graph for batch size: N"` (one per size)
- **Warmup**: `"Warm up with max_batch_size=N before graph capture"`
- **Nsys**: CUDA graph capture and replay events visible per batch size
- **Deprecated**: The old top-level `cuda_graph_batch_sizes` key and `compile_model.cuda_graph_batch_sizes` in `default.yaml` are removed. If a user config still uses the old key, it may be silently ignored — check for the absence of `"Using cuda_graph_batch_sizes"` log as a clue.

### `enable_chunked_prefill` `[nsys]`
- **No explicit log**. Applied at executor level.
- **Nsys**: chunked prefill manifests as multiple shorter prefill kernels instead of one large prefill per request

### `model_factory` `[log]`
- **No explicit log for selection**. The value is known from the YAML config itself.
- **Verification**: Check that `build_and_load_factory_model` transform succeeded via `[SUMMARY]`. If the factory was invalid, a `ValueError` would appear in the log (`"model_factory '...' not found"`).
- Mark as APPLIED if the model built successfully (no error in log).

---

## `kv_cache_config` Parameters

### `enable_block_reuse` `[nsys]`
- **No explicit log**. Passed to cache initializers.
- **Nsys**: block reuse manifests as reduced KV cache allocation kernels on repeated prompts with shared prefixes

### `free_gpu_memory_fraction` `[nsys]`
- **No explicit log**. Used in KV cache allocation.
- **Nsys**: total GPU memory allocated for KV cache visible in memory allocation events

### `tokens_per_block` `[nsys]`
- **No explicit log**. Used in KV cache allocation.
- **Nsys**: block size affects cache management kernel patterns

---

## Transform Parameters

All transforms log `[stage=X, transform=<name>]` prefix. Check for `[SUMMARY]` line.

### `compile_model.piecewise_enabled` `[log]` `[nsys]`
- **Transform key**: `compile_model`
- **Success indicators**:
  - `"TorchCudagraphCompiler: dual-mode enabled (monolithic + piecewise)"`
  - `"Auto-generated piecewise_num_tokens from max_num_tokens=N: [buckets]"`
  - `"PiecewiseCapturedGraph: prepared with N submodules"`
  - `"PiecewiseCapturedGraph: warming up for num_tokens=N"`
  - `"PiecewiseCapturedGraph: captured graphs for num_tokens=N"`
- **Failure indicators**:
  - `"model is not a GraphModule, piecewise CUDA graph requires an FX GraphModule. Falling back to eager execution"`
  - `"Dropping piecewise_num_tokens [...] (too small for mixed batch"`
  - `"Dropping piecewise buckets [...] that exceed mixed-batch capacity"`

### `multi_stream_moe.enabled` `[log]` `[graph]` `[nsys]`
- **Transform key**: `multi_stream_moe`
- **Success**: `[SUMMARY] matches=N` (N > 0)
- **Failure/skip indicators**:
  - `"No merge add found downstream of MoE node"`
  - `"Could not identify shared-expert subgraph"`
  - `"First shared-expert op ... does not directly consume fork point"`
- **Nsys**: multi-stream concurrency visible as overlapping kernels on separate CUDA streams
- **Graph**: fork/join ops inserted around MoE subgraph

### `multi_stream_gemm.enabled` `[log]` `[graph]` `[nsys]`
- **Transform key**: `multi_stream_gemm`
- **Success**: `"Fork point ...: moving ... to aux stream"`
- **Skip**: `"Skipping fork point ...: already has multi-stream ops."`
- **Nsys**: overlapping GEMM + allreduce kernels on separate streams
- **Graph**: fork/join ops around GEMM nodes

### `multi_stream_mla_attn.enabled` `[log]` `[graph]` `[nsys]`
- **Transform key**: `multi_stream_mla_attn`
- **Success**: `[SUMMARY] matches=N` (N > 0)
- **Success**: `"Fork point ...: moving ... kv ... to aux stream"`
- **Failure**: `"Could not find KV projection fork point"`
- **Failure**: `"No fork point ... MLA"`
- **Check**: `[SUMMARY]` line for matches vs skipped.
- **Nsys**: overlapping KV projection and Q projection kernels on separate streams

### `gather_logits_before_lm_head.enabled` `[log]` `[graph]`
- **Transform key**: `gather_logits_before_lm_head`
- **Success**: `[SUMMARY] matches=1`
- **No explicit log otherwise**. Check summary line.
- **Graph**: gather op inserted before lm_head in post-transform graph

### `detect_sharding.*` `[log]` `[graph]`
- **Transform key**: `detect_sharding`

#### `allreduce_strategy`
- **Confirm**: `"Using allreduce strategy: <STRATEGY>, dist backend: <backend>"`

#### `sharding_dims` / `sharding_source`
- **TP from manual**: `"Applying sharding from manual config"`
- **TP from factory**: `"Applying sharding from factory config"`
- **TP heuristic**: `"Running autodeploy TP sharding heuristics"`
- **EP**: `"Running autodeploy EP sharding heuristics"`
- **BMM**: `"Running autodeploy BMM sharding heuristics"`
- **Skip TP**: `"Skipping TP sharding for single device"` or `"No manual config found. Skipping sharding from manual config"`
- **Result**: `"Applied N TP shards from config. Simple: N, row-col: N ..."`
- **Attention DP**: `"Attention DP is enabled. Skipping TP sharding, only MoE-all-to-all"`

### `fuse_gemms_mixed_children.enabled` `[log]` `[graph]`
- **Transform key**: `fuse_gemms_mixed_children`
- **Success**: `"Fusing N GEMMs (...) into ... (dtype=...)"`
- **Skip**: `"Skipping GEMM fusion for ...: mixed dtypes ..."`
- **Graph**: fused GEMM ops replace individual linear ops in post-transform graph

### `fuse_add_rms_norm.enabled` `[log]` `[graph]`
- **Transform key**: `fuse_add_rms_norm`
- **Check**: `[SUMMARY]` line.
- **Graph**: fused add+rmsnorm custom ops in post-transform graph

### `fuse_swiglu.enabled` `[log]` `[graph]`
- **Transform key**: `fuse_swiglu`
- **Check**: `[SUMMARY]` line.
- **Graph**: fused swiglu custom op in post-transform graph

### `match_swiglu_pattern.enabled`, `match_nvfp4_swiglu_pattern.enabled`, `match_finegrained_fp8_swiglu_pattern.enabled` `[log]` `[graph]`
- **Check**: `[SUMMARY]` line for respective transform key.
- **Graph**: matched pattern ops replaced with custom ops

### `fuse_gemms.enabled`, `fuse_fp4_gemms.enabled`, `fuse_fp8_gemms.enabled` `[log]` `[graph]`
- **Check**: `[SUMMARY]` line for respective transform key.
- **Graph**: reduced linear op count in post-transform graph

### `fuse_rmsnorm_quant_fp8.enabled`, `fuse_trtllm_attn_quant_fp8.enabled` `[log]` `[graph]`
- **Check**: `[SUMMARY]` line for respective transform key.
- **Graph**: fused rmsnorm+quant or attn+quant custom ops in post-transform graph

### `export_to_gm.num_moe_experts_for_export` `[log]`
- **Transform key**: `export_to_gm`
- **No explicit log for the value**. Check that `export_to_gm` transform succeeded via `[SUMMARY]`.

### `initialize_mrope_delta_cache.enabled` `[log]`
- **Transform key**: `initialize_mrope_delta_cache`
- **Check**: `[SUMMARY]` line.

---

## General Failure Patterns

These log patterns indicate something went wrong regardless of specific config:
- `"Transform <name> failed: <error>"` — transform errored and was skipped
- `"Falling back"` — a feature was attempted but reverted
- `"Skipping"` — a step was intentionally skipped (may or may not be a problem)
