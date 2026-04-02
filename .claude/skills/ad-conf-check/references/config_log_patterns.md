# AutoDeploy Config-to-Log Pattern Reference

This file maps each YAML config parameter to the log patterns that confirm or deny its application.

## How Transforms Are Logged

Every transform logs with prefix: `[stage=<stage>, transform=<name>]`

Summary lines after each transform:
- **Applied**: `[SUMMARY] matches=N | time: ...`
- **Skipped**: `[SUMMARY] skipped | time: ...`
- **Disabled**: `[SUMMARY] disabled`

Transform apply logs are prefixed with `[APPLY]`.

---

## Top-Level Config Parameters

### `runtime`
- **Values**: `trtllm`, `demollm`
- **No direct log**. Infer from overall server startup.

### `compile_backend`
- **Values**: `torch-cudagraph`, `torch-compile`, `torch-opt`
- `torch-cudagraph`: Look for `"Warm up with max_batch_size="` or `"Capturing graph for batch size:"`
- `torch-compile`: Look for `"Torch Dynamo cache size limit"`
- `torch-opt`: Look for `"Setting Torch Dynamo recompile limit"`

### `attn_backend`
- **Values**: `trtllm`, `flashinfer`, `triton`, etc.
- **No explicit log for selection**. Check for attention-related ops in transform logs (e.g., `insert_cached_attention` transform).

### `max_seq_len`, `max_num_tokens`, `max_batch_size`
- **No direct confirmation log**. These are consumed by other transforms (e.g., piecewise bucket generation).
- `max_num_tokens` appears in: `"Auto-generated piecewise_num_tokens from max_num_tokens=N"`

### `cuda_graph_batch_sizes`
- **Confirm**: `"Using cuda_graph_batch_sizes: [...]"`
- **Confirm per-size**: `"Capturing graph for batch size: N"` (one per size)
- **Warmup**: `"Warm up with max_batch_size=N before graph capture"`

### `enable_chunked_prefill`
- **No explicit log**. Applied at executor level.

### `model_factory`
- **No explicit log for selection**. Check for model class in build_model transform logs.

---

## `kv_cache_config` Parameters

### `enable_block_reuse`
- **No explicit log**. Passed to cache initializers.

### `free_gpu_memory_fraction`
- **No explicit log**. Used in KV cache allocation.

### `tokens_per_block`
- **No explicit log**. Used in KV cache allocation.

---

## Transform Parameters

All transforms log `[stage=X, transform=<name>]` prefix. Check for `[SUMMARY]` line.

### `compile_model.piecewise_enabled`
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

### `multi_stream_moe.enabled`
- **Transform key**: `multi_stream_moe`
- **Success**: `[SUMMARY] matches=N` (N > 0)
- **Failure/skip indicators**:
  - `"No merge add found downstream of MoE node"`
  - `"Could not identify shared-expert subgraph"`
  - `"First shared-expert op ... does not directly consume fork point"`

### `multi_stream_gemm.enabled`
- **Transform key**: `multi_stream_gemm`
- **Success**: `"Fork point ...: moving ... to aux stream"`
- **Skip**: `"Skipping fork point ...: already has multi-stream ops."`

### `multi_stream_mla_attn.enabled`
- **Transform key**: `multi_stream_mla_attn`
- **Check**: `[SUMMARY]` line for matches vs skipped.

### `gather_logits_before_lm_head.enabled`
- **Transform key**: `gather_logits_before_lm_head`
- **Success**: `[SUMMARY] matches=1`
- **No explicit log otherwise**. Check summary line.

### `detect_sharding.*`
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

### `fuse_gemms_mixed_children.enabled`
- **Transform key**: `fuse_gemms_mixed_children`
- **Success**: `"Fusing N GEMMs (...) into ... (dtype=...)"`
- **Skip**: `"Skipping GEMM fusion for ...: mixed dtypes ..."`

### `fuse_add_rms_norm.enabled`
- **Transform key**: `fuse_add_rms_norm`
- **Check**: `[SUMMARY]` line.

### `fuse_swiglu.enabled`
- **Transform key**: `fuse_swiglu`
- **Check**: `[SUMMARY]` line.

### `match_swiglu_pattern.enabled`, `match_nvfp4_swiglu_pattern.enabled`, `match_finegrained_fp8_swiglu_pattern.enabled`
- **Check**: `[SUMMARY]` line for respective transform key.

### `fuse_gemms.enabled`, `fuse_fp4_gemms.enabled`, `fuse_fp8_gemms.enabled`
- **Check**: `[SUMMARY]` line for respective transform key.

### `fuse_rmsnorm_quant_fp8.enabled`, `fuse_trtllm_attn_quant_fp8.enabled`
- **Check**: `[SUMMARY]` line for respective transform key.

### `export_to_gm.num_moe_experts_for_export`
- **Transform key**: `export_to_gm`
- **No explicit log for the value**. Check that `export_to_gm` transform succeeded via `[SUMMARY]`.

### `initialize_mrope_delta_cache.enabled`
- **Transform key**: `initialize_mrope_delta_cache`
- **Check**: `[SUMMARY]` line.

---

## General Failure Patterns

These log patterns indicate something went wrong regardless of specific config:
- `"Transform <name> failed: <error>"` — transform errored and was skipped
- `"Falling back"` — a feature was attempted but reverted
- `"Skipping"` — a step was intentionally skipped (may or may not be a problem)
