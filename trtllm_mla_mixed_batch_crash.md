# MLA Context thop.attention Crash on Second Invocation

## Summary

`thop.attention` with `is_mla_enable=True` and `attention_input_type=context_only`
crashes with `cudaStreamIsCapturing: illegal memory access` on the **second**
invocation of a context AttentionOp at a given `layer_idx`. The first call always
succeeds. The crash reproduces with both random BF16 and real FP8 weights.

Originally attributed to mixed prefill+decode batches, but further investigation
shows the crash occurs even with **pure prefill batches** (host_request_types = all
zeros) — it's specifically the second call to the same AttentionOp that crashes.

## Reproduction

**Branch**: `eg/ds_with_mla_enablement`  
**SHA**: `650a436eaf`  
**Fork**: `nv-auto-deploy/TensorRT-LLM`

### Steps

1. Remove the SDPA mixed-batch fallback in `_handle_prefill_thop` (lines ~862-949)
   so ALL batches go through C++ FMHA. Replace the `if not is_mixed_batch` / `else`
   block with a single unconditional `wrapper.plan()` + `wrapper.run()` call.

2. Run:
```bash
# Create 2-request dataset
head -2 DeepSeek-V3-Lite_128_128_64.inp > ds_2req.inp

# Config (ds_ad.yaml):
# runtime: trtllm
# attn_backend: trtllm
# compile_backend: torch-simple
# model_factory: AutoModelForCausalLM
# skip_loading_weights: true
# max_seq_len: 512
# kv_cache_config: {}
# transforms:
#   insert_cached_mla_attention:
#     backend: trtllm_mla
#   fuse_rope_into_trtllm_mla:
#     stage: cache_init
#     enabled: true

trtllm-bench --model deepseek-ai/DeepSeek-V3-Lite \
    --model_path ${LLM_MODELS_ROOT}/DeepSeek-V3-Lite/fp8 \
    throughput --dataset ds_2req.inp \
    --backend _autodeploy --max_batch_size 64 --max_num_tokens 4096 \
    --extra_llm_api_options ds_ad.yaml
```

3. The benchmark tool runs a 2-request warm-up, then a 2-request main iteration.
   The warm-up creates 2 sequences. During the main iteration, the scheduler creates
   a mixed batch: 1 new prefill + 1 still-decoding warm-up sequence.

### Expected result
All requests complete (as they do with the PT backend).

### Actual result
```
RuntimeError: [TensorRT-LLM][ERROR] CUDA runtime error in
cudaStreamIsCapturing(stream, &status): an illegal memory access was encountered
(../include/tensorrt_llm/common/cudaUtils.h:157)
```

## Crash Details

- **Always fails on the 2nd context call** (31st `_call_thop_attention_mla` invocation,
  i.e., first layer of the second forward pass that has context sequences)
- **1st context call** (warm-up, pure prefill `pf=1, dec=0`): succeeds
- **2nd context call** (main batch, mixed `pf=1, dec=1`): crashes
- The crash is at the START of `thop.attention` (`cudaStreamIsCapturing`), before
  any kernel runs
- With `CUDA_LAUNCH_BLOCKING=1`, a `torch.cuda.synchronize()` immediately before
  the call succeeds — the CUDA context is clean
- K/V magnitudes are identical between the passing and crashing calls (~200)
- All scalar parameters match the PT backend exactly (verified via arg dump comparison)

## What We Verified

| Variation | Result |
|-----------|--------|
| Pure prefill batch (no decode in host_request_types) | Works |
| Mixed batch (pf=1, dec=1) | Crashes |
| skip_loading_weights=true (random BF16 weights) | Same crash |
| skip_loading_weights=false (real FP8 weights) | Same crash |
| layer_idx (shared with decode) | 2 failures/256 |
| layer_idx+30 (separate context op) | 1 failure/256 |
| Full metadata (unsliced, including decode entries) | Crashes |
| Sliced metadata (prefill-only) | Crashes |
| Zeroed decode block_offsets | Crashes |
| Persistent workspace (shared with decode) | Crashes |
| Fresh empty workspace each call | Crashes |
| torch.cuda.synchronize() before wrapper.run() | Crashes |
| Warmup the context op during decode (update_kv_cache=False) | Crashes |
| Direct _call_thop_attention_mla (bypass wrapper) | Crashes |

## Why PT Backend Works

The PT backend also processes mixed batches with full metadata passed to
`thop.attention` with `attention_input_type=context_only`. PT uses the same
`layer_idx` for both `self.mha` (context) and `self.mqa` (decode) wrappers.
Since these have different `head_size` (192 vs 576) and `num_kv_heads` (32 vs 1),
the C++ code creates separate AttentionOps keyed by the full parameter set.

Both the PT context op and the AD context op have the same parameters:
- head_size=192, num_kv_heads=32, v_head_dim=128
- is_fused_qkv=False, attention_input_type=context_only
- q_lora_rank=2560, quant_mode=1024 (FP8 block scaling)
- Same kv_cache_block_offsets encoding (page_id * 30)

The only remaining difference: PT's context op shares `layer_idx` with decode
(the C++ code creates a separate op based on param hash), while AD uses
`layer_idx+30` (explicit separation). This may affect how the C++ code
initializes internal state for the context op.

## Update: Not a Mixed-Batch Issue — Second-Call Crash

### PT backend never sends decode entries to context kernel

Dumped PT backend's `host_request_types` for all 5 captured context calls:
```
Call 1: host_request_types = [0]           (1 context sequence)
Call 2: host_request_types = [0,0,0,0,0,0,0,0]  (8 context sequences)
Call 3: host_request_types = [0]
Call 4: host_request_types = [0]
Call 5: host_request_types = [0]
```

**PT ONLY passes context sequences to the context kernel.** Decode entries are
never present. The PT scheduler/metadata pipeline filters them out before the
`self.mha.forward()` call.

### Slicing metadata to context-only doesn't fully fix it

After slicing all per-sequence metadata to `[:num_prefill]` (matching PT's
behavior), the crash persists: **255/256 pass, 1 failure**.

The failing call was confirmed to be a **pure prefill batch** (after slicing):
```
[CTX CRASH] pf=1 tokens=128 err=CUDA error: an illegal memory access was encountered
```

This is NOT a mixed-batch issue. It's the **second invocation** of the context
AttentionOp at `layer_idx+30` that crashes:
- **Call 1** (warm-up forward, layer 0, pure prefill, `pf=1`): succeeds
- **Call 31** (main-batch forward, layer 0, pure prefill after slicing, `pf=1`): crashes

Both calls have identical parameters: same `host_request_types=[0]`, same
`num_tokens=128`, same Q/K/V magnitudes. The only difference: call 31 reuses
the C++ AttentionOp that was created by call 1.

### Theory: C++ AttentionOp second-call state corruption

The C++ AttentionOp caches internal state (workspace, stream handle, buffer
pointers) from the first call. On the second call:
- The workspace tensor may have been reallocated (different GPU address)
- The CUDA stream may have changed between forwards
- Internal buffer sizes may not match the new batch

The PT backend avoids this because its scheduling ensures context AttentionOps
are called consistently (same metadata shape patterns across forwards). The AD
scheduler produces different batch compositions between warm-up and main batch.

### Repro (SHA `9dd41a3a88`)

Same as before, but now the crash reproduces even with sliced (context-only)
metadata. The crash is on the second context call, regardless of batch
composition.

## Current Workaround

SDPA fallback for mixed batches (detects via `host_request_types`):
- Pure prefill → C++ FMHA (fast, ~90% of prefill iterations)
- Mixed prefill+decode → PyTorch SDPA + manual index_copy cache write (~10%)

This gives 256/256 pass with torch-simple, 5.7 tps/user.

Note: the SDPA fallback works because it's the mixed-batch calls that happen
to be the second invocation (first is always pure prefill during warm-up).
If the AD scheduler produced two consecutive pure-prefill calls, the second
would also crash.
