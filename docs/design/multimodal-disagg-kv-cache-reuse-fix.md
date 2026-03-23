# Design Document: Fix Multimodal KV Cache Block Reuse in Disaggregated Serving

**Authors:** ibhosale
**Date:** March 20, 2026
**Status:** Implemented (workaround); full optimization pending
**Components:** KV Cache Manager (C++), PyExecutor (Python), CacheFormatter (C++)
**Related files:**
- `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp`
- `tensorrt_llm/_torch/pyexecutor/py_executor.py`
- `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`
- `cpp/tensorrt_llm/batch_manager/blockKey.cpp`
- `cpp/include/tensorrt_llm/batch_manager/kvCacheUtils.h`

---

## 1. Problem Statement

Multimodal disaggregated serving (Encoder-Prefill-Decode) crashes when
`enable_block_reuse: true`. The crash occurs during the Prefill→Decode KV cache
transfer on the very first multimodal request.

### 1.1 Observed Crash: Reuse Tree Assertion Failure

When `enable_block_reuse: true` and `enable_partial_reuse: true` (the default),
the Prefill worker hits a fatal assertion during the async KV cache transfer:

```
[TensorRT-LLM][WARNING] [kv cache manager] storeContextBlocks: Can not find sequence for request 1
python3: ../include/tensorrt_llm/batch_manager/kvCacheUtils.h:84:
  static tensorrt_llm::batch_manager::kv_cache_manager::BlockRange
  tensorrt_llm::batch_manager::kv_cache_manager::BlockRange::fromReuseTree(
    tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager&,
    const tensorrt_llm::batch_manager::kv_cache_manager::BlockKey&,
    int32_t):
  Assertion `lastBlock' failed.

  TLLM_CHECK_WITH_INFO: Couldn't find the requested block in the reuse tree
```

Full stack trace at the point of crash:

```
Thread 93 (LWP 40335):
#0  __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
#1  __GI_abort () at abort.c:79
#2  __assert_fail_base (...) at assert.c:92
#3  __GI___assert_fail (...) at assert.c:101
#4  tensorrt_llm::batch_manager::kv_cache_manager::BlockRange::fromReuseTree(...)
      at ../include/tensorrt_llm/batch_manager/kvCacheUtils.h:84
#5  tensorrt_llm::batch_manager::kv_cache_manager::getBlockRangeForSending(...)
      at cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp:245
#6  tensorrt_llm::batch_manager::kv_cache_manager::CacheFormatter::format(...)
      at cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp:277
#7  tensorrt_llm::batch_manager::CacheTransmitter::sendSync(...)
      ...
```

### 1.2 Observed Crash: Sequence Not Found (After Bypassing Reuse Tree)

When the reuse tree is bypassed and `fromAllBlockIds` is used instead (e.g.,
by routing multimodal requests away from `fromReuseTree`), a second crash
occurs because the sequence has already been removed by early termination:

```
[TensorRT-LLM][WARNING] [kv cache manager] storeContextBlocks: Can not find sequence for request 1
[TensorRT-LLM][ERROR] Exception in sendAndRemoveResponse: unordered_map::at request id: 1
```

This happens because `_handle_responses()` terminates the request (removing
the sequence from the KV cache manager) before the async sender thread
calls `fromAllBlockIds(requestId)`.

### 1.3 Related User-Facing Error (Dynamo / Raw Embeddings Path)

Users on the Dynamo side also observed a related error when sending the same
multimodal request twice with KV cache enabled (raw embeddings path):

```
tensorrt_llm.executor.utils.RequestError: Multimodal token count mismatch:
  found 0 image tokens in input_ids but received 2046 image embeddings.
  This is likely due to KV cache reuse, chunk prefill, or other optimizations
  that cause token count mismatches within the inference batch.
```

This occurs on the **second identical request** because KV cache reuse replaces
the image placeholder tokens with cached K/V, but the engine still receives
the full embeddings. This is a separate issue from the disagg crash but shares
the same root area (multimodal + KV cache reuse interaction).

Reference: [TRT-LLM PR #11967](https://github.com/NVIDIA/TensorRT-LLM/pull/11967),
[NVBug 5956071](https://nvbugspro.nvidia.com/bug/5956071)

### 1.4 Reproduction

**Environment:**
- TRT-LLM release container: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc8`
- GPU: NVIDIA B200 (compute capability 9.0)
- Model: Qwen3-VL-30B-A3B-Instruct-FP8

**Config triggering the crash:**
```yaml
kv_cache_config:
  enable_block_reuse: true    # This triggers the crash
  # enable_partial_reuse defaults to true when block_reuse is true
```

**Steps:**
1. Set up E-P-D disaggregated serving with any multimodal model
2. Enable `enable_block_reuse: true`
3. Send any multimodal request (image + text prompt)
4. Crash occurs during Prefill→Decode KV cache transfer on the first request

**Workaround (before fix):** Set `enable_block_reuse: false`. This disables all
KV cache reuse (including for text-only requests), which is a significant
performance regression for production serving.

**Test to reproduce:**
```bash
LLM_MODELS_ROOT=/tmp pytest tests/unittest/_torch/multimodal/test_mm_encoder_standalone.py \
  -k "test_epd_disagg_mm_hash_kv_cache_reuse and prompts0" -v -s
```

---

## 2. Background

### 2.1 KV Cache Block Reuse

TRT-LLM uses a paged KV cache where GPU memory is divided into fixed-size
**blocks** (default 32 tokens per block). When `enable_block_reuse` is enabled,
completed blocks are stored in a **reuse tree** (radix/prefix tree) indexed by
token content. Subsequent requests with matching prefixes skip K/V computation
and reuse existing blocks.

Each block is identified by a `BlockKey`:

```cpp
struct BlockKey {
    VecUniqueTokens uniqueTokens;  // Token IDs in this block
    bool usesExtraIds;             // Whether extra IDs (mm hashes) are present
    IdType loraTaskId;             // LoRA adapter ID
    MmKeys extraKeys;              // Multimodal hash keys (per-block)
    SizeType32 cacheSaltID;        // Cache partition ID
};
```

Two `BlockKey`s must match exactly (all fields, including `extraKeys`) for reuse.

### 2.2 Multimodal `extraKeys`

For multimodal requests, placeholder tokens (e.g., `<image>`) all share the same
token ID. The actual K/V values depend on the image content. To prevent false
cache hits, `generateBlockHashExtraKeys()` adds `MmKey` entries (hash + offset)
to blocks that overlap with multimodal token positions:

```
Block 0 (text only):   extraKeys = []
Block 1 (image start): extraKeys = [{hash=0xABC, offset=0}]
Block 2 (image cont.): extraKeys = [{hash=0xABC, offset=32}]
Block 3 (text only):   extraKeys = []
```

These `extraKeys` are **per-block** — they vary based on each block's overlap
with multimodal content.

### 2.3 Disaggregated Serving (E-P-D) KV Cache Transfer

In disaggregated mode, Prefill and Decode run on separate GPUs. After Prefill
computes K/V, the blocks must be transferred to Decode via network (NIXL/UCX).

`getBlockRangeForSending()` in `cacheFormatter.cpp` determines which blocks to
send. It has two code paths:

| Path | Lookup method | Use case |
|------|---------------|----------|
| `fromReuseTree(blockKey)` | By BlockKey in the reuse tree | Partial transfer: only send blocks the Decode worker doesn't already have |
| `fromAllBlockIds(requestId)` | By request ID in sequence map | Full transfer: send all blocks for the request |

### 2.4 Early Request Termination (`enable_partial_reuse_for_disagg`)

When `enable_block_reuse=True`, `enable_partial_reuse` defaults to `True`,
which enables `enable_partial_reuse_for_disagg`. This optimization:

1. **Stores and pins blocks** in the reuse tree during `start_transfer()`
2. **Terminates the request immediately** in `_handle_responses()` — freeing
   the sequence metadata (request ID → block mapping)
3. The async sender thread uses `fromReuseTree` to find blocks by `BlockKey`
   (does not need the request ID)

This allows the Prefill worker to accept new requests without waiting for the
KV cache transfer to complete.

---

## 3. Root Cause Analysis

Two bugs combine to cause the crash:

### Bug 1: `fromReuseTree` cannot handle per-block multimodal `extraKeys`

`BlockRange::fromReuseTree()` calls `findBlocksInReuseTreeByBlockKey()`, which
receives a single `BlockKey` (for the last block) and iterates backwards to find
all blocks in the sequence. For each sub-block, it **copies the top-level
`extraKeys`** to use as the search key.

This is incorrect for multimodal: each block has unique `extraKeys` based on its
position relative to image tokens. The copied keys don't match the stored keys,
so the lookup fails:

```
Stored:  Block 1 extraKeys = [{hash=0xABC, offset=0}]
Search:  Block 1 extraKeys = [{hash=0xABC, offset=32}]  ← copied from last block
                                                           MISMATCH → not found
```

Additionally, the Decode side constructs `lastBlockKey` from its own `LlmRequest`,
which does **not** carry multimodal metadata (`multimodal_hashes`,
`multimodal_positions`). The resulting key has empty `extraKeys`, guaranteeing
a mismatch with any multimodal block.

**Result:** `TLLM_CHECK_WITH_INFO(lastBlock, "Couldn't find the requested block in the reuse tree")` assertion fires.

### Bug 2: Sequence removed before async sender can use `fromAllBlockIds`

If we bypass `fromReuseTree` and use `fromAllBlockIds` instead, it looks up
blocks by request ID. But `enable_partial_reuse_for_disagg` causes
`_handle_responses()` to terminate the request **immediately** — removing the
sequence from the KV cache manager — while the async sender thread is still
running.

```
Timeline (race condition):
  t=0  start_transfer()         → stores blocks, pins them
  t=1  respond_and_send_async() → launches async sender thread
  t=2  _handle_responses()      → TERMINATES request, removes sequence
  t=3  async sender thread      → fromAllBlockIds(requestId) → sequence gone!
                                  → unordered_map::at crash
```

---

## 4. Implemented Fix (Workaround)

### 4.1 C++ Change: Route multimodal to `fromAllBlockIds`

**File:** `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp`

Added `senderHasMultimodalBlockKeys()` to detect multimodal requests and added
`senderHasMultimodal` to the condition that selects the `fromAllBlockIds` path:

```cpp
bool senderHasMultimodalBlockKeys(LlmRequest const& llmRequest)
{
    auto const mmHashes = llmRequest.getMultimodalHashes();
    return mmHashes.has_value() && *mmHashes && !(*mmHashes)->empty();
}

// In getBlockRangeForSending():
bool const senderHasMultimodal = senderHasMultimodalBlockKeys(llmRequest);

if (poolNum > 1 || !cacheManager->isEnableBlockReuse()
    || !cacheManager->isEnablePartialReuse()
    || lastBlockKey.uniqueTokens.size() == 0
    || recvSideHasCP || ppSize > 1
    || senderHasMultimodal)            // ← NEW: bypass reuse tree for MM
{
    return BlockRange::fromAllBlockIds(...);
}
```

**Effect:** Multimodal requests always send all blocks (full transfer), avoiding
the broken reuse tree lookup. Text-only requests are unaffected.

### 4.2 Python Change: Skip early termination for multimodal requests

**File:** `tensorrt_llm/_torch/pyexecutor/py_executor.py`

Modified `_handle_responses()` to not terminate multimodal requests early,
keeping the sequence alive for `fromAllBlockIds`:

```python
has_multimodal = request.multimodal_hashes is not None
if (self.enable_partial_reuse_for_disagg
    and not self.kv_cache_manager.is_vswa
    and self.dist.pp_size == 1
    and not has_multimodal):           # ← NEW: skip MM requests
    requests_to_terminate.append(request)
else:
    if not request.is_disagg_context_transmission_state:
        requests_to_terminate.append(request)
```

**Effect:** Multimodal requests follow the standard disagg lifecycle:
1. `start_transfer()` → stores blocks, sets state to `DISAGG_CONTEXT_TRANS_IN_PROGRESS`
2. `_handle_responses()` → skips early termination → sequence stays alive
3. Async sender → `fromAllBlockIds(requestId)` → sequence found → transfer succeeds
4. `_end_transfer_and_maybe_terminate()` → cleans up after transfer completes

Non-multimodal requests continue using the fast early-termination path.

---

## 5. What This Fix Achieves vs. Doesn't

### Achieved

| Capability | Status |
|-----------|--------|
| Multimodal + disagg + `enable_block_reuse=true` | **Works** (was crashing) |
| Text-only disagg partial reuse | **Unchanged** (still fast path) |
| Single-worker multimodal block reuse | **Unchanged** |
| Correctness of multimodal KV cache transfers | **Correct** |

### Not Yet Achieved (Future Work)

| Gap | Impact | Required changes |
|-----|--------|-----------------|
| No partial KV transfer for multimodal disagg | Repeated MM requests re-send all blocks even if Decode has them cached | Fix `findBlocksInReuseTreeByBlockKey` + propagate MM metadata to Decode |
| No early termination for MM disagg requests | Prefill holds sequence slightly longer (until transfer completes) | Same as above — once `fromReuseTree` works for MM, early termination is safe |

### Performance Impact

- **Text-only disagg requests:** Zero impact. Same code path as before.
- **Multimodal disagg requests (first occurrence):** No impact — full transfer
  is required anyway since Decode has no cached blocks.
- **Multimodal disagg requests (repeated same image):** Sub-optimal. Prefill
  sends all blocks instead of only the delta. Network bandwidth overhead is
  proportional to the number of cached blocks Decode already has.
- **Single-worker (non-disagg):** Zero impact. The fix only touches the
  disaggregated transfer path.

---

## 6. Full Fix Design (Future Work)

To enable partial KV transfer for multimodal disagg requests, three changes
are needed:

### 6.1 Fix `findBlocksInReuseTreeByBlockKey` for per-block `extraKeys`

**File:** `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`

Currently copies the last block's `extraKeys` to all sub-blocks. Must instead
reconstruct per-block `extraKeys` using `generateBlockHashExtraKeys()` for each
block's token range:

```cpp
// Current (broken for multimodal):
subBlockKey.extraKeys = topLevelKey.extraKeys;  // same for all blocks

// Fixed:
subBlockKey.extraKeys = generateBlockHashExtraKeys(
    llmRequest, blockStartTokenIdx, blockEndTokenIdx);  // per-block
```

This requires passing the `LlmRequest` (or at minimum multimodal positions,
lengths, and hashes) into `findBlocksInReuseTreeByBlockKey`.

### 6.2 Propagate multimodal metadata to Decode side

The Decode worker's `LlmRequest` must carry `multimodal_hashes`,
`multimodal_positions`, and `multimodal_lengths` so it can construct correct
`BlockKey`s. Options:

- **Via `DisaggregatedParams`:** Add fields to the generation-only request's
  `DisaggregatedParams` during the orchestrator handoff.
- **Via `CacheTransceiverConfig`:** Embed MM metadata in the transfer protocol
  so the Decode worker receives it alongside the KV data.

### 6.3 Reconstruct correct `lastBlockKey` on Decode side

With multimodal metadata available, the Decode side can call
`generateBlockHashExtraKeys()` for its last block and produce a `lastBlockKey`
that matches what the Prefill side stored.

### Estimated Scope

- `kvCacheManager.cpp`: Modify `findBlocksInReuseTreeByBlockKey` signature and logic
- `blockKey.cpp`: Possibly refactor `generateBlockHashExtraKeys` for reuse
- `cacheFormatter.cpp`: Remove the `senderHasMultimodal` bypass
- `py_executor.py`: Remove the `has_multimodal` early-termination skip
- Disagg params / transceiver: Add MM metadata propagation
- Tests: Verify partial transfer works with multimodal

---

## 7. Testing

### Unit Test

`tests/unittest/_torch/multimodal/test_mm_encoder_standalone.py`:
- `test_epd_disagg_mm_hash_kv_cache_reuse`: Full E-P-D flow with
  `enable_block_reuse=True`, verifies multimodal requests complete without crash
- `test_epd_disagg_output_matches_raw_with_block_reuse`: Verifies output
  correctness matches non-disagg path

### Manual Verification

```bash
LLM_MODELS_ROOT=/tmp pytest tests/unittest/_torch/multimodal/test_mm_encoder_standalone.py \
  -k "test_epd_disagg_mm_hash_kv_cache_reuse and prompts0" -v -s
```

### CI

The fix should be validated on GPU nodes with disaggregated serving enabled.
Tests require multi-GPU setup for true E-P-D flow.

---

## 8. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Regression in text-only disagg | Low | Text-only path is completely unchanged; no new conditions in that flow |
| Regression in single-worker MM | None | Fix only touches disagg transfer path (`getBlockRangeForSending` and `_handle_responses` disagg branch) |
| Memory pressure from delayed MM sequence cleanup | Low | Sequences are held only for transfer duration (seconds); standard eviction handles memory pressure |
| `request.multimodal_hashes` attribute missing on some request types | Low | Property is defined via nanobind on all `GenLlmReq` objects; returns `None` for non-MM requests |

---

## 9. References

- KV Cache Manager: `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`
- Block Key generation: `cpp/tensorrt_llm/batch_manager/blockKey.cpp`
- Reuse tree lookup: `cpp/include/tensorrt_llm/batch_manager/kvCacheUtils.h`
- Cache transfer formatting: `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp`
- PyExecutor disagg flow: `tensorrt_llm/_torch/pyexecutor/py_executor.py`
- Async transfer manager: `tensorrt_llm/_torch/pyexecutor/py_executor.py` (class `AsyncTransferManager`)
- Nanobind bindings: `cpp/tensorrt_llm/nanobind/batch_manager/bindings.cpp`
