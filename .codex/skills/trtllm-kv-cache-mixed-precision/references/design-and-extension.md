<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mixed-Precision KV Cache: Data Flow and Extension Recipes

Symbols are named rather than given line numbers. Locate them on the current
main before relying on them.

## Data Flow

```text
CONFIG   kv_cache_config.dtype = mixed_nvfp4_<fp8|float16|bfloat16>, mixed_precision_first_n/last_n
         KvCacheConfig validators; TorchLlmArgs.validate_mixed_precision_kv
MODEL    model_loader: mixed_nvfp4_fp8 -> FP8 KV; float16/bfloat16 -> unquantized (must equal model dtype);
         dsa/deepseek_v4 keep their NVFP4 pool as the low pool
MANAGER  KVCacheManagerV2:
           _init_mixed_precision          (Blackwell, dtype checks, shared-KV layers stay native)
           low pools sized as packed NVFP4 (dense) / native NVFP4 (DSA, DeepSeek-V4)
           _add_mixed_precision_layers:   per mixed local layer l append an attention layer
               (high dtype, window last_n + 1, sinks first_n, ReusePolicy.REQUIRED) -> one extra layer group
           mixed_precision_high_layers[l] = high layer id
           disaggregated ids: for global model layer L = pp_layers[l], high(L) = 2L + 1, every other layer 2 * id
         C++ core: ordinary full-attention life cycle (low) + sliding-window-with-sinks life cycle (high)
REQUEST  prefix match in the radix tree; the REQUIRED high group needs its sink and live-window pages at the
         endpoint, so the match shrinks to an endpoint they cover; a partial last block is copied to a private page
STEP     metadata.prepare():
           path A (TRTLLM, FlashInfer): build staging once per metadata object; per step fill page tables and
             lengths [cached, cached + new] into fixed device buffers (TRTLLM also builds staging block offsets)
           path B (DSA, DeepSeek-V4): refresh the static high page table(s); -1 marks pages not held
FORWARD  per mixed layer:
           A: gather staging[t] = high[t] if held else decode(low[t]); swap attention view to staging;
              unchanged kernel appends and reads; restore; write back: encode new tokens into low (always),
              copy into high (when held)
           B: append into the NVFP4 pool; store new rows into high; after top-k gather into FP8 scratch,
              overlay the rows high holds
UPDATE   context: commit per block_reuse_config.policy (low and high groups together)
         generation: resize(capacity - rewound draft slots, history); the high window slides and stale high
         pages are released; generation blocks are committed only with all_reusable
FREE     close the request cache; committed pages stay in the tree until evicted
EVICT    high sink page evicted -> whole prefix subtree removed; high window page evicted -> later matches shorter
```

## KVCM V2 Reuse Policies (Per Life Cycle)

`ReusePolicy` is defined in the C++ core configuration header and bound to
Python.

| Policy | Prefix match | Commit | Resume |
|---|---|---|---|
| `REQUIRED` | Constrains the match: sinks and live window for windowed life cycles; every block for full attention | Yes | Claimed with the match |
| `PRIVATE` | Ignored | Never | Fresh pages |
| `OPTIONAL` | Does not constrain the match | Yes | Selected at resume (`optional_reuse_groups`) only when that group's reuse status is complete; otherwise fresh pages to recompute |
| `SINK_REQUIRED_ONLY` | Sinks only | Yes | Sinks claimed with the match; window selectable at resume |

`block_reuse_config.policy` (`all_reusable`, `per_request`,
`per_conversation`) is different: it controls **when** blocks are committed
for the whole manager. It does not set the per-life-cycle policy.

## Recipe: Mixed Precision in a New Attention Backend

**Path A, staging.** The backend reads one plain paged KV cache.

1. Allow the backend in `TorchLlmArgs.validate_mixed_precision_kv`.
2. In the metadata's `prepare()`, call
   `prepare_mixed_precision_kv(layout, tokens_per_block)` when
   `_has_mixed_precision_staging()` is true. Choose the layout and page size
   your kernels need; staging pages may be smaller than manager pages.
3. Does the backend read manager-derived tensors besides the manager and the
   layout? If so:
   - build them in `_on_mixed_precision_kv_prepared` from the staging view;
   - swap them in `_swap_mixed_precision_kv`, and restore them in
     `_restore_mixed_precision_kv`.

   TRTLLM does this for its block offsets.
4. Make sure each layer's forward runs inside
   `mixed_precision_kv_staging(layer_idx)`. `Attention.forward` and the MLA
   forward already do.
5. Keep staging lengths in sync when device KV lengths change after
   `prepare()`, and swap in the draft staging for draft forwards.
6. Disable any fast path that bypasses `prepare()` or appends K/V outside
   staging.
7. Extend the backend parametrization and the CUDA-graph and length tests in
   `tests/unittest/_torch/attention/test_mixed_precision_backends.py`.

**Path B, own low-format pool.** For sparse-MLA-style backends.

1. On the cache manager, set `_supports_mixed_precision_kv = True` and
   `_own_pool_is_mixed_precision_low = True`. On the backend's attention
   metadata class, override `_has_mixed_precision_staging()` to return
   `False`, as the DSA metadata does. An override on the manager is never
   called.
2. Allocate a static high page table with the graph-aware buffer allocator,
   and refresh it in every `prepare()` with the manager's page-table fill.
3. Store new rows into the high pool after the append, and overlay the high
   rows after the gather, using the helpers in
   `sparse/mixed_precision_high_rows.py`.
4. Swap or hide the table for draft forwards.
5. Allow the sparse algorithm in the validator, and in the model loader's
   low-pool mapping.

## Recipe: A New Low-Precision Storage Format

The low format is chosen by the dtype name `mixed_<low>_<high>`.

1. **Configuration.**
   - Extend the dtype pattern and the list of accepted names in `llm_args.py`
     (`_MIXED_PRECISION_KV_DTYPE`, `_MIXED_PRECISION_KV_DTYPES`).
   - Map the name in the model loader's KV-quantization step.
   - Regenerate the golden manifest and the serve schemas, and document the
     name in `kvcache.md`.
2. **Pool sizing.** Define the encoded size of one token-head record, padded
   to the record alignment. The manager sizes the low pool from it through
   `get_mixed_precision_kv_row_bytes()`.
3. **Conversion kernels.**
   - Add a codec next to the NVFP4 one in
     `tensorrt_llm/_torch/attention/kernels/blackwell/mixed_precision_kv.py`,
     and register it in `_LOW_PRECISION_CODECS`. It provides decode, for the
     staging gather, and encode, for the write-back.
   - `convert_mixed_precision_kv()` compiles one kernel per key: dtypes,
     shapes, and strides. Warm up every key the deployment uses before CUDA
     graph capture.
   - Follow `../../trtllm-kv-cache-develop/references/adding-kernels.md`.
4. **Own-pool backends (path B).** A backend whose native pool already uses
   the format needs only the high-row store and overlay, not the codec.
5. **Tests.** Round-trip invariants of the codec, paged conversion with
   non-contiguous pages, CUDA-graph replay with changing lengths, and the
   manager's pool-size checks. Then evaluate accuracy against FP8 KV and
   against an all-low-format cache.

## Recipe: A New Cache Manager or Model Family

- Set `_supports_mixed_precision_kv = True` on the manager. Without it,
  construction raises.
- Restrict layers by overriding `_is_mixed_precision_layer`. Override
  `_mixed_precision_rows_per_page` when a page does not hold
  `tokens_per_block` rows.
- If the manager overrides pool-role mapping, map the appended high layers'
  roles too.
- If the manager has virtual layers, provide the mapping that the
  disaggregated layer-id function uses.
- Update the supported and unsupported manager sets in the tests. Add a GPU
  test on a registered Blackwell stage.

## Recipe: Changing Reuse for the High Group

With the `REQUIRED` policy, a missing high window ends the match. To let a
match survive it, the design must provide another source for those tokens.
The low pool always holds the full history; recomputation is the other
option. The steps are:

1. Choose the per-life-cycle policy (`SINK_REQUIRED_ONLY` or `OPTIONAL`)
   where the high layers are added.
2. In a `_resume_context_cache` override, select the groups whose reuse status
   is complete. The CSA2 cache manager is the existing example.
3. Fill unselected high pages before attention reads them: decode them from
   the low pool, or recompute them. Show that this is correct on the TRTLLM,
   FlashInfer, and sparse paths, under CUDA graphs and overlap.
4. Check that disaggregated transfer reads `reuse_status` correctly.
5. Test match length and selection behavior in
   `tests/unittest/kv_cache_manager_v2_tests/`, and the end-to-end reused
   depth in the mixed-precision tests.
6. Measure accuracy with forced reuse, and the per-request reused depth,
   against the `REQUIRED` baseline. Keep the baseline configuration
   unchanged.

A **new** C++ reuse policy is a native-core change. Follow the KVCM V2 C++
guide's high-risk rules:
- the enum;
- the life-cycle predicates and validation;
- match pruning;
- the subtree eviction rule;
- claim, resume and commit paths;
- bindings and stubs;
- C++ tests;
- the C++ guide itself, which should list every policy.
