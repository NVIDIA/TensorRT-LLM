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
MODEL    model_loader: mixed_nvfp4_fp8 -> FP8 KV cache; float16/bfloat16 -> unquantized (must equal model dtype);
         dsa/deepseek_v4 keep their NVFP4 pool as the mixed-precision low pool
MANAGER  KVCacheManagerV2:
           _init_mixed_precision          (Blackwell, dtype checks, layers sharing another layer's KV cache stay native)
           mixed-precision low pool sized as packed NVFP4 (dense) / native NVFP4 (DeepSeek sparse attention, DeepSeek-V4)
           _add_mixed_precision_layers:   per local mixed-precision layer l, append an attention layer
               (high-precision dtype, sliding window last_n + 1, sink tokens first_n, ReusePolicy.REQUIRED)
               -> one extra layer group
           mixed_precision_high_layers[l] = mixed-precision high layer id
           disaggregated ids: for global model layer L = pp_layers[l], its mixed-precision high layer gets 2L + 1,
             every other layer 2 * id
         native core: ordinary full-attention life cycle (mixed-precision low pool)
                    + sliding-window-with-sinks life cycle (mixed-precision high pool)
REQUEST  prefix match in the radix tree; the REQUIRED mixed-precision high layer group needs the mixed-precision
         high sink pages and mixed-precision high window pages at the endpoint, so the match shrinks to an
         endpoint they cover; a partial last block is copied to a private page
STEP     metadata.prepare():
           staging-copy path (TRTLLM, FlashInfer): build the mixed-precision staging copy once per metadata
             object; per step fill page tables and lengths [cached, cached + new] into fixed device buffers
             (the TRTLLM attention backend also builds the block offsets of the mixed-precision staging copy)
           high-row overlay path (DeepSeek sparse attention, DeepSeek-V4): refresh the static mixed-precision
             high page table(s); -1 marks pages not held
FORWARD  per mixed-precision layer:
           staging-copy path: gather each token t into the mixed-precision staging copy, from the
             mixed-precision high pool if held there, else decoded from the mixed-precision low pool;
             swap the attention view to the mixed-precision staging copy; unchanged kernel appends and
             reads; restore; write back: encode new tokens into the mixed-precision low pool (always),
             copy them into the mixed-precision high pool (when held)
           high-row overlay path: append into the NVFP4 pool; store new rows into the mixed-precision
             high pool; after the top-k gather into FP8 scratch, overlay the rows the mixed-precision
             high pool holds
UPDATE   context: commit per block_reuse_config.policy (the model's layer groups and the
           mixed-precision high layer group together)
         generation: resize(capacity - rewound draft slots, history); the mixed-precision high window
           slides and stale mixed-precision high pages are released; generation blocks are committed
           only with all_reusable
FREE     close the request cache; committed pages stay in the radix tree until evicted
EVICT    mixed-precision high sink page evicted -> whole prefix subtree removed;
         mixed-precision high window page evicted -> later matches shorter
```

## KV Cache Manager V2 Reuse Policies (Per Life Cycle)

`ReusePolicy` is defined in the configuration header of the KV cache manager
V2 native core and bound to Python.

| Policy | Prefix match | Commit | Resume |
|---|---|---|---|
| `REQUIRED` | Constrains the match: sinks and the live sliding window for sliding-window life cycles; every block for full attention | Yes | Claimed with the match |
| `PRIVATE` | Ignored | Never | Fresh pages |
| `OPTIONAL` | Does not constrain the match | Yes | Selected at resume (`optional_reuse_groups`) only when that layer group's reuse status is complete; otherwise fresh pages to recompute |
| `SINK_REQUIRED_ONLY` | Sinks only | Yes | Sinks claimed with the match; sliding window selectable at resume |

`block_reuse_config.policy` (`all_reusable`, `per_request`,
`per_conversation`) is different: it controls **when** blocks are committed
for the whole manager. It does not set the per-life-cycle policy.

## Recipe: Mixed-Precision KV Cache in a New Attention Backend

**Staging-copy path.** The backend reads one plain paged KV cache.

1. Allow the backend in `TorchLlmArgs.validate_mixed_precision_kv`.
2. In the metadata's `prepare()`, call
   `prepare_mixed_precision_kv(layout, tokens_per_block)` when
   `_has_mixed_precision_staging()` is true. Choose the layout and page size
   your kernels need; pages of the mixed-precision staging copy may be
   smaller than the KV cache manager's pages.
3. Does the backend read manager-derived tensors besides the manager and the
   layout? If so:
   - build them in `_on_mixed_precision_kv_prepared` from the view of the
     mixed-precision staging copy;
   - swap them in `_swap_mixed_precision_kv`, and restore them in
     `_restore_mixed_precision_kv`.

   The TRTLLM attention backend does this for its block offsets.
4. Make sure each layer's forward runs inside
   `mixed_precision_kv_staging(layer_idx)`. `Attention.forward` and the
   multi-head latent attention (MLA) forward already do.
5. Keep the lengths of the mixed-precision staging copy in sync when device
   KV cache lengths change after `prepare()`, and swap in the draft
   mixed-precision staging copy for draft forwards.
6. Disable any fast path that bypasses `prepare()` or appends K/V outside
   the mixed-precision staging copy.
7. Extend the backend parametrization and the CUDA-graph and length tests in
   `tests/unittest/_torch/attention/test_mixed_precision_backends.py`.

**High-row overlay path, own pool in the low-precision format.** For
sparse-MLA-style backends.

1. On the KV cache manager, set `_supports_mixed_precision_kv = True` and
   `_own_pool_is_mixed_precision_low = True`. On the backend's attention
   metadata class, override `_has_mixed_precision_staging()` to return
   `False`, as the DeepSeek sparse attention (DSA) metadata does. An override
   on the manager is never called.
2. Allocate a static mixed-precision high page table with the graph-aware
   buffer allocator, and refresh it in every `prepare()` with the manager's
   page-table fill.
3. Store new rows into the mixed-precision high pool after the append, and
   overlay the mixed-precision high rows after the gather, using the helpers
   in `sparse/mixed_precision_high_rows.py`.
4. Swap or hide the mixed-precision high page table for draft forwards.
5. Allow the sparse algorithm in the validator, and in the model loader's
   mixed-precision low pool mapping.

## Recipe: A New Low-Precision Storage Format

The low-precision format is chosen by the dtype name `mixed_<low>_<high>`.

1. **Configuration.**
   - Extend the dtype pattern and the list of accepted names in `llm_args.py`
     (`_MIXED_PRECISION_KV_DTYPE`, `_MIXED_PRECISION_KV_DTYPES`).
   - Map the name in the model loader's KV cache quantization step.
   - Regenerate the LLM-args golden manifest and the serve schemas, and
     document the name in `kvcache.md`.
2. **Pool sizing.** Define the encoded size of one token-head record, padded
   to the record alignment. The manager sizes the mixed-precision low pool
   from it through `get_mixed_precision_kv_row_bytes()`.
3. **Conversion kernels.**
   - Add a low-precision codec next to the NVFP4 one in
     `tensorrt_llm/_torch/attention/kernels/blackwell/mixed_precision_kv.py`,
     and register it in `_LOW_PRECISION_CODECS`. It provides decode, for the
     gather into the mixed-precision staging copy, and encode, for the
     write-back.
   - `convert_mixed_precision_kv()` compiles one kernel per key: dtypes,
     shapes, and strides. Warm up every key the deployment uses before CUDA
     graph capture.
   - Follow `../../trtllm-kv-cache-compression/references/adding-kernels.md`.
4. **Own-pool backends (high-row overlay path).** A backend whose native pool
   already uses the low-precision format needs only the high-row store and
   overlay, not the low-precision codec.
5. **Tests.** Round-trip invariants of the low-precision codec, paged
   conversion with non-contiguous pages, CUDA-graph replay with changing
   lengths, and the manager's pool-size checks. Then, locally, check the
   codec's encode/decode round-trip error for every supported
   high-precision dtype on real KV cache data, and evaluate accuracy on real
   runs against the FP8 KV cache and against a KV cache entirely in the
   low-precision format. Write the commands and results in the PR
   description; pytest keeps only the checks above.

## Recipe: A New KV Cache Manager or Model Family

- Set `_supports_mixed_precision_kv = True` on the manager. Without it,
  construction raises.
- Restrict layers by overriding `_is_mixed_precision_layer`. Override
  `_mixed_precision_rows_per_page` when a page does not hold
  `tokens_per_block` rows.
- If the manager overrides pool-role mapping, map the appended
  mixed-precision high layers' roles too.
- If the manager has virtual layers, provide the mapping that the
  disaggregated layer-id function uses.
- Update the supported and unsupported manager sets in the tests. Add a GPU
  test on a registered Blackwell stage.

## Recipe: Changing Reuse for the Mixed-Precision High Layer Group

With the `REQUIRED` policy, a missing mixed-precision high window ends the
match. To let a match survive it, the design must provide another source for
those tokens. The mixed-precision low pool always holds the full history;
recomputation is the other option. The steps are:

1. Choose the per-life-cycle policy (`SINK_REQUIRED_ONLY` or `OPTIONAL`)
   where the mixed-precision high layers are added.
2. In a `_resume_context_cache` override, select the layer groups whose reuse
   status is complete. `CSA2CacheManager` is the existing example.
3. Fill unselected mixed-precision high pages before attention reads them:
   decode them from the mixed-precision low pool, or recompute them. Show
   that this is correct on the TRTLLM, FlashInfer, and sparse attention
   backends, under CUDA graphs and the overlap scheduler, in local real runs.
4. Check that disaggregated transfer reads `reuse_status` correctly.
5. Test match length and selection behavior in
   `tests/unittest/kv_cache_manager_v2_tests/`, and the reused depth at
   manager level in
   `tests/unittest/_torch/pyexecutor/kv_cache/test_mixed_precision_kv_cache.py`.
6. Measure accuracy with forced reuse, and the per-request reused depth end
   to end, against the `REQUIRED` baseline, in local real runs. Keep the
   baseline configuration unchanged. Write the commands and results of
   steps 3 and 6 in the PR description.

A **new** C++ reuse policy is a change to the KV cache manager V2 native
core. Follow the high-risk rules of the KV cache manager V2 C++ guide
(`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md`):
- the enum;
- the life-cycle predicates and validation;
- match pruning;
- the subtree eviction rule;
- claim, resume and commit paths;
- bindings and stubs;
- C++ tests;
- the KV cache manager V2 C++ guide itself, which should list every policy.
