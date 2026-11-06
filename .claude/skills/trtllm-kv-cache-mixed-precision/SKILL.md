---
name: trtllm-kv-cache-mixed-precision
description: >-
  Develop, review, or debug the TensorRT-LLM mixed-precision KV cache, which
  stores the full KV history in NVFP4 and keeps the first N and last N tokens
  in the attention dtype (kv_cache_config.dtype mixed_nvfp4_fp8,
  mixed_nvfp4_float16 or mixed_nvfp4_bfloat16, mixed_precision_first_n,
  mixed_precision_last_n) on KV Cache Manager V2. Use when enabling it for a
  new attention backend, cache manager, model family, or speculative mode;
  when touching its high-precision layers, staging copy, high-row overlay,
  disaggregated transfer ids, or prefix-reuse behavior; and when measuring
  its accuracy or per-step cost.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Mixed-Precision KV Cache

The mixed-precision KV cache (a prototype) stores the KV history of every
full-attention layer in NVFP4, in the layer's own *low* pool. It also keeps
the first `mixed_precision_first_n` and the last `mixed_precision_last_n`
tokens of each sequence in a *high* pool, in the attention dtype. Attention
kernels are unchanged: before a mixed layer runs, its KV is presented to
attention in the attention dtype.

The user-facing contract is in `docs/source/features/kvcache.md`, section
"Mixed-Precision KV Cache (Prototype)". This skill covers the design, the
invariants, the extension steps, and the pitfalls. Load
`trtllm-kv-cache-develop` first. The data flow and extension recipes are in
`references/design-and-extension.md`.

## Required Context

1. `docs/source/features/kvcache.md`: "Mixed-Precision KV Cache (Prototype)",
   "SWA Endpoint Retention", "Partial Reuse".
2. `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md`: life
   cycles, sliding windows and sinks, reuse policies, eviction.
3. `tensorrt_llm/_torch/attention/ATTENTION_DEVELOPER_GUIDE.md`: metadata
   families, and how device KV lengths are refreshed after `prepare()`.
4. Code:
   - `tensorrt_llm/llmapi/llm_args.py`: the `KvCacheConfig` dtype and
     first/last-N fields, and `TorchLlmArgs.validate_mixed_precision_kv`.
   - `tensorrt_llm/_torch/pyexecutor/model_loader.py`: KV quantization mapping
     for the mixed dtypes.
   - `tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`:
     `_init_mixed_precision`, `_is_mixed_precision_layer`,
     `_add_mixed_precision_layers`, the mixed-precision page-table helpers,
     and the disaggregated layer-id mapping.
   - `tensorrt_llm/_torch/pyexecutor/kv_cache/mixed_precision_staging.py`.
   - `tensorrt_llm/_torch/attention/backends/interface.py`
     (`prepare_mixed_precision_kv`, `mixed_precision_kv_staging`), and the
     `trtllm.py` and `flashinfer.py` metadata.
   - `tensorrt_llm/_torch/attention/kernels/blackwell/mixed_precision_kv.py`
     (conversion kernels).
   - `tensorrt_llm/_torch/attention/backends/sparse/mixed_precision_high_rows.py`,
     for DSA and DeepSeek-V4.
   - `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py`.

## How to Use It

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: mixed_nvfp4_bfloat16      # or mixed_nvfp4_fp8, mixed_nvfp4_float16
  mixed_precision_first_n: 256
  mixed_precision_last_n: 256
  pool_ratio: [0.75, 0.25]         # optional; one entry more than the model's layer groups
```

The second dtype in the name is the attention dtype. It must match the model
dtype unless it is `fp8`. Confirm the result in the startup log:
- the mixed-precision layers and the extra high layer group;
- the staging size, for the TRTLLM and FlashInfer backends.

Then report reused depth and per-step cost (`perf-kv-cache-tuning`) next to
accuracy (`trtllm-kv-cache-accuracy-eval`).

## How It Is Built (Reuse of Existing Machinery)

The feature adds no C++ code. It is a model of how to build on existing KVCM
V2 mechanisms:

- **Which layers.** Full-attention self-attention layers with KV heads.
  Sliding-window layers, layers that share KV with another layer, and the
  layers of drafters with their own architecture stay native.
- **Low pool.** The layer's own pool:
  - dense managers size each token-head record as `head_dim / 16` groups of
    8 packed data bytes plus one FP8 scale byte each (data first, then
    scales), rounded up to a multiple of 8 bytes. Use the padded size in
    capacity estimates;
  - DSA and DeepSeek-V4 keep their native NVFP4 pool
    (`_own_pool_is_mixed_precision_low`).
- **High copy.** `_add_mixed_precision_layers` appends one extra attention
  layer per mixed layer, in the high dtype, with:
  - `sliding_window_size = last_n + 1`, which includes the next write
    position;
  - `num_sink_tokens = first_n`;
  - `reuse_policy = ReusePolicy.REQUIRED`.

  All high layers share one life cycle, so they form **one extra layer
  group, placed last**. A `pool_ratio`, if set, needs one more entry than the
  model has groups. The C++ core sees an ordinary sliding-window-plus-sink
  life cycle. Coverage is in whole pages, so slightly more than N tokens can
  be high precision.
- **Write rule.** New tokens are always encoded into the low pool, which
  therefore holds the full history. They are also copied into the high pool
  when their page is held there.

| Read path | Backends | Mechanism |
|---|---|---|
| A: staging copy | TRTLLM, FlashInfer (dense GQA/MHA; dense MLA path exists) | Metadata `prepare()` builds a staging copy. Per mixed layer: gather high rows where held and decode the rest from low, swap the attention view to staging, run the unchanged kernel, restore, then write the new tokens back. TRTLLM stages HND with pages of at most 32 tokens |
| B: high-row overlay | DSA, DeepSeek-V4 (sparse MLA) | New rows are stored into the high pool after the append. After the top-k gather decodes NVFP4 into scratch, the high pool's rows overwrite their scratch rows |

## Supported Combinations and Rejections

Configuration time (`TorchLlmArgs.validate_mixed_precision_kv` and
`KvCacheConfig`):
- requires `use_kv_cache_manager_v2: true` (`auto` is rejected);
- attention backend TRTLLM or FlashInfer;
- sparse algorithm, if any, `dsa` or `deepseek_v4`;
- no KV connector, no `sleep_config`, no KV cache compression;
- `max_beam_width == 1`;
- no dynamic-tree speculative decoding;
- at least one of first/last N above zero.

Model load and manager construction:
- the high dtype must match the model dtype when unquantized;
- Blackwell GPUs only;
- context parallelism 1;
- dense self-attention only;
- `head_dim` divisible by 16;
- the manager class must opt in (`_supports_mixed_precision_kv`);
- the DFlash unified-KV layout is excluded.

Speculative decoding:

| Drafter | Draft KV |
|---|---|
| One-model MTP | Mixed, like the target layers. A separate MTP draft manager gets its own high layers and staging |
| One-model EAGLE3 and other drafters with their own architecture | Plain cache. On DSA and DeepSeek-V4 the drafter inherits the NVFP4 pool |
| Dynamic-tree drafting | Rejected |

Disaggregated serving:
- the high layer of model layer `L` uses transfer id `2L + 1`, and every
  other layer uses twice its id, so ids are collision-free and equal on all
  pipeline ranks;
- the receiver resolves a high layer's KV-head count from its owning layer;
- use the Python transceiver path. The C++ cache transceiver is deprecated
  and is not a target for new work;
- context and generation workers must use the same mixed dtype and first/last
  N.

## Prefix Reuse: The Part That Surprises People

The low and high pools belong to the same per-request cache. They match,
commit, and evict through one radix tree. Because the high group is
`REQUIRED`:

- The reusable depth is the deepest endpoint at which **both** the high sink
  pages and the high live-window pages still exist.
- With `block_reuse_config.policy: per_request` or `per_conversation`, only
  the sinks and the final window are committed. A later request that shares
  a shorter prefix reuses only up to the sink-page boundary. With
  `all_reusable`, high pages stay committed for every block, subject to
  eviction from the high pool.
- Evicting a high **sink** page drops the whole prefix subtree below it.
  Evicting a high **window** page shortens later matches.
- Under memory pressure the high pool fills first. Reuse can collapse to the
  sink pages while the low pool still holds the full history. Always report
  the reused depth per request, not only a hit count.
- The high pool is sized by the last `pool_ratio` entry, or from
  `avg_seq_len`. Without it, the manager warns and assumes `max_seq_len // 2`.
- Choose the commit policy and the high-pool size deliberately, from reused
  depth measured on the target trace:
  - `all_reusable` keeps high pages for every committed block. Prefixes reuse
    deeper, but the high pool fills, and its evictions can take sink pages.
  - `per_request` and `per_conversation` keep only the sinks and the final
    window, which bounds high-pool use. With
    `block_reuse_config.enable_branch_snapshot` on, window life cycles are
    also committed at branch points. The high group counts as one, so a
    shared prefix can then reuse up to the fork. Confirm this in a test
    before relying on it.

  Report the configuration, and treat any change from a benchmark's standard
  configuration as its own arm.

Before describing reuse behavior, read where the high layers are added and
which reuse policy they get on the current main, and check for options that
change it. With the `REQUIRED` policy, evicted high pages are not rebuilt,
and a match ends where the high sinks or window are missing.

Changing the high group's reuse policy, or rebuilding missing high pages, is
a design change. It needs:
- a resume-time selection of which groups to reuse;
- a way to fill the missing high pages before attention reads them;
- the reuse-policy rules of the KVCM V2 C++ guide.

See `references/design-and-extension.md`. Changing an experiment's reuse
configuration, for example to `per_request`, to make reuse numbers look
better is not a fix. Measure it as a separate arm.

## Performance

Per-step costs to measure and report against FP8 KV, pure NVFP4 KV, and the
model dtype:
- **Per mixed layer per step** (path A): a gather and a write-back launch.
  They cover the longest sequence in eager steps and the **full span inside
  CUDA graphs**.
- **Host work per step**:
  - page tables gathered with NumPy for each page-table key;
  - TRTLLM staging block offsets;
  - a FlashInfer staging page list built in Python over every page.

  Keep these off the critical path when you change them.
- **Fast paths off.** The TRTLLM steady-decode fast path is off for every step
  of a manager with mixed-precision layers, including its non-mixed layers.
  The module-side FP8 append plan is skipped for each mixed layer.
- **Memory.** Staging is reserved once per rank:
  `rows x ceil(max_seq_len / page) x largest page`. Its size is logged at
  startup. Report it, and say how it affects the KV budget.
- **Expected benefit.** More history tokens per byte than FP8, at accuracy
  close to the high dtype. It shows up as capacity, concurrency, and reuse.
  Show the capacity gain and the per-step cost at the same operating points
  (`perf-kv-cache-tuning`).

## Accuracy

- Compare against FP8 KV and pure NVFP4 KV with the aligned protocol of
  `trtllm-kv-cache-accuracy-eval`. Use long-context retrieval with several
  needles and long reasoning, where distant tokens matter.
- Include forced-reuse runs. A reused prefix reads high pages for the sinks
  and the window, and NVFP4 for the middle. Report the reused depth.
- State `first_n`, `last_n`, and the page size. Effective coverage is whole
  pages.

## Tests

- `tests/unittest/_torch/pyexecutor/kv_cache/test_mixed_precision_kv_cache.py`:
  layout, quota, pool ratio, reuse and window retention, disaggregated
  export, drafters, DSA page table. Requires Blackwell.
- `tests/unittest/_torch/attention/test_mixed_precision_backends.py`:
  conversion, staging, CUDA-graph lengths, drafters, public options,
  validation.
- The mixed cases in
  `tests/unittest/_torch/pyexecutor/kv_cache/test_mamba_cache_manager.py`,
  `test_kvcm2_integration.py`, and `test_dflash_unified_kv_gate.py`, and
  `tests/unittest/disaggregated/test_extractor.py`.
- A `cpu_only` test inside a module that is skipped without a Blackwell GPU
  never runs in CI: GPU stages deselect `cpu_only`, and CPU stages skip the
  module. Put CPU-only checks in a module that is not GPU-gated.

## Pitfalls Seen Before

- **Raw pool reads.** `get_buffers(layer, ...)` on a mixed layer returns the
  *low* pool as packed `uint8` by default. Code that reads raw KV must pass
  `mixed_precision_pool="low"` or `"high"` and decode explicitly.
- **Feature-off invariance.** Every change for this feature must leave
  non-mixed runs identical: outputs, memory, speed, errors, API. Gate on the
  mixed configuration. Propose a more correct shared bookkeeping rule
  separately.
- **Stale lengths.** Device KV lengths can change after `prepare()` (overlap
  corrections, speculative decoding). Staging lengths must follow the device
  lengths, and draft staging must be re-synced before draft forwards.
- **"Once" claims.** Staging is built and zeroed once per metadata object.
  Before claiming any "once" behavior, read the call-site guard: a guard that
  rebuilds the object per step makes it per step.
- **Disaggregated heads.** A transfer layer whose KV-head count is resolved
  from the wrong layer breaks only in disaggregated runs. Test both pools and
  the id pairing on every claimed model family.
- **Custom managers.** A manager that does not opt in fails at construction.
  Add the opt-in and its layer and row hooks, rather than special-casing the
  manager elsewhere.

## Design Gate

Extend the general gate from `trtllm-kv-cache-develop` with:

```markdown
## Mixed-Precision Design Gate
- Change area: <layers/pools | staging path A | overlay path B | spec | disagg | reuse | validation>
- Existing mechanism reused: <SWA/sink life cycle, layer group, page tables, staging, transfer ids>
- Non-mixed invariance: <how gated; tests run with mixed off>
- Reuse impact: <effect on reusable depth; reuse policy of the high group unchanged? why>
- CUDA graph and overlap: <static buffers, lengths re-synced, fast paths>
- Per-step cost and memory: <...>
- Tests: <files and stages; Blackwell availability>
```
