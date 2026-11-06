---
name: trtllm-kv-cache-mixed-precision
description: >-
  Develop, review, or debug the TensorRT-LLM mixed-precision KV cache, which
  stores the full KV cache history in NVFP4 and keeps the first N and last N
  tokens in the high-precision dtype (kv_cache_config.dtype mixed_nvfp4_fp8,
  mixed_nvfp4_float16 or mixed_nvfp4_bfloat16, mixed_precision_first_n,
  mixed_precision_last_n) on KV cache manager V2. Use when enabling it for a
  new attention backend, KV cache manager, model family, or speculative
  decoding mode; when touching its mixed-precision high layers,
  mixed-precision staging copy, high-row overlay path, disaggregated transfer
  ids, or prefix-reuse behavior; and when measuring its accuracy or per-step
  cost.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Mixed-Precision KV Cache

The mixed-precision KV cache (a prototype) stores the KV cache history of
every full-attention layer in NVFP4, in the layer's own *mixed-precision low
pool*. It also keeps the first `mixed_precision_first_n` and the last
`mixed_precision_last_n` tokens of each sequence in a *mixed-precision high
pool*, in the high-precision dtype. Attention kernels are unchanged: before a
mixed-precision layer runs, its KV cache is presented to attention in the
high-precision dtype.

The user-facing contract is in `docs/source/features/kvcache.md`, section
"Mixed-Precision KV Cache (Prototype)". This skill covers the design, the
invariants, the extension steps, and the pitfalls. The mixed-precision KV
cache is built on KV cache manager V2, not on the KV cache compression
framework: read `trtllm-kv-cache-manager-v2`
(`../trtllm-kv-cache-manager-v2/SKILL.md`) first. The
data flow and extension recipes are in `references/design-and-extension.md`.

## Required Context

1. `docs/source/features/kvcache.md`: "Mixed-Precision KV Cache (Prototype)",
   the sliding-window attention (SWA) section "SWA Endpoint Retention", and
   "Partial Reuse".
2. `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md` (the KV
   cache manager V2 C++ guide): life cycles, sliding windows and sinks, reuse
   policies, eviction.
3. `tensorrt_llm/_torch/attention/ATTENTION_DEVELOPER_GUIDE.md`: metadata
   families, and how device KV cache lengths are refreshed after `prepare()`.
4. Code:
   - `tensorrt_llm/llmapi/llm_args.py`: the `KvCacheConfig` dtype and
     first/last-N fields, and `TorchLlmArgs.validate_mixed_precision_kv`.
   - `tensorrt_llm/_torch/pyexecutor/model_loader.py`: KV cache quantization mapping
     for the mixed-precision dtypes.
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
     for DeepSeek sparse attention (DSA) and DeepSeek-V4.
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

The second dtype in the name is the high-precision dtype, the dtype attention
computes in. It must match the model dtype unless it is `fp8`. Confirm the
result in the startup log:
- the mixed-precision layers and the extra mixed-precision high layer group;
- the size of the mixed-precision staging copy, for the TRTLLM and FlashInfer
  attention backends.

Then report reused depth and per-step cost (`perf-kv-cache-tuning`,
`../perf-kv-cache-tuning/SKILL.md`) next to accuracy
(`trtllm-kv-cache-accuracy-eval`, `../trtllm-kv-cache-accuracy-eval/SKILL.md`).

## How It Is Built (Reuse of Existing Machinery)

The feature adds no C++ code. It is a model of how to build on existing KV
cache manager V2 mechanisms:

- **Which layers.** Full-attention self-attention layers with KV heads.
  Sliding-window layers, layers that share a KV cache with another layer, and
  the layers of drafters with their own architecture stay native.
- **Mixed-precision low pool.** The layer's own pool:
  - dense managers size each token-head record as `head_dim / 16` groups of
    8 packed data bytes plus one FP8 scale byte each (data first, then
    scales), rounded up to a multiple of 8 bytes. Use the padded size in
    capacity estimates;
  - DSA and DeepSeek-V4 keep their native NVFP4 pool
    (`_own_pool_is_mixed_precision_low`).
- **Mixed-precision high layers.** `_add_mixed_precision_layers` appends one
  extra attention layer per mixed-precision layer, in the high-precision
  dtype, with:
  - `sliding_window_size = last_n + 1`, which includes the next write
    position;
  - `num_sink_tokens = first_n`;
  - `reuse_policy = ReusePolicy.REQUIRED`.

  All mixed-precision high layers share one life cycle, so they form **one
  extra layer group, placed last**. A `pool_ratio`, if set, needs one more
  entry than the model has layer groups. The KV cache manager V2 native core
  sees an ordinary sliding-window-plus-sink life cycle. Coverage is in whole
  pages, so slightly more than N tokens can be high precision.
- **Write rule.** New tokens are always encoded into the mixed-precision low
  pool, which therefore holds the full history. They are also copied into the
  mixed-precision high pool when their page is held there.

| Read path | Backends | Mechanism |
|---|---|---|
| Staging-copy path | TRTLLM, FlashInfer: dense grouped-query attention (GQA) and multi-head attention (MHA); a dense multi-head latent attention (MLA) path exists | Metadata `prepare()` builds a mixed-precision staging copy. Per mixed-precision layer: gather mixed-precision high rows where held and decode the rest from the mixed-precision low pool, swap the attention view to the mixed-precision staging copy, run the unchanged kernel, restore, then write the new tokens back. The TRTLLM attention backend builds the mixed-precision staging copy in the `HND` layout (heads before tokens), with pages of at most 32 tokens |
| High-row overlay path | DSA, DeepSeek-V4 (sparse MLA) | New rows are stored into the mixed-precision high pool after the append. After the top-k gather decodes NVFP4 into scratch, the mixed-precision high pool's rows overwrite their scratch rows |

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
- the high-precision dtype must match the model dtype when unquantized;
- Blackwell GPUs only;
- context parallelism 1;
- dense self-attention only;
- `head_dim` divisible by 16;
- the manager class must opt in (`_supports_mixed_precision_kv`);
- the DFlash unified-KV layout is excluded.

Speculative decoding:

| Drafter | Draft KV cache |
|---|---|
| One-model multi-token prediction (MTP) | Mixed-precision KV cache, like the target layers. A separate MTP draft KV cache manager gets its own mixed-precision high layers and mixed-precision staging copy |
| One-model EAGLE3 and other drafters with their own architecture | Plain KV cache. On DSA and DeepSeek-V4 the drafter inherits the NVFP4 pool |
| Dynamic-tree drafting | Rejected |

Disaggregated serving:
- the mixed-precision high layer of model layer `L` uses transfer id `2L + 1`,
  and every other layer uses twice its id, so ids are collision-free and equal
  on all pipeline ranks;
- the receiver resolves a mixed-precision high layer's KV-head count from
  its owning layer;
- use the Python transceiver path. The C++ cache transceiver is deprecated
  and is not a target for new work;
- context and generation workers must use the same mixed-precision dtype and
  first/last N.

## Prefix Reuse: The Part That Surprises People

The mixed-precision low pool and the mixed-precision high pool belong to the
same per-request cache. They match, commit, and evict through one radix tree.
Because the mixed-precision high layer group is `REQUIRED`:

- The reusable depth is the deepest endpoint at which **both** the
  mixed-precision high sink pages and the pages of the mixed-precision high
  window at that endpoint still exist.
- With `block_reuse_config.policy: per_request` or `per_conversation`, only
  the mixed-precision high sink pages and the final mixed-precision high
  window are committed. A later request that shares a shorter prefix reuses
  only up to the end of the mixed-precision high sink pages. With
  `all_reusable`, mixed-precision high pages stay committed for every block,
  subject to eviction from the mixed-precision high pool.
- Evicting a mixed-precision high **sink** page drops the whole prefix
  subtree below it. Evicting a mixed-precision high **window** page shortens
  later matches.
- Under memory pressure the mixed-precision high pool fills first. Reuse can
  collapse to the mixed-precision high sink pages while the mixed-precision
  low pool still holds the full history. Always report the reused depth per
  request, not only a hit count.
- The mixed-precision high pool is sized by the last `pool_ratio` entry, or from
  `avg_seq_len`. Without it, the manager warns and assumes `max_seq_len // 2`.
- Choose the commit policy and the size of the mixed-precision high pool
  deliberately, from reused depth measured on the target trace:
  - `all_reusable` keeps mixed-precision high pages for every committed
    block. Prefixes reuse deeper, but the mixed-precision high pool fills,
    and its evictions can take mixed-precision high sink pages.
  - `per_request` and `per_conversation` keep only the mixed-precision high
    sink pages and the final mixed-precision high window, which bounds the
    use of the mixed-precision high pool. With
    `block_reuse_config.enable_branch_snapshot` on, sliding-window life
    cycles are also committed at branch points. The mixed-precision high
    layer group counts as one, so a shared prefix can then reuse up to the
    fork. Confirm this in a test before relying on it.

  Report the configuration, and treat any change from a benchmark's standard
  configuration as its own arm (one configuration being compared).

Before describing reuse behavior, read where the mixed-precision high layers
are added and which reuse policy they get on the current main, and check for
options that change it. With the `REQUIRED` policy, evicted mixed-precision
high pages are not rebuilt, and a match ends where the mixed-precision high
sink pages or the mixed-precision high window are missing.

Changing the mixed-precision high layer group's reuse policy, or rebuilding
missing mixed-precision high pages, is a design change. It needs:
- a resume-time selection of which layer groups to reuse;
- a way to fill the missing mixed-precision high pages before attention
  reads them;
- the reuse-policy rules of the KV cache manager V2 C++ guide.

See `references/design-and-extension.md`. Changing an experiment's reuse
configuration, for example to `per_request`, to make reuse numbers look
better is not a fix. Measure it as a separate arm.

## Performance

Per-step costs to measure and report against the FP8 KV cache, the pure NVFP4
KV cache, and a KV cache in the model dtype:
- **Per mixed-precision layer per step** (staging-copy path): a gather and a
  write-back launch. They cover the longest sequence in eager steps and the
  **full span inside CUDA graphs**.
- **Host work per step**:
  - page tables gathered with NumPy for each page-table key;
  - for the TRTLLM attention backend, the block offsets of the
    mixed-precision staging copy;
  - for the FlashInfer attention backend, a page list of the
    mixed-precision staging copy, built in Python over every page.

  Keep these off the critical path when you change them.
- **Fast paths off.** The steady-decode fast path of the TRTLLM attention
  backend is off for every step of a manager with mixed-precision layers,
  including its other layers. The module-side FP8 KV cache append plan is
  skipped for each mixed-precision layer.
- **Memory.** The mixed-precision staging copy is reserved once per rank:
  `rows x ceil(max_seq_len / page) x largest page`. Its size is logged at
  startup. Report it, and say how it affects the KV cache memory budget.
- **Expected benefit.** More history tokens per byte than the FP8 KV cache,
  at accuracy close to a KV cache in the high-precision dtype. It shows up as
  capacity, concurrency, and reuse. Show the capacity gain and the per-step
  cost at the same operating points (`perf-kv-cache-tuning`,
  `../perf-kv-cache-tuning/SKILL.md`).

## Accuracy

- Compare against the FP8 KV cache and the pure NVFP4 KV cache with the
  aligned protocol of `trtllm-kv-cache-accuracy-eval`
  (`../trtllm-kv-cache-accuracy-eval/SKILL.md`). Use long-context retrieval
  with several needles and long reasoning, where distant tokens matter.
- Include forced-reuse runs. A reused prefix reads the mixed-precision high
  sink pages and the mixed-precision high window, and reads the middle in
  NVFP4 from the mixed-precision low pool. Report the reused depth.
- State `first_n`, `last_n`, and the page size. Effective coverage is whole
  pages.

## Tests

- `tests/unittest/_torch/pyexecutor/kv_cache/test_mixed_precision_kv_cache.py`:
  layout, quota, pool ratio, reuse and retention of the mixed-precision high
  window, disaggregated export, drafters, DSA page table. Requires Blackwell.
- `tests/unittest/_torch/attention/test_mixed_precision_backends.py`:
  conversion, the mixed-precision staging copy, CUDA-graph lengths, drafters,
  public options, validation. Its conversion test compares bytes only
  (layout, addressing, exact decoding). After changing the conversion
  kernels, check the NVFP4 encode/decode round trip for every high-precision
  dtype locally on real KV cache data, and write the commands and results in
  the PR description.
- The mixed-precision cases in
  `tests/unittest/_torch/pyexecutor/kv_cache/test_mamba_cache_manager.py`,
  `test_kvcm2_integration.py`, and `test_dflash_unified_kv_gate.py`, and
  `tests/unittest/disaggregated/test_extractor.py`.
- A `cpu_only` test inside a module that is skipped without a Blackwell GPU
  never runs in CI: GPU stages deselect `cpu_only`, and CPU stages skip the
  module. Put CPU-only checks in a module that is not GPU-gated.

## Pitfalls Seen Before

- **Raw pool reads.** `get_buffers(layer, ...)` on a mixed-precision layer
  returns the mixed-precision low pool as packed `uint8` by default. Code
  that reads raw KV cache data must pass `mixed_precision_pool="low"` or
  `"high"` and decode explicitly.
- **Feature-off invariance.** Every change for this feature must leave runs
  without the mixed-precision KV cache identical: outputs, memory, speed,
  errors, API. Gate on the mixed-precision configuration. Propose a more
  correct shared bookkeeping rule separately.
- **Stale lengths.** Device KV cache lengths can change after `prepare()`
  (overlap scheduler corrections, speculative decoding). The lengths of the
  mixed-precision staging copy must follow the device lengths, and the draft
  mixed-precision staging copy must be re-synced before draft forwards.
- **"Once" claims.** The mixed-precision staging copy is built and zeroed
  once per metadata object. Before claiming any "once" behavior, read the
  call-site guard: a guard that rebuilds the object per step makes it per
  step.
- **Disaggregated heads.** A transfer layer whose KV-head count is resolved
  from the wrong layer breaks only in disaggregated runs. Test both the
  mixed-precision low pool and the mixed-precision high pool, and the id
  pairing, on every claimed model family.
- **Custom managers.** A manager that does not opt in fails at construction.
  Add the opt-in and its layer and row hooks, rather than special-casing the
  manager elsewhere.

## Design Gate

Extend the general design gate in
`../trtllm-kv-cache-manager-v2/references/engineering-rules.md` with:

```markdown
## Mixed-Precision KV Cache Design Gate
- Change area: <layers and pools | staging-copy path | high-row overlay path | speculative decoding | disaggregated serving | reuse | validation>
- Existing mechanism reused: <sliding-window-with-sinks life cycle, layer group, page tables, mixed-precision staging copy, disaggregated transfer ids>
- Invariance with the mixed-precision KV cache off: <how gated; tests run with it off>
- Reuse impact: <effect on reusable depth; reuse policy of the mixed-precision high layer group unchanged? why>
- CUDA graphs and the overlap scheduler: <static buffers, lengths re-synced, fast paths>
- Per-step cost and memory: <...>
- Tests: <pytest files and stages; Blackwell availability>
- Local runs (round trip, accuracy, end to end): <commands and results, for the PR description>
```
