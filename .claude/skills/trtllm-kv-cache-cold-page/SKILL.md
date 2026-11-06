---
name: trtllm-kv-cache-cold-page
description: >-
  Develop, review, or debug storage-bound KV cache compression for the
  TensorRT-LLM KV cache manager V2 cold tiers: the native cold-page codec
  contract (IKvCacheColdPageCodec, the NativeColdPageCodec adapter and its
  Python bridge), the NVFP4 cold-page method
  (ColdPageQuantizationCompressionConfig, quantization_for_cold_page),
  rotary position embedding (RoPE) precision options (skip_rope_quantization,
  nvfp4_residual_dim), per-model KV cache layout coverage including
  DeepSeek-V4, Host and Disk offload/onboard, and how compressed cold pages
  interact with reuse, speculative decoding, and serving. Use when adding a
  cold-page format or cold-page codec, enabling a model or layout, changing
  encode/decode kernels, or investigating the accuracy or performance of
  compressed cold pages.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Cold-Page KV Cache Compression

Cold-page compression changes only how KV cache manager V2 stores pages that
leave the GPU:
- a page is encoded when it moves from the GPU to a Host or Disk tier;
- it is decoded when it moves back;
- Host-to-Disk moves copy the encoded blob unchanged.

The active KV cache on the GPU, attention kernels, and model code stay
unchanged. KV cache manager V2 still owns every allocation, migration, event,
and staging buffer.

The contracts live in:
- `docs/source/developer-guide/kv-cache-compression-development.md`, sections
  "Storage-bound methods", "Which numbers of a K or V vector become NVFP4",
  and "Validation" (check its encode/decode round-trip accuracy item
  locally on real KV cache data, not in pytest or gtests);
- `docs/source/developer-guide/kv-cache-cold-page-codec.md`: the native
  cold-page codec application binary interface (ABI), the memory contract,
  the migration transaction, and the initialization order.

This skill is the workflow and the pitfalls on top of them. Cold-page
compression is a KV cache compression method: read
`trtllm-kv-cache-compression` (`../trtllm-kv-cache-compression/SKILL.md`)
first. The KV cache manager V2 side of the cold-page codec interface is
covered by `trtllm-kv-cache-manager-v2`
(`../trtllm-kv-cache-manager-v2/SKILL.md`).

## Required Context

1. The two developer guides above, the "Cold-Page Quantization" section of
   `docs/source/features/kv-cache-compression.md`, and
   `examples/kv_cache_compression/nvfp4_cold_page.md`.
2. Python code:
   - `ColdPageQuantizationCompressionConfig` in `tensorrt_llm/llmapi/llm_args.py`;
   - `tensorrt_llm/_torch/kv_cache_compression/quantization_for_cold_page/`.
     It contains the `ColdPageQuantizationCompression` compression provider
     base and `Nvfp4ColdPageQuantizationCompression`.
3. Native code:
   - `cpp/tensorrt_llm/batch_manager/kv_cache_compression/` (`NativeColdPageCodec`);
   - `cpp/tensorrt_llm/nanobind/kvCacheCompression/bindings.cpp`, the Python
     bridge and kernel entry points;
   - `cpp/tensorrt_llm/kernels/nvfp4ColdPageKernels.*`;
   - `coldPageCodec.*` and `storageManager.*` in
     `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`.
4. Tests:
   - Python: `tests/unittest/_torch/kv_cache_compression/`, including
     `test_quantization_for_cold_page.py` and `test_nvfp4_cold_page_rope_gpu.py`.
   - Native: the NVFP4 kernel gtest
     (`cpp/tests/unit_tests/kernels/nvfp4ColdPageKernelsTest.cpp`) runs through
     `test_kv_cache_compression_unit_tests` in
     `tests/integration/defs/cpp/test_unit_tests.py`. The cold-page codec
     adapter and KV cache manager V2 cold-page gtests in
     `cpp/tests/unit_tests/batch_manager/` run with the batch-manager unit
     tests.

## How to Use It

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  host_cache_size: 8589934592        # bytes; without a cold tier nothing is compressed
kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: nvfp4
  # skip_rope_quantization: true     # prototype; supported model types only
  # scale_checkpoint_path: /path     # optional ModelOpt NVFP4 K/V scales
```

The same options exist in Python as `ColdPageQuantizationCompressionConfig`.
`examples/kv_cache_compression/nvfp4_cold_page.md` has runnable recipes,
including aggregated and disaggregated serving and the activation check. Pages
are compressed only when they leave the GPU, so the workload must put
pressure on GPU capacity, for example multi-turn sessions longer than what
the GPU holds.

## Composing the Pipeline

```text
startup
  factory builds the compression provider (one per LLM) -> KV cache manager V2 construction asks it for a cold-page codec
  -> provider.build_codec_state()            (layers it owns, runtime dtype, layouts)
  -> KV cache manager V2 allocates hot pools, then configure(hot pool descriptors)
  -> provider.configure() -> build_lifecycle_metadata() per life cycle -> fixed cold-page bytes
  -> KV cache manager V2 sizes and allocates the Host / Disk cold tiers
offload (hot -> cold), driven by the eviction controller
  allocate cold slots -> wait on readiness events -> encode_cold_pages() on the migration stream
  -> one finish event -> ownership moves to the cold slot
onboard (cold -> hot), before a reused page is locked
  allocate hot slots -> decode_cold_pages() -> finish event -> attention reads the hot page as usual
host <-> disk: the encoded blob is copied unchanged; no cold-page codec call
```

The compression provider's encode and decode receive host page-index pairs
and the migration stream, and launch the cold-page format kernels. The NVFP4
launcher splits a batch into fixed-size chunks of pages. Kernels for a new
cold-page format follow
`../trtllm-kv-cache-compression/references/adding-kernels.md`.

## Layers and Ownership

```text
KV cache manager V2 StorageManager (C++)
                              -- owns pools, cold tiers, staging buffers, events, migration transaction
   | IKvCacheColdPageCodec: configure / queryColdPageBytes / batching id / index location / encode / decode
NativeColdPageCodec (C++)     -- resolves hot buffers per life cycle, routes each life cycle to the compression provider
   |                             or to its built-in lossless fallback, drains the stream after a compression provider error
Python bridge (nanobind)      -- forwards configure/encode/decode to the compression provider with pointers as integers
ColdPageQuantizationCompression subclass (Python)
                              -- format state, per-life-cycle layout and fixed cold-page size, kernel launches
```

| Component | Owns | Never does |
|---|---|---|
| Config and factory | Method selection, cold-page format options, admission | Touch pages |
| Python compression provider | Cold-page format state per layer, lossy and lossless spans, layout validation, launches | Allocate, publish, or release KV cache manager V2 pages; synchronize the stream |
| `NativeColdPageCodec` | Layout resolution, routing to the compression provider or the lossless fallback, lifetime bridge | Hold quantization policy |
| KV cache manager V2 storage | Hot and cold allocation, staging, migration, events, rollback | Know the cold-page format |

## Cold-Page Codec Contract Checklist

Each item is stated in the cold-page codec design doc or the KV cache
compression development guide, or enforced by the `NativeColdPageCodec`
adapter:

- **Fixed size.** The cold-page size is fixed per life cycle and positive.
  Variable-length pages are not supported.
- **One owner per life cycle.** A life cycle is either wholly owned by the
  compression provider or wholly on the lossless fallback; mixing is
  rejected. Non-attention life cycles, such as recurrent state, stay on the
  lossless fallback.
- **Index location.** The page-index location is chosen per life cycle, and
  both directions use the same one.
  - The NVFP4 compression provider publishes host indices and copies them
    into kernel parameters in chunks.
  - Host indices are valid only during the call.
  - Device indices are valid until the stream work completes.
- **Asynchronous.** Work is enqueued only on the supplied stream, without
  synchronizing. The return value means "submitted", not "completed".
- **No retained pointers.** Base and index pointers are never kept. Disk
  addresses never reach the cold-page codec; the Disk path goes through
  KV cache manager V2's pinned staging buffers.
- **Exact lossless spans.** Spans the cold-page format declares lossless are
  copied byte for byte.
- **Batches.** Non-contiguous pages, partial pages, and large multi-page
  batches are handled. Handle whatever batch a call receives:
  KV cache manager V2 may split large batches across calls, and a call need
  not contain every member of a batching class.
- **One-shot ownership.** One cold-page codec instance for every KV cache
  manager V2 the executor builds (target and draft), consumed at
  construction. Never share cold-page codec state between KV cache managers.

## Adding a Cold-Page Format or Cold-Page Codec

1. **Write the representation down.** Per life cycle and buffer, record:
   - which contiguous span is quantized;
   - what is copied unchanged;
   - the scale-group size and alignment;
   - where scales live;
   - the fixed encoded size.

   The NVFP4 cold-page format quantizes one contiguous range per row in
   groups of 16 with an FP8 scale per group. Prefixes and suffixes are
   copied, and the layer layout is 16-byte aligned.
2. **Configuration.** Add a new `quant` value with its factory and
   architecture-gate branches, or add a new config class following
   `trtllm-kv-cache-compression` (`../trtllm-kv-cache-compression/SKILL.md`).
   Report `supports_block_reuse()` and `supports_speculative_decoding()`
   truthfully.
3. **Compression provider.** Subclass `ColdPageQuantizationCompression`. It
   already sets `provides_cold_page_codec = True` and
   `uses_iteration_lifecycle = False`, and implements
   `create_cold_page_codec()` and `configure()`, which publishes host page
   indices. Implement:
   - `build_codec_state(...)`, returning state with unique
     KV cache manager V2 `layer_ids`;
   - `build_lifecycle_metadata(...)` for each life cycle, returning a fixed
     `cold_page_bytes > 0`;
   - `encode_cold_pages(...)` and `decode_cold_pages(...)`.

   Map local layers to global ones through the pipeline-parallel layer list.
   A compression provider that subclasses `KVCacheCompressionManager`
   directly must also implement `create_cold_page_codec()` and `configure()`,
   and expose `codec_state.layer_ids`. Override `configure()` only to publish
   device page indices.
4. **Kernels.**
   - A CUDA kernel library goes in `cpp/tensorrt_llm/kernels/`, registered
     with `add_tllm_kernel_library(...)` in `kernelComponents.cmake`. Expose a
     thin entry point in `cpp/tensorrt_llm/nanobind/kvCacheCompression/bindings.cpp`,
     as `nvfp4_cold_page_encode` and `nvfp4_cold_page_decode` do. Call it from
     the compression provider through a local
     `from tensorrt_llm.bindings.internal import kv_cache_compression as native`.
   - Triton or CuTe DSL launchers may live in the Python package.
   - KV cache manager V2 storage and migration code need no change.
   - Give an actionable error on unsupported GPUs.
5. **Lossy flag.** The KV cache lender (`kv_cache/sharing`) refuses to
   publish original KV cache bytes from a KV cache manager whose
   cold-page codec is lossy. Today `KVCacheManagerV2` detects this by
   comparing the algorithm name with `quantization_for_cold_page`. Do not add
   a second name to that check. Propose an algorithm-neutral declaration
   instead, for example a capability on the compression provider or the
   config, and agree on it with the KV cache manager V2 owners.
6. **Admission.** Reject layouts the cold-page format cannot express, with
   actionable messages, at construction or in the cold-page codec state
   builder. Never compress a layer the user asked for by silently falling
   back. The lossless fallback is for life cycles the cold-page format
   defines as lossless.
7. **Tests.**
   - Add a native gtest for the kernels, and extend the KV cache compression
     unit-test fixture so CI builds and runs it.
   - Add Python tests for cold-page codec state, layouts, admission, and
     activation.
   - In both, test lossy kernels through the layout contract in "Numerics
     Pitfalls Seen Before", never with a tolerance against a hand-written
     numeric reference. Measure the encode/decode round-trip accuracy for
     every supported runtime dtype, and any drift over repeated round trips,
     locally on real KV cache data, and write the commands and results in
     the PR description.

## Enabling a Model for RoPE-Preserving Cold Pages

`skip_rope_quantization` keeps the position-encoded part of each K vector in
its active precision. It is honored only for the model types in
`_SKIP_ROPE_QUANTIZATION_MODEL_TYPES`, in
`quantization_for_cold_page/nvfp4_quantization.py`. Other models log a warning
and quantize whole vectors. To add a model:

1. **Verify the physical KV cache layout.** Read the attention module and
   the KV cache manager V2 buffers, not just the model config.
   - A key-only buffer is treated as a multi-head latent attention (MLA)
     latent with trailing RoPE.
   - Separate key and value buffers are treated as partial-rotary
     grouped-query attention (GQA) with leading RoPE.
   - DeepSeek-V4 has its own per-role layout rules.
   - Interleaved rotary dimensions, RoPE in the middle of a vector, or an MLA
     KV cache with a value buffer need an explicit placement rule and a test.
2. **Derive the RoPE width.** Use rotary-width fields or partial-rotary
   factors, never `rope_theta`.
3. **Keep the refusals.** The NVFP4 cold-page format refuses:
   - an all-RoPE vector;
   - RoPE in the middle of a vector;
   - spans that are not multiples of 16 elements.

   V vectors are always quantized in full. Draft KV caches always quantize
   whole vectors with single NVFP4 and identity global scales. They log a
   warning when `skip_rope_quantization` is set, and ignore
   `nvfp4_residual_dim` without a message.
4. **Add the model type with a test.** The test must check placement on that
   model's real layout.

## When Pages Become Cold

- The hot eviction controller evicts pages in least recently used (LRU)
  order within a priority. Unreferenced pages at or above
  `cold_tier_offload_min_priority` (default 30) move to a cold tier and are
  encoded. Lower-priority pages are dropped.
- Suspended requests' pages and reusable radix-tree pages are the usual
  candidates. A restore before locking decodes them.
- Without a Host or Disk tier, the method has nothing to do, and that is not
  an error. A Host tier is configured with `kv_cache_config.host_cache_size`
  in bytes; the default is 0. See "Enable Offloading to Host Memory" in
  `kvcache.md`.

## Interactions

- **Block reuse.** Supported, because token identity is preserved. A prefix
  restored from a cold page is the decoded, lossy version. Accuracy
  evaluation must include requests that reuse onboarded pages.
- **Speculative decoding.** Target and separate draft KV cache managers
  encode independently. Admission currently accepts one-model
  MTP-EAGLE, the EAGLE variant of multi-token prediction (MTP), and one-model
  EAGLE3, and rejects other modes, including vanilla MTP. The list lives in
  `validate_kv_cache_compression_compatibility()`; read it there rather than
  copying it.
- **Mixed-precision KV cache.** Rejected together, by
  `TorchLlmArgs.validate_mixed_precision_kv`, which rejects any compression
  configuration with a `mixed_nvfp4_*` dtype.
- **Where checks run.** Different checks run in different places, so look in
  all of them before adding or relaxing one:
  - the architecture, block-reuse and speculative-mode checks are in
    `validate_kv_cache_compression_compatibility()`;
  - context-parallel Helix and an unsupported `quant` value are rejected in
    `create_kv_cache_compression_manager()`;
  - layout limits are enforced in the cold-page codec state builder.

  The validator and the factory run during executor construction, before
  KV cache manager V2 allocation, not when the configuration is parsed.
- **Active KV cache that is already NVFP4.** The method is skipped, with an
  info log.
- **CUDA graphs and the overlap scheduler.** Encode and decode run on the
  migration stream, outside CUDA graph capture, so keep CUDA graphs and the
  overlap scheduler on, as in every production measurement.
- **Not guarded today.** Disaggregated serving and beam search have no
  explicit check. The NVFP4 example documents a disaggregated-serving
  recipe. Treat beam search and any other combination nobody has validated
  as unsupported. Add an explicit rejection when you find one that fails.
- **Mixed-dtype KV cache layer specs.** Rejected.
- **The KV cache lender.** It refuses lossy cold-page codecs.

## Numerics Pitfalls Seen Before

- **Repeated round trips.** A page can be offloaded, onboarded, and
  offloaded again.
  - A cold-page format that re-derives scales from already-quantized values
    drifts on every round trip.
  - Design it so that re-encoding a decoded page reproduces the same codes,
    or measure and document the drift.
  - Test several round trips, not one. Where the cold-page format promises
    it, check in pytest or the kernel gtest that re-encoding a decoded page
    reproduces the same codes. Measure any drift locally on real KV cache
    data, and write the commands and results in the PR description.
- **Hand-written reference implementations can be wrong**, for example by
  grouping dimensions in a different order than the kernel. Test the layout
  contract instead: spans, scale groups, byte-exact lossless parts, and the
  same codes on re-encode where the cold-page format promises it.
- **Global scales.** Identity by default. With `scale_checkpoint_path`,
  per-layer K/V scales come from an NVFP4 ModelOpt checkpoint, and are loaded
  and validated at initialization.
  - They apply to target layers with separate key and value buffers.
  - Key-only (MLA) layers and draft KV caches keep identity scales.
  - The DeepSeek-V4 compressed KV cache uses only the K scale.

  Never compute scales while serving.
- **RoPE values are the sensitive part.** `skip_rope_quantization` and the
  DeepSeek-V4 two-component residual (`nvfp4_residual_dim: 64`, the default)
  are accuracy options. Measure them per model.
- **Apparent drops can be artifacts.** An apparent accuracy drop can come
  from truncation or a protocol mismatch. Check with
  `trtllm-kv-cache-accuracy-eval` (`../trtllm-kv-cache-accuracy-eval/SKILL.md`)
  before blaming the cold-page format.

## Activation and Performance

- **Activation evidence.** The offload and onboard byte counters
  (`trtllm_kv_cache_offload_bytes_total`, `trtllm_kv_cache_onboard_bytes_total`)
  also count lossless-fallback life cycles. Show that compressed life cycles
  moved, for example from cold-page sizes or compression provider call
  counts, not only that bytes moved.
- **Expected benefit.** More pages per cold-tier byte and fewer bytes moved.
  This shows up as a higher reuse hit rate and lower time to first token
  (TTFT) when the working set exceeds GPU capacity. Use multi-turn or
  shared-prefix traces that force offload.
- **Baselines.** The default lossless cold-page codec at the same tier
  capacity (equal bytes), and at the same page count (equal token capacity).
  State which.
- **Attribution.** Report onboarding latency, bytes moved, and encode and
  decode time against copy time. Say whether they overlap model execution or
  block a request; see `perf-kv-cache-tuning`
  (`../perf-kv-cache-tuning/SKILL.md`).

## Review Checklist

```markdown
## Cold-page review
- Representation written down (spans, scale groups, fixed size per life cycle): <yes/no>
- Cold-page codec contract items kept (size, ownership per life cycle, index location, asynchronous completion, pointers): <risks>
- Layout verified on the real cache layout for each enabled model: <models>
- Unsupported layouts and combinations rejected with actionable errors: <where>
- Lossy cold-page codec declared without adding a method name to KV cache manager V2: <how / not applicable>
- Same codes on re-encode tested, where the cold-page format promises it: <test / not promised>
- Encode/decode round-trip accuracy for every supported runtime dtype, and drift over repeated round trips, measured locally on real KV cache data: <commands and results in the PR description>
- Reuse of onboarded pages covered in accuracy evaluation: <yes/no>
- Host and Disk paths, failure rollback, non-default streams tested: <tests>
- Activation shown for compressed life cycles, not only byte counters: <how>
- Performance: comparison at equal bytes or equal token capacity, with migration time attributed: <summary>
```
