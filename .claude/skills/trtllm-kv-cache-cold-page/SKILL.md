---
name: trtllm-kv-cache-cold-page
description: >-
  Develop, review, or debug storage-bound KV cache compression for the
  TensorRT-LLM KV Cache Manager V2 (KVCM V2) cold tiers: the native
  cold-page codec contract (IKvCacheColdPageCodec, the NativeColdPageCodec
  adapter and its Python bridge), the NVFP4 cold-page method
  (ColdPageQuantizationCompressionConfig, quantization_for_cold_page),
  RoPE precision options (skip_rope_quantization, nvfp4_residual_dim),
  per-model cache-layout coverage including DeepSeek-V4, Host and Disk
  offload/onboard, and how compressed cold pages interact with reuse,
  speculative decoding, and serving. Use when adding a cold format or codec,
  enabling a model or layout, changing encode/decode kernels, or
  investigating the accuracy or performance of compressed cold pages.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Cold-Page KV Compression

Cold-page compression changes only how KVCM V2 stores pages that leave the
GPU:
- a page is encoded when it moves from the GPU to a Host or Disk tier;
- it is decoded when it moves back;
- Host-to-Disk moves copy the encoded blob unchanged.

The active GPU cache, attention kernels, and model code stay unchanged. KVCM
still owns every allocation, migration, event, and staging buffer.

The contracts live in:
- `docs/source/developer-guide/kv-cache-compression-development.md`, sections
  "Storage-bound methods", "Which numbers of a K or V vector become NVFP4",
  and "Validation";
- `docs/source/developer-guide/kv-cache-cold-page-codec.md`: the native
  codec ABI, the memory contract, the migration transaction, and the
  initialization order.

This skill is the workflow and the pitfalls on top of them. Load
`trtllm-kv-cache-develop` and `trtllm-kv-cache-compression` first.

## Required Context

1. The two developer guides above, the "Cold-Page Quantization" section of
   `docs/source/features/kv-cache-compression.md`, and
   `examples/kv_cache_compression/nvfp4_cold_page.md`.
2. Python code:
   - `ColdPageQuantizationCompressionConfig` in `tensorrt_llm/llmapi/llm_args.py`;
   - `tensorrt_llm/_torch/kv_cache_compression/quantization_for_cold_page/`.
     It contains the `ColdPageQuantizationCompression` provider base and
     `Nvfp4ColdPageQuantizationCompression`.
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
     `tests/integration/defs/cpp/test_unit_tests.py`. The codec-adapter and
     KVCM cold-page gtests in `cpp/tests/unit_tests/batch_manager/` run with the
     batch-manager unit tests.

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
  factory builds the provider (one per LLM) -> KVCM construction asks it for a codec
  -> provider.build_codec_state()            (layers it owns, runtime dtype, layouts)
  -> KVCM allocates hot pools, then configure(hot pool descriptors)
  -> provider.configure() -> build_lifecycle_metadata() per life cycle -> fixed cold-page bytes
  -> KVCM sizes and allocates the Host / Disk cold tiers
offload (hot -> cold), driven by the eviction controller
  allocate cold slots -> wait on readiness events -> encode_cold_pages() on the migration stream
  -> one finish event -> ownership moves to the cold slot
onboard (cold -> hot), before a reused page is locked
  allocate hot slots -> decode_cold_pages() -> finish event -> attention reads the hot page as usual
host <-> disk: the encoded blob is copied unchanged; no codec call
```

The provider's encode and decode receive host page-index pairs and the
migration stream, and launch the format kernels. The NVFP4 launcher splits a
batch into fixed-size chunks of pages. Kernels for a new format follow
`../trtllm-kv-cache-develop/references/adding-kernels.md`.

## Layers and Ownership

```text
KVCM V2 StorageManager (C++)  -- owns pools, cold tiers, staging, events, migration transaction
   | IKvCacheColdPageCodec: configure / queryColdPageBytes / batching id / index location / encode / decode
NativeColdPageCodec (C++)     -- resolves hot buffers per life cycle, routes each life cycle to the provider
   |                             or to its built-in lossless fallback, drains the stream after a provider error
Python bridge (nanobind)      -- forwards configure/encode/decode to the provider with pointers as integers
ColdPageQuantizationCompression subclass (Python)
                              -- format state, per-life-cycle layout and fixed cold-page size, kernel launches
```

| Component | Owns | Never does |
|---|---|---|
| Config and factory | Method selection, format options, admission | Touch pages |
| Python provider | Format state per layer, lossy and lossless spans, layout validation, launches | Allocate, publish, or release KVCM pages; synchronize the stream |
| `NativeColdPageCodec` | Layout resolution, provider-or-fallback routing, lifetime bridge | Hold quantization policy |
| KVCM V2 storage | Hot and cold allocation, staging, migration, events, rollback | Know the format |

## Codec Contract Checklist

Each item is stated in the codec design doc or the compression development
guide, or enforced by the adapter:

- **Fixed size.** The cold-page size is fixed per life cycle and positive.
  Variable-length pages are not supported.
- **One owner per life cycle.** A life cycle is either wholly provider-owned
  or wholly on the lossless fallback; mixing is rejected. Non-attention life
  cycles, such as recurrent state, stay on the fallback.
- **Index location.** The page-index location is chosen per life cycle, and
  both directions use the same one.
  - The NVFP4 provider publishes host indices and copies them into kernel
    parameters in chunks.
  - Host indices are valid only during the call.
  - Device indices are valid until the stream work completes.
- **Asynchronous.** Work is enqueued only on the supplied stream, without
  synchronizing. The return value means "submitted", not "completed".
- **No retained pointers.** Base and index pointers are never kept. Disk
  addresses never reach the codec; the Disk path goes through KVCM's pinned
  staging.
- **Exact lossless spans.** Spans the format declares lossless are copied
  byte for byte.
- **Batches.** Non-contiguous pages, partial pages, and large multi-page
  batches are handled. Handle whatever batch a call receives: KVCM may split
  large batches across calls, and a call need not contain every member of a
  batching class.
- **One-shot ownership.** One codec instance for every KVCM the executor
  builds (target and draft), consumed at construction. Never share codec
  state between managers.

## Adding a Cold Format or Codec

1. **Write the representation down.** Per life cycle and buffer, record:
   - which contiguous span is quantized;
   - what is copied unchanged;
   - the scale-group size and alignment;
   - where scales live;
   - the fixed encoded size.

   The NVFP4 format quantizes one contiguous range per row in groups of 16
   with an FP8 scale per group. Prefixes and suffixes are copied, and the
   layer layout is 16-byte aligned.
2. **Configuration.** Add a new `quant` value with its factory and
   architecture-gate branches, or add a new config class following
   `trtllm-kv-cache-compression`. Report `supports_block_reuse()` and
   `supports_speculative_decoding()` truthfully.
3. **Provider.** Subclass `ColdPageQuantizationCompression`. It already
   sets `provides_cold_page_codec = True` and `uses_iteration_lifecycle = False`,
   and implements `create_cold_page_codec()` and `configure()`, which
   publishes host page indices. Implement:
   - `build_codec_state(...)`, returning state with unique KVCM `layer_ids`;
   - `build_lifecycle_metadata(...)` for each life cycle, returning a fixed
     `cold_page_bytes > 0`;
   - `encode_cold_pages(...)` and `decode_cold_pages(...)`.

   Map local layers to global ones through the pipeline layer list. A provider
   that subclasses `KVCacheCompressionManager` directly must also implement
   `create_cold_page_codec()` and `configure()`, and expose
   `codec_state.layer_ids`. Override `configure()` only to publish device page
   indices.
4. **Kernels.**
   - A CUDA kernel library goes in `cpp/tensorrt_llm/kernels/`, registered
     with `add_tllm_kernel_library(...)` in `kernelComponents.cmake`. Expose a
     thin entry point in `cpp/tensorrt_llm/nanobind/kvCacheCompression/bindings.cpp`,
     as `nvfp4_cold_page_encode` and `nvfp4_cold_page_decode` do. Call it from
     the provider through a local
     `from tensorrt_llm.bindings.internal import kv_cache_compression as native`.
   - Triton or CuTe DSL launchers may live in the Python package.
   - KVCM storage and migration code need no change.
   - Give an actionable error on unsupported GPUs.
5. **Lossy flag.** The KV-sharing lender (`kv_cache/sharing`) refuses to
   publish original KV bytes from a manager whose cold codec is lossy. Today
   `KVCacheManagerV2` detects this by comparing the algorithm name with
   `quantization_for_cold_page`. Do not add a second name to that check.
   Propose an algorithm-neutral declaration instead, for example a capability
   on the provider or the config, and agree on it with the KVCM owners.
6. **Admission.** Reject layouts the format cannot express, with actionable
   messages, at construction or in the codec state builder. Never compress a
   layer the user asked for by silently falling back. The fallback is for
   life cycles the format defines as lossless.
7. **Tests.**
   - Add a native gtest for the kernels, and extend the compression unit-test
     fixture so CI builds and runs it.
   - Add Python tests for codec state, layouts, admission, and activation.

## Enabling a Model for RoPE-Preserving Cold Pages

`skip_rope_quantization` keeps the position-encoded part of each K vector in
its active precision. It is honored only for the model types in
`_SKIP_ROPE_QUANTIZATION_MODEL_TYPES`, in
`quantization_for_cold_page/nvfp4_quantization.py`. Other models log a warning
and quantize whole vectors. To add a model:

1. **Verify the physical cache layout.** Read the attention module and the
   KVCM buffers, not just the model config.
   - A key-only buffer is treated as an MLA latent with trailing RoPE.
   - Separate key and value buffers are treated as partial-rotary GQA with
     leading RoPE.
   - DeepSeek-V4 has its own per-role layout rules.
   - Interleaved rotary dimensions, RoPE in the middle of a vector, or an MLA
     cache with a value buffer need an explicit placement rule and a test.
2. **Derive the RoPE width.** Use rotary-width fields or partial-rotary
   factors, never `rope_theta`.
3. **Keep the refusals.** The format refuses:
   - an all-RoPE vector;
   - RoPE in the middle of a vector;
   - spans that are not multiples of 16 elements.

   V vectors are always quantized in full. Draft caches always quantize whole
   vectors with single NVFP4 and identity global scales. They log a warning
   when `skip_rope_quantization` is set, and ignore `nvfp4_residual_dim`
   without a message.
4. **Add the model type with a test.** The test must check placement on that
   model's real layout.

## When Pages Become Cold

- The hot eviction controller evicts pages LRU within a priority.
  Unreferenced pages at or above `cold_tier_offload_min_priority` (default 30)
  move to a cold tier and are encoded. Lower-priority pages are dropped.
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
- **Speculative decoding.** Target and separate draft KVCMs encode
  independently. Admission currently accepts one-model MTP-EAGLE and
  one-model EAGLE3 and rejects other modes, including vanilla MTP. The list
  lives in `validate_kv_cache_compression_compatibility()`; read it there
  rather than copying it.
- **Mixed-precision KV.** Rejected together, by
  `TorchLlmArgs.validate_mixed_precision_kv`, which rejects any compression
  configuration with a `mixed_nvfp4_*` dtype.
- **Where checks run.** Different checks run in different places, so look in
  all of them before adding or relaxing one:
  - the architecture, block-reuse and speculative-mode checks are in
    `validate_kv_cache_compression_compatibility()`;
  - context-parallel Helix and an unsupported `quant` value are rejected in
    `create_kv_cache_compression_manager()`;
  - layout limits are enforced in the codec-state builder.

  The validator and the factory run during executor construction, before KVCM
  allocation, not when the configuration is parsed.
- **Active KV that is already NVFP4.** The method is skipped, with an info
  log.
- **CUDA graphs and overlap.** Encode and decode run on the migration stream,
  outside graph capture, so keep CUDA graphs and the overlap scheduler on, as
  in every production measurement.
- **Not guarded today.** Disaggregated serving and beam search have no
  explicit check. The NVFP4 example documents a disaggregated recipe. Treat
  beam search and any other combination nobody has validated as unsupported.
  Add an explicit rejection when you find one that fails.
- **Mixed-dtype KV layer specs.** Rejected.
- **The KV-sharing lender.** It refuses lossy cold codecs.

## Numerics Pitfalls Seen Before

- **Repeated round trips.** A page can be offloaded, onboarded, and
  offloaded again.
  - A format that re-derives scales from already-quantized values drifts on
    every round trip.
  - Design it so that re-encoding a decoded page reproduces the same codes,
    or measure and document the drift.
  - Test several round trips, not one.
- **Hand-written references can be wrong**, for example by grouping
  dimensions in a different order than the kernel. Test the layout contract
  instead: spans, scale groups, byte-exact lossless parts, round-trip
  stability.
- **Global scales.** Identity by default. With `scale_checkpoint_path`,
  per-layer K/V scales come from an NVFP4 ModelOpt checkpoint, and are loaded
  and validated at initialization.
  - They apply to target layers with separate key and value buffers.
  - Key-only (MLA) layers and draft caches keep identity scales.
  - The DeepSeek-V4 compressed cache uses only the K scale.

  Never compute scales while serving.
- **RoPE values are the sensitive part.** `skip_rope_quantization` and the
  DeepSeek-V4 two-component residual (`nvfp4_residual_dim: 64`, the default)
  are accuracy options. Measure them per model.
- **Apparent drops can be artifacts.** An apparent accuracy drop can come
  from truncation or a protocol mismatch. Check with
  `trtllm-kv-cache-accuracy-eval` before blaming the format.

## Activation and Performance

- **Activation evidence.** The offload and onboard byte counters
  (`trtllm_kv_cache_offload_bytes_total`, `trtllm_kv_cache_onboard_bytes_total`)
  also count lossless-fallback life cycles. Show that compressed life cycles
  moved, for example from cold-page sizes or provider call counts, not only
  that bytes moved.
- **Expected benefit.** More pages per cold-tier byte and fewer bytes moved.
  This shows up as a higher reuse hit rate and lower TTFT when the working set
  exceeds GPU capacity. Use multi-turn or shared-prefix traces that force
  offload.
- **Baselines.** The default lossless codec at the same tier capacity
  (iso-bytes), and at the same page count (iso-capacity). State which.
- **Attribution.** Report onboarding latency, bytes moved, and encode and
  decode time against copy time. Say whether they overlap model execution or
  block a request (`perf-kv-cache-tuning`).

## Review Checklist

```markdown
## Cold-page review
- Representation written down (spans, scale groups, fixed size per life cycle): <yes/no>
- Codec contract items kept (size, ownership per life cycle, index location, async, pointers): <risks>
- Layout verified on the real cache layout for each enabled model: <models>
- Unsupported layouts and combinations rejected with actionable errors: <where>
- Lossy codec declared without adding a method name to KVCM: <how / n.a.>
- Repeated round-trip stability tested: <test>
- Reuse of onboarded pages covered in accuracy evaluation: <yes/no>
- Host and Disk paths, failure rollback, non-default streams tested: <tests>
- Activation shown for compressed life cycles, not only byte counters: <how>
- Perf: iso-bytes or iso-capacity A/B with migration attribution: <summary>
```
