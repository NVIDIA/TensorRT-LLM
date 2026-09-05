<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV Cache Compression Development Guide

This guide is for contributors adding a KV cache compression method to the
TensorRT-LLM PyTorch backend. It describes the common compression-manager
lifecycle, the two extension models, and the ownership boundary with
`KVCacheManagerV2` (KVCM V2).

For user-facing configuration and method selection, see
[KV Cache Compression](../features/kv-cache-compression.md). For the C++ hot/cold
storage ABI, page-index lifetime, staging, and migration transaction, see the
[KVCM V2 Cold-Page Codec Design](kv-cache-cold-page-codec.md).

- [Architecture](#architecture)
- [Configuration contract](#configuration-contract)
- [Configuration, construction, and binding](#configuration-construction-and-binding)
- [Iteration-driven methods](#iteration-driven-methods)
- [Storage-bound codec providers](#storage-bound-codec-providers)
- [Ownership and failure boundaries](#ownership-and-failure-boundaries)
- [Calibration and offline artifacts](#calibration-and-offline-artifacts)
- [Adding a compression method](#adding-a-compression-method)
- [Validation](#validation)

## Architecture

KV cache compression works through `KVCacheManagerV2` (KVCM V2). A compression
method defines when compression runs and which method-specific transformation
it applies. All KV-cache interactions go through KVCM V2, which remains the
authority for Pages, Slots, cache levels, allocation, migration, reuse,
completion ordering, and mapping publication.

```text
KvCacheCompressionConfig
        |
        v
factory and compatibility checks
        |
        v
KVCacheCompressionManager
        |
        +-- iteration-driven method
        |     lifecycle hook -> method-specific compression action
        |
        `-- storage-bound method
              codec provider -> native codec adapter -> KVCM migration
```

The framework provides two integration paths. A manager selects either or both
paths through class attributes.

| Integration path | Manager attribute | Execution point | Example |
|---|---|---|---|
| Iteration-driven | `uses_iteration_lifecycle = True` | PyExecutor resource-manager callbacks around model iterations | TriAttention periodic token eviction |
| Storage-bound | `provides_cold_page_codec = True` | KVCM hot/cold migration | NVFP4 cold-page quantization |

An iteration-driven method runs compression logic at request or model-iteration
lifecycle boundaries. A storage-bound method runs when KVCM moves Pages across
a representation boundary.

Keep the two mechanisms separate. Do not add migration policy to an iteration
hook, and do not make a cold-page provider allocate or publish KVCM Pages.

## Configuration contract

Compression configurations expose attributes and compatibility methods that
the framework uses during construction.

| Member | Type | Purpose |
|---|---|---|
| `changes_physical_kv_length` | Class attribute | Tells KVCM whether compression manages physical-history reconciliation |
| `supports_block_reuse()` | Method | Reports whether block reuse remains valid |
| `supports_speculative_decoding()` | Method | Reports whether the current configuration supports speculative decoding; method-specific mode checks still apply |

`changes_physical_kv_length` does not conflict with
`supports_block_reuse()`. They describe independent properties. A method can
change or compact only the generation/decode suffix while preserving the
reusable prompt-prefix blocks and their identity. Such a method can set
`changes_physical_kv_length = True` and still return `True` from
`supports_block_reuse()`. A method should report block reuse as unsupported
only when it cannot preserve the reusable prefix or its mapping contract.

Compatibility methods are admission predicates, not runtime fallbacks. A method
may still impose narrower mode, backend, or model-layout checks in the common
compatibility validator or its construction path.

## Configuration, construction, and binding

Public method configurations inherit from `KvCacheCompressionConfig` in
`tensorrt_llm/llmapi/llm_args.py`. The `algorithm` field selects a concrete
configuration and manager. Method-specific options remain on the concrete
configuration rather than the common base.

`create_kv_cache_compression_manager()` in
`tensorrt_llm/_torch/pyexecutor/_util.py` performs admission checks and creates
the concrete manager. Validate unsupported combinations here, before model
execution or storage migration. This includes backend, GPU architecture, block
reuse, speculative-decoding mode, and other method-level restrictions.

The factory runs before KVCM construction because a storage-bound manager must
be available as a codec provider while KVCM builds its cold layout. The two
integration paths are wired differently:

- An iteration-driven manager is registered for lifecycle callbacks. If it
  needs access to KVCM, `bind_kv_cache_managers()` binds it after KVCM
  construction.
- A storage-bound manager provides a `NativeColdPageCodec` during KVCM
  construction and is not registered for iteration callbacks.

## Iteration-driven methods

KV cache compression can run at stable boundaries in the inference lifecycle.
During a prefill or decode forward pass, Attention consumes a stable view of the
KV cache. Once that stage completes and before the next one begins, the
framework can update the physical KV state used by subsequent execution.

After the final prefill chunk, a method can compress the completed context cache
before generation begins, reducing the KV footprint carried into decode.
Between decode iterations, a method can periodically compress newly accumulated
KV state to control cache growth during long generation.

Compression methods that operate at these lifecycle boundaries are
iteration-driven. For example, TriAttention periodically evicts selected decode
KV between generation iterations.

```text
request initialization
  -> prepare method state

prefill / chunked prefill
  -> final context chunk completes
  -> optional context-final compression

each decode iteration
  -> iteration begins
  -> model forward and KVCM update
  -> optional compression before the next iteration

request completes or aborts
  -> release method state
```

`KVCacheCompressionManager` exposes five semantic hooks at these lifecycle
points. Algorithms override only the hooks they need.

### Lifecycle hooks

| Hook | Exact trigger | Appropriate work |
|---|---|---|
| `on_request_init(request)` | Before a request's first prefill chunk | Initialize request-local compression state |
| `on_context_step_end(requests)` | After a request's final prefill chunk | Compress the completed context before generation, when needed |
| `on_generation_step_begin(scheduled_batch)` | Before each scheduled forward iteration | Prepare an iteration-level compression action, when needed |
| `on_generation_step_end(scheduled_batch)` | After each scheduled forward iteration and KV-cache update | Compress the updated KV state before the next iteration, when needed |
| `on_request_finish(request)` | When a request completes or aborts | Release request-local compression state |

All five hooks default to no-op. For example, an algorithm that can derive all
round inputs at generation end does not need a generation-begin snapshot.

These lifecycle points reuse PyExecutor's existing request cycle. When
iteration-driven compression is enabled, the framework registers the
compression manager and invokes the corresponding hooks. Developers only need
to subclass `KVCacheCompressionManager` and implement the hooks their method
uses; the framework handles registration and callback wiring.

## Storage-bound codec providers

Cold-page compression is a storage-bound method that encodes KV Pages into a
compressed format during offloading and decodes them during onboarding. Each
method defines a compressed layout for the target cold tier, such as Host
memory or Disk. The compression and transfer kernels can also be fused or
co-optimized.

Quantization is one concrete compression approach: quantization runs during
offloading and dequantization runs during onboarding. Other compression
approaches use the same storage-bound flow by defining their compressed layout
and encode/decode operations.

```text
GPU hot Page
  -> offloading: encode
  -> Host or Disk cold Page
  -> onboarding: decode
  -> GPU hot Page
```

### Cold-page compression APIs

KVCM manages Page allocation and migration. A storage-bound compression method
defines its compressed format and implements the relevant APIs below.

| API | Purpose |
|---|---|
| `create_cold_page_codec()` | Create the cold-page codec |
| `build_codec_state()` | Define the format state and the layers it handles |
| `build_lifecycle_metadata()` | Define and validate the cold-page layout |
| `encode_cold_pages()` | Encode a batch of KV Pages into the cold-page representation |
| `decode_cold_pages()` | Decode a batch of cold Pages back into the hot KV representation |

## Ownership and failure boundaries

| Component | Owns | Does not own |
|---|---|---|
| [`KvCacheCompressionConfig`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/llm_args.py#L3783) and [`create_kv_cache_compression_manager()`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/_util.py#L2830) | Method selection and supported-combination admission | Pages, kernels, or request mappings |
| [`KVCacheCompressionManager`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/resource_manager.py#L2775) | Algorithm cadence, request state, decisions, format metadata, and algorithm launches | KVCM allocation policy or Attention runtime state |
| [`NativeColdPageCodec`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/kv_cache_compression/nativeColdPageCodec.h#L61) | KVCM-layout resolution, provider routing, fallback routing, and Python/native lifetime bridge | Format-specific quantization policy |
| [`KVCacheManagerV2`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py#L789) | Pages, Slots, pools, mappings, migration streams, events, publication, release, rollback, and cold storage | Algorithm scores or quantization decisions |
| [`AttentionBackend`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/attention_backend/interface.py#L1001) | Consumption of the published active GPU representation | Cold storage and migration |

Before physical mutation, a method may reject, defer, or perform a legal no-op.
After it submits work or moves bytes, it must follow the framework's completion
and failure contract; it must not silently fall back while leaving visible state
partially updated.

## Calibration and offline artifacts

Some compression methods require calibration or other offline artifacts.
TensorRT-LLM is an inference platform, so a compression method must not perform
model calibration, corpus collection, parameter fitting, or calibration-file
generation in the inference critical path. Users should produce any required
artifacts before serving with the method's upstream tooling,
[NVIDIA ModelOpt](https://nvidia.github.io/Model-Optimizer/), or another
documented offline workflow.

The runtime may accept an artifact path in the method configuration, then load
and validate that artifact during manager or model initialization. Once request
execution begins, lifecycle hooks and cold-page `encode()`/`decode()` calls must
consume ready-to-use immutable state. They must not launch a calibration
workflow from prefill, generation, Page migration, or request teardown.

Fail fast during initialization when a required artifact is missing,
incompatible with the model, or malformed. Loading optional checkpoint metadata
is also an initialization concern. Dynamic quantities that are mathematically
part of each compression operation, such as per-block scales computed while
encoding a Page, are runtime transform state rather than model calibration.

## Adding a compression method

### 1. Define the configuration and telemetry

1. Add a `KvCacheCompressionConfig` subclass in `llm_args.py`.
2. Choose a unique `algorithm` value and keep method-specific fields on the
   concrete class.
3. Declare physical-length, block-reuse, and speculative-decoding capabilities.
4. Add the class to `KvCacheCompressionConfigType`.
5. If the method requires calibration, expose only the path or identifier of a
   precomputed artifact. Document the external offline workflow; do not add
   calibration execution to TensorRT-LLM.
6. Add the algorithm value to the existing compression-algorithm telemetry
   allowlist. Opt in only safe, low-cardinality method fields; model paths,
   calibration paths, and other user data must remain excluded.
7. Run `python3 scripts/generate_llm_args_golden_manifest.py`, review the
   generated schema change, and run the telemetry manifest tests. Public
   configuration is not complete until the manifest is updated deliberately.

### 2. Register Python admission and construction

1. Extend `validate_kv_cache_compression_compatibility()` with only the
   method's real unsupported combinations.
2. Add a factory branch in `create_kv_cache_compression_manager()`.
3. Keep optional heavy algorithm imports local to the selected factory branch.
4. Fail before construction for an explicitly requested unsupported method or
   combination.

The KV cache compression framework owns a method's configuration dispatch,
compatibility policy, manager construction, lifecycle integration, and
algorithm policy. Keep that method-specific Python logic in `llm_args.py`,
`_util.py`, and the method's package under
`tensorrt_llm/_torch/kv_cache_compression/`, together with only the native
launchers or bindings it needs. Use the existing executor and KVCM V2
interfaces; do not add method-specific branches to the runtime or KVCM C++.
Change those shared components only when extending an algorithm-neutral
framework contract.

### 3A. Implement an iteration-driven method

1. Subclass `KVCacheCompressionManager`; retain the default
   `uses_iteration_lifecycle = True`.
2. If needed, bind stable KVCM geometry in `bind_kv_cache_managers()`.
3. Override only the required semantic hooks.
4. Keep selection policy separate from generic movement or compaction.
5. Publish completion before resizing or releasing KVCM-owned capacity.

### 3B. Implement a storage-bound method

1. Set `provides_cold_page_codec = True`. Set
   `uses_iteration_lifecycle = False` when the method does not use iteration
   hooks.
2. Implement the relevant APIs from
   [Cold-page compression APIs](#cold-page-compression-apis).
3. Define the method's format state, supported layers, compressed layout, and
   batched encode/decode operations through those APIs.

The framework connects storage-bound compression to KVCM V2 through the
existing interfaces. A new compression format implements its provider APIs and
algorithm launcher.

### 4. Add method-specific kernels

Keep the manager responsible for policy and the kernel responsible for a
batched transform. Put Python launchers with the method implementation. Triton
and CuTe DSL kernels can live in the method package; a CUDA implementation can
live under `cpp/tensorrt_llm/kernels/` with the smallest required binding.

For a storage-bound method, the launcher must consume the resolved lifecycle
metadata, Page-index batch, cold base pointer, and CUDA stream provided through
the existing codec adapter. Adding a CUDA kernel or binding does not require
changing the KVCM storage or migration code. Preserve non-contiguous Pages,
partial Pages, batching, supplied-stream execution, pointer lifetime, and the
asynchronous failure contract.

Prefer an existing production primitive when its numerical and layout contract
matches. Keep backend-specific imports local, provide an actionable unsupported
architecture error, and test the launcher separately before wiring it into a
manager lifecycle.

### 5. Preserve reuse and serving compatibility where possible

KV-cache block reuse and disaggregated serving are important compression use
cases. Preserve and test them when the algorithm's semantics permit it; they
are not unconditional requirements for every method. Declare
`supports_block_reuse()` truthfully, validate context/generation ownership and
transfer behavior for disaggregated serving, and avoid silently disabling
either feature. If a combination cannot be supported, reject it during
compatibility validation and document the limitation.

### 6. Document the method

Add a concise method section to the user-facing feature page and a complete,
runnable example under `examples/kv_cache_compression/`. Document the method's
measured accuracy or output-quality impact, tested model and workload scope,
performance evidence, and known unsupported combinations. Keep algorithm
details out of the KVCM storage-ABI document.

## Validation

Validation is an evidence workflow, not only a collection of unit tests. A new
method is not ready when its kernels merely run or its configuration parses.
Complete the following stages in order, then preserve the validated behavior
with automated regression tests.

```text
Accuracy and output quality
  -> real end-to-end activation and compatibility
  -> end-to-end performance
  -> automated regression gates
```

### 1. Establish the accuracy and output-quality contract

Choose the uncompressed or full-KV path as the reference before measuring
performance. Define the method's acceptance criterion and evaluate the real
end-to-end model path on representative models, cache layouts, context lengths,
and workloads. Lossy compression is allowed, but its impact must be measured;
do not assume that a numerically small tensor error implies unchanged model
output.

Record enough information to reproduce the comparison:

- exact model checkpoint, backend, runtime KV dtype, and compression settings;
- calibration artifact and generation workflow, when the method requires one;
- dataset or traffic trace, prompt/chat template, context and output lengths,
  sampling parameters, and repeat protocol;
- reference and compressed scores, absolute and relative deltas, and run-to-run
  variation; and
- compression pressure or frequency, including repeated lossy transforms when
  they can occur in production.

Lossless behavior is preferred when the method can provide it, but it is not a
universal requirement. The acceptable trade-off is method- and
workload-specific. Publish the measured quality impact, tested scope, and known
limitations in the user-facing method documentation so users do not have to
infer them from unit tests.

### 2. Prove real end-to-end behavior and compatibility

Run supported models through the production executor, model engine, KVCM, and
native kernels. Prove that compression actually activated; a successfully
parsed configuration is not activation evidence. Use counters, traces, or
other route evidence to show that the intended lifecycle hook or hot/cold codec
ran and that the compressed state was later consumed correctly.

Exercise the combinations the method claims to support, including:

- prefill and generation lifecycles, request completion and abort;
- non-contiguous and partial Pages, suspend/resume, rewind, and repeated
  compression where applicable;
- block reuse, including reuse of a previously compressed or restored prefix;
- disaggregated serving, including context/generation ownership and KV
  transfer boundaries;
- target and independent draft caches for supported speculative modes; and
- every supported model layout, runtime KV dtype, and cold-tier route,
  including secondary GPU, Host, or Disk as applicable.

Block reuse and disaggregated serving are high-value compression scenarios.
Preserve and validate them whenever the method's semantics permit it. If a
combination is unsupported, reject it before execution and document the
limitation instead of silently disabling the feature. A kernel round-trip test
alone does not establish end-to-end compatibility.

### 3. Measure end-to-end performance

Measure performance only after the accuracy and functional gates pass. Start
with an uninstrumented A/B comparison against the uncompressed path in which
only the compression setting changes. Hold the model, workload, sampling
contract, parallel topology, cache quotas, and serving duration constant. When
capacity is the expected benefit, state whether the comparison holds physical
bytes or logical Page capacity constant.

Report the metrics affected by the method, such as TTFT, inter-token latency,
request and token throughput, cache hit rate, retained KV capacity, migration
bytes, and Host/Disk traffic. Use repeated runs and report variation rather
than selecting one favorable sample. A transform or migration microbenchmark
is useful for kernel development, but it is not evidence of end-to-end benefit.

After the unprofiled result is established, profile the same workload to
separate kernel-work sum, GPU-busy critical path, exposed CPU submission, and
application wall time. Preserve the raw profiler artifacts and distinguish
work that overlaps model execution from latency exposed to the request.

### 4. Turn validated behavior into regression gates

Automated tests should protect the contracts established above. Keep focused
CPU tests fast, add native and kernel tests for data-path contracts, and retain
small end-to-end smoke gates for activation and integration. These tests catch
regressions; they do not replace the documented accuracy and performance
validation.

#### Common framework tests

- configuration parsing, serialization, and factory dispatch;
- telemetry allowlisting, privacy exclusions, and golden-manifest parity;
- admission of supported combinations and early rejection of unsupported ones;
- target-only and independent target/draft construction;
- resource-manager ordering and exact lifecycle-hook cadence;
- block-reuse and disaggregated-serving paths when the method declares them
  supported;
- request completion and abort cleanup; and
- proof that compression activated rather than merely parsing a configuration.

#### Iteration-driven tests

- stable-prefix and protected-tail boundaries;
- repeated compression, Page reuse, rewind, suspend/resume, and overlap;
- byte-correct movement or compaction against a CPU oracle;
- completion ordering before published lengths and KVCM resize; and
- aligned end-to-end accuracy and performance checks.

#### Storage-bound tests

- independent codec state for every KVCM construction;
- hot-layout resolution, provider/fallback lifecycle routing, and rejection of
  mixed ownership;
- fixed cold-page geometry and byte-exact preservation of buffers or spans
  declared lossless;
- encode/decode round-trip accuracy for every supported runtime dtype;
- non-contiguous, partial, and large multi-Page batches;
- non-default stream behavior and pointer lifetime;
- Host and Disk offload/onboard paths, including failure rollback; and
- activation evidence that hot/cold conversion invoked the intended codec.
