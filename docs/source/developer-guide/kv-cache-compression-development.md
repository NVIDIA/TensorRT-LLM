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
- [Capability declarations](#capability-declarations)
- [Configuration, construction, and binding](#configuration-construction-and-binding)
- [Calibration and offline artifacts](#calibration-and-offline-artifacts)
- [Iteration-driven methods](#iteration-driven-methods)
- [Storage-bound codec providers](#storage-bound-codec-providers)
- [Ownership and failure boundaries](#ownership-and-failure-boundaries)
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

The framework supports two extension models.

| Extension model | Manager flags | Execution point | Example |
|---|---|---|---|
| Iteration-driven | `uses_iteration_lifecycle = True` | PyExecutor resource-manager callbacks around model iterations | TriAttention periodic token eviction |
| Storage-bound | `provides_cold_page_codec = True` | KVCM hot/cold migration | NVFP4 cold-page quantization |

An iteration-driven method runs compression logic at request or model-iteration
lifecycle boundaries. A storage-bound method runs when KVCM moves Pages across
a representation boundary.

Keep the two mechanisms separate. Do not add migration policy to an iteration
hook, and do not make a cold-page provider allocate or publish KVCM Pages.

## Capability declarations

There are two independent capability layers.

### Manager execution model

| Attribute | Meaning |
|---|---|
| `uses_iteration_lifecycle` | Register the manager for per-iteration resource callbacks |
| `provides_cold_page_codec` | Pass the manager to KVCM during construction as a codec provider |

### Configuration behavior

| Member | Meaning | Framework effect |
|---|---|---|
| `changes_physical_kv_length` | Physical and logical KV lengths can diverge | Binding tells KVCM that compression manages history reconciliation |
| `supports_block_reuse()` | The method preserves the block-reuse contract | Admission can keep block reuse enabled |
| `supports_speculative_decoding()` | The configuration supports at least one speculative mode | Admission proceeds to method-specific mode checks |

`changes_physical_kv_length` does not conflict with
`supports_block_reuse()`. They describe independent properties. A method can
change or compact only the generation/decode suffix while preserving the
reusable prompt-prefix blocks and their identity. Such a method can set
`changes_physical_kv_length = True` and still return `True` from
`supports_block_reuse()`. A method should report block reuse as unsupported
only when it cannot preserve the reusable prefix or its mapping contract.

Capability methods are admission predicates, not runtime fallbacks. A method
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
extension models then take different paths:

- An iteration-driven manager is registered as
  `ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER`. After the target and
  optional draft KVCMs have been constructed, `bind_kv_cache_managers()` binds
  their concrete V2 instances. The resource-manager order places the
  compression manager after the native KV managers so its generation-end hook
  observes finalized writes and rewinds.
- A storage-bound manager is passed to KVCM as a cold-page codec provider. It is
  not registered for iteration callbacks when `uses_iteration_lifecycle` is
  `False`. KVCM retains the native codec adapter, which in turn retains the
  provider and its codec state for the KVCM lifetime.

### `bind_kv_cache_managers()`

The base implementation:

1. requires the target manager and optional draft manager to be
   `KVCacheManagerV2` instances;
2. stores the target and optional draft references; and
3. sets `kv_compression_manages_history` from
   `config.changes_physical_kv_length` on each bound manager.

Override this method only when the algorithm must derive manager-lifetime state
from the constructed KVCM, such as pool geometry, batch capacity, or draft-cache
participation. Call `super().bind_kv_cache_managers(...)` first. Do not use the
method to copy KVCM allocation state into a second owner.

`has_independent_draft_kv_cache` reports whether an optional draft KVCM was
bound. Target and draft caches remain independent owners: an algorithm must not
reuse target mappings or decisions for a draft cache unless its contract
explicitly defines that relationship.

## Calibration and offline artifacts

TensorRT-LLM is an inference platform. A compression method must not perform
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

## Iteration-driven methods

`KVCacheCompressionManager` is a `BaseResourceManager` adapter. PyExecutor calls
the resource-manager interface; the base class translates those calls into five
semantic hooks that algorithms may override.

```text
first context chunk
  prepare_resources()
    -> on_request_init(request), once

every scheduled iteration, before forward
  prepare_resources()
    -> on_generation_step_begin(scheduled_batch)

every scheduled iteration, after forward and native KVCM update
  update_resources()
    -> on_context_step_end(final-prefill requests), when present
    -> on_generation_step_end(scheduled_batch)

request completes or aborts
  free_resources()
    -> on_request_finish(request)
```

### Lifecycle hooks

| Hook | Exact trigger | Appropriate work |
|---|---|---|
| `on_request_init(request)` | The request's first context chunk reaches `prepare_resources()` | Allocate request-local algorithm state or raise a capacity high-water mark |
| `on_context_step_end(requests)` | After forward, for requests whose final context chunk ran in that iteration; intermediate chunks are excluded | Perform one batched context-final action for that cohort |
| `on_generation_step_begin(scheduled_batch)` | Every scheduled iteration, before forward | Inspect the generation cohort or snapshot scheduler-owned information that cannot be reconstructed after overlap |
| `on_generation_step_end(scheduled_batch)` | Every scheduled iteration, after forward and the native KVCM update | Process the generation cohort against authoritative KV state |
| `on_request_finish(request)` | Request completion or abort | Release request-local algorithm state; do not release KVCM Pages |

All five hooks default to no-op. Override only hooks required by the algorithm.
For example, an algorithm that can derive all round inputs at generation end
does not need a generation-begin snapshot.

`on_context_step_end()` receives the scheduler's
`context_requests_last_chunk` cohort. It does not infer context completion by
watching request-state transitions. This matters for short-output requests and
the overlap scheduler, where a request may move directly toward completion.

The generation begin and end hooks run for every scheduled batch, including a
context-only or mixed batch. An algorithm must select the generation requests
it handles from `scheduled_batch`. `on_generation_step_end()` is the normal
final-reconciliation point for physical eviction. Because the compression
manager is ordered after KVCM, the hook sees accepted writes, rewinds, and
current mappings before it compacts, publishes the new visible length, and asks
KVCM to resize or reclaim.

### Resource-manager adapter methods

Algorithm subclasses normally **do not override** `prepare_resources()`,
`update_resources()`, or `free_resources()`. These methods define the common
translation from PyExecutor callbacks to the semantic hooks above. Overriding
them can bypass first-chunk gating, final-context batching, or the required
post-KVCM ordering.

The base class also returns zero from `get_max_resource_count()` and
`get_needed_resource_to_completion()`. Compression does not own schedulable
physical capacity; KVCM does. A new method should not override these accessors
to model KV capacity a second time.

## Storage-bound codec providers

A storage-bound manager sets:

```python
uses_iteration_lifecycle = False
provides_cold_page_codec = True
```

KVCM calls the provider only for representation-changing hot/cold migration.
Hot/hot and cold/cold movement remains a KVCM copy operation.

The current provider path is:

```text
KVCM construction
  -> provider.create_cold_page_codec(...)
       -> provider.build_codec_state(...)
       -> create_python_cold_page_codec(provider, codec_state)
            -> PythonColdPageCodec
            -> NativeColdPageCodec

KVCM configures all hot pool groups
  -> NativeColdPageCodec resolves provider-owned lifecycles and hot buffers
  -> provider.configure(codec_state, resolved_lifecycles)
       -> provider.build_lifecycle_metadata(...) for each lifecycle
       -> fixed cold-page size and Page-index location

KVCM hot/cold migration
  -> NativeColdPageCodec.encode()/decode()
  -> provider.encode_cold_pages()/decode_cold_pages()
  -> native algorithm launcher on the supplied CUDA stream
```

### Cold-page storage foundation and format layout

KVCM V2's cold-page mechanism is the algorithm-neutral storage foundation for
storage-bound compression; it is not itself a compression method. KVCM treats
each cold Page as one fixed-size opaque byte record. It allocates and releases
cold Slots, routes Pages across cache levels, stages Disk I/O, tracks completion
events, publishes the completed mapping, and rolls back failed migrations. It
does not interpret the compression format inside the record.

A storage-bound compression method supplies that missing format contract
through the existing cold-page codec interface. For every provider-owned
lifecycle or layer group, define:

- which hot buffers are encoded and which are preserved losslessly;
- the order, byte offsets, alignment, and padding of records in the cold Page;
- the packed data, scale, and auxiliary-buffer representation;
- one fixed cold Page byte size; and
- the matching batched encode and decode operations.

The native adapter derives resolved hot lifecycle and buffer descriptors from
KVCM's `PoolGroupDesc`. The provider can build and validate its layout in Python
from those resolved descriptors, then pass immutable layout metadata to its
native launcher. The layout belongs to the compression method and codec; do not
add a format-specific branch to KVCM. Once the codec reports the fixed Page size
and implements the transform, the existing cold-page mechanism manages the
compressed Slots and their lifecycle without compression-specific Page
management changes.

Iteration-driven methods that do not introduce a separate storage
representation do not need a cold-page layout. Storage-bound methods that store
a different cold-tier representation, for example in Host or Disk memory,
should use this interface rather than building a second Page/Slot manager. For
the storage ABI, see the
[KVCM V2 Cold-Page Codec Design](kv-cache-cold-page-codec.md). See
[PR #18091](https://github.com/NVIDIA/TensorRT-LLM/pull/18091) for a concrete
NVFP4 layout and provider implementation.

### General storage-bound provider contract

A storage-bound method can subclass `KVCacheCompressionManager` directly and
provide its own cold-page codec. This is the general path for storage formats
that do not use the token-wise quantization helper.

| Member | `KVCacheCompressionManager` behavior | Storage-bound subclass responsibility |
|---|---|---|
| `__init__()` | Stores the configuration and initializes target/draft KVCM references | Normally inherit; call `super().__init__()` when adding manager-lifetime state |
| `uses_iteration_lifecycle` | Defaults to `True` | Set to `False` |
| `provides_cold_page_codec` | Defaults to `False` | Set to `True` |
| `create_cold_page_codec()` | Returns `None` | Required: return an independent owning native `IKvCacheColdPageCodec` for each call |
| `encode_cold_pages()`, `decode_cold_pages()` | Raise `NotImplementedError` | Implement when the returned codec delegates format transforms back to the Python provider; a self-contained native codec may own these operations directly |

The returned codec must satisfy the same ownership, lifetime, stream, and
migration contracts regardless of where its implementation lives. A different
storage format is not a reason to add an algorithm-specific path to KVCM.

### Token-wise cold-page quantization helper

For token-wise cold-page quantization, subclass
`ColdPageQuantizationCompression`. It supplies the common registration and
Python/native bridge, so the format subclass defines only its state, layout,
and batched transform.

| Member | `ColdPageQuantizationCompression` behavior | Quantization format responsibility |
|---|---|---|
| `__init__()` | Inherits compression configuration and KVCM state from the general base | Override only to load immutable format metadata, and call `super().__init__()` |
| `uses_iteration_lifecycle`, `provides_cold_page_codec` | Selects storage-bound execution with `False` and `True` | Inherit |
| `create_cold_page_codec()` | Builds independent codec state and returns an owning native wrapper | Inherit |
| `configure()` | Builds and retains metadata for each provider-owned lifecycle, reports one fixed cold Page size per lifecycle, and selects host-resident `PageIndexPair` arrays | Inherit when this index contract fits |
| `build_codec_state()` | Raises `NotImplementedError` | Required: define codec-lifetime format state and provider-owned layers |
| `build_lifecycle_metadata()` | Raises `NotImplementedError` | Required: resolve and validate one lifecycle's physical layout and launch metadata |
| `encode_cold_pages()`, `decode_cold_pages()` | Inherit the general base placeholders | Required: dispatch the batched format-specific transforms |

### Required helper method: codec state

KVCM supplies the cache configuration, runtime KV dtype, PP-local layer
mapping, per-layer KV-head count, per-layer head dimension, and whether the
cache belongs to a draft model. `build_codec_state()` should resolve stable
facts available before KVCM allocates its cold tiers, such as:

- provider-owned layer IDs;
- runtime dtype and per-layer geometry;
- model-supplied quantization metadata; and
- immutable format parameters.

The returned object must expose unique `layer_ids`. The native adapter uses
them to determine which KVCM lifecycles the provider owns. Each
`create_cold_page_codec()` call must create independent state, and the native
wrapper retains that state for the codec lifetime. Do not keep mutable
lifecycle metadata only on the shared compression-manager object.

### Required helper method: lifecycle layout

The native adapter converts KVCM's authoritative hot pool descriptors into
resolved lifecycles before it calls `configure()`. The common implementation
then calls `build_lifecycle_metadata(codec_state, lifecycle)` for each
provider-owned lifecycle and retains the results on the codec state.

`build_lifecycle_metadata()` must validate the resolved hot buffer roles,
addresses, Slot strides, byte sizes, alignment, and transform geometry. It must
produce the fixed cold Page byte size and immutable metadata needed by later
launches. Derive this information from the resolved descriptors rather than
guessing a model layout from its name.

A lifecycle must be entirely provider-owned or entirely handled by the
embedded lossless codec. The native adapter rejects mixed ownership, duplicate
provider layers, and provider layers missing from KVCM's descriptors.
Provider-unowned lifecycles, such as recurrent state in a hybrid model, remain
lossless.[^mixed-lifecycle-host-limit]

[^mixed-lifecycle-host-limit]: On hosts where KVCM uses chunked pinned-memory
    registration, the current adapter rejects a model that combines
    provider-owned and lossless-fallback lifecycles because the embedded
    lossless codec cannot split its batched copies at those registration
    boundaries.

### Required helper methods: batched transforms

`encode_cold_pages()` and `decode_cold_pages()` receive the codec state,
provider-lifecycle index, cold allocation base address, `PageIndexPair` array,
Page count, and KVCM-owned CUDA stream.

The common quantization base selects host-resident Page indices. Consume or
copy the `PageIndexPair` array before the Python callback returns; the GPU
transform itself should remain asynchronous on the supplied stream.

Submit every Page in the call to the format-specific native launcher. Avoid a
Python loop over Pages or layers; launcher-internal tiling or chunking belongs
below this interface. A call may be the original migration batch or a chunk
created by Page-index or Disk staging, so do not assume it contains every Page
or lifecycle from the original KVCM operation.

Enqueue work only on the supplied stream, and do not retain the cold pointer,
Page-index pointer, or stream past its documented lifetime. The provider must
not synchronize successful work, publish Page mappings, release Slots, or
perform Disk I/O. If a Python provider throws after beginning submission, the
native adapter drains that stream before reporting failure so KVCM can roll
back safely.

The exact `IKvCacheColdPageCodec` ABI, host/device index lifetimes, batching,
staging, and failure transaction are documented in the
[Cold-Page Codec Design](kv-cache-cold-page-codec.md).

## Ownership and failure boundaries

| Component | Owns | Does not own |
|---|---|---|
| Compression configuration and factory | Method selection and supported-combination admission | Pages, kernels, or request mappings |
| Compression manager | Algorithm cadence, request state, decisions, format metadata, and algorithm launches | KVCM allocation policy or Attention runtime state |
| Native cold-page adapter | KVCM-layout resolution, provider routing, fallback routing, and Python/native lifetime bridge | Format-specific quantization policy |
| KVCM V2 | Pages, Slots, pools, mappings, migration streams, events, publication, release, rollback, and cold storage | Algorithm scores or quantization decisions |
| Attention backend | Consumption of the published active GPU representation | Cold storage and migration |

Before physical mutation, a method may reject, defer, or perform a legal no-op.
After it submits work or moves bytes, it must follow the framework's completion
and failure contract; it must not silently fall back while leaving visible state
partially updated.

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
2. Bind stable KVCM geometry in `bind_kv_cache_managers()`.
3. Override only the required semantic hooks.
4. Keep selection policy separate from generic movement or compaction.
5. Publish completion before resizing or releasing KVCM-owned capacity.
6. Do not override the resource-manager adapter methods unless the framework
   contract itself is being changed.

### 3B. Implement a storage-bound method

1. Set `uses_iteration_lifecycle = False` and
   `provides_cold_page_codec = True`.
2. Define the method's fixed cold-page layout: encoded and lossless buffers,
   record order, offsets, alignment, padding, auxiliary metadata, and total Page
   bytes. Follow the
   [cold-page layout contract](#cold-page-storage-foundation-and-format-layout).
3. Choose the provider base that matches the method:
   - for token-wise cold-page quantization, subclass
     `ColdPageQuantizationCompression` and implement `build_codec_state()` plus
     `build_lifecycle_metadata()`; or
   - for another storage format, subclass `KVCacheCompressionManager` directly
     and implement `create_cold_page_codec()`.
4. Define the provider-owned layer set and a lossless policy for unowned
   lifecycles.
5. Implement batched `encode_cold_pages()` and `decode_cold_pages()` using the
   supplied stream.
6. Keep fixed cold-page size, Page-index location, pointer lifetime, and
   asynchronous failure behavior consistent with the native codec contract.

The native `IKvCacheColdPageCodec` interface and Python/native adapter already
connect a storage-bound provider to KVCM V2. A new compression format should
implement the Python provider and its algorithm launcher, not add a format
branch to `storageManager.cpp`, `kvCache.cpp`, or the KVCM migration engine.

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
