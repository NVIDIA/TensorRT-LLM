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

KV cache compression is a strategy layer over KVCM V2. The compression method
decides *when* and *how* to reduce the stored KV representation. KVCM V2 remains
the authority for Pages, Slots, cache levels, allocation, migration, reuse,
events, and publication of a completed mapping.

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
        |     lifecycle hook -> decision/transform -> KVCM resize or reclaim
        |
        `-- storage-bound method
              codec provider -> native codec adapter -> KVCM migration
```

The framework supports two extension models.

| Extension model | Manager flags | Execution point | Example |
|---|---|---|---|
| Iteration-driven | `uses_iteration_lifecycle = True` | PyExecutor resource-manager callbacks around model iterations | TriAttention periodic token eviction |
| Storage-bound | `provides_cold_page_codec = True` | KVCM hot/cold migration | NVFP4 cold-page quantization |

An iteration-driven method can change the token set or physical length while a
request is running. A storage-bound method leaves the active GPU representation
unchanged and transforms a complete Page batch only when KVCM moves it across a
representation boundary.

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
artifacts before serving with the method's upstream tooling, NVIDIA ModelOpt,
or another documented offline workflow.

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

after forward and native KVCM update
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
| `on_context_step_end(requests)` | The iteration containing each request's final context chunk has completed | Perform one batched context-final action for that cohort |
| `on_generation_step_begin(scheduled_batch)` | Before the current iteration's forward | Snapshot scheduler-owned information that cannot be reconstructed after overlap |
| `on_generation_step_end(scheduled_batch)` | After forward and after the native KVCM update | Apply periodic or budget-triggered compression against authoritative KV state |
| `on_request_finish(request)` | Request completion or abort | Release request-local algorithm state; do not release KVCM Pages |

All five hooks default to no-op. Override only hooks required by the algorithm.
For example, an algorithm that can derive all round inputs at generation end
does not need a generation-begin snapshot.

`on_context_step_end()` receives the scheduler's
`context_requests_last_chunk` cohort. It does not infer context completion by
watching request-state transitions. This matters for short-output requests and
the overlap scheduler, where a request may move directly toward completion.

`on_generation_step_end()` is the normal final-reconciliation point for
physical eviction. Because the compression manager is ordered after KVCM, the
hook sees accepted writes, rewinds, and current mappings before it compacts,
publishes the new visible length, and asks KVCM to resize or reclaim.

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

### `create_cold_page_codec()`

KVCM supplies the resolved cache configuration, runtime KV dtype, PP-local
layer mapping, per-layer KV-head count, per-layer head dimension, and whether
the cache belongs to a draft model. The method must return one owning native
`IKvCacheColdPageCodec` object.

KVCM can request more than one codec, for example for target and independent
draft caches or while rebuilding after a construction fallback. Every call
must therefore create independent codec state. Do not store mutable
lifecycle-specific metadata only on the shared compression-manager object.

`ColdPageQuantizationCompression.create_cold_page_codec()` implements the
common bridge:

1. call `build_codec_state()` for format-specific codec-lifetime state;
2. pass the provider and that state to
   `create_python_cold_page_codec()`; and
3. return the resulting native codec to KVCM.

### `build_codec_state()`

Implement this method in a format-specific subclass. It should resolve stable
facts available before KVCM allocates its cold tiers, for example:

- provider-owned layer IDs;
- runtime dtype and per-layer geometry;
- model-supplied quantization metadata; and
- immutable format parameters.

The returned object must expose unique `layer_ids`. The native adapter uses
them to determine which KVCM lifecycles the provider owns. State is retained by
the native codec wrapper, so tensors and metadata referenced by later launches
must remain alive for the codec lifetime.

### `configure()` and `build_lifecycle_metadata()`

During `NativeColdPageCodec.configure()`, the adapter converts KVCM's
authoritative hot pool descriptors into resolved lifecycles. A lifecycle is
either entirely provider-owned or entirely handled by the default lossless
codec. Mixing provider-owned and fallback layers inside one lifecycle is
rejected.

`ColdPageQuantizationCompression.configure()` calls
`build_lifecycle_metadata(codec_state, lifecycle)` for every provider-owned
lifecycle, stores the results on the codec state, and returns:

- the fixed number of bytes in one cold Page; and
- whether the `PageIndexPair` array is in host or device memory.

Use `build_lifecycle_metadata()` to validate the actual hot buffer roles,
addresses, Slot strides, byte sizes, alignment, and transform geometry. Derive
launch metadata from these resolved descriptors rather than guessing a model
layout from its name.

Provider-unowned lifecycles, such as recurrent state in a hybrid model, use the
embedded lossless codec. The native adapter verifies that every declared
provider layer appears exactly once and rejects ambiguous lifecycle ownership.
On host kernels where KVCM requires chunked pinned-memory registration, the
current adapter rejects a configuration that combines provider-owned and
fallback lifecycles because the embedded lossless codec cannot split its
batched copies at those registration boundaries.

### `encode_cold_pages()` and `decode_cold_pages()`

These methods receive:

- the codec state and provider-lifecycle index;
- the cold allocation base address;
- a `PageIndexPair` array address;
- the number of complete Pages; and
- the KVCM-owned CUDA stream.

They must submit the complete batch to the format-specific native launcher.
Avoid a Python loop over Pages or layers; launcher-internal tiling or chunking
belongs below this interface. The current adapter forwards one complete KVCM
batch in one provider call.

The provider may enqueue work only on the supplied stream and must not retain
the cold pointer, Page-index pointer, or stream past the documented lifetime.
It must not synchronize successful work, publish Page mappings, release Slots,
or perform disk I/O. If a Python provider throws after beginning submission,
the native adapter drains that same stream before reporting failure so KVCM can
roll back the migration transaction safely.

The exact `IKvCacheColdPageCodec` ABI, host/device index lifetimes, batching
representatives, staging rules, and failure transaction are documented in the
[Cold-Page Codec Design](kv-cache-cold-page-codec.md).

## Ownership and failure boundaries

| Component | Owns | Does not own |
|---|---|---|
| Compression configuration and factory | Method selection and supported-combination admission | Pages, kernels, or request mappings |
| Compression manager | Algorithm cadence, request state, decisions, format metadata, and algorithm launches | KVCM allocation policy or Attention-private state |
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

Configuration dispatch, compatibility policy, manager construction, and
algorithm policy belong in Python. Keep them in `llm_args.py`, `_util.py`, and
the method's package under `tensorrt_llm/_torch/kv_cache_compression/`; do not
add algorithm selection to KVCM C++.

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
2. Implement `create_cold_page_codec()` directly, or reuse
   `ColdPageQuantizationCompression` and implement `build_codec_state()` plus
   `build_lifecycle_metadata()`.
3. Define the provider-owned layer set and a lossless policy for unowned
   lifecycles.
4. Implement batched `encode_cold_pages()` and `decode_cold_pages()` using the
   supplied stream.
5. Keep fixed cold-page size, Page-index location, pointer lifetime, and
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
runnable example under `examples/kv_cache_compression/`. Keep algorithm details
out of the KVCM storage-ABI document.

## Validation

Start with focused CPU tests, then exercise the real native and model paths.

### Common framework tests

- configuration parsing, serialization, and factory dispatch;
- telemetry allowlisting, privacy exclusions, and golden-manifest parity;
- admission of supported combinations and early rejection of unsupported ones;
- target-only and independent target/draft construction;
- resource-manager ordering and exact lifecycle-hook cadence;
- block-reuse and disaggregated-serving paths when the method declares them
  supported;
- request completion and abort cleanup; and
- proof that compression activated rather than merely parsing a configuration.

### Iteration-driven tests

- stable-prefix and protected-tail boundaries;
- repeated compression, Page reuse, rewind, suspend/resume, and overlap;
- byte-correct movement or compaction against a CPU oracle;
- completion ordering before published lengths and KVCM resize; and
- aligned end-to-end accuracy and performance checks.

### Storage-bound tests

- independent codec state for every KVCM construction;
- hot-layout resolution, provider/fallback lifecycle routing, and rejection of
  mixed ownership;
- fixed cold-page geometry and byte-exact lossless spans;
- encode/decode round-trip accuracy for every supported runtime dtype;
- non-contiguous, partial, and large multi-Page batches;
- non-default stream behavior and pointer lifetime;
- Host and Disk offload/onboard paths, including failure rollback; and
- activation evidence that hot/cold conversion invoked the intended codec.

Performance validation follows correctness. Compare an uninstrumented,
same-shape baseline before using a profiler to attribute kernel work, GPU-busy
critical path, CPU submission, and application wall time separately.
