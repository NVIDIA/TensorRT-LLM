<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCacheManagerV2 Cold-Page Codec Design

## Summary

KVCM2 supports an internal cold-page representation for pages evicted from the hot tier. A hot page may span
multiple GPU-facing pools, while a cold page is stored as one fixed-size opaque blob. An `IKvCacheColdPageCodec`
transforms between these representations.

```text
Hot representation                         Cold representation
multiple kernel-facing pools  <-------->   one opaque blob per page
                                 codec
```

Compression is optional. The default codec concatenates all hot-pool data into the cold blob without reducing its
size. Compression codecs may produce smaller cold pages. Both cases use the same KVCM allocation, migration, and
event-management paths.

This design reduces the number of files and I/O operations used by disk storage. Instead of issuing one operation per
hot pool per page, KVCM can issue one operation per encoded cold page. Further coalescing of adjacent pages remains a
possible optimization.

## Codec API

The C++ interface is named `IKvCacheColdPageCodec` because it defines a representation transform, not necessarily a
compression algorithm.

```cpp
enum class PageIndexLocation : int
{
    kBadLocation = -1,
    kHost,
    kDevice,
};

struct alignas(8) PageIndexPair
{
    int32_t dst;
    int32_t src;
};

static_assert(sizeof(PageIndexPair) == 8);
static_assert(std::is_trivially_copyable_v<PageIndexPair>);

class IKvCacheColdPageCodec
{
public:
    IKvCacheColdPageCodec();
    virtual ~IKvCacheColdPageCodec();

    virtual bool configure(
        PoolGroupDesc const* gpuDescs, PoolGroupIndex numGpuDescs) noexcept = 0;

    // Zero indicates failure.
    [[nodiscard]] virtual size_t queryColdPageBytes(LayerGroupId layerGroupId) const noexcept = 0;

    // The default implementation returns layerGroupId.
    [[nodiscard]] virtual LayerGroupId getBatchingLayerGroupId(
        LayerGroupId layerGroupId) const noexcept;

    // Applies to both encode and decode.
    [[nodiscard]] virtual PageIndexLocation queryPageIndexLocation(
        LayerGroupId layerGroupId) const noexcept = 0;

    virtual bool encode(LayerGroupId layerGroupId, void* dstBasePtr,
        PageIndexPair const* pageIndices, size_t numBasePages,
        cudaStream_t stream) noexcept = 0;
    virtual bool decode(LayerGroupId layerGroupId, void const* srcBasePtr,
        PageIndexPair const* pageIndices, size_t numBasePages,
        cudaStream_t stream) noexcept = 0;
};
```

`LayerGroupId` is an alias of `LifeCycleId`. It is passed to the size query and both transforms because lifecycles that
share a hot pool group may use different cold formats and cold-page sizes.

`configure()` supplies every hot GPU pool layout and base address in one call. The descriptors form a contiguous
array ordered by `PoolGroupIndex`, and every descriptor contains that same index. KVCM calls it exactly once; returning
`false` rejects the complete configuration.

The methods have these contracts:

- `queryColdPageBytes()` returns a fixed cold-page payload size for a lifecycle. Variable-sized output is not supported.
- `getBatchingLayerGroupId(lifecycle)` returns the smallest lifecycle ID in the same codec-equivalence class. Equal
  IDs promise identical encode/decode behavior, including algorithm, parameters, cold-page size, and encoded
  representation. Membership in the same configured GPU pool group is a necessary additional precondition.
- The default `getBatchingLayerGroupId()` returns its argument, disabling cross-lifecycle batching.
- `getBatchingLayerGroupId()` returns a negative ID on failure or for an unknown lifecycle.
- `queryPageIndexLocation()` explicitly selects the pointer location used by both `encode()` and `decode()`.
  Lifecycles with the same batching ID must return the same location. Compressing codecs normally select `kDevice`; a
  codec that consumes indices synchronously to construct batched DMA descriptors, page-wise `cudaMemcpyAsync` calls,
  or kernel parameters may select `kHost`. It returns `kBadLocation` on failure or for an unknown lifecycle;
  `kBadLocation` is never a valid index-array location.
- KVCM may concatenate index pairs from any subset of lifecycles with the same batching ID and call `encode()` or
  `decode()` once using that representative ID. KVCM does not promise that every member of the class is present.
- Encode and decode enqueue work on the supplied CUDA stream and do not synchronize it.
- A `bool` result reports synchronous validation or submission success, not asynchronous CUDA-work completion.
- For `kHost`, index pointers remain valid only until the codec call returns and asynchronous work must not retain or
  dereference them.
- For `kDevice`, index pointers remain valid until all codec work enqueued on the supplied stream completes. The
  codec must use them only on that stream and must not retain them beyond that work.
- Each `PageIndexPair` contains the destination and source index for one logical copy. The array-of-structures layout is
  shared by host and device codecs. A GPU work item can fetch both indices with one aligned 64-bit load, while a host
  codec can iterate the same array to issue page-wise copies.
- GPU-accessible data base pointers may be used by work enqueued on the supplied stream, but must not be retained beyond
  that work.

Storage-bound KV cache compression reuses this cold-page codec ABI through the
C++ `NativeColdPageCodec` adapter. For the compression-provider lifecycle and
extension interface, see the
[KV Cache Compression Development Guide](kv-cache-compression-development.md).

## Storage Layout and Grouping

`LifeCycleId` is the stable semantic key. Pool groups are physical allocation equivalence classes and are specific to
a storage representation.

```text
LifeCycleId ----> hotPoolGroup(lifecycle):  multi-pool hot layout
            +---> coldPoolGroup(lifecycle): one-pool cold layout
```

The lifecycle-to-pool-group mappings for hot and cold storage are independent. For example, two lifecycles may share a
GPU pool group because their unencoded layouts are identical but map to different cold pool groups because they encode
to different sizes.

Each non-hot pool group contains exactly one physical pool. Lifecycles with compatible fixed cold-page strides may
share a cold pool group. Their encoded contents may differ because the lifecycle passed to `decode()` selects the
correct interpretation.

All cold tiers use the same logical encoded representation and lifecycle-to-cold-pool-group mapping. This permits raw
blob copies between host and disk without decoding and re-encoding. Physical tiers may still apply their own allocation
granularity or padding around the logical payload.

KVCM must therefore maintain level-specific mappings, conceptually:

```cpp
TypedVec<CacheLevel, TypedVec<LifeCycleId, PoolGroupIndex>> lifeCycleToPoolGroup;
```

Memory partitioning is layer-group-based rather than hot-pool-group-based. KVCM first derives one positive, normalized
hot-tier byte-quota weight per public layer group, whether the source is `initial_pool_ratio`, a typical batch, or
constraints. `LayerGroupId` currently maps one-to-one to the internal `LifeCycleId`, so the implementation stores and
computes these weights by lifecycle. Hot initialization sums the lifecycle byte weights that map to each hot pool
group. Cold initialization first converts each hot byte weight to its implied slot-count weight, then converts that
weight to cold bytes using the lifecycle's cold-page size and sums it into the cold pool group. This preserves
layer-group slot-count proportions when hot and cold representations use different page sizes or pool-group mappings.
Runtime sampling remains level-specific and byte-based. The low-level `initial_pool_ratio` and the higher-level
`KvCacheConfig.pool_ratio` therefore contain exactly one hot-tier byte ratio per layer group in layer-group ID order,
not one per hot pool group.
Hot-level constraints remain feasibility floors and may clamp the resulting hot allocation; they are not projected into
cold storage. Cold pool groups use only the structural minimum needed by their allocators.

Migration resolves source and destination groups independently from the page lifecycle and cache level. Pool group
indices must not be compared across representations as though they identify the same physical layout.

At construction, KVCM validates that every batching representative is configured, is no greater than each member,
maps to itself, belongs to the same configured GPU and cold pool groups as every member, and has the same cold-page
size and page-index location. For a hot/cold conversion, the migration key includes the physical source and destination
pool groups plus the batching representative. KVCM then concatenates the already ordered index pairs and makes one
codec call. Hot/hot and cold/cold copies do not use the codec batching ID.

Existing public pool-layout and GPU base-address queries continue to describe only the hot representation. APIs that
take an explicit cache level, such as `getAndResetIterationPeakBlockStats(level)`, return pool-group data in that
level's grouping order; hot and cold results may therefore have different lengths and unrelated group indices.

## Memory Contract

KVCM owns all hot, cold, and staging allocations. Cold storage is not exposed to codec users, and only GPU storage is
otherwise exposed through the existing KVCM API.

KVCM guarantees that every hot or cold data base pointer passed to the codec is GPU-accessible. The current UVM
assumption that mapped pinned-host memory has the same host and GPU address remains unchanged. No public memory
descriptor is required.

KVCM first builds a packed `thread_local std::vector<PageIndexPair>` before acquiring scarce staging resources. For
host indices, KVCM passes this vector directly and the codec consumes it before returning. For device indices, KVCM
acquires device staging and submits one H2D `cuMemcpyBatchAsync` entry with
`CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL`. CUDA must finish every access to the pageable host vector before the API
returns, while the H2D transfer remains asynchronous on the codec stream. KVCM may therefore reuse the host vector
immediately and does not need a pinned-host index ring. The upload also sets
`CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE` to prefer copy engines when the platform honors the hint.
Before CUDA 12.8, where `cuMemcpyBatchAsync` is unavailable, KVCM instead uploads the pageable-host indices with a
version-gated segmented kernel whose parameters are consumed during launch.

The hot-side storage used by the codec must be GPU-accessible, either GPU memory or pinned mapped host memory. Disk
addresses are never passed to the codec. When no host tier exists, KVCM uses temporary pinned-host staging buffers for
GPU-to-disk and disk-to-GPU migrations.

KVCM keeps every GPU-accessible data allocation alive until the work enqueued on the codec stream has completed.

The generalized staging facility is independent of `CopyEngine`. When any cold tier is neither GPU nor host memory,
`StorageManager` uniquely owns a pinned page ring. After querying and validating the cold-page size of every lifecycle,
it sizes that ring to `max(64 MiB, 3 * maxColdPageBytes)`, with checked multiplication. GPU/host-only cold storage does
not allocate a page ring. `CopyEngine` is constructed with a nullable, non-owning pointer and requires the ring only for
two-hop transfers. It is destroyed before the ring during teardown. Final ring destruction synchronizes all
retired-range events before freeing the backing allocation. `StagingBuffer` also remains a non-owning lease and must not
outlive its `StorageManager`.

Staging managers are intentionally not internally thread-safe. Each belongs to one KVCM instance, whose APIs currently
serialize access. If KVCM later supports concurrent public calls, synchronization belongs at the KVCM boundary through
one mutex per instance rather than inside each staging manager.

If any lifecycle requests device indices, its `StorageManager` owns one device index ring, allocated lazily after codec
configuration. The default host-index codec allocates no index staging. CUDA handles any internal host staging required
by `CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL`; KVCM does not retain a pinned index ring. Per-`StorageManager` device rings
prevent codec traffic in one KVCM instance from consuming another instance's index capacity.

Each manager stores its backing allocation as `std::variant<HostMem, CudaUniqPtr>` and derives its memory kind and
base address from the active owner, so host and device ownership cannot diverge. Each manager keeps a byte-accurate
next-fit cursor and a spatially ordered list partition of its backing buffer. A live
range represents the current lease; a retired range retains the completion event of the previous temporal owner of
those bytes. Allocation scans contiguous retired runs from the cursor to the physical end while skipping live ranges,
then makes at most one additional pass from the beginning. The cursor changes only after a successful allocation. If no
run satisfies the minimum request, allocation fails rather than waiting for a live lease that cannot retire on the same
thread. There is no manager-wide fixed allocation granularity.

Each request specifies a positive size granularity and a power-of-two address alignment. The returned size is a
multiple of the size granularity: the minimum bound is rounded up and the maximum available bound is rounded down.
Size granularity need not be a power of two, but it must be a multiple of the address alignment, so that every
granularity-sized slice inside a lease starts at an aligned address. Disk staging uses the cold-page size, while index
staging uses `sizeof(PageIndexPair)`; each chooses its own alignment subject to that divisibility rule. An allocation
carves only its payload bytes into a new live range.
Skipped alignment padding and end-of-ring slack remain retired with their inherited events. If padding crosses multiple
previous ranges, it remains split at those boundaries so every padding fragment keeps the event protecting its bytes.
A stream acquisition waits on all source events without invalidating them, allowing other fragments and streams to wait
on the same events later. A synchronous acquisition synchronizes the source events; their shared state may then be
closed because future consumers no longer require a wait.

The device ring retains enough capacity to overlap H2D index copies and codec execution. If a batch exceeds the 64 KiB
`kMaxIndexBatchBytes` scheduling cap, KVCM splits it into codec chunks; codecs already cannot require a complete
equivalence class in one call.

`StagingBuffer` has one current owner represented by `std::optional<CUstream>`. `nullopt` means synchronous CPU
ownership with no outstanding asynchronous access. Acquiring with `nullopt` host-synchronizes the slice's previous
completion events before returning; acquiring with a stream inserts waits on that stream. `stream()` returns the
optional owner and `setStream()` transfers ownership as follows:

| Old owner | New owner | Transition |
|---|---|---|
| `nullopt` | `nullopt` | No action |
| `nullopt` | stream | No action; preceding CPU access is already complete |
| stream A | stream A | No action |
| stream A | stream B | Record an event on A and make B wait |
| stream | `nullopt` | Host-synchronize the old stream |

On destruction, a stream-owned slice records its reusable event on that stream; a synchronously owned slice becomes
immediately reusable. Callers must finish CPU access before a `nullopt`-to-stream transition and must call
`setStream(stream)` before submitting asynchronous work that accesses a synchronously acquired slice.

For a device-index codec, KVCM acquires a device slice on the codec stream and submits the pageable host array with
`CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL`. The call returns only after CUDA has consumed the host source, so KVCM can
immediately reuse it. The device lease remains alive through the codec call and records its event after the H2D copy
and all codec work enqueued on the stream.

## Migration Paths

The expected paths are:

```text
Hot -> Host: encode into the host cold-page slot
Host -> Hot: decode from the host cold-page slot

Hot -> Disk: encode into temporary pinned staging, then write the blob
Disk -> Hot: read the blob into temporary pinned staging, then decode

Host -> Disk: raw encoded-blob copy
Disk -> Host: raw encoded-blob copy

Hot -> Hot: existing per-pool copy, including defragmentation
Cold -> Cold: raw one-pool copy
```

The default concatenating codec reports `kHost` and consumes the `PageIndexPair` array synchronously. It builds one
`cuMemcpyBatchAsync` descriptor for each page and hot pool, then submits the complete descriptor array once per codec
call. Each descriptor copies the full pool contribution for that page. There is no codec-level 2 MiB work size,
transfer chunking policy, or 16-byte alignment requirement; DMA scheduling and partitioning belong to CUDA.
Before CUDA 12.8, the version-gated fallback submits the same descriptors as individual stream-ordered
`cuMemcpyAsync` calls.

The batch uses stream-ordered source access and sets `CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE` to prefer
copy engines when the platform honors the hint. Non-empty calls require a non-legacy
CUDA stream, as required by `cuMemcpyBatchAsync`. Host memory may be registered as adjacent 2 GiB regions by
`HostMem`; a descriptor crossing such a registration boundary is split only to satisfy that allocation constraint.
This boundary handling is unrelated to page-size or DMA work-unit tuning.

The concrete codec, its configuration logic, and its factory live entirely in `coldPageCodec.cpp`. The class remains
translation-unit-private and needs neither a pImpl nor a CUDA implementation file. Custom codecs remain free to choose
host or device indices and their own implementation, subject to the corresponding pointer-lifetime contract.

### Migration Transaction

Migration preserves the existing event and ownership transaction around every representation-specific transfer:

1. Allocate all destination slots before submitting work.
2. Wait for source-page and destination-slot ready events on the migration stream.
3. Submit the transfer and record one finish event.
4. Attach that event to the source pages and destination slots, then notify migration observers.
5. Transfer slot ownership and release the source slots only after successful submission.

A synchronous submission failure releases newly allocated destination slots and leaves source pages retryable. The
source and destination allocations remain alive until the recorded work completes.

## Construction and Ownership

When `cold_page_codec` is omitted or set to `None`, KVCM creates the default lossless concatenating codec
internally. Codec configuration, validation, and ownership transfer also occur for a GPU-only manager, even though no
migration invokes the codec; this detects invalid explicit codecs consistently and preserves one construction contract.

Normal Python users therefore select the default with:

```python
manager = KVCacheManager(config)
```

Concrete codec classes remain implementation details. Python-defined codec subclasses are not supported, and the
pure-Python KVCM2 backend does not support the codec feature.

KVCM accepts `std::unique_ptr<IKvCacheColdPageCodec>` by value and transfers it directly to `StorageManager`, the
component that executes migrations. Supplying a codec is consumptive as soon as KVCM construction is invoked, whether
construction succeeds or fails. In Python, the relinquished codec wrapper must not be used again.

`create_default_kv_cache_cold_page_codec()` is not needed to select the default. It remains exposed primarily as a
reference for authors binding native compression codecs and as an end-to-end demonstration of the ownership-transfer
contract:

```python
codec = create_default_kv_cache_cold_page_codec()
manager = KVCacheManager(
    config,
    cold_page_codec=codec,
)
```

Construction does not attempt to restore codec ownership on failure. Configuration may already have mutated the codec
before a later validation or allocation fails, so restoring only the pointer would not provide a useful rollback
guarantee. `StorageManager` owns the codec throughout configuration and destroys it automatically if construction
throws.

## Initialization Sequence

The intended initialization order is:

1. Consume the supplied codec at the KVCM call boundary, validate the KVCM configuration, and construct the hot
   storage layout. `StorageManager` creates the default codec when none was supplied.
2. Allocate hot storage so GPU base addresses are available.
3. Configure the codec with the hot pool-group descriptions.
4. Query the fixed cold-page size, batching representative, and page-index location for every lifecycle.
5. Validate all batching equivalence classes, then build the cold lifecycle-to-pool-group mapping from encoded sizes
   and physical stride requirements.
6. Allocate every configured cold tier using its one-pool layout.
7. Retain the configured codec in `StorageManager` for all migrations.

## Required KVCM Refactoring

Existing assumptions that pool count, pool sizes, and pool-group indices are uniform across cache levels must be
removed.
In particular:

- Slot allocation must aggregate lifecycle demand using the mapping for the requested cache level.
- Migration must derive source and destination pool groups separately and batch hot/cold conversions by the codec's
  validated batching representative.
- Eviction controllers must use the lifecycle grouping of their cache level.
- Pool ratios must be derived per layer group and projected through each cache level's internal lifecycle mapping.
- Constraint-derived minimum slots apply only to the hot level, where batches execute; cold levels use structural
  allocator minima.
- Statistics keyed by pool group must not assume that groups at different levels have matching indices or counts.
- Hot/cold migration must use encode/decode, while cold/cold migration remains an opaque blob copy.

## Initial Non-Goals

- Python implementations or subclasses of `IKvCacheColdPageCodec`.
- Support in the pure-Python KVCM2 backend.
- Variable-length encoded pages or per-page size metadata.
- Direct codec access to disk addresses or disk I/O.
- Changing the existing UVM and mapped-pinned-address assumptions.

## Validation

The implementation should cover at least:

- Byte-exact round trips through the default concatenating codec.
- Codec-specific correctness for compressed or lossy implementations.
- Lifecycles sharing a hot pool group but mapping to different cold pool groups.
- Lifecycles from different hot groups sharing a compatible cold pool group.
- GPU-host, GPU-disk staging, host-disk, resize, eviction, and defragmentation paths.
- Correct CUDA event ordering and staging-buffer lifetime.
- Every optional-stream ownership transition, including synchronous acquisition before CPU overwrite.
- Lazy device index-ring allocation, reuse, capacity pressure, and chunking for a device-index codec.
- Default-codec DMA descriptor construction cost and `cuMemcpyBatchAsync` throughput for non-contiguous pages.
- Device-codec AoS index-load behavior and the cost of ephemeral-source H2D submission.
- Layer-group pool-ratio projection across different hot and cold pool-group mappings.
- Hot-only constraint floors and structural cold-tier minima.
- Consumptive Python ownership transfer on both successful and failed construction attempts.
