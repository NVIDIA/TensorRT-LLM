<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCacheManagerV2 C++ Guide

This directory contains the C++ implementation of KVCacheManagerV2: allocation,
prefix reuse, eviction, GPU/host/disk movement, page locking, and KV-cache event
generation. Python accesses it through the nanobind bindings in
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp`.

This guide is self-contained. Do not make changes here depend on the Python
KVCacheManagerV2 implementation or its migration documents; those files are
temporary and will be removed after the migration.

The current C++ headers and tests are the source of truth. Historical Python
behavior remains useful for compatibility testing, but migration-era proposed
layouts or ownership models must not override the implementation.

`SPARSE_ATTENTION_DESIGN.md` covers how a sparse buffer's pages are held in host
memory and read through a per-request GPU cache, and `INDEX_CONVERSION_DESIGN.md`
how page indices become the compact arena indices attention kernels take. Read
them before changing `batch.*`, `sparseCache.*` or the index conversion.

## Layout

- `common.h`, `config.h`, `tokenIdExt.h`, and `exceptions.h`: shared types,
  configuration, token encoding, and error types.
- `blockRadixTree.*`: the shared prefix-reuse tree and SHA-256 block keys.
- `page.*`, `kvCache.*`, and `kvCacheManager.*`: page lifecycle, per-request
  cache state, and the top-level manager.
- `batch.*`: the caches stepped together and their device-side mirror — row
  ownership, the per-layer-group page tables a sparse fetch reads, and the
  per-row watermarks a step derives its counts from.
- `sparseCache.*`, `sparseCacheKernels.*`, and `indexConversionKernels.*`: the
  per-request GPU cache over host-backed sparse pages, and the conversion from
  page indices to the compact arena indices attention kernels take.
- `storage/`, `storageManager.*`, `evictionController.*`, and `copyEngine.*`:
  pools, eviction ownership, migration, and data movement.
- `lifeCycleRegistry.*`: layer-group/lifecycle mapping, including attention and
  SSM behavior.
- `eventManager.*` and `eventSink.h`: KV-cache event derivation and delivery.
- `utils/`: typed indices, ownership helpers, CUDA events, host memory, and
  math utilities.

## Design principles

- Prefer composition when it expresses ownership or containment. Use
  inheritance only for a real IS-A relationship or required virtual dispatch,
  such as `CommittedPage : Page` and `EventManager : EventSink`.
- Preserve strongly typed indices. `CacheGroupId` (and legacy `LayerGroupId`) is a public lifecycle
  semantic, `PoolGroupIndex` is a storage-layout index, `BlockOrdinal` is a
  sequence position, and `SlotId`/page indices address storage. Do not convert
  or compare them implicitly.
- Keep hot-path checks under `TLLM_CHECK_DEBUG*` or `gDebug` when they are too
  expensive for release builds. Do not remove debug invariants simply because
  production does not execute them.
- Check with `TLLM_CHECK*` rather than a hand-rolled `if (bad) { throw ...; }`.
  It is the convention across the project, and it keeps a condition and the
  message explaining it in one place that reads as a check.
  The exception is a failure whose *type* is the contract. The bindings make a
  Python class of each one in `exceptions.h`, so callers -- and the poison latch
  -- act on which class they caught; throw those directly:
  `OutOfMemoryError` and its `Host`/`Disk`/`Cu` forms, `LogicError`,
  `AssertionError`, `CuError`, `CorruptedError`, `ResourceBusyError`,
  `OutOfPagesError`.
  Validation at the binding boundary that throws `std::invalid_argument` or
  `std::out_of_range` is legacy and should migrate to `TLLM_CHECK*` as it is
  touched, so that KVCM2 ends up on the same footing as the rest of the repo.
  Migrate deliberately rather than in passing: nanobind maps those to
  `ValueError` and `IndexError` where `TLLM_CHECK*` raises `RuntimeError`, and
  the suites assert the first two in well over a hundred places, so each
  conversion carries its own test updates and is a Python API change under the
  nanobind rule on preserving exception types.
- Keep the core independent of Python. C++ storage, copying, CUDA, hashing, and
  lifecycle operations call C++ APIs directly; Python adaptation belongs in
  nanobind.

## Architecture and request flow

`KvCacheManager` owns the global services: a `LifeCycleRegistry`,
`BlockRadixTree`, and `StorageManager`. It creates a `KvCache` per request.
The request cache starts `SUSPENDED`, becomes `ACTIVE` through `resume()`, and
must eventually be `close()`d. Its normal flow is:

1. Match the input `TokenSpan` against `BlockRadixTree` within a `ReuseScope`.
2. Allocate request-local pages for unmatched blocks through `StorageManager`.
3. Lock or restore required pages before model execution. Every page locks at the
   hot level; a sparse pool's hot allocation is host memory, which is a property
   of that pool rather than of the level or of the group holding it.
4. Commit completed blocks to the tree, making their immutable pages available
   for later requests; `stopCommitting()` finalizes this process.
5. Suspend or close the request, returning pages to holding/eviction ownership.

`LifeCycleRegistry` maps layer buffers to lifecycle groups. Attention lifecycles
may have sliding-window and sink-token rules; SSM lifecycles represent a
recurrent-state checkpoint. Both layer configs expose `reusePolicy`: `REQUIRED`
constrains the prefix match, `OPTIONAL` offers state at the required endpoint for
explicit selection on first resume, and `PRIVATE` never publishes reusable pages.
A buffer may override its layer's default through `BufferConfig::reusePolicy`.
This applies to attention and SSM buffers, so one model layer may span several
lifecycle groups. Use `getBufferGroupId(layerId, role)` and `bufferGrouping()`
(`get_buffer_group_id` and `buffer_grouping` in Python) for exact membership.
These APIs return or index by `CacheGroupId`, a public alias of `LifeCycleId`.
`LayerGroupId` remains a lifecycle alias for legacy APIs.
The legacy `getLayerGroupId(layerId)` and `layerGrouping()` are deprecated in
documentation only: uniform layers retain their results, while a split layer
fails a `TLLM_CHECK_WITH_INFO` (for the queried layer or the entire partition,
respectively), exposed as `RuntimeError` in Python. No deprecation warnings are emitted.

Page conversion reads `BufferAttr::scratchPagesPerBlock`, counting only the
layer's buffers in that coalesced slot. Scratch capacity uses the maximum exact
layer share across pools within each lifecycle. Layer counts are temporary
layout calculations, not persistent `(layer, lifecycle)` attributes.

Execution backends that share a page or state index across coupled buffers must
validate that they use the same policy; independent policies require independent
index tables. In particular, the default attention view couples K/V and scales,
and Mamba kernels couple recurrent, convolution, PLE, and replay state. Native
KVCM supports independent policies for callers that provide those indices.
The sparse fetch path configured by `sparseCache` still uses one group per layer;
its setup uses the checked layer lookup and rejects split layers.

SSM lifecycles are deduplicated by policy, so a manager can have up to three SSM
groups; iterate `ssmLifeCycleIds()` for working state and use
`requiredSsmLifeCycleId()` only for prefix pruning.
Optional SSM checkpoints require an exact endpoint match, including partial
blocks. Fresh or unselected recurrent state must be initialized or reconstructed
by the caller before inference. A pool-group index is a storage-layout index and is
not interchangeable with a layer ID or lifecycle ID.

Sparsity is a property of a pool, not of a lifecycle: a layer's buffers may be
partly sparse, and the pool each lands in carries the flag. Sparse buffers are
invalid for SSM layers. Sparse and dense pools stay separate even when their slot
sizes match, which is what lets a sparse pool be backed by host memory while a
dense pool of the same group and level is device memory; the group reports
GPU_MEM either way. Compatible sparse buffers still coalesce. Sparse buffers place no requirement on the cold tiers.

`StorageManager` coordinates GPU, host, and disk cache levels. It allocates
slots, schedules pages for eviction, migrates pages between levels, and resizes
pools. `CopyEngine` performs the actual batched transfers; C++ code calls it
directly and must not round-trip through Python bindings.

`Batch` groups the `KvCache`s stepped together and owns their row order. Rows are
slots rather than positions in a compacted list, so removing a member leaves a
hole instead of shifting later rows and the shapes and addresses a captured graph
baked in stay valid. Membership is exclusive and carries the sparse arena lease,
so a request joins only once it is ready to be stepped; `move()` relocates a
member between rows without ending membership, which `removeKvCache()` followed
by `addKvCache()` would. A member may not be resumed, suspended or closed while
seated, because those write page indices through the Batch's staged row.

`publish()` uploads a step's page tables and watermarks from pinned staging and
returns with the copy still in flight, so every later host write to a member's
page indices — a `KvCache::resize()`, a commit, a beam fork, an SWA recycle,
anything that locks or unlocks a page — must be separated from it by
`waitPublished()`, or it races the copy and publishes a torn table. The Batch's
own mutators are exempt: `resize()`, `move()`, `addKvCache()` and
`removeKvCache()` each wait for themselves. `publish()` throws rather than waits
when the previous publish has not been waited on, because waiting there would
cover only its own write and hide a member already mutated into the hazard; the
lock and unlock paths carry a debug check for the member side of the same rule.

A sparse member writes back the units its history passes through
`flushSparseUnits()`, and must have done so for the history it is giving up
before it leaves, since removal returns the arena lease and discards whatever the
GPU cache held.

The dependency direction is broadly:

```text
types/config/exceptions
  -> lifecycle + memory/CUDA utilities
  -> storage pools + eviction + copy engine + radix tree
  -> pages + storage manager
  -> per-request KvCache
  -> Batch + sparse cache
  -> KvCacheManager
  -> nanobind API
```

## State machines

### Per-request cache

- `SUSPENDED`: no active CUDA-stream use; committed pages can be held or
  evicted.
- `ACTIVE`: pages required by the request are locked at their intended storage
  levels and use the cache's CUDA stream.
- `CLOSED`: resources are released; further use is invalid.

`commit()` finalizes full blocks. Its `isEnd=true` form is a terminal-memory
contract: later writes to the request's KV memory are invalid, because final
live pages may be moved into the radix tree rather than copied.

`stopCommitting()` is a distinct transition and must not call `commit()`:
doing so would append the same tokens twice. It also releases stale held SWA
pages and performs final commit-state bookkeeping.

### Page status

- `LOCKED`: pinned at the current storage level; neither eviction nor dropping
  is permitted. Every lifecycle locks at the hot level, sparse or not. A sparse
  lifecycle's hot pool group is host memory, but that is a property of the
  allocation rather than of the level, so locking does not distinguish them.
- `HELD`: eviction is allowed, but dropping is not.
- `DROPPABLE`: both eviction and dropping are allowed.

The `PageHolder`, `UniqPageLock`, and `SharedPageLock` types implement these
transitions. CUDA ready/finish events are part of their correctness contract:
they establish write completion, migration ordering, and safe reuse across
streams. A stream change for an active cache intentionally synchronizes the
new stream with the old one.

`batchedLockPages()` restores cold pages to the hot level before locking them,
deduplicating pages shared by multiple owners. The destination is always the hot
level: an earlier revision carried one per page, for sparse lifecycles that
resided at level 1, and that is gone along with the residency split. A partial
prefix copies from the source's actual tier into a private hot page without
moving a shared source. `ScratchSlotLock` remains GPU-only. This does not itself
demote history; offload requires a separate transfer and slot ownership handoff.

## Ownership and lifetime

The high-level ownership shape is:

```text
KvCacheManager
|- LifeCycleRegistry (value)
|- StorageManager (shared)
|- BlockRadixTree (shared)
`- living KvCache registry (non-owning pointers)

KvCache
|- KvCacheManager (shared; cache keeps manager alive)
`- per-beam/per-block page holders and locks

BlockRadixTree
`- roots -> child Blocks (strong ownership through next maps)
              `- lifecycle page entries (raw observer links)

Eviction controller
`- prioritized LRU lists (strong ownership of droppable Pages)
```

- Avoid shared ownership unless it is needed; prefer values, `std::unique_ptr`,
  or documented raw observers. When it is needed, prefer the non-atomic
  `SharedPtr`/`WeakPtr` in `utils/sharedPtr.h` over `std::shared_ptr`: atomic
  refcounts are costly on ARM (Grace). `SharedPtr` is fine across threads when
  the manager's API lock serializes every copy and release. Reserve
  `std::shared_ptr` for process-wide resources or references released outside
  that lock (for example, from Python). This overrides the generic `shared_ptr`
  advice in `CODING_GUIDELINES.md`.
- Treat every ownership edge as intentional; do not replace weak edges with
  strong ones merely to simplify access.
- `KvCache` keeps its `KvCacheManager` alive. The manager's registry of living
  caches must not create the reverse strong-reference cycle.
- A committed page is referenced by the radix tree without making the tree its
  permanent owner. Eviction queues may be the only strong owner of a droppable
  page, so never store a raw pointer past the operation that obtained it.
- The eviction queue stores strong page ownership and its `NodeRef` is valid
  only for the eviction policy that created it. Exclude a page from eviction
  before moving it to another cache level; only then schedule it in the new
  level's policy.
- Destructors can trigger tree detachment, page unlinking, or eviction updates.
  Keep teardown order explicit, make cleanup idempotent where needed, and audit
  re-entrancy before changing a destructor or `close`/`shutdown` path.
- `CommittedPage::numTokensInBlock` can be smaller than its block's token span.
  For attention it describes a reusable prefix; for SSM it is an exact state
  checkpoint. Do not assume every page covers its whole block.
- `Block::prev`, `Block::storage`, `CommittedPage::block`, and page manager
  pointers are observer/back-reference links with lifetime invariants, not
  ownership. Explicit unlinking and teardown order keep them valid.
- Orphan blocks may retain pages while a live `KvCache` still references them;
  `Block::~Block()` reclaims those pages when the last block owner releases them.
  Every `KvCache` must be closed before manager shutdown so this deferred cleanup
  runs while `StorageManager` is still alive.

### Beam-search pages

Prefill normally runs at beam width 1. Expanding the beam width does not
physically duplicate the whole prompt; ownership splits at the prompt tail:

- `_appendBeams` starts at block `floor(expectedPromptLength / tokensPerBlock)`.
  Full prompt-prefix blocks before that ordinal stay on beam 0, and other beams
  read that shared prefix through attention cache indirection.
- The partially filled prompt tail, if any, gets a private writable page per
  beam, copied from beam 0. Preallocated uncommitted generation blocks are
  handled the same way. Newly allocated decode blocks have one row per beam.
  Already committed blocks remain canonical on beam 0, even when the
  prompt-length hint is conservative.
- Raw page-index rows of non-leading beams contain `BAD_PAGE_INDEX` for the
  shared prefix. `getAggregatedPageIndices()` and `Batch` publication expose the
  actual storage rows. Cache indirection resolves shared attention history; do
  not duplicate beam-0 pages into every raw row.
- `getPagePriorities(layerGroupId, beamIndex)` preserves block ordinals. For a
  block with one shared row it reads beam 0; otherwise it reads the requested
  beam's private row. A missing lifecycle page yields `kPriorityDefault`.
- Commit canonicalizes a reusable block to a single beam-0 row and releases the
  alternative beam rows.
- The wrapper disables `enablePartialCommit` when `maxBeamWidth > 1`, and
  changing the beam width while partial commit is enabled is rejected.
  Full-block reuse remains enabled. Partial reuse is also allowed: on resume,
  a matched partial prefix is copied into a private uncommitted page before
  beams expand.
- SSM current state lives separately in `mSsmBlocks` and is copied per beam on
  expansion. It is per-beam writable state, not a shared attention prefix.

## Correctness invariants

- `ReuseScope::salt` is an optional string used directly by scope equality and
  root/block-key hashing. Hash its complete bytes, including embedded NULs, and
  preserve the distinction between an absent salt and an empty string. Integer
  conversion is confined to the V1 event hash format and must agree with the router.
- `Block::root` is a non-owning back-pointer for reading scope metadata while
  attached. Set it from the parent on construction and attachment; clear it with
  `prev` on detachment. Stored events copy the root's salt while it is valid.
- Block keys are SHA-256 digests over the reuse scope and the token sequence.
  They are a security boundary for cross-request reuse. Do not replace,
  truncate, or use a non-cryptographic hash unless prefix matching also gains a
  token-content equality check.
- `knownNoDigest=true` is an external guarantee (for example `text_only`), not
  a hint derived by scanning tokens. Passing it incorrectly corrupts key hashes.
- The radix tree owns attached child blocks. Detach subtrees leaf-first; live
  caches may retain orphan blocks after their former root is destroyed. Restore
  `prev` and `root` when reattaching them, before publishing stored events.
- Root removal is deferred through `proposeToEraseEmptyRoot()` and drained only
  at tree safe points. Do not erase roots directly from a destructor chain.
- Pages are immutable after commit. Locking and CUDA events establish when their
  data is safe to read or migrate; preserve those synchronization boundaries.
- Event payloads describe complete blocks. A lifecycle with partial page coverage
  must not emit an event for the full block.
- Hash bytes and token encoding must stay compatible across all callers. Normal
  token IDs use the little-endian `TokenIdExt` representation; digest tokens and
  `ReuseScope` values participate in the same chained block-key protocol.
- Partial attention coverage uses a `>=` prefix check. SSM coverage names an
  exact recurrent-state checkpoint and must be truncated to that boundary.
- Multi-beam block arrays, sliding-window stale ranges, sink tokens, partial
  block reuse, SSM snapshots, intra-batch rebasing, and rollback after OOM are
  coupled inside `KvCache`; changes there require broad invariant testing.

## Storage, eviction, and memory

- Every cache level has storage and a per-level eviction controller; each pool
  group has a priority-sorted set of LRU queues. Lower priority is evicted first,
  then least-recently-used within that priority.
- `NodeRef` is a stable `std::list` iterator, but only within the exact
  `LRUEvictionPolicy` that issued it. Iterators from different lists are not
  comparable or interchangeable.
- Eviction failure must preserve queue consistency. If a multi-pool eviction
  cannot satisfy all requested slots, restore pages already removed before
  propagating `OutOfPagesError`.
- Host and disk memory code directly uses `mmap`, `munmap`, `mremap`,
  `madvise`, `posix_fallocate`, and CUDA host registration. Preserve cleanup on
  partial failure, CUDA unregister/register ordering across resize, and the
  distinction between host and disk OOM errors.
- CUDA virtual memory and copy operations use the driver/runtime APIs directly.
  Keep allocation granularity, stream ordering, and source/destination lifetime
  valid through asynchronous copies.

## Testing a change here

The Python suite (`tests/unittest/kv_cache_manager_v2_tests/`) does **not** build this
package's C++ tests, so a green Python run says nothing about whether they still compile:
it loads a prebuilt `.so` and never sees the headers. Removing a symbol the C++ tests use
therefore breaks them silently. Build and run them too:

```bash
cd cpp/build
for t in kvCacheManagerV2ConfigTest kvCacheManagerV2ColdPageTest \
         kvCacheManagerV2ColdPageCopyTest kvCacheManagerV2TypedIndexTest \
         kvCacheManagerV2HostMemTest kvCacheManagerV2StatsTest \
         kvCacheManagerV2ConcurrencyTest kvCacheManagerV2DigestPoolTest; do
  cmake --build . --target "$t" -j"$(nproc)" || echo "$t: BUILD FAIL"
done
```

## Interfaces, bindings, and build

- `TokenSpan` is non-owning. The caller retains its backing storage for the full
  call; the manager reads tokens but never stores the span itself. This enables
  the int32 zero-copy matching path.
- Keep C++ implementation sources co-located here and add every compiled source
  to this directory's `CMakeLists.txt`. The parent target consumes its source
  list; do not add a separate shared library for this subsystem.
- SHA-256 support is vendored under `cpp/tensorrt_llm/common/sha256`
  and configured by this directory's `CMakeLists.txt`. Preserve the
  architecture-specific SHA extension flags when changing the hash integration.
  Do not add OpenSSL/libcrypto merely for block hashing; avoiding that dependency
  is intentional for wheel portability.
- Nanobind bindings belong in
  `cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp`. Keep the
  public Python surface compatible with the runtime package; use the
  introspection API only for white-box tests and diagnostics.

## Concurrency model

`KvCacheManager` owns a single reader-writer lock (`ReentrantSharedMutex`,
`utils/reentrantSharedMutex.h`) guarding every piece of mutable state reachable
from it: the radix tree, the storage manager, and the living-`KvCache` set.
Mutating APIs take it exclusively; read-only queries take it shared, so several
`probeReuse()` lookups run concurrently while a writer is excluded.

**What the lock does and does not cover.** It protects state *shared between*
`KvCache`s, not the fields of an individual one, and it is a guarantee about the
bound (Python-visible) API only -- see "Where the guarantee applies" below.

A `KvCache` belongs to one request and is driven by its owning thread; calling
methods on the *same* `KvCache` from two threads concurrently is not supported,
and the lock does not make it safe.

**Rule when adding or editing an API:** a method needs the lock iff it reaches
the manager, the storage manager, or the radix tree.

- Purely per-object accessors (`stopCommitting`, `updateBasePageIndex`, ...) do
  not take it.
- A wrapper whose only shared-state access is a call to an already-locking API
  (`setCapacity` -> `resize`, or the cached-token attribution drop methods ->
  `markStatsDirty`/`clearStatsDirty`) does not take it either.
- Internal statistics accumulators (`commitStats`, `commitSsmSnapshotIterationStats`,
  `commitReusedBlocksByLevel`, `commitConnectorReusedBlocks`, `commitCachedTokensByLevel`,
  `commitConnectorCachedTokens`, and `recordDiskPrefetchBlocks`)
  use the caller's exclusive lock. `commitPendingStats()` and `prefetch()` provide it;
  direct C++ callers, including tests, must do the same.
- `std::shared_mutex` is not recursive, but public APIs call each other freely
  here. `ReentrantSharedMutex` compares the owning thread before it looks at the
  requested mode, so on a thread that already holds the lock **exclusively**
  both a nested exclusive and a nested shared acquisition are no-ops. Internal
  call sites therefore need no annotation: `adjust()` holds it exclusively and
  calls `getQuota()`, which asks for it shared.
  What does not work is nesting inside a *shared* acquisition -- either a second
  shared one, which `std::shared_mutex` leaves undefined, or an upgrade to
  exclusive, which deadlocks. Neither is checked -- a violation is a hang, not a
  diagnostic -- so both have to be respected by construction. See the
  "deliberate non-features" list in `utils/reentrantSharedMutex.h`.

**Where the guarantee applies.** The thread-safety contract covers the API
surface exposed through nanobind and declared in the `.pyi` stubs. Internal C++
entry points carry no guarantee: a C++ caller is inside the implementation and
is responsible for the lock discipline itself.

That distinction matters because `blocks()`, `storage()` and `radixTree()` hand
out references to shared structures, which locking cannot make safe -- the
caller uses the reference after the lock drops. None of them is bound, so they
are internal-only and out of scope rather than holes in the contract. Do not
expose them without solving that first.

Two exposed accessors look like they might leak shared state and do not:

- `manager` returns the `KvCacheManager` itself, which is the lock owner. A
  method on it locks iff it touches the radix tree, the storage manager's mutable
  state, the eviction lists, or the stats aggregates. Accessors that read only
  state fixed at construction are lock-free: `num_layers`, `layer_ids`,
  `tokens_per_block`, `cache_tier_list`, `all_buffer_ids`, `get_layer_group_id`,
  `get_life_cycle_pool_group_indices`,
  `get_page_stride`, `get_page_index_scale`, `get_page_index_converter`,
  `get_mem_pool_base_address`, `supports_index_mode`, `init_config`, and the
  config flags. `get_page_index_upper_bound` resembles that group but reads a
  pool's current `numSlots`, which `_adjustLevel()` changes, so it takes the
  shared lock. Anything added here that reads resize-mutable state must too.
- `get_base_page_indices` returns a zero-copy numpy view, but into
  `mBasePageIndices`, which is per-`KvCache` (an internal vector or a
  caller-owned buffer). Consistent with one-owner-thread; not shared state. Its
  only shared access is the `gDebug` cross-check calling
  `getAggregatedPageIndices()`, which does take the shared lock because it reads
  each `Page`'s slot id, and slot assignment is what migration changes.

**Granularity is deliberate.** Per-block locks would not help: the hazards are
structural mutation of the radix tree's `unordered_map`s (insert/erase/rehash)
and the *non-atomic* refcounts in `utils/sharedPtr.h`, neither of which a
per-node lock protects. Eviction is global (LRU across the whole tree), so a
writer cannot be scoped to a subtree.

**Event retirement sits outside the lock.** `CachedCudaEvent` returns its
`CUevent` to the process-wide `CudaEventPool` from places that hold different
locks or none: `scrubEvents()` under the exclusive lock, and the destructor on
whichever thread drops the last copy. The pool is shared by every
`KvCacheManager`, so no manager's lock could cover it anyway.

Three properties make that sound, and a change that breaks any one of them
reintroduces a hazard the type system will not catch:

- `PooledEvent` holds the handle in an atomic and retires with an exchange, so
  concurrent retirers race and exactly one returns the handle. A plain reset
  would let two callers put the same `CUevent`, and the pool would hand one
  event to two owners.
- *While you hold a copy*, any retirement you can observe came from
  `queryComplete()`, `synchronize()` or `close()`, and each of those runs only
  after the work has completed -- so a copy that finds the handle gone skips a
  dependency that was already satisfied, never a live one. Holding the copy is
  what rules out the destructor, which retires unconditionally: the last drop
  returns the handle even with work still in flight. That is why a raw `CUevent`
  must not outlive the `CachedCudaEvent` it came from. Once the handle is back
  in the pool another owner can re-record it, and a stale waiter would then wait
  on that owner's work, which can finish earlier -- a missed wait, not a
  spurious one.
- `CudaEventPool` is unbounded and FIFO (see its declaration). Unbounded so a
  handle already copied out of a `CachedCudaEvent` is never destroyed under its
  user; FIFO so the reuse distance keeps a copied-out handle from being
  re-recorded by another owner mid-call.

Two rules follow for code that touches events:

- Load the handle once per call and operate on the loaded value. Testing the
  handle and then re-reading it can observe a concurrent retirement in between
  and pass `nullptr` to the driver.
- A raw `CUevent` that has left a `CachedCudaEvent` -- `handle()`,
  `streamWaitEvents()`, `synchronizeAll()`, `mergeEvents()` -- is valid only
  while the caller holds the **exclusive** lock, which excludes the shared-lock
  lookups that would otherwise retire it. Do not copy handles out on a
  shared-lock path.
- Copies may be used concurrently; a single `CachedCudaEvent` **object** may
  not. Two mechanisms cover two pieces of state, and confusing them is the easy
  mistake here: **the atomic protects the payload, the lock protects the
  object.** `PooledEvent`'s handle is an atomic, safe for any number of
  concurrent callers; the `shared_ptr` that names it -- and the `std::optional`
  that may wrap it, as in `KvCache::mFinishEvent` -- is ordinary memory and
  needs the API lock.

  So for a `CachedCudaEvent` held as a member:

  | Operation on the member | Lock |
  |---|---|
  | assign, move-assign, `emplace`, `reset`, destroy | **exclusive** |
  | copy it out, `handle()`, `isClosed()` | **shared** |
  | `queryComplete()`, `synchronize()`, `close()`, `waitInStream()` | **shared** |

  The third row is the counter-intuitive one. Those methods retire the handle,
  which looks like a write, but they mutate only the atomic payload and never
  `mEvent` -- which is why they are `const`. For locking purposes they are
  readers.

  `Slot::readyEvent` and `Page::readyEvent` are assigned in about ten places;
  each is a first-row operation. An *unlocked* reader races those writers no
  matter how carefully the writers lock: that is why `finish_event` is no longer
  bound, since the binding copied out of a live `KvCache` with no lock at all.

  Two riders. Once a copy exists it is independent and needs no lock -- that is
  the point of the shared payload, and what makes handing a copy to another
  thread viable. But a raw `CUevent` taken from it is not a copy: because
  `queryComplete()` is a shared-lock operation that can retire, a handle read
  under the shared lock can be reissued by a concurrent reader, so handles stay
  exclusive-only as above.

**Storage statistics use the manager API.** `get_storage_statistics(cache_level)`
samples all pools in the requested level under the manager's shared lock and
returns independent value copies. The binding releases the GIL before acquiring
the lock. `get_life_cycle_pool_group_indices(cache_level)` returns a copied
lifecycle-to-pool mapping in that level's numbering without locking: the mapping
and its level bounds are fixed at construction.
Production code and tests must use these manager APIs for storage statistics and
lifecycle-to-pool mappings. To query one lifecycle, index the returned mapping.

**Introspection is out of scope.** The `_introspection` submodule (`storage_utilization`,
`set_target_ratio_list_gpu`, `reuse_match_pages`, test block/codec helpers, ...)
reaches private members directly, by design, and takes no locks. It is
test/white-box only and is **not thread-safe**. Do not call it concurrently with
anything, and do not treat it as part of the contract above.

### Using KVCM2 from more than one thread

The supported pattern is to build a cache on a helper thread and hand it to the
consumer, e.g. `create_kv_cache()` -> `resume()` -> `resize()` in the background,
then transfer ownership. Three requirements:

1. **The handoff needs a happens-before edge** (a queue, or joining the thread).
   `SharedPtr` refcounts are non-atomic, so sequential access is only safe with
   real synchronisation between the threads.
2. **Adopt the object with the `cuda_stream` setter, do not block.**
   `setCudaStream()` records an event on the old stream and makes the new one
   wait on it -- a GPU-side dependency. Calling `synchronize()` instead stalls
   the helper until the offload completes, serialising the work that was moved
   off the critical path in the first place.
3. **The helper thread needs its own CUDA context** (`torch.cuda.set_device`
   plus any allocation), otherwise the first KVCM2 call fails with
   `CUDA_ERROR_INVALID_CONTEXT`.

Writer hold time is the latency cost of the coarse lock: a `resize()` that
evicts holds the exclusive lock for milliseconds, and a concurrent reader waits
that long. If low-latency reads matter more than that, chunk the writer or give
readers a separate index rather than making the lock finer-grained.

## Nanobind

- Binding code may release the GIL only while it touches no Python objects. Keep
  Python conversions and callbacks under the GIL, and keep non-owning token
  buffers alive for the entire C++ call.
- Reacquire the GIL before invoking a Python callback, wrapping a C++ result,
  creating an `nb::object`, or changing Python reference counts.
- **Every binding that can block on the API lock must release the GIL first.**
  This is a correctness requirement, not a throughput one. `KvCacheManager` calls
  back into Python while holding the lock (the priority callback and the event
  sink both do `nb::gil_scoped_acquire`), so a thread that waits for the lock
  *while holding the GIL* deadlocks against a lock holder that is waiting for the
  GIL. A binding that blocks with the GIL held also stops every other Python
  thread for the duration of a `resize`, which is the opposite of the point.
- Prefer `nb::call_guard<nb::gil_scoped_release>()` and bind the C++ method
  directly. The guard wraps only the registered function: nanobind converts the
  arguments before constructing it and the return value after destroying it, so
  both conversions keep the GIL. A wrapper lambda that exists only to call
  `nb::cast()` on the result is doing by hand what nanobind already does outside
  the guard -- drop the lambda instead of working around it.
- Use an inner `nb::gil_scoped_release`, scoped to the blocking call alone, only
  when the lambda itself must do interpreter work: building a container
  element-by-element (`castIterationStatsByLifeCycle`, `castPeakBlockStats`,
  `castRequestIds`) or consuming an `nb::object` argument before the call
  (`get_aggregated_pages`). `nb::call_guard` would run that work GIL-free.
  Keep the scope tight, and never let an `nb::object` be destroyed inside it --
  that is a GIL-free decref.
- Exceptions crossing the binding boundary need an explicit nanobind mapping.
  Preserve Python exception type and attributes when adding or changing a C++
  exception.
- Changing a class layout in a header requires rebuilding **everything that
  compiles against it**: `tensorrt_llm`, `bindings`, and the `kvCacheManagerV2*`
  gtest binaries. A stale consumer constructs the object with the old size,
  which surfaces as heap corruption (`free(): invalid size`, or an abort inside
  a destructor) rather than an obvious ABI error -- and the corruption appears
  in whatever code runs next, not at the mismatch.

## High-risk changes

Use extra review and tests for changes involving:

- destructor, `shutdown()`, `close()`, tree detachment, or page unlink order;
- eviction ownership, `NodeRef`, migration, pool resizing, or OOM rollback;
- block hashes, token encoding, salts/LoRA reuse scopes, or `knownNoDigest`;
- partial attention coverage, SSM checkpoints, SWA windows/sinks, or final
  snapshots;
- CUDA stream/event lifetime or host-memory registration;
- `KvCache` commit state, beam forks, reuse rebasing, or page-index buffers;
- nanobind GIL release, callbacks, non-owning buffers, or exception translation.

## Development and tests

- Focused C++ unit tests are in `cpp/tests/unit_tests/batch_manager/`, notably
  `radixBlockTreeTest.cpp`, `kvCacheManagerTest.cpp`,
  `kvCacheManagerV2DigestPoolTest.cpp`, `kvCacheManagerV2HostMemTest.cpp`,
  `kvCacheManagerV2StatsTest.cpp`, and `kvCacheManagerV2TypedIndexTest.cpp`.
- Python behavior tests are in `tests/unittest/kv_cache_manager_v2_tests/`, and
  drive this C++ implementation through the nanobind bindings. During
  development, prefer the fast path below: set `PYTHONPATH` to
  `tensorrt_llm/runtime/` and execute the test file directly with `python`. Do
  not use `pytest` for this fast path; the file's test runner avoids importing
  the full `tensorrt_llm` package.

  ```bash
  REPO_ROOT="$(git rev-parse --show-toplevel)"
  PYTHONPATH="$REPO_ROOT/tensorrt_llm/runtime/" \
      python "$REPO_ROOT/tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py" -v
  ```

- Run one test class or method by passing its unittest name:

  ```bash
  REPO_ROOT="$(git rev-parse --show-toplevel)"
  PYTHONPATH="$REPO_ROOT/tensorrt_llm/runtime/" \
      python "$REPO_ROOT/tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py" \
      TestNoBatching.test_basic -v
  ```

- Before final validation, also exercise the production import path:

  ```bash
  REPO_ROOT="$(git rev-parse --show-toplevel)"
  PYTHONPATH="$REPO_ROOT/" \
      python "$REPO_ROOT/tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py" -v
  ```

- Run `test_kv_cache_event_manager.py` after event changes,
  `test_kv_cache_salting.py` after hashing/reuse-scope changes, and the stats
  tests after allocation or event-accounting changes. Run both available
  backends for changes shared by the C++ and Python surfaces.
- Run focused tests when possible, then the affected KVCacheManagerV2 Python
  suite on both available backends. Use `TLLM_DEBUG_MODE=1` when diagnosing an
  invariant failure.
- Follow the repository C++ standards in `CODING_GUIDELINES.md`: Allman braces,
  east-const, explicit ownership, and clang-format. Do not make unrelated
  formatting changes in this high-churn subsystem.
