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

## Layout

- `common.h`, `config.h`, `tokenIdExt.h`, and `exceptions.h`: shared types,
  configuration, token encoding, and error types.
- `blockRadixTree.*`: the shared prefix-reuse tree and SHA-256 block keys.
- `page.*`, `kvCache.*`, and `kvCacheManager.*`: page lifecycle, per-request
  cache state, and the top-level manager.
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
- Preserve strongly typed indices. `LayerGroupId` is a public lifecycle
  semantic, `PoolGroupIndex` is a storage-layout index, `BlockOrdinal` is a
  sequence position, and `SlotId`/page indices address storage. Do not convert
  or compare them implicitly.
- Keep hot-path checks under `TLLM_CHECK_DEBUG*` or `gDebug` when they are too
  expensive for release builds. Do not remove debug invariants simply because
  production does not execute them.
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
3. Lock or migrate required committed pages to GPU before model execution.
4. Commit completed blocks to the tree, making their immutable pages available
   for later requests; `stopCommitting()` finalizes this process.
5. Suspend or close the request, returning pages to holding/eviction ownership.

`LifeCycleRegistry` maps model layers to lifecycle groups. Attention lifecycles
may have sliding-window and sink-token rules; SSM lifecycles represent a
recurrent-state checkpoint. A pool-group index is a storage-layout index and is
not interchangeable with a layer ID or lifecycle ID.

`StorageManager` coordinates GPU, host, and disk cache levels. It allocates
slots, schedules pages for eviction, migrates pages between levels, and resizes
pools. `CopyEngine` performs the actual batched transfers; C++ code calls it
directly and must not round-trip through Python bindings.

The dependency direction is broadly:

```text
types/config/exceptions
  -> lifecycle + memory/CUDA utilities
  -> storage pools + eviction + copy engine + radix tree
  -> pages + storage manager
  -> per-request KvCache
  -> KvCacheManager
  -> nanobind API
```

## State machines

### Per-request cache

- `SUSPENDED`: no active CUDA-stream use; committed pages can be held or
  evicted.
- `ACTIVE`: pages required by the request are locked to GPU and use the cache's
  CUDA stream.
- `CLOSED`: resources are released; further use is invalid.

`commit()` finalizes full blocks. Its `isEnd=true` form is a terminal-memory
contract: later writes to the request's KV memory are invalid, because final
live pages may be moved into the radix tree rather than copied.

`stopCommitting()` is a distinct transition and must not call `commit()`:
doing so would append the same tokens twice. It also releases stale held SWA
pages and performs final commit-state bookkeeping.

### Page status

- `LOCKED`: required on GPU; neither eviction nor dropping is permitted.
- `HELD`: eviction is allowed, but dropping is not.
- `DROPPABLE`: both eviction and dropping are allowed.

The `PageHolder`, `UniqPageLock`, and `SharedPageLock` types implement these
transitions. CUDA ready/finish events are part of their correctness contract:
they establish write completion, migration ordering, and safe reuse across
streams. A stream change for an active cache intentionally synchronizes the
new stream with the old one.

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

- Follow the existing `SharedPtr`/`WeakPtr` conventions in `utils/sharedPtr.h`.
  Treat every ownership edge as intentional; do not replace weak edges with
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

## Correctness invariants

- Block keys are SHA-256 digests over the reuse scope and the token sequence.
  They are a security boundary for cross-request reuse. Do not replace,
  truncate, or use a non-cryptographic hash unless prefix matching also gains a
  token-content equality check.
- `knownNoDigest=true` is an external guarantee (for example `text_only`), not
  a hint derived by scanning tokens. Passing it incorrectly corrupts key hashes.
- The radix tree owns child blocks and a child never outlives its parent. Preserve
  parent/child attachment order when replacing or removing blocks.
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

## Nanobind and concurrency

- C++ public APIs are called under the executor's single-threaded KV-cache
  access model. Do not add mutexes or relax that model without auditing every
  manager, cache, page, and callback path.
- Binding code may release the GIL only while it touches no Python objects. Keep
  Python conversions and callbacks under the GIL, and keep non-owning token
  buffers alive for the entire C++ call.
- Reacquire the GIL before invoking a Python callback, wrapping a C++ result,
  creating an `nb::object`, or changing Python reference counts. If the
  single-threaded access precondition is ever relaxed, concurrency protection
  must be designed for the whole manager rather than added piecemeal.
- Exceptions crossing the binding boundary need an explicit nanobind mapping.
  Preserve Python exception type and attributes when adding or changing a C++
  exception.

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
- Python behavior and backend-parity tests are in
  `tests/unittest/kv_cache_manager_v2_tests/`. During development, prefer the
  fast path below: set `PYTHONPATH` to `tensorrt_llm/runtime/` and execute the
  test file directly with `python`. Do not use `pytest` for this fast path; the
  file's test runner avoids importing the full `tensorrt_llm` package.

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
