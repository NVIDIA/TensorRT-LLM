# KVCacheManagerV2 C++ Migration — Open Issues

This file tracks design decisions that were made without explicit guidance.
Review each item before closing the migration.

---

## Issue 1: BLAKE3 vs SHA-256 for block keys

**Decision:** BLAKE3 is used in C++ (replacing Python's `hashlib.sha256`).

**Impact:** The 32-byte `BlockKey` values will differ between the Python and C++
implementations for the same token sequence. Python and C++ cannot share a warm
radix tree via serialized block cache. This is acceptable since this is a full
replacement, but note that any on-disk KV cache persisted by the Python version
cannot be loaded by the C++ version.

**File:** `3rdparty/blake3/`, `include/kv_cache_manager_v2/blockRadixTree.h`

---

## Issue 2: File naming convention

**Decision:** camelCase used for `.h/.cpp` filenames (e.g., `lifeCycleRegistry.h`,
`kvCache.h`) matching `kvCacheManagerV2Utils.cpp`. Include subdirectory paths
(`kv_cache_manager_v2/`, `utils/`, `storage/`) retain snake_case for consistency
with the existing namespace path.

**Rationale:** User instruction: "use camel case instead of snake case for C++
identifiers." The migration plan used snake_case in its file listing.

---

## Issue 3: `TokenIdExt` ABI (int vs bytes)

**Decision:** `TokenIdExt = std::variant<TokenId, std::vector<uint8_t>>` in C++,
mirroring Python's `TokenId | bytes`.

**Impact:** When passing `TokenIdExt` through the nanobind layer, Python `bytes`
objects will be serialized to `std::vector<uint8_t>`. This should be transparent
but adds a copy. If performance is critical for multi-modal workloads, a zero-copy
buffer approach (memoryview → span) should be considered.

**File:** `include/kv_cache_manager_v2/common.h`

---

## Issue 4: `llist` package dependency

**Decision:** After Phase 19, the Python `llist` package (which provides
`dllist`/`dllistnode`) is no longer needed. Before removing it:

1. Check `pyproject.toml` and `requirements*.txt` for `llist` dependency entries.
2. Remove those entries to avoid pulling in a C-extension that is no longer used.

**Status:** Phase 19 (Python cleanup) not yet performed. Do not remove until
Phase 19 is complete and tests pass.

---

## Issue 5: `MemAddress` / `DiskAddress` duplication ✓ RESOLVED

**Resolution:** `kvCacheManagerV2Utils.h` now `#include`s
`kv_cache_manager_v2/common.h` and removes its own `DiskAddress` struct and
`MemAddress` typedef. The canonical definitions live in `common.h`.
`common.h` is the include path root for all kv_cache_manager_v2 code.

**Files:** `kvCacheManagerV2Utils.h`, `common.h`

---

## Issue 6: `copyEngine.cpp` — CUDA stream for migration ✓ RESOLVED

**Resolution:** Added `TemporaryCudaStream` RAII class (in `utils/cudaEvent.h/.cpp`)
that acquires a stream from the pool and calls `cuStreamWaitEvent` for each prior
event in the constructor. `StorageManager::_batchedMigrate` now uses
`TemporaryCudaStream(priorEvents)` instead of `CachedCudaStream`, collecting
`src->readyEvent` and `dst.readyEvent` for all slots as prior events.

This fixes `test_inflight_batching_4` (CUDA `assertEq` data race with
`tokens_per_block=64` under tight memory pressure).

---

## Issue 13: `getAggregatedPages` — `layer_group_id` was pool group index ✓ RESOLVED

**Resolution:** `KvCacheManager::getAggregatedPages` was returning
`static_cast<LifeCycleId>(pgIdx)` (pool group index) as `layer_group_id` in
`AggregatedPageDesc`. The correct value is `lc` (the lifecycle ID).
`LayerGroupId` is the public alias for `LifeCycleId`; pool group index is a
separate concept (multiple lifecycles can share one pool group when their slot
layouts match).

This caused `TestDisaggregatedServing` to use wrong page indices during KV
transfer, resulting in CUDA data mismatches.

**File:** `kvCacheManager.cpp:225`

---

## Issue 14: `KvCache::setCudaStream` — missing cross-stream sync ✓ RESOLVED

**Resolution:** Python's `cuda_stream` setter records an event on the OLD
stream and makes the NEW stream wait. C++ `setCudaStream` was just
`mCudaStream = stream` with no sync. Fixed to mirror Python behavior.

**File:** `kvCache.h`

---

## Issue 15: `KvCache::~KvCache` — exception in destructor ✓ RESOLVED

**Resolution:** `close()` can throw, but destructors are implicitly `noexcept`
in C++11. Added try/catch to suppress exceptions.

**File:** `kvCache.cpp`

---

## Issue 16: `get_aggregated_pages` nanobind GIL bug ✓ RESOLVED

**Resolution:** `nb::call_guard<nb::gil_scoped_release>()` on the outer
binding released the GIL before the lambda's Python API calls (`nb::cast`,
iteration). Moved `nb::gil_scoped_release` inside the lambda body after
Python→C++ conversion.

**File:** `kvCacheManagerV2.cpp`

---

## Issue 17: Missing `max_util_for_resume` in KVCacheManagerConfig binding ✓ RESOLVED

**Resolution:** Added `max_util_for_resume` and `enable_partial_reuse` as
optional kwargs (with defaults 0.97 and true) to the `KVCacheManagerConfig`
nanobind `__init__`. `TestComplexModels` now passes.

**File:** `kvCacheManagerV2.cpp`

---

## Issue 18: `stopCommitting()` double-appended tokens ✓ RESOLVED

**Resolution:** C++ `stopCommitting()` called `commit(partial)` which
re-appended the last partial tokens already in `mCommittedTokens`, inflating
the count past `mCapacity`. Python's `stop_committing()` calls
`_commit_block()` directly without re-appending. Fixed by inlining the
block-commit logic in `stopCommitting()` using existing `mCommittedTokens`.

**File:** `kvCache.cpp`

---

## Issue 19: `_adjustLevel` stub — dynamic cache level resizing ✓ RESOLVED

**Resolution:** Implemented `KvCacheManager::_adjustLevel`,
`StorageManager::adjustCacheLevel/shrinkPoolGroup/expandPoolGroup`,
`CacheLevelStorage::computeSlotCountList`, and
`PerLevelEvictionController::pageIterator`. Also added
`KvCacheManager::_gatherPersistentPages` for last-level persistent page
tracking.

**Files:** `kvCacheManager.cpp`, `storageManager.cpp`, `storage/core.cpp`,
`evictionController.cpp`, and corresponding headers.

---

## Issue 7: `storageManager.cpp` — `numPools_` name collision

**Decision:** A private method `numPools_` was added to `StorageManager` (with
trailing underscore) because there is also a public `numPools(PoolGroupIndex)`
method. This inconsistency should be cleaned up by merging into a single method.

---

## Issue 8: `blockRadixTree.cpp` — block deletion and radix tree cleanup

**Decision:** `Block::~Block()` in C++ is simplified compared to the Python
version. The Python `__del__` does:
1. For droppable+scheduled pages in storage: exclude from eviction.
2. If prev is RootBlock and next is empty: remove RootBlock from tree.

The C++ destructor handles (1) but relies on `shared_ptr` reference counting for
(2). The RootBlock removal is deferred to when the `shared_ptr<Block>` refcount
drops to zero inside the root's `next` map. This should work correctly with
`shared_ptr` but differs from the Python behavior where it happens eagerly in
`__del__`.

---

## Issue 9: `KvCache::_increaseCapacity` — multi-beam support

**Decision:** `_increaseCapacity` only allocates pages for `beam_index=0`.
Full beam search support (allocating pages for each beam) is deferred.

**Python reference:** `_KVCache.resize()` with `beam_width > 1` raises
`NotImplementedError("Not implemented yet for beam search")`. The C++ mirrors
this limitation.

---

## Issue 10: Phase 19 (Python cleanup) not yet performed

The following Python files have NOT been deleted yet. They should be removed
after the C++ migration is validated against the test suite:

- `tensorrt_llm/runtime/kv_cache_manager_v2/_core/`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_page.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_storage_manager.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_eviction_controller/`
- `tensorrt_llm/runtime/kv_cache_manager_v2/rawref/`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_common.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_config.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_exceptions.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_life_cycle_registry.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_utils.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_cuda_virt_mem.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_copy_engine.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_block_radix_tree.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_storage/`

Also update `tensorrt_llm/runtime/kv_cache_manager_v2/__init__.py` to re-export
from `tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2`.

---

## Issue 11: `storageManager.h` — missing `layerToLifeCycleIds()` accessor ✓ RESOLVED

`layerToLifeCycleIds_` member and inline `layerToLifeCycleIds()` accessor added
to `StorageManager`. Initialized in constructor from `config.layerToLifeCycleIds()`.

---

## Issue 12: `storageManager.h` — missing storage internal accessors ✓ RESOLVED

`friend class KvCacheManager;` added to `StorageManager` in `storageManager.h`.
`KvCacheManager` can now access `bufferAttr_` and `slotToPageIndices_` directly.

---

## Verification commands

```bash
# After build:
PYTHONPATH=~/tekit/tensorrt_llm/runtime/ \
    pytest tests/unittest/kv_cache_manager_v2_tests/ -v

# Full path:
PYTHONPATH=~/tekit \
    pytest tests/unittest/kv_cache_manager_v2_tests/ -v
```
