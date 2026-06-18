# KVCacheManagerV2: Python → C++ Migration Plan

## Goal

Migrate `tensorrt_llm/runtime/kv_cache_manager_v2/` from Python to C++ with nanobind Python
bindings, preserving the **same public interface** as the current Python package.
nanobind is already the binding layer used throughout TRT-LLM's C++ core.

## Design Decisions

### Composition vs. inheritance
The Python code favors inheritance over composition to avoid slow attribute chasing: Python walks
the MRO for each attribute lookup, so a flat class with inherited fields is faster than a composed
object where you'd write `self.inner.field`. In C++, member access through composition compiles to
the same pointer offset arithmetic as base-class member access — there is no performance difference.

**Prefer composition in C++ where it expresses ownership or containment more clearly.**
Only use inheritance where there is genuine IS-A polymorphism (e.g. `CommittedPage : Page`,
or virtual dispatch via an abstract interface).

### Reference model (`rawref` replacement)
`rawref.ref[T]` exists solely to work around mypyc's lack of `weakref` support.
In C++, replace directly with `std::weak_ptr<T>`:

```cpp
// rawref.ref[T]           → std::weak_ptr<T>
// rawref.NULL             → std::weak_ptr<T>{}  (default-constructed, always expired)
// __rawref__ field        → not needed (see below)
// rawref.ref(obj)         → std::weak_ptr<T>(shared_ptr_to_obj)
// ref.invalidate()        → let the weak_ptr expire (no strong owner)
// ref.is_valid            → !weak_ptr.expired()
// ref()  (dereference)    → weak_ptr.lock()   (returns nullptr if expired)
```

The Python `__rawref__` attribute was the singleton backing store that made `rawref.ref(obj)`
return the same ref object on repeated calls — a workaround for mypyc. In C++, callers who hold
a `shared_ptr<T>` construct `std::weak_ptr<T>` from it directly; no extra field or factory
function is needed. If a method needs `shared_ptr<this>` internally, inherit from
`std::enable_shared_from_this<T>`.

### Eviction queue linked list (`llist.dllist` replacement)
`std::list<T>` provides O(1) `erase(iterator)` with stable iterators — exactly what we need.
No extra library required.

```cpp
// NodeRef = std::list<std::shared_ptr<Page>>::iterator
// std::list iterator is stable: insert/erase elsewhere does not invalidate it.

struct LRUEvictionPolicy {
    std::list<std::shared_ptr<Page>> queue;

    // push → O(1)
    NodeRef push(std::shared_ptr<Page> page, bool evictFirst = false) {
        return evictFirst ? queue.insert(queue.begin(), std::move(page))
                           : queue.insert(queue.end(), std::move(page));
    }
    // pop → O(1)
    std::shared_ptr<Page> pop() {
        auto p = std::move(queue.front());
        queue.erase(queue.begin());
        return p;
    }
    // remove by stored iterator → O(1)
    std::shared_ptr<Page> remove(NodeRef node) {
        auto p = std::move(*node);
        queue.erase(node);
        return p;
    }
};
```

`Page` stores `std::optional<NodeRef> node_ref`. The queue uses `shared_ptr<Page>` because for
`DROPPABLE` `CommittedPage`s the eviction queue is the **sole strong owner**: `_PageHolder` is
absent (`_holder == nullptr`) and `Block::storage[lc]` holds only a `weak_ptr<CommittedPage>`.
Using `Page*` would be an immediate dangling pointer for this case.

`PrioritizedEvictionPolicy` uses `std::map<Priority, LRUEvictionPolicy>` (sorted by key).
`remove(node)` looks up the priority via `(*node)->priority` to find the right sub-list.

### Python bindings
Use **nanobind** — already the binding layer in TRT-LLM. All TRT-LLM bindings compile into a
single `bindings.so` (`cpp/tensorrt_llm/nanobind/`). Add kv_cache_manager_v2 bindings as a new
file `batch_manager/kv_cache_manager_v2.cpp` in that directory, registered as a submodule:

```cpp
// batch_manager/kv_cache_manager_v2.cpp
void kvCacheManagerV2Bindings(nb::module_& m) {
    nb::class_<KVCacheManager>(m, "KVCacheManager")
        .def(nb::init<KVCacheManagerConfig const&>())
        .def("create_kv_cache", &KVCacheManager::createKvCache, ...)
        .def("shutdown", &KVCacheManager::shutdown)
        // ...
    ;
    nb::class_<_KVCache>(m, "_KVCache")
        // ...
    ;
}
```

### Copy engine: no Python bindings round-trip
`_copy_engine.py` calls `copy_device_to_device`, `copy_host_to_device`, etc. through the Python
nanobind layer. After migration, `copy_engine.cpp` calls the underlying C++ functions directly —
no interop needed since both sides are C++ (see Gotchas section for details).

### CUDA
Direct `#include <cuda.h>` (driver API) and `#include <cuda_runtime.h>`.
No FFI layer, no crate, no bindgen. Link against `CUDA::cuda_driver` in CMake.

### GIL management

**Precondition**: KVCacheManager and _KVCache are accessed from a single Python thread
(TRT-LLM's executor model). This means no mutex is needed — the single-thread invariant
already prevents concurrent access.

GIL can be released on **every public API entry** that works purely with C++ objects, allowing
other Python threads (loggers, monitors, signal handlers) to run during C++ work:

```cpp
// Pure C++ work — release GIL for the entire duration:
void KVCacheManager::resize(CacheLevel level, int quota) {
    nb::gil_scoped_release _;
    adjustLevel(level, quota);   // all C++ internals, no Python objects touched
}

// Methods that return Python objects: do C++ work with GIL released,
// then reacquire to wrap the result:
nb::ref<_KVCache> KVCacheManager::create_kv_cache(...) {
    std::shared_ptr<_KVCache> result;
    {
        nb::gil_scoped_release _;
        result = createKvCacheImpl(...);   // C++ allocation only
    }                                           // GIL reacquired here
    return nb::cast(result);                   // nanobind wrapping requires GIL
}

// Methods with Python callbacks: reacquire GIL only for the callback:
Priority callPriorityCb(nb::callable const& cb, BlockOrdinal ord, LifeCycle lc) {
    nb::gil_scoped_acquire _;   // reacquire just for the Python call
    return nb::cast<Priority>(cb(ord, lc));
}
```

No data structure changes required — unlike Rust, C++ has no compile-time constraints on
what types can be used across GIL release points.

### Memory operations
Direct `mmap`, `munmap`, `mremap`, `madvise`, `posix_fallocate` via `<sys/mman.h>` and
`<fcntl.h>`. Identical semantics to the `ctypes`-based calls in `_utils.py`.

### Build system
Add sources to the existing `bindings` CMake target in `cpp/tensorrt_llm/nanobind/CMakeLists.txt`:

```cmake
set(SRCS
    ${SRCS}
    batch_manager/kvCacheManagerV2.cpp      # nanobind bindings
    # C++ implementation sources:
    ../batch_manager/kv_cache_manager_v2/kvCacheManager.cpp
    ../batch_manager/kv_cache_manager_v2/kvCache.cpp
    # ...
)
```

Link CUDA into the existing `bindings` target if not already present:
```cmake
target_link_libraries(bindings PRIVATE CUDA::cuda_driver CUDA::cudart)
```

---

## Proposed C++ Layout

The implementation lives under `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/` — co-located
with the existing KV cache C++ code (`kvCacheManager.cpp`, `kvCacheManagerV2Utils.cpp`, etc.) in
`batch_manager/`. The nanobind bindings go in `cpp/tensorrt_llm/nanobind/batch_manager/`.
`cpp/tensorrt_llm/runtime/` is reserved for buffer management and decoder infrastructure and is not
the right home for KV cache orchestration.

```
cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/
├── CMakeLists.txt
├── common.h                 # CacheTier, CacheLevel, PageStatus, newtypes
├── config.h/cpp             # KVCacheManagerConfig, AttentionLayerConfig, etc.
├── exceptions.h             # OutOfMemoryError, LogicError, CuError, etc.
├── lifeCycleRegistry.h/cpp  # LifeCycle, LifeCycleRegistry
├── movingAverage.h          # MovingAverage, Average (header-only)
├── utils/
│   ├── math.h               # divUp, roundUp, DynamicBitset, Array2D (header-only)
│   ├── cudaEvent.h/cpp      # CachedCudaEvent, SimplePool, stream wrappers
│   └── hostMem.h/cpp        # HostMem (mmap + CUDA pinning + mremap)
├── cudaVirtMem.h/cpp        # VirtMem, PooledPhysMemAllocator
├── storage/
│   ├── config.h/cpp         # BufferId, StorageConfig, SlotDesc
│   └── core.h/cpp           # SlotAllocator, PoolGroupBase, *CacheLevelStorage
├── evictionController.h/cpp # LRUEvictionPolicy, PrioritizedEvictionPolicy,
│                            # PerLevelEvictionController, NodeRef
├── copyEngine.h/cpp         # CopyEngine, StagingBuffer, batchedCopy
├── blockRadixTree.h/cpp     # Block, RootBlock, BlockRadixTree
├── page.h/cpp               # Page, CommittedPage, _PageHolder, lock types
├── storageManager.h/cpp     # StorageManager, CacheLevelManager
├── kvCache.h/cpp            # _KVCache
└── kvCacheManager.h/cpp     # KVCacheManager, AggregatedPageDesc
```

The nanobind binding file lives separately at:
```
cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp
```

---

## Ownership Model

```
KVCacheManager  (shared_ptr, held by Python via nanobind)
├── LifeCycleRegistry                      (owned, unique_ptr or value)
├── BlockRadixTree                         (shared_ptr)
│   └── roots: vector<RootBlock>
│       └── next: unordered_map<TokenIdExt, shared_ptr<Block>>
│           └── storage[lc]: weak_ptr<CommittedPage>
└── StorageManager                         (shared_ptr)
    └── levels: vector<CacheLevelManager>  (owned)
        ├── storage: unique_ptr<CacheLevelStorage>
        └── controller: PerLevelEvictionController
            └── policies: map<Priority, LRUEvictionPolicy>
                └── queue: list<shared_ptr<Page>>   // sole owner for DROPPABLE pages

Page  (shared_ptr<Page>, owned by Slot)
├── manager:   weak_ptr<StorageManager>
├── holder:    weak_ptr<_PageHolder>
│   └── lock:  weak_ptr<_UniqPageLock>
│       └── finish_events: vector<CachedCudaEvent>
└── node_ref:  optional<list<shared_ptr<Page>>::iterator>

CommittedPage : Page
├── block: weak_ptr<Block>
└── (inherits manager, holder, node_ref from Page)

_KVCache  (shared_ptr, held by Python via nanobind)
├── manager: shared_ptr<KVCacheManager>   (not weak; cache keeps manager alive)
└── blocks:  vector<BlockData>
```

---

## Phase-by-Phase Plan

| Phase | Python source | C++ target | Notes |
|-------|--------------|------------|-------|
| **0** | — | `CMakeLists.txt`, skeleton `kvCacheManagerV2.cpp` | nanobind submodule setup |
| **1** | `_common.py` | `common.h` | Enums (`CacheTier`, `PageStatus`), `using` newtypes |
| **2** | `_config.py` | `config.h/cpp` | Config structs; mirror Python dataclass fields |
| **3** | `_exceptions.py` | `exceptions.h` | Exception hierarchy; map CUDA errors via `cuGetErrorString` |
| **4** | `_life_cycle_registry.py` | `lifeCycleRegistry.h/cpp` | `LifeCycle`, `LifeCycleRegistry` |
| **5** | `_utils.py` (math/data) | `utils/math.h` | `divUp`, `DynamicBitset`, `Array2D`, typed-index helpers |
| **6** | `_utils.py` (CUDA/mem) | `utils/cudaEvent.h/cpp`, `utils/hostMem.h/cpp` | `HostMem`, `CachedCudaEvent`, `SimplePool`, stream wrappers |
| **7** | `_cuda_virt_mem.py` | `cudaVirtMem.h/cpp` | CUDA virtual memory; `VirtMem`, `PooledPhysMemAllocator` |
| **8** | `_storage/_config.py` | `storage/config.h/cpp` | `BufferId`, `StorageConfig`, `createStorageConfig` |
| **9** | `_storage/_core.py` | `storage/core.h/cpp` | `SlotAllocator`, `PoolGroupBase`, `*CacheLevelStorage`, resize logic |
| **10** | `_eviction_controller/` | `evictionController.h/cpp` | `LRUEvictionPolicy` with `std::list`; `PerLevelEvictionController` |
| **11** | `_copy_engine.py` | `copyEngine.h/cpp` | `CopyEngine`, `StagingBuffer`; direct calls to existing copy functions |
| **12** | `_block_radix_tree.py` | `blockRadixTree.h/cpp` | `Block`, `BlockRadixTree`; BLAKE3 for token sequence hashing |
| **13** | `_page.py` | `page.h/cpp` | `Page`, `CommittedPage`, `_PageHolder`, lock types; state machine |
| **14** | `_storage_manager.py` | `storageManager.h/cpp` | `StorageManager`, migration/eviction orchestration |
| **15** | `_core/_moving_average.py` | `movingAverage.h` | `MovingAverage`; trivial, header-only |
| **16** | `_core/_kv_cache.py` | `kvCache.h/cpp` | Largest file; SUSPENDED/ACTIVE state machine, beam search, sliding window |
| **17** | `_core/_kv_cache_manager.py` | `kvCacheManager.h/cpp` | `KVCacheManager`, ratio adjustment, `AggregatedPageDesc` |
| **18** | all | `nanobind/batch_manager/kvCacheManagerV2.cpp` added to `bindings.so` | All `nb::class_<>` definitions; update `__init__.py` with dual-PYTHONPATH trick |
| **19** | all `.py` impl files | — | Delete `_core/`, `_page.py`, `_storage_manager.py`, `_eviction_controller/`, `rawref/`; remove `llist` dep |

### Dependency graph

```
Phase 0
  └── Phases 1–3 (types, config, exceptions)
          └── Phase 4 (life_cycle_registry)
                └── Phase 5 (math utils)
                      └── Phase 6 (cuda/mem utils)
                            ├── Phase 7 (cuda_virt_mem)
                            └── Phase 8 (storage config)
                                  └── Phase 9 (storage core)
                                        ├── Phase 10 (eviction controller)
                                        ├── Phase 11 (copy engine)
                                        └── Phase 12 (block radix tree)
                                              └── Phase 13 (page)
                                                    └── Phase 14 (storage manager)
                                                          ├── Phase 16 (_kv_cache)
                                                          └── Phase 17 (kv_cache_manager)
Phase 15 (moving average) -- independent, any time before Phase 17
All phases feed Phase 18 (bindings) then Phase 19 (cleanup)
```

---

## Gotchas & Risks

### `shared_ptr` cycles
The Python code uses `rawref`/`weakref` deliberately to break cycles. The same discipline is
required with `weak_ptr` vs `shared_ptr`:
- `Page → StorageManager`: **weak**
- `Page → _PageHolder`: **weak**
- `CommittedPage → Block`: **weak**
- `Block → parent Block`: **weak** (or raw pointer; parent outlives child)
- `_KVCache → KVCacheManager`: **strong** (cache keeps manager alive)
- `KVCacheManager → _KVCache` (via `living_kv_caches`): **weak**

Inherit from `std::enable_shared_from_this<T>` on classes that need `shared_from_this()` internally. No factory functions are required — construct with `std::make_shared<T>()` directly.

### `_KVCache` complexity
`_core/_kv_cache.py` is the largest and most complex file: SUSPENDED/ACTIVE state machine,
multi-beam block arrays, sliding window eviction, partial block reuse, intra-batch rebasing,
and extensive invariant checking. Budget the most time and testing effort here.

### Destructor ordering and cascades
Python `CommittedPage.__del__` calls `block.unset_page()` which may cascade deletions.
In C++, `shared_ptr` destruction runs the destructor synchronously when the last owner drops.
Audit all destructors for re-entrancy. Use `std::optional<T>` fields with `.reset()` to guard
against double-destruction, mirroring the Python pattern.

### Copy engine: direct C++ calls, no bindings round-trip
`_copy_engine.py` currently calls `copy_device_to_device`, `copy_host_to_host`,
`copy_disk_to_host`, etc. by importing them through the Python nanobind layer
(`bindings.internal.batch_manager`), which wraps `copyDeviceToDevice()`, `copyHostToHost()`,
`copyDiskToHost()`, etc. in `batch_manager/kvCacheManagerV2Utils.cpp`.

After migration, `copy_engine.cpp` **calls those C++ functions directly** — `#include
"kvCacheManagerV2Utils.h"` and link against the same C++ library. The Python binding wrappers in
`nanobind/batch_manager/kvCacheManagerV2Utils.cpp` are not involved at all.

### `std::list` iterator as `NodeRef`
Iterators from different `std::list` instances are incomparable and must not be mixed.
This is the same invariant as the Python `dllistnode` — `node_ref` is always used with the
controller that issued it. Document this invariant clearly in the header.

When a page changes cache level, `exclude_from_eviction` must be called before the page is
moved (which clears `node_ref` and releases the queue's `shared_ptr`) and
`schedule_for_eviction` called on the new controller (which sets a new `node_ref` into the
new list and takes ownership via a new `shared_ptr`).

### Hashing for block keys
`_block_radix_tree.py` uses Python's `hashlib.sha256` to hash token sequences into block keys.
In C++, use **BLAKE3** via the [`blake3`](https://github.com/BLAKE3-team/BLAKE3/tree/master/c)
C implementation (CC0/Apache-2.0, no license concerns). OpenSSL/libcrypto are **not** existing
dependencies of TRT-LLM's C++ code, so adding them would be unnecessary overhead.
BLAKE3 is also significantly faster than SHA-256 for this use case.

```cmake
# Add directly to the existing bindings target — no library dependency needed:
target_sources(bindings PRIVATE
    third_party/blake3/blake3.c
    third_party/blake3/blake3_dispatch.c
    third_party/blake3/blake3_portable.c
)
target_include_directories(bindings PRIVATE third_party/blake3)
```

```cpp
#include "blake3.h"

std::array<uint8_t, BLAKE3_OUT_LEN> hashTokens(std::span<TokenIdExt const> tokens) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, tokens.data(), tokens.size_bytes());
    std::array<uint8_t, BLAKE3_OUT_LEN> out;
    blake3_hasher_finalize(&hasher, out.data(), BLAKE3_OUT_LEN);
    return out;
}
```

### Nanobind exception translation
C++ exceptions must be registered with nanobind to propagate correctly to Python:
```cpp
nb::register_exception<OutOfMemoryError>(m, "OutOfMemoryError");
nb::register_exception<CuError>(m, "CuError");
// etc.
```

### GIL release scope
The single-threaded caller precondition means no mutex is needed. Rules for GIL release:
- **Release** at the top of any public API that works purely with C++ objects
- **Never** access Python objects (`nb::object`, `nb::callable`, reference counts) with GIL released
- **Reacquire** (via `nb::gil_scoped_acquire`) before calling Python callbacks or creating Python return values
- If the single-threaded precondition is ever relaxed, add a `std::recursive_mutex` per `KVCacheManager` and acquire it after releasing the GIL

---

## Post-migration Python package structure

All TRT-LLM C++ nanobind bindings compile into a single **`bindings.so`**
(`cpp/tensorrt_llm/nanobind/`). The kv_cache_manager_v2 bindings follow the same pattern —
add `batch_manager/kv_cache_manager_v2.cpp` to the existing `SRCS` list in
`cpp/tensorrt_llm/nanobind/CMakeLists.txt`. No separate `.so` is created.

Keep `tensorrt_llm/runtime/kv_cache_manager_v2/` as a thin re-export package, using the same
conditional import trick already used in `_copy_engine.py` to handle both PYTHONPATH modes:

```python
# tensorrt_llm/runtime/kv_cache_manager_v2/__init__.py
import sys
from importlib.util import find_spec
from pathlib import Path

if "tensorrt_llm" in sys.modules:
    # Production: tensorrt_llm already loaded (PYTHONPATH=~/tekit)
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2 import (
        KVCacheManager, _KVCache, ...
    )
else:
    # Fast dev path: only kv_cache_manager_v2 is on sys.path
    # (PYTHONPATH=~/tekit/tensorrt_llm/runtime/)
    spec = find_spec("kv_cache_manager_v2")
    assert spec is not None and spec.origin is not None
    _trtllm_root = str(Path(spec.origin).parent.parent.parent)
    with temporary_sys_path(_trtllm_root):
        from bindings.internal.batch_manager.kv_cache_manager_v2 import (
            KVCacheManager, _KVCache, ...
        )

__all__ = [...]  # same list as today
```

(`temporary_sys_path` is a small context manager already implemented in `_utils.py` — copy it
into the thin wrapper or inline it.)

All `.py` implementation files (`_core/`, `_storage_manager.py`, `_page.py`,
`_eviction_controller/`, `rawref/`) are deleted. Only `__init__.py` survives.

---

## Verification

The test suite is at `tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py`.
Run in two stages:

### Stage 1 — isolated fast path (during migration, per-phase)
```bash
PYTHONPATH=~/tekit/tensorrt_llm/runtime/ \
    pytest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py -v
```
Only `kv_cache_manager_v2` and `bindings.so` are loaded — the rest of `tensorrt_llm`'s Python
code is not imported. Use this for fast iteration during development; run after each phase that
adds new bindings.

### Stage 2 — production path (final validation)
```bash
PYTHONPATH=~/tekit \
    pytest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py -v
```
Imports via `tensorrt_llm.runtime.kv_cache_manager_v2`. Run this before marking the migration
complete.

Both stages must pass before a phase is considered done.

---

## Progress Tracker

- [ ] Phase 0: Infrastructure (CMakeLists.txt, nanobind submodule skeleton in `bindings.so`)
- [ ] Phase 1: `common.h`
- [ ] Phase 2: `config.h/cpp`
- [ ] Phase 3: `exceptions.h`
- [ ] Phase 4: `lifeCycleRegistry.h/cpp`
- [ ] Phase 5: `utils/math.h`
- [ ] Phase 6: `utils/cudaEvent.h/cpp`, `utils/hostMem.h/cpp`
- [ ] Phase 7: `cudaVirtMem.h/cpp`
- [ ] Phase 8: `storage/config.h/cpp`
- [ ] Phase 9: `storage/core.h/cpp`
- [ ] Phase 10: `evictionController.h/cpp`
- [ ] Phase 11: `copyEngine.h/cpp`
- [ ] Phase 12: `blockRadixTree.h/cpp`
- [ ] Phase 13: `page.h/cpp`
- [ ] Phase 14: `storageManager.h/cpp`
- [ ] Phase 15: `movingAverage.h`
- [ ] Phase 16: `kvCache.h/cpp`
- [ ] Phase 17: `kvCacheManager.h/cpp`
- [ ] Phase 18: `nanobind/batch_manager/kvCacheManagerV2.cpp` (nanobind bindings added to `bindings.so`), update `__init__.py` with dual-PYTHONPATH trick
- [ ] Phase 19: Remove all `.py` implementation files (`_core/`, `_page.py`, `_storage_manager.py`, `_eviction_controller/`, `rawref/`, `llist` dep)
