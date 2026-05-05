# KVCacheManagerV2 Python-to-C++ Parity Audit

Date: 2026-04-30

Scope: `tensorrt_llm/runtime/kv_cache_manager_v2` Python reference vs. C++ implementation in
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2` and nanobind surface in
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp`.

Python is treated as the behavioral reference unless an intentional non-parity decision is explicitly documented.

## Dynamic Checks Run

Fast-dev import path:

```bash
PYTHONPATH=/home/yaoy/tekit/tensorrt_llm/runtime TLLM_KV_CACHE_MANAGER_V2_BACKEND=python python -c "import kv_cache_manager_v2 as kv; ..."
PYTHONPATH=/home/yaoy/tekit/tensorrt_llm/runtime TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp python -c "import kv_cache_manager_v2 as kv; ..."
```

Confirmed:

- Python backend exposes `AggregatedPageDesc`, `gen_multi_modal_tokens`, and `rawref`; C++ backend does not.
- Python backend exposes `KVCacheManager.allow_seq_rebasing`, `adjust`, `need_adjustment`, `_KVCache.is_active`, and `_KVCache.finish_event`; C++ backend does not.
- Python `GpuCacheTierConfig` exposes `tier` and `assert_valid`; C++ backend does not.
- Python `BufferId(1, "key")` is tuple-like and hash-compatible with equality; C++ `BufferId` is not tuple-like and equal instances have different hashes.

Direct unittest runs through `python`:

```bash
PYTHONPATH=/home/yaoy/tekit/tensorrt_llm/runtime:/home/yaoy/tekit/tests/unittest/kv_cache_manager_v2_tests \
TLLM_KV_CACHE_MANAGER_V2_BACKEND=python \
python tests/unittest/kv_cache_manager_v2_tests/test_branch_reuse.py

PYTHONPATH=/home/yaoy/tekit/tensorrt_llm/runtime:/home/yaoy/tekit/tests/unittest/kv_cache_manager_v2_tests \
TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp \
python tests/unittest/kv_cache_manager_v2_tests/test_branch_reuse.py
```

Both backends passed `test_branch_reuse.py`: 1 test, 0 failures.

```bash
PYTHONPATH=/home/yaoy/tekit/tensorrt_llm/runtime:/home/yaoy/tekit/tests/unittest/kv_cache_manager_v2_tests \
TLLM_KV_CACHE_MANAGER_V2_BACKEND=python \
python tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py

PYTHONPATH=/home/yaoy/tekit/tensorrt_llm/runtime:/home/yaoy/tekit/tests/unittest/kv_cache_manager_v2_tests \
TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp \
python tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py
```

Both backends passed `test_kv_cache_manager_v2.py`: 55 tests, 12 skipped.

Pytest-specific blocked checks:

- `pytest tests/unittest/kv_cache_manager_v2_tests/test_branch_reuse.py -q` with fast-dev `PYTHONPATH` fails before tests because `tests/unittest/conftest.py` imports top-level `tensorrt_llm`, which is absent from that path.
- Production import path with `PYTHONPATH=/home/yaoy/tekit` fails before tests on an unrelated top-level import error: `ImportError: cannot import name 'deep_gemm' from partially initialized module 'tensorrt_llm'`.

## High Severity Findings

### 1. C++ staging buffer can index past the ring end

Python caps each staging allocation to the remaining contiguous suffix and asserts it does not pass the ring end:
`tensorrt_llm/runtime/kv_cache_manager_v2/_copy_engine.py:247` through `:253`.

C++ computes suffix availability, raises `actualSize` back to `minSize`, then indexes `mGrains[mStartGrain + i]`:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/copyEngine.cpp:81` through `:98`, and again in the destructor at `:113`.

Expected: staging allocations never span past `numGrains`; wrap happens before allocation or allocation is capped.
Actual: if `mNext` is near the end and the suffix is smaller than `minSize`, C++ can access past `mGrains`.

Minimal repro: 64 MiB staging pool, 1 MiB grains, `mNext == 63`, request `minSize = 2 MiB`. C++ starts at grain 63 with `mNumGrains = 2` and touches grain 64.

### 2. `create_kv_cache` binding is not keyword/type compatible

Python exposes `custom_priority_callback` and accepts `input_tokens: Sequence[...]`:
`tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache_manager.py:259` through `:278`, with the callback stored as `Callable[[BlockOrdinal, LifeCycle], Priority]` in `_core/_kv_cache.py:221` through `:228`.

C++ binding names the callback `priority_cb`, only converts `input_tokens` when it is exactly a Python list, and otherwise passes an empty vector:
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:380` through `:391`. The C++ callback type is `(BlockOrdinal, LifeCycleId)` in `kvCache.h:144` through `:145`.

Expected: `custom_priority_callback=` works, any sequence is honored, and callback semantics match Python.
Actual: Python keyword users fail, tuple/other sequence inputs silently disable prefix matching, and callback receives an integer lifecycle id instead of the Python lifecycle object.

### 3. Public manager controls are omitted from nanobind

Python public surface includes `allow_seq_rebasing`, `adjust`, and `need_adjustment`:
`tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache_manager.py:311` through `:318`, `:483` through `:492`.

C++ has equivalent declarations:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h:137` through `:168`.

The binding exposes nearby manager properties but omits those three:
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:398` through `:434`.

Expected: default C++ backend exposes the same public manager controls.
Actual: users get `AttributeError`.

### 4. Top-level C++ shim omits descriptor/converter classes returned by APIs

The stub declares `ExpandedBuffer`, `AggregatedPageDesc`, and `PageIndexConverter`:
`tensorrt_llm/runtime/kv_cache_manager_v2/__init__.pyi:230` through `:255`.

C++ binding defines them:
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:436` through `:455`.

The default C++ shim import lists omit them:
`tensorrt_llm/runtime/kv_cache_manager_v2/__init__.py:77` through `:95`, and `:102` through `:120`.

Expected: top-level imports work under both backends.
Actual: `from kv_cache_manager_v2 import AggregatedPageDesc` succeeds under Python backend and fails under C++ backend.

### 5. `BufferId` equality/hash protocol diverges

Python `BufferId` is a `NamedTuple`:
`tensorrt_llm/runtime/kv_cache_manager_v2/_storage/_config.py:36` through `:38`.

C++ binding exposes `BufferId` as a mutable class with `__eq__` only:
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:107` through `:112`.

Dynamic check:

```text
python: a == b True, hash(a) == hash(b) True, {a: 7}[b] == 7
cpp:    a == b True, hash(a) == hash(b) False, {a: 7}.get(b) == "miss"
```

Expected: equal `BufferId`s work as equivalent dict/set keys, as in Python.
Actual: C++ equal objects have identity-like hashes, breaking dict/set behavior and any user code mirroring Python storage maps.

### 6. Capacity decrease can regrow C++ base-page-index buffers

Python deletes blocks first, letting lock destructors write `BAD_PAGE_INDEX`, then truncates/fills index buffers:
`tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py:427` through `:432`.

C++ shrinks page-index buffers before destroying dropped block locks:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp:683` through `:688`.
`SharedPageLock` destruction calls `updateBasePageIndex`:
`page.cpp:318`, `page.cpp:374`.
For internal vectors, `updateBasePageIndex` resizes to `ord + 1`:
`kvCache.cpp:1351`.

Expected: after decreasing capacity, base index buffer length tracks the new block count.
Actual: dropping a block after shrinking can regrow internal index buffers with `BAD_PAGE_INDEX` tail entries.

Minimal repro: create an active cache with 2 blocks, set capacity to one block, then `get_base_page_indices(0)` can have length 2 while `num_blocks == 1`.

### 7. `resume()` utilization gate uses different semantics

Python rejects resume if any GPU pool group exceeds threshold:
`tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py:621` through `:623`,
with per-pool-group utilization from `_storage_manager.py:590` through `:596`.

C++ computes one weighted aggregate:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp:232`,
`storageManager.cpp:662` through `:671`.

Expected: one over-threshold pool group blocks resume.
Actual: a saturated small pool can be masked by low utilization elsewhere.

Minimal repro: threshold `0.9`, two pool groups with utilizations `[1.0, 0.0]`. Python returns `False`; C++ may resume.

### 8. Ratio resize rounding diverges

Python uses `round()` in quota math:
`tensorrt_llm/runtime/kv_cache_manager_v2/_storage/_core.py:771` through `:778`, and `:831` through `:836`.

C++ uses `std::round` / `std::llround`:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.cpp:680` through `:683`, and `:755` through `:756`.

Expected: C++ matches Python banker rounding.
Actual: `.5` values round away from zero, so pool-group slot counts can differ.

Minimal repro: `total_quota=5`, two pool groups, `ratio=[0.5, 0.5]`, `min_slots=[1, 1]`, `slot_size_lists=[[1], [1]]`. Python gives `[2, 3]`; C++ gives `[3, 2]`.

### 9. Storage layout pool-group ordering diverges

Python builds groups in first-seen insertion order:
`tensorrt_llm/runtime/kv_cache_manager_v2/_storage/_config.py:158` through `:198`.

C++ uses `std::map` for lifecycle and slot-size grouping, then emits sorted groups:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/config.cpp:108` through `:110`,
`:156` through `:172`.

Expected: pool-group indices and layout order match Python first-seen order.
Actual: C++ sorts by lifecycle id and lexicographic slot-size vector, changing visible group indices and memory layout order.

Minimal repro: first-seen slot-size groups `[200]` then `[100]`. Python emits `[200], [100]`; C++ emits `[100], [200]`.

### 10. Block key hashing is not parity-compatible

Python uses SHA-256 and encodes ints as 8-byte little endian:
`tensorrt_llm/runtime/kv_cache_manager_v2/_block_radix_tree.py:49` through `:85`.

C++ uses BLAKE3:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.cpp:47` through `:99`.

Expected for strict parity: identical root/block keys for the same LoRA id and token sequence.
Actual: every digest differs from Python, including the empty root for `lora_task_id=None`.

This is documented in `issues.md`, so it is an intentional non-parity decision, not an accidental translation error. It still violates strict behavioral parity if block keys are observable, serialized, or compared across implementations.

## Medium Severity Findings

### 11. Config tier methods and Helix config are declared but not bound

Python tier configs expose `tier` and `assert_valid`:
`tensorrt_llm/runtime/kv_cache_manager_v2/_config.py:44` through `:81`.

C++ headers have `tier()` / `assertValid()`:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h:34` through `:70`.

Binding exposes only `quota`/`path`:
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:122` through `:133`.

Python declares `HelixConfig` and `KVCacheManagerConfig.helix_config`:
`_config.py:164` through `:216`.
C++ header has them:
`config.h:220` through `:257`.
Binding constructor/properties stop at `ssm_reuse_interval`:
`kvCacheManagerV2.cpp:179` through `:218`.

Expected: config objects match the stub and Python reference.
Actual: `tier`, `assert_valid`, `HelixConfig`, and `helix_config` are unavailable through C++ backend.

### 12. `_KVCache.finish_event` and `_KVCache.is_active` are missing from binding

Python defines `finish_event` and `is_active`:
`tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py:304` through `:307`, and `:767` through `:772`.

C++ has corresponding methods:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h:216` through `:219`, and `:292`.

Binding exposes nearby properties but omits both:
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:271` through `:289`.

Expected: same public properties.
Actual: C++ backend raises `AttributeError`, including for the synchronization event surface.

### 13. `_KVCache.commit` signature is narrower in C++

Python accepts `accepted_input_tokens` plus optional `beam_search_indices`:
`tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py:539` through `:548`.

C++ only has `commit(std::vector<TokenIdExt> const& tokens)`:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h:180` through `:183`,
bound as one arg named `tokens`:
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:243` through `:250`.

Expected: `commit(accepted_input_tokens=..., beam_search_indices=None)` remains valid.
Actual: documented Python keywords/optional arg fail under C++.

### 14. Base page index API return and buffer validation diverge

Python returns `array.array | memoryview`, asserts external buffer length is at least `num_blocks`, and fills the tail with `BAD_PAGE_INDEX`:
`tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py:265` through `:285`,
`:346` through `:354`.

C++ binding returns an ownerless read-only NumPy ndarray from an internal span:
`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp:261` through `:266`.
C++ accepts an external buffer and defers size safety to later asserts:
`kvCache.cpp:1429`, `kvCache.cpp:1333`.

Expected: same return contract and immediate rejection of undersized buffers.
Actual: different return type with lifetime risk, and undersized external buffers can be installed before later assertion or out-of-bounds paths.

### 15. SSM snapshot level probing wraps in C++

Python tries `src_page.cache_level + i` directly:
`tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py:797` through `:803`,
using `StorageManager.new_slots` at `_storage_manager.py:296`.

C++ wraps levels modulo `numCacheLevels()`:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp:477` through `:479`,
then calls `StorageManager::newSlots` at `storageManager.cpp:256`.

Expected: no wraparound fallback under Python reference.
Actual: C++ may store an SSM snapshot in an earlier cache level that Python would not reach.

Minimal repro: two levels, source SSM page at level 1, level 1 cannot allocate, GPU level can allocate. Python reaches invalid level 2 and raises; C++ wraps to level 0 and succeeds.

### 16. Last-level held-page eviction ordering is unstable in C++

Python `remove_if` is stable:
`tensorrt_llm/runtime/kv_cache_manager_v2/_utils.py:174` through `:183`.
Python uses it for held pages, appends them back, then accepts from the tail:
`_storage_manager.py:394`, `:425`, `:431`.

C++ uses unstable `std::partition`, appends held pages back, then accepts from the tail:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.cpp:383`,
`:420`, `:429`.

Expected: retained and held fallen pages preserve Python ordering before tail acceptance.
Actual: C++ may accept a different held page than Python.

Minimal repro: `fallenPages[pg] = [H1, D, H2]`, capacity to accept one held page. Python accepts `H2`; C++ can reorder and accept `H1`.

### 17. Explicit CUDA virtual-memory destroy drops CUDA errors

Python `VirtMem.destroy()` checks `cuCtxSynchronize` and `cuMemAddressFree` via `_unwrap`:
`tensorrt_llm/runtime/kv_cache_manager_v2/_cuda_virt_mem.py:150` through `:156`,
with `_unwrap` raising on non-success at `_utils.py:66`.

C++ `VirtMem::destroy()` calls both APIs unchecked, then clears state:
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/cudaVirtMem.cpp:198` through `:204`.

Expected: explicit destroy surfaces CUDA failures or avoids marking VA as freed after failed free.
Actual: C++ silently ignores CUDA sync/free failures and clears `mAddr/mVmSize`.

### 18. `NDEBUG` environment interpretation differs

Python reference:
`tensorrt_llm/runtime/kv_cache_manager_v2/_common.py:21`

```python
NDEBUG = int(os.environ.get("TLLM_KV_CACHE_MANAGER_V2_DEBUG", "0")) == 0
```

C++ backend shim:
`tensorrt_llm/runtime/kv_cache_manager_v2/__init__.py:141`

```python
NDEBUG = os.environ.get("TLLM_KV_CACHE_MANAGER_V2_DEBUG", "") == ""
```

Expected: same env values produce same debug mode.
Actual: `TLLM_KV_CACHE_MANAGER_V2_DEBUG=0` yields `NDEBUG=True` in Python reference and `NDEBUG=False` in C++ shim.

## Test Gaps To Add

- Fresh-process A/B harness for `TLLM_KV_CACHE_MANAGER_V2_BACKEND=python/cpp`, because backend selection is import-time global.
- Import smoke tests for every top-level name in `__all__` and every public name in `__init__.pyi`.
- Signature parity tests for constructors and public methods, including keyword calls.
- Differential trace tests that compare status, commit state, committed tokens, base/aggregated page indices, storage ratios, and radix-tree shape after each operation.
- Property tests for storage ratio rounding and pool-group ordering.
- Copy-engine tests that force staging ring wrap near the end.
- Resize tests that shrink capacity after external/internal page-index buffers are installed.
- SSM snapshot tests for source pages at the last cache level.
- CUDA error-path tests for explicit virtual-memory destroy.
