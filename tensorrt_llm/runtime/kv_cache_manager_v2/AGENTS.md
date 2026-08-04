# AGENTS.md

This file provides guidance to coding agents when working with code in this directory.

## What This Is

KVCacheManagerV2 is the KV cache management subsystem for TensorRT-LLM. It manages GPU/host/disk memory for key-value caches used during LLM inference, handling page allocation, eviction, multi-tier caching, radix-tree-based prefix sharing, and disaggregated serving.

This is a **pure Python implementation** designed to be compilable with **mypyc** for production performance. There is also a `rawref` C extension for mutable object references.

## Commands

### Running Tests

**Fast mode** (avoids loading full `tensorrt_llm` — preferred during development):
```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
PYTHONPATH="$REPO_ROOT/tensorrt_llm/runtime/" \
    python "$REPO_ROOT/tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py" -v
```

**Single test class or method:**
```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
PYTHONPATH="$REPO_ROOT/tensorrt_llm/runtime/" \
    python "$REPO_ROOT/tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py" \
    TestNoBatching.test_basic -v
```

**Production mode** (imports via `tensorrt_llm.runtime.kv_cache_manager_v2`):
```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
PYTHONPATH="$REPO_ROOT/" \
    python "$REPO_ROOT/tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py" -v
```

### Building Extensions

```bash
# Build rawref C extension
cd rawref && python setup.py build_ext --inplace

# Build mypyc-compiled version (run from runtime/ parent dir)
cd .. && python kv_cache_manager_v2/setup_mypyc.py build_ext --inplace

# Or use the Makefile (builds both)
make all
```

### Debug Mode

Set `TLLM_KV_CACHE_MANAGER_V2_DEBUG=1` to enable debug assertions (`NDEBUG=False`). Default is release mode (`NDEBUG=True`).

## Architecture

### Dual Import Trick

The test file uses `find_spec("kv_cache_manager_v2")` to detect whether the package is importable as a top-level module (fast mode via `PYTHONPATH=.../runtime/`) or must be imported via the full path `tensorrt_llm.runtime.kv_cache_manager_v2` (production mode). This allows the same test file to work in both contexts.

### Core Layers (bottom-up)

1. **`_common.py`** — Fundamental types: `TokenId`, `TokenIdExt`, `CacheLevel`, `CacheTier`, `PageStatus`, `CudaStream`, `BeamIndex`, `BlockOrdinal`. All use `NewType` for type safety.

2. **`rawref/`** — C extension providing `ref[T]`, a substitute for `weakref.ref` which is not compatible with mypyc. Uses raw object IDs instead of Python's weak reference machinery. Used throughout for parent/back-references that must not prevent GC. Objects must define `__rawref__ = NULL` class attribute and call `invalidate()` in `__del__`.

3. **`_storage/`** — Low-level memory pool management. `_config.py` defines buffer/pool configurations. `_core.py` provides `CacheLevelStorage` with slot-based allocation across `PoolGroup`s and `Pool`s.

4. **`_storage_manager.py`** — Coordinates storage across cache tiers (GPU → Host → Disk). Manages the `PerLevelEvictionController` and `batched_copy` for cross-tier data movement.

5. **`_page.py`** — Page abstraction: `CommittedPage` (finalized, prefix-shareable), `UncommittedPage` (being filled), `BlockPage` (within a radix tree block). Includes `_SharedPageLock` and `batched_lock_to_gpu` for multi-level page locking.

6. **`_life_cycle_registry.py`** — Maps `LayerGroupId`→`LifeCycleId`. Each layer group has either `AttnLifeCycle` (with optional sliding window + sink tokens) or `SsmLifeCycle`. Controls which blocks are "stale" and eligible for eviction.

7. **`_block_radix_tree.py`** — Radix tree for prefix sharing across sequences. Blocks store pages and token IDs. Supports multi-modal tokens via `gen_multi_modal_tokens`.

8. **`_eviction_controller/`** — Decides which pages to evict when memory is low, per cache level.

9. **`_copy_engine.py`** — `CopyTask` and `batched_copy` for efficient GPU↔Host↔Disk data transfers.

10. **`_core/_kv_cache.py`** (`_KVCache`) — Per-sequence cache state. Manages the block chain, commit/uncommit lifecycle, beam search forks, page locking, and cross-level migration. This is the most complex module.

11. **`_core/_kv_cache_manager.py`** (`KVCacheManager`) — Top-level manager. Owns all `_KVCache` instances, handles batched operations (`prepareStep`, `acceptStep`, `cleanStep`), quota management, and disaggregated serving coordination.

### Key Type Aliases

- `LayerGroupId` = public alias of `LifeCycleId` (same value, different semantic)
- `PoolGroupIndex` ≠ `LayerGroupId` — pool group index is the storage-level index
- `PageIndex` = index into a pool's slot array
- `BlockOrdinal` = position of a block in a sequence's block chain

### Page Lifecycle

`UncommittedPage` → (commit) → `CommittedPage` → (evict to host) → `CommittedPage` at lower level → (recall to GPU) → `CommittedPage` at GPU level

Pages inside radix tree blocks are `BlockPage` wrappers that delegate to the underlying `CommittedPage`/`UncommittedPage`.

## Gotchas

- **`rawref` must be built first** — the package imports `rawref` at the top level. Run `make rawref` before anything else.
- **mypyc compilation is from the `runtime/` directory** — `setup_mypyc.py` expects `kv_cache_manager_v2/` prefixes in module paths.
- **`_exceptions.py` excluded from mypyc** — mypyc can't compile classes inheriting from builtin `Exception`.
- **`NDEBUG` controls assertions** — many hot-path assertions are gated behind `if not NDEBUG:`. Don't remove these guards.
- **`stopCommitting()` must NOT call `commit()`** — it would double-append tokens to the block.
- **Cross-stream sync on `cuda_stream` setter** — changing the CUDA stream records an event on the old stream and waits on the new one. This is intentional.
