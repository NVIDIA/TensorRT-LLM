# AGENTS.md

This file provides guidance to coding agents when working with code in this directory.

## What This Is

The Python surface of KVCacheManagerV2, the KV cache management subsystem for TensorRT-LLM.
It manages GPU/host/disk memory for key-value caches used during LLM inference, handling page
allocation, eviction, multi-tier caching, radix-tree-based prefix sharing, and disaggregated
serving.

**The implementation is C++**, in `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`, exposed
through nanobind as `tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2`. Read
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md` for the architecture, the
concurrency contract, and the C++ test suite.

This directory contains only:

- `__init__.py` — re-exports the nanobind surface, plus the plain-Python aliases, constants and
  two `__dataclass_fields__` grafts the bindings do not carry. Adding a type to the bindings
  means adding a rebind here and an entry in `__all__`.
- `_introspection.py` — white-box hooks for tests and accuracy harnesses. Every function
  forwards to the bindings' native `_introspection` submodule; the indirection exists so callers
  import a stable Python path and get plain Python containers back.

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

### Rebuilding after a C++ change

```bash
cd cpp/build
cmake --build . --target bindings -j$(nproc)
cp tensorrt_llm/libtensorrt_llm.so ../../tensorrt_llm/libs/
cp tensorrt_llm/thop/libth_common.so ../../tensorrt_llm/libs/
cp tensorrt_llm/nanobind/bindings.cpython-*.so ../../tensorrt_llm/
```

### Debug Mode

Set `TLLM_DEBUG_MODE=1` to enable debug assertions (`NDEBUG=False`). Default is release mode
(`NDEBUG=True`).

## Gotchas

- **Dual import trick.** `_load_cpp_module()` reaches the bindings two ways: through
  `tensorrt_llm.bindings...` when `tensorrt_llm` is already imported, otherwise by walking up
  from `find_spec("kv_cache_manager_v2")` to the `tensorrt_llm` root and importing
  `bindings.internal.batch_manager.kv_cache_manager_v2` directly. The second path is what makes
  fast mode work; the test files mirror the same branch. Keep both in sync.
- **`__dataclass_fields__` grafts.** `BatchDesc` and `KVCacheManagerConfig` are nanobind classes
  that callers pass to `dataclasses.replace()`. `replace()` is keyed on `__dataclass_fields__`,
  so `__init__.py` attaches a field spec to each. A new constructor field on either binding must
  be added to the matching `_*FieldSpec` or `replace()` silently drops it.
- **Block-key hashing is a cross-language contract.** `root_block_key` and `block_key` must stay
  byte-identical to the C++ `RootBlock::makeKey` / `Block::makeKey` they wrap, because
  `tensorrt_llm/serve/router_utils.py` computes routing hashes with them and compares against
  hashes the engine produced. `TestBlockKeyHashing` in the unit tests guards the format.
- **Streaming KV events have no implementation.** `kv_cache_config.kv_events_config` is rejected
  by `validate_streaming_support`; the supported route is the buffered path via
  `kv_cache_config.event_buffer_max_size`. A native event sink is what would re-enable it — a
  Python sink cannot work, because the C++ radix tree calls its sink natively.
