---
name: trtllm-kv-cache-manager-v2
description: >-
  Change, extend, review, or debug TensorRT-LLM KV Cache Manager V2 (KVCM V2)
  itself: the executor-side KVCacheManagerV2 wrapper and its model- and
  backend-specific subclasses, layer groups and life cycles, per-life-cycle
  reuse policies and commit timing, retention priorities, extra buffers, pool
  sizing, Host and Disk tiers, events and statistics, the nanobind surface, and
  the C++ core. Use before adding a buffer, life cycle, reuse or retention
  rule, or tier behavior; before changing capacity, history, commit, or
  release; when a feature needs something KVCM does not yet provide; and when
  reviewing such a change.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Changing KV Cache Manager V2

KVCM V2 has four layers:
- a C++ core (`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`);
- nanobind bindings (`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp`);
- a thin Python surface that re-exports them (`tensorrt_llm/runtime/kv_cache_manager_v2/`);
- the executor-side wrapper and its subclasses
  (`tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`,
  `kv_cache/mamba_cache_manager/`, and the backend managers under
  `tensorrt_llm/_torch/attention/backends/`).

The two `AGENTS.md` files of the core and of the Python surface are the source
of truth for architecture, concurrency, invariants, and commands. This skill
decides where a change belongs, which extension points to use, and how to
test and review it. Load `trtllm-kv-cache-develop` first.

## Required Context

1. `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md`, in full.
   When you touch the batch, the sparse cache, or index conversion, also read
   `SPARSE_ATTENTION_DESIGN.md` and `INDEX_CONVERSION_DESIGN.md` next to it.
2. `tensorrt_llm/runtime/kv_cache_manager_v2/AGENTS.md`: test commands, the
   rebuild steps after a C++ change, and the gotchas (string salts,
   dataclass-field grafts, the cross-language block-key format).
3. The wrapper (`kv_cache_manager_v2.py`) and the subclass you touch.
4. `docs/source/features/kvcache.md`, for user-facing behavior.

## How It Is Used

Users configure KVCM V2 through `kv_cache_config`:
- `use_kv_cache_manager_v2: true` (the default);
- pool size: `free_gpu_memory_fraction`, `max_tokens`, `max_gpu_total_bytes`;
- `tokens_per_block`;
- reuse: `enable_block_reuse`, `enable_partial_reuse`, `block_reuse_config`;
- layer-group sizing: `pool_ratio`, `avg_seq_len`, `enable_kv_pool_rebalance`;
- the host tier: `host_cache_size`, `cold_tier_offload_min_priority`.

`docs/source/features/kvcache.md` documents them, and `perf-kv-cache-tuning`
lists their performance effects. Check what a run actually built:
- `isinstance(manager, KVCacheManagerV2)` for the manager class;
- the startup log for the layer groups, pool sizes, and window sizes;
- `LLM.get_stats()` and `LLM.get_kv_cache_events()` at runtime.

## Where a Change Belongs

Prefer the highest row that can express the change.

| Need | Layer | Extension point on main |
|---|---|---|
| A model or backend needs extra per-layer buffers | Manager subclass | `_extra_buffers_per_layer()`; `get_disagg_role_mapper_kinds()` so disaggregated serving transfers them |
| Which layers share a pool, a window, sinks, or a reuse policy | Wrapper configuration build | `_build_cache_config()` in a subclass; `reuse_policy` on the layer specification; `_add_mixed_precision_layers()` is the worked example |
| Which reusable groups a request claims at resume | Wrapper | A `_resume_context_cache()` override that passes `optional_reuse_groups` for groups whose reuse status is complete. The CSA2 cache manager is the example |
| When blocks are committed for reuse | Configuration and wrapper | `block_reuse_config.policy` (`BlockReusePolicy`); commit timing in `update_context_resources()` |
| Which pages survive pressure | Wrapper callbacks and configuration | Priority callbacks (for example `swa_endpoint_rewind_tokens`); `cold_tier_offload_min_priority` |
| Transfer ids and geometry for disaggregated serving | Wrapper | The optional hooks `get_disagg_global_layer_ids` and `get_disagg_role_mapper_kinds`, read by the KV extractor |
| A new reuse policy, eviction rule, tier, migration step, or hash | C++ core | Follow the core guide's high-risk rules, below |

A method or feature never subclasses KVCM to hold algorithm state. It holds a
manager and uses its API. Subclass the wrapper only for a model or backend
whose cache layout differs.

## Contracts to Keep

The core guide has the full list. The ones changes break most often:

- **Lengths.** History never decreases, and capacity is at least history.
  `resize(capacity, None)` changes capacity and leaves history alone.
  Rejected draft slots are reclaimed by shrinking capacity. V2 has no
  `rewind_kv_cache`.
- **Commit.** Pages are immutable after commit, and `commit(is_end=True)` is
  terminal. Events describe complete, published blocks.
- **Matching.** Every `REQUIRED` life cycle must cover a match at its
  endpoint. `SINK_REQUIRED_ONLY` needs a window with sinks and applies to
  attention layers only. Coupled buffers that share a page index share a
  reuse policy.
- **Identity.** Block keys are SHA-256 digests over the reuse scope and the
  tokens, and must stay byte-identical with the router's computation. A salt
  is a string; `None`, `""`, and `"0"` are different.
- **Transactions.** Prepare, lock, submit, fence, commit. Roll back on
  failure, and publish mappings, statistics, and events only for work that
  succeeded.
- **Typed indices.** Life-cycle ids, pool-group indices, block ordinals, and
  slots are never mixed. Use `TLLM_CHECK*`, and keep expensive hot-path checks
  under the debug variants.
- **Bindings.** Preserve Python exception types. A new constructor field on a
  binding that Python passes to `dataclasses.replace()` must be added to its
  field graft, or `replace()` drops it silently.
- **Feature-off invariance.** With no new option set, allocation, page
  content, statistics, events, and outputs are identical to main.

## Adding Native Code

1. **Core.** Put C++ in `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`
   and add sources to its `CMakeLists.txt`. Follow the core guide's design
   principles: composition, typed indices, `TLLM_CHECK*`, and no Python
   dependency in the core.
2. **Bindings.** Expose the new surface in
   `cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp`, keeping
   exception types and GIL rules as the core guide describes.
3. **Python surface.**
   - Update the stub `tensorrt_llm/runtime/kv_cache_manager_v2/__init__.pyi`.
   - Add the rebind and the `__all__` entry in `__init__.py`.
   - Extend the dataclass field graft when a passed-through constructor gains
     a field.
4. **Rebuild.** Rebuild the bindings and copy the libraries, as the Python
   surface's `AGENTS.md` shows. A stale binary combined with new Python
   silently tests the wrong code.
5. **Tests.** C++ gtests in `cpp/tests/unit_tests/batch_manager/`, and the
   Python suites below, in both fast mode and production mode.

## Recipe: A New Per-Life-Cycle Reuse Policy

1. **C++ core**: the enum and its comment; the life-cycle predicates and
   validation; match pruning; the subtree eviction rule; the claim, resume,
   and commit paths in the per-request cache; storage. Update the core guide,
   which should list every policy.
2. **Bindings and stubs**: bind the value, regenerate or edit the `.pyi`
   stub, and rebuild every consumer.
3. **Python selection**: set the policy per layer or buffer when the
   configuration is built. Select optional groups at resume. Fill unselected
   pages, by recomputation or another source, before attention reads them.
4. **Disaggregated serving**: check that the transceiver's reuse reporting
   reads the new status correctly.
5. **Tests**: C++ gtests for matching and eviction, and
   `tests/unittest/kv_cache_manager_v2_tests/test_optional_checkpoint.py` for
   the claim and selection behavior.

## Recipe: A New User-Facing Commit Policy

1. Add the value to `block_reuse_config.policy` in `llm_args.py` and to
   `BlockReusePolicy`.
2. Implement its commit timing in `update_context_resources()` and the
   generation-commit gate.
3. Set the manager configuration (for example `commit_min_snapshot`) and any
   conversation-retention behavior. Check hybrid-model overrides.
4. Regenerate the LLM-args golden manifest and the serve schemas, and update
   `kvcache.md`.

## Testing

- **Python surface, fast mode.** Run the `kv_cache_manager_v2_tests` files
  directly with `PYTHONPATH` set to `tensorrt_llm/runtime/`, as the surface's
  `AGENTS.md` shows. Before final validation, also run the production import
  path.
- **Targeted suites**:
  - `test_kv_cache_event_manager.py` after event changes;
  - `test_kv_cache_salting.py` after hashing or reuse-scope changes;
  - `test_kv_cache_stats_api.py` and `test_kv_cache_stats_behavior.py` after
    allocation or accounting changes;
  - `test_optional_checkpoint.py` after reuse-policy changes.
- **C++.** The gtests in `cpp/tests/unit_tests/batch_manager/` (radix tree,
  manager, and V2 suites), after rebuilding the bindings. Use
  `TLLM_DEBUG_MODE=1` when diagnosing an invariant failure.
- **Wrapper.** `tests/unittest/_torch/pyexecutor/kv_cache/`, including
  `test_kvcm2_integration.py`, for executor-side changes.
- **Debug fills.** Run affected end-to-end paths with
  `TRTLLM_KV_FRESH_PAGE_FILL=nan` and `TRTLLM_KV_GUARD_PAGE` to catch reads of
  pages that were never written.
- **Pressure.** After any change to capacity, history, resize, suspend, or
  resume, run a high-pressure serving test with the overlap scheduler on, in
  which requests are suspended as eviction victims. Assert that no tail page
  entry is `-1`.
- Then run the feature-level gates in `trtllm-kv-cache-compat-testing`.

## Review Checklist

```markdown
## KVCM V2 review
- Layer of the change (subclass hook / wrapper / core) and why not higher: <...>
- Core-guide high-risk areas touched: <list, or none>
- Contracts at risk (lengths, commit, matching, identity, transactions, bindings): <...>
- Feature-off invariance: allocation, pages, stats, events, outputs: <how checked>
- Tests: fast mode + production import, targeted suites, C++ gtests, debug fills, pressure run: <...>
- Docs: core guide, surface guide, kvcache.md updated where behavior changed: <...>
```
