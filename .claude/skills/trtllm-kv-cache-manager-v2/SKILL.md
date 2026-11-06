---
name: trtllm-kv-cache-manager-v2
description: >-
  Entry point for KV cache manager V2 development in TensorRT-LLM: the
  executor-side KVCacheManagerV2 wrapper and its model- and backend-specific
  subclasses, layer groups and life cycles, per-life-cycle reuse policies and
  commit timing, retention priorities, extra buffers, pool sizing, host and
  disk tiers, events and statistics, the nanobind surface, and the native
  core, plus features built directly on them such as the mixed-precision KV
  cache. Read it first before designing, implementing, debugging, or
  reviewing such a change. Gives the hard rules, required reading, current
  and deprecated paths, owner map, where a change belongs, contracts, and
  tests. Also hosts the engineering rules, design principles, and review
  lessons that the KV cache compression framework skill
  (trtllm-kv-cache-compression) uses as well.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# KV Cache Manager V2 Development

KV cache manager V2 has four layers:
- the native core, in C++ (`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`);
- nanobind bindings (`cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp`);
- a thin Python surface that re-exports them (`tensorrt_llm/runtime/kv_cache_manager_v2/`);
- the executor-side wrapper and its subclasses
  (`tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`,
  `kv_cache/mamba_cache_manager/`, and the backend managers under
  `tensorrt_llm/_torch/attention/backends/`).

This skill is the entry point for work on KV cache manager V2 and on
features built directly on it, such as the mixed-precision KV cache. Work on
the KV cache compression framework, meaning compression methods such as
TriAttention eviction and cold-page compression, starts from
`trtllm-kv-cache-compression` (`../trtllm-kv-cache-compression/SKILL.md`),
which builds on the contracts described here.

The two `AGENTS.md` files of the native core and of the Python surface are the
source of truth for architecture, concurrency, invariants, and commands. This
skill decides where a change belongs, which extension points to use, and how
to test and review it. The design workflow, engineering rules, agent working
rules, and PR hygiene that both entry skills share are in
`references/engineering-rules.md`.

## Hard Rules

The job is a working feature: rules 1 and 2 come first, and the rest keep
the result correct and reviewable. `references/engineering-rules.md` and
`references/design-principles.md` explain each one.

1. Deliver the feature end to end first. Wire the whole path
   (configuration, KV cache manager V2, attention metadata, executor
   callbacks, outputs) and show it running on a real model in the
   production configuration before polishing anything.
2. Make it work on every model family KV cache manager V2 serves (dense,
   multi-head latent attention (MLA), sparse-attention, sliding-window and
   hybrid, hybrid recurrent, and multimodal models, and speculative-decoding
   draft layers) and on every GPU architecture it claims, such as Hopper
   (H100, H200) and Blackwell (B200, GB200, B300, GB300). A family or
   architecture that cannot be supported yet is rejected at configuration
   time and listed as a gap, never silently skipped.
3. Read the required context below before designing or editing, and build
   only on current paths (KV cache manager V2, the Python cache
   transceiver, `tensorrt_llm/_torch/attention/`), never on deprecated ones.
4. Change the code where the behavior belongs. Use an existing mechanism
   when it fits; when it does not, extend it or write the new code. Do not
   add a flag, a wrapper, or a parallel path to avoid touching the right
   file.
5. Write direct code: no defensive try/except, no fallback that hides an
   error or changes results, no checks for states that cannot happen, no
   options nobody asked for. A counted fallback that gives the same results
   under a resource limit, such as per-request work when an optional batch
   workspace does not fit in memory, is fine. Reject unsupported
   combinations at configuration time.
6. With the feature off, outputs, memory, speed, errors, and API stay
   identical to main.
7. Write few, strong tests: each one runs the real path and fails if the
   feature breaks. Never edit, skip, or loosen a test to make it pass; fix
   the code, or stop and report the conflict. When an approved behavior
   change makes an expectation obsolete, update the test and say why in the
   PR. Keep pytest small, with few GPU tests; check lossy numerics and
   end-to-end behavior locally on real runs, and write the commands and
   results in the PR description.
8. Decide and test every compatibility axis, including block reuse, CUDA
   graphs, the overlap scheduler, disaggregated serving, speculative
   decoding, and parallelism (`../trtllm-kv-cache-compat-testing/SKILL.md`),
   and measure performance end to end in the production configuration
   (`../perf-kv-cache-tuning/SKILL.md`).
9. Never remove a feature, mode, model support, or optimization to make
   code smaller, and never change an experiment's settings to make a result
   pass.
10. Give each concept one full name and use it everywhere: identifiers, test
    names, comments, docs, PR text, and replies to people. Write
    "mixed-precision high window", never "window"; "FP8 KV cache", never
    "FP8" or "KV". Abbreviate only with common knowledge (LLM, GPU, CUDA,
    `attn`) or after writing "full name (ABBR)" at first use.
11. Say "done", "passed", or "pushed" only with evidence you can show.

## Route the Task

Paths are relative to this file.

| Task | Read next |
|---|---|
| The mixed-precision KV cache (`mixed_nvfp4_*` dtypes, first/last N tokens) | `../trtllm-kv-cache-mixed-precision/SKILL.md` |
| A KV cache compression method, or the KV cache compression framework itself (iteration-driven or storage-bound, including cold-page compression) | `../trtllm-kv-cache-compression/SKILL.md` |
| Whether a change works with reuse, CUDA graphs, the overlap scheduler, disaggregated serving, speculative decoding, or parallelism; test design | `../trtllm-kv-cache-compat-testing/SKILL.md` |
| Throughput, latency, memory, profiling, tuning | `../perf-kv-cache-tuning/SKILL.md` |
| Accuracy of a lossy or state-changing feature | `../trtllm-kv-cache-accuracy-eval/SKILL.md` |
| Kernels for KV cache manager V2 data movement or for a feature built on it | `../trtllm-kv-cache-compression/references/adding-kernels.md` |
| A sparse attention algorithm that selects or skips KV cache entries inside attention (`SparseAttentionConfig`) | `docs/source/developer-guide/sparse-attention-development-guide.md` in the repository, then the compatibility, performance, and accuracy skills above |

## Required Context

Read before proposing a design or editing code:

1. `CODING_GUIDELINES.md` and the root `AGENTS.md`.
2. `docs/source/features/kvcache.md`: manager selection, reuse, partial
   reuse, retention, mixed precision, offloading.
3. `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md` (the KV
   cache manager V2 C++ guide), in full: ownership, invariants, the
   concurrency model, and high-risk areas. When you touch the batch, the
   sparse cache, or index conversion, also read `SPARSE_ATTENTION_DESIGN.md`
   and `INDEX_CONVERSION_DESIGN.md` next to it.
4. `tensorrt_llm/runtime/kv_cache_manager_v2/AGENTS.md`: test commands, the
   rebuild steps after a C++ change, and the gotchas (string salts,
   dataclass-field grafts, the cross-language block-key format).
5. The wrapper (`kv_cache_manager_v2.py`) and the subclass you touch.
6. `tensorrt_llm/_torch/attention/ATTENTION_DEVELOPER_GUIDE.md`, at least
   "KV-cache and decode-time semantics", "Evaluating New Attention", and
   "Anti-Patterns".
7. `docs/source/features/feature-combination-matrix.md`.
8. The docs of the area you touch:
   - disaggregated serving: `docs/source/features/disagg-serving.md` and
     `docs/source/developer-guide/kv-transfer.md`;
   - sparse attention: `docs/source/features/sparse-attention.md` and
     `docs/source/developer-guide/sparse-attention-development-guide.md`;
   - the cold-page codec interface:
     `docs/source/developer-guide/kv-cache-cold-page-codec.md`;
   - KV cache compression methods: start from
     `../trtllm-kv-cache-compression/SKILL.md` instead.
9. The target code, its tests, and its recent history (`git log -- <files>`):
   who changed it, and why.

When docs and code disagree, the code and its tests win. Say so in the change,
and either fix the doc in the same PR or report it as a follow-up. Known
stale spots to read past:
- some docs still mention a pure-Python KV cache manager V2 backend, or ask
  for tests on "both backends". There is one backend: the C++ native core;
- the sparse-attention development guide still offers KV cache manager V1
  for auxiliary pools. Put side buffers on KV cache manager V2 through
  `_extra_buffers_per_layer` instead;
- the root `AGENTS.md` lists NIXL, UCX, or MPI for KV cache exchange, and
  `kv-transfer.md` says a V1 context worker can talk to a V2 generation
  worker. Both predate the deprecation: the current transceiver uses NIXL,
  with KV cache manager V2 on both sides;
- the opening sections of `kvcache.md` ("The Basics", "Reuse Across
  Requests") describe KV cache manager V1 pools. KV cache manager V2 sizes
  one pool per layer group, from `pool_ratio` or `avg_seq_len`;
- the "Multi-backend Support" overview in `disagg-serving.md` describes the
  deprecated C++ transceiver. Use its transceiver configuration section
  instead;
- the cold-page codec design's section on the KV cache manager refactoring
  it requires (titled "Required KVCM Refactoring") is already implemented.
  Read it as invariants, not as open work;
- the note in `tensorrt_llm/runtime/kv_cache_manager_v2/AGENTS.md` that
  streaming KV cache events have no implementation predates the native
  streaming path. The "KV Cache Events" section of `kvcache.md` is current;
- line anchors in GitHub links can drift. Locate symbols by name.

## Build on Current Paths Only

| Build on | Do not build on (deprecated, removed, or legacy) |
|---|---|
| KV cache manager V2 (`use_kv_cache_manager_v2: true`, the default) | KV cache manager V1. It is deprecated and scheduled for removal, together with its V1-based subclasses, such as the hybrid managers in `kv_cache/mamba_cache_manager/legacy.py`. A few features still depend on V1 while they gain V2 support; new features target V2 and do not reuse V1-specific code paths |
| The Python cache transceiver (`transceiver_runtime: PYTHON`) on the NIXL backend, with KV cache manager V2 on both workers | The deprecated C++ cache transceiver, which is selected by `transceiver_runtime: CPP`, by an explicit null, or by `auto` when it falls back. Also the `UCX`, `MPI`, and `MOONCAKE` transceiver backends, `backend: DEFAULT`, and the `TRTLLM_USE_*_KVCACHE` backend selectors. New features target the Python transceiver |
| Explicit settings in recipes and tests (`use_kv_cache_manager_v2: true`, `transceiver_runtime: PYTHON`) | `auto` values that can resolve to a deprecated path, and example or curated configurations that pin `use_kv_cache_manager_v2: false` |
| Current speculative-decoding configurations: multi-token prediction (MTP), EAGLE3, and the other one-model modes | The deprecated `NGramDecodingConfig` and `UserProvidedDecodingConfig` |
| The KV cache manager V2 native core (C++), driven through `tensorrt_llm/_torch/pyexecutor/kv_cache/` | A pure-Python KV cache manager V2 core. Main has none, and no code reads the `TLLM_KV_CACHE_MANAGER_V2_BACKEND` variable that some docs and tests still mention |
| `tensorrt_llm/_torch/attention/` (`backends/`, `attention.py`, `ATTENTION_DEVELOPER_GUIDE.md`) | The deprecated compatibility modules `tensorrt_llm._torch.attention_backend` and `tensorrt_llm._torch.modules.attention`, which only re-export the new location |
| `tensorrt_llm/_torch/pyexecutor/kv_cache/` and `tensorrt_llm/_torch/disaggregation/` | Flat module paths from before the KV cache managers and the transceiver were regrouped |
| The KV cache compression framework (`KvCacheCompressionConfig`) for work between steps or at tier boundaries | Sparse-attention hooks for KV cache reduction that does not change the attention computation |
| Current field names: `cold_tier_offload_min_priority`, `block_reuse_config.policy`, `block_reuse_config.enable_branch_snapshot`, `mamba_state_config.periodic_snapshot_interval` | Old names: `secondary_offload_min_priority` (alias of `cold_tier_offload_min_priority`); `block_reuse_config.block_reuse_policy` (rejected; renamed to `policy`); `mamba_state_config.enable_branch_snapshot` (alias of `block_reuse_config.enable_branch_snapshot`); `mamba_state_cache_interval` (alias of `mamba_state_config.periodic_snapshot_interval`); `sink_token_length` (ignored on the PyTorch backend) |
| `tests/unittest/_torch/pyexecutor/` for executor tests, `tests/unittest/_torch/pyexecutor/kv_cache/` for the KV cache manager V2 wrapper, `tests/unittest/kv_cache_manager_v2_tests/` for KV cache manager V2 core behavior | `tests/unittest/_torch/executor/`, the name from before the executor tests were renamed |

Before relying on any path, confirm it on the current main. Search open pull
requests for planned removals or renames of the files you will touch.

## Owner Map

| Layer | Location | Owns | Must not |
|---|---|---|---|
| User configuration | `tensorrt_llm/llmapi/llm_args.py` (`KvCacheConfig`, `KvCacheCompressionConfig` subclasses, `SparseAttentionConfig`) | Settings, defaults, validation, rejection of unsupported combinations | Hold runtime state; switch features off silently |
| Executor construction | `tensorrt_llm/_torch/pyexecutor/_util.py`, `py_executor_creator.py` | Building managers in the right order, compatibility admission, KV cache memory budget estimation | Contain method-specific runtime logic |
| Executor loop and resource managers | `tensorrt_llm/_torch/pyexecutor/py_executor.py`, `resource_manager.py` | Request lifecycle, scheduling, the order of resource callbacks | Know a KV layout or a method's name |
| KV cache manager V2, executor side | `tensorrt_llm/_torch/pyexecutor/kv_cache/` (`kv_cache_manager_v2.py`, `mixed_precision_staging.py`, `mamba_cache_manager/mamba_cache_manager_v2.py` for hybrid models), plus model- and backend-specific subclasses under `tensorrt_llm/_torch/attention/backends/` that declare side buffers through `_extra_buffers_per_layer` | Mapping layers to layer groups, pools, life cycles, priorities and reuse policies; extra buffers; the page tables that the mixed-precision staging copy reads (the mixed-precision staging copy itself belongs to the attention metadata) | Make algorithm decisions; branch on one method |
| KV cache manager V2 native core | `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`, surfaced by `tensorrt_llm/runtime/kv_cache_manager_v2/` | Pages, slots, pools, the radix tree, eviction, migration, events, publication, release | Know models or methods |
| Compression methods | `tensorrt_llm/_torch/kv_cache_compression/`, plus the KV cache compression manager base class (`KVCacheCompressionManager`) in `resource_manager.py` | Cadence, request-local state, selection, transforms, method kernels | Allocate or publish pages; release capacity except through the KV cache manager V2 API after publishing completion |
| Attention | `tensorrt_llm/_torch/attention/` (backends, metadata) | Consuming the published active representation; per-step metadata; declaring backend side buffers through the KV cache manager V2 hooks | Own storage policy beyond those declarations; grow per-feature kernel branches |
| Disaggregated transfer | `tensorrt_llm/_torch/disaggregation/` (Python transceiver, NIXL backend) | Moving published state between context and generation workers; resolving layout and transfer ids on both sides | Assume layouts the sender did not publish; target the deprecated C++ transceiver |


## How It Is Used

Users configure KV cache manager V2 through `kv_cache_config`:
- `use_kv_cache_manager_v2: true` (the default);
- pool size: `free_gpu_memory_fraction`, `max_tokens`, `max_gpu_total_bytes`;
- `tokens_per_block`;
- reuse: `enable_block_reuse`, `enable_partial_reuse`, `block_reuse_config`;
- layer-group sizing: `pool_ratio`, `avg_seq_len`, `enable_kv_pool_rebalance`;
- the host tier: `host_cache_size`, `cold_tier_offload_min_priority`.

`docs/source/features/kvcache.md` documents them, and `perf-kv-cache-tuning`
(`../perf-kv-cache-tuning/SKILL.md`) lists their performance effects. Check
what a run actually built:
- `isinstance(manager, KVCacheManagerV2)` for the manager class;
- the startup log for the layer groups, pool sizes, and attention window
  sizes;
- `LLM.get_stats()` and `LLM.get_kv_cache_events()` at runtime.

## Where a Change Belongs

Prefer the highest row that can express the change.

| Need | Layer | Extension point on main |
|---|---|---|
| A model or backend needs extra per-layer buffers | Manager subclass | `_extra_buffers_per_layer()`; `get_disagg_role_mapper_kinds()` so disaggregated serving transfers them |
| Which layers share a pool, a sliding window, sinks, or a reuse policy | Wrapper configuration build | `_build_cache_config()` in a subclass; `reuse_policy` on the layer specification; `_add_mixed_precision_layers()` is the worked example |
| Which reusable layer groups a request claims at resume | Wrapper | A `_resume_context_cache()` override that passes `optional_reuse_groups` for layer groups whose reuse status is complete. `CSA2CacheManager` is the example |
| When blocks are committed for reuse | Configuration and wrapper | `block_reuse_config.policy` (`BlockReusePolicy`); commit timing in `update_context_resources()` |
| Which pages survive pressure | Wrapper callbacks and configuration | Priority callbacks (for example `swa_endpoint_rewind_tokens`); `cold_tier_offload_min_priority` |
| Transfer ids and geometry for disaggregated serving | Wrapper | The optional hooks `get_disagg_global_layer_ids` and `get_disagg_role_mapper_kinds`, read by the KV cache extractor (`kv_extractor.py`) |
| A new reuse policy, eviction rule, tier, migration step, or hash | Native core | Follow the high-risk rules of the KV cache manager V2 C++ guide, below |

A method or feature never subclasses KV cache manager V2 to hold algorithm
state. It holds a manager and uses its API. Subclass the wrapper only for a
model or backend whose cache layout differs.

## Contracts to Keep

The KV cache manager V2 C++ guide has the full list. The ones changes break
most often:

- **Lengths.** History never decreases, and capacity is at least history.
  `resize(capacity, None)` changes capacity and leaves history alone.
  Rejected draft slots are reclaimed by shrinking capacity. KV cache manager
  V2 has no `rewind_kv_cache`.
- **Commit.** Pages are immutable after commit, and `commit(is_end=True)` is
  terminal. Events describe complete, published blocks.
- **Matching.** Every `REQUIRED` life cycle must cover a match at its
  endpoint. `SINK_REQUIRED_ONLY` needs a sliding window with sinks and
  applies to attention layers only. Coupled buffers that share a page index
  share a reuse policy.
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

1. **Native core.** Put C++ in
   `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/` and add sources to
   its `CMakeLists.txt`. Follow the design principles of the KV cache manager
   V2 C++ guide: composition, typed indices, `TLLM_CHECK*`, and no Python
   dependency in the native core.
2. **Bindings.** Expose the new surface in
   `cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp`, keeping
   exception types and global interpreter lock (GIL) rules as the KV cache
   manager V2 C++ guide describes.
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

1. **Native core**: the enum and its comment; the life-cycle predicates and
   validation; match pruning; the subtree eviction rule; the claim, resume,
   and commit paths in the per-request cache; storage. Update the KV cache
   manager V2 C++ guide, which should list every policy.
2. **Bindings and stubs**: bind the value, regenerate or edit the `.pyi`
   stub, and rebuild every consumer.
3. **Python selection**: set the policy per layer or buffer when the
   configuration is built. Select optional layer groups at resume. Fill
   unselected pages, by recomputation or another source, before attention
   reads them.
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
  directly with `PYTHONPATH` set to `tensorrt_llm/runtime/`, as the Python
  surface's `AGENTS.md` shows. Before final validation, also run the
  production import path.
- **Targeted suites**:
  - `test_kv_cache_event_manager.py` after event changes;
  - `test_kv_cache_salting.py` after hashing or reuse-scope changes;
  - `test_kv_cache_stats_api.py` and `test_kv_cache_stats_behavior.py` after
    allocation or accounting changes;
  - `test_optional_checkpoint.py` after reuse-policy changes.
- **C++.** The `kvCacheManagerV2*` gtests in
  `cpp/tests/unit_tests/batch_manager/`, after rebuilding the bindings. The
  `radixBlockTreeTest.cpp` and `kvCacheManagerTest.cpp` suites in the same
  directory test KV cache manager V1. Use `TLLM_DEBUG_MODE=1` when
  diagnosing an invariant failure.
- **Wrapper.** `tests/unittest/_torch/pyexecutor/kv_cache/`, including
  `test_kvcm2_integration.py`, for executor-side changes.
- **Debug fills.** Run affected end-to-end paths with
  `TRTLLM_KV_FRESH_PAGE_FILL=nan` and `TRTLLM_KV_GUARD_PAGE` to catch reads of
  pages that were never written.
- **Pressure.** After any change to capacity, history, resize, suspend, or
  resume, run a high-pressure serving workload locally with the overlap
  scheduler on, in which requests are suspended as eviction victims. Check
  that no tail page entry is `-1`, and write the command and result in the
  PR description. In pytest, cover the same invariant with a small
  manager-level test of resize, suspend, and resume at a page boundary.
- Then run the feature-level gates in `trtllm-kv-cache-compat-testing`
  (`../trtllm-kv-cache-compat-testing/SKILL.md`).

## Review Checklist

```markdown
## KV cache manager V2 review
- Layer of the change (subclass hook / executor-side wrapper / native core) and why not higher: <...>
- High-risk areas of the KV cache manager V2 C++ guide touched: <list, or none>
- Contracts at risk (lengths, commit, matching, identity, transactions, bindings): <...>
- Feature-off invariance: allocation, pages, stats, events, outputs: <how checked>
- Tests: fast mode + production import, targeted suites, C++ gtests, debug fills, pressure run: <...>
- Docs: KV cache manager V2 C++ guide, Python surface guide, kvcache.md updated where behavior changed: <...>
```

## References

- `references/engineering-rules.md`: the design workflow and design gate,
  engineering rules, what machines check, agent working rules, PR hygiene,
  and the review output format. Shared with `trtllm-kv-cache-compression`.
- `references/design-principles.md`: the review bar, with rationale.
- `references/lessons-learned.md`: recurring review findings and the rules
  they produced, by category.
- `references/kv-cache-architecture.md`: a map of the stack, with the
  invariants changes most often break.
- `../trtllm-kv-cache-compression/references/adding-kernels.md`: choosing,
  writing, wiring, and testing kernels.
