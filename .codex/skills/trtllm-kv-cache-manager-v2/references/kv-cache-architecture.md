<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Stack Map Around KV Cache Manager V2

This file is a map of the PyTorch-backend stack around KV cache manager V2:
the executor, attention metadata, the KV cache compression framework, and
disaggregated transfer. It also lists the invariants changes most often
break. The authoritative details live in:
- the KV cache manager V2 C++ guide,
  `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md`;
- `tensorrt_llm/runtime/kv_cache_manager_v2/AGENTS.md`;
- `docs/source/features/kvcache.md`;
- the attention developer guide.

Symbols are named rather than given line numbers. Confirm them on the current
main.

## 1. Which KV Cache Manager Runs

- **KV cache manager V2 is the default** (`use_kv_cache_manager_v2: true`).
  Build every new KV-cache feature on KV cache manager V2.
- **KV cache manager V1** (`use_kv_cache_manager_v2: false`, or `auto`
  resolving to KV cache manager V1) is deprecated and scheduled for removal,
  together with its subclasses. Examples of those subclasses are the hybrid
  managers in `kv_cache/mamba_cache_manager/legacy.py`, such as
  `MixedMambaHybridCacheManager` and `CppMambaHybridCacheManager`. The hybrid
  manager built on KV cache manager V2 is `MambaHybridCacheManagerV2`.

  Do not add features to KV cache manager V1, and do not design around
  behavior that only KV cache manager V1 has. A few features still depend on
  KV cache manager V1 until their KV cache manager V2 support lands. The
  "Migration prerequisites" in `docs/source/release-notes.md` and the
  "Selecting the KV Cache Manager" section of `docs/source/features/kvcache.md`
  list them. New features target KV cache manager V2 and do not reuse code
  paths specific to KV cache manager V1. Do not write `auto` in recipes.
- **How the class is chosen.** The executor's KV cache creator resolves the
  KV cache manager class from the model, its sparse-attention algorithm, and
  the configuration. Sparse algorithms route through
  `tensorrt_llm/_torch/attention/backends/sparse/registry.py`: some always use
  KV cache manager V2, some follow `use_kv_cache_manager_v2`, and some still
  use KV cache manager V1. Do not trust a doc table or the flag alone. Check
  the result with `isinstance(manager, KVCacheManagerV2)`. KV cache manager V2
  is not a subclass of the KV cache manager V1 class `KVCacheManager`.
- **Do not build on old notes.** Any note or document describing a
  pure-Python implementation of KV cache manager V2, or an environment
  variable that selects it, is obsolete. The KV cache manager V2 native core
  is C++.

## 2. KV Cache Manager V2 Layers

| Layer | Location | Role |
|---|---|---|
| Executor-side wrapper | `tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py` (`KVCacheManagerV2`) | Builds the KV cache manager V2 native core configuration from the model: layers, layer groups, buffers, attention windows, sinks, reuse policies, pool ratios, extra buffers. Adapts executor requests, keeps the pinned host page-index tensor, and copies block offsets to the device. Subclasses exist for hybrid recurrent and sparse models |
| Python surface | `tensorrt_llm/runtime/kv_cache_manager_v2/` | Re-exports the nanobind bindings, plus introspection helpers for tests |
| Native core | `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/` | Life-cycle registry, block radix tree (SHA-256 block keys), per-request caches, storage manager and cold tiers, eviction controller, copy engine, events |
| Bindings | `cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.cpp` | Type, exception and lifetime adaptation |

## 3. Concepts

- **Layer group (life cycle).** Buffers that share an attention window size,
  a sink count, and a reuse policy form one life cycle, or layer group. A
  layer's buffers can land in different layer groups through a per-buffer
  reuse-policy override, and several life cycles can share one physical pool
  group. The grouping decides pool sizing (`pool_ratio` has one entry per
  layer group) and the prefix-match rules.
- **Reuse policy per life cycle** (`ReusePolicy`):

  | Policy | Effect |
  |---|---|
  | `REQUIRED` | Constrains prefix matching |
  | `PRIVATE` | Never published |
  | `OPTIONAL` | Committed, but selected only at resume |
  | `SINK_REQUIRED_ONLY` | Only the sinks constrain the match |

  It is separate from `block_reuse_config.policy` (`all_reusable`,
  `per_request`, `per_conversation`), which sets **when** blocks are
  committed.
- **Partial reuse.** KV cache manager V2 always copies a partially matched
  last block into a private page before appending. `copy_on_partial_reuse` is
  read only by KV cache manager V1. Some KV cache managers force partial reuse
  off, so check the effective setting of the KV cache manager in use, not the
  configuration value.
- **Block keys and scope.** Keys are SHA-256 digests over the reuse scope
  (low-rank adaptation (LoRA) id, salt) and the tokens. They are a security
  boundary for cross-request reuse; never weaken them. Routers depend on
  their exact format.
- **Priorities and tiers.** Eviction is least recently used (LRU) within a
  priority. Pages at or above `cold_tier_offload_min_priority` can move to
  host memory or disk, where a cold-page codec may encode them;
  lower-priority pages are dropped.

## 4. Request Flow

```text
prepare_context      match the prompt minus its last token (plus draft look-ahead evidence) in the radix tree;
                     prune the match until every REQUIRED life cycle covers it at the endpoint;
                     claim matched pages; copy a partial last block into a private page; settle the cursor
context steps        commit per block_reuse_config.policy (every chunk, or only at the end / branch points)
generation steps     resize(capacity, history): capacity drops by rewound draft slots, history advances;
                     generation blocks commit only under all_reusable and when no feature manages history
finish / abort       close the request cache; committed pages stay in the tree until evicted
```

## 5. Length Invariants

- `history <= capacity`. History never decreases: the KV cache manager V2
  native core rejects a smaller history. Capacity may shrink.
- KV cache manager V2 has no `rewind_kv_cache`. Rejected draft slots are
  reclaimed by shrinking capacity during the generation update. Under the
  overlap scheduler, the rewind uses the width the in-flight step was
  allocated with.
- The generation update sets history to `max_beam_num_tokens - 1`; the newest
  sampled token has no KV cache entry yet. It leaves history unchanged when a
  compression method manages history (`kv_compression_manages_history`; see
  `trtllm-kv-cache-compression`
  (`../../trtllm-kv-cache-compression/SKILL.md`)) or context-parallel Helix
  is active.
- `commit(is_end=True)` is terminal. Writing to committed memory afterwards is
  invalid. Pages are immutable after commit.

## 6. Page Tables

- Each request cache writes page indices, without copying, into the
  executor-side wrapper's pinned host tensor `host_kv_cache_block_offsets`.
  Slots come from the index mapper, and device copies are made per batch.
- Sparse backends such as DeepSeek sparse attention (DSA) and DeepSeek-V4
  build their page tables from the same pinned host tensor and copy them per
  batch. The native `Batch` (slot rows that keep captured addresses valid)
  exists in the KV cache manager V2 native core and its bindings, but the
  executor-side wrapper does not build one on main. Do not design against it
  until a sparse fetch path uses it.
- Feature-specific tables, such as mixed-precision high pages, are gathered
  from the same host tensor. Never cache raw page pointers across steps.

## 7. Executor Interplay

- Resource managers run in the resource manager's dictionary order for
  `prepare_resources` and `update_resources`, and in **reverse** order for
  `free_resources`. The executor builder places the KV cache compression
  manager after KV cache manager V2 and the cross-attention KV cache
  manager V2. In every executor loop, the target KV cache manager V2 is
  updated by a separate call before the other managers' update. A new
  manager that must run after KV cache manager V2 needs the same placement.
- In the overlap-scheduler loop, iteration N+1's `prepare_resources` runs
  before iteration N's `update_resources`. A feature that mutates KV cache
  state at step end must account for the already-prepared next batch.
- KV cache manager V2 is updated before other managers in each executor
  loop. A step-end feature therefore sees the step's final capacity and
  rewinds.
- Attention metadata `prepare()` runs on the host every step, also with CUDA
  graphs. On steady one-model multi-token prediction (MTP) decode steps,
  metadata that supports it runs `prepare_steady_decode_step()` instead. Keep
  this work proportional to what changed.

## 8. CUDA-Graph Contract for KV-Cache Features

- Device buffers that captured kernels read must keep their addresses. Get
  them from the graph-aware buffer allocator, and update their contents in
  place.
- Shapes and launch bounds are frozen at capture. Read the actual lengths
  from device memory, and size the work to a static maximum.
- When device KV cache lengths change after `prepare()` (overlap-scheduler
  corrections, speculative decoding), refresh dependent buffers in place
  through the existing length-update path.
- Compile every kernel variant before capture, including just-in-time (JIT)
  compiled ones.

## 9. Disaggregated Serving

- **Current path: the Python transceiver** (`transceiver_runtime: PYTHON`)
  with the NIXL backend and KV cache manager V2 on both workers. The C++
  cache transceiver is deprecated. New features target the Python
  transceiver only.
- The Python transceiver and the KV cache region extractor in
  `tensorrt_llm/_torch/disaggregation/` resolve per-layer geometry and global
  layer ids on both sides. A feature that adds layers, pools, or buffers
  must:
  - define their global transfer ids, collision-free and equal on every
    pipeline rank;
  - give their KV-head counts and dtypes;
  - prove that the receiver resolves them.
- Both workers must run the same KV-cache feature configuration. Reject
  mismatches.

## 10. Invariants That Changes Break Most Often

- Feature-off runs must stay identical to main.
- History monotonic; capacity at least history; publish before shrinking.
- Pages are immutable after commit; events describe complete blocks.
- Block-key bytes and token encoding stay compatible across languages and
  with routers.
- Every REQUIRED life cycle must cover a match. A new retained copy with
  `REQUIRED` reuse shortens matches when its pages are evicted.
- Free order: hooks run before KV cache manager V2 frees blocks. A feature
  hook never releases a request's blocks; methods return trailing pages only
  by shrinking capacity through `resize`.
- Typed indices in the KV cache manager V2 native core (layer group, pool
  group, block ordinal, slot) are never mixed.
- Coupled buffers that share a page index share a reuse policy.
- Changes inside KV cache manager V2 follow its transaction order: prepare,
  lock, submit, fence, commit, with rollback on failure. Publish mappings,
  statistics, and events only for work that succeeded.
- Size per-sequence page-index tables from the requested maximum sequence
  length plus speculative slack. Host-memory and disk tiers let a sequence be
  longer than what fits on the GPU at once.
