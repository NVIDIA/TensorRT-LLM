---
id: case-auxiliary-cache-in-kv-manager
type: case
family: runtime-execution
maturity: full
bottleneck: [memory]
signals: [memory-capacity-bound, long-context]
architectures: [any-sm]
model_scope: [model-agnostic, sparse-attention, mla, deepseek-v32]
phase: [any-phase]
patterns: [pattern-fold-side-cache-into-kv-manager]
accuracy_risk: lossless
apply_via_kind: [code-change, config-knob]
knobs: [kv_cache_config.enable_block_reuse, host_cache_size]
specialists: [kernel-cuda-specialist, trtllm-serve-config-guide]
commits: ['ae6875fe10', '356a52edf5', '402a056ac6', '6254f3a161', 'c37924f37b']
interactions:
  - {case: case-mla-kv-cache-reuse, relation: composes-with, note: "#9383 extends that reuse pattern to the side cache"}
  - {case: case-move-bookkeeping-into-cpp-op, relation: composes-with, note: the scatter op there writes into this pool}
measured: []
---

# Fold a model-specific auxiliary cache into the unified paged KV-cache manager

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `ae6875fe10` [feat] Move indexer-k-cache to KVCacheManager (#8699);
  related: `356a52edf5` KV-cache reuse for DSv32 (#9383), `402a056ac6` host cache
  offload for DSA (#12010), `6254f3a161` count indexer K-cache as UINT8 in the size
  estimate (#15088), `c37924f37b` clear indexer-k-cache refs before release (#9110).
- **Applies when:** a model needs an **auxiliary/side cache** alongside the main KV
  cache — DSA keeps a separate small FP8/UINT8 **indexer K cache** for the lightning
  indexer; also block-scale/scale-factor pools, sparse-attention landmark caches, any
  per-layer compressed side state. Signals: a bespoke second allocator
  (`indexer_k_cache_manager = BlockManager(...)`) with hand-rolled
  `add_dummy_requests` / `prepare_resources` / `update_resources` / `free_resources`
  / `rewind` overrides; the side cache **can't** be reused across prefixes or
  offloaded to host because it isn't paged; a "reuse not supported" guard; its memory
  isn't counted correctly in the auto-sizing estimate.
- **Mechanism:** make the auxiliary cache a **first-class typed pool inside the
  unified paged `KVCacheManager`** (a `containsIndexerKCache` pool in
  `WindowBlockManager`, `createIndexerKCachePools()`, allocated `kUINT8`) that
  **shares block IDs** with the main KV pool. The parallel shadow allocator and its
  overrides are deleted; the manager pages, reuses, offloads, and accounts for the
  side pool alongside the main blocks. That single relocation then unlocks, for free:
  **prefix/block reuse** (drop the "unsupported" guard, teach `BlockRange` to carry
  the side pool via `isEnableIndexerKCache()`, #9383) and **host/secondary offload**
  (#12010). Three corollaries the relocation exposes: (1) a **mixed-dtype** pool set
  (UINT8 side cache beside a BF16/FP8 KV) must be summed **additively per physical
  pool dtype**, never via one multiplicative dtype factor (#15088); (2) once blocks
  are onboarded from a secondary/host pool, **index them by the decoded memory-pool
  index**, not the logical block ID (they diverge — `_get_pool_block_indices()` from
  `host_kv_cache_block_offsets`, #12010); (3) **null the Python tensor views before
  the C++ owner frees** the device memory at teardown (#9110).
- **Generalizes to:** the pattern "**manage an auxiliary/compressed side cache as a
  typed pool inside the unified paged KV-cache manager instead of a bolt-on shadow
  allocator — it inherits paging, block reuse, host/secondary offload, eviction, and
  correct per-pool memory accounting, and deletes duplicated resource-tracking
  code.**" Carries to: scale-factor / block-scale pools, sparse-attention or
  landmark caches, adapter/state caches, any model-specific representation that lives
  as long as the KV blocks. Adapt by: registering the pool type + `createXPools()`,
  sharing block IDs with the main pool, wiring size accounting to the pool's
  *physical* dtype, and carrying the pool through the block-range/transfer machinery
  so reuse/offload include it.
- **Apply via:** **not a single server knob** — a KV-cache-manager code change
  (the pool becomes first-class). The capabilities it unlocks ride existing knobs:
  `kv_cache_config.enable_block_reuse` (reuse), the host-offload config
  (`host_cache_size` / free-fraction). Delegate to **kernel-cuda-specialist** /
  the KVCacheManager owner; expose YAML knobs via **trtllm-serve-config-guide**.
- **Expected effect:** the side cache gains prefix reuse (higher prefill throughput /
  lower TTFT on shared prefixes) and host offload (more/longer sequences fit), plus
  correct auto-sizing (no OOM / no wasted capacity) and less duplicated code.
  Direction only — measured Δ (KV-hit rate, TTFT, max concurrency/seq-len, memory
  headroom) to be recorded from a prefix-heavy / long-context trace.
- **Accuracy risk:** **lossless** — pure storage/bookkeeping relocation. The one
  correctness-sensitive path is **reuse/offload lockstep**: the side pool must be
  reused/offloaded in lockstep with the MLA KV blocks (shared block IDs) and indexed
  by the correct pool index; a mismatch would silently feed wrong indexer state →
  verify with an accuracy eval on a reuse/offload trace.
- **Verify:** correctness — DSA e2e accuracy unchanged with `enable_block_reuse=True`
  and with host offload on (vs off); size-estimate matches measured pool bytes. Perf
  — KV-cache-hit rate + TTFT on a repeated-prefix trace, max concurrency/seq-len with
  offload, memory headroom.
- **Rollback:** `kv_cache_config.enable_block_reuse=False` / disable host offload to
  isolate; the pool-relocation itself is structural. Trigger: accuracy drift on a
  reuse/offload trace, or a size-estimate error causing OOM.
- **Prior art:** PRs #8699, #9383, #12010, #15088, #9110. Files:
  `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` (`createIndexerKCachePools`,
  `containsIndexerKCache`, `isEnableIndexerKCache`), `kvCacheUtils.h` (`BlockRange`),
  `tensorrt_llm/_torch/attention/backends/sparse/dsa.py` (`DSACacheManager`,
  `_get_pool_block_indices`). Owning specialist: **kernel-cuda-specialist**. Related:
  the [MLA KV-cache reuse case](mla-kv-cache-reuse.md) (#9383 is the side-cache
  instance of that pattern) and [move bookkeeping into a C++ op](move-bookkeeping-into-cpp-op.md)
  (the scatter op that writes into this pool).
