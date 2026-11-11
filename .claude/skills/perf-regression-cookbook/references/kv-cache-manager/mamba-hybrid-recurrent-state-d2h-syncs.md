---
id: case-mamba-hybrid-recurrent-state-d2h-syncs
type: regression-case
family: execution-and-graph
module: kv-cache-manager
maturity: full
regression_class: [sync-introduced, host-work-added]
signals: [throughput-drop, itl-increase, gpu-idle-between-steps, host-time-increase, perf-ci-bar-failure]
subsystems: [kv-cache, cuda-graph, runtime-cpp]
introduced_via: [new-feature]
phase: [decode]
patterns: [pattern-per-step-sync-added, pattern-host-work-on-hot-path]
nvbugs: ["6176224", "6175923", "6144334"]
commits: ["79ede08f31bb"]
success_prs: [14003]
failed_prs: []
---

# Mamba-hybrid prefix caching added per-slot D2H syncs to the decode prep path

> Part of the [KV-cache manager regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbugs `6176224` / `6175923` / `6144334` · commit
  `79ede08f31bb` · PR #14003 — "[None][fix] Fix CppMambaHybridCacheManager
  functional and perf issues". The PR carries no NVBug tag. Nvbugs
  `6175923` / `6144334` are also cited by PR #14612, which fixes an unrelated
  gpt_oss_20b measurement artifact recorded as
  [perf-test gpt-oss-20b MoE backend pin](../measurement-and-test/perf-test-gpt-oss-20b-moe-backend-pin.md);
  do not treat #14612 as this defect's fix, or this PR as that one's.
- **Symptom:** decode-prep host stalls on mamba-hybrid models with prefix
  caching. PR #14003's perf evidence is an nsys capture on
  Qwen3.5-A17B-NVFP4 + MTP Eagle one-model + CUDA graphs showing
  `numGenReq × (1 + max_draft_len)` `cudaMemcpyAsync` +
  `cudaStreamSynchronize` pairs per iteration inside `_prepare_inputs`, plus a
  `refresh_blocks` stream sync at the tail of `prepare_resources`. The PR
  states no end-to-end number.
- **Root cause:** three host/device sync points on the mamba-hybrid prep
  path. (a) `Mamba2Metadata.prepare` did
  `for i, idx in enumerate(indices): state_indices_cpu[i] = idx`, where
  `indices` is `CppMambaHybridCacheManager.cuda_state_indices` — a CUDA
  tensor. Iterating it yields 0-d CUDA slices, so each assignment into the
  CPU tensor issues its own `cudaMemcpyAsync` + `cudaStreamSynchronize`:
  one blocking round-trip per batch slot per iteration. (b)
  `KVCacheTransferManager::copyBlock` issued one `cudaMemcpyAsync` per
  layer for the layer-first pool layout `{numLayers, numBlocks, kvFactor,
  blockSize}`. (c) `refresh_blocks()` (`syncTransfers`) ran unconditionally
  at the tail of `_prepare_resources`, blocking the remaining prep work
  even when no transfer had been scheduled.
- **How introduced:** a new feature added the cost; the fix PR names no
  culprit commit.
- **Fix mechanism:** alias instead of copy, batch instead of loop, defer
  instead of block. `Mamba2Metadata.state_indices` takes a direct
  reference (`self.state_indices = indices`) when the source is on CUDA,
  guarded by a `data_ptr()` invariant assert so a future buffer
  reallocation cannot silently break CUDA-graph replays (CPU/list paths
  keep the original copy). The per-layer memcpy loop becomes a single
  pitched `cudaMemcpy2DAsync` — for a fixed block index the per-layer
  slices are equal-length rows at stride `numBlocks * rowBytes`. And
  `_prepare_resources` splits into an async onboard-issue phase plus a new
  `flush_state_transfers()` that calls `refresh_blocks()` only when a
  transfer was actually scheduled, invoked at the end of
  `Mamba2Metadata.prepare()` so the rest of `_prepare_tp_inputs` overlaps
  the in-flight onboards; `KVCacheManager::copyLinearAttentionBlock` now
  returns `bool` through all three layers (`KVCacheManager` →
  `BlockManager` → `WindowBlockManager`) to support that skip. The same PR
  also fixes two *functional* issues in the manager — invalid state on PP
  ranks with zero local mamba layers, and recurrent-state slot
  under-reservation (`+1` for the CUDA-graph padding sentinel and
  `+ spec_config.max_draft_tokens` for the draft-len sentinels) that
  surfaced under load as block-allocation failures — which are correctness,
  not the measured regression.
- **Detection signal:** in nsys, `cudaMemcpyAsync` + `cudaStreamSynchronize`
  pairs inside `_prepare_inputs` whose **count scales with generation batch
  size × (1 + max_draft_len)** — a per-slot, not per-step, sync. Statically,
  `grep -n "state_indices" tensorrt_llm/_torch/modules/mamba/mamba2_metadata.py`:
  any per-element assignment out of a CUDA tensor into a CPU tensor is one
  sync per element, and the loop reads as cheap host bookkeeping. The
  deferred sync is now marked by the `hybrid_flush_state_transfers` nvtx
  range, so its position relative to the forward is directly visible.
- **Prevention/guard:** the fix adds the `data_ptr()` invariant assert on
  the aliased buffer and unit tests at
  `tests/unittest/_torch/executor/test_mamba_cache_manager.py` — but those
  cover the slot-reservation logic, not the absence of syncs. Gap: nothing
  fails when a per-element D2H creeps back onto the prep path; a per-step
  sync budget (a counter asserted in a unit test, or an nsys-derived
  per-iteration sync count in perf CI) is what would catch the next one.
- **Generalizes to:** `pattern-per-step-sync-added` and
  `pattern-host-work-on-hot-path` — a correctness-motivated feature paid
  for in blocking round-trips on the step path. Carries to: iterating a
  CUDA tensor in Python anywhere on the hot path (every element is a
  hidden D2H sync); copying a device-side index buffer to host when it
  could be aliased; an unconditional stream sync at the tail of
  resource-prep that can be deferred past the host work or skipped when
  nothing was scheduled; per-layer memcpy loops over a layer-strided pool
  that one 2D copy covers; and the sibling case
  [dsa-indexer-host-overhead](../attention-fmha/dsa-indexer-host-overhead.md), where a debug
  assert's `.all()` forced the very same per-step D2H.
