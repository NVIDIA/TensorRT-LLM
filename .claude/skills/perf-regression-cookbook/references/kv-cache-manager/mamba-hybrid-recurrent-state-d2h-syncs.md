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

- **Provenance:** nvbug `6176224` · commit `79ede08f31bb` · PR #14003 —
  "[None][fix] Fix CppMambaHybridCacheManager functional and perf issues".
  The PR carries no NVBug tag; the link lives only in nvbug 6176224, which
  records #14003 as the merged fix.
  **Two more bug ids cover the same defect**, because the same QA sweep
  (1.3.0rc13 `b9ce4b69` → 1.3.0rc14 `93cb6518`) also filed two *multi-model*
  bugs whose `nemotron_nano_12b_v2` rows are this regression: nvbug `6175923`
  (gpt_oss_20b and nemotron_nano_12b_v2, closed without verification) and
  nvbug `6144334` (nemotron_nano and gpt_oss models, on H200). Their
  *other* half — the gpt_oss_20b rows — is an unrelated measurement artifact
  fixed by PR #14612, recorded as
  [perf-test gpt-oss-20b MoE backend pin](../measurement-and-test/perf-test-gpt-oss-20b-moe-backend-pin.md).
  So 6175923 has two fix PRs (#14612 and this one) and neither closes it
  alone; do not treat #14612 as this bug's fix, or this PR as that bug's fix.
  The bug records themselves resolve both sweep bugs onto this one: 6144334
  is marked as a duplicate of 6175923, which is marked as a duplicate of
  **6176224**, and 6176224 lists both as its duplicates — so the duplicate
  graph is *affirmative* evidence for listing all three ids here, and it is
  why the mechanical one-more-hop from the gpt_oss case would land on this
  defect. (Read both links from the full bug record; a summary view can show
  them blank.)
  Two caveats when reading those two ids as this defect: 6144334's
  `nemotron_nano_12b_v2` rows are **H200**, not B300 (+5.02% to +22.38%
  Inference Time — the defect is not GPU-specific, the sweep just ran a
  different fleet), and neither sweep bug's own trail ever names #14003 —
  6144334 only points at 6175923 as the same issue, and 6175923 defers its
  nemotron half to the other nemotron bugs. The link from those ids to this
  fix runs through 6176224, not through their own history.
- **Symptom:** NVIDIA-Nemotron-Nano-12B-v2 (a mamba-hybrid model) on B300,
  1.3.0rc14 (`93cb6518`) vs 1.3.0rc13 (`b9ce4b69`); the bug reports
  Inference Time +23.66% / +7.61%, Seq Throughput -19.13% / -7.08% and
  Output Token Time +27.02%, surfaced by the QA perf sweep. PR #14003's own
  perf evidence is an nsys capture on Qwen3.5-A17B-NVFP4 + MTP Eagle
  one-model + CUDA graphs showing `numGenReq × (1 + max_draft_len)`
  `cudaMemcpyAsync` + `cudaStreamSynchronize` pairs per iteration inside
  `_prepare_inputs`, plus a `refresh_blocks` stream sync at the tail of
  `prepare_resources`. The bug's co-reported KV Cache Size +17.61%
  (184.83 → 217.38) is **not** part of this defect — per the NVBug it is a
  change in what the statistic covers, and the baseline should be updated.
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
- **How introduced:** a new feature added the cost. The fix PR names no
  culprit; per the NVBug, a bisect over the 152 commits in rc13..rc14
  identifies commit `035de5d18515`, "[TRTLLM-10061][feat] Prefix caching
  support for mamba hybrid models (Qwen3.5 & Nemotron Super V3) (#12185)",
  which added the `LinearCacheType.RECURRENT_STATES` allocation path.
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
