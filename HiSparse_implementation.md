<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HiSparse implementation tasks

Keep sparse KV history on host and fetch selected KV into a GPU cache with a fixed budget.
Keep host copies after fetching. Preserve each model's GPU-only sparse-attention results.

Follow [the V2 design](hisparse_kvcm_v2_design.md) and
[the generalization principle](hisparse_design_principle.md). Start from current `main` and its
local residency changes. Targets: GLM5.2, DeepSeek-V4, and MiniMax M3. MTP is out of scope.

**Status (2026-09-16):** Steps 2–4 are implemented locally. The current focus is to finish their
review and validation. Step 5 is unimplemented; scheduler and model wiring follow later.

## Rules shared by all tasks

- Keep indexer/scoring data, required dense KV, and full-history lookup tables on GPU.
- Fetch tokens or the model's compressed entries; allocation can still use pages.
- Keep host copies valid and at stable addresses while decode can fetch them. Protect new writes
  until backup finishes; replace GPU entries only when a valid host copy exists and protected uses finish.
- Run model selection on GPU, then HiSparse's fused lookup/LRU/refetch kernel within the decode
  graph. Allocate buffers beforehand. Keep selection/layout and ownership interfaces separate
  from kernel execution so future kernels can use the same inputs.
- Reserve GPU/host capacity before admitting work. Unexpected failures need safe cleanup;
  automatic retry of a partly executed forward is separate work.
- Support local prefill/decode and prefix reuse. Disaggregation is optional: prefer direct host
  transfers, with fixed-size staging buffers where needed.

Use the design's [key terms](hisparse_kvcm_v2_design.md#key-terms): page, GPU cache entry, slot,
and metadata. Host copies are data kept in host memory.

## Implementation order

Dependencies apply to completed integration; independent tasks can start together. Use synthetic
KV and selections for storage/cache tests, then real models for attention checks. Each task adds
its own focused tests. Later combined tests do not replace them.

### 1. API and owner review

**Work:** Review the agreed design with KVCM, attention, executor, scheduler, and transfer owners.
Set page/copy ownership, model selection formats and layouts, APIs, and budget units/scope.
Review the KVCM integration of the chosen HiSparse kernel, including its host-address,
short-sequence, and newest-token assumptions. Retain V2 prefix reuse and local prefill/decode.

**Validation:** Record owner review and check the risky API assumptions with small prototypes:
changing graph inputs, full-cache progress, shared prefixes, and host exhaustion. Final model
checks follow in later tasks.

### 2. Residency groups and active-request lifecycle

**Depends on:** 1 for final review.

**Status:** `residencyGroup` and `setResidencyWindow()` are implemented locally. They provide
page protection, not arbitrary top-K fetching or the new GPU cache.

**Work:** Use the split in design section 3.1: separate KVCM layer entries with unique IDs,
`residencyGroup = 0` for data on the existing GPU path, and `residencyGroup = 1` for sparse KV
kept on host. These values separate pages; tasks 4–5 add host copies and the small GPU cache.
Model adapters apply the split in tasks 7, 9, and 10.

**Validation:** Check mixed held/locked pages, independent groups, sinks/window, writable tails,
resize, commit, suspend/resume, shared locks, and cleanup. Include fully written uncommitted KV
with prefix reuse disabled. Full-history sparse KV must retain old unselected data.

### 3. Selection and layout interfaces

**Depends on:** 1 for final review; can run alongside 2.

**Status:** Implemented. Logical DSA top-K positions keep their order and duplicates.
`SelectionContext` borrows DSA's existing `req_idx_per_token` and `kv_lens_cuda`, plus explicit
host-source rows and generations. No separate selection policy or request-ID lookup is needed.

`EntryFormat`, keyed by `BufferId`/`DataRole`, describes model tensor shapes, dtypes, entry axes,
and compression. KVCM owns lifecycle and pool mapping; the codec supplies byte offsets and sizes.
Opaque codecs are rejected when host sources are enabled. `HostSourceView` borrows KVCM's table.
These interfaces do not move KV. HiSparse will perform the single GPU hit lookup in step 5.
Model wiring follows in 7, 9, and 10.

**Validation:** 25 selection/format and DSA tests passed on CPU. They cover borrowed metadata,
order, masks, invalid formats, context/decode, and IndexShare. GPU lookup and fetching remain part of step 5.
Full package and model inference validation remain pending.

See [the interface guide](docs/source/developer-guide/sparse-attention-development-guide.md#selected-kv-interfaces).

### 4. Host storage and backup

**Depends on:** 2, 3.

**Status:** Implemented in the C++ core with internal Python bindings. Model wiring is a later step.

**Work:** Reuse V2 allocation, pinned host memory, events, and suitable copy code. Add retained
host copies and write invalidation. Mark host copies valid after backup finishes; protect GPU
sources until safe reuse. Keep active host addresses fixed: holding a page alone does not prevent
movement. Restore needed disk data to host before decode.

**Validation:** 68 native tests passed under CUDA memcheck with zero errors; 12 focused Python
host-source tests and two executor row tests passed. Coverage includes pending/completed copies, partial pages,
invalidation, offload, shared prefixes, explicit row reuse, stale generations, fixed addresses,
graph metadata reads, batch-scoped readers, codec offsets, unsupported codecs, and cleanup.
Validation uses rebuilt C++ components and bindings. The two executor tests ran with isolated
source methods and the rebuilt `IndexMapper`; full-package and model inference checks remain pending.

The new methods are `backupToHost`, `invalidateHostCopy`, `offloadToHost`, and
`acquireHostCopy`. They use the configured host tier's budget and codec. Read handles
keep addresses alive through GPU use; live copies prevent host pool movement. Offload
reuses the host slot, and later full-page GPU restoration keeps that host copy.
Disk restore uses one full GPU page before creating a retained host copy.

KVCM owns a fixed-size, GPU-readable `HostSourceTable`. The executor derives capacity from its
existing limits and binds existing `IndexMapper` rows. Early index release clears a row without
releasing the request's host copies. Generations reject stale references when a row is reused.
`HostSourceView` borrows addresses without allocating or copying.

Page changes update affected entries and shared users. Append updates touch the tail and newly
stale pages; other structural changes rebuild only the affected request. Read acquisition polls
pending copies and protects published sources for the batch.
Unrelated requests remain usable. Manual event polling is test-only. `HostSourceRead` and
`HostPageRead` preserve CUDA-event and lifetime protection. GPU caching, refetch, LRU, and model
wiring remain unimplemented.

See [host-copy usage and layout](docs/source/developer-guide/kv-cache-host-copies.md).

### 5. GPU cache, refetch, and graph execution

**Depends on:** 2, 3, 4.

**Status:** Unimplemented. Start after steps 2–4 are finalized.

**Work:** Use SGLang's [HiSparse kernels][hisparse-kernel] directly behind `ensure_resident()`.

- Accept `SelectedEntries`, model `EntryFormat`, KVCM/codec layouts, `HostSourceView`, and GPU-cache state directly.
- Integrate `load_cache_to_device_buffer_kernel` for GPU lookup, LRU replacement, host copies,
  and attention indices. Record the upstream revision and preserve its license.
- Check host readiness, row generations, capacity, duplicate/padded selections, layouts/scales,
  and active readers on GPU. Support short sequences starting from an empty cache.
- Support newest entries and shared miss plans through `copy_cache_planned_kernel`.
  Keep each layer's KV and read protection separate; release protection after attention.
- Prepare compilation and all workspaces before graph capture. Include required kernel source
  and licenses in package data.

**Validation:** Check fetched KV/scales against host bytes. Test hits, misses, full capacity,
newest entries, host ownership, shared plans, side-stream readers, and changing graph inputs.
Check that selected slots stay protected through attention.
Run CUDA memory and race checks. Record cache memory use and hit/miss timings; broader tuning
and model measurements follow in later steps.

### 6. Capacity, configuration, and scheduler

**Depends on:** 4, 5.

**Work:** Reserve host history/generation space and GPU cache, including HiSparse's extra
newest-token storage, indexer/dense KV, sinks/window, writes, mappings, prefill, and temporary
buffers. Check growth before a step starts. Count shared host allocations once and request-local
GPU copies separately; optional prefetch must also fit.
Reuse `host_cache_size` and top-K settings; add enablement and a GPU budget as needed.

**Validation:** Verify admitted steps fit reservations. Test exact budgets, mixed prefill/decode,
invalid settings, and errors after partial writes. Protect shared data and pending work during cleanup; report failure before returning
outputs or reusing affected slots. New public settings need documentation, golden-manifest updates,
and telemetry/privacy CODEOWNER review in the change that adds them.

### 7. DSA and GLM5.2 local prefill/decode

**Depends on:** 5, 6.

**Work:** Wire `DSACacheManagerV2` and `sparse/registry.py` to the HiSparse cache. Split indexer
and sparse KV lifetimes. Use existing GPU prefill, finish host backup, then release/shrink full-history
sparse GPU storage. Run GPU selection/refetch and attention per layer. IndexShare can share
selections; each layer still needs its own KV.

**Validation:** Match GLM5.2 selections and outputs within the existing tolerance. Use new cache
indices so host selections are not lost as `-1` page entries. Test changing request IDs, selections,
and padding across graph replays, including `nvbugs/6018172`. Measure bounded decode cache use and
the separate prefill peak; this initial path still needs history to fit GPU prefill.

### 8. Prefix reuse and request cleanup

**Depends on:** 7. Basic shared-ownership tests also belong in 2, 4, and 5.

**Work:** Keep shared logical pages under V2 ownership and build each request's GPU mapping over
them. Preserve committed values and pending-copy events. Suspended requests keep references to needed pages;
close releases references safely. Clear mappings before request slots are reused.

**Validation:** Use shared prefixes with different selections. Test reuse on/off, commit,
suspend/resume, concurrent reads/copies, close, reuse hits, output correctness, and final release.

### 9. DeepSeek-V4 enablement

**Depends on:** 7, 8; independent of 10 and 11.

**Work:** Wire `DeepseekV4CacheManager` through the shared interfaces to HiSparse's DSv4 layout
variant. Separate sparse compressed KV from indexer data; preserve required sliding-window and
other state. Fetch compressed entries/scales. Copy compressed entries to host when produced;
a new entry need not appear every token step.

**Validation:** Repeat layout, model-output, selection, graph, capacity, prefill, prefix reuse,
and cleanup checks against the GPU-only DeepSeek-V4 sparse baseline.

### 10. MiniMax M3 enablement

**Depends on:** 7, 8; independent of 9 and 11.

**Work:** Wire `MiniMaxM3KVCacheManagerV2` and its attention backend through the shared interfaces
to the HiSparse kernel with its KV/scale layout. Separate indexer and sparse KV lifetimes while
keeping required dense KV on GPU.

**Validation:** Repeat layout, model-output, selection, graph, capacity, prefill, prefix reuse,
and cleanup checks against the GPU-only MiniMax M3 sparse baseline.

### 11. Optional disaggregated serving

**Depends on:** 7, 8 for the first model; 9 or 10 for those models' checks.

**Work:** Add direct host transfers for sparse history and GPU destinations for required resident
data. Use fixed-size staging buffers when the transport needs them. Exchange all required history,
including unselected KV. Keep addresses valid through completion; document NIXL/UCX/MPI support.

**Validation:** Transfer history larger than the decode sparse GPU cache, while required GPU data
still fits. Test prefix reuse, concurrent transfers, destination readiness, outputs, bounded
staging, and cleanup. This task does not block local model support.

### 12. HiSparse kernel tuning and performance

**Depends on:** 7 for initial results; 9 and 10 for all layouts; 11 for disaggregation results.

**Work:** Measure and tune the integrated HiSparse kernel and its supported layout variants.
Vary context length, top-K, batch size, and miss rate, including small batches with many misses.
Measure shared-index copy-only prefetch where supported. Compare tuned kernels with recorded
HiSparse measurements to quantify changes with the same KV format, usable capacity, and protection rules.

**Validation:** Run the same correctness/ordering checks after tuning. Measure total allocated
GPU memory, host use, prefill peak, throughput, latency, and bytes/time copied both ways. Include
newest-token storage, indexer/dense KV, mappings, and temporary buffers. Report extra bytes from
scattered selections and any regressions. Freed slots alone do not prove lower GPU allocation.

### 13. Combined correctness and regression checks

**Depends on:** 8, 9, 10, 12; include 11 for supported disaggregation combinations.

**Work:** Combine all target models, supported HiSparse kernel variants, graph modes, and
prefix reuse settings. Include long runs with changing selections, generated writes, exhaustion, and cleanup.

**Validation:** Match baseline selections/outputs within existing tolerances; physical indices
may differ. Require full-cache progress, safe partial-write cleanup, correct sharing, and no leaks.
Run relevant C++ tests and production Python imports. Set `LLM_MODELS_ROOT` for model-weight tests.

### 14. User documentation and release checks

**Depends on:** 12, 13 for the combinations being released.

**Work:** Add a feature guide, examples/YAML, supported models/transports, budget guidance, and
measured tradeoffs. Explain prefill limits and optional disaggregation. Document final APIs and
codec changes; link validation and benchmark results.

**Validation:** Run documented configurations and confirm they match tested support and limits.
Publish tested HiSparse launch defaults; keep unnecessary tuning settings internal.

[hisparse-kernel]: https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh
