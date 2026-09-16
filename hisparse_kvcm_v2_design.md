<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DSA KV offload on KVCacheManagerV2

## 1. Goal and terms

Sparse attention reads only selected KV, but the current full-history DSA path keeps all its KV
on GPU. The goal is to keep that history in host memory and fetch selected KV into a GPU cache
with a fixed budget. Keep the host copies after fetching, so GPU copies can be replaced and
fetched again without another copy back to host.

The GPU cache may retain recently used KV as well as the current selection. Preserve the model's
selections and results. Indexer history, dense KV, metadata, temporary buffers, and prefill have
separate memory costs; the sparse GPU-cache budget does not bound those costs.

Port SGLang's `load_cache_to_device_buffer_kernel` from [hisparse.cuh](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh)
for GPU lookup, LRU replacement, refetch, and attention indices. Integrate the kernel source
with KVCM V2 through a small wrapper.

### Key terms


| Term            | Meaning                                                                                                                                      |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Layer buffer    | One kind of saved data from one layer, such as layer 0's K or V.                                                                             |
| Block           | A fixed group of token positions. With 64 tokens per block, block 0 covers tokens 0–63.                                                      |
| Layer group     | A set of layer buffers that KVCM manages together under the same rules for keeping and releasing data. One group can include several layers. |
| Page            | Cached data managed together: held, locked, moved, or dropped as one unit. An attention page contains one block's data for one layer group.  |
| Slot            | A fixed-size piece of memory that can hold several buffers.                                                                                  |
| Pool            | A collection of slots. Every slot in that pool has the same byte size.                                                                       |
| GPU cache entry | One token's KV, or one compressed entry, from one layer, kept in the small GPU cache.                                                        |
| Metadata        | Records of which tokens and layers the KV belongs to, and where it is stored.                                                                |


For example, four 4 KiB buffers can share one 16 KiB slot. Every slot in that pool is 16 KiB;
a different pool can use a different slot size. See [slot layout](cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/config.h#L101) and
[fixed slot size](cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.h#L172).

An attention page includes the group's configured buffers for that block, which may include
indexer data as well as K/V. Its buffers can occupy one slot in each of several pools, so a page
need not be one continuous piece of memory. See [Page](cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/page.h#L43) and [pages within a block](cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h#L72).

HiSparse can copy one selected token's KV from a page into the GPU cache while keeping the full
page on host. A **host copy** means KV data kept in host memory.

## 2. What exists today



### 2.1 Sparse model storage

Full-history DSA keeps active history GPU-locked unless residency controls are enabled. Top-K
changes which KV attention reads; it does not reduce that allocation. Some NVFP4 paths gather
selected KV into temporary GPU buffers while keeping the full GPU history.

DSA and MiniMax M3 put indexer K and attention KV in the same layer configuration. Buffer sizes
may place them in different pools, but they share the same rules for keeping and releasing pages.
DeepSeek-V4 separates sliding-window attention (SWA) from other state, while its sparse compressed
KV and compressed indexer K still share those rules.

Sources: [DSA buffers](tensorrt_llm/_torch/attention/backends/sparse/dsa/cache_manager.py#L538), [MiniMax buffers](tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py#L235), and [DeepSeek-V4 groups](tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py#L928).

### 2.2 Page storage and protection

V2 defaults to C++; Python is a reference backend. Existing components provide:


| Component                         | What it does                                                                                          |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `StorageManager`                  | Allocates GPU, host, and disk storage. Moving a whole page to another level releases its source slot. |
| `LRUEvictionPolicy`               | Uses CPU-managed page queues ordered by priority and least-recently-used order (LRU).                 |
| `BlockRadixTree`                  | Shares pages for prefix reuse.                                                                        |
| `IKvCacheColdPageCodec`           | Converts and copies whole pages between GPU and host/disk formats.                                    |
| `kv_cache_config.host_cache_size` | Sets the host-memory budget.                                                                          |


A page's protection controls what storage management may do:

- **HELD:** keep the data; it may move to another storage level.
- **LOCKED:** keep the data on GPU. Completion events also prevent reuse before earlier GPU work
finishes.

The local `residencyGroup` change lets otherwise-identical layer configurations have separate
page lifetimes. `setResidencyWindow()` lets a full-history request keep GPU locks for sinks,
recent history, and writable pages while holding older pages. Sinks are required prefix tokens.
Use V2's assigned `LayerGroupId` when calling `setResidencyWindow()`, not the `residencyGroup` value.

This window API chooses protection from token ranges. Sparse top-K can select scattered tokens
anywhere in history, so it needs a separate operation to find and fetch those selections. That
operation is proposed in section 3.4.

Sources: [KvCache](cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp), [residency rules](cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md#internal-sparse-residency), [page locks](cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/page.cpp),
[storage management](cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.cpp), and [the codec](cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/coldPageCodec.h).


## 3. Proposed changes



### 3.1 Use `residencyGroup` to separate data with different GPU needs

For HiSparse, use this split:


| `residencyGroup` | Data                                                 | Where it stays                               |
| ---------------- | ---------------------------------------------------- | -------------------------------------------- |
| `0`              | Indexer/scoring data and required dense-attention KV | GPU while needed                             |
| `1`              | Sparse-attention KV, including its scales            | Full history on host, with a small GPU cache |


**Sparse KV still has a full history.** The difference is where that history is stored.
Full-history lookup tables must be GPU-readable.

Two implementation steps are required:

1. Put indexer data and sparse KV into **separate KVCM layer entries**, each with a unique ID.
2. Give those entries different `residencyGroup` values so V2 cannot combine them into one page,
  even when their window and sink settings match.

The field applies to every buffer in a layer entry. **It only separates pages.** It does not copy
data or create a GPU cache; the new storage/cache code does that.

> **KVCM and attention owner review:** confirm each model's split and entry mappings. Check that
> releasing sparse GPU locks leaves required indexer/dense KV locked.

Sources: `AttentionLayerConfig` and [the grouping rule](cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/lifeCycleRegistry.h#L39).

### 3.2 Fetch only the selected data

**Fetch the tokens or compressed entries the model selects, including their scales.** Pages can
remain the allocation unit. Selecting one token from a 64-token page should fetch that token's KV.

Keep the model's existing scoring and top-K settings:

1. `SelectionPolicy` returns request/layer/lifecycle IDs, positions, and valid entries.
2. The copy code fetches duplicate selections once, preserving the order attention expects.
3. The host layout supports reads of selected entries in the model's existing KV format.

Extra lossy compression needs separate accuracy tests. Whole-page codecs may need changes to
support these small reads.

> **KVCM and attention owner review:** confirm selection IDs, offsets, compressed entries/scales,
> valid lengths, and shared-prefix mappings.



### 3.3 Keep host copies and protect unfinished work

**Keep host copies after fetching. Replacing a GPU copy must not lose the stored history.**

For newly written KV:

1. Write into protected GPU storage.
2. Copy to host after the writes finish. The host copy becomes valid when the copy finishes.
3. Reuse the GPU slot only when the host copy is valid and no protected use remains.

Protect selected/prefetched KV through attention, required sinks/window and writable data while
needed, and unfinished reads/copies until completion. Before changing data with a host copy,
mark that copy invalid; mark the updated copy valid only after its transfer finishes.

Fully written uncommitted KV also needs host copies. Committed values remain read-only.

Hold needed pages to prevent dropping their data. Also keep active host addresses fixed: a hold
alone still allows movement. Restore any needed disk data to host before decode; GPU fetches
cannot read disk.

**KVCM owns one `HostSourceTable`.** Reserve its request, beam, and page limits before creating
requests or capturing a graph. Its mapped, pinned memory has fixed addresses until manager
shutdown. For each request slot it stores the request ID and a reuse generation. For each
beam, layer group, and block it stores the retained host slot, host level, and completed token
count. Pool metadata gives the host base address, slot size, capacity, and pool group.

KVCM updates these records with the request and page lifecycle. Pending or invalid copies have
no published slot and zero coverage. Refresh polls backup events without waiting for the backup;
only completed copies become visible. Commit and prefix reuse follow the actual shared page.
Suspend keeps retained sources; resume replaces sources when it copies a partial prefix into a
new writable page. Close clears the row before its slot can be reused.

`HostSourceView` only borrows the table's addresses and dimensions. It owns no memory, builds no
second table, allocates no storage, and copies no data. Keep model-specific `EntryLayout` separate:
KVCM reports completed **tokens**; the model layout defines stored entries, strides, and scales.

Before GPU use, acquire a table read scope. It refreshes completed sources and protects them
with existing `HostPageRead` handles. Close the scope after submitting GPU work, including graph
replay. Its completion event protects the table and host slots. Table changes are rejected while
a read scope is open; after close, CPU updates wait for its GPU work before changing mapped
metadata. This is a scheduler boundary, not a per-layer operation. The view itself is not a
lifetime guard. A captured graph must acquire a fresh read scope for each replay.
See [host-source table usage](docs/source/developer-guide/kv-cache-host-copies.md#host-source-table).

> **KVCM owner review:** confirm copy validity, safe reuse, fixed host addresses, partial pages,
> shared data, and host/disk exhaustion.



### 3.4 Find GPU hits, fetch misses, and give attention its indices

**Pass logical selections directly to `ensure_resident()`, backed by HiSparse's
`load_cache_to_device_buffer_kernel`.** Its inputs are `SelectedEntries`, a separate `EntryLayout`,
KVCM's `HostSourceView` protected by a read scope, and mutable GPU-cache state. It handles:

1. Validate selection IDs and bounds.
2. Find GPU hits and choose replaceable LRU entries for misses.
3. Resolve valid host sources for misses and fetch their KV.
4. Produce attention indices and protect the selected GPU slots through attention.

The HiSparse kernel combines lookup, LRU, and copying. Attention must run after its copies finish;
returning from the Python call does not wait for GPU completion. Keep host sources alive through
copy completion and prevent selected GPU slots from being replaced until attention finishes.

Allocate buffers before graph capture. The KVCM integration must handle layouts/scales,
duplicate and padded selections, request-slot reuse, and source validity. Keep required checks
on GPU and report unavailable selected data; no per-layer CPU allocation or synchronization.

Attention must use this new mapping. The existing `convert_req_index_to_global` returns `-1` for
missing GPU page entries; using it before fetch would lose host-resident selections.

CPU page LRU and `batchedLockToGpu()` handle whole pages. They do not implement this GPU operation.

> **KVCM, attention, and executor owner review:** confirm lookup, replacement, copy ordering,
> indices, error reporting, and progress when the cache is full.



### 3.5 Reuse V2 code and connect the models

**Keep one** `KvCache` **per request, with separate host-storage and GPU-cache allocators.** Track
host copies and GPU entries separately; `Page::cacheLevel` alone cannot describe both.

- Reuse V2 allocation, pinned host memory, events, and whole-page backup code.
- Integrate the HiSparse kernel source and required helpers. Record the upstream revision and
preserve its license. Limit changes to the KVCM binding, layouts, and required lifetime checks.
- Keep selection/layout interfaces and KVCM ownership separate from kernel execution.
HiSparse handles LRU and transfer together through `ensure_resident`. Future kernels can use
the same selection, layout, and ownership interfaces.
This follows the [generalization principle](hisparse_design_principle.md).
- Connect `DSACacheManagerV2`, `DeepseekV4CacheManager`, and `MiniMaxM3KVCacheManagerV2` through
`sparse/registry.py`, using each model's layout and selection format.

> **KVCM, transfer, and attention owner review:** confirm ownership, copy addresses,
> index-buffer lifetimes, ordering, and the kernel binding.



## 4. One decode step

**Each layer writes new KV, prepares selected KV on GPU, and runs attention.**

Reserve table and cache capacity before graph capture. Before each replay, update request IDs,
lengths, and valid rows, including padding. Acquire a KVCM host-source read scope for that replay
and close it after submission. Host source addresses must stay valid until the reads finish.

Each layer runs these steps on GPU:

1. **Write** new KV and scoring data. Wait for prior uses before overwriting a slot.
2. **Select** the KV to read. IndexShare layers can share a selection but need their own KV.
3. **Fetch** missing KV by passing the selection, `EntryLayout`, `HostSourceView`, and GPU-cache state directly
  to `ensure_resident()`. New KV can use its protected GPU copy while backup is pending.
4. **Run attention** using the returned indices, after the required copies finish.

Copy newly completed KV to host after its writes. This may overlap attention when both read the
source. For DeepSeek-V4, copy compressed entries when produced, not necessarily every token step.

Prefetch only when the selection is known and space is reserved. At full capacity, wait for old
uses before replacing slots. Follow section 3.3's protection rules throughout.

For shared-index layers, use `copy_cache_planned_kernel` to replay a recorded miss plan only when
their corresponding GPU slots hold the same token positions and use the same replacements.
Each layer copies its own KV. Sharing top-K positions alone does not establish this condition.

Existing CPU lock releases still use `recordEventScope()` and `KvCache::finishEvent()`.
GPU-entry reuse needs stream/event ordering without CPU lock calls per token.

> **KVCM, attention, and executor owner review:** confirm writes, copies, attention, and reuse
> stay correctly ordered, including full capacity and side-stream prefetch.

Sources: [DSA selection](tensorrt_llm/_torch/attention/backends/sparse/dsa/backend.py) and [page-index conversion](cpp/tensorrt_llm/kernels/convertReqIndexToGlobal.cu).

## 5. Request lifecycle and memory limits



### Prefill and transition to decode

**Use existing GPU prefill, then switch to the small decode cache.**

1. Compute prefill KV on GPU.
2. Copy sparse history to host and wait for completion.
3. Free or shrink the large sparse GPU-history allocation before using the small decode cache.

This reduces decode memory. Prefill still needs its original GPU capacity; larger contexts need
separate prefill work.

> **KVCM and attention owner review:** confirm prefill capacity, copy completion, and actual
> release of full-history sparse GPU storage.



### Capacity and failure

**Reserve GPU and host memory before admitting a request.**


| Memory | Include in the budget                                                                                                                                  |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GPU    | Selected cache, HiSparse's extra newest-token storage, indexer/dense KV, sinks/window, writes, mappings, prefill, and temporary copy/prefetch buffers. |
| Host   | History already present and the admitted generation budget.                                                                                            |


Check further reservations before a step starts. Count each shared allocation once and each
request-local copy separately. Include pending work; measure allocated memory, not just locked pages.

If a step fails, report the error before returning its outputs or reusing affected slots. Preserve
shared data and wait for pending work before freeing memory. Do not automatically retry a partly
executed forward; safe retry needs separate work.

> **KVCM and scheduler owner review:** confirm reservations, history growth, and cleanup after
> partial writes or failed copies.



### Prefix reuse, suspension, and close

**Requests can share pages while choosing different KV for their GPU caches.** Each request keeps
its own mapping; selections must not change shared token IDs or committed values.


| Action               | Required behavior                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------ |
| Commit               | Keep values read-only and preserve pending-copy events. Do not recreate dropped data.      |
| Suspend              | Keep references to pages the request still needs.                                          |
| Close                | Release this request's references; other requests or transfers may still need those pages. |
| Reuse a request slot | Clear its old GPU-cache mappings first.                                                    |


Memory reuse must wait for pending reads/copies. Full-history sparse KV must remain available
when unselected; native SWA follows its own window rules.

> **KVCM owner review:** confirm shared-page identity, commit rules, and page/GPU-copy lifetimes.



### Disaggregated serving

**Exchange all history the destination needs, including unselected sparse KV.** Extend the
[disaggregation adapter](tensorrt_llm/_torch/disaggregation/resource/cache_reuse.py) to support host addresses:

1. Send or receive sparse history in host memory; put required indexer/dense data on GPU.
2. If a transport needs staging, transfer chunks through a fixed-size temporary buffer.
3. Keep addresses valid through completion and initialize cache mappings after the data is ready.

Sparse history can exceed the decode GPU-cache budget; required GPU data must still fit.
Validate NIXL/UCX/MPI support and limits. Local prefill/decode must also work without disaggregation.

> **KVCM and disaggregation owner review:** confirm addresses, memory registration, completion
> tracking, and staging limits for each supported transport.



### Configuration and defaults

**Reuse** `host_cache_size` **and the model's selection settings.** Add feature enablement and a
GPU-cache budget if existing settings do not cover them.

Define the budget's units, whether it is per request or shared, and space for writes and other
protected data. Reject budgets that cannot hold a valid selection.

Use the HiSparse kernel; choose its launch settings from tests and benchmarks. Keep tuning internal
unless a user setting is needed. New public settings need the review listed in section 6.


## 6. Validation and deliverables


| Area                  | Required checks                                                                                                                                                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correctness           | Match each GPU-only sparse baseline's selections and output tolerance. Physical indices may differ.                                                                                                                               |
| Kernel integration    | Test the actual HiSparse kernel with KVCM host handles, model layouts/scales, short-sequence initialization, newest-token storage, and required protection/error checks.                                                          |
| Transfers             | Page boundaries, compressed entries/scales, duplicate/invalid/empty selections, partial entries, and extra bytes from scattered top-K.                                                                                            |
| Host copies           | Pending copies, writes, uncommitted KV, repeated fetches, and GPU replacement without another GPU-to-host copy.                                                                                                                   |
| Full capacity         | All hits/misses, overlap, and completely different selections with protected writes, sinks/window, prefetch, and delayed work.                                                                                                    |
| CUDA graphs           | Changed selections/request IDs across same-shape replays, padding, stable buffers/host addresses, side streams, shared-plan validity, and the `nvbugs/6018172` mapping case.                                                      |
| Lifetimes and sharing | Commit, suspend/resume, close, prefix reuse on/off, different selections over shared pages, reused request slots, and cleanup.                                                                                                    |
| Prefill               | Host-copy readiness, actual release/shrink of full-history GPU storage, and the separate prefill peak.                                                                                                                            |
| Disaggregation        | Full required history larger than decode sparse-cache capacity, bounded staging, concurrent transfers, prefix reuse, and destination readiness.                                                                                   |
| Failures              | GPU/host exhaustion, supported disk-restore errors, rejected copy submissions, partial writes, shared-data protection, and cleanup.                                                                                               |
| Memory/performance    | Measure latency, throughput, and transfer bytes across batch sizes and miss rates. Count all GPU/host storage, including newest-token space and metadata. Compare with recorded HiSparse timings and check for leaks. |


Deliver reviewed ownership and lifecycle rules, selection/transfer APIs, reservations and cleanup,
model layouts, and graph/performance results. Document codec changes and supported combinations.
Keep [HiSparse_implementation.md](HiSparse_implementation.md) aligned with those results.
