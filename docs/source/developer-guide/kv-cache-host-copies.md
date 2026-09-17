<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Retained host copies in KVCacheManagerV2

KVCM V2 can keep a host copy while a page is on GPU. After the page becomes
`HELD`, the caller can release its GPU slot and keep the host copy. This is an
internal C++ API for sparse KV offload. Model integration and the small GPU
cache for selected entries are separate steps.

## Operations

All operations below use an active request's CUDA stream. Drive each request
from one CPU thread; the manager lock protects shared state across requests.
Order GPU writes before backup on that stream.

| C++ method on `KvCache` | Internal Python binding | What it does |
| --- | --- | --- |
| `backupToHost(group, ordinal, validTokens, hostLevel, beam)` | `_backup_to_host` | Allocate a retained host slot and enqueue backup. Reuse an existing copy if it already covers the requested prefix. |
| `invalidateHostCopy(group, ordinal, beam)` | `_invalidate_host_copy` | Mark a writable page's host copy stale before submitting new GPU writes. |
| `offloadToHost(group, ordinal, beam)` | `_offload_to_host` | Make the retained host slot the page's primary slot and release its GPU slot. It does not copy KV again. |
| `acquireHostCopy(group, ordinal, beam)` | `_acquire_host_copy` | Return a `HostPageRead` handle and wait for backup in the request's stream. |

`group` is a layer/lifecycle group ID. `ordinal` is the request's block number.
`validTokens` counts written tokens within that page, starting at zero. It must
not exceed the request's history or a committed page's recorded coverage.
`hostLevel` defaults to 1 and must name a configured host-memory tier. `beam`
defaults to zero.

Backup uses the existing host tier's allocator, pinned memory, budget, and
cold-page codec. It copies a whole page, while recording only the written token
prefix as usable. Full, uncommitted pages can be backed up and offloaded. Backup
counts once in the existing offload byte/block counters; changing the primary
slot does not count another transfer.

## Completion and reads

A reader exposes the host slot ID, pool group, address, pool base address, page
size, and pool capacity. Keep the reader open while submitting work that uses
those addresses on its request stream. Call `close()` after submitting that
work. Closing records an event; it does not wait on the CPU. The handle can
outlive request closure, but must close before manager shutdown.

In Python, `reader.valid_tokens` is the prefix covered **after** the wait
inserted by acquisition. `reader.ready` reports whether backup has finished.
`reader.completed_tokens` returns zero while it is pending or stale. Use the
completed count to inspect readiness. KVCM publishes its host-source table itself.
A CPU read of the address also requires backup to have finished.

Before changing GPU data that has a host copy, close its readers and call
`invalidateHostCopy()`. This requires an uncommitted, GPU-locked page. The next
backup waits for prior host reads and refreshes the same host address. Readers
cannot acquire a stale copy. Backup and read failures must be handled before
publishing selected-entry mappings.

## Releasing GPU storage and cleanup

Use `setResidencyWindow()` to let old pages become `HELD`. Offload requires a
held page and a backup covering all its written tokens. A pending backup is
allowed: the released GPU slot carries an event that prevents reuse before the
copy and prior GPU uses finish. The host page also carries its readiness event.
Ordinary page-table entries for held pages remain invalid.

A later full-page GPU restore keeps the host copy, so offloading that page
again needs no new host allocation or backup. Moving an uncommitted page into
the prefix tree transfers ownership of its copy too.

Active pages with retained copies use explicit offload; ordinary tier eviction
does not move them. A held page is never dropped to make host space. Exhaustion
raises an error, leaving the GPU source available. When the last request lets
go of a cached GPU page, its extra host copy is released. An inactive host page
can move to disk or be dropped by the usual policy. Readers protect their host
slot independently until their work finishes.

A pool cannot move or resize while it contains retained host copies. After
those copies are released, resizing waits for any remaining reads or backups.
This protects addresses as well as page contents.

## Host-source table

The C++ manager owns one optional `HostSourceTable` in mapped pinned memory.
Its addresses stay fixed until shutdown. The executor's `initialize_host_sources()`
takes no capacity settings: it uses the existing `IndexMapper` capacity,
`max_blocks_per_seq`, and `max_beam_width`. Call it before graph capture.

Each request uses its existing `IndexMapper` row. The native binding
`cache._bind_host_source_row(row)` connects that row; it does not allocate another.
The native initialization API receives the executor's limits (standalone tests
supply their own limits). No GPU request-ID lookup is needed.

| Array | Shape and meaning |
| --- | --- |
| `request_ids`, `generations`, `request_valid` | `[max_requests]`. Request ID for diagnostics, row generation, and validity. Generation increases on row reuse or ID change. |
| `slot_ids`, `host_levels`, `completed_tokens` | `[max_requests, max_beams, num_life_cycles, max_pages]`. Retained host slot, host tier, and completed token prefix. An absent source is `(-1, -1, 0)`. |
| `pool_metadata` | `[num_levels, num_life_cycles, 4]`. GPU-readable base, slot bytes, pool bytes, and pool-group index. Non-host tiers contain zeros. Each cold page occupies one pool slot. |

`HostSourceView` borrows these arrays. Getting a view does not allocate, copy,
refresh, or protect data. Arrays are read-only NumPy views; `*_address` properties
give CUDA addresses. Keep the manager alive and discard borrowed views after shutdown.

`cache._host_source_ref` returns `(row, generation)`. A batch passes these pairs
in the same order as its existing request metadata. The future kernel checks
row validity and generation directly. `EntryFormat` remains separate model data.

A request can outlive its row: disaggregated prefill may release its index while
KV transfer continues. `release_index_slot()` unbinds the host-source row first;
retained host copies stay on the request's pages. Rebinding uses the new row and
generation. Assigning `cache.id = None` also unbinds; assigning a new ID alone
does not choose a row. Close clears the row before another request can bind it.

Backup, invalidation, and offload update the affected page and its shared-prefix
users. Append updates touch only the tail and newly stale pages. Other structural changes,
such as commit and resume, rebuild only the changed request. Suspend keeps sources; cloning a writable partial prefix does
not inherit its old source. Pool metadata changes only when pools change.
Completed coverage is capped by the request history and committed page coverage.
It counts input tokens, including for compressed model entries.

Read acquisition polls only pending copies in the requested batch. It never
waits for backup completion and never publishes pending or invalid copies.
`_introspection.refresh_host_source_table(manager)` is a test helper. The manager's
production interface has no manual refresh method.

Acquire one batch read scope before each GPU submission or graph replay:

```python
view = manager._host_source_view
rows = [cache._host_source_ref for cache in batch_caches]
read = manager._acquire_host_sources(rows, stream)
try:
    # Future ensure_resident consumes selections, model format, codec layout,
    # this borrowed view, and mutable GPU-cache state.
    # Submit GPU reads or graph replay on stream here.
    pass
finally:
    read.close()
```

`HostSourceRead` retains only these requests and uses `HostPageRead` handles for
their published copies. Unrelated requests can change or close. Shared pages
retain their normal read protection. Changes to a protected row are rejected
while its scope is open; after close they wait for the GPU completion event.
Pool changes and shutdown require all scopes to close. If closing a scope drops
the last request owner, request cleanup may wait for completion.

Acquire and close outside graph capture, once per batch submission. Overlapping
scopes for the same row use its unchanged metadata; pending copies are published
on a later acquisition after those scopes close. A per-page reader can still
outlive request close. Neither read API performs GPU hit lookup, LRU, or refetch.

## Host layout and disk restore

Model `EntryFormat` is keyed by `BufferId` (layer ID and `DataRole`). It describes
component tensor shapes, dtypes, entry axes, and tokens per entry. Components can
include KV and scale tensors. It contains no lifecycle, pool, slot-size, or
coalesced-buffer offset fields.

Use `manager._host_buffer_layout(BufferId(...))` for physical layout. KVCM validates
the codec's `ColdBufferLayout` and returns the lifecycle, byte offset and size within
a cold slot, and number of native model buffers per logical page (`expansion`).
The view supplies that lifecycle's host pool base, slot size, and pool group.
The future consumer combines these with the model format to locate entry bytes.

The default codec concatenates hot pool slots in pool order. It computes buffer
offsets once from KVCM's coalesced-buffer descriptors. Models must not repeat that
calculation. A custom codec can implement `queryBufferLayout()` only if it exposes
byte-preserving random access to each buffer. Otherwise host-source initialization
rejects it clearly; its existing whole-page encode/decode API remains available.

If a page is on disk, `backupToHost()` first restores that page through the
existing codec to GPU, then creates its retained host copy. This needs room for
one full GPU page. Arrange disk restore before decode; selected-entry reads do
not perform disk I/O. This step does not add selected-entry fetch, LRU
replacement, graph execution, or model wiring.
