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

The C++ manager owns one optional `HostSourceTable`. It uses mapped pinned
memory, readable by the GPU, with fixed addresses until manager shutdown.
Reserve its limits before creating requests or capturing graphs:

```python
manager._reserve_host_source_table(max_requests=32, max_pages=2048, max_beams=1)
view = manager._host_source_view
```

`HostSourceView` borrows that storage. Getting a view does not refresh, copy,
allocate, or protect data. Its arrays are read-only NumPy views; corresponding
`*_address` properties give CUDA addresses for kernels. Keep the manager alive
while using them and discard the view after shutdown. `EntryLayout` is a
separate model object and is not stored in this table.

| Array | Shape and meaning |
| --- | --- |
| `request_ids`, `generations`, `request_valid` | `[max_requests]`. A live request with an ID gets a row; validity is separate so every uint64 ID is supported. Reusing a row increments its generation, even if the request ID repeats. |
| `slot_ids`, `host_levels`, `completed_tokens` | `[max_requests, max_beams, num_life_cycles, max_pages]`. Retained host slot, host tier, and completed token prefix. An absent source is `(-1, -1, 0)`. |
| `pool_metadata` | `[num_levels, num_life_cycles, 4]`. GPU-readable pool base, bytes per slot, pool capacity in bytes, and pool-group index. Non-host tiers contain zeros. The cold-page codec has one host pool per lifecycle. |

`cache._host_source_slot` identifies the row. Anonymous caches have no row.
Requests cannot exceed the reserved limits or share a live request ID. KVCM
rejects these cases instead of reallocating the table. Consumers must check
row validity, request identity/generation, page bounds, and completed coverage.
Changing `cache.id` updates its table row and generation too; assigning `None`
removes the row until an ID is assigned again.

KVCM updates the table on backup, invalidation, offload, commit, prefix reuse,
suspend/resume, and close. Suspend keeps retained sources. A writable clone of
a partial reused page starts without a host source. Close clears all source
entries before a request row can be reused. Shared prefixes remain visible to
other requests that still hold the page. Releasing the final copy leaves no
published source.

Backup completion is polled at refresh or the next lifecycle update. Call
`manager._refresh_host_source_table()` to publish newly completed backups;
it does not wait for those backups. Pending and invalid copies stay absent.
Coverage is capped by both request history and the page's written prefix.
It counts input tokens, not model-specific compressed entries.

Before GPU work, including each graph replay, acquire a read scope:

```python
read = manager._acquire_host_sources(stream)
try:
    # Future consumer: SelectedEntries + EntryLayout + view.
    # Submit all GPU reads (or graph replay) on stream here.
    pass
finally:
    read.close()
```

Acquisition refreshes the table and uses `HostPageRead` handles to protect all
published sources. It also keeps requests alive until the scope closes, so
dropping a Python request reference cannot change the table during a read.
Concurrent scopes share the same unchanged table. Close records completion;
if it releases the last owner of a request, that request's cleanup waits for
completion. Lifecycle changes and explicit
refresh are rejected while any scope is open. After close, changing the mapped
table waits for its readers' GPU completion. Do this at the scheduler boundary,
outside capture, not once per attention layer. Pending copies may complete
during a read scope but are published only after that scope ends.

The existing per-page reader still works independently and can outlive request
close. Neither read API implements GPU hit lookup, LRU, refetch, or model wiring.

## Host layout and disk restore

The default cold-page codec joins the GPU page's pool slots into one host slot,
in pool order. For a buffer in GPU pool `p`, its host offset within a page is:

```text
sum(GPU slot sizes for pools before p) + buffer offset within GPU pool p
```

For `EntryLayout`, use the host pool group, one host pool (`pool_index = 0`),
this offset plus the model's entry offset, and the model's byte stride and size.
Include scale bytes and coalesced layer offsets. Native compressed models must
convert completed token coverage to completed stored entries using their own
layout rules.

A custom cold-page codec defines its own host layout. Its encoded bytes are
not automatically addressable as selected entries; the adapter must supply a
matching layout or decode them first.

If a page is on disk, `backupToHost()` first restores that page through the
existing codec to GPU, then creates its retained host copy. This needs room for
one full GPU page. Arrange disk restore before decode; selected-entry reads do
not perform disk I/O. This step does not add selected-entry fetch, LRU
replacement, graph execution, or model wiring.
