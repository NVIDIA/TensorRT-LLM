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

All operations below use an active request's CUDA stream and follow KVCM's
single-threaded CPU access rule. Order GPU writes before backup on that stream.

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
completed count when publishing a host-valid table without a stream dependency.
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
