<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCacheManagerV2 NIXL GDS Offload Design

## Summary

This design adds a persistent NIXL GDS transport to `KVCacheManagerV2`. It
preserves V2's existing GPU, host, and disk levels and supports:

- GPU to/from host through the existing CUDA copy engine.
- GPU to/from disk through NIXL `VRAM_SEG <-> FILE_SEG` transfers.
- Host to/from disk through NIXL `DRAM_SEG <-> FILE_SEG` transfers.
- Direct GPU-to-disk bypass when a cold page should not consume host capacity.
- Batched, transactional page migration through
  `StorageManager._batched_migrate()`.

The design is limited to `tensorrt_llm/runtime/kv_cache_manager_v2`. It does
not route migrations through the legacy C++ `KVCacheTransferManager` path. It
does extend the shared NIXL loopback descriptors so V2 can use borrowed disk
file descriptors and nonzero slot offsets.

## Implemented slice

The first implementation slice provides the direct data path while preserving
the current synchronous V2 migration contract:

- `StorageManager._batched_migrate()` selects the configured disk backend.
- One reusable NIXL `GDS_MT` loopback agent is created per thread-count value.
- GPU/disk and host/disk tasks are submitted to NIXL in pool-sized batches.
- GPU/disk transfers bypass V2's 64 MiB pinned-host staging buffer.
- Host/disk transfers use NIXL `DRAM_SEG <-> FILE_SEG` descriptors.
- Disk/disk and all non-disk transfers retain their existing paths.
- NIXL registrations are currently scoped to one batch, and completion is
  awaited before `batched_copy()` returns.
- If NIXL cannot register the file or memory descriptors for GDS, the existing
  NIXL loopback behavior falls back to positional POSIX I/O. GPU transfers in
  that fallback use an internal host buffer.

Persistent registrations, asynchronous migration tickets, transactional
rollback, and backend telemetry remain follow-up work described below.

## Architecture

### Current V2 structure

```text
+----------------------+
| KVCacheV2Scheduler   |
| capacity scheduling  |
| suspend/resume       |
+----------+-----------+
           |
           v
+----------------------+
| KVCacheManagerV2     |
| _KVCache objects     |
| radix-tree reuse     |
| lifecycle registry   |
+----------+-----------+
           |
           v
+----------------------+
| StorageManager       |
| prepare_free_slots() |
| _batched_migrate()   |
| release_slot()       |
+----------+-----------+
           |
           +-------------------+-------------------+
           |                   |                   |
           v                   v                   v
+------------------+  +------------------+  +------------------+
| GPU level        |  | HOST level       |  | DISK level       |
| GPU slots        |  | Host slots       |  | Disk slots       |
| CUDA events      |  | CPU buffers      |  | File-backed data |
+------------------+  +------------------+  +------------------+
           ^                   ^                   ^
           |                   |                   |
           +-------------------+-------------------+
                               |
                     +---------+---------+
                     | CopyEngine        |
                     | page migration    |
                     +-------------------+
```

### Proposed V2 structure

```text
+----------------------+
| KVCacheV2Scheduler   |
| existing scheduling  |
| and suspend/resume   |
+----------+-----------+
           |
           v
+----------------------+
| KVCacheManagerV2     |
| no GDS-specific      |
| allocation policy    |
+----------+-----------+
           |
           v
+-------------------------------+
| StorageManager                |
| prepare_free_slots()          |
| _batched_migrate()            |
| migration commit/rollback     |
+---------------+---------------+
                |
       Select transport by level pair
                |
       +--------+---------+-------------------+
       |                  |                   |
       v                  v                   v
+--------------+   +--------------+   +------------------------+
| GPU <-> HOST |   | HOST <-> DISK|   | GPU <-> DISK          |
| CUDA copy    |   | NIXL GDS or  |   | NIXL GDS              |
| engine       |   | POSIX fallback|  | direct storage path    |
+--------------+   +--------------+   +-----------+------------+
                                                  |
                                                  v
                                      +------------------------+
                                      | NixlStorageCopyEngine  |
                                      | persistent NIXL agent  |
                                      | GDS_MT backend         |
                                      | persistent registration|
                                      | batched requests       |
                                      +-----------+------------+
                                                  |
                                                  v
                                      +------------------------+
                                      | Preallocated arenas    |
                                      | per rank/device/group  |
                                      +------------------------+
```

The copy engine provides transport only. Page eviction policy, prefix reuse,
priorities, request scheduling, and slot allocation remain owned by V2.

## Supported Migration Edges

| Migration | Transport |
|---|---|
| GPU to host | Existing CUDA device-to-host copy |
| Host to GPU | Existing CUDA host-to-device copy |
| GPU to disk | NIXL `VRAM_SEG -> FILE_SEG` |
| Disk to GPU | NIXL `FILE_SEG -> VRAM_SEG` |
| Host to disk | NIXL `DRAM_SEG -> FILE_SEG` |
| Disk to host | NIXL `FILE_SEG -> DRAM_SEG` |

GPU-to-disk is the GPUDirect Storage path. Host-to-disk may use the same NIXL
GDS backend and cuFile APIs, but it is not GPU-direct and must be benchmarked
against an optimized POSIX or `io_uring` path.

## Persistent Storage and Registration

Create one arena per session, rank, device, and pool group:

```text
<root>/<session>/rank-<rank>/device-<device>/pool-group-<group>.bin
```

Map V2 disk slots to stable file ranges:

```text
file_offset = disk_slot.slot_id * aligned_slot_stride[pool_group]
transfer_size = slot_payload_size[pool_group]
```

At initialization:

1. Open and preallocate each disk arena.
2. Register arena descriptors as NIXL `FILE_SEG` regions.
3. Register stable GPU ranges as `VRAM_SEG` regions.
4. Register persistent host-cache ranges as `DRAM_SEG` regions.
5. Create one local NIXL agent with a `GDS_MT` backend.

Registration and file-open operations must not occur per page. CUDA VMM
allocations must be split at actual allocation boundaries. If V2 dynamically
remaps GPU backing, registrations must follow map/unmap lifetime through a
refcounted registry.

## Migration Transactions

Represent each asynchronous migration with a ticket:

```text
MigrationTicket
    source_level
    destination_level
    pool_group
    pages[]
    source_slots[]
    destination_slots[]
    nixl_request
    state:
        PREPARED
        SUBMITTED
        COMPLETED
        COMMITTED
        FAILED
```

While a migration is in flight:

- Source and destination slots remain reserved.
- The page cannot be evicted, locked for execution, or migrated again.
- The page's visible level and slot remain unchanged until commit.
- GPU capacity is not reported as reclaimed until offload commits.
- An onboarded page is not published in a request page table until commit.

On success, move the page to the destination slot, update its level, publish
readiness, release the source slot, and restore eviction registration. On
failure, release the destination, preserve the source, restore prior eviction
state, and surface a tier-specific error.

## Direct GPU-to-Disk Offload

Direct GPU-to-disk migration bypasses host cache memory:

```text
GPU CacheLevelStorage                         Disk CacheLevelStorage
+----------------------+                     +----------------------+
| GPU page slot        |                     | Disk page slot       |
| GPU address + length |                     | arena fd + offset    |
+----------+-----------+                     +-----------+----------+
           |                                             ^
           | NIXL GDS: VRAM_SEG -> FILE_SEG              |
           +---------------------------------------------+
                         no host-cache copy
```

### Offload sequence

```text
Scheduler       EvictionController   StorageManager    NIXL/GDS      NVMe
    |                   |                  |                |           |
    | need GPU capacity |                  |                |           |
    |------------------>| choose GPU page  |                |           |
    |                   | target=DISK      |                |           |
    |                   |----------------->| reserve disk slot           |
    |                   |                  | wait for last GPU use        |
    |                   |                  | create MigrationTicket       |
    |                   |                  | submit NIXL_WRITE            |
    |                   |                  |--------------->| cuFileWrite|
    |                   |                  |                |==========>|
    |                   |                  |<---------------| future    |
    | scheduler processes independent work while transfer is pending    |
    |                   |                  |                              |
    | need reclaimed GPU capacity         |                              |
    |------------------------------------->| poll/wait ticket              |
    |                   |                  |<---------------| complete   |
    |                   |                  | commit GPU -> DISK            |
    |                   |                  | release GPU slot              |
    |<-------------------------------------| capacity available            |
```

### CUDA ordering

NIXL GDS work is not automatically ordered with the model CUDA stream. Before
submitting a write, the GDS worker must wait for the page's last-use event:

```text
model CUDA stream
    +-- attention uses page
    +-- record page.finish_event
                              \
                               \ wait
                                v
GDS worker
    +-- wait(page.finish_event)
    +-- submit NIXL_WRITE(GPU address -> disk offset)
```

The GPU slot is released only after successful NIXL completion.

### Batched offload

```text
_batched_migrate(
    pool_group=0,
    source=GPU,
    destination=DISK,
    pages=[P0, P1, P2, P3],
)
        |
        v
one NIXL GDS_MT request
    +-- P0 GPU range -> disk slot 19
    +-- P1 GPU range -> disk slot 20
    +-- P2 GPU range -> disk slot 21
    +-- P3 GPU range -> disk slot 22
```

Coalesce descriptors only when both GPU ranges and disk offsets are
contiguous.

## Disk-to-GPU Onboarding

`batched_lock_to_gpu()` remains the onboarding entry point:

```text
batched_lock_to_gpu(tasks)
    +-- hold pages and exclude them from eviction
    +-- calculate GPU requirements per pool group
    +-- prepare_free_slots(GPU_LEVEL, requirements)
    +-- partition by (source_level, pool_group)
    +-- _batched_migrate(DISK -> GPU)
    +-- wait for pages required by this request
    +-- commit destination pages
    +-- publish GPU page indices
    +-- return shared GPU page locks
```

```text
Scheduler      KVCacheManagerV2   StorageManager   NIXL/GDS      GPU
    |                  |                |               |           |
    | prepare request  |                |               |           |
    |----------------->| find disk pages|               |           |
    |                  |--------------->| reserve GPU slots          |
    |                  |                | batch by pool group        |
    |                  |                |-------------->| cuFileRead|
    |                  |                |               |==========>|
    |                  |                |<--------------| future    |
    |                  |                | wait required tickets      |
    |                  |                |<--------------| complete  |
    |                  |                | commit DISK -> GPU          |
    |                  |<---------------| publish page indices       |
    |<-----------------| request ready  |               |           |
```

Use disk-to-GPU for pages immediately required by execution. Use disk-to-host
for speculative prefetch that should not consume GPU capacity yet.

## Host-to-Disk Offload

Host cache can still be demoted to disk when GDS is enabled:

```text
HOST CacheLevelStorage                        Disk CacheLevelStorage
+----------------------+                     +----------------------+
| Host page slot       |                     | Disk page slot       |
| DRAM/pinned address  |                     | arena fd + offset    |
+----------+-----------+                     +-----------+----------+
           |                                             ^
           | NIXL: DRAM_SEG -> FILE_SEG                  |
           +---------------------------------------------+
```

This path uses the same arenas, tickets, batching, and rollback rules as
GPU-to-disk. Pinned host memory is preferred. Its performance must be compared
with the non-GDS host-storage path because no GPU bounce copy is eliminated.

## Tiering Policy

Preserve all three demotion paths:

```text
GPU -> HOST     warm-page demotion
HOST -> DISK    normal cold-page demotion
GPU -> DISK     cold-page or host-pressure bypass
```

| Condition | Migration |
|---|---|
| Host has capacity and page is likely to be reused soon | GPU to host |
| GPU page is low priority or expected to remain cold | GPU to disk |
| Host tier is disabled | GPU to disk |
| Host is full and GPU capacity is urgently required | GPU to disk |
| Host page becomes cold | Host to disk |
| Disk page is immediately required | Disk to GPU |
| Disk page is prefetched speculatively | Disk to host |

Initial bypass policy:

```python
if host_has_capacity and page.priority >= disk_bypass_priority:
    destination = HOST_LEVEL
else:
    destination = DISK_LEVEL
```

Direct GPU-to-disk is valuable under host pressure:

```text
without bypass:                 with bypass:

HOST victim -> DISK             GPU victim -> DISK
GPU victim  -> HOST             GPU slot becomes free
GPU slot becomes free
```

The eviction controller selects the destination. The copy engine only
implements migration edges.

## Copy-Engine Interface

```python
class CopyEngine:
    def supports(
        self,
        source_level: CacheLevel,
        destination_level: CacheLevel,
    ) -> bool: ...

    def submit(
        self,
        pool_group: int,
        copies: Sequence[SlotCopy],
    ) -> MigrationTicket: ...

    def poll(self, ticket: MigrationTicket) -> TransferState: ...

    def wait(self, ticket: MigrationTicket) -> None: ...

    def shutdown(self) -> None: ...
```

`NixlStorageCopyEngine` owns the NIXL agent, GDS_MT backend, registrations,
request handles, completion polling, descriptor batching, coalescing, and
telemetry. It does not own eviction, scheduling, prefix reuse, page priority,
or slot allocation.

## Configuration

Extend the V2 configuration rather than request retention configuration:

```yaml
kv_cache_config:
  use_v2_kv_cache: true
  host_cache_size: 128GiB
  disk_cache_size: 2TiB
  disk_cache_path: /mnt/nvme/trtllm-kv
  disk_cache_backend: nixl_gds       # nixl_gds | posix
  disk_cache_gds_thread_count: 16
```

Rules:

- Configuration is global to the V2 manager, not per request.
- GPU, host, and disk levels may coexist.
- `posix` preserves the existing host-staged GPU/disk behavior.
- `nixl_gds` requests direct transfers and retains NIXL's POSIX compatibility
  fallback if GDS registration is unavailable.
- `disk_cache_gds_thread_count` must be positive and is passed to the NIXL
  `GDS_MT` backend as `thread_count`.
- Disk capacity participates in V2 slot-count and constraint calculations.

### Slurm and Pyxis deployment

GDS inside a Pyxis/Enroot container requires the host udev database in
addition to the disk-cache mount. On the validated `batch_3` configuration,
mounting `/scratch` without `/run/udev` causes `cuFileHandleRegister()` to
return error 5027 (`CU_FILE_HANDLE_NOT_REGISTERED`). The same `gdsio` binary
then reports `file descriptor is not registered`, even though native `gdsio`
on the host reports `XferType: GPUD`.

Mount `/run/udev` read-only and use a shared, writable `/scratch` mount:

```bash
#SBATCH --container-image=/path/to/trtllm.sqsh
#SBATCH --container-mounts=/scratch:/scratch:rw+rshared,/run/udev:/run/udev:ro

export TLLM_KVCACHE_V2_GDS_PATH="/scratch/pyxis/runtime/$(id -u)"
```

The `/run/udev/data` database lets cuFile resolve the mounted RAID device to
its underlying NVMe devices and select the GDS path. The mount is read-only;
privileged container execution and a host cuFile override were not required
on the validated system.

Do not use `/scratch` itself as the test or cache directory unless the job has
write permission. The cluster root is commonly owned by `root`; use a
per-user or per-job directory on the same filesystem.

Validate the final container environment, not only the host:

```bash
GDS_PATH="/scratch/pyxis/runtime/$(id -u)"

gdsio -D "${GDS_PATH}/gds_test" \
    -d 0 -w 4 -s 500M -i 1M -x 0 -I 1 -T 60

TLLM_KVCACHE_V2_GDS_PATH="${GDS_PATH}" \
python tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py \
    TestNixlGdsCopyEngine.test_gds_gpu_and_host_disk_round_trips -v
```

The `gdsio` result must report `XferType: GPUD`. The KV-cache test must pass,
not skip, and checks `last_transfer_used_gds()` after each GPU/disk and
host/disk operation so a POSIX compatibility fallback cannot produce a false
pass.

The integration test is discovered in standard test runs. It skips with a
specific reason when `TLLM_KVCACHE_V2_GDS_PATH` is unset or the configured
environment cannot register the file with NIXL GDS.

## Observability

Expose:

- Requested and effective disk backend.
- Direct GDS versus compatibility/fallback status.
- Registered GPU, host, and file bytes.
- Migration count and bytes by level pair.
- Descriptor count, batch size, and coalesced bytes.
- Submission, queue, wait, and end-to-end latency.
- Outstanding tickets and failures.
- Disk capacity, valid slots, and evictions.
- GPU-to-disk bypass decisions and reasons.

## Test Plan

V2 unit tests must cover:

- Transport selection for every level pair.
- Stable disk-slot-to-file-offset mapping.
- Persistent registration lifetime.
- GPU VMM boundary splitting and remapping callbacks.
- Pool-group batching and safe descriptor coalescing.
- Source and destination reservation while tickets are pending.
- GPU capacity not reclaimed before offload commit.
- GPU indices not published before onboard commit.
- Exact GPU/disk and host/disk round trips.
- Failed transfer rollback preserving the source page.
- Eviction exclusion for migrating pages.
- Concurrent reuse of one disk page causing one migration.
- Shutdown draining tickets before deregistration.
- Explicit GDS failure and configured POSIX fallback.
- Unchanged GPU/host behavior when disk is disabled.

GPU integration and performance tests must compare:

- GPU-to-disk with GPU-to-host-to-disk.
- Disk-to-GPU with disk-to-host-to-GPU.
- NIXL GDS with POSIX for host-to-disk.
- Prefix reuse and suspend/resume under three-tier pressure.
- Multi-rank arena isolation.
- CPU utilization, NVMe bandwidth, batch size, and request latency.
- Real GDS selection through NIXL/cuFile telemetry.
- Standard-test behavior with `TLLM_KVCACHE_V2_GDS_PATH` unset, supported, and
  unsupported.
- Pyxis execution with `/run/udev` mounted read-only, ensuring the test passes
  on a GDS-capable local filesystem instead of falling back or skipping.

## Implementation Order

1. Add V2 disk arenas and stable slot offsets.
2. Add `NixlStorageCopyEngine` and startup capability checks.
3. Add persistent GPU, host, and file registrations.
4. Add migration tickets and transactional commit/rollback.
5. Connect GPU-to-disk and disk-to-GPU to `_batched_migrate()`.
6. Connect host-to-disk and disk-to-host to the same engine.
7. Add eviction selection for warm demotion, cold demotion, and bypass.
8. Add metrics, cleanup, correctness tests, and benchmarks.
