# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where an Engram hash table lives: pinned host memory the device reads through a VMM mapping.

The four Engram tables are BF16 ``[164994380, 96]`` -- 31.7 GB each, 126.7 GB over the four
layers -- so none fits on a device at any member of the deployment, and every unit (core
included) reads them from host memory. This module is that path. It is ported from
``models/qwen4_exp/ple_host_table.py`` and its ``chunk_broker`` (Apache-2.0, NVIDIA) and
narrowed to what Engram needs: host residency is not an option here, so there is no
device-resident or per-rank mode, and no fallback that silently keeps a second copy.

Geometry. The rows are cut into `CHUNKS` chunks of `chunk_rows` rows; the ranks of one host
split the chunks, rank ``r`` of ``world`` owning chunks
``[r * CHUNKS / world, (r + 1) * CHUNKS / world)``. Every rank maps every chunk into one
device VA range, so the gather reads the whole table at one pointer: it creates and fills the
chunks it owns and attaches the others through the owner's file descriptor. One copy per
host; the other host of a two-node deployment keeps its own. ``world`` must divide `CHUNKS`
(the deployment's 4 and 8 do).

The gather is ``torch.index_select`` over the mapped view -- a lookup, so no catalog entry.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import socket
import struct
import threading
import time
from typing import Callable, Optional

import torch
from cuda.bindings import driver as cuda_driver
from cuda.bindings import runtime as cudart

from tensorrt_llm._dlpack_utils import pack_strided_memory
from tensorrt_llm._utils import CUASSERT
from tensorrt_llm.logger import logger

#: Fixed chunks of a table; every host size that divides it maps whole chunks.
CHUNKS = 8
#: Chunk byte sizes are multiples of this: the VMM host NUMA granularity, so the chunks map back
#: to back as one table.
CHUNK_ALIGN = 2 << 20
#: How long a rank waits for another process of the host to publish a chunk.
_ATTACH_TIMEOUT_S = 3600.0
#: A chunk's sample: this many blocks of `_SAMPLE_ROWS` rows at an even stride.
_SAMPLE_BLOCKS = 256
_SAMPLE_ROWS = 8
#: ``cudaMemoryTypeHost``: what `cudaPointerGetAttributes` reports for the mapped table.
_CUDA_MEMORY_TYPE_HOST = 1

Copy = Callable[[torch.Tensor, int, int, int], None]
Read = Callable[[int, int], torch.Tensor]


# ------------------------------------------------------------------------ geometry --


def chunk_rows(rows: int, cols: int, align: int = CHUNK_ALIGN) -> int:
    """Rows of one chunk: an even split of `rows` over `CHUNKS`, rounded up so a chunk's bytes
    are a multiple of `align` for any whole-byte element size. The element count is aligned,
    so the row partition is independent of dtype."""
    unit = align // math.gcd(align, cols)
    per = -(-rows // CHUNKS)
    return -(-per // unit) * unit


def chunk_range(k: int, rows: int, cols: int, align: int = CHUNK_ALIGN) -> tuple[int, int]:
    """Global rows ``[lo, hi)`` of chunk `k` (empty past the table's end)."""
    per = chunk_rows(rows, cols, align)
    return min(k * per, rows), min((k + 1) * per, rows)


def owned_chunks(rank: int, world: int) -> range:
    """The chunks rank `rank` of `world` is responsible for."""
    assert world > 0 and CHUNKS % world == 0, (rank, world)
    per = CHUNKS // world
    return range(rank * per, (rank + 1) * per)


def owned_rows(rank: int, world: int, rows: int, cols: int, align: int = CHUNK_ALIGN) -> tuple:
    """Global rows ``[lo, hi)`` of rank `rank`'s chunks."""
    chunks = owned_chunks(rank, world)
    return (
        chunk_range(chunks[0], rows, cols, align)[0],
        chunk_range(chunks[-1], rows, cols, align)[1],
    )


def sample_blocks(k: int, rows: int, cols: int, align: int = CHUNK_ALIGN, phase: int = 0) -> list:
    """Row blocks ``[lo, hi)`` of chunk `k` that an attaching rank compares: `_SAMPLE_BLOCKS`
    blocks of `_SAMPLE_ROWS` rows at an even stride, `phase` 1 halfway between phase 0's; the
    whole chunk when it is too short to stride."""
    lo, hi = chunk_range(k, rows, cols, align)
    if hi - lo <= 2 * _SAMPLE_BLOCKS * _SAMPLE_ROWS:
        return [(lo, hi)] if hi > lo else []
    stride = (hi - lo) // _SAMPLE_BLOCKS
    first = lo + phase * (stride // 2)
    return [
        (start, min(start + _SAMPLE_ROWS, hi))
        for start in range(first, hi, stride)[:_SAMPLE_BLOCKS]
    ]


def _sample(read: Read, blocks) -> torch.Tensor:
    """The checkpoint's rows at `blocks`, in order, as bytes on the CPU."""
    return torch.cat([read(lo, hi).contiguous().view(torch.uint8).cpu() for lo, hi in blocks])


# ------------------------------------------------------------------------ reading --


def shard_rows(weights, keys, rows_per_shard: int, read_entry) -> tuple[Copy, Read]:
    """``(copy, read)`` over a table stored as equal contiguous row shards `keys`.

    ``copy(dst, lo, hi, base)`` writes global rows ``[lo, hi)`` into `dst`, whose row 0 is
    global row `base`, one shard at a time; ``read(lo, hi)`` returns those rows.
    ``read_entry(entry, cut)`` reads rows `cut` of a checkpoint entry."""

    def pieces(lo: int, hi: int):
        s = lo // rows_per_shard
        while lo < hi:
            a = lo - s * rows_per_shard
            b = min(hi - s * rows_per_shard, rows_per_shard)
            yield lo, read_entry(weights[keys[s]], (slice(a, b),))
            lo += b - a
            s += 1

    def copy(dst: torch.Tensor, lo: int, hi: int, base: int) -> None:
        for start, src in pieces(lo, hi):
            dst[start - base : start - base + src.shape[0]].copy_(src.to(dst.device))

    def read(lo: int, hi: int) -> torch.Tensor:
        return torch.cat([src for _, src in pieces(lo, hi)])

    return copy, read


# ---------------------------------------------------------------------- host NUMA --


def _parse_cpulist(text: str) -> set:
    cpus = set()
    for part in filter(None, text.strip().split(",")):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def device_numa_node(device) -> Optional[int]:
    """The NUMA node of `device`'s local CPUs, or None."""
    try:
        props = torch.cuda.get_device_properties(device)
        pci = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        with open(f"/sys/bus/pci/devices/{pci}/local_cpulist") as fh:
            first = min(_parse_cpulist(fh.read()))
        for name in sorted(os.listdir("/sys/devices/system/node")):
            if name.startswith("node") and name[4:].isdigit():
                with open(f"/sys/devices/system/node/{name}/cpulist") as fh:
                    if first in _parse_cpulist(fh.read()):
                        return int(name[4:])
    except (OSError, AttributeError, RuntimeError, ValueError):
        return None
    return None


# --------------------------------------------------------------------------- VMM --


def _cu(result):
    err, *values = result
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return values[0] if len(values) == 1 else tuple(values)


class _VmmChunks:
    """`count` chunk allocations of `chunk_bytes` on host NUMA memory, mapped back to back for
    one device, each created here or imported from a POSIX fd another process exported."""

    def __init__(self, device_index: int, chunk_bytes: int, count: int, node: int) -> None:
        self.device_index = device_index
        self.chunk_bytes = chunk_bytes
        self.count = count
        self.node = node
        granularity = int(
            _cu(
                cuda_driver.cuMemGetAllocationGranularity(
                    self._prop(node),
                    cuda_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                )
            )
        )
        if chunk_bytes % granularity:
            raise RuntimeError(
                f"chunks of {chunk_bytes} bytes are not a multiple of the {granularity}-byte "
                "VMM host NUMA granularity"
            )
        self.va = int(_cu(cuda_driver.cuMemAddressReserve(count * chunk_bytes, granularity, 0, 0)))
        self._handles: dict[int, object] = {}

    @staticmethod
    def _prop(node: int):
        D = cuda_driver
        prop = D.CUmemAllocationProp()
        prop.type = D.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = D.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.location.id = node
        prop.requestedHandleTypes = (
            D.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        return prop

    def _map(self, k: int, handle, read_only: bool) -> None:
        address = self.va + k * self.chunk_bytes
        _cu(cuda_driver.cuMemMap(address, self.chunk_bytes, 0, handle, 0))
        try:
            self._set_access(address, read_only)
        except BaseException:
            # The caller releases `handle`; nothing stays mapped to it.
            cuda_driver.cuMemUnmap(address, self.chunk_bytes)
            raise
        self._handles[k] = handle

    def _set_access(self, address: int, read_only: bool) -> None:
        D = cuda_driver
        desc = D.CUmemAccessDesc()
        desc.location.type = D.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = self.device_index
        desc.flags = (
            D.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
            if read_only
            else D.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        )
        try:
            _cu(D.cuMemSetAccess(address, self.chunk_bytes, [desc], 1))
        except RuntimeError:
            if not read_only:
                raise
            desc.flags = D.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _cu(D.cuMemSetAccess(address, self.chunk_bytes, [desc], 1))

    def create(self, k: int) -> None:
        handle = _cu(cuda_driver.cuMemCreate(self.chunk_bytes, self._prop(self.node), 0))
        try:
            self._map(k, handle, read_only=False)
        except BaseException:
            cuda_driver.cuMemRelease(handle)
            raise

    def export(self, k: int) -> int:
        D = cuda_driver
        return int(
            _cu(
                D.cuMemExportToShareableHandle(
                    self._handles[k],
                    D.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                    0,
                )
            )
        )

    def attach(self, k: int, fd: int) -> None:
        D = cuda_driver
        handle = _cu(
            D.cuMemImportFromShareableHandle(
                fd, D.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
        )
        try:
            self._map(k, handle, read_only=True)
        except BaseException:
            D.cuMemRelease(handle)
            raise

    def complete(self) -> bool:
        return len(self._handles) == self.count

    def view(self, rows: int, cols: int, dtype, device) -> torch.Tensor:
        index = device.index if device.index is not None else torch.cuda.current_device()
        nbytes = rows * cols * torch.empty(0, dtype=dtype).element_size()
        flat = pack_strided_memory(self.va, nbytes, nbytes, 1, torch.uint8, index)
        return flat.view(dtype).view(rows, cols)

    def resident_on_host(self) -> bool:
        host = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        return all(
            _cu(cuda_driver.cuMemGetAllocationPropertiesFromHandle(h)).location.type == host
            for h in self._handles.values()
        )

    def close(self) -> None:
        D = cuda_driver
        for k, handle in self._handles.items():
            D.cuMemUnmap(self.va + k * self.chunk_bytes, self.chunk_bytes)
            D.cuMemRelease(handle)
        self._handles.clear()
        if self.va:
            D.cuMemAddressFree(self.va, self.count * self.chunk_bytes)
            self.va = 0


# ----------------------------------------------------------------------- handoff --
#
# One process creates and fills a chunk; every other process of the host attaches to the same
# allocation through its file descriptor. A chunk is named by a key whose abstract Unix socket
# (`\0trtllm-engram-<uid>-<key>`) the first process to bind owns; it serves the descriptor to
# every same-uid peer (`SCM_RIGHTS`). Abstract names carry no permissions, so both sides check
# the other's uid through `SO_PEERCRED`. Nothing is written to the filesystem, so a crash
# leaves nothing behind, and a process that cannot bind connects and waits for the owner.
# Ported from ``models/qwen4_exp/chunk_broker.py``.

OWNER = "owner"
ATTACHED = "attached"

_CRED = struct.Struct("iII")  # struct ucred: pid_t pid, uid_t uid, gid_t gid
_POLL_SECONDS = 1.0


def _socket_name(key: str) -> str:
    """The abstract socket name of `key` (per uid)."""
    return f"\0trtllm-engram-{os.getuid()}-{key}"


def _peer_uid(conn: socket.socket) -> int:
    """The uid of the process at the other end of the connected Unix socket `conn`."""
    _, uid, _ = _CRED.unpack(conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _CRED.size))
    return uid


class _OwnedSegment:
    """The owner's side: serves `publish`ed ``(fd, meta)`` to every same-uid peer, until `close`."""

    def __init__(self, server: socket.socket, key: str) -> None:
        self.key = key
        self._server = server
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._fd = -1
        self._meta: dict = {}
        self._answers: list[threading.Thread] = []
        self._thread = threading.Thread(
            target=self._serve, name=f"engram-broker-{key[:12]}", daemon=True
        )
        self._thread.start()

    def publish(self, fd: int, meta: dict) -> None:
        """Make the allocation available. `fd` stays owned by the caller and must stay open."""
        self._fd = fd
        self._meta = dict(meta)
        self._ready.set()

    def close(self) -> None:
        """Stop serving; peers still waiting see the connection close and retry."""
        self._closed.set()
        try:
            # Closing alone does not wake the thread blocked in `accept`; shutting down does.
            self._server.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._server.close()
        # `_serve` alone starts answers: once it has returned, the list of answers is final.
        self._thread.join(_POLL_SECONDS * 5)
        for thread in self._answers:
            thread.join(_POLL_SECONDS * 5)

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                # `close`, or `accept` failing (EMFILE, ENFILE, ENOMEM, ...). Either way the name
                # is freed with the thread: connected peers already answered keep their answer,
                # and every other peer sees its connection reset and owns the key itself.
                self._server.close()
                return
            answer = threading.Thread(target=self._answer, args=(conn,), daemon=True)
            self._answers = [t for t in self._answers if t.is_alive()] + [answer]
            answer.start()

    def _answer(self, conn: socket.socket) -> None:
        with conn:
            try:
                if _peer_uid(conn) != os.getuid():
                    return
                while not self._ready.wait(_POLL_SECONDS):
                    if self._closed.is_set():
                        return
                if self._closed.is_set():
                    return
                socket.send_fds(conn, [json.dumps(self._meta).encode()], [self._fd])
            except OSError:
                return


def _try_own(key: str) -> Optional[_OwnedSegment]:
    """Own `key` if no process does (non-blocking), else None."""
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
    try:
        server.bind(_socket_name(key))
        server.listen(64)
    except OSError as error:
        server.close()
        if error.errno == errno.EADDRINUSE:
            return None
        raise
    return _OwnedSegment(server, key)


def _acquire(
    key: str, timeout_s: float, log: Callable[[str], None] = print, report_s: float = 60.0
) -> tuple:
    """``(OWNER, segment)`` when no process owns `key`, else ``(ATTACHED, fd, meta)`` once the
    owner publishes; an owner that exits, aborts or stops accepting first hands `key` to its
    waiters. Raises `TimeoutError` when no owner has published within `timeout_s`, and
    `PermissionError` when a process of another uid holds the name."""
    name = _socket_name(key)
    deadline = time.monotonic() + timeout_s
    next_report = time.monotonic() + report_s
    while True:
        owned = _try_own(key)
        if owned is not None:
            return OWNER, owned
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
        try:
            conn.connect(name)
        except (ConnectionRefusedError, FileNotFoundError):
            # The owner just exited, or is between `bind` and `listen` (or never listens).
            conn.close()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no owner of {key!r} listens within {timeout_s:.0f} s")
            time.sleep(0.05)
            continue
        got: Optional[tuple] = None
        try:
            owner_uid = _peer_uid(conn)
            if owner_uid != os.getuid():
                raise PermissionError(f"the owner of {key!r} runs as uid {owner_uid}")
            while got is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no allocation for {key!r} within {timeout_s:.0f} s")
                conn.settimeout(min(remaining, report_s))
                try:
                    got = socket.recv_fds(conn, 4096, 1)
                except socket.timeout:
                    if time.monotonic() >= next_report:
                        log(f"engram host table: waiting for the owner of {key!r} to publish it")
                        next_report = time.monotonic() + report_s
                except ConnectionResetError:
                    # The owner closed its listener before accepting this connection.
                    got = (b"", [], 0, None)
        finally:
            conn.close()
        msg, fds, _, _ = got
        if fds:
            return ATTACHED, fds[0], json.loads(msg.decode())
        # The connection closed without a descriptor: the owner exited or aborted. Retry.


# ------------------------------------------------------------------------- table --


class EngramHostTable:
    """One Engram table (``rows`` x ``cols`` elements of ``dtype``) in host memory, mapped for
    ``device`` and read by the gather.

    Construct it at declare time (it reserves this rank's chunks and the whole table's device
    VA), then call `fill` once with the checkpoint's shards. ``rank`` / ``world`` are this
    rank's place among the ranks of the host that share the table; ``name`` identifies the
    table across those processes (the checkpoint's tensor name). ``table`` is the mapped view
    after `fill` -- wrap it in an ``nn.Parameter`` without copying, as the caller does.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        rank: int,
        world: int,
        name: str,
        align: int = CHUNK_ALIGN,
        identity: Optional[str] = None,
        log: Callable[[str], None] = logger.info,
    ) -> None:
        if not 0 <= rank < world:
            raise ValueError(f"rank {rank} is not one of {world}")
        if world > 1 and CHUNKS % world:
            raise ValueError(f"{CHUNKS} chunks do not split over {world} ranks")
        self.itemsize = torch.empty(0, dtype=dtype).element_size()
        self.rows, self.cols, self.dtype, self.device = rows, cols, dtype, device
        self.rank, self.world = rank, world
        self.name = name
        self.align = align
        self.identity = (
            identity or hashlib.sha256(f"{name}\0{rows}x{cols}\0{dtype}".encode()).hexdigest()[:32]
        )
        self._log = log
        self.table: Optional[torch.Tensor] = None
        self._vmm: Optional[_VmmChunks] = None
        #: The chunks this rank created (and will fill); `_segments` is the subset it serves.
        self._created: set[int] = set()
        self._segments: dict[int, _OwnedSegment] = {}
        self._fds: list[int] = []
        self._attached = 0
        if device.type != "cuda":
            self.table = torch.empty(rows, cols, dtype=dtype, device=device)
            return
        self.chunk_bytes = chunk_rows(rows, cols, align) * cols * self.itemsize
        node = device_numa_node(device)
        index = device.index if device.index is not None else torch.cuda.current_device()
        self._vmm = _VmmChunks(index, self.chunk_bytes, CHUNKS, 0 if node is None else node)
        for k in owned_chunks(rank, world):
            if world > 1:
                segment = _try_own(self._key(k))
                if segment is None:
                    # Another process of this host holds the name: attach in `fill`.
                    continue
                # Recorded first: if `create` fails, `close` releases the name, and the
                # processes waiting on it take the chunk over.
                self._segments[k] = segment
            self._vmm.create(k)
            self._created.add(k)

    @property
    def chunk_rows(self) -> int:
        """Rows of one chunk."""
        return chunk_rows(self.rows, self.cols, self.align)

    @property
    def owned_rows(self) -> tuple[int, int]:
        """Global rows ``[lo, hi)`` this rank creates and fills."""
        return owned_rows(self.rank, self.world, self.rows, self.cols, self.align)

    def _key(self, k: int) -> str:
        return f"{self.identity}-c{k}"

    # ----------------------------------------------------------------------- fill --

    def fill(self, copy: Copy, read: Read) -> None:
        """Fill this rank's chunks, publish them, attach the rest; ends with `table` mapped.

        ``copy(dst, lo, hi, base)`` writes global rows ``[lo, hi)`` into `dst`, whose row 0 is
        global row `base`; ``read(lo, hi)`` returns those rows from the checkpoint. Every rank
        of the host calls it."""
        if self.device.type != "cuda":
            copy(self.table, 0, self.rows, 0)
            return
        if self.table is None:
            self.table = self._vmm.view(self.rows, self.cols, self.dtype, self.device)
        for k in sorted(self._created):
            self._fill_and_publish(k, copy)
        for k in range(CHUNKS):
            if k in self._created:
                continue
            got = _acquire(self._key(k), _ATTACH_TIMEOUT_S, self._log)
            if got[0] == OWNER:
                self._segments[k] = got[1]
                self._vmm.create(k)
                self._created.add(k)
                self._fill_and_publish(k, copy)
                continue
            _, fd, meta = got
            try:
                if meta.get("chunk_bytes") != self.chunk_bytes:
                    raise RuntimeError(
                        f"chunk {k} of {self.name} is {meta.get('chunk_bytes')} bytes, this "
                        f"rank expects {self.chunk_bytes}"
                    )
                self._vmm.attach(k, fd)
            finally:
                os.close(fd)
            self._attached += 1
            self._check_chunk(k, read)
        if not self._vmm.complete():
            missing = sorted(set(range(CHUNKS)) - set(self._vmm._handles))
            raise RuntimeError(f"{self.name}: chunks {missing} are not mapped")

    def _fill_and_publish(self, k: int, copy: Copy) -> None:
        lo, hi = chunk_range(k, self.rows, self.cols, self.align)
        if hi > lo:
            copy(self.table, lo, hi, 0)
        torch.cuda.synchronize(self.device)
        if k not in self._segments:
            # `world` 1: nothing to hand the descriptor to.
            return
        fd = self._vmm.export(k)
        self._fds.append(fd)
        self._segments[k].publish(fd, {"chunk": k, "chunk_bytes": self._vmm.chunk_bytes})

    def _check_chunk(self, k: int, read: Read) -> None:
        """An attached chunk holds the checkpoint's bytes at its phase-1 sample, rows the key
        does not hash; raises `RuntimeError` otherwise."""
        blocks = sample_blocks(k, self.rows, self.cols, self.align, phase=1)
        if not blocks:
            return
        rows = torch.cat([torch.arange(lo, hi) for lo, hi in blocks])
        got = self.table.view(torch.uint8).index_select(0, rows.to(self.device)).cpu()
        want = _sample(read, blocks)
        if not torch.equal(got, want):
            first = int(rows[(got != want).any(1)][0])
            raise RuntimeError(f"shared chunk {k} differs from the checkpoint at row {first}")

    # ------------------------------------------------------------------- reading --

    def gather(self, ids: torch.Tensor) -> torch.Tensor:
        """The table's rows `ids` names: ``[..., cols]``. A negative id reads row 0, which the
        caller masks (the HF module's ``masked_fill`` for a position that owns no n-gram)."""
        flat = ids.clamp_min(0).reshape(-1).to(device=self.device, dtype=torch.int64)
        return self.table.index_select(0, flat).view(*ids.shape, self.cols)

    def resident_on_host(self) -> bool:
        """Whether the table's pages are host memory, from the driver."""
        if self.device.type != "cuda":
            return True
        if self.table is None or self.table.numel() == 0:
            return False
        (attributes,) = CUASSERT(cudart.cudaPointerGetAttributes(self.table.data_ptr()))
        return int(attributes.type) == _CUDA_MEMORY_TYPE_HOST

    def describe(self) -> str:
        return (
            f"{self.name}: {CHUNKS} chunks of {self.chunk_rows} rows, rank {self.rank}/"
            f"{self.world} created {sorted(self._created)} and attached {self._attached}"
        )

    def close(self) -> None:
        for segment in self._segments.values():
            segment.close()
        self._segments.clear()
        self._created.clear()
        for fd in self._fds:
            os.close(fd)
        self._fds.clear()
        if self._vmm is not None:
            self._vmm.close()
            self._vmm = None
        self.table = None
