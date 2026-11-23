# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram hash tables in pinned host memory, read by the device through one VMM mapping.

Each table (31.7 GB) is cut into `CHUNKS` chunks; the ranks of a host split them, create and fill
their own, and attach the others through the owner's exported file descriptor, so a host keeps one
copy and every rank maps the whole table at one pointer.
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

from tensorrt_llm._dlpack_utils import pack_strided_memory
from tensorrt_llm.logger import logger

CHUNKS = 8
#: Chunk byte sizes are multiples of the VMM host NUMA granularity, so chunks map back to back.
CHUNK_ALIGN = 2 << 20
_ATTACH_TIMEOUT_S = 3600.0
#: An attached chunk is checked at this many blocks of `_SAMPLE_ROWS` rows.
_SAMPLE_BLOCKS = 256
_SAMPLE_ROWS = 8

Copy = Callable[[torch.Tensor, int, int, int], None]
Read = Callable[[int, int], torch.Tensor]


def chunk_rows(rows: int, cols: int) -> int:
    """Rows of one chunk: `rows` / `CHUNKS`, rounded up so its element count is `CHUNK_ALIGN`-aligned."""
    unit = CHUNK_ALIGN // math.gcd(CHUNK_ALIGN, cols)
    per = -(-rows // CHUNKS)
    return -(-per // unit) * unit


def chunk_range(k: int, rows: int, cols: int) -> tuple[int, int]:
    """Global rows ``[lo, hi)`` of chunk `k` (empty past the table's end)."""
    per = chunk_rows(rows, cols)
    return min(k * per, rows), min((k + 1) * per, rows)


def owned_chunks(rank: int, world: int) -> range:
    """The chunks rank `rank` of `world` creates and fills."""
    assert world > 0 and CHUNKS % world == 0, (rank, world)
    per = CHUNKS // world
    return range(rank * per, (rank + 1) * per)


def sample_blocks(k: int, rows: int, cols: int) -> list:
    """Row blocks ``[lo, hi)`` of chunk `k` compared on attach (the whole chunk if short)."""
    lo, hi = chunk_range(k, rows, cols)
    if hi - lo <= 2 * _SAMPLE_BLOCKS * _SAMPLE_ROWS:
        return [(lo, hi)] if hi > lo else []
    stride = (hi - lo) // _SAMPLE_BLOCKS
    first = lo + stride // 2
    return [
        (start, min(start + _SAMPLE_ROWS, hi))
        for start in range(first, hi, stride)[:_SAMPLE_BLOCKS]
    ]


def shard_rows(weights, keys, rows_per_shard: int, read_entry) -> tuple[Copy, Read]:
    """``(copy, read)`` over a table stored as equal contiguous row shards `keys`.

    ``copy(dst, lo, hi, base)`` writes global rows ``[lo, hi)`` into `dst`, whose row 0 is global
    row `base`; ``read(lo, hi)`` returns those rows; ``read_entry(entry, cut)`` reads a checkpoint
    entry's rows `cut`.
    """

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


def _cu(result):
    err, *values = result
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return values[0] if len(values) == 1 else tuple(values)


class _VmmChunks:
    """`count` host-NUMA chunk allocations mapped back to back for one device, each created here
    or imported from a POSIX fd another process exported."""

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
            # The caller releases `handle`.
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

    def close(self) -> None:
        D = cuda_driver
        for k, handle in self._handles.items():
            D.cuMemUnmap(self.va + k * self.chunk_bytes, self.chunk_bytes)
            D.cuMemRelease(handle)
        self._handles.clear()
        if self.va:
            D.cuMemAddressFree(self.va, self.count * self.chunk_bytes)
            self.va = 0


# A chunk's key names an abstract Unix socket (`\0trtllm-engram-<uid>-<key>`); the first process
# to bind it owns the chunk and serves its fd to same-uid peers (`SCM_RIGHTS`, uid checked through
# `SO_PEERCRED` on both sides). Nothing touches the filesystem, so a crash leaves nothing behind.

OWNER = "owner"
ATTACHED = "attached"

_CRED = struct.Struct("iII")  # struct ucred: pid_t pid, uid_t uid, gid_t gid
_POLL_SECONDS = 1.0
_REPORT_SECONDS = 60.0


def _socket_name(key: str) -> str:
    return f"\0trtllm-engram-{os.getuid()}-{key}"


def _peer_uid(conn: socket.socket) -> int:
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
        # Only `_serve` starts answers: once it has returned, the list is final.
        self._thread.join(_POLL_SECONDS * 5)
        for thread in self._answers:
            thread.join(_POLL_SECONDS * 5)

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                # `close` or a failing `accept`: free the name; unanswered peers take the key over.
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


def _acquire(key: str, timeout_s: float) -> tuple:
    """``(OWNER, segment)`` when no process owns `key`, else ``(ATTACHED, fd, meta)`` once the
    owner publishes; an owner that exits first hands `key` to its waiters."""
    name = _socket_name(key)
    deadline = time.monotonic() + timeout_s
    next_report = time.monotonic() + _REPORT_SECONDS
    while True:
        owned = _try_own(key)
        if owned is not None:
            return OWNER, owned
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
        try:
            conn.connect(name)
        except (ConnectionRefusedError, FileNotFoundError):
            # The owner just exited, or is between `bind` and `listen`.
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
                conn.settimeout(min(remaining, _REPORT_SECONDS))
                try:
                    got = socket.recv_fds(conn, 4096, 1)
                except socket.timeout:
                    if time.monotonic() >= next_report:
                        logger.info(
                            f"engram host table: waiting for the owner of {key!r} to publish it"
                        )
                        next_report = time.monotonic() + _REPORT_SECONDS
                except ConnectionResetError:
                    # The owner closed its listener before accepting this connection.
                    got = (b"", [], 0, None)
        finally:
            conn.close()
        msg, fds, _, _ = got
        if fds:
            return ATTACHED, fds[0], json.loads(msg.decode())
        # Closed without a descriptor: the owner exited or aborted. Retry.


class EngramHostTable:
    """One Engram table (``rows`` x ``cols`` of ``dtype``) in host memory, mapped for ``device``.

    Construct at declare time (reserves this rank's chunks and the table's device VA), then
    `fill` once. ``rank`` / ``world`` place this rank among the host's ranks sharing the table;
    ``name`` (the checkpoint tensor name) identifies it across their processes.
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
    ) -> None:
        if not 0 <= rank < world:
            raise ValueError(f"rank {rank} is not one of {world}")
        if world > 1 and CHUNKS % world:
            raise ValueError(f"{CHUNKS} chunks do not split over {world} ranks")
        self.itemsize = torch.empty(0, dtype=dtype).element_size()
        self.rows, self.cols, self.dtype, self.device = rows, cols, dtype, device
        self.rank, self.world = rank, world
        self.name = name
        self.identity = hashlib.sha256(f"{name}\0{rows}x{cols}\0{dtype}".encode()).hexdigest()[:32]
        self.table: Optional[torch.Tensor] = None
        self._vmm: Optional[_VmmChunks] = None
        #: The chunks this rank created (and fills); `_segments` is the subset it serves.
        self._created: set[int] = set()
        self._segments: dict[int, _OwnedSegment] = {}
        self._fds: list[int] = []
        if device.type != "cuda":
            self.table = torch.empty(rows, cols, dtype=dtype, device=device)
            return
        self.chunk_bytes = chunk_rows(rows, cols) * cols * self.itemsize
        node = device_numa_node(device)
        index = device.index if device.index is not None else torch.cuda.current_device()
        self._vmm = _VmmChunks(index, self.chunk_bytes, CHUNKS, 0 if node is None else node)
        for k in owned_chunks(rank, world):
            if world > 1:
                segment = _try_own(self._key(k))
                if segment is None:
                    # Another process of this host holds the name: attach in `fill`.
                    continue
                # Recorded first so a failing `create` still releases the name in `close`.
                self._segments[k] = segment
            self._vmm.create(k)
            self._created.add(k)

    def _key(self, k: int) -> str:
        return f"{self.identity}-c{k}"

    def fill(self, copy: Copy, read: Read) -> None:
        """Fill this rank's chunks, publish them, attach the rest; `copy`/`read` as `shard_rows`.

        Every rank of the host calls it."""
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
            got = _acquire(self._key(k), _ATTACH_TIMEOUT_S)
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
            self._check_chunk(k, read)
        if not self._vmm.complete():
            missing = sorted(set(range(CHUNKS)) - set(self._vmm._handles))
            raise RuntimeError(f"{self.name}: chunks {missing} are not mapped")

    def _fill_and_publish(self, k: int, copy: Copy) -> None:
        lo, hi = chunk_range(k, self.rows, self.cols)
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
        """Raise `RuntimeError` unless attached chunk `k` holds the checkpoint's sampled bytes."""
        blocks = sample_blocks(k, self.rows, self.cols)
        if not blocks:
            return
        rows = torch.cat([torch.arange(lo, hi) for lo, hi in blocks])
        got = self.table.view(torch.uint8).index_select(0, rows.to(self.device)).cpu()
        want = torch.cat([read(lo, hi).contiguous().view(torch.uint8).cpu() for lo, hi in blocks])
        if not torch.equal(got, want):
            first = int(rows[(got != want).any(1)][0])
            raise RuntimeError(f"shared chunk {k} differs from the checkpoint at row {first}")

    def gather(self, ids: torch.Tensor) -> torch.Tensor:
        """The table's rows `ids`: ``[..., cols]``; a negative id reads row 0, which the caller masks."""
        flat = ids.clamp_min(0).reshape(-1).to(device=self.device, dtype=torch.int64)
        return self.table.index_select(0, flat).view(*ids.shape, self.cols)

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
