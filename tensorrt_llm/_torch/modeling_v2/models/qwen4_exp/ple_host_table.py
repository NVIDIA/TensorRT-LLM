# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the PLE n-gram table lives in host memory, for both Qwen3.8-Flash-Next targets.

The table's rows are cut into `CHUNKS` fixed chunks (`chunk_rows`); a rank of a server of
`world` ranks is responsible for chunks `[rank * CHUNKS / world, (rank + 1) * CHUNKS / world)`.
`PLE_HOST_TABLE_ENV` picks how the rows are stored and looked up:

* `rank` (default): every rank holds the whole table in private host memory.
* `server`: under attention TP every rank holds only the rows of its chunks; the n-gram gather
  reads those rows and writes zeros for the others (`window`), and the caller sums the gathered
  rows over the attention-TP group (`needs_sum`). One copy per server, no inter-process state.
  Exact: every element has one nonzero term.
* `node`: every chunk is one CUDA VMM allocation on host NUMA memory, shared by every process of
  the host through `chunk_broker`; every rank maps all chunks back to back and gathers locally.
  A rank fills the chunks it is responsible for and attaches to the others. One copy per node.
  Processes share a chunk when their tables have one `table_identity`: the bytes of a strided
  sample of every chunk's rows, whatever directory the checkpoint was read from. A rank that
  attaches compares the chunk with its own checkpoint at other sampled rows.

`server` needs attention TP with a size that divides `CHUNKS`, else it runs as `rank`. When a
rank cannot take part in `node` (no VMM host NUMA allocation, no table identity, no owner
within the timeout), every rank of the server runs `server` instead, or `rank`. A requested
mode that does not run is a warning, once per process.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Callable, Optional

import torch
from cuda.bindings import driver as cuda_driver
from cuda.bindings import runtime as cudart

from tensorrt_llm._dlpack_utils import pack_strided_memory
from tensorrt_llm._utils import CUASSERT
from tensorrt_llm.logger import logger

from . import chunk_broker as _broker

#: `rank`, `server` or `node` (see the module docstring); default `rank`.
PLE_HOST_TABLE_ENV = "TRTLLM_MODELING_V2_PLE_HOST_TABLE"
RANK = "rank"
SERVER = "server"
NODE = "node"
_MODES = (RANK, SERVER, NODE)

#: Fixed chunks of the table; every server size that divides it maps whole chunks.
CHUNKS = 8
#: Chunk byte sizes are multiples of this: the VMM host NUMA granularity, so `node` can map the
#: chunks back to back as one table.
CHUNK_ALIGN = 2 << 20
#: How long a rank waits for another process to publish a chunk before giving up on `node`.
_ATTACH_TIMEOUT_S = 3600.0
#: A chunk's sample (`sample_blocks`): this many blocks of `_SAMPLE_ROWS` rows at an even stride.
_SAMPLE_BLOCKS = 256
_SAMPLE_ROWS = 8
_CUDA_MEMORY_TYPE_HOST = 1
#: `(requested, running)` fallbacks this process has warned about.
_WARNED: set = set()

Copy = Callable[[torch.Tensor, int, int, int], None]
Read = Callable[[int, int], torch.Tensor]
Private = Callable[[int, int, torch.dtype, torch.device], tuple]


def host_table_mode() -> str:
    """The requested mode from `PLE_HOST_TABLE_ENV`."""
    value = os.environ.get(PLE_HOST_TABLE_ENV, RANK).strip().lower() or RANK
    if value not in _MODES:
        raise ValueError(f"{PLE_HOST_TABLE_ENV}={value!r}: expected one of {', '.join(_MODES)}")
    return value


def _warn_fallback(requested: str, running: str, why: str) -> None:
    """Warn, once per process and pair, that the requested mode runs as `running`, and why: the
    table then takes more host memory than the deployment asked for."""
    if (requested, running) in _WARNED:
        return
    _WARNED.add((requested, running))
    logger.warning(
        f"ple host table: {PLE_HOST_TABLE_ENV}={requested} requested, running {running}: {why}"
    )


def chunk_rows(rows: int, cols: int, align: int = CHUNK_ALIGN) -> int:
    """Rows of one chunk: an even split of `rows` over `CHUNKS`, rounded up so a chunk's bytes
    (`cols` one-byte columns a row) are a multiple of `align`."""
    unit = align // math.gcd(align, cols)
    per = -(-rows // CHUNKS)
    return -(-per // unit) * unit


def chunk_range(k: int, rows: int, cols: int, align: int = CHUNK_ALIGN) -> tuple[int, int]:
    """Global rows `[lo, hi)` of chunk `k` (empty past the table's end)."""
    per = chunk_rows(rows, cols, align)
    return min(k * per, rows), min((k + 1) * per, rows)


def owned_chunks(rank: int, world: int) -> range:
    """The chunks rank `rank` of `world` is responsible for."""
    assert world > 0 and CHUNKS % world == 0, (rank, world)
    per = CHUNKS // world
    return range(rank * per, (rank + 1) * per)


def owned_rows(rank: int, world: int, rows: int, cols: int, align: int = CHUNK_ALIGN) -> tuple:
    """Global rows `[lo, hi)` of rank `rank`'s chunks."""
    chunks = owned_chunks(rank, world)
    return chunk_range(chunks[0], rows, cols, align)[0], chunk_range(chunks[-1], rows, cols, align)[
        1
    ]


def sample_blocks(k: int, rows: int, cols: int, align: int = CHUNK_ALIGN, phase: int = 0) -> list:
    """Row blocks `[lo, hi)` of chunk `k` that `node` compares: `_SAMPLE_BLOCKS` blocks of
    `_SAMPLE_ROWS` rows at an even stride, `phase` 1 halfway between phase 0's; the whole chunk
    when it is too short to stride."""
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


def table_identity(read: Read, rows: int, cols: int, *parts, align: int = CHUNK_ALIGN) -> str:
    """The `node` key of a `rows` x `cols` table that `read` returns rows of: a digest of
    `parts` and of every chunk's phase-0 sample (`sample_blocks`). Two checkpoints share it only
    if their tables agree at those rows, wherever each was read from."""
    digest = hashlib.sha256(f"{rows}x{cols}".encode())
    for part in parts:
        digest.update(f"\0{part}".encode())
    for k in range(CHUNKS):
        blocks = sample_blocks(k, rows, cols, align)
        if blocks:
            digest.update(_sample(read, blocks).numpy())
    return digest.hexdigest()[:32]


def shard_rows(weights, keys, rows_per_shard: int, read_entry) -> tuple[Copy, Read]:
    """`(copy, read)` over a table stored as equal contiguous row shards `keys`.

    `copy(dst, lo, hi, base)` writes global rows `[lo, hi)` into `dst`, whose row 0 is global row
    `base`, one shard at a time; `read(lo, hi)` returns those rows. `read_entry(entry, cut)` reads
    rows `cut` of a checkpoint entry."""

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
        flat = pack_strided_memory(self.va, rows * cols, rows * cols, 1, torch.uint8, index)
        return flat.view(rows, cols).view(dtype)

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


def _all_agree(ok: bool) -> bool:
    """Whether every rank of the server reports `ok` (a collective: every rank calls it)."""
    from tensorrt_llm._utils import mpi_allgather, mpi_disabled

    if not mpi_disabled():
        result = mpi_allgather(bool(ok))
        return all(result) if isinstance(result, list) else bool(result)
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return ok
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


class PleHostTable:
    """One rank's storage of the PLE n-gram table (`rows` x `cols` one-byte elements).

    After `fill`: `table` is the view the gather reads, `window` its `row_window` (None for the
    whole table), `needs_sum` whether the caller sums the gathered rows over the attention-TP
    group, and `mode` the mode that runs. `rank` / `world` are this rank's place in the TP group
    and `dp` the attention-DP size; `private(rows, cols, dtype, device)` returns
    `(view, owner)` of the target's private host allocation."""

    def __init__(
        self,
        mode: str,
        rows: int,
        cols: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        rank: int,
        world: int,
        dp: int,
        private: Private,
        identity: Optional[str] = None,
        align: int = CHUNK_ALIGN,
        log: Callable[[str], None] = logger.info,
    ) -> None:
        assert mode in _MODES, mode
        assert torch.empty(0, dtype=dtype).element_size() == 1, dtype
        self.requested = mode
        self.rows, self.cols, self.dtype, self.device = rows, cols, dtype, device
        self.rank, self.world, self.dp = rank, world, dp
        self.align = align
        self.identity = identity
        self._private = private
        self._log = log
        self.mode = RANK
        self.window: Optional[tuple[int, int, int]] = None
        self.table: Optional[torch.Tensor] = None
        self._owner = None
        self._vmm: Optional[_VmmChunks] = None
        self._segments: dict[int, _broker.OwnedSegment] = {}
        self._fds: list[int] = []
        self._node_error: Optional[str] = None
        self._attached = 0
        if device.type != "cuda":
            self.table = torch.empty(rows, cols, dtype=dtype, device=device)
        elif mode == NODE:
            self._declare_node()
        else:
            self.mode = SERVER if mode == SERVER and self._shards() else RANK
            if mode == SERVER and self.mode == RANK and world > 1:
                _warn_fallback(
                    mode,
                    self.mode,
                    f"{SERVER} needs attention TP over a size that divides {CHUNKS} (tp {world}, "
                    f"attention DP {dp})",
                )
            self._declare_private()

    @property
    def needs_sum(self) -> bool:
        return self.mode == SERVER

    def _shards(self) -> bool:
        return self.dp == 1 and self.world > 1 and CHUNKS % self.world == 0

    def _key(self, k: int) -> str:
        return f"{self.identity}-c{k}"

    # -------------------------------------------------------------------- declare --

    def _declare_private(self) -> None:
        if self.mode == SERVER:
            lo, hi = owned_rows(self.rank, self.world, self.rows, self.cols, self.align)
            self.window = (lo, hi, lo)
        else:
            lo, hi = 0, self.rows
        if hi > lo:
            self.table, self._owner = self._private(hi - lo, self.cols, self.dtype, self.device)
        else:
            self.table = torch.empty(0, self.cols, dtype=self.dtype, device=self.device)

    def _declare_node(self) -> None:
        """Map nothing yet; own (create) the chunks this rank is responsible for if no process
        does. A failure is kept for `fill`, where every rank agrees on the mode."""
        if self.identity is None:
            self._node_error = "no table identity"
            return
        try:
            chunk_bytes = chunk_rows(self.rows, self.cols, self.align) * self.cols
            node = device_numa_node(self.device)
            index = (
                self.device.index if self.device.index is not None else torch.cuda.current_device()
            )
            self._vmm = _VmmChunks(index, chunk_bytes, CHUNKS, 0 if node is None else node)
            if CHUNKS % self.world == 0:
                for k in owned_chunks(self.rank, self.world):
                    segment = _broker.try_own(self._key(k))
                    if segment is not None:
                        # Recorded first: if `create` fails, `_abort_node` closes it, and the
                        # processes waiting on it take the chunk over.
                        self._segments[k] = segment
                        self._vmm.create(k)
        except (RuntimeError, OSError) as error:
            self._node_error = str(error)
            self._abort_node()

    # ----------------------------------------------------------------------- fill --

    def fill(self, copy: Copy, read: Read) -> None:
        """Fill (and in `node`, share and gather) the rows this rank stores; ends with `table`,
        `window` and `mode` set. Every rank of the server calls it."""
        if self.device.type != "cuda":
            copy(self.table, 0, self.rows, 0)
            return
        if self.requested != NODE:
            lo, hi, base = self.window or (0, self.rows, 0)
            if hi > lo:
                copy(self.table, lo, hi, base)
            return
        ok = self._node_error is None and self._fill_node(copy, read)
        if _all_agree(ok):
            self.mode = NODE
            return
        reason = self._node_error or "another rank of this server could not share the table"
        self.mode = SERVER if self._shards() else RANK
        _warn_fallback(NODE, self.mode, reason)
        if ok:
            # Keep the full shared view; `server` reads only this rank's rows from it.
            if self.mode == SERVER:
                lo, hi = owned_rows(self.rank, self.world, self.rows, self.cols, self.align)
                self.window = (lo, hi, 0)
            return
        self._abort_node()
        self._declare_private()
        lo, hi, base = self.window or (0, self.rows, 0)
        if hi > lo:
            copy(self.table, lo, hi, base)

    def _fill_node(self, copy: Copy, read: Read) -> bool:
        vmm = self._vmm
        self.table = vmm.view(self.rows, self.cols, self.dtype, self.device)
        try:
            for k in list(self._segments):
                self._fill_and_publish(k, copy)
            for k in range(CHUNKS):
                if k in self._segments:
                    continue
                got = _broker.acquire(self._key(k), _ATTACH_TIMEOUT_S, self._log)
                if got[0] == _broker.OWNER:
                    self._segments[k] = got[1]
                    vmm.create(k)
                    self._fill_and_publish(k, copy)
                    continue
                _, fd, meta = got
                try:
                    if meta.get("chunk_bytes") != vmm.chunk_bytes:
                        raise RuntimeError(f"chunk {k} is {meta.get('chunk_bytes')} bytes")
                    vmm.attach(k, fd)
                finally:
                    os.close(fd)
                self._attached += 1
                self._check_chunk(k, read)
        except (RuntimeError, OSError, TimeoutError) as error:
            self._node_error = str(error)
            self._abort_node()
            return False
        return vmm.complete()

    def _fill_and_publish(self, k: int, copy: Copy) -> None:
        lo, hi = chunk_range(k, self.rows, self.cols, self.align)
        if hi > lo:
            copy(self.table, lo, hi, 0)
        torch.cuda.synchronize(self.device)
        fd = self._vmm.export(k)
        self._fds.append(fd)
        self._segments[k].publish(fd, {"chunk": k, "chunk_bytes": self._vmm.chunk_bytes})

    def _check_chunk(self, k: int, read: Read) -> None:
        """An attached chunk holds the checkpoint's bytes at its phase-1 sample, rows the key does
        not hash; raises `RuntimeError` otherwise."""
        blocks = sample_blocks(k, self.rows, self.cols, self.align, phase=1)
        if not blocks:
            return
        rows = torch.cat([torch.arange(lo, hi) for lo, hi in blocks])
        got = self.table.view(torch.uint8).index_select(0, rows.to(self.device)).cpu()
        want = _sample(read, blocks)
        if not torch.equal(got, want):
            first = int(rows[(got != want).any(1)][0])
            raise RuntimeError(f"shared chunk {k} differs from the checkpoint at row {first}")

    def _abort_node(self) -> None:
        for segment in self._segments.values():
            segment.close()
        self._segments.clear()
        for fd in self._fds:
            os.close(fd)
        self._fds.clear()
        if self._vmm is not None:
            self._vmm.close()
            self._vmm = None
        self.table = None

    # ----------------------------------------------------------------- inspection --

    def resident_on_host(self) -> bool:
        """Whether the table's pages are host memory, from the driver."""
        if self._vmm is not None:
            return self._vmm.resident_on_host()
        if self.table is None or self.table.numel() == 0:
            return True
        (attributes,) = CUASSERT(cudart.cudaPointerGetAttributes(self.table.data_ptr()))
        return int(attributes.type) == _CUDA_MEMORY_TYPE_HOST

    def describe(self) -> str:
        if self.mode == NODE:
            return (
                f"{NODE}: {CHUNKS} shared chunks, this process created {len(self._segments)} and "
                f"attached {self._attached}"
            )
        if self.mode == SERVER:
            lo, hi, _ = self.window
            return f"{SERVER}: rows [{lo}, {hi}) of {self.rows}, summed over {self.world} ranks"
        return f"{RANK}: the whole table"

    def close(self) -> None:
        self._abort_node()
        self._owner = None
