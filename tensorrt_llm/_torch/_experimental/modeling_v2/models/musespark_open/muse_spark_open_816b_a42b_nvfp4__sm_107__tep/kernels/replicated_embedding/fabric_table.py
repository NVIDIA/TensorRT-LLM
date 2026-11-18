# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The replicated embedding table as one fabric mapping that each TP rank holds a part of.

``replicated_embedding`` gives every rank the whole ``[vocab, hidden]`` bf16 table so that a
call is one row gather. The table is instead built as
one contiguous virtual address range on every rank whose physical pages are split over the
TP group: rank ``r`` allocates rows ``[begin_r, end_r)`` in its own device memory, exports
them as a fabric handle, and every rank maps all ranks' parts back to back
(``cuMemMap``). Each rank then holds about ``1 / tp_size`` of the table instead of all of it
(TP4: 0.77 of 3.08 GiB, the rank's shard rows, which the shard parameter used to hold), and
the remaining rows are read over NVLink.

Every consumer of the table (the gather entries, the fused model boundary, the drafter's
noise gathers) keeps its ``table`` tensor: a contiguous bf16 ``[vocab, hidden]`` view with
the same bytes, so the values are the replicated table's, bit for bit; only where the bytes
live changes.

Part boundaries are rounded to the allocation granularity (2 MiB = 128 rows of 16 KiB on
the targets this serves), so a part is not exactly the rank's vocabulary shard: up to one
granule of the shard sits in a neighbour's part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


@dataclass(frozen=True)
class Layout:
    """Which rows of the table each rank's physical part holds, and the part sizes in bytes.

    ``bounds[r]`` .. ``bounds[r + 1]`` are rank ``r``'s rows; every bound but the last is a
    multiple of ``rows_per_granule``, so every part starts on a granule. ``part_bytes[r]`` is
    the part's size rounded up to the granularity; ``reserved_bytes`` is their sum, the
    virtual range every rank maps.
    """

    rows: int
    row_bytes: int
    granularity: int
    bounds: Tuple[int, ...]
    part_bytes: Tuple[int, ...]

    @property
    def rows_per_granule(self) -> int:
        return self.granularity // self.row_bytes

    @property
    def reserved_bytes(self) -> int:
        return sum(self.part_bytes)

    def offset(self, rank: int) -> int:
        """Byte offset of rank ``rank``'s part in the mapped range."""
        return self.bounds[rank] * self.row_bytes


def plan_layout(rows: int, row_bytes: int, world: int, granularity: int) -> Layout:
    """Split ``rows`` rows of ``row_bytes`` bytes into ``world`` granule-aligned parts.

    Rank ``r``'s part starts at ``r * rows / world`` rounded to the nearest granule, so each
    part is within half a granule of an even split and of the rank's vocabulary shard.

    Raises:
        ValueError: if a granule is not a whole number of rows, or a part would be empty.
    """
    if world < 1 or rows < world:
        raise ValueError(f"cannot split {rows} rows over {world} ranks")
    if granularity <= 0 or row_bytes <= 0 or granularity % row_bytes:
        raise ValueError(
            f"the allocation granularity {granularity} B is not a whole number of "
            f"{row_bytes}-byte rows"
        )
    per_granule = granularity // row_bytes
    bounds: List[int] = [0]
    for rank in range(1, world):
        bound = int(round(rank * rows / world / per_granule)) * per_granule
        bounds.append(bound)
    bounds.append(rows)
    part_bytes = []
    for rank in range(world):
        count = bounds[rank + 1] - bounds[rank]
        if count <= 0:
            raise ValueError(f"rank {rank} would hold no rows of the {rows}-row table")
        part_bytes.append(-(-count * row_bytes // granularity) * granularity)
    return Layout(rows, row_bytes, granularity, tuple(bounds), tuple(part_bytes))


def _comm_class():
    """A ``MnnvlMemory`` subclass, used only for its TP communicator (its own cached state).

    ``MnnvlMemory`` groups ranks by PP + CP + MoE-TP, which is the TP group only when the MoE
    runs expert-parallel over the whole TP group; the table needs exactly the TP group.
    """
    from tensorrt_llm._mnnvl_utils import MnnvlMemory

    class _FabricTableComm(MnnvlMemory):
        @classmethod
        def comm_split_color_key(cls, mapping: Mapping) -> Tuple[int, int]:
            """Group ranks by PP + CP (one TP group each), ordering each group by TP rank."""
            return (mapping.pp_rank * mapping.cp_size + mapping.cp_rank, mapping.tp_rank)

    return _FabricTableComm


_COMM_CLASS = None


def _comm(mapping: Mapping):
    global _COMM_CLASS
    if _COMM_CLASS is None:
        _COMM_CLASS = _comm_class()
    return _COMM_CLASS, _COMM_CLASS.get_comm(mapping)


class FabricTable:
    """A ``[rows, hidden]`` tensor mapped on every TP rank, its pages split over the ranks.

    Collective over the TP group (``__init__``). The mapping lives as long as the object;
    ``replicated_embedding.install`` keeps it on the embedding module next to the table view.
    """

    def __init__(self, rows: int, hidden: int, dtype: torch.dtype, mapping: Mapping):
        from tensorrt_llm._mnnvl_utils import _check_cu_result, cuda

        self._cuda = cuda
        self._check = _check_cu_result
        cls, comm = _comm(mapping)
        if comm.Get_size() != mapping.tp_size or comm.Get_rank() != mapping.tp_rank:
            raise RuntimeError(
                f"the fabric communicator is rank {comm.Get_rank()} of {comm.Get_size()}, "
                f"not TP rank {mapping.tp_rank} of {mapping.tp_size}"
            )
        cls.initialize()
        device = torch.cuda.current_device()
        prop = cls.get_allocation_prop(device)
        if prop.requestedHandleTypes != cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC:
            raise RuntimeError("the fabric table needs fabric handles (aarch64 MNNVL)")
        granularity = int(cls.get_allocation_granularity(device))
        element = torch.empty((), dtype=dtype).element_size()
        self.layout = plan_layout(rows, hidden * element, mapping.tp_size, granularity)
        self.rank = mapping.tp_rank
        self.device = device
        self._handles: List = []
        self._mapped: List[Tuple[int, int]] = []
        self._base = 0
        local = None
        error = None
        handle_data = None
        try:
            local = self._check(cuda.cuMemCreate(self.layout.part_bytes[self.rank], prop, flags=0))
            exported = self._check(
                cuda.cuMemExportToShareableHandle(local, prop.requestedHandleTypes, 0)
            )
            handle_data = exported.data
        except RuntimeError as exc:
            error = f"rank {self.rank}: {exc}"
        payloads = comm.allgather((error, handle_data))
        errors = [e for e, _ in payloads if e is not None]
        if errors:
            if local is not None:
                self._check(cuda.cuMemRelease(local))
            raise RuntimeError("fabric table export failed: " + "; ".join(errors))
        try:
            self._base = int(
                self._check(cuda.cuMemAddressReserve(self.layout.reserved_bytes, granularity, 0, 0))
            )
            access = cuda.CUmemAccessDesc()
            access.location = prop.location
            access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            for peer, (_, data) in enumerate(payloads):
                handle = (
                    local
                    if peer == self.rank
                    else self._check(
                        cuda.cuMemImportFromShareableHandle(data, prop.requestedHandleTypes)
                    )
                )
                self._handles.append(handle)
                address = self._base + self.layout.offset(peer)
                size = self.layout.part_bytes[peer]
                self._check(cuda.cuMemMap(address, size, 0, handle, 0))
                self._mapped.append((address, size))
                self._check(cuda.cuMemSetAccess(address, size, [access], 1))
        except RuntimeError:
            self.close()
            if local is not None and local not in self._handles:
                self._check(cuda.cuMemRelease(local))
            raise
        from tensorrt_llm._dlpack_utils import pack_strided_memory

        total = rows * hidden * element
        # Described as bytes and reinterpreted: the DLPack helper encodes bfloat16 as a 16-bit
        # float (kDLFloat), which torch reads back as float16.
        flat = pack_strided_memory(self._base, total, total, 1, torch.uint8, device)
        self.tensor = flat.view(dtype).view(rows, hidden)
        if self.tensor.dtype != dtype:
            raise RuntimeError(f"the fabric table view is {self.tensor.dtype}, not {dtype}")
        self._comm = comm

    @property
    def local_bytes(self) -> int:
        """Bytes of device memory this rank holds for the table."""
        return self.layout.part_bytes[self.rank]

    def local_rows(self) -> Tuple[int, int]:
        """``[begin, end)`` rows whose pages live on this rank."""
        return self.layout.bounds[self.rank], self.layout.bounds[self.rank + 1]

    def barrier(self) -> None:
        """Host barrier over the TP group (after a device synchronize on the caller's side)."""
        self._comm.barrier()

    def close(self) -> None:
        """Unmap and release every part and the address range (idempotent)."""
        cuda = self._cuda
        for address, size in reversed(self._mapped):
            try:
                self._check(cuda.cuMemUnmap(address, size))
            except RuntimeError as exc:
                logger.warning(f"[musespark_open] fabric table: unmap failed: {exc}")
        self._mapped = []
        for handle in self._handles:
            try:
                self._check(cuda.cuMemRelease(handle))
            except RuntimeError as exc:
                logger.warning(f"[musespark_open] fabric table: release failed: {exc}")
        self._handles = []
        if self._base:
            try:
                self._check(
                    cuda.cuMemAddressFree(cuda.CUdeviceptr(self._base), self.layout.reserved_bytes)
                )
            except RuntimeError as exc:
                logger.warning(f"[musespark_open] fabric table: address free failed: {exc}")
            self._base = 0


def fill_from_shard(table: torch.Tensor, shard: torch.Tensor, mapping: Mapping) -> None:
    """Write this rank's vocabulary shard into its rows of ``table`` (some land on peers)."""
    width = shard.shape[0]
    table[mapping.tp_rank * width : (mapping.tp_rank + 1) * width].copy_(shard)


def mismatched_rows(table: torch.Tensor, shard: torch.Tensor, mapping: Mapping, chunk: int) -> int:
    """Rows of ``table`` that differ (any bit) from the all-gathered shards (collective).

    Gathers ``chunk`` shard rows at a time, as ``_gather_table`` does, so it needs no second
    whole table.
    """
    width = shard.shape[0]
    world = mapping.tp_size
    bad = 0
    for begin in range(0, width, chunk):
        part = shard[begin : begin + chunk].contiguous()
        count = part.shape[0]
        gathered = allgather(part, mapping, dim=0).view(torch.int16)
        for rank in range(world):
            want = gathered[rank * count : (rank + 1) * count]
            have = table[rank * width + begin : rank * width + begin + count].view(torch.int16)
            bad += int((want != have).any(dim=1).sum().item())
        del gathered
    return bad


def make_table(
    rows: int, hidden: int, dtype: torch.dtype, mapping: Mapping
) -> Optional[FabricTable]:
    """A ``FabricTable``, or None (with a warning) when this rank cannot build one.

    Not agreed here: the caller agrees the outcome over the TP group.
    """
    try:
        return FabricTable(rows, hidden, dtype, mapping)
    except (RuntimeError, ValueError, ImportError) as exc:
        logger.warning(f"[musespark_open] fabric embedding table unavailable: {exc}")
        return None
