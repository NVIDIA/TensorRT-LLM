# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one-sided MNNVL push of a context-parallel (cep) group: the entry and its workspace.

A cep context server moves rows between its CP ranks (the KDA layers' Ulysses all-to-alls, the
indexer's row moves, the sequence-split KDA's state hand-offs). ``rubin_cp_push`` writes each
destination's row blocks straight into that rank's receive region over MNNVL (fabric memory),
then meets every rank of the group on an epoch flag (``kernels/comm/rubin_cp_push``): no NCCL
communicator, so no NCCL point-to-point buffers per peer, and no staging copy on either side.

The workspace (:func:`rubin_cp_push_workspace`, a runtime-helper builder called once at
construction, collective over the group) is one fabric segment per rank, mapped on every rank of
the group: two receive regions of ``region`` bytes and a flag area. Consecutive pushes alternate
the regions. A push's region holds what it received until the next push is issued: read it, on
the stream of the push, before issuing the next one. Then the region a push writes was last read
before the push two back completed on every rank (a push completes only once every rank has
started it), so no other barrier is needed.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, List, NamedTuple, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell, OpWrapper

from ...kernels.comm.rubin_cp_push import FLAG_BYTES, PushArgs, run_cp_push

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping


@functools.cache
def _push_memory() -> type:
    """The MNNVL memory class of the cep push: the Helix CP grouping (one segment per rank of the
    CP group, ordered by CP rank) with allocation state of its own."""
    from tensorrt_llm._torch.distributed.mnnvl_memory import HelixCpMnnvlMemory

    return type("CpPushMemory", (HelixCpMnnvlMemory,), {"__doc__": _push_memory.__doc__})


@dataclasses.dataclass
class CpPushWorkspace:
    """One rank's view of the push workspace of a CP group (:func:`rubin_cp_push_workspace`).

    ``segment`` is this rank's segment (uint8: two regions, then the flags), ``peers`` the int64
    device array of every rank's segment address on this rank, ``counter`` the push kernel's CTA
    counter, ``pushes`` the pushes issued so far (the epoch of the last one).
    """

    world: int
    rank: int
    region: int
    segment: torch.Tensor
    peers: torch.Tensor
    counter: torch.Tensor
    memory: object
    pushes: int = 0

    @property
    def flags_off(self) -> int:
        """Byte offset of the flag area in every segment."""
        return 2 * self.region


class CpPushBlock(NamedTuple):
    """One row block per destination of a push (bytes; see ``kernels/comm/rubin_cp_push``).

    For destination ``d`` and the push table's ``(s0, d0, n)`` of ``d``, rows ``r < n`` of
    ``row_bytes`` bytes go from ``src + d * src_dest + (s0 + r) * src_row`` (offsets in bytes from
    ``src``'s first element) to ``dst_off + me * dst_src + (d0 + r) * dst_row`` of ``d``'s receive
    region. ``max_rows`` bounds every ``n``.
    """

    src: torch.Tensor
    src_row: int
    src_dest: int
    row_bytes: int
    dst_off: int
    dst_row: int
    dst_src: int
    max_rows: int


# One push of an exchange: ((s0, d0, n) per destination, flat; the most rows of a destination; the
# rows this rank receives).
Push = Tuple[List[int], int, int]


def push_chunks(
    blocks: Sequence[Tuple[int, int, int]], recv_rows: Sequence[int], rank: int, cap: int
) -> List[Push]:
    """The pushes of one exchange: ``blocks[d] = (s0, d0, n)`` sends this rank's rows
    ``[s0, s0 + n)`` to rows ``[d0, d0 + n)`` of rank ``d``'s receive buffer of ``recv_rows[d]``
    rows. Push ``k`` moves rows ``[k cap, (k + 1) cap)`` of every destination's buffer (the same
    number of pushes on every rank)."""
    out = []
    for k in range(max(1, -(-max(recv_rows, default=0) // cap))):
        lo_w, hi_w = k * cap, (k + 1) * cap
        vals, most = [], 0
        for s0, d0, n in blocks:
            lo, hi = max(d0, lo_w), min(d0 + n, hi_w)
            m = max(0, hi - lo)
            vals += [s0 + lo - d0 if m else 0, lo - lo_w if m else 0, m]
            most = max(most, m)
        out.append((vals, most, max(0, min(recv_rows[rank], hi_w) - lo_w)))
    return out


def push_table(blocks: Sequence[tuple[int, int, int]], device) -> torch.Tensor:
    """The int32 device table of a push: ``(s0, d0, n)`` per destination, in rank order."""
    return torch.tensor([x for b in blocks for x in b], dtype=torch.int32, device=device)


class RubinCpPush(OpWrapper):
    """Push ``blocks`` into every destination's next receive region of ``ws``; return this rank's
    region (uint8 ``[ws.region]``) once every rank's blocks for it have landed.

    ``table`` (int32 ``[3 * world]``, device) holds ``(s0, d0, n)`` per destination. Every rank of
    the group makes the same sequence of pushes on ``ws``; the result is valid until the next push
    is issued (see the module docstring).
    """

    ARCHS = frozenset()
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"cep{n}, {what}",
            spec=dict(world=n, blocks=blocks, rows=rows),
        )
        for n in (4, 8)
        for what, blocks, rows in (
            ("indexer row move: FP8 queries and FP32 weights", ((4096,), (128,)), (1, 700, 4097)),
            ("indexer Top-K back", ((8192,),), (1, 700)),
            (
                "KDA Ulysses out: q / k / v / beta / f_a columns of each rank's heads",
                "kda1",
                (1, 513),
            ),
            ("KDA Ulysses back: each rank's heads as columns of the rows' rank", "kda2", (1, 513)),
            ("sequence-split KDA hand-off: one FP32 state", ((3072,),), (2048,)),
        )
    )
    note = """
    One CuTe DSL launch (`kernels/comm/rubin_cp_push.run_cp_push`): the blocks'
    rows are copied by the SMs straight from the sources into the peers' fabric
    segments, then the last CTA signals every rank and waits for every source
    (it spins: not beside another kernel that waits on the other ranks). A push
    is a copy, so the region is bit-exact against the reference.

    Not driven by single-GPU cells: ARCHS is empty. The cells need 4 or 8 ranks;
    `rubin_multirank/test_rubin_cp_push.py` drives them on one VR200 tray (4)
    and two (8).
    """

    def __call__(
        self,
        ws: CpPushWorkspace,
        blocks: Sequence[CpPushBlock],
        table: torch.Tensor,
    ) -> torch.Tensor:
        ws.pushes += 1
        offset = (ws.pushes % 2) * ws.region
        args = [
            PushArgs(
                src=b.src.data_ptr(),
                src_row=b.src_row,
                src_dest=b.src_dest,
                row_bytes=b.row_bytes,
                dst_off=offset + b.dst_off,
                dst_row=b.dst_row,
                dst_src=b.dst_src,
                max_rows=b.max_rows,
            )
            for b in blocks
        ]
        run_cp_push(
            ws.peers.data_ptr(),
            table,
            ws.counter.data_ptr(),
            ws.flags_off,
            ws.pushes,
            ws.rank,
            ws.world,
            args,
        )
        return ws.segment[offset : offset + ws.region]

    def reference(
        self,
        blocks_in_rank_order: Sequence[Sequence[CpPushBlock]],
        tables_in_rank_order: Sequence[torch.Tensor],
        rank: int,
        region: torch.Tensor,
    ) -> torch.Tensor:
        """``region`` (uint8, what rank ``rank``'s region held before) with every source's blocks
        for ``rank`` written into it.

        The documented collective exception to mirroring ``__call__``: the region is not a
        function of what the calling rank holds, so this takes every rank's blocks and table in
        rank order.
        """
        out = region.clone()
        for me, (blocks, table) in enumerate(zip(blocks_in_rank_order, tables_in_rank_order)):
            s0, d0, n = (int(x) for x in table.view(-1, 3)[rank].tolist())
            if not n:
                continue
            for b in blocks:
                src = _bytes_from(b.src)
                rows = src.as_strided(
                    (n, b.row_bytes),
                    (b.src_row, 1),
                    src.storage_offset() + rank * b.src_dest + s0 * b.src_row,
                )
                dst = out.as_strided(
                    (n, b.row_bytes),
                    (b.dst_row, 1),
                    out.storage_offset() + b.dst_off + me * b.dst_src + d0 * b.dst_row,
                )
                dst.copy_(rows)
        return out

    def is_valid(
        self,
        ws: CpPushWorkspace,
        blocks: Sequence[CpPushBlock],
        table: torch.Tensor,
    ) -> None:
        assert table.dtype == torch.int32 and table.is_contiguous(), "table: contiguous int32"
        assert table.numel() == 3 * ws.world, "table: one (s0, d0, n) per destination"
        rows = table.view(-1, 3).tolist()
        for b in blocks:
            assert b.row_bytes % 4 == 0 and b.row_bytes > 0, "rows of whole 4-byte words"
            assert all(n <= b.max_rows for _, _, n in rows), "max_rows bounds every n"
            for d, (s0, d0, n) in enumerate(rows):
                if not n:
                    continue
                first = d * b.src_dest + s0 * b.src_row
                last = d * b.src_dest + (s0 + n - 1) * b.src_row + b.row_bytes
                span = _bytes_from(b.src)
                assert first >= 0 and last <= span.numel(), "a block reads past its storage"
                end = b.dst_off + (ws.world - 1) * b.dst_src + (d0 + n - 1) * b.dst_row
                assert end + b.row_bytes <= ws.region, "a block writes past the region"


rubin_cp_push = RubinCpPush()


def _bytes_from(t: torch.Tensor) -> torch.Tensor:
    """The bytes of ``t``'s storage from its first element on, as a flat uint8 tensor."""
    base = torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage())
    offset = t.storage_offset() * t.element_size()
    return base[offset:]


def rubin_cp_push_workspace(mapping: "Mapping", region: int) -> CpPushWorkspace:
    """Build the push workspace of ``mapping``'s CP group with two receive regions of ``region``
    bytes (rounded up to 256): allocate this rank's fabric segment, map every rank's, zero the
    flags and meet the group. Collective over the CP group: every rank calls it at the same
    point of construction, before the KV cache is sized (the segments are outside PyTorch's
    allocator, so the sizing counts them)."""
    region = -(-region // 256) * 256
    memory_cls = _push_memory()
    memory = memory_cls(mapping, 2 * region + FLAG_BYTES)
    world, rank = mapping.cp_size, mapping.cp_rank
    segments = memory.as_torch_strided_tensor(torch.uint8)
    device = torch.device("cuda", torch.cuda.current_device())
    peers = torch.tensor(
        [memory.ptr + r * memory.rank_stride for r in range(world)],
        dtype=torch.int64,
        device=device,
    )
    segment = segments[rank]
    segment[2 * region :].zero_()
    torch.cuda.synchronize()
    memory_cls.get_comm(mapping).barrier()
    return CpPushWorkspace(
        world=world,
        rank=rank,
        region=region,
        segment=segment,
        peers=peers,
        counter=torch.zeros(4, dtype=torch.int32, device=device),
        memory=memory,
    )


__all__ = [
    "CpPushBlock",
    "CpPushWorkspace",
    "Push",
    "push_chunks",
    "push_table",
    "rubin_cp_push",
    "rubin_cp_push_workspace",
]
