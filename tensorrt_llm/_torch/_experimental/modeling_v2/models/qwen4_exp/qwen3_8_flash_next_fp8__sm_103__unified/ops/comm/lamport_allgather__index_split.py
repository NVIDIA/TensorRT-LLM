# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gather of a decode step's QSA selection split by request over the TP group, one launch.

Each rank selects some requests' rows (`_decode_index_selection`); one Triton launch gathers the
int32 selection rows in rank order as a one-shot Lamport exchange over CUDA IPC buffers: each
rank stores its rows into every peer's buffer (NVLink) and its own output, then copies the
peers' rows out of its own buffer as they land. Bitwise the NCCL gather.

Buffers: `[SLOTS, max_words]` int32 a rank, every word `SENTINEL` (int32 min; tokens are >= -1)
until a peer's store lands. Slot protocol: every program reads the call's slot (`state[0]`) and
the previous call's share (`state[1]`) after its grid-dependency wait and takes a ticket
(`state[2]`, acq_rel) after its stores; the last ticket sets the next slot and share and resets
the ticket. Each call clears the previous call's slot back to `SENTINEL`. A peer stores into
call k's slot only after reading this rank's call k - 1 stores, made after call k - 2 completed
(stream order), so call k's slot (call k - 3's) was cleared by call k - 2. Every rank of the
group must make the same sequence of calls (the split depends only on the row count).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber

SLOTS = 3
SENTINEL = -(2**31)
_BLOCK = 1024
_PROGRAMS = 256
#: Words of one selection row: the target's `token_topk + compress_ratio - 1`.
_SELECTION_WIDTH = 2051


if TYPE_CHECKING:
    from tensorrt_llm._ipc_utils import IpcMemory


@dataclass(frozen=True)
class SelectionLamport:
    """One rank's view of the group's buffers: `buffers` the `IpcMemory` (kept alive here; it
    frees this rank's buffer and closes the peers' with the object), `state` the `[4]` int32
    slot state (`state[0]` the next call's slot, `state[1]` the previous call's share in words,
    `state[2]` the ticket counter), `args` the `[8]` int64 the kernel reads: the peers' buffer
    addresses at [0, world), this rank's own at `rank`."""

    rank: int
    world: int
    max_words: int
    buffers: IpcMemory
    state: torch.Tensor
    args: torch.Tensor

    def fits(self, words: int) -> bool:
        """Whether a call gathering `words` int32 in all (every rank's share) fits a slot."""
        return 0 < words <= self.max_words


def build(mapping, max_words: int) -> SelectionLamport:
    """Allocate this rank's buffers and open the peers' over `mapping`'s TP group (CUDA IPC,
    one node, `IpcMemory`). A TP collective: every rank calls `build` at the same point, before
    any CUDA graph capture."""
    from tensorrt_llm._ipc_utils import IpcMemory, cuda
    from tensorrt_llm._torch.distributed.communicator import Distributed

    rank, world = mapping.tp_rank, mapping.tp_size
    assert 2 <= world <= 4, "the kernel addresses 2 to 4 ranks"
    ipc = IpcMemory(mapping, SLOTS * max_words * 4, open_ipc=True)
    assert ipc.open_ipc, "the TP group spans nodes: no CUDA IPC"
    (err,) = cuda.cuMemsetD32(ipc.local_ptr, SENTINEL & 0xFFFFFFFF, SLOTS * max_words)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemsetD32 of the selection buffers: {err!r}")
    dev = torch.device("cuda", torch.cuda.current_device())
    state = torch.zeros(4, dtype=torch.int32, device=dev)
    peers = [int(ptr) for ptr in ipc.peer_ptrs]
    args = torch.tensor(peers + [0] * (8 - world), dtype=torch.int64, device=dev)
    torch.cuda.synchronize()
    # every rank's buffer is filled and opened before any rank's first push
    Distributed.get(mapping).tp_allgather(b"")
    return SelectionLamport(
        rank=rank,
        world=world,
        max_words=max_words,
        buffers=ipc,
        state=state,
        args=args,
    )


@triton.jit
def _dst_words(dst_rows, g, m, width, ROW_MAP: tl.constexpr):
    """Output word of gathered word `g`: itself, or with `ROW_MAP` the same column of the row
    `dst_rows` names for its row."""
    if ROW_MAP:
        # int32 arithmetic (a slot holds < 2^31 words): cheaper than a 64-bit division per word.
        g32 = g.to(tl.int32)
        row = g32 // width
        to = tl.load(dst_rows + row, mask=m, other=0).to(tl.int64)
        return to * width + (g32 - row * width)
    return g


@triton.jit(do_not_specialize=["share", "rank", "slot_words", "width"])
def _lamport_allgather_kernel(
    src,
    dst,
    args,
    state,
    share,
    rank,
    slot_words,
    dst_rows,
    width,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_MAP: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Program p moves words p * BLOCK, (p + P) * BLOCK, ... of each share (`share` words a
    rank): pushes this rank's from `src` to every peer's slot and to `dst`, clears the previous
    call's peer shares of the previous slot, then copies each peer's share out of this rank's
    slot as it lands. With `ROW_MAP`, gathered row `j` (of `width` words) lands in row
    `dst_rows[j]` of `dst` instead of row `j`."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if USE_PDL:
        # With a memory clobber: no load of the predecessor's output (`src`) or of the state is
        # hoisted above the wait.
        gdc_wait_with_memory_clobber()
    slot = tl.load(state)
    prev_share = tl.load(state + 1)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    offs = tl.arange(0, BLOCK)
    local = tl.load(args + rank).to(tl.pointer_type(tl.int32))
    slot_off = slot.to(tl.int64) * slot_words
    own0 = rank.to(tl.int64) * share
    # 1. this rank's share: to every peer's slot (same offset) and to the output
    for start in range(pid * BLOCK, share, nprog * BLOCK):
        w = start + offs
        m = w < share
        out = dst + _dst_words(dst_rows, own0 + w, m, width, ROW_MAP)
        v = tl.load(src + w, mask=m, other=0)
        tl.store(out, v, mask=m)
        for q in tl.static_range(WORLD):
            if q != rank:
                peer = tl.load(args + q).to(tl.pointer_type(tl.int32))
                tl.store(peer + slot_off + own0 + w, v, mask=m)
    # 2. the previous call's peer shares of the previous slot back to the marker
    prev_off = ((slot + 2) % 3).to(tl.int64) * slot_words
    for q in tl.static_range(WORLD):
        if q != rank:
            q0 = q * prev_share.to(tl.int64)
            for start in range(pid * BLOCK, prev_share, nprog * BLOCK):
                w = start + offs
                tl.store(
                    local + prev_off + q0 + w,
                    tl.full([BLOCK], -2147483648, tl.int32),
                    mask=w < prev_share,
                )
    # Every program has read the state (at its start): the last ticket moves it on.
    ticket = tl.atomic_add(state + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == nprog - 1:
        tl.store(state, (slot + 1) % 3)
        tl.store(state + 1, share)
        tl.store(state + 2, 0)
    # 3. each peer's share, once it has landed in this rank's slot, to the output
    for q in tl.static_range(WORLD):
        if q != rank:
            q0 = q * share.to(tl.int64)
            for start in range(pid * BLOCK, share, nprog * BLOCK):
                w = start + offs
                m = w < share
                out = dst + _dst_words(dst_rows, q0 + w, m, width, ROW_MAP)
                v = tl.load(local + slot_off + q0 + w, mask=m, other=0, volatile=True)
                pending = tl.sum(((v == -2147483648) & m).to(tl.int32), axis=0)
                while pending > 0:
                    v = tl.load(local + slot_off + q0 + w, mask=m, other=0, volatile=True)
                    pending = tl.sum(((v == -2147483648) & m).to(tl.int32), axis=0)
                tl.store(out, v, mask=m)


class LamportAllgather(OpWrapper):
    """`src`'s rows from every rank of `lam`'s group concatenated in rank order (the NCCL
    `allgather` with equal shares): `src` is this rank's contiguous int32 `[rows, width]`
    share. Returns a fresh `[world * rows, width]` int32 tensor. With `dst_rows` (int32
    `[world * rows]`, a permutation, the same on every rank) the gathered row `j` is stored at
    row `dst_rows[j]` instead (`out[dst_rows] = gathered`), in the same launch. `lam` comes from
    `build`."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"tep{world}: a served step of {requests} requests, 12 trunk calls at "
            f"{3 * requests} rows and 2 draft calls at {requests}",
            spec=dict(world=world, requests=requests, width=_SELECTION_WIDTH, dtype=torch.int32),
        )
        for world in (2, 4)
        for requests in (16, 32, 64, 128, 192)
    )

    note = """
    Certified bitwise NCCL's gather over a long mixed sequence of each cell's calls (the three
    slots rotating), with and without the row map, eagerly and replayed from a CUDA graph.
    Not certified: groups spanning nodes, world sizes other than 2 and 4, widths other than
    the target's selection rows.
    """

    def __call__(
        self, src: torch.Tensor, lam: SelectionLamport, dst_rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert src.dtype == torch.int32 and src.is_contiguous() and src.ndim == 2
        rows, width = src.shape
        share = rows * width
        assert lam.fits(share * lam.world), (share, lam.world, lam.max_words)
        dst = torch.empty([lam.world * rows, width], dtype=torch.int32, device=src.device)
        if dst_rows is not None:
            assert dst_rows.dtype == torch.int32 and dst_rows.is_contiguous()
            assert dst_rows.numel() == lam.world * rows
        pdl = get_env_enable_pdl()
        _lamport_allgather_kernel[(_PROGRAMS,)](
            src,
            dst,
            lam.args,
            lam.state,
            share,
            lam.rank,
            lam.max_words,
            dst if dst_rows is None else dst_rows,
            width,
            WORLD=lam.world,
            BLOCK=_BLOCK,
            ROW_MAP=dst_rows is not None,
            USE_PDL=pdl,
            num_warps=4,
            launch_pdl=pdl,
        )
        return dst

    def reference(
        self, srcs_in_rank_order: Sequence[torch.Tensor], dst_rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Every rank's `src` concatenated in rank order, its row `j` at row `dst_rows[j]` with
        `dst_rows`. The documented collective exception to mirroring `__call__`."""
        gathered = torch.cat(list(srcs_in_rank_order))
        if dst_rows is None:
            return gathered
        out = torch.empty_like(gathered)
        out[dst_rows.long()] = gathered
        return out


lamport_allgather = LamportAllgather()
