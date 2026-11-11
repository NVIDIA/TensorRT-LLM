# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected pools -> latent-cache row ids of their members and of the incomplete tail, in one
launch.

Pool ``j`` of a row is its request's positions ``[j * kpool, (j + 1) * kpool)``; the row's
``kv_len // kpool`` complete pools are its candidates and the ``kv_len % kpool`` positions after
them (the incomplete tail) are always visible to it. Cache position ``p`` of block table ``t``
is latent row ``base_row + block_tables[t, p // tokens_per_block] * rows_per_slot + p %
tokens_per_block`` of the uniformly strided row view of the latent pool.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _kpool_expand_kernel(
    SEL,
    KV_LENS,
    BT,
    REQ,
    OUT,
    sel_stride,
    bt_stride,
    tpb,
    base_row,
    rows_per_slot,
    out_width,
    SELECT_K: tl.constexpr,
    KPOOL: tl.constexpr,
    KPOOL_P2: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    HAS_REQ: tl.constexpr,
):
    """Expand selected pools into latent-cache row ids and append the tail.

    Output row ``b`` is ``[SELECT_K * KPOOL expanded members | KPOOL - 1 tail
    positions | -1 padding]``; every invalid entry is ``-1`` (the FlashMLA
    sparse kernel's own invalid marker). Row id of cache position ``p`` is
    ``base_row + block_table[p // tpb] * rows_per_slot + p % tpb``.
    """
    b = tl.program_id(0)
    if HAS_REQ:
        tbl = tl.load(REQ + b).to(tl.int64)
    else:
        tbl = b.to(tl.int64)
    kv_len = tl.load(KV_LENS + b).to(tl.int64)
    num_cand = kv_len // KPOOL

    i = tl.arange(0, SELECT_K)
    j = tl.load(SEL + b * sel_stride + i).to(tl.int64)
    # A radix top-k pads short rows with negative ids; those are invalid too.
    valid = (j >= 0) & (j < num_cand)
    m = tl.arange(0, KPOOL_P2)
    mmask = m < KPOOL
    pos = j[:, None] * KPOOL + m[None, :]
    page = pos // tpb
    off = pos - page * tpb
    ok = valid[:, None] & mmask[None, :]
    slot = tl.load(BT + tbl * bt_stride + page, mask=ok, other=0).to(tl.int64)
    rows = base_row + slot * rows_per_slot + off
    rows = tl.where(ok, rows, -1)
    tl.store(
        OUT + b * out_width + i[:, None] * KPOOL + m[None, :],
        rows.to(tl.int32),
        mask=mmask[None, :] & (i[:, None] >= 0),
    )

    # Tail: the incomplete trailing group, always visible to its own query.
    tail_count = kv_len - num_cand * KPOOL
    tail_start = kv_len - tail_count
    t = tl.arange(0, KPOOL_P2)
    tpos = tail_start + t
    tvalid = t < tail_count
    tpage = tpos // tpb
    toff = tpos - tpage * tpb
    tslot = tl.load(BT + tbl * bt_stride + tpage, mask=tvalid, other=0).to(tl.int64)
    trows = tl.where(tvalid, base_row + tslot * rows_per_slot + toff, -1)
    tidx = SELECT_K * KPOOL + t
    tl.store(OUT + b * out_width + tidx, trows.to(tl.int32), mask=tidx < out_width)

    # -1 padding up to the kernel-aligned width.
    pad0 = SELECT_K * KPOOL + KPOOL_P2
    for start in range(pad0, out_width, PAD_BLOCK):
        p = start + tl.arange(0, PAD_BLOCK)
        tl.store(OUT + b * out_width + p, tl.full([PAD_BLOCK], -1, tl.int32), mask=p < out_width)


class KpoolExpand(OpWrapper):
    """``selected`` int32 ``[N, select_k]`` pool ids -> int32 ``[N, out_width]`` latent row ids.

    Column ``i * kpool + m`` holds member ``m`` of ``selected[n, i]`` when that id is a
    candidate of row ``n`` (``0 <= id < kv_lens[n] // kpool``), else ``-1``; columns
    ``select_k * kpool + t`` hold the tail positions ``(kv_lens[n] // kpool) * kpool + t`` for
    ``t < kv_lens[n] % kpool``, then ``-1`` to ``out_width``. The order of ``selected`` is kept.
    Row ``n`` reads block table ``request_ids[n]`` when given, else table ``n``.
    """

    ARCHS = frozenset({Arch.SM_100})
    # select_k = index_topk / index_kpool = 512 pools of kpool 4, out_width the padded
    # kernel_output_width 2112, 32-token blocks. `kv` is the prompt length (context cells) or
    # the requests' KV length after the step (generation cells).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP-3 verify of one 13K-token request",
            spec=dict(requests=1, tokens=4, kv=13312),
        ),
        Cell(
            why="MTP-3 verify of eight 13K-token requests",
            spec=dict(requests=8, tokens=4, kv=13312),
        ),
        Cell(
            why="MTP-3 verify of 64 requests, the largest CUDA-graph batch",
            spec=dict(requests=64, tokens=4, kv=13312),
        ),
        Cell(
            why="single-token decode of eight requests: tables by row, no request ids",
            spec=dict(requests=8, tokens=1, kv=13312),
        ),
        Cell(
            why="one rank's sequence-parallel quarter of a 13,312-token prompt (rows 0-3327)",
            spec=dict(context=((0, 13312),), rows=(0, 3328)),
        ),
        Cell(
            why="a 13,312-token prompt's last sequence-parallel quarter (rows 9984-13311)",
            spec=dict(context=((0, 13312),), rows=(9984, 13312)),
        ),
        Cell(
            why="rank 1's quarter (rows 10-19) of a 37-token prompt: short rows",
            spec=dict(context=((0, 37),), rows=(10, 20)),
        ),
    )

    note = """
    The caller's `selected` comes from the decode top-k: candidate pool ids in an
    unspecified order, padded with -1 when a row has fewer than select_k candidates. Ids at
    or past a row's candidate count are written as -1 too.

    The kernel spans the selection with `tl.arange(0, select_k)`, so select_k is a power of
    two (512 for this model); `out_width < select_k * kpool + kpool - 1` raises.
    """

    def __call__(
        self,
        selected: torch.Tensor,
        kv_lens: torch.Tensor,
        block_tables: torch.Tensor,
        tokens_per_block: int,
        *,
        base_row: int,
        rows_per_slot: int,
        kpool: int,
        out_width: int,
        request_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, select_k = selected.shape
        if out_width < select_k * kpool + kpool - 1:
            raise ValueError(
                f"kpool_expand: out_width {out_width} < {select_k * kpool + kpool - 1}"
            )
        out = torch.empty(n, out_width, dtype=torch.int32, device=selected.device)
        if n == 0:
            return out
        _kpool_expand_kernel[(n,)](
            selected,
            kv_lens,
            block_tables,
            kv_lens if request_ids is None else request_ids,
            out,
            selected.stride(0),
            block_tables.stride(0),
            tokens_per_block,
            base_row,
            rows_per_slot,
            out_width,
            SELECT_K=select_k,
            KPOOL=kpool,
            KPOOL_P2=triton.next_power_of_2(kpool),
            PAD_BLOCK=64,
            HAS_REQ=request_ids is not None,
            num_warps=4,
        )
        return out

    def reference(
        self,
        selected: torch.Tensor,
        kv_lens: torch.Tensor,
        block_tables: torch.Tensor,
        tokens_per_block: int,
        *,
        base_row: int,
        rows_per_slot: int,
        kpool: int,
        out_width: int,
        request_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The member and tail positions of each row, mapped through its block table."""
        n, select_k = selected.shape
        dev = selected.device
        out = torch.full((n, out_width), -1, dtype=torch.int32, device=dev)
        if n == 0:
            return out
        table = torch.arange(n, device=dev) if request_ids is None else request_ids.long()
        kv = kv_lens.long()
        num_cand = kv // kpool
        last_page = block_tables.shape[1] - 1

        def row_ids(positions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            page = (positions // tokens_per_block).clamp(0, last_page)
            slot = block_tables[table.view(-1, *([1] * (positions.dim() - 1))), page].long()
            ids = base_row + slot * rows_per_slot + positions % tokens_per_block
            return torch.where(valid, ids, -1).to(torch.int32)

        pools = selected.long()
        member = torch.arange(kpool, device=dev)
        positions = pools[:, :, None] * kpool + member  # [n, select_k, kpool]
        valid = ((pools >= 0) & (pools < num_cand[:, None]))[:, :, None].expand_as(positions)
        out[:, : select_k * kpool] = row_ids(positions.clamp(min=0), valid).view(n, -1)
        tail = num_cand[:, None] * kpool + member  # [n, kpool]
        width = min(kpool, out_width - select_k * kpool)
        tail_ids = row_ids(tail, tail < kv[:, None])
        out[:, select_k * kpool : select_k * kpool + width] = tail_ids[:, :width]
        return out

    def is_valid(
        self,
        selected: torch.Tensor,
        kv_lens: torch.Tensor,
        block_tables: torch.Tensor,
        tokens_per_block: int,
        *,
        base_row: int,
        rows_per_slot: int,
        kpool: int,
        out_width: int,
        request_ids: torch.Tensor | None = None,
    ) -> None:
        n, select_k = selected.shape
        assert selected.dtype == torch.int32 and selected.stride(1) == 1, (
            f"selected must be int32 with unit-stride rows; got {selected.dtype} "
            f"{selected.stride()}"
        )
        for name, width in (("select_k", select_k), ("kpool", kpool)):
            assert width > 0 and width & (width - 1) == 0, f"{name} {width} must be a power of 2"
        assert kv_lens.shape == (n,)
        assert block_tables.dim() == 2 and block_tables.stride(1) == 1
        if n == 0:
            return
        if request_ids is None:
            assert block_tables.shape[0] >= n, "row n reads block table n"
        else:
            assert request_ids.shape == (n,)
            assert 0 <= int(request_ids.min()) and int(request_ids.max()) < block_tables.shape[0]
        # A visible position past the table's width reads the next table's pages.
        widest = int(kv_lens.max())
        assert widest <= block_tables.shape[1] * tokens_per_block, (
            f"a visible length of {widest} exceeds the tables' "
            f"{block_tables.shape[1] * tokens_per_block} positions"
        )
        # Row ids are int32.
        top = base_row + (int(block_tables.max()) + 1) * rows_per_slot
        assert top < 2**31, f"latent row ids up to {top} overflow int32"


kpool_expand = KpoolExpand()
