# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode target's per-step setup of a verify step in one launch.

Tier 5 (Triton written for this target). Before the trunk runs, a verify step needs:

* `pos3`: a contiguous int32 `[3, T]` copy of the step's mRoPE positions;
* the PLE segmentation vector `cu_seqlens` (`[batch + 1]` int64, leading zero) of the
  per-request row counts;
* the speculative rollback's staging: each row's QSA slot (`pages`, `within`, int64; page
  `block_table[request, position // tokens_per_block]` and offset `position % tokens_per_block`,
  floor semantics, a negative column wrapping as torch indexing does), the requests' recurrent
  slots as int64, and every attention layer's index-K plane at those slots (`[L, T, 128]` bf16,
  read before the trunk writes any plane).

Program `(0, b)` writes block `b` of the rows' positions and slots (block 0 also the
segmentation vector and the int64 slots); program `(1 + l, b)` copies plane `l` at the same
rows, resolving each row's slot as program `(0, b)` does.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .qsa_rollback_planes import ROW

#: Rows a program covers.
_BLOCK_T = 16
#: Most requests the segmentation vector covers (`batch + 1` entries).
_BLOCK_S = 512


@triton.jit
def _slot(req_idx, logical, block_table, bt_stride, bt_cols, rows, tpb):
    req = tl.load(req_idx + rows).to(tl.int64)
    pos = tl.load(logical + rows).to(tl.int64)
    q = pos // tpb
    r = pos % tpb
    neg = (r != 0) & (pos < 0)
    q = tl.where(neg, q - 1, q)
    r = tl.where(neg, r + tpb, r)
    q = tl.where(q < 0, q + bt_cols, q)
    page = tl.load(block_table + req * bt_stride + q).to(tl.int64)
    return page, r


@triton.jit
def _setup_kernel(
    positions,
    pos_stride,
    seq_lens,
    slots,
    req_idx,
    logical,
    block_table,
    bt_stride,
    bt_cols,
    plane_ptrs,
    plane_strides,
    pos3,
    cu_seqlens,
    slots_out,
    pages_out,
    within_out,
    index_out,
    num_rows,
    batch,
    tpb,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROW: tl.constexpr,
):
    who = tl.program_id(0)
    tb = tl.program_id(1)
    t = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    tm = t < num_rows
    tt = tl.where(tm, t, 0)
    page, within = _slot(req_idx, logical, block_table, bt_stride, bt_cols, tt, tpb)
    if who == 0:
        tl.store(pages_out + t, page, mask=tm)
        tl.store(within_out + t, within, mask=tm)
        for a in tl.static_range(3):
            p = tl.load(positions + a * pos_stride + t, mask=tm)
            tl.store(pos3 + a * num_rows + t, p.to(tl.int32), mask=tm)
        if tb == 0:
            s = tl.arange(0, BLOCK_S)
            sm = s < batch
            lengths = tl.load(seq_lens + s, mask=sm, other=0).to(tl.int64)
            tl.store(cu_seqlens + 1 + s, tl.cumsum(lengths, axis=0), mask=sm)
            tl.store(cu_seqlens + s, tl.zeros([BLOCK_S], dtype=tl.int64), mask=s == 0)
            tl.store(slots_out + s, tl.load(slots + s, mask=sm).to(tl.int64), mask=sm)
    else:
        plane = who - 1
        base = tl.load(plane_ptrs + plane).to(tl.pointer_type(tl.bfloat16))
        stride = tl.load(plane_strides + plane)
        e = tl.arange(0, ROW)
        vals = tl.load(
            base + page[:, None] * stride + within[:, None] * ROW + e[None, :], mask=tm[:, None]
        )
        tl.store(
            index_out + (plane.to(tl.int64) * num_rows + t[:, None]) * ROW + e[None, :],
            vals,
            mask=tm[:, None],
        )


class VerifyStepSetup(OpWrapper):
    """`(pos3, cu_seqlens, slots.long(), pages, within, index_old)`:

    * `pos3 = position_ids.reshape(3, -1)` as a fresh contiguous int32 tensor;
    * `cu_seqlens = [0, cumsum(seq_lens)]` int64;
    * `pages = block_table[request_indices.long(), logical_positions // tokens_per_block]`
      and `within = logical_positions % tokens_per_block`, int64;
    * `index_old = qsa_rollback_gather(plane_ptrs, plane_strides, pages, within, L)`.

    Fusion boundary: the decode target's step setup before the trunk; which rows, slots and
    planes are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP3, one request, 12 planes", spec=dict(batch=1, p=4, planes=12)),
        Cell(why="MTP3, 32 requests, 12 planes", spec=dict(batch=32, p=4, planes=12)),
    )
    note = """
    Metadata consumed: none (the caller passes `seq_lens_cuda[:batch]`, the QSA row mapping
    and block table). Call site: `DecodeTarget.forward` of a captured verify step, after the
    captured preprocessing corrected positions and lengths and before the trunk.
    Preconditions: `position_ids` reshapes to `[3, T]` with unit last stride; at most
    `_BLOCK_S - 1` requests.
    """

    def __call__(
        self,
        position_ids,
        seq_lens,
        slots,
        request_indices,
        logical_positions,
        block_table,
        tokens_per_block: int,
        plane_ptrs,
        plane_strides,
        num_planes: int,
    ):
        pos2 = torch.reshape(position_ids, [3, -1])
        rows = pos2.shape[1]
        batch = seq_lens.shape[0]
        dev = pos2.device
        pos3 = torch.empty([3, rows], dtype=torch.int32, device=dev)
        cu_seqlens = torch.empty(batch + 1, dtype=torch.int64, device=dev)
        slots_out = torch.empty(batch, dtype=torch.int64, device=dev)
        pages = torch.empty(rows, dtype=torch.int64, device=dev)
        within = torch.empty(rows, dtype=torch.int64, device=dev)
        index_old = torch.empty([num_planes, rows, ROW], dtype=torch.bfloat16, device=dev)
        if rows:
            _setup_kernel[(1 + num_planes, triton.cdiv(rows, _BLOCK_T))](
                pos2,
                pos2.stride(0),
                seq_lens,
                slots,
                request_indices,
                logical_positions,
                block_table,
                block_table.stride(0),
                block_table.shape[1],
                plane_ptrs,
                plane_strides,
                pos3,
                cu_seqlens,
                slots_out,
                pages,
                within,
                index_old,
                rows,
                batch,
                tokens_per_block,
                BLOCK_T=_BLOCK_T,
                BLOCK_S=_BLOCK_S,
                ROW=ROW,
                num_warps=4,
            )
        return pos3, cu_seqlens, slots_out, pages, within, index_old

    def reference(self, *args):
        raise NotImplementedError("test_modeling_v2_verify_step_setup compares the composed ops")

    def is_valid(
        self,
        position_ids,
        seq_lens,
        slots,
        request_indices,
        logical_positions,
        block_table,
        tokens_per_block: int,
        plane_ptrs,
        plane_strides,
        num_planes: int,
    ) -> None:
        pos2 = torch.reshape(position_ids, [3, -1])
        assert pos2.stride(1) == 1
        assert 0 < seq_lens.shape[0] < _BLOCK_S and seq_lens.is_contiguous()
        assert slots.shape == seq_lens.shape
        assert request_indices.shape[0] >= pos2.shape[1]
        assert logical_positions.shape[0] >= pos2.shape[1]
        assert block_table.stride(1) == 1 and tokens_per_block > 0
        assert plane_ptrs.numel() == plane_strides.numel() == num_planes


verify_step_setup = VerifyStepSetup()
