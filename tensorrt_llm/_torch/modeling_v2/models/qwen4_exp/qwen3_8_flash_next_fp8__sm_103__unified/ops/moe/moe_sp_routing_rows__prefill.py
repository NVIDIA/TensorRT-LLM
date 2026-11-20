# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sequence-parallel MoE block's routing rows in one launch each way.

Each rank packs, for its own rows, one bf16 row a token of `[scales | ids | weights]` (`nb` fp32
1x128 dequantization scales as bf16 pairs, `topk` int32 expert ids as bf16 pairs, `topk` bf16
weights, one pad column when the width is odd), gathers the rows of every rank, and unpacks them
into the runner's operands. Both directions are pure data movement: the same bytes as the slice
copies they replace (`xsf.t()`, the ids after the padding rows' no-expert fill, the weights; then
`.t().contiguous()` of the scales and `.contiguous()` of the ids and weights), in one launch
instead of four and three.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Rows a program.
_BLOCK_R = 32


@triton.jit
def _pack_kernel(
    xsf_ptr,
    ids_ptr,
    w_ptr,
    pad_ptr,
    meta32_ptr,
    meta16_ptr,
    rows,
    W32,
    no_expert,
    NB: tl.constexpr,
    NBP: tl.constexpr,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    HAS_PAD: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < rows
    r64 = r.to(tl.int64)
    b = tl.arange(0, NBP)
    bm = rm[:, None] & (b < NB)[None, :]
    sc = tl.load(xsf_ptr + b[None, :].to(tl.int64) * rows + r64[:, None], mask=bm, other=0.0)
    tl.store(meta32_ptr + r64[:, None] * W32 + b[None, :], sc.to(tl.int32, bitcast=True), mask=bm)
    k = tl.arange(0, TOPKP)
    km = rm[:, None] & (k < TOPK)[None, :]
    ids = tl.load(ids_ptr + r64[:, None] * TOPK + k[None, :], mask=km, other=0)
    if HAS_PAD:
        pad = tl.load(pad_ptr + r, mask=rm, other=0) != 0
        ids = tl.where(pad[:, None], no_expert, ids)
    tl.store(meta32_ptr + r64[:, None] * W32 + NB + k[None, :], ids, mask=km)
    w = tl.load(w_ptr + r64[:, None] * TOPK + k[None, :], mask=km, other=0.0)
    tl.store(meta16_ptr + r64[:, None] * (2 * W32) + 2 * NB + 2 * TOPK + k[None, :], w, mask=km)


@triton.jit
def _unpack_kernel(
    every32_ptr,
    every16_ptr,
    sc_ptr,
    ids_ptr,
    w_ptr,
    total,
    W32,
    NB: tl.constexpr,
    NBP: tl.constexpr,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < total
    r64 = r.to(tl.int64)
    b = tl.arange(0, NBP)
    bm = rm[:, None] & (b < NB)[None, :]
    sc = tl.load(every32_ptr + r64[:, None] * W32 + b[None, :], mask=bm, other=0)
    tl.store(sc_ptr + b[None, :].to(tl.int64) * total + r64[:, None], sc, mask=bm)
    k = tl.arange(0, TOPKP)
    km = rm[:, None] & (k < TOPK)[None, :]
    ids = tl.load(every32_ptr + r64[:, None] * W32 + NB + k[None, :], mask=km, other=0)
    tl.store(ids_ptr + r64[:, None] * TOPK + k[None, :], ids, mask=km)
    w = tl.load(
        every16_ptr + r64[:, None] * (2 * W32) + 2 * NB + 2 * TOPK + k[None, :], mask=km, other=0.0
    )
    tl.store(w_ptr + r64[:, None] * TOPK + k[None, :], w, mask=km)


#: The checkpoint's routing row: 20 scales (hidden 2560) and top-10 ids and weights, 70 bf16 wide.
_CKPT = dict(nb=20, topk=10)


class PackRoutingRows(OpWrapper):
    """`meta = pack_routing_rows(xsf, topk_ids, topk_w, pad_mask, no_expert)`: a fresh
    `[rows, width]` bf16 tensor, `width = 2 * nb + 3 * topk` rounded up to even, whose row `r`
    holds the bytes of `xsf[:, r]`, then of `topk_ids[r]`, then of `topk_w[r]`.

    - `xsf` `[nb, rows]` fp32, contiguous (`fp8_quantize_1x128`'s transposed scales);
    - `topk_ids` `[rows, topk]` int32 and `topk_w` `[rows, topk]` bf16, contiguous;
    - `pad_mask` bool `[rows]` or `[rows, 1]`, or None: its rows get `no_expert` ids.

    The pad column of an odd `2 * nb + 3 * topk` is not written. No input is written.
    """

    ARCHS = frozenset({Arch.SM_103})
    # spec: rows, nb, topk, padded (trailing rows the pad mask sets; 0: no mask).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="a tep2 rank's 16384 rows of a 32768-token step, no padding",
            spec=dict(rows=16384, padded=0, **_CKPT),
        ),
        Cell(
            why="a rank's 2048 rows of a 4096-token bucket, its last 300 padding",
            spec=dict(rows=2048, padded=300, **_CKPT),
        ),
        Cell(
            why="2050 rows, not a multiple of the 32-row program, 7 padding",
            spec=dict(rows=2050, padded=7, **_CKPT),
        ),
    )

    note = """
    Source: one Triton kernel (`_pack_kernel`), grid `ceil(rows / 32)`, JIT-compiled per
    (nb, topk, with or without a pad mask). Byte-exact the slice copies of `reference`.
    Not certified: an odd width (the pad column); `rows == 0` (returns the empty tensor without
    a launch).
    """

    def __call__(
        self,
        xsf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_w: torch.Tensor,
        pad_mask: torch.Tensor | None,
        no_expert: int,
    ) -> torch.Tensor:
        nb, rows = xsf.shape
        topk = topk_ids.shape[1]
        c2 = 2 * nb + 3 * topk
        width = c2 + (c2 & 1)
        meta = torch.empty([rows, width], dtype=torch.bfloat16, device=xsf.device)
        if rows:
            pad = pad_mask.reshape(rows) if pad_mask is not None else topk_ids
            _pack_kernel[(triton.cdiv(rows, _BLOCK_R),)](
                xsf,
                topk_ids,
                topk_w,
                pad,
                meta.view(torch.int32),
                meta,
                rows,
                width // 2,
                no_expert,
                NB=nb,
                NBP=triton.next_power_of_2(nb),
                TOPK=topk,
                TOPKP=triton.next_power_of_2(topk),
                HAS_PAD=pad_mask is not None,
                BLOCK_R=_BLOCK_R,
                num_warps=4,
            )
        return meta

    def reference(
        self,
        xsf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_w: torch.Tensor,
        pad_mask: torch.Tensor | None,
        no_expert: int,
    ) -> torch.Tensor:
        """The slice copies the op replaces; a pad column is zero here."""
        nb, rows = xsf.shape
        topk = topk_ids.shape[1]
        c0, c1, c2 = 2 * nb, 2 * nb + 2 * topk, 2 * nb + 3 * topk
        ids = topk_ids
        if pad_mask is not None:
            ids = topk_ids.masked_fill(pad_mask.reshape(rows, 1), no_expert)
        meta = torch.zeros([rows, c2 + (c2 & 1)], dtype=torch.bfloat16, device=xsf.device)
        meta[:, :c0].view(torch.float32).copy_(xsf.t())
        meta[:, c0:c1].view(torch.int32).copy_(ids)
        meta[:, c1:c2].copy_(topk_w)
        return meta

    def is_valid(
        self,
        xsf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_w: torch.Tensor,
        pad_mask: torch.Tensor | None,
        no_expert: int,
    ) -> None:
        # The kernel reads each operand at its contiguous strides.
        assert xsf.dim() == 2 and xsf.dtype == torch.float32 and xsf.is_contiguous()
        rows = xsf.shape[1]
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
        assert topk_w.dtype == torch.bfloat16 and topk_w.is_contiguous()
        assert topk_ids.dim() == 2 and topk_ids.shape[0] == rows, (tuple(topk_ids.shape), rows)
        assert topk_w.shape == topk_ids.shape, (tuple(topk_w.shape), tuple(topk_ids.shape))
        if pad_mask is not None:
            assert pad_mask.dtype == torch.bool and pad_mask.numel() == rows


pack_routing_rows = PackRoutingRows()


class UnpackRoutingRows(OpWrapper):
    """`scales, ids, weights = unpack_routing_rows(every, nb, topk)` from the gathered
    `[T, width]` bf16 routing rows (`pack_routing_rows`' layout, contiguous): `scales` `[nb, T]`
    fp32, `ids` `[T, topk]` int32 and `weights` `[T, topk]` bf16, each fresh and contiguous.
    `every` is not written.
    """

    ARCHS = frozenset({Arch.SM_103})
    # spec: rows (every rank's), nb, topk.
    CELLS: tuple[Cell, ...] = (
        Cell(why="both tep2 ranks' rows of a 32768-token step", spec=dict(rows=32768, **_CKPT)),
        Cell(why="both ranks' rows of a 4096-token bucket", spec=dict(rows=4096, **_CKPT)),
        Cell(
            why="4100 rows, not a multiple of the 32-row program",
            spec=dict(rows=4100, **_CKPT),
        ),
    )

    note = """
    Source: one Triton kernel (`_unpack_kernel`), grid `ceil(T / 32)`, JIT-compiled per
    (nb, topk). Byte-exact the slice copies of `reference`. Not certified: an odd width;
    `T == 0` (returns the empty tensors without a launch).
    """

    def __call__(self, every: torch.Tensor, nb: int, topk: int) -> tuple:
        total, width = every.shape
        scales = torch.empty([nb, total], dtype=torch.float32, device=every.device)
        ids = torch.empty([total, topk], dtype=torch.int32, device=every.device)
        weights = torch.empty([total, topk], dtype=torch.bfloat16, device=every.device)
        if total:
            _unpack_kernel[(triton.cdiv(total, _BLOCK_R),)](
                every.view(torch.int32),
                every,
                scales.view(torch.int32),
                ids,
                weights,
                total,
                width // 2,
                NB=nb,
                NBP=triton.next_power_of_2(nb),
                TOPK=topk,
                TOPKP=triton.next_power_of_2(topk),
                BLOCK_R=_BLOCK_R,
                num_warps=4,
            )
        return scales, ids, weights

    def reference(self, every: torch.Tensor, nb: int, topk: int) -> tuple:
        """The slice copies the op replaces."""
        c0, c1, c2 = 2 * nb, 2 * nb + 2 * topk, 2 * nb + 3 * topk
        scales = every[:, :c0].view(torch.float32).t().contiguous()
        ids = every[:, c0:c1].contiguous().view(torch.int32)
        weights = every[:, c1:c2].contiguous()
        return scales, ids, weights

    def is_valid(self, every: torch.Tensor, nb: int, topk: int) -> None:
        # The kernel reads `every` at its contiguous strides, as int32 and as bf16.
        c2 = 2 * nb + 3 * topk
        assert every.dim() == 2 and every.dtype == torch.bfloat16 and every.is_contiguous()
        assert every.shape[1] == c2 + (c2 & 1), (tuple(every.shape), nb, topk)


unpack_routing_rows = UnpackRoutingRows()
