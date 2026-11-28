# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero the rows of a routed-experts output that hold a pair of this rank's experts (Triton).

The CuteDSL FC2 finalize scatter-adds every (token, local expert) pair into its token's row, so
those rows have to start at zero, and no other row is written. ``zero_local_rows`` zeroes exactly
them: a row whose routing names an expert of ``[local_expert_offset, local_expert_offset +
local_num_experts)``. Every other row keeps its contents. One launch that reads the routing on the
device (no host read, CUDA-graph safe), one compiled variant per routing width and row length.
No torch op is registered; the family entry calls this launcher directly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Rows per program and row elements per store; the program stores a ``[ROWS, COLS]`` tile at a
#: time, a 16-byte vector per thread and step at a BF16 row of 3584.
_ROWS, _COLS, _WARPS = 8, 512, 4


@triton.jit(do_not_specialize=["num_rows", "lo", "hi"])
def _zero_local_rows_kernel(
    out_ptr,  # [num_rows, HIDDEN], dense
    ids_ptr,  # int32 [num_rows, TOP_K], dense
    num_rows,
    lo,  # the window's first expert
    hi,  # one past the window's last expert
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    in_range = rows < num_rows
    rows64 = rows.to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    ids = tl.load(
        ids_ptr + rows64[:, None] * TOP_K + k[None, :],
        mask=in_range[:, None] & (k[None, :] < TOP_K),
        other=-1,
    )
    local = ((ids >= lo) & (ids < hi)).to(tl.int32)
    live = in_range & (tl.max(local, axis=1) > 0)
    zero = tl.zeros((BLOCK_R, BLOCK_H), dtype=tl.float32).to(out_ptr.dtype.element_ty)
    for h0 in tl.static_range(0, HIDDEN, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        ptrs = out_ptr + rows64[:, None] * HIDDEN + cols[None, :]
        if EVEN_H:
            tl.store(ptrs, zero, mask=live[:, None])
        else:
            tl.store(ptrs, zero, mask=live[:, None] & (cols[None, :] < HIDDEN))


def zero_local_rows(
    out: torch.Tensor, ids: torch.Tensor, local_expert_offset: int, local_num_experts: int
) -> None:
    """Zero ``out[r]`` for every row ``r`` whose routing ``ids[r]`` names an expert of the window.

    ``out``: ``[T, hidden]``, dense (rows are addressed by the row length). ``ids``: int32
    ``[T, top_k]``, dense; an id outside the window (another rank's expert, or -1 for a slot that
    routes nowhere) leaves its row alone. In place.
    """
    if out.dim() != 2 or not out.is_contiguous():
        raise ValueError(f"zero_local_rows: out {tuple(out.shape)} must be a dense [T, hidden]")
    if (
        ids.dim() != 2
        or ids.dtype != torch.int32
        or not ids.is_contiguous()
        or ids.shape[0] != out.shape[0]
        or ids.device != out.device
    ):
        raise ValueError(
            f"zero_local_rows: ids {tuple(ids.shape)} {ids.dtype} must be a dense int32 "
            f"[T, top_k] beside out {tuple(out.shape)} on its device"
        )
    num_rows, hidden = out.shape
    if num_rows == 0 or hidden == 0:
        return
    top_k = ids.shape[1]
    lo = int(local_expert_offset)
    hi = lo + int(local_num_experts)
    block_h = min(_COLS, triton.next_power_of_2(hidden))
    _zero_local_rows_kernel[(triton.cdiv(num_rows, _ROWS),)](
        out,
        ids,
        num_rows,
        lo,
        hi,
        TOP_K=top_k,
        BLOCK_K=triton.next_power_of_2(max(top_k, 1)),
        HIDDEN=hidden,
        BLOCK_R=_ROWS,
        BLOCK_H=block_h,
        EVEN_H=hidden % block_h == 0,
        num_warps=_WARPS,
    )
