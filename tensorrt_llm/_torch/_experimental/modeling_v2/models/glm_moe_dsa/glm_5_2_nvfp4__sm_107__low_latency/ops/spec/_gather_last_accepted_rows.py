# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernel behind ``gather_last_accepted_rows``: the step-0 row selection of the MTP draft loop.

Request ``r`` selects row ``g[r] = batch_indices[r] * rows_per_request + num_accepted[r] - 1 + row_offset``, the
row of its last accepted token. One launch writes the int64 ``g`` and copies row ``g[r]`` of one or two row
tensors into row ``r`` of fresh outputs.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

#: Columns one program copies.
_BLOCK = 1024


@triton.jit
def _gather_last_accepted_rows_kernel(
    src_ptr,
    out_ptr,
    src2_ptr,
    out2_ptr,
    batch_indices_ptr,
    num_accepted_ptr,
    gather_ids_ptr,
    stride_src,
    stride_out,
    stride_src2,
    stride_out2,
    rows_per_request,
    row_offset,
    n_cols,
    HAS_SECOND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    chunk = tl.program_id(1)
    row = (
        tl.load(batch_indices_ptr + req).to(tl.int64) * rows_per_request
        + tl.load(num_accepted_ptr + req).to(tl.int64)
        - 1
        + row_offset
    )
    if chunk == 0:
        tl.store(gather_ids_ptr + req, row)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(src_ptr + row * stride_src + cols, mask=mask)
    tl.store(out_ptr + req.to(tl.int64) * stride_out + cols, values, mask=mask)
    if HAS_SECOND:
        values2 = tl.load(src2_ptr + row * stride_src2 + cols, mask=mask)
        tl.store(out2_ptr + req.to(tl.int64) * stride_out2 + cols, values2, mask=mask)


def gather_last_accepted_rows(
    rows: torch.Tensor,
    batch_indices: torch.Tensor,
    num_accepted: torch.Tensor,
    rows_per_request: int,
    row_offset: int,
    second: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(g, rows[g])``, plus ``second[g]`` as a third element when ``second`` is given, in one launch.

    ``rows`` and ``second`` are 2D with unit column stride; ``batch_indices`` and ``num_accepted`` are contiguous
    integer vectors of one length.
    """
    num_requests = batch_indices.shape[0]
    n_cols = rows.shape[1]
    gather_ids = torch.empty((num_requests,), dtype=torch.int64, device=rows.device)
    out = rows.new_empty((num_requests, n_cols))
    out2 = second.new_empty((num_requests, second.shape[1])) if second is not None else out
    if num_requests > 0:
        _gather_last_accepted_rows_kernel[(num_requests, triton.cdiv(n_cols, _BLOCK))](
            rows,
            out,
            second if second is not None else rows,
            out2,
            batch_indices,
            num_accepted,
            gather_ids,
            rows.stride(0),
            out.stride(0),
            second.stride(0) if second is not None else 0,
            out2.stride(0),
            rows_per_request,
            row_offset,
            n_cols,
            HAS_SECOND=second is not None,
            BLOCK=_BLOCK,
        )
    if second is not None:
        return gather_ids, out, out2
    return gather_ids, out
