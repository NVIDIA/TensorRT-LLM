# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MTP index share: each request's last accepted TopK row of draft step 0, in one Triton launch.

With MTP index share, draft step 0's indexer computes the TopK of the verify step's next_n rows,
and the draft steps after it reuse, per request, the row of its last accepted token. The kernel
copies those rows, the rows ``torch.index_select(gen_topk, 0, index)`` would copy for
``index[r] = r * next_n + clamp(num_accepted[r] - 1, 0, next_n - 1)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _stash_last_accepted_rows_kernel(
    src_ptr,  # [num_requests * next_n, n_cols] int32 TopK rows
    src_stride,
    num_accepted_ptr,  # [num_requests] accepted tokens per request
    dst_ptr,  # [num_requests, n_cols] int32
    dst_stride,
    next_n,
    n_cols,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    chunk = tl.program_id(1)
    # The request's last accepted row of its next_n: num_accepted - 1, clamped to [0, next_n - 1].
    accepted = tl.load(num_accepted_ptr + req)
    offset = tl.minimum(tl.maximum(accepted - 1, 0), next_n - 1)
    row = req.to(tl.int64) * next_n + offset.to(tl.int64)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(src_ptr + row * src_stride + cols, mask=mask)
    tl.store(dst_ptr + req.to(tl.int64) * dst_stride + cols, values, mask=mask)


def launch_stash_last_accepted_rows(
    gen_topk: torch.Tensor, num_accepted: torch.Tensor, dst: torch.Tensor, next_n: int
) -> None:
    """dst[r] = gen_topk[r * next_n + clamp(num_accepted[r] - 1, 0, next_n - 1)] for each request.

    Args:
        gen_topk: [num_requests * next_n, n_cols] int32 TopK rows, unit column stride.
        num_accepted: [num_requests] accepted tokens per request (contiguous).
        dst: [num_requests, n_cols] int32, unit column stride.
    """
    num_requests, n_cols = dst.shape
    if num_requests == 0:
        return
    block = min(triton.next_power_of_2(n_cols), 2048)
    _stash_last_accepted_rows_kernel[(num_requests, triton.cdiv(n_cols, block))](
        gen_topk,
        gen_topk.stride(0),
        num_accepted,
        dst,
        dst.stride(0),
        next_n,
        n_cols,
        BLOCK=block,
    )
