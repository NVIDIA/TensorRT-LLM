# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Debug check of the DSA core's index operands (``features.K31Features.dsa_core_check``).

SparseDcpDecode reads KV row ``indices[b, i]`` for every ``i < valid[b]`` with no bounds test,
so its contract is: ``0 <= valid[b] <= 2048`` and the first ``valid[b]`` indices of row ``b``
are pool rows in ``[0, pool_rows)``. One program per row checks it on the device (capturable,
no host sync); a violating row prints ``K31_DSA_CORE_CHECK fail`` with its rank, layer, row,
count, leading non-negative entries, first bad position, that entry's pool row and
request-local key (``local``, else the pool row again) and the pool size, then fails a device
assert (the step stops with a CUDA error instead of reading outside the pool).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(debug=True)
def _core_rows_check(indices, local, valid, pool_rows, rank, layer, TOPK: tl.constexpr):
    b = tl.program_id(0)
    n = tl.load(valid + b)
    offs = tl.arange(0, TOPK)
    idx = tl.load(indices + b * TOPK + offs)
    lead = tl.min(tl.where(idx < 0, offs, TOPK), axis=0)
    bad = (offs < n) & ((idx < 0) | (idx.to(tl.int64) >= pool_rows))
    first = tl.min(tl.where(bad, offs, TOPK), axis=0)
    ok = (n >= 0) & (n <= TOPK) & (first == TOPK)
    if (first < TOPK) | (n < 0) | (n > TOPK):
        at = tl.minimum(first, TOPK - 1)
        tl.device_print(
            "K31_DSA_CORE_CHECK fail rank layer row valid lead first index local pool_rows",
            rank,
            layer,
            b,
            n,
            lead,
            first,
            tl.load(indices + b * TOPK + at),
            tl.load(local + b * TOPK + at),
            pool_rows,
        )
    tl.device_assert(ok, "K31_DSA_CORE_CHECK: a kept count reaches past the row's pool rows")


def check_core_rows(
    indices: torch.Tensor,
    valid: torch.Tensor,
    pool_rows: int,
    rank: int,
    layer: int,
    local: torch.Tensor | None = None,
) -> None:
    """Launch the check on the current stream (``local``: the request-local Top-K the indices
    were remapped from, printed next to a bad entry; defaults to ``indices``)."""
    rows, topk = indices.shape
    if not indices.is_contiguous():
        raise ValueError("The DSA core check reads contiguous [rows, top-k] indices")
    if local is None or local.shape != indices.shape or not local.is_contiguous():
        local = indices
    _core_rows_check[(rows,)](indices, local, valid, pool_rows, rank, layer, TOPK=topk, num_warps=4)
