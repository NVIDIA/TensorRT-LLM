# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The token-embedding gather as a vectorized row copy (the ``spec_head_fusion`` constant).

``F.embedding(ids, weight)`` for 1-D ids and a contiguous table: one program per (token, 1024
hidden columns) copies the row with 16-byte accesses instead of the scalar-load index-select
kernel. A copy, so bitwise ``nn.Embedding``; the launch shape depends only on the token count,
so it is CUDA-graph capturable.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_gather_kernel(
    ids_ptr, weight_ptr, out_ptr, vocab, hidden, BLOCK_H: tl.constexpr, VEC: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    offs = tl.max_contiguous(tl.multiple_of(offs, VEC), VEC)
    mask = offs < hidden
    token = tl.load(ids_ptr + row).to(tl.int64)
    valid = (token >= 0) & (token < vocab)
    src = weight_ptr + tl.where(valid, token, 0) * hidden
    vals = tl.load(src + offs, mask=mask & valid, other=0.0)
    tl.store(out_ptr + row.to(tl.int64) * hidden + offs, vals, mask=mask)


def embedding_gather(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.embedding(ids, weight)`` for 1-D ids and a contiguous table, as a
    vectorized row copy (ids outside the table give a zero row)."""
    assert ids.dim() == 1 and weight.dim() == 2 and weight.is_contiguous()
    vocab, hidden = weight.shape
    out = torch.empty((ids.shape[0], hidden), dtype=weight.dtype, device=weight.device)
    if ids.shape[0] == 0:
        return out
    vec = 16 // weight.element_size()
    assert hidden % vec == 0
    block_h = 1024
    _embedding_gather_kernel[(ids.shape[0], triton.cdiv(hidden, block_h))](
        ids, weight, out, vocab, hidden, BLOCK_H=block_h, VEC=vec, num_warps=4
    )
    return out
