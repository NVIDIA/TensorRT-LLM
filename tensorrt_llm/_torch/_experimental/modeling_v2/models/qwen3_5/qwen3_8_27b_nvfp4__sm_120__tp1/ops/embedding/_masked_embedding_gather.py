# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token embedding as one Triton gather with the vocab-range mask fused in.

Replaces the Embedding module's masked-input -> ``F.embedding`` -> ``masked_fill_`` chain
for a table held whole: valid rows are copied unchanged, ids outside ``[lo, hi)`` give
+0.0 rows.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_embedding_gather_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    lo,
    hi,
    weight_stride_row,
    out_stride_row,
    embedding_dim,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    idx = tl.load(ids_ptr + token).to(tl.int64)
    valid = (idx >= lo) & (idx < hi)
    row = tl.where(valid, idx - lo, 0)
    offsets = block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    values = tl.load(weight_ptr + row * weight_stride_row + offsets, mask=mask & valid, other=0.0)
    tl.store(out_ptr + token * out_stride_row + offsets, values, mask=mask)


def masked_embedding_gather(
    input_: torch.Tensor, weight: torch.Tensor, vocab_start_index: int, vocab_end_index: int
) -> torch.Tensor:
    """out[t] = W[id - lo] if lo <= id < hi else 0, in one kernel.

    Bitwise equal to get_masked_input_and_mask + F.embedding + masked_fill_: valid rows
    are copied unchanged and invalid ids give +0.0 rows.
    """
    ids = input_.reshape(-1)
    num_tokens = ids.numel()
    embedding_dim = weight.shape[1]
    output = torch.empty((num_tokens, embedding_dim), dtype=weight.dtype, device=weight.device)
    if num_tokens > 0:
        block_d = min(1024, triton.next_power_of_2(embedding_dim))
        grid = (num_tokens, triton.cdiv(embedding_dim, block_d))
        _masked_embedding_gather_kernel[grid](
            ids,
            weight,
            output,
            vocab_start_index,
            vocab_end_index,
            weight.stride(0),
            output.stride(0),
            embedding_dim,
            BLOCK_D=block_d,
            num_warps=4,
        )
    return output.view(*input_.shape, embedding_dim)
