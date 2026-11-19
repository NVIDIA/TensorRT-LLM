# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 rows widened to FP32 with full-width loads and stores (Triton).

``x.float()`` on a contiguous BF16 tensor runs torch's casting copy, which loads and stores one element at a time:
the DEP16 context server's indexer widens its ``[16384, 7168]`` input that way in every DSA layer. The FP32 bits of a
BF16 value are its 16 bits followed by 16 zero bits, so ``_upcast`` moves the bits as integers, 16 bytes per access:
the output is ``x.float()`` bit for bit (a NaN keeps its payload, which torch's GPU cast may make canonical).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _upcast_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    """dst[i] = src[i] << 16 over the bits: src int16 (the BF16 bits), dst int32 (the FP32 bits)."""
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(src_ptr + off, mask=m, other=0)
    # Sign extension fills bits 16..31, which the shift then pushes out of the word.
    tl.store(dst_ptr + off, v.to(tl.int32) << 16, mask=m)


def _upcast(x: torch.Tensor) -> torch.Tensor:
    """A new contiguous FP32 tensor with ``x``'s shape and values; ``x`` contiguous BF16."""
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    n = x.numel()
    if n == 0:
        return out
    block = 4096
    _upcast_kernel[(triton.cdiv(n, block),)](
        x.view(torch.int16), out.view(torch.int32), n, BLOCK=block, num_warps=8
    )
    return out
