# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FP8 K and V of non-absorption MLA context attention, in one pass.

The FP8 context MLA FMHA reads contiguous E4M3 K ``[T, H * (nope + rope)]`` and
V ``[T, H * v]``. :func:`mla_kv_to_fp8` writes them straight from the BF16
kv_b_proj output and k_pe, with the bytes the attention op's context copy
(``quantizeCopyInputToFp8Kernel`` with its unit scale) writes for the
concatenated BF16 K and V: every element becomes
``cvt.rn.satfinite.e4m3(float(x))``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3x2_from_bf16x2(pairs):
    """Convert int32-packed BF16 pairs to int16-packed E4M3 pairs, first element in the low byte."""
    lo = (pairs << 16).to(tl.float32, bitcast=True)
    hi = (pairs & -65536).to(tl.float32, bitcast=True)
    return tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
        "=h,f,f",
        [lo, hi],
        dtype=tl.int16,
        is_pure=True,
        pack=1,
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _mla_kv_fp8_kernel(
    kv_ptr,
    pe_ptr,
    k_ptr,
    v_ptr,
    num_tokens,
    kv_row_stride,
    pe_row_stride,
    NUM_HEADS: tl.constexpr,
    NOPE_PAIRS: tl.constexpr,
    ROPE_PAIRS: tl.constexpr,
    V_PAIRS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    mask = (rows < num_tokens)[:, None]
    head0 = tl.program_id(1) * HEADS_PER_PROGRAM
    nope = tl.arange(0, NOPE_PAIRS)[None, :]
    rope = tl.arange(0, ROPE_PAIRS)[None, :]
    v_cols = tl.arange(0, V_PAIRS)[None, :]
    kv_rows = kv_ptr + rows[:, None] * kv_row_stride
    k_rows = k_ptr + rows[:, None] * (NUM_HEADS * (NOPE_PAIRS + ROPE_PAIRS))
    v_rows = v_ptr + rows[:, None] * (NUM_HEADS * V_PAIRS)
    pe = tl.load(pe_ptr + rows[:, None] * pe_row_stride + rope, mask=mask, other=0)
    pe = _e4m3x2_from_bf16x2(pe)
    for i in tl.static_range(HEADS_PER_PROGRAM):
        head = head0 + i
        k_head = k_rows + head * (NOPE_PAIRS + ROPE_PAIRS)
        x = tl.load(kv_rows + head * NOPE_PAIRS + nope, mask=mask, other=0)
        tl.store(k_head + nope, _e4m3x2_from_bf16x2(x), mask=mask)
        tl.store(k_head + NOPE_PAIRS + rope, pe, mask=mask)
        x = tl.load(kv_rows + NUM_HEADS * NOPE_PAIRS + head * V_PAIRS + v_cols, mask=mask, other=0)
        tl.store(v_rows + head * V_PAIRS + v_cols, _e4m3x2_from_bf16x2(x), mask=mask)


def _pairs_view(x: torch.Tensor, name: str) -> torch.Tensor:
    """int32 view of a 2-D BF16 tensor with unit column stride and even row stride / offset."""
    if (
        x.dtype != torch.bfloat16
        or x.dim() != 2
        or x.stride(1) != 1
        or x.stride(0) % 2
        or x.storage_offset() % 2
        or x.shape[1] % 2
    ):
        raise ValueError(
            f"{name} must be a 2-D BF16 view with unit column stride and even width, row stride "
            f"and offset; got {x.dtype}, shape {tuple(x.shape)}, stride {x.stride()}."
        )
    return x.view(torch.int32)


def mla_kv_to_fp8(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 K and V for the FP8 context FMHA of non-absorption MLA.

    Args:
        kv: BF16 kv_b_proj output ``[T, H * (nope + v)]``: the nope columns of
            all heads, then their V columns.
        k_pe: BF16 RoPE key ``[T, rope]`` shared by all heads; unit column
            stride and any even row stride (e.g. a ``latent_cache`` view).
        num_heads: H.
        qk_nope_head_dim: nope; a power of two.
        v_head_dim: v; a power of two.

    Returns:
        Contiguous K ``[T, H * (nope + rope)]`` (each head's nope columns
        followed by k_pe) and V ``[T, H * v]``.
    """
    num_tokens, rope_head_dim = k_pe.shape
    nope_cols, v_cols = num_heads * qk_nope_head_dim, num_heads * v_head_dim
    if kv.dim() != 2 or tuple(kv.shape) != (num_tokens, nope_cols + v_cols):
        raise ValueError(f"kv must be [{num_tokens}, {nope_cols + v_cols}], got {tuple(kv.shape)}.")
    for name, dim in (
        ("qk_nope_head_dim", qk_nope_head_dim),
        ("qk_rope_head_dim", rope_head_dim),
        ("v_head_dim", v_head_dim),
    ):
        if dim < 2 or dim & (dim - 1):
            raise ValueError(f"{name} must be a power of two >= 2, got {dim}.")
    heads_per_program = 4 if num_heads % 4 == 0 else 1
    kv_pairs = _pairs_view(kv, "kv")
    pe_pairs = _pairs_view(k_pe, "k_pe")
    k = torch.empty(
        (num_tokens, num_heads * (qk_nope_head_dim + rope_head_dim)),
        dtype=torch.float8_e4m3fn,
        device=kv.device,
    )
    v = torch.empty((num_tokens, v_cols), dtype=torch.float8_e4m3fn, device=kv.device)
    if num_tokens == 0:
        return k, v
    block_tokens = 32
    grid = (triton.cdiv(num_tokens, block_tokens), num_heads // heads_per_program)
    _mla_kv_fp8_kernel[grid](
        kv_pairs,
        pe_pairs,
        k.view(torch.int16),
        v.view(torch.int16),
        num_tokens,
        kv_pairs.stride(0),
        pe_pairs.stride(0),
        NUM_HEADS=num_heads,
        NOPE_PAIRS=qk_nope_head_dim // 2,
        ROPE_PAIRS=rope_head_dim // 2,
        V_PAIRS=v_head_dim // 2,
        HEADS_PER_PROGRAM=heads_per_program,
        BLOCK_TOKENS=block_tokens,
        num_warps=4,
    )
    return k, v
