# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fused QKV prep for Gemma4: per-head RMSNorm + RoPE + FP8 quant in one kernel.

Replaces the unfused chain executed per layer per step on the FlashInfer
FP8-KV path:

    split_qkv (strided views) -> reshape copies (3x direct_copy)
    -> q_norm / k_norm / v_norm (3x rmsnorm)
    -> flashinfer rope in-place on q, k (1 kernel)
    -> q/k/v .to(float8_e4m3fn) in the backend (3x float8_copy)

with a single Triton kernel that reads the packed QKV GEMM output directly
(per-head strided access, no contiguous intermediate), normalizes each head,
applies neox-style RoPE to Q/K heads from the module's fp32 cos/sin table,
and writes packed FP8 (or BF16) Q, K, V.

Numerics deliberately replicate the unfused path: fp32 accumulation with a
round to bf16 after the norm and again after RoPE, so the final FP8 values
match the reference chain (see tests/unittest/_torch/modules/
test_gemma4_fused_qkv_prep.py).

The kernel processes BLOCK_N tokens per program (2D [BLOCK_N, HALF] tiles)
so each thread issues wide vectorized loads/stores instead of one scalar
element; tile size and warp count are fixed host-side as a pure function of
head_dim (no runtime autotuning, so launches stay deterministic under
CUDA-graph capture). Tuned on B200: 363 -> ~75 us/call at ~6.5k tokens for
the 64-head hd=256 shape (~4.3 TB/s effective), 2.8x on the graph-replayed
decode shape.

The unfused path is kept only for configurations this kernel does not
support (KV-shared layers, non-FP8 KV cache, custom-mask multimodal
prefill, torch.compile).
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma4_qkv_norm_rope_quant_kernel(
    qkv_ptr,  # [N, W] bf16, row stride SW; heads packed [q | k | v], head h at col h*HD
    q_out_ptr,  # [N, NQ*HD] fp8/bf16 contiguous
    k_out_ptr,  # [N, NK*HD]
    v_out_ptr,  # [N, NK*HD]
    pos_ptr,  # [N] int positions
    cos_sin_ptr,  # [max_pos, 2, HALF] fp32: [:, 0, :]=cos, [:, 1, :]=sin
    qw_ptr,  # [HD] bf16 q_norm weight
    kw_ptr,  # [HD] bf16 k_norm weight
    SW,  # qkv row stride (elements)
    NQ,  # num q heads
    NK,  # num kv heads
    EPS,
    N,  # num tokens (rows)
    HD: tl.constexpr,  # head dim
    HALF: tl.constexpr,  # HD // 2 (power of two)
    OUT_FP8: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tokens per program (power of two)
):
    h = tl.program_id(0)
    rows = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    rmask = rows < N
    offs = tl.arange(0, HALF)

    # Head h occupies columns [h*HD, (h+1)*HD) of the packed qkv rows.
    base = qkv_ptr + rows[:, None] * SW + h * HD + offs[None, :]
    x1 = tl.load(base, mask=rmask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(base + HALF, mask=rmask[:, None], other=0.0).to(tl.float32)

    ssq = tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)
    rms = tl.math.rsqrt(ssq / HD + EPS)  # [BLOCK_N]

    if h < NQ + NK:
        # Q/K head: norm (with weight) -> bf16 round -> rope -> bf16 round.
        wp = qw_ptr if h < NQ else kw_ptr
        w1 = tl.load(wp + offs).to(tl.float32)
        w2 = tl.load(wp + HALF + offs).to(tl.float32)
        y1 = (x1 * rms[:, None] * w1[None, :]).to(tl.bfloat16).to(tl.float32)
        y2 = (x2 * rms[:, None] * w2[None, :]).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(pos_ptr + rows, mask=rmask, other=0).to(tl.int64)
        cs = cos_sin_ptr + pos[:, None] * (2 * HALF) + offs[None, :]
        cos = tl.load(cs, mask=rmask[:, None], other=0.0)
        sin = tl.load(cs + HALF, mask=rmask[:, None], other=0.0)
        o1 = (y1 * cos - y2 * sin).to(tl.bfloat16)
        o2 = (y2 * cos + y1 * sin).to(tl.bfloat16)
        if h < NQ:
            out = q_out_ptr + rows[:, None] * (NQ * HD) + h * HD + offs[None, :]
        else:
            out = k_out_ptr + rows[:, None] * (NK * HD) + (h - NQ) * HD + offs[None, :]
        if OUT_FP8:
            tl.store(out, o1.to(tl.float8e4nv), mask=rmask[:, None])
            tl.store(out + HALF, o2.to(tl.float8e4nv), mask=rmask[:, None])
        else:
            tl.store(out, o1, mask=rmask[:, None])
            tl.store(out + HALF, o2, mask=rmask[:, None])
    else:
        # V head: weightless norm, no rope (v_norm is applied to the raw v
        # slice; reads happen before any write, matching HF's
        # v_norm-before-k_norm ordering for K=V layers).
        o1 = (x1 * rms[:, None]).to(tl.bfloat16)
        o2 = (x2 * rms[:, None]).to(tl.bfloat16)
        out = v_out_ptr + rows[:, None] * (NK * HD) + (h - NQ - NK) * HD + offs[None, :]
        if OUT_FP8:
            tl.store(out, o1.to(tl.float8e4nv), mask=rmask[:, None])
            tl.store(out + HALF, o2.to(tl.float8e4nv), mask=rmask[:, None])
        else:
            tl.store(out, o1, mask=rmask[:, None])
            tl.store(out + HALF, o2, mask=rmask[:, None])


def gemma4_fused_qkv_norm_rope_quant(
    qkv: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    out_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused per-head norm + rope + quant over a packed QKV tensor.

    Args:
        qkv: [N, num_q_heads*head_dim + 2*num_kv_heads*head_dim] bf16, packed
            [q | k | v]; may have a row stride larger than its width.
        position_ids: flattenable to [N]; integer rope positions per token.
        cos_sin: [max_positions, 2, head_dim // 2] fp32 contiguous table
            (RotaryEmbedding.rotary_cos_sin layout).
        q_weight/k_weight: [head_dim] q_norm/k_norm weights.
        eps: shared RMSNorm epsilon.
        out_fp8: emit float8_e4m3fn outputs (KV-cache dtype) when True,
            bf16 otherwise.

    Returns:
        (q, k, v): contiguous [N, q_size], [N, kv_size], [N, kv_size] in the
        requested output dtype; q/k are roped, v is norm-only.
    """
    assert qkv.dim() == 2 and qkv.stride(-1) == 1
    assert qkv.dtype == torch.bfloat16
    assert cos_sin.is_contiguous() and cos_sin.dtype == torch.float32
    half = head_dim // 2
    assert cos_sin.dim() == 3 and cos_sin.shape[1] == 2 and cos_sin.shape[2] == half
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    assert qkv.shape[-1] == q_size + 2 * kv_size

    n = qkv.shape[0]
    out_dtype = torch.float8_e4m3fn if out_fp8 else torch.bfloat16
    q_out = torch.empty((n, q_size), dtype=out_dtype, device=qkv.device)
    k_out = torch.empty((n, kv_size), dtype=out_dtype, device=qkv.device)
    v_out = torch.empty((n, kv_size), dtype=out_dtype, device=qkv.device)
    if n == 0:
        return q_out, k_out, v_out

    # ~2k-element tiles keep every thread on wide vectorized accesses without
    # spilling: 16 tokens/program for hd=256, 8 for hd=512 (B200-tuned, see
    # module docstring). head_dim is a power of two on every gated shape, so
    # block_n is too (required by tl.arange).
    block_n = max(1, min(32, 4096 // head_dim))
    grid = (num_q_heads + 2 * num_kv_heads, triton.cdiv(n, block_n))
    _gemma4_qkv_norm_rope_quant_kernel[grid](
        qkv,
        q_out,
        k_out,
        v_out,
        position_ids.view(-1),
        cos_sin,
        q_weight,
        k_weight,
        qkv.stride(0),
        num_q_heads,
        num_kv_heads,
        eps,
        n,
        HD=head_dim,
        HALF=half,
        OUT_FP8=out_fp8,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return q_out, k_out, v_out
