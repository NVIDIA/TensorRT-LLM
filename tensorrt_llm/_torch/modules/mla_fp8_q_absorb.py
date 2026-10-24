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
"""FP8 query of the absorbed MLA decode for a NoPE model, in one kernel.

The FP8-KV MLA decode reads its query as FP8 ``[tokens, heads, kv_lora_rank +
rope_dim]``: the absorbed ``q_nope @ W_UK`` followed by the RoPE'd ``q_pe``. For
a NoPE model (identity rotation, e.g. Kimi K3) the RoPE part is a copy of
``q_pe``, so both halves come straight out of one batched GEMM kernel whose
epilogue converts to FP8, with no BF16 ``fused_q`` for the MLA RoPE kernel to
re-read and convert.

FP8 conversion is round-to-nearest-even with saturation to +-448, as the MLA
RoPE kernel's; the q_nope part is rounded once from the FP32 accumulator
instead of through BF16.

With ``TRTLLM_K3_MLA_FP8_Q_ABSORB_CUTE=1`` on SM100/SM103/SM107 the same bytes come from
the persistent tcgen05 kernel in
:mod:`tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_mla_q_absorb` (faster at decode
token counts); unsupported operands keep the Triton kernel.
"""

import os

import torch
import triton
import triton.language as tl

_USE_CUTE = os.environ.get("TRTLLM_K3_MLA_FP8_Q_ABSORB_CUTE", "0") == "1"


@triton.jit
def _mla_fp8_q_absorb_nope_kernel(
    q_ptr,
    w_ptr,
    out_ptr,
    num_tokens,
    stride_q_tok,
    stride_q_head,
    stride_w_head,
    stride_w_n,
    stride_o_tok,
    stride_o_head,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    LORA_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    FP8_MAX: tl.constexpr = 448.0,
):
    # One program per (head, N block, token split): the W_UK tile stays in
    # registers/shared memory while the program walks its token blocks.
    head = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, NOPE_DIM)
    b = tl.load(
        w_ptr + head.to(tl.int64) * stride_w_head + offs_n[:, None] * stride_w_n + offs_k[None, :]
    )
    bt = tl.trans(b)
    offs_r = tl.arange(0, ROPE_DIM)
    num_m = tl.cdiv(num_tokens, BLOCK_M)
    for m in tl.range(pid_s, num_m, M_SPLITS, num_stages=NUM_STAGES):
        offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < num_tokens
        q_rows = q_ptr + offs_m.to(tl.int64)[:, None] * stride_q_tok + head * stride_q_head
        a = tl.load(q_rows + offs_k[None, :], mask=m_mask[:, None], other=0.0)
        acc = tl.dot(a, bt)
        o_rows = out_ptr + offs_m.to(tl.int64)[:, None] * stride_o_tok + head * stride_o_head
        acc = tl.minimum(tl.maximum(acc, -FP8_MAX), FP8_MAX)
        tl.store(o_rows + offs_n[None, :], acc.to(tl.float8e4nv), mask=m_mask[:, None])
        if pid_n == 0:
            q_pe = tl.load(q_rows + NOPE_DIM + offs_r[None, :], mask=m_mask[:, None], other=0.0)
            q_pe = tl.minimum(tl.maximum(q_pe.to(tl.float32), -FP8_MAX), FP8_MAX)
            tl.store(
                o_rows + LORA_DIM + offs_r[None, :],
                q_pe.to(tl.float8e4nv),
                mask=m_mask[:, None],
            )


def _launch_config(num_tokens: int) -> dict:
    """Tile and grid shape by token count (measured on GB300)."""
    if num_tokens <= 64:
        return dict(BLOCK_M=32, BLOCK_N=128, M_SPLITS=1, NUM_STAGES=2, num_warps=4)
    if num_tokens <= 448:
        return dict(BLOCK_M=64, BLOCK_N=256, M_SPLITS=1, NUM_STAGES=2, num_warps=4)
    m_splits = 2 if num_tokens <= 1024 else 4
    return dict(BLOCK_M=64, BLOCK_N=256, M_SPLITS=m_splits, NUM_STAGES=2, num_warps=4)


def mla_fp8_q_absorb_nope(
    q: torch.Tensor,
    w_kc: torch.Tensor,
    out: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
) -> None:
    """Write the FP8 absorbed query of a NoPE MLA decode step into ``out``.

    ``q``: ``[tokens, heads, nope_dim + rope_dim]`` BF16 (last dim contiguous),
    ``w_kc``: ``[heads, kv_lora_rank, nope_dim]`` BF16 (absorbed W_UK, contiguous),
    ``out``: ``[tokens, heads, kv_lora_rank + rope_dim]`` FP8 E4M3 (or its uint8
    view), last dim contiguous. ``out[t, h, :kv_lora_rank] = fp8(q_nope[t, h] @
    w_kc[h].T)`` and ``out[t, h, kv_lora_rank:] = fp8(q_pe[t, h])``.
    """
    num_tokens, num_heads, head_dim = q.shape
    lora_dim = w_kc.shape[1]
    assert head_dim == nope_dim + rope_dim and w_kc.shape == (num_heads, lora_dim, nope_dim)
    assert out.shape == (num_tokens, num_heads, lora_dim + rope_dim)
    assert q.stride(2) == 1 and w_kc.stride(2) == 1 and out.stride(2) == 1
    if out.dtype == torch.uint8:
        out = out.view(torch.float8_e4m3fn)
    if num_tokens == 0:
        return
    if _USE_CUTE:
        from ..cute_dsl_kernels.blackwell import k3_mla_q_absorb

        if k3_mla_q_absorb.can_run(q, w_kc, out, nope_dim, rope_dim):
            k3_mla_q_absorb.k3_mla_q_absorb(q, w_kc, out, nope_dim, rope_dim)
            return
    cfg = _launch_config(num_tokens)
    assert lora_dim % cfg["BLOCK_N"] == 0
    grid = (num_heads, lora_dim // cfg["BLOCK_N"], cfg["M_SPLITS"])
    _mla_fp8_q_absorb_nope_kernel[grid](
        q,
        w_kc,
        out,
        num_tokens,
        q.stride(0),
        q.stride(1),
        w_kc.stride(0),
        w_kc.stride(1),
        out.stride(0),
        out.stride(1),
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        LORA_DIM=lora_dim,
        **cfg,
    )
