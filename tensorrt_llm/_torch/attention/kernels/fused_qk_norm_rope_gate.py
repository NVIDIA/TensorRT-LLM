# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py
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
"""Fused Qwen3.5 full-attention preprocessing and output gating.

The projection produces interleaved ``[q0, gate0, q1, gate1, ..., K, V]``.
The preprocessing kernel reads that layout directly and writes packed
``[Q, K, V]`` while applying per-head Gemma RMSNorm and NeoX RoPE to Q/K.
Gate deinterleave is fused into the same pass.  Keeping the checkpoint layout
unchanged avoids coupling the optimization to weight loading, quantization,
LoRA, or attention-backend fallback behavior.
"""

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_QKV_ROPE_TILING_ENABLED = os.environ.get("TRTLLM_ENABLE_QKV_ROPE_TILING", "0") == "1"
_MIN_TILED_HEAD_ROWS = 3072  # 384 eight-token CTAs.


@triton.jit
def _fused_qkv_gemma_rmsnorm_rope_gate_kernel(
    qkv_ptr,
    qkv_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    positions_ptr,
    num_tokens,
    max_positions,
    qkv_stride_token,
    qkv_out_stride_token,
    gate_out_stride_token,
    cos_sin_stride_position,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    head_block: tl.constexpr,
    rotary_block: tl.constexpr,
    eps: tl.constexpr,
    output_fp16: tl.constexpr,
    has_pass_through: tl.constexpr,
    use_mrope: tl.constexpr,
    mrope_section1: tl.constexpr,
    mrope_section2: tl.constexpr,
    block_tokens: tl.constexpr,
):
    """Normalize and rotate independent rows in a token/head tile.

    Normalized Q/K round through the output dtype before RoPE, as in the
    single-token kernel. FP32 reduction order can differ across layouts.
    """
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    output_dtype = tl.float16 if output_fp16 else tl.bfloat16
    head = tl.program_id(1)
    head_offsets = tl.arange(0, head_block)
    rotary_offsets = tl.arange(0, rotary_block)
    # Each program owns one Q, K, or V head across block_tokens tokens.
    # Tiled tensors have shape [tokens, dimensions]; reductions stay per token.
    if block_tokens == 1:
        tokens = tl.program_id(0).to(tl.int64)
    else:
        tokens = (tl.program_id(0).to(tl.int64) * block_tokens + tl.arange(0, block_tokens))[
            :, None
        ]
        head_offsets = head_offsets[None, :]
        rotary_offsets = rotary_offsets[None, :]
    token_mask = tokens < num_tokens
    tile_mask = token_mask & (head_offsets < head_dim)
    if head < num_q_heads + num_kv_heads:
        # Input Q heads alternate with gates; output Q/K heads are packed.
        is_k = head >= num_q_heads
        if is_k:
            local_head = head - num_q_heads
            input_offset = 2 * q_size + local_head * head_dim
            output_offset = q_size + local_head * head_dim
            weight_ptr = k_weight_ptr
        else:
            local_head = head
            input_offset = local_head * 2 * head_dim
            output_offset = local_head * head_dim
            weight_ptr = q_weight_ptr
        input_rows = qkv_ptr + tokens * qkv_stride_token + input_offset
        output_rows = qkv_out_ptr + tokens * qkv_out_stride_token + output_offset
        x = tl.load(input_rows + head_offsets, mask=tile_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + head_offsets, mask=head_offsets < head_dim, other=0.0).to(
            tl.float32
        )
        # Masked lanes contribute zero; RMS uses the true, unpadded head width.
        inverse_rms = tl.rsqrt(tl.sum(x * x, axis=-1) / head_dim + eps)
        if block_tokens != 1:
            inverse_rms = inverse_rms[:, None]
        # Gemma uses (1 + weight). Round to BF16/FP16 before FP32 RoPE math.
        normalized = (x * inverse_rms * (weight + 1.0)).to(output_dtype).to(tl.float32)
        if has_pass_through:
            tl.store(
                output_rows + head_offsets,
                normalized,
                mask=tile_mask & (head_offsets >= rotary_dim),
            )

        rotary_tile_mask = token_mask & (rotary_offsets < half_rotary)
        if block_tokens == 1 or has_pass_through or (head_dim & (head_dim - 1)) != 0:
            # Preserve scalar rounding and limit live registers for partial RoPE.
            # Non-power-of-two heads also need loads that exclude padded lanes.
            x_first = tl.load(input_rows + rotary_offsets, mask=rotary_tile_mask, other=0.0).to(
                tl.float32
            )
            x_second = tl.load(
                input_rows + half_rotary + rotary_offsets, mask=rotary_tile_mask, other=0.0
            ).to(tl.float32)
            weight_first = tl.load(
                weight_ptr + rotary_offsets, mask=rotary_offsets < half_rotary, other=0.0
            ).to(tl.float32)
            weight_second = tl.load(
                weight_ptr + half_rotary + rotary_offsets,
                mask=rotary_offsets < half_rotary,
                other=0.0,
            ).to(tl.float32)
            x_first = (x_first * inverse_rms * (weight_first + 1.0)).to(output_dtype).to(tl.float32)
            x_second = (
                (x_second * inverse_rms * (weight_second + 1.0)).to(output_dtype).to(tl.float32)
            )
        else:
            # Full power-of-two RoPE can split normalized registers into NeoX halves directly,
            # avoiding another set of input and weight loads.
            x_first, x_second = tl.split(
                tl.permute(tl.reshape(normalized, (block_tokens, 2, half_rotary)), (0, 2, 1))
            )

        if use_mrope:
            # Interleave height/width frequencies within their section lengths; remaining
            # frequencies use temporal positions (section zero).
            section = tl.where(
                (rotary_offsets % 3 == 1) & (rotary_offsets < mrope_section1 * 3),
                1,
                tl.where((rotary_offsets % 3 == 2) & (rotary_offsets < mrope_section2 * 3), 2, 0),
            )
            position = tl.load(
                positions_ptr + section * num_tokens + tokens,
                mask=rotary_tile_mask,
                other=0,
            ).to(tl.int64)
        else:
            position = tl.load(positions_ptr + tokens, mask=token_mask, other=0).to(tl.int64)
        position_is_valid = (position >= 0) & (position < max_positions)
        tl.device_assert(
            position_is_valid | ~rotary_tile_mask, "position is outside the RoPE table"
        )
        # Assertions may be disabled: keep table loads safe, then propagate NaNs
        # for invalid positions rather than silently using position zero.
        safe_position = tl.where(position_is_valid, position, 0)
        cos_sin_base = cos_sin_ptr + safe_position * cos_sin_stride_position
        cos = tl.load(cos_sin_base + rotary_offsets, mask=rotary_tile_mask, other=0.0).to(
            tl.float32
        )
        sin = tl.load(
            cos_sin_base + half_rotary + rotary_offsets, mask=rotary_tile_mask, other=0.0
        ).to(tl.float32)
        cos = tl.where(position_is_valid, cos, float("nan"))
        sin = tl.where(position_is_valid, sin, float("nan"))
        tl.store(
            output_rows + rotary_offsets,
            x_first * cos - x_second * sin,
            mask=rotary_tile_mask,
        )
        tl.store(
            output_rows + half_rotary + rotary_offsets,
            x_second * cos + x_first * sin,
            mask=rotary_tile_mask,
        )

        if not is_k:
            # Gates bypass normalization and RoPE; preserve their input bits.
            gate = tl.load(input_rows + head_dim + head_offsets, mask=tile_mask, other=0.0)
            gate_rows = gate_out_ptr + tokens * gate_out_stride_token + local_head * head_dim
            tl.store(gate_rows + head_offsets, gate, mask=tile_mask)
    else:
        # V heads only move from the interleaved projection to packed QKV.
        local_head = head - num_q_heads - num_kv_heads
        value = tl.load(
            qkv_ptr
            + tokens * qkv_stride_token
            + 2 * q_size
            + kv_size
            + local_head * head_dim
            + head_offsets,
            mask=tile_mask,
            other=0.0,
        )
        tl.store(
            qkv_out_ptr
            + tokens * qkv_out_stride_token
            + q_size
            + kv_size
            + local_head * head_dim
            + head_offsets,
            value,
            mask=tile_mask,
        )


def _use_tiled_qkv_rope(
    qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    enable_tiling: bool,
) -> bool:
    return (
        # Opt in because changing tile layouts can change Q/K rounding.
        enable_tiling
        # Restrict tiling to the full- and partial-RoPE geometries benchmarked.
        and (head_dim, rotary_dim) in ((128, 128), (256, 64))
        # Count Q, K, and V head rows so small TP-local head counts still need
        # enough work: 3072 rows yield at least 384 eight-token programs.
        and qkv.shape[0] * (num_q_heads + 2 * num_kv_heads) >= _MIN_TILED_HEAD_ROWS
        # Avoid measured small-batch regressions. Larger inputs only add tiles.
        and qkv.shape[0] >= 256
    )


@torch.library.custom_op(
    "trtllm::fused_qkv_gemma_rmsnorm_rope_gate",
    mutates_args=(),
)
def _fused_qkv_gemma_rmsnorm_rope_gate(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    use_mrope: bool,
    mrope_section1: int,
    mrope_section2: int,
    enable_tiling: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert qkv.dim() == 2 and qkv.stride(-1) == 1
    assert qkv.dtype in (torch.bfloat16, torch.float16)
    assert q_weight.shape == (head_dim,) and k_weight.shape == (head_dim,)
    assert q_weight.device == qkv.device and k_weight.device == qkv.device
    assert cos_sin.dtype == torch.float32 and cos_sin.is_contiguous()
    assert cos_sin.device == qkv.device and positions.device == qkv.device
    assert rotary_dim > 0 and rotary_dim <= head_dim and rotary_dim % 2 == 0
    assert cos_sin.shape == (cos_sin.shape[0], 2, rotary_dim // 2)
    assert cos_sin.shape[0] > 0

    num_tokens = qkv.shape[0]
    if use_mrope:
        assert mrope_section1 > 0 and mrope_section2 > 0
        assert positions.numel() == 3 * num_tokens
        positions = positions.reshape(3, num_tokens)
    else:
        assert positions.numel() == num_tokens
        positions = positions.reshape(-1)
    assert positions.is_contiguous()
    assert positions.dtype in (torch.int32, torch.int64)

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    assert qkv.shape[1] == 2 * q_size + 2 * kv_size

    qkv_out = torch.empty((num_tokens, q_size + 2 * kv_size), dtype=qkv.dtype, device=qkv.device)
    gate_out = torch.empty((num_tokens, num_q_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    if num_tokens == 0:
        return qkv_out, gate_out

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rotary_block = triton.next_power_of_2(half_rotary)
    # Eight tokens per program amortize scheduling overhead across independent rows.
    # Two warps are the benchmarked configuration for eligible workloads; the scalar fallback keeps
    # one token and four warps.
    block_tokens, num_warps = (
        (8, 2)
        if _use_tiled_qkv_rope(qkv, num_q_heads, num_kv_heads, head_dim, rotary_dim, enable_tiling)
        else (1, 4)
    )
    # Eight-token tiles reduce program count; token masks handle the final tile.
    grid = (triton.cdiv(num_tokens, block_tokens), num_q_heads + 2 * num_kv_heads)
    _fused_qkv_gemma_rmsnorm_rope_gate_kernel[grid](
        qkv,
        qkv_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        num_tokens,
        cos_sin.shape[0],
        qkv.stride(0),
        qkv_out.stride(0),
        gate_out.stride(0),
        cos_sin.stride(0),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        half_rotary=half_rotary,
        head_block=head_block,
        rotary_block=rotary_block,
        eps=eps,
        output_fp16=qkv.dtype == torch.float16,
        has_pass_through=rotary_dim < head_dim,
        use_mrope=use_mrope,
        mrope_section1=mrope_section1,
        mrope_section2=mrope_section2,
        block_tokens=block_tokens,
        num_warps=num_warps,
    )
    return qkv_out, gate_out


@_fused_qkv_gemma_rmsnorm_rope_gate.register_fake
def _(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    use_mrope: bool,
    mrope_section1: int,
    mrope_section2: int,
    enable_tiling: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = qkv.shape[0]
    qkv_width = (num_q_heads + 2 * num_kv_heads) * head_dim
    return (
        qkv.new_empty((num_tokens, qkv_width)),
        qkv.new_empty((num_tokens, num_q_heads, head_dim)),
    )


def fused_qkv_gemma_rmsnorm_rope_gate(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: Optional[Tuple[int, int, int]] = None,
    *,
    enable_tiling: bool = _QKV_ROPE_TILING_ENABLED,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare packed QKV and gate with one Triton kernel.

    `TRTLLM_ENABLE_QKV_ROPE_TILING=1` opts into eight-token tiles for at least 256 tokens with
    (head dim, rotary dim) equal to (128, 128) or (256, 64) and at least 3072 token/head pairs
    across Q/K/V.
    Head counts can vary with tensor parallelism. The flag sets the default at import; callers can
    override it with `enable_tiling`.
    Other calls use the single-token kernel.

    Tiled FP32 reductions and compiler contraction can change Q/K rounding. With tiling enabled,
    changing a request's batch size across a dispatch boundary can change its Q/K values, cached K,
    and generated tokens.
    Leave the flag off when batch-invariant results are required.

    Args:
        qkv: ``[num_tokens, 2 * q_size + 2 * kv_size]`` BF16/FP16 projection
            output in per-head interleaved Q/G layout.
        q_weight: ``[head_dim]`` raw Gemma RMSNorm Q weights.
        k_weight: ``[head_dim]`` raw Gemma RMSNorm K weights.
        cos_sin: ``[max_positions, 2, rotary_dim // 2]`` FP32 NeoX table.
        positions: Flattenable ``[num_tokens]`` plain-RoPE positions, or
            ``[3, ..., num_tokens]`` interleaved-MRoPE positions.
        eps: RMSNorm epsilon.
        num_q_heads: Local query-head count.
        num_kv_heads: Local key/value-head count.
        head_dim: Per-head Q/K/V dimension.
        rotary_dim: Prefix dimension receiving NeoX RoPE.
        mrope_section: Temporal/height/width rotary-half dimensions. ``None``
            selects plain RoPE; a tuple selects Qwen-style interleaved MRoPE.
        enable_tiling: Allow eight-token tiling for eligible workloads. Defaults
            to the import-time environment flag; ``True`` does not bypass eligibility.

    Returns:
        Packed QKV ``[num_tokens, q_size + 2 * kv_size]`` and gate
        ``[num_tokens, num_q_heads, head_dim]``.
    """
    use_mrope = mrope_section is not None
    if use_mrope:
        assert len(mrope_section) == 3
        assert sum(mrope_section) == rotary_dim // 2
        mrope_section1 = mrope_section[1]
        mrope_section2 = mrope_section[2]
    else:
        mrope_section1 = 0
        mrope_section2 = 0
    return _fused_qkv_gemma_rmsnorm_rope_gate(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        use_mrope,
        mrope_section1,
        mrope_section2,
        enable_tiling,
    )


@triton.jit
def _fused_sigmoid_mul_kernel(
    output_ptr,
    attention_ptr,
    gate_ptr,
    output_stride_token,
    attention_stride_token,
    gate_stride_token,
    gate_stride_head,
    hidden_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < hidden_size
    head = offsets // head_dim
    dim = offsets - head * head_dim

    attention = tl.load(
        attention_ptr + token * attention_stride_token + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + token * gate_stride_token + head * gate_stride_head + dim,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + token * output_stride_token + offsets,
        attention * tl.sigmoid(gate),
        mask=mask,
    )


@torch.library.custom_op(
    "trtllm::fused_sigmoid_mul_inplace",
    mutates_args=("attention_output",),
)
def fused_sigmoid_mul_inplace(
    attention_output: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    """Apply ``attention_output *= sigmoid(gate)`` with a Triton kernel."""
    assert attention_output.dim() == 2 and attention_output.stride(-1) == 1
    num_tokens, hidden_size = attention_output.shape
    if gate.dim() == 3:
        assert gate.shape[0] == num_tokens
        assert gate.shape[1] * gate.shape[2] == hidden_size
        assert gate.stride(-1) == 1
        head_dim = gate.shape[2]
        gate_stride_token = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        assert gate.dim() == 2 and gate.shape == attention_output.shape
        assert gate.stride(-1) == 1
        head_dim = hidden_size
        gate_stride_token = gate.stride(0)
        gate_stride_head = hidden_size

    if num_tokens == 0:
        return

    max_block_size = 1024 if num_tokens < 1024 else 2048
    block_size = min(triton.next_power_of_2(hidden_size), max_block_size)
    grid = (num_tokens, triton.cdiv(hidden_size, block_size))
    _fused_sigmoid_mul_kernel[grid](
        attention_output,
        attention_output,
        gate,
        attention_output.stride(0),
        attention_output.stride(0),
        gate_stride_token,
        gate_stride_head,
        hidden_size=hidden_size,
        head_dim=head_dim,
        block_size=block_size,
        num_warps=4,
    )
