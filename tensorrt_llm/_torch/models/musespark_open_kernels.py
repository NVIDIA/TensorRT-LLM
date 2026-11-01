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
"""Triton kernels for the Muse Spark Open decoder's fused paths.

Each kernel stands in for a short chain of elementwise and reduction ops in
`modeling_musespark_open`. The `*_supported` check next to each launcher states
the inputs the kernel takes; the caller keeps the reference composition for
everything else. The kernels round to the activation dtype wherever the
reference does, so they differ from it only in reduction order.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Warps per program, measured on GB300 at the checkpoint's widths with 1, 32
# and 256 tokens: each is the fastest, or within noise of it, at every count.
_FINALIZE_NUM_WARPS = 8
_RESIDUAL_NUM_WARPS = 16


@triton.jit
def _post_expert_norm_combine_kernel(
    out_ptr,  # [num_tokens, HIDDEN] FP32
    gemm2_ptr,  # [rows, >= HIDDEN], row-major
    index_ptr,  # [num_tokens * TOP_K]; -1 marks a slot whose expert is not local
    weights_ptr,  # [num_tokens * TOP_K]
    norm_weight_ptr,  # [HIDDEN], stored as an offset from 1
    gemm2_row_stride,
    gemm2_num_rows,
    eps,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per token: weight-first RMSNorm of each selected row, then the weighted sum.

    A slot whose index is negative or past the end of `gemm2` is never read:
    the MoE kernel leaves those rows unwritten, and a NaN there must not reach
    the sum.
    """
    token = tl.program_id(0)
    hoff = tl.arange(0, BLOCK_H)
    koff = tl.arange(0, BLOCK_K)
    in_row = hoff < HIDDEN
    in_slot = koff < TOP_K

    gain = tl.load(norm_weight_ptr + hoff, mask=in_row, other=0.0).to(tl.float32) + 1.0
    index = tl.load(index_ptr + token * TOP_K + koff, mask=in_slot, other=-1)
    local = in_slot & (index >= 0) & (index < gemm2_num_rows)
    safe = tl.where(local, index, 0).to(tl.int64)

    rows = tl.load(
        gemm2_ptr + safe[:, None] * gemm2_row_stride + hoff[None, :],
        mask=local[:, None] & in_row[None, :],
        other=0.0,
    ).to(tl.float32)
    scaled = rows * gain[None, :]
    variance = tl.sum(scaled * scaled, axis=1) / HIDDEN
    normed = scaled * tl.rsqrt(variance + eps)[:, None]
    weights = tl.load(weights_ptr + token * TOP_K + koff, mask=in_slot, other=0.0).to(tl.float32)
    acc = tl.sum(tl.where(local[:, None], normed * weights[:, None], 0.0), axis=0)
    tl.store(out_ptr + token * HIDDEN + hoff, acc, mask=in_row)


def post_expert_norm_combine_supported(gemm2_output: torch.Tensor, num_tokens: int) -> bool:
    """Whether `post_expert_norm_combine` takes these inputs: CUDA, row-major, some tokens."""
    return gemm2_output.is_cuda and num_tokens > 0 and gemm2_output.stride(-1) == 1


def post_expert_norm_combine(
    gemm2_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    routing_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
) -> torch.Tensor:
    """The per-expert post-norm and the top-k combine, in FP32, as one kernel."""
    out = torch.empty((num_tokens, hidden_size), dtype=torch.float32, device=gemm2_output.device)
    _post_expert_norm_combine_kernel[(num_tokens,)](
        out,
        gemm2_output,
        expanded_idx_to_permuted_idx.reshape(-1),
        routing_weights.reshape(-1),
        norm_weight,
        gemm2_output.stride(0),
        gemm2_output.shape[0],
        eps,
        TOP_K=top_k,
        HIDDEN=hidden_size,
        BLOCK_K=triton.next_power_of_2(top_k),
        BLOCK_H=triton.next_power_of_2(hidden_size),
        num_warps=_FINALIZE_NUM_WARPS,
    )
    return out


@triton.jit
def _gated_residual_after_rmsnorm_kernel(
    out_ptr,  # [T, HIDDEN] FP32, the updated residual stream
    residual_ptr,  # [T, HIDDEN] FP32
    branch_ptr,  # [T, HIDDEN], activation dtype
    gain_ptr,  # [HIDDEN] `1 + w`; read only with HAS_GAIN
    alpha_ptr,  # [HIDDEN] FP32
    beta_ptr,  # [HIDDEN] FP32
    eps,
    next_out_ptr,  # [T, HIDDEN], activation dtype; written only with FUSE_NEXT_NORM
    next_weight_ptr,  # [HIDDEN]; read only with FUSE_NEXT_NORM
    next_eps,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_GAIN: tl.constexpr,
    FUSE_NEXT_NORM: tl.constexpr,
):
    """`alpha * residual + beta * rmsnorm(gain * branch)` for one token.

    The gain product and the normalized branch are each rounded to the
    activation dtype, as in the reference. With `FUSE_NEXT_NORM`, the updated
    residual is also rounded to the activation dtype and passed through the
    next Gemma RMSNorm, which applies `1 + w` after normalizing.
    """
    token = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < HIDDEN
    base = token * HIDDEN + offs

    branch = tl.load(branch_ptr + base, mask=mask, other=0.0)
    scaled = branch.to(tl.float32)
    if HAS_GAIN:
        gain = tl.load(gain_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        scaled = (scaled * gain).to(branch.dtype).to(tl.float32)
    variance = tl.sum(scaled * scaled, axis=0) / HIDDEN
    normed = (scaled * tl.rsqrt(variance + eps)).to(branch.dtype).to(tl.float32)

    residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    alpha = tl.load(alpha_ptr + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    updated = alpha * residual + beta * normed
    tl.store(out_ptr + base, updated, mask=mask)
    if FUSE_NEXT_NORM:
        next_input = updated.to(branch.dtype).to(tl.float32)
        next_variance = tl.sum(next_input * next_input, axis=0) / HIDDEN
        next_weight = tl.load(next_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        next_output = next_input * tl.rsqrt(next_variance + next_eps)
        next_output = next_output * (next_weight + 1.0)
        tl.store(next_out_ptr + base, next_output, mask=mask)


def gated_residual_supported(
    residual: torch.Tensor, branch: torch.Tensor, gain: Optional[torch.Tensor]
) -> bool:
    """Whether `gated_residual_after_rmsnorm` takes these inputs."""
    return (
        residual.is_cuda
        and residual.ndim == 2
        and residual.shape == branch.shape
        and residual.shape[0] > 0
        and residual.is_contiguous()
        and branch.is_contiguous()
        # A gain of another dtype makes the reference promote `gain * branch`
        # and skip the round to the activation dtype that the kernel takes.
        and (gain is None or (gain.dtype == branch.dtype and gain.is_contiguous()))
    )


def gated_residual_after_rmsnorm(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gain: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """One residual join: the FP32 `alpha * residual + beta * rmsnorm(gain * branch)`."""
    out = torch.empty_like(residual, dtype=torch.float32)
    hidden_size = residual.shape[-1]
    _gated_residual_after_rmsnorm_kernel[(residual.shape[0],)](
        out,
        residual,
        branch,
        # Unread without a gain, but a pointer argument still needs a tensor.
        alpha if gain is None else gain,
        alpha,
        beta,
        eps,
        out,
        alpha,
        eps,
        HIDDEN=hidden_size,
        BLOCK_H=triton.next_power_of_2(hidden_size),
        HAS_GAIN=gain is not None,
        FUSE_NEXT_NORM=False,
        num_warps=_RESIDUAL_NUM_WARPS,
    )
    return out


def gated_residual_and_norm_supported(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gain: Optional[torch.Tensor],
    next_weight: torch.Tensor,
) -> bool:
    """Whether `gated_residual_after_rmsnorm_and_norm` takes these inputs."""
    return (
        residual.dtype == torch.float32
        and branch.dtype == torch.bfloat16
        and gated_residual_supported(residual, branch, gain)
        and next_weight.dtype == branch.dtype
        and next_weight.is_contiguous()
    )


def gated_residual_after_rmsnorm_and_norm(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gain: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    next_weight: torch.Tensor,
    next_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A residual join and the next Gemma RMSNorm; returns `(residual, next_input)`."""
    updated = torch.empty_like(residual)
    next_input = torch.empty_like(branch)
    hidden_size = residual.shape[-1]
    _gated_residual_after_rmsnorm_kernel[(residual.shape[0],)](
        updated,
        residual,
        branch,
        alpha if gain is None else gain,
        alpha,
        beta,
        eps,
        next_input,
        next_weight,
        next_eps,
        HIDDEN=hidden_size,
        BLOCK_H=triton.next_power_of_2(hidden_size),
        HAS_GAIN=gain is not None,
        FUSE_NEXT_NORM=True,
        num_warps=_RESIDUAL_NUM_WARPS,
    )
    return updated, next_input


@triton.jit
def _routing_weights_kernel(
    scores_ptr,
    indices_ptr,
    selected_ptr,
    weights_ptr,
    score_stride,
    eps,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK_K)
    mask = slots < TOP_K
    indices = tl.load(indices_ptr + row * TOP_K + slots, mask=mask, other=0)
    scores = tl.load(scores_ptr + row * score_stride + indices, mask=mask, other=0.0)
    denominator = tl.sum(scores, axis=0) + eps
    weights = tl.div_rn(scores, denominator)
    tl.store(selected_ptr + row * TOP_K + slots, indices.to(tl.int32), mask=mask)
    tl.store(weights_ptr + row * TOP_K + slots, weights, mask=mask)


def routing_weights_supported(
    scores: torch.Tensor, topk_idx: torch.Tensor, output_dtype: torch.dtype
) -> bool:
    """Whether `routing_weights` takes these inputs: FP32 scores and int64 top-8 ids on CUDA."""
    return (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and scores.is_contiguous()
        and scores.shape[0] > 0
        and topk_idx.is_cuda
        and topk_idx.dtype == torch.int64
        and topk_idx.ndim == 2
        and topk_idx.is_contiguous()
        and topk_idx.shape == (scores.shape[0], 8)
        and output_dtype == torch.float32
    )


def routing_weights(
    scores: torch.Tensor, topk_idx: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The int32 top-8 ids and their scores, renormalized to sum to one."""
    selected = torch.empty_like(topk_idx, dtype=torch.int32)
    weights = torch.empty_like(topk_idx, dtype=torch.float32)
    _routing_weights_kernel[(scores.shape[0],)](
        scores,
        topk_idx,
        selected,
        weights,
        scores.stride(0),
        eps,
        TOP_K=8,
        BLOCK_K=8,
        num_warps=1,
    )
    return selected, weights


@triton.jit
def _router_projection_kernel(
    hidden_states,
    weight,
    output,
    scale,
    EXPERTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    offsets = tl.arange(0, HIDDEN)
    x = tl.load(hidden_states + token * HIDDEN + offsets).to(tl.float32)
    w = tl.load(
        weight + expert[:, None] * HIDDEN + offsets[None, :],
        expert[:, None] < EXPERTS,
        other=0.0,
    )
    logits = tl.sum(w * x[None, :], axis=1) * scale
    tl.store(output + token * EXPERTS + expert, logits, expert < EXPERTS)


@triton.jit
def _router_tensorcore_kernel(
    hidden_states,
    weight,
    partials,
    TOKENS: tl.constexpr,
):
    # Sixteen tokens share each weight tile. The 8192-wide dot product is split
    # into sixteen 512-wide chunks whose FP32 partials the reduce kernel sums;
    # TF32x3 keeps the FP32 weights' precision.
    tokens = tl.program_id(0) * 16 + tl.arange(0, 16)
    experts = tl.program_id(1) * 32 + tl.arange(0, 32)
    split = tl.program_id(2)
    acc = tl.zeros((16, 32), tl.float32)
    for block in range(4):
        hidden = split * 512 + block * 128 + tl.arange(0, 128)
        x = tl.load(
            hidden_states + tokens[:, None] * 8192 + hidden[None, :],
            tokens[:, None] < TOKENS,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(weight + experts[None, :] * 8192 + hidden[:, None])
        acc = tl.dot(x, w, acc, input_precision="tf32x3")
    tl.store(
        partials + split * TOKENS * 256 + tokens[:, None] * 256 + experts[None, :],
        acc,
        tokens[:, None] < TOKENS,
    )


@triton.jit
def _router_tensorcore_reduce_kernel(partials, output, scale, SIZE: tl.constexpr):
    offsets = tl.program_id(0) * 128 + tl.arange(0, 128)
    splits = tl.arange(0, 16)
    values = tl.load(
        partials + splits[:, None] * SIZE + offsets[None, :],
        offsets[None, :] < SIZE,
        other=0.0,
    )
    tl.store(output + offsets, tl.sum(values, axis=0) * scale, offsets < SIZE)


def router_projection_supported(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether `router_projection` takes these inputs: up to 128 BF16 tokens of width 8192."""
    return (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and 0 < hidden_states.shape[0] <= 128
        and hidden_states.shape[1] == 8192
        and hidden_states.is_contiguous()
        and weight.device == hidden_states.device
        and weight.dtype == torch.float32
        and weight.shape == (256, 8192)
        and weight.is_contiguous()
    )


def _router_launch(tokens: int, overlapped: bool) -> Tuple[int, int]:
    """(experts per program, warps) of the CUDA-core router, measured on GB300.

    `overlapped` is the shape measured while the pooler runs on another stream.
    """
    if overlapped:
        if tokens in (3, 5):
            return 1, 16
        if 4 <= tokens <= 11:
            return 2, 16
        if tokens in (13, 15):
            return 2, 8
        if tokens == 14:
            return 4, 8
    return (1, 16) if tokens <= 2 else (2, 4)


def router_projection(
    hidden_states: torch.Tensor, weight: torch.Tensor, scale: float, *, overlapped: bool
) -> torch.Tensor:
    """`scale * hidden_states @ weight.T` with FP32 weights and accumulation.

    Fewer than 16 tokens use CUDA-core dot products; 16 or more share weight
    tiles on the tensor cores.
    """
    tokens = hidden_states.shape[0]
    output = torch.empty((tokens, 256), device=hidden_states.device, dtype=torch.float32)
    if tokens >= 16:
        partials = torch.empty((16, tokens, 256), device=hidden_states.device, dtype=torch.float32)
        _router_tensorcore_kernel[(triton.cdiv(tokens, 16), 8, 16)](
            hidden_states, weight, partials, TOKENS=tokens, num_warps=4
        )
        _router_tensorcore_reduce_kernel[(triton.cdiv(tokens * 256, 128),)](
            partials, output, scale, SIZE=tokens * 256, num_warps=4
        )
        return output
    rows, warps = _router_launch(tokens, overlapped)
    _router_projection_kernel[(tokens, 256 // rows)](
        hidden_states,
        weight,
        output,
        scale,
        EXPERTS=256,
        HIDDEN=8192,
        ROWS=rows,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _full_routing_kernel(logits, bias, selected, weights, eps, EXPERTS: tl.constexpr):
    """Top-8 of `sigmoid(logits) + bias`, weighted by the renormalized unbiased scores.

    The picks come out in descending order and exact ties go to the lowest
    expert id, so the order in which the finalize sums the experts is fixed.
    """
    # The `255 - expert` tie-break key needs at most 256 experts.
    tl.static_assert(EXPERTS <= 256)
    row = tl.program_id(0)
    expert = tl.arange(0, EXPERTS)
    scores = tl.sigmoid(tl.load(logits + row * EXPERTS + expert))
    ranked = scores + tl.load(bias + expert)
    # Both signed zeros must sort as one value before the id breaks the tie.
    ranked = tl.where(ranked == 0.0, 0.0, ranked)
    bits = ranked.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits >> 31) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    keys = (ordered.to(tl.uint64) << 32) | (255 - expert).to(tl.uint64)
    sorted_keys = tl.sort(keys, descending=True)
    ids = 255 - (sorted_keys & 0xFFFFFFFF).to(tl.int32)
    picked = tl.gather(scores, ids, axis=0)
    picked = tl.where(expert < 8, picked, 0.0)
    normalized = tl.div_rn(picked, tl.sum(picked, axis=0) + eps)
    tl.store(selected + row * 8 + expert, ids, expert < 8)
    tl.store(weights + row * 8 + expert, normalized, expert < 8)


def full_routing_supported(
    router_logits: torch.Tensor,
    bias: Optional[torch.Tensor],
    top_k: int,
    output_dtype: torch.dtype,
) -> bool:
    """Whether `full_routing` takes these inputs: up to 128 tokens over 256 experts, top-8."""
    return (
        router_logits.is_cuda
        and router_logits.dtype == torch.float32
        and router_logits.ndim == 2
        and router_logits.is_contiguous()
        and 0 < router_logits.shape[0] <= 128
        and router_logits.shape[1] == 256
        and top_k == 8
        and output_dtype == torch.float32
        and bias is not None
        and bias.is_cuda
        and bias.dtype == torch.float32
        and bias.shape == (256,)
        and bias.is_contiguous()
    )


def full_routing(
    router_logits: torch.Tensor, bias: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The whole routing step for 256 experts and top-8; returns int32 ids and FP32 weights."""
    tokens = router_logits.shape[0]
    selected = torch.empty((tokens, 8), device=router_logits.device, dtype=torch.int32)
    weights = torch.empty((tokens, 8), device=router_logits.device, dtype=torch.float32)
    _full_routing_kernel[(tokens,)](
        router_logits, bias, selected, weights, eps, EXPERTS=256, num_warps=4
    )
    return selected, weights


@triton.jit
def _attention_norm_gate_kernel(
    out,
    attention,
    gate,
    attention_stride,
    gate_token_stride,
    gate_head_stride,
    eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Weightless RMSNorm of one head of the attention output, times `sigmoid(gate)`.

    The norm and the sigmoid are each rounded to BF16 before the product, as
    the reference's separate ops do.
    """
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(attention + token * attention_stride + head * HEAD_DIM + offsets).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    normed = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
    g = tl.load(gate + token * gate_token_stride + head * gate_head_stride + offsets).to(tl.float32)
    sigmoid = tl.sigmoid(g).to(tl.bfloat16)
    result = normed.to(tl.float32) * sigmoid.to(tl.float32)
    tl.store(out + row * HEAD_DIM + offsets, result)


def attention_norm_gate_supported(
    attention_output: torch.Tensor, gate: torch.Tensor, head_dim: int
) -> bool:
    """Whether `attention_norm_gate` takes these inputs: up to 128 BF16 tokens, 64-wide heads."""
    return (
        attention_output.is_cuda
        and attention_output.dtype == torch.bfloat16
        and attention_output.ndim == 2
        and attention_output.stride(-1) == 1
        and 0 < attention_output.shape[0] <= 128
        and head_dim == 64
        and attention_output.shape[1] % head_dim == 0
        and gate.is_cuda
        and gate.dtype == torch.bfloat16
        and gate.ndim in (2, 3)
        and gate.stride(-1) == 1
        and gate.shape[0] == attention_output.shape[0]
        and gate.numel() == attention_output.numel()
        and (gate.ndim == 2 or gate.shape[-1] == head_dim)
    )


def attention_norm_gate(
    attention_output: torch.Tensor, gate: torch.Tensor, head_dim: int, eps: float
) -> torch.Tensor:
    """Per-head weightless RMSNorm of `[tokens, heads * head_dim]`, gated by `sigmoid(gate)`.

    `gate` is `[tokens, heads * head_dim]` or `[tokens, heads, head_dim]`, with
    any token and head strides.
    """
    output = torch.empty_like(attention_output, memory_format=torch.contiguous_format)
    heads = attention_output.shape[1] // head_dim
    _attention_norm_gate_kernel[(attention_output.shape[0] * heads,)](
        output,
        attention_output,
        gate,
        attention_output.stride(0),
        gate.stride(0),
        gate.stride(1) if gate.ndim == 3 else head_dim,
        eps,
        HEADS=heads,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return output
