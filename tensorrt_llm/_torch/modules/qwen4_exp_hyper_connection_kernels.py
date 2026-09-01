# Adapted from vLLM's Qwen3.8-Flash-Next Hyper-Connection kernels.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for the Qwen4-Exp gated Hyper-Connection.

These kernels fuse only numerically adjacent pointwise and normalization
operations. The two low-rank projections remain regular GEMMs so TRT-LLM can
select the best cuBLAS or low-M CuTe DSL implementation for each token count.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version

#: Widest element block ``hc_combine_norm`` will ask for in one tile. 4096
#: lanes is what the previous fixed 512-wide tiling already held in registers
#: for a 2560-wide hidden size; wider rows fall back to tiling rather than
#: growing per-thread state.
_COMBINE_NORM_MAX_BLOCK = 4096


def _pdl_enabled(rows: int) -> bool:
    # PDL hides the serial launch dependency at decode sizes, but its
    # synchronization overhead becomes measurable once M supplies enough CTA
    # parallelism on its own. Keep the shape-aware boundary used by the HC
    # low-latency path rather than regressing prefill or large IFB batches.
    return rows <= 16 and os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1" and get_sm_version() >= 90


@triton.jit
def _load_block_output(
    block_ptr,
    shared_expert_ptr,
    shared_expert_gate_ptr,
    row,
    stride_block,
    stride_shared_expert,
    offsets_inner,
    mask,
    gate_shared_expert: tl.constexpr,
):
    """Load the block output a combine injects into the residual streams.

    With ``gate_shared_expert`` the block pointer holds only the routed experts'
    output and this adds the MoE block's shared-expert branch,
    ``sigmoid(gate) * shared`` with one gate scalar per row. A combine reads
    the routed output anyway, so folding the branch in costs one extra vector
    load and removes the pointwise launch that would otherwise produce the sum.
    """
    block = tl.load(
        block_ptr + row * stride_block + offsets_inner,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if gate_shared_expert:
        shared_expert = tl.load(
            shared_expert_ptr + row * stride_shared_expert + offsets_inner,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sigmoid(tl.load(shared_expert_gate_ptr + row).to(tl.float32))
        # Round back to the routed tensor's dtype: that is where the separate
        # shared-expert kernel materializes the sum before a combine reads it,
        # so both paths inject exactly the same values.
        block = (block + gate * shared_expert).to(block_ptr.dtype.element_ty).to(tl.float32)
    return block


def _check_shared_expert_branch(
    block_output: torch.Tensor,
    shared_expert_output: Optional[torch.Tensor],
    shared_expert_gate_logits: Optional[torch.Tensor],
) -> bool:
    """Validate an optional shared-expert branch and report whether it is set."""
    if shared_expert_output is None:
        assert shared_expert_gate_logits is None, (
            "shared_expert_gate_logits needs shared_expert_output"
        )
        return False
    assert shared_expert_gate_logits is not None, (
        "shared_expert_output needs shared_expert_gate_logits"
    )
    assert shared_expert_output.shape == block_output.shape
    assert shared_expert_output.dtype == block_output.dtype
    assert shared_expert_output.stride(1) == 1
    assert shared_expert_gate_logits.numel() == block_output.shape[0]
    assert shared_expert_gate_logits.is_contiguous()
    return True


@triton.jit
def _hc_silu_kernel(
    input_ptr,
    output_ptr,
    stride_input,
    stride_output,
    width: tl.constexpr,
    hc_count: tl.constexpr,
    block_size: tl.constexpr,
    launch_with_pdl: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < width
    if launch_with_pdl:
        tl.extra.cuda.gdc_wait()
    value = tl.load(
        input_ptr + row * stride_input + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    value /= hc_count
    if launch_with_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        output_ptr + row * stride_output + offsets,
        value * tl.sigmoid(value),
        mask=mask,
    )


@torch.library.custom_op("trtllm::qwen4_exp_hc_silu", mutates_args=())
def hc_silu(input: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Fuse the HC scaling and SiLU applied after the down projection."""
    rows, width = input.shape
    assert input.is_cuda and input.stride(1) == 1
    output = torch.empty_like(input)
    block_size = triton.next_power_of_2(width)
    launch_with_pdl = _pdl_enabled(rows)
    with torch.cuda.device(input.device.index):
        _hc_silu_kernel[(rows,)](
            input,
            output,
            input.stride(0),
            output.stride(0),
            width=width,
            hc_count=hc_count,
            block_size=block_size,
            launch_with_pdl=launch_with_pdl,
            launch_pdl=launch_with_pdl,
        )
    return output


@hc_silu.register_fake
def _(input: torch.Tensor, hc_count: int) -> torch.Tensor:
    del hc_count
    return torch.empty_like(input)


@triton.jit
def _hc_gate_mix_kernel(
    input_ptr,
    gate_ptr,
    output_ptr,
    stride_input,
    stride_gate,
    stride_output,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_size: tl.constexpr,
    launch_with_pdl: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets_inner = tile * block_size + tl.arange(0, block_size)
    mask = offsets_inner < hidden_size

    if launch_with_pdl:
        tl.extra.cuda.gdc_wait()
    accumulator = tl.zeros([block_size], dtype=tl.float32)
    for stream in tl.static_range(hc_count):
        offsets = stream * hidden_size + offsets_inner
        gate = tl.load(
            gate_ptr + row * stride_gate + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            input_ptr + row * stride_input + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sigmoid(gate) * value
    accumulator /= hc_count
    if launch_with_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        output_ptr + row * stride_output + offsets_inner,
        accumulator,
        mask=mask,
    )


@torch.library.custom_op("trtllm::qwen4_exp_hc_gate_mix", mutates_args=())
def hc_gate_mix(
    input: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """Fuse sigmoid, per-stream multiplication, and the stream mean."""
    rows, hyper_hidden_size = input.shape
    assert gate.shape == input.shape
    assert hyper_hidden_size % hc_count == 0
    assert input.is_cuda and input.stride(1) == 1 and gate.stride(1) == 1
    hidden_size = hyper_hidden_size // hc_count
    output = input.new_empty((rows, hidden_size))
    block_size = 512
    grid = (rows, triton.cdiv(hidden_size, block_size))
    launch_with_pdl = _pdl_enabled(rows)
    with torch.cuda.device(input.device.index):
        _hc_gate_mix_kernel[grid](
            input,
            gate,
            output,
            input.stride(0),
            gate.stride(0),
            output.stride(0),
            hidden_size=hidden_size,
            hc_count=hc_count,
            block_size=block_size,
            launch_with_pdl=launch_with_pdl,
            launch_pdl=launch_with_pdl,
        )
    return output


@hc_gate_mix.register_fake
def _(input: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    del gate
    return input.new_empty((input.shape[0], input.shape[1] // hc_count))


@triton.jit
def _hc_combine_kernel(
    residual_ptr,
    block_ptr,
    injection_ptr,
    output_ptr,
    shared_expert_ptr,
    shared_expert_gate_ptr,
    stride_residual,
    stride_block,
    stride_injection,
    stride_output,
    stride_shared_expert,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_size: tl.constexpr,
    gate_shared_expert: tl.constexpr,
    launch_with_pdl: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets_inner = tile * block_size + tl.arange(0, block_size)
    mask = offsets_inner < hidden_size

    if launch_with_pdl:
        tl.extra.cuda.gdc_wait()
    block = _load_block_output(
        block_ptr,
        shared_expert_ptr,
        shared_expert_gate_ptr,
        row,
        stride_block,
        stride_shared_expert,
        offsets_inner,
        mask,
        gate_shared_expert,
    )
    for stream in tl.static_range(hc_count):
        offset = stream * hidden_size + offsets_inner
        residual = tl.load(
            residual_ptr + row * stride_residual + offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        injection = tl.load(injection_ptr + row * stride_injection + stream).to(tl.float32)
        injection = 2.0 * tl.sigmoid(injection / hc_count)
        tl.store(
            output_ptr + row * stride_output + offset,
            residual + injection * block,
            mask=mask,
        )
    if launch_with_pdl:
        tl.extra.cuda.gdc_launch_dependents()


@torch.library.custom_op("trtllm::qwen4_exp_hc_combine", mutates_args=())
def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
    shared_expert_output: Optional[torch.Tensor] = None,
    shared_expert_gate_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fuse injection sigmoid, broadcast multiply, and residual addition.

    Passing the shared-expert pair means ``block_output`` carries only the
    routed experts' output and the kernel completes the MoE block's
    ``+ sigmoid(gate) * shared`` itself, one launch fewer.
    """
    rows, hyper_hidden_size = residual.shape
    assert hyper_hidden_size % hc_count == 0
    hidden_size = hyper_hidden_size // hc_count
    assert block_output.shape == (rows, hidden_size)
    assert injection_logits.shape == (rows, hc_count)
    assert residual.is_cuda
    assert residual.stride(1) == block_output.stride(1) == injection_logits.stride(1) == 1
    gate_shared_expert = _check_shared_expert_branch(
        block_output, shared_expert_output, shared_expert_gate_logits
    )
    output = torch.empty_like(residual)
    block_size = 512
    grid = (rows, triton.cdiv(hidden_size, block_size))
    launch_with_pdl = _pdl_enabled(rows)
    with torch.cuda.device(residual.device.index):
        _hc_combine_kernel[grid](
            residual,
            block_output,
            injection_logits,
            output,
            shared_expert_output,
            shared_expert_gate_logits,
            residual.stride(0),
            block_output.stride(0),
            injection_logits.stride(0),
            output.stride(0),
            shared_expert_output.stride(0) if gate_shared_expert else 0,
            hidden_size=hidden_size,
            hc_count=hc_count,
            block_size=block_size,
            gate_shared_expert=gate_shared_expert,
            launch_with_pdl=launch_with_pdl,
            launch_pdl=launch_with_pdl,
        )
    return output


@hc_combine.register_fake
def _(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
    shared_expert_output: Optional[torch.Tensor] = None,
    shared_expert_gate_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del block_output, injection_logits, hc_count
    del shared_expert_output, shared_expert_gate_logits
    return torch.empty_like(residual)


@triton.jit
def _hc_combine_norm_kernel(
    residual_ptr,
    block_ptr,
    injection_ptr,
    weight_ptr,
    output_ptr,
    normed_ptr,
    shared_expert_ptr,
    shared_expert_gate_ptr,
    stride_residual,
    stride_block,
    stride_injection,
    stride_output,
    stride_normed,
    stride_shared_expert,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    shared_weight: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
    padded_tiles: tl.constexpr,
    gate_shared_expert: tl.constexpr,
    launch_with_pdl: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    stream = tl.program_id(1)
    tile_ids = tl.arange(0, padded_tiles)
    offsets_inner = tile_ids[:, None] * block_size + tl.arange(0, block_size)[None, :]
    mask = offsets_inner < hidden_size
    offsets = stream * hidden_size + offsets_inner
    weight_offsets = offsets_inner if shared_weight else offsets

    if launch_with_pdl:
        tl.extra.cuda.gdc_wait()
    residual = tl.load(
        residual_ptr + row * stride_residual + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    block = _load_block_output(
        block_ptr,
        shared_expert_ptr,
        shared_expert_gate_ptr,
        row,
        stride_block,
        stride_shared_expert,
        offsets_inner,
        mask,
        gate_shared_expert,
    )
    # Issue the norm weight with the input loads even though it is only needed
    # after the reduction below: that reduction is block-wide, and its barriers
    # keep the compiler from hoisting the load, which would otherwise serialize
    # a whole memory round-trip behind it.
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
    injection = tl.load(injection_ptr + row * stride_injection + stream).to(tl.float32)
    injection = 2.0 * tl.sigmoid(injection / hc_count)

    # Preserve the materialized BF16/FP16 boundary before normalization.
    output = (residual + injection * block).to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + row * stride_output + offsets, output, mask=mask)

    output_fp32 = output.to(tl.float32)
    sum_squares = tl.sum(tl.sum(output_fp32 * output_fp32, axis=1), axis=0)
    reciprocal_rms = tl.rsqrt(sum_squares / hidden_size + eps)
    normed = output_fp32 * reciprocal_rms
    normed += normed * weight
    if launch_with_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(normed_ptr + row * stride_normed + offsets, normed, mask=mask)


@torch.library.custom_op("trtllm::qwen4_exp_hc_combine_norm", mutates_args=())
def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
    shared_expert_output: Optional[torch.Tensor] = None,
    shared_expert_gate_logits: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse combine with the next grouped Gemma RMSNorm.

    Passing the shared-expert pair means ``block_output`` carries only the
    routed experts' output and the kernel completes the MoE block's
    ``+ sigmoid(gate) * shared`` itself, one launch fewer.
    """
    rows, hyper_hidden_size = residual.shape
    assert hyper_hidden_size % hc_count == 0
    hidden_size = hyper_hidden_size // hc_count
    assert block_output.shape == (rows, hidden_size)
    assert injection_logits.shape == (rows, hc_count)
    assert norm_weight.numel() in (hidden_size, hyper_hidden_size)
    assert residual.is_cuda
    assert residual.stride(1) == block_output.stride(1) == injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous()
    gate_shared_expert = _check_shared_expert_branch(
        block_output, shared_expert_output, shared_expert_gate_logits
    )
    output = torch.empty_like(residual)
    normed = torch.empty_like(residual)
    # One tile per row wherever it fits. The grid is (rows, hc_count), so at
    # decode sizes only hc_count CTAs cover the whole GPU and the kernel is
    # bound by the load-reduce-store chain inside each of them: a single
    # contiguous [1, next_pow2(hidden)] block reads that chain's inputs in
    # wider vectors than the equivalent [hidden/512, 512] tiling and needs one
    # less level of reduction, for the same number of lanes. The cap keeps
    # per-thread work bounded for rows too wide to hold in registers at once.
    block_size = min(triton.next_power_of_2(hidden_size), _COMBINE_NORM_MAX_BLOCK)
    padded_tiles = triton.next_power_of_2(triton.cdiv(hidden_size, block_size))
    launch_with_pdl = _pdl_enabled(rows)
    with torch.cuda.device(residual.device.index):
        _hc_combine_norm_kernel[(rows, hc_count)](
            residual,
            block_output,
            injection_logits,
            norm_weight,
            output,
            normed,
            shared_expert_output,
            shared_expert_gate_logits,
            residual.stride(0),
            block_output.stride(0),
            injection_logits.stride(0),
            output.stride(0),
            normed.stride(0),
            shared_expert_output.stride(0) if gate_shared_expert else 0,
            hidden_size=hidden_size,
            hc_count=hc_count,
            shared_weight=norm_weight.numel() == hidden_size,
            eps=eps,
            block_size=block_size,
            padded_tiles=padded_tiles,
            gate_shared_expert=gate_shared_expert,
            launch_with_pdl=launch_with_pdl,
            # Four warps avoid oversubscribing the fixed-width HC reduction.
            num_warps=4,
            launch_pdl=launch_with_pdl,
        )
    return output, normed


@hc_combine_norm.register_fake
def _(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
    shared_expert_output: Optional[torch.Tensor] = None,
    shared_expert_gate_logits: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, norm_weight, eps, hc_count
    del shared_expert_output, shared_expert_gate_logits
    return torch.empty_like(residual), torch.empty_like(residual)


__all__ = ["hc_combine", "hc_combine_norm", "hc_gate_mix", "hc_silu"]
