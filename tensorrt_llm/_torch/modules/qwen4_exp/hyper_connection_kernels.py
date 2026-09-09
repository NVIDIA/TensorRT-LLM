# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's Qwen3.8-Flash-Next Hyper-Connection kernels.
"""CUDA kernels for the Qwen4-Exp gated Hyper-Connection.

These kernels fuse only numerically adjacent pointwise and normalization
operations. The two low-rank projections remain regular GEMMs so TRT-LLM can
select the best cuBLAS or low-M CuTe DSL implementation for each token count.
"""

import os

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
    if input.ndim != 2 or not input.is_cuda or input.stride(1) != 1:
        raise ValueError("HC SiLU requires a row-major two-dimensional CUDA tensor")
    if hc_count <= 0:
        raise ValueError("HC stream count must be positive")
    rows, width = input.shape
    output = torch.empty_like(input)
    if rows == 0:
        return output
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
    if input.ndim != 2 or gate.shape != input.shape:
        raise ValueError("HC gate and input must have matching two-dimensional shapes")
    if hc_count <= 0 or input.shape[1] % hc_count:
        raise ValueError("HC stream count must divide the hidden dimension")
    if (
        not input.is_cuda
        or gate.device != input.device
        or gate.dtype != input.dtype
        or input.stride(1) != 1
        or gate.stride(1) != 1
    ):
        raise ValueError("HC gate and input must be matching row-major CUDA tensors")
    rows, hyper_hidden_size = input.shape
    hidden_size = hyper_hidden_size // hc_count
    output = input.new_empty((rows, hidden_size))
    if rows == 0:
        return output
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
    stride_residual,
    stride_block,
    stride_injection,
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
    block = tl.load(
        block_ptr + row * stride_block + offsets_inner,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
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
) -> torch.Tensor:
    """Fuse injection sigmoid, broadcast multiply, and residual addition."""
    if residual.ndim != 2 or hc_count <= 0 or residual.shape[1] % hc_count:
        raise ValueError("HC residual geometry is incompatible with its stream count")
    rows, hyper_hidden_size = residual.shape
    hidden_size = hyper_hidden_size // hc_count
    if block_output.shape != (rows, hidden_size) or injection_logits.shape != (rows, hc_count):
        raise ValueError("HC combine operands have incompatible shapes")
    if (
        not residual.is_cuda
        or block_output.device != residual.device
        or injection_logits.device != residual.device
        or block_output.dtype != residual.dtype
        or injection_logits.dtype != residual.dtype
        or residual.stride(1) != 1
        or block_output.stride(1) != 1
        or injection_logits.stride(1) != 1
    ):
        raise ValueError("HC combine operands must be matching row-major CUDA tensors")
    output = torch.empty_like(residual)
    if rows == 0:
        return output
    block_size = 512
    grid = (rows, triton.cdiv(hidden_size, block_size))
    launch_with_pdl = _pdl_enabled(rows)
    with torch.cuda.device(residual.device.index):
        _hc_combine_kernel[grid](
            residual,
            block_output,
            injection_logits,
            output,
            residual.stride(0),
            block_output.stride(0),
            injection_logits.stride(0),
            output.stride(0),
            hidden_size=hidden_size,
            hc_count=hc_count,
            block_size=block_size,
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
) -> torch.Tensor:
    del block_output, injection_logits, hc_count
    return torch.empty_like(residual)


@triton.jit
def _hc_combine_norm_kernel(
    residual_ptr,
    block_ptr,
    injection_ptr,
    weight_ptr,
    output_ptr,
    normed_ptr,
    stride_residual,
    stride_block,
    stride_injection,
    stride_output,
    stride_normed,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    shared_weight: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
    padded_tiles: tl.constexpr,
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
    block = tl.load(
        block_ptr + row * stride_block + offsets_inner,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse combine with the next grouped Gemma RMSNorm."""
    if residual.ndim != 2 or hc_count <= 0 or residual.shape[1] % hc_count:
        raise ValueError("HC residual geometry is incompatible with its stream count")
    rows, hyper_hidden_size = residual.shape
    hidden_size = hyper_hidden_size // hc_count
    if block_output.shape != (rows, hidden_size) or injection_logits.shape != (rows, hc_count):
        raise ValueError("HC combine-norm operands have incompatible shapes")
    if norm_weight.ndim != 1 or norm_weight.shape[0] not in (hidden_size, hyper_hidden_size):
        raise ValueError("HC norm weight has an incompatible shape")
    if (
        not residual.is_cuda
        or block_output.device != residual.device
        or injection_logits.device != residual.device
        or norm_weight.device != residual.device
        or block_output.dtype != residual.dtype
        or injection_logits.dtype != residual.dtype
        or norm_weight.dtype != residual.dtype
        or residual.stride(1) != 1
        or block_output.stride(1) != 1
        or injection_logits.stride(1) != 1
        or not norm_weight.is_contiguous()
    ):
        raise ValueError("HC combine-norm operands must be matching contiguous CUDA tensors")
    output = torch.empty_like(residual)
    normed = torch.empty_like(residual)
    if rows == 0:
        return output, normed
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
            residual.stride(0),
            block_output.stride(0),
            injection_logits.stride(0),
            output.stride(0),
            normed.stride(0),
            hidden_size=hidden_size,
            hc_count=hc_count,
            shared_weight=norm_weight.numel() == hidden_size,
            eps=eps,
            block_size=block_size,
            padded_tiles=padded_tiles,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, norm_weight, eps, hc_count
    return torch.empty_like(residual), torch.empty_like(residual)


__all__ = ["hc_combine", "hc_combine_norm", "hc_gate_mix", "hc_silu"]
