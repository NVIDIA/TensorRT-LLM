# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch custom op for fused DSpark RMSNorm and RoPE."""

import functools

import cutlass
import cutlass.cute as cute
import torch

from ..._utils import get_sm_version
from ..cute_dsl_kernels.blackwell.dspark_rmsnorm_rope import (
    DSparkConcatFP8QuantKernel,
    DSparkRMSNormFP8QuantKernel,
    DSparkRMSNormRoPEBlockPageKernel,
    DSparkRMSNormRoPECacheWriteKernel,
    DSparkRMSNormRoPEKernel,
)

_DSPARK_HEAD_DIM = 512
_DSPARK_ROPE_DIM = 64
_DSPARK_WINDOW_SIZE = 128
_DSPARK_PAGE_SIZE = 8
_DSPARK_BLOCK_SIZES = (5, 6)
_DSPARK_ARCH_BY_SM = {
    100: "sm_100",
    103: "sm_103",
    107: "sm_107",
}

# Worker initialization is process-local and single-threaded. These flags only
# permit compile-cache misses during that initialization window.
_dspark_preparation_compile_allowed = False
_dspark_fp8_input_compile_allowed = False


@functools.cache
def _get_dspark_arch_str(sm_version: int | None = None) -> str | None:
    """Return the CuTe allocator arch for a supported DSpark GPU."""
    if sm_version is None:
        sm_version = get_sm_version()
    return _DSPARK_ARCH_BY_SM.get(sm_version)


def _has_regular_row_stride(x: torch.Tensor) -> bool:
    """Return whether leading dimensions flatten to non-overlapping strided rows."""
    if x.is_contiguous():
        return True
    if x.stride(-1) != 1:
        return False

    outer_dims = [dim for dim in range(x.ndim - 1) if x.shape[dim] > 1]
    if not outer_dims:
        return True

    row_dim = outer_dims[-1]
    if x.stride(row_dim) < x.shape[-1]:
        return False
    return all(
        x.stride(dim) == x.stride(next_dim) * x.shape[next_dim]
        for dim, next_dim in zip(outer_dims, outer_dims[1:])
    )


def is_fused_dspark_rmsnorm_rope_supported(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    norm_dim: int | None = None,
) -> bool:
    """Return whether tensors satisfy the production fused-op contract.

    ``weight`` is None for a kernel built with ``apply_weight=False``; the
    weight checks are then vacuous rather than a reason to reject, and the
    caller has no tensor to offer in the first place.
    """
    operands = (x, freqs) if weight is None else (x, weight, freqs)
    if _get_dspark_arch_str() is None or not all(t.is_cuda for t in operands):
        return False
    if x.dtype != torch.bfloat16 or (weight is not None and weight.dtype != torch.bfloat16):
        return False
    if freqs.dtype != torch.float32:
        return False
    if x.ndim < 2 or x.shape[-1] % 32 != 0:
        return False
    if rope_dim < 0 or rope_dim > x.shape[-1] or rope_dim % 2 != 0:
        return False
    # norm_dim defaults to the whole row; the only other supported value is the
    # nope prefix, where the weight spans just that prefix.
    effective_norm_dim = x.shape[-1] if norm_dim is None else norm_dim
    if effective_norm_dim not in (x.shape[-1], x.shape[-1] - rope_dim):
        return False
    if effective_norm_dim % 32 != 0:
        return False
    if weight is not None and weight.shape != (effective_norm_dim,):
        return False
    if (x.shape[-1] - rope_dim) % 32 != 0 or (rope_dim // 2) % 32 != 0:
        return False
    rows = x.numel() // x.shape[-1]
    if num_heads <= 0 or rows % num_heads != 0:
        return False
    return (
        freqs.ndim == 3
        and freqs.shape[0] == rows // num_heads
        and freqs.shape[1] >= max(1, rope_dim // 2)
        and freqs.shape[2] == 2
        and _has_regular_row_stride(x)
        and (weight is None or weight.is_contiguous())
        and freqs.is_contiguous()
    )


def _is_dspark_cache_write_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
) -> bool:
    batch = x.shape[0] if x.ndim == 3 else -1
    return (
        x.shape == (batch, 1, _DSPARK_HEAD_DIM)
        and is_fused_dspark_rmsnorm_rope_supported(
            x, weight, freqs, num_heads=1, rope_dim=_DSPARK_ROPE_DIM
        )
        and kv_cache.is_cuda
        and kv_cache.dtype == x.dtype
        and kv_cache.ndim == 3
        and kv_cache.shape[1:] == (_DSPARK_WINDOW_SIZE, _DSPARK_HEAD_DIM)
        and kv_cache.stride(1) == _DSPARK_HEAD_DIM
        and kv_cache.stride(2) == 1
        and slots.is_cuda
        and slots.dtype == torch.int64
        and slots.shape == (batch,)
        and slots.is_contiguous()
        and start_pos.is_cuda
        and start_pos.dtype == torch.int64
        and start_pos.shape == (batch,)
        and start_pos.is_contiguous()
    )


def _is_dspark_block_page_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
) -> bool:
    block_size = x.shape[1] if x.ndim == 3 else -1
    return (
        block_size in _DSPARK_BLOCK_SIZES
        and x.shape[-1:] == (_DSPARK_HEAD_DIM,)
        and is_fused_dspark_rmsnorm_rope_supported(
            x, weight, freqs, num_heads=1, rope_dim=_DSPARK_ROPE_DIM
        )
    )


def is_fused_dspark_attention_preparation_supported(
    main_x: torch.Tensor,
    block_x: torch.Tensor,
    weight: torch.Tensor,
    main_freqs: torch.Tensor,
    block_freqs: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
) -> bool:
    """Return whether RMSNorm/RoPE can directly prepare the attention inputs."""
    return (
        main_x.shape[0] == block_x.shape[0]
        and _is_dspark_cache_write_supported(main_x, weight, main_freqs, kv_cache, slots, start_pos)
        and _is_dspark_block_page_supported(block_x, weight, block_freqs)
    )


@functools.cache
def _compile_fused_dspark_rmsnorm_rope(
    hidden_dim: int,
    rope_dim: int,
    num_heads: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
    norm_dim: int,
):
    rows = cute.sym_int()
    freq_rows = cute.sym_int()
    x_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (rows, hidden_dim),
        stride=(cute.sym_int64(), 1),
    )
    # None, not a fake tensor, when the kernel does not scale by a weight: the
    # operand then does not exist in the compiled signature, so the call site
    # has nothing to pass and nothing to allocate.
    weight_fake = (
        cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (norm_dim,), stride_order=(0,))
        if apply_weight
        else None
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (freq_rows, cute.sym_int(), 2),
        stride_order=(2, 1, 0),
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (rows, hidden_dim), stride_order=(1, 0)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkRMSNormRoPEKernel(
        hidden_dim,
        rope_dim,
        num_heads,
        eps,
        apply_weight,
        apply_rmsnorm,
        inverse_rope,
        norm_dim=norm_dim,
    )
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        freqs_fake,
        output_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.cache
def _compile_fused_dspark_rope_into(
    hidden_dim: int,
    rope_dim: int,
    num_heads: int,
    eps: float,
    out_dim: int,
    out_rope_offset: int,
):
    """Rope-only variant that writes into a wider destination row.

    Every argument is part of the @functools.cache key on purpose: out_dim and
    out_rope_offset change the generated addressing, so sharing a compiled
    kernel across two offsets would write the rotated pairs to the wrong
    columns -- wrong numbers, no error, and for a speculative drafter that only
    shows up as a lower acceptance length.
    """
    rows = cute.sym_int()
    freq_rows = cute.sym_int()
    x_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (rows, hidden_dim),
        stride=(cute.sym_int64(), 1),
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (freq_rows, cute.sym_int(), 2),
        stride_order=(2, 1, 0),
    )
    output_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (rows, out_dim),
        stride=(cute.sym_int64(), 1),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkRMSNormRoPEKernel(
        hidden_dim,
        rope_dim,
        num_heads,
        eps,
        False,
        False,
        False,
        norm_dim=hidden_dim - rope_dim,
        out_rope_offset=out_rope_offset,
        write_nope=False,
    )
    return cute.compile(
        kernel,
        x_fake,
        None,
        freqs_fake,
        output_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def _r128c4_scale_numel(rows: int, hidden_dim: int) -> int:
    return ((rows + 127) // 128 * 128) * (hidden_dim // 32)


def precompile_dspark_fp8_input_fusions(
    qkv_hidden_dim: int | None,
    q_lora_rank: int | None,
    eps: float,
) -> None:
    """Precompile dynamic-row DSpark FP8 input producers during model init."""
    global _dspark_fp8_input_compile_allowed
    if get_sm_version() != 107:
        return
    if _dspark_fp8_input_compile_allowed:
        raise RuntimeError("DSpark FP8 input precompile is already in progress")

    _dspark_fp8_input_compile_allowed = True
    try:
        if qkv_hidden_dim is not None:
            _compile_dspark_concat_fp8_quant(qkv_hidden_dim)
        if q_lora_rank is not None:
            _compile_dspark_rmsnorm_fp8_quant(q_lora_rank, eps)
    finally:
        _dspark_fp8_input_compile_allowed = False


@functools.cache
def _compile_dspark_rmsnorm_fp8_quant(hidden_dim: int, eps: float):
    if not _dspark_fp8_input_compile_allowed:
        raise RuntimeError(
            "DSpark RMSNorm/FP8 quant kernel was not precompiled during worker initialization"
        )

    rows = cute.sym_int()
    scale_elems = cute.sym_int()
    x_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (rows, hidden_dim),
        stride=(cute.sym_int64(), 1),
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (hidden_dim,), stride_order=(0,)
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (rows, hidden_dim), stride_order=(1, 0)
    )
    scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E8M0FNU, (scale_elems,), stride_order=(0,)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkRMSNormFP8QuantKernel(hidden_dim, eps)
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        output_fake,
        scale_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.cache
def _compile_dspark_concat_fp8_quant(hidden_dim: int):
    if not _dspark_fp8_input_compile_allowed:
        raise RuntimeError(
            "DSpark concat/FP8 quant kernel was not precompiled during worker initialization"
        )

    first_rows = cute.sym_int()
    second_rows = cute.sym_int()
    output_rows = cute.sym_int()
    scale_elems = cute.sym_int()
    first_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (first_rows, hidden_dim),
        stride=(cute.sym_int64(), 1),
    )
    second_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (second_rows, hidden_dim),
        stride=(cute.sym_int64(), 1),
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (output_rows, hidden_dim), stride_order=(1, 0)
    )
    scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E8M0FNU, (scale_elems,), stride_order=(0,)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkConcatFP8QuantKernel(hidden_dim)
    return cute.compile(
        kernel,
        first_fake,
        second_fake,
        output_fake,
        scale_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def precompile_dspark_attention_preparation(block_size: int, eps: float) -> None:
    """Precompile the dynamic-batch cache-write and block-page variants."""
    global _dspark_preparation_compile_allowed
    if _get_dspark_arch_str() is None or block_size not in _DSPARK_BLOCK_SIZES:
        return
    if _dspark_preparation_compile_allowed:
        raise RuntimeError("DSpark attention preparation precompile is already in progress")

    _dspark_preparation_compile_allowed = True
    try:
        _compile_dspark_rmsnorm_rope_cache_write(eps)
        _compile_dspark_rmsnorm_rope_block_page(block_size, eps)
    finally:
        _dspark_preparation_compile_allowed = False


@functools.cache
def _compile_dspark_rmsnorm_rope_cache_write(eps: float):
    if not _dspark_preparation_compile_allowed:
        raise RuntimeError(
            "DSpark RMSNorm/RoPE cache-write kernel was not precompiled "
            "during worker initialization"
        )

    rows = cute.sym_int()
    pages = cute.sym_int()
    x_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (rows, _DSPARK_HEAD_DIM),
        stride=(cute.sym_int64(), 1),
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (_DSPARK_HEAD_DIM,), stride_order=(0,)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (rows, _DSPARK_ROPE_DIM // 2, 2),
        stride_order=(2, 1, 0),
    )
    cache_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (pages, _DSPARK_WINDOW_SIZE, _DSPARK_HEAD_DIM),
        stride=(cute.sym_int64(), _DSPARK_HEAD_DIM, 1),
    )
    slots_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (rows,), stride_order=(0,))
    start_pos_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (rows,), stride_order=(0,)
    )
    slots_i32_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (rows,), stride_order=(0,)
    )
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (rows,), stride_order=(0,)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkRMSNormRoPECacheWriteKernel(
        _DSPARK_HEAD_DIM,
        _DSPARK_ROPE_DIM,
        eps,
        _DSPARK_WINDOW_SIZE,
    )
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        freqs_fake,
        cache_fake,
        slots_fake,
        start_pos_fake,
        slots_i32_fake,
        cache_seqs_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.cache
def _compile_dspark_rmsnorm_rope_block_page(block_size: int, eps: float):
    if not _dspark_preparation_compile_allowed:
        raise RuntimeError(
            "DSpark RMSNorm/RoPE block-page kernel was not precompiled during worker initialization"
        )

    rows = cute.sym_int()
    batch = cute.sym_int()
    x_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (rows, _DSPARK_HEAD_DIM),
        stride=(cute.sym_int64(), 1),
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (_DSPARK_HEAD_DIM,), stride_order=(0,)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (rows, _DSPARK_ROPE_DIM // 2, 2),
        stride_order=(2, 1, 0),
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch, _DSPARK_PAGE_SIZE, _DSPARK_HEAD_DIM),
        stride_order=(2, 1, 0),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkRMSNormRoPEBlockPageKernel(
        _DSPARK_HEAD_DIM,
        _DSPARK_ROPE_DIM,
        eps,
        block_size,
        _DSPARK_PAGE_SIZE,
    )
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        freqs_fake,
        output_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_rmsnorm_fp8_quant",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_dspark_rmsnorm_fp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce a prequantized MXFP8 Linear input after model-init precompile."""
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    output = x.new_empty(x_flat.shape, dtype=torch.float8_e4m3fn)
    output_scale = x.new_empty(
        (_r128c4_scale_numel(x_flat.shape[0], hidden_dim),), dtype=torch.uint8
    )
    compiled = _compile_dspark_rmsnorm_fp8_quant(hidden_dim, eps)
    compiled(
        x_flat,
        weight,
        output,
        output_scale.view(torch.float8_e8m0fnu),
    )
    return output, output_scale


@torch.library.register_fake("trtllm::cute_dsl_dspark_rmsnorm_fp8_quant")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, eps
    rows = x.numel() // x.shape[-1]
    return (
        x.new_empty((rows, x.shape[-1]), dtype=torch.float8_e4m3fn),
        x.new_empty((_r128c4_scale_numel(rows, x.shape[-1]),), dtype=torch.uint8),
    )


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_concat_fp8_quant",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_dspark_concat_fp8_quant(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate row ranges into one prequantized MXFP8 Linear input."""
    rows = first.shape[0] + second.shape[0]
    hidden_dim = first.shape[1]
    output = first.new_empty((rows, hidden_dim), dtype=torch.float8_e4m3fn)
    output_scale = first.new_empty((_r128c4_scale_numel(rows, hidden_dim),), dtype=torch.uint8)
    compiled = _compile_dspark_concat_fp8_quant(hidden_dim)
    compiled(
        first,
        second,
        output,
        output_scale.view(torch.float8_e8m0fnu),
    )
    return output, output_scale


@torch.library.register_fake("trtllm::cute_dsl_dspark_concat_fp8_quant")
def _(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = first.shape[0] + second.shape[0]
    hidden_dim = first.shape[1]
    return (
        first.new_empty((rows, hidden_dim), dtype=torch.float8_e4m3fn),
        first.new_empty((_r128c4_scale_numel(rows, hidden_dim),), dtype=torch.uint8),
    )


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_rmsnorm_rope",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_dspark_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
    norm_dim: int | None = None,
) -> torch.Tensor:
    """Apply fused RMSNorm and adjacent-pair RoPE to regular BF16 rows.

    ``weight`` may be None when ``apply_weight`` is False. It is dropped on the
    way to the kernel either way, so a caller that has a weight lying around
    can keep passing it, and one that does not need not invent one.
    """
    if apply_weight and weight is None:
        raise ValueError("cute_dsl_dspark_rmsnorm_rope needs a weight when apply_weight is set")
    weight = weight if apply_weight else None
    if not is_fused_dspark_rmsnorm_rope_supported(x, weight, freqs, num_heads, rope_dim, norm_dim):
        raise ValueError(
            "cute_dsl_dspark_rmsnorm_rope requires regular row-strided BF16 tensors on "
            "an SM100, SM103, or SM107 GPU with a valid FP32 frequency view; "
            f"got SM {get_sm_version()}"
        )

    original_shape = x.shape
    x_flat = x.view(-1, x.shape[-1])
    output = torch.empty_like(x_flat)
    compiled = _compile_fused_dspark_rmsnorm_rope(
        x.shape[-1],
        rope_dim,
        num_heads,
        eps,
        apply_weight,
        apply_rmsnorm,
        inverse_rope,
        x.shape[-1] if norm_dim is None else norm_dim,
    )
    compiled(x_flat, weight, freqs, output)
    return output.view(original_shape)


@torch.library.register_fake("trtllm::cute_dsl_dspark_rmsnorm_rope")
def _(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
    norm_dim: int | None = None,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_rope_into",
    mutates_args=("out",),
    device_types="cuda",
)
def cute_dsl_dspark_rope_into(
    x: torch.Tensor,
    freqs: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    out_rope_offset: int,
) -> None:
    """Rotate x's trailing rope_dim and store it at out[..., out_rope_offset:].

    x keeps its own (narrower) row width; nothing is written outside the rope
    columns of `out`, so the caller owns the rest of the destination row.
    """
    # No weight operand: _compile_fused_dspark_rope_into builds the kernel with
    # apply_weight=False, so there is nothing for the call to scale by and
    # nothing to allocate per decode step.
    if not is_fused_dspark_rmsnorm_rope_supported(
        x, None, freqs, num_heads, rope_dim, x.shape[-1] - rope_dim
    ):
        raise ValueError(
            "cute_dsl_dspark_rope_into requires regular row-strided BF16 tensors on "
            "an SM100, SM103, or SM107 GPU with a valid FP32 frequency view; "
            f"got SM {get_sm_version()}"
        )
    # The destination is compiled with x's dtype and a (dyn, 1) stride, so it
    # needs the same checks as the source; view() below would otherwise fail
    # with an opaque RuntimeError, and a mismatched dtype would reach a kernel
    # compiled for the other one.
    if out.dtype != x.dtype or out.device != x.device or not _has_regular_row_stride(out):
        raise ValueError(
            "cute_dsl_dspark_rope_into needs a row-strided out on x's device with "
            f"x's dtype; got dtype={out.dtype}, device={out.device}, "
            f"stride={tuple(out.stride())}"
        )
    out_flat = out.view(-1, out.shape[-1])
    x_flat = x.view(-1, x.shape[-1])
    if out_flat.shape[0] != x_flat.shape[0]:
        raise ValueError(
            f"out must have one row per x row; got {out_flat.shape[0]} vs {x_flat.shape[0]}"
        )
    if out_rope_offset + rope_dim > out.shape[-1]:
        raise ValueError(
            f"rope slice [{out_rope_offset}, {out_rope_offset + rope_dim}) does not fit "
            f"a row of width {out.shape[-1]}"
        )
    compiled = _compile_fused_dspark_rope_into(
        x.shape[-1],
        rope_dim,
        num_heads,
        0.0,
        out.shape[-1],
        out_rope_offset,
    )
    compiled(x_flat, None, freqs, out_flat)


@torch.library.register_fake("trtllm::cute_dsl_dspark_rope_into")
def _(
    x: torch.Tensor,
    freqs: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    out_rope_offset: int,
) -> None:
    return None


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_rmsnorm_rope_cache_write",
    mutates_args=("kv_cache",),
    device_types="cuda",
)
def cute_dsl_dspark_rmsnorm_rope_cache_write(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare attention metadata after the caller validates the fused contract."""
    compiled = _compile_dspark_rmsnorm_rope_cache_write(eps)
    slots_i32 = torch.empty_like(slots, dtype=torch.int32)
    cache_seqs = torch.empty_like(start_pos, dtype=torch.int32)
    compiled(
        x.view(-1, _DSPARK_HEAD_DIM),
        weight,
        freqs,
        kv_cache,
        slots,
        start_pos,
        slots_i32,
        cache_seqs,
    )
    return slots_i32, cache_seqs


@torch.library.register_fake("trtllm::cute_dsl_dspark_rmsnorm_rope_cache_write")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del x, weight, freqs, kv_cache, eps
    return (
        torch.empty_like(slots, dtype=torch.int32),
        torch.empty_like(start_pos, dtype=torch.int32),
    )


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_rmsnorm_rope_block_page",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_dspark_rmsnorm_rope_block_page(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Prepare an eight-row page after the caller validates the fused contract."""
    block_size = x.shape[1]
    compiled = _compile_dspark_rmsnorm_rope_block_page(block_size, eps)
    output = x.new_empty((x.shape[0], _DSPARK_PAGE_SIZE, _DSPARK_HEAD_DIM))
    compiled(
        x.view(-1, _DSPARK_HEAD_DIM),
        weight,
        freqs,
        output,
    )
    return output


@torch.library.register_fake("trtllm::cute_dsl_dspark_rmsnorm_rope_block_page")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del weight, freqs, eps
    return x.new_empty((x.shape[0], _DSPARK_PAGE_SIZE, _DSPARK_HEAD_DIM))
