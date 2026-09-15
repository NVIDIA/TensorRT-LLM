# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSpark RMSNorm and RoPE."""

import functools

import cutlass
import cutlass.cute as cute
import torch

from ..._utils import get_sm_version
from ..cute_dsl_kernels.blackwell.dspark_rmsnorm_rope import (
    DSparkRMSNormRoPECacheWriteKernel,
    DSparkRMSNormRoPEDraftBlockKernel,
    DSparkRMSNormRoPEKernel,
)

_DSV4_DSPARK_HEAD_DIM = 512
_DSV4_DSPARK_ROPE_DIM = 64
_DSV4_DSPARK_WINDOW_SIZE = 128
_DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE = 8
_DSV4_DSPARK_BLOCK_SIZES = (5, 6)
_DSV4_DSPARK_ARCH_BY_SM = {
    100: "sm_100",
    103: "sm_103",
}


@functools.cache
def _get_dspark_arch_str(sm_version: int | None = None) -> str | None:
    """Return the CuTe allocator arch for a supported DSpark GPU."""
    if sm_version is None:
        sm_version = get_sm_version()
    return _DSV4_DSPARK_ARCH_BY_SM.get(sm_version)


def is_fused_dspark_rmsnorm_rope_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
) -> bool:
    """Return whether tensors satisfy the production fused-op contract."""
    if _get_dspark_arch_str() is None or not all(t.is_cuda for t in (x, weight, freqs)):
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if freqs.dtype != torch.float32:
        return False
    if x.ndim < 2 or x.shape[-1] % 32 != 0:
        return False
    if weight.shape != (x.shape[-1],):
        return False
    if rope_dim < 0 or rope_dim > x.shape[-1] or rope_dim % 2 != 0:
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
        and x.is_contiguous()
        and weight.is_contiguous()
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
        x.shape == (batch, 1, _DSV4_DSPARK_HEAD_DIM)
        and is_fused_dspark_rmsnorm_rope_supported(
            x, weight, freqs, num_heads=1, rope_dim=_DSV4_DSPARK_ROPE_DIM
        )
        and kv_cache.is_cuda
        and kv_cache.dtype == x.dtype
        and kv_cache.ndim == 3
        and kv_cache.shape[1:] == (_DSV4_DSPARK_WINDOW_SIZE, _DSV4_DSPARK_HEAD_DIM)
        and kv_cache.stride(1) == _DSV4_DSPARK_HEAD_DIM
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


def _is_dspark_draft_block_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
) -> bool:
    block_size = x.shape[1] if x.ndim == 3 else -1
    return (
        block_size in _DSV4_DSPARK_BLOCK_SIZES
        and x.shape[-1:] == (_DSV4_DSPARK_HEAD_DIM,)
        and is_fused_dspark_rmsnorm_rope_supported(
            x, weight, freqs, num_heads=1, rope_dim=_DSV4_DSPARK_ROPE_DIM
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
        and _is_dspark_draft_block_supported(block_x, weight, block_freqs)
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
):
    rows = cute.sym_int()
    freq_rows = cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (rows, hidden_dim), stride_order=(1, 0)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (hidden_dim,), stride_order=(0,)
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
def _compile_dspark_rmsnorm_rope_cache_write(eps: float):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSpark RMSNorm/RoPE cache-write must be warmed up before CUDA graph capture"
        )
    rows = cute.sym_int()
    pages = cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (rows, _DSV4_DSPARK_HEAD_DIM), stride_order=(1, 0)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (_DSV4_DSPARK_HEAD_DIM,), stride_order=(0,)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (rows, _DSV4_DSPARK_ROPE_DIM // 2, 2),
        stride_order=(2, 1, 0),
    )
    cache_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (pages, _DSV4_DSPARK_WINDOW_SIZE, _DSV4_DSPARK_HEAD_DIM),
        stride=(cute.sym_int64(), _DSV4_DSPARK_HEAD_DIM, 1),
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
        _DSV4_DSPARK_HEAD_DIM,
        _DSV4_DSPARK_ROPE_DIM,
        eps,
        _DSV4_DSPARK_WINDOW_SIZE,
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
def _compile_dspark_rmsnorm_rope_draft_block(block_size: int, eps: float):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSpark RMSNorm/RoPE draft-block must be warmed up before CUDA graph capture"
        )
    rows = cute.sym_int()
    batch = cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (rows, _DSV4_DSPARK_HEAD_DIM), stride_order=(1, 0)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (_DSV4_DSPARK_HEAD_DIM,), stride_order=(0,)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (rows, _DSV4_DSPARK_ROPE_DIM // 2, 2),
        stride_order=(2, 1, 0),
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch, _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE, _DSV4_DSPARK_HEAD_DIM),
        stride_order=(2, 1, 0),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkRMSNormRoPEDraftBlockKernel(
        _DSV4_DSPARK_HEAD_DIM,
        _DSV4_DSPARK_ROPE_DIM,
        eps,
        block_size,
        _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE,
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
    "trtllm::cute_dsl_dspark_rmsnorm_rope",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_dspark_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
) -> torch.Tensor:
    """Apply fused RMSNorm and adjacent-pair RoPE to contiguous BF16 rows."""
    if not is_fused_dspark_rmsnorm_rope_supported(x, weight, freqs, num_heads, rope_dim):
        raise ValueError(
            "cute_dsl_dspark_rmsnorm_rope requires contiguous BF16 tensors on "
            "an SM100 or SM103 GPU with a valid FP32 frequency view; "
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
    )
    compiled(x_flat, weight, freqs, output)
    return output.view(original_shape)


@torch.library.register_fake("trtllm::cute_dsl_dspark_rmsnorm_rope")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
) -> torch.Tensor:
    return torch.empty_like(x)


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
        x.view(-1, _DSV4_DSPARK_HEAD_DIM),
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
    "trtllm::cute_dsl_dspark_rmsnorm_rope_draft_block",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_dspark_rmsnorm_rope_draft_block(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Prepare a zero-padded draft block after validating the fused contract."""
    block_size = x.shape[1]
    compiled = _compile_dspark_rmsnorm_rope_draft_block(block_size, eps)
    output = x.new_empty((x.shape[0], _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE, _DSV4_DSPARK_HEAD_DIM))
    compiled(
        x.view(-1, _DSV4_DSPARK_HEAD_DIM),
        weight,
        freqs,
        output,
    )
    return output


@torch.library.register_fake("trtllm::cute_dsl_dspark_rmsnorm_rope_draft_block")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del weight, freqs, eps
    return x.new_empty((x.shape[0], _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE, _DSV4_DSPARK_HEAD_DIM))
