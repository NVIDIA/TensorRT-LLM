# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSpark RMSNorm and RoPE."""

import functools

import cutlass
import cutlass.cute as cute
import torch

from ..._utils import get_sm_version
from ..attention.kernels.blackwell.dspark.rmsnorm_rope import (
    DSparkBackfillKernel,
    DSparkConcatFP8QuantKernel,
    DSparkPagedBackfillKernel,
    DSparkRMSNormFP8QuantKernel,
    DSparkRMSNormRoPECacheWriteKernel,
    DSparkRMSNormRoPEDraftBlockKernel,
    DSparkRMSNormRoPEKernel,
    DSparkRMSNormRoPEPageWriteKernel,
)
from ..utils import mxfp8_scale_infer_shape

_DSV4_DSPARK_HEAD_DIM = 512
_DSV4_DSPARK_ROPE_DIM = 64
_DSV4_DSPARK_WINDOW_SIZE = 128
_DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE = 8
_DSV4_DSPARK_BLOCK_SIZES = tuple(range(1, _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE + 1))
_DSV4_DSPARK_ARCH_BY_SM = {
    100: "sm_100",
    103: "sm_103",
    107: "sm_107",
}
# RMSNorm/RoPE support is checked independently of attention allocator selection.
_DSPARK_RMSNORM_ROPE_SM_VERSIONS = frozenset((100, 103, 107))


_dspark_fp8_input_compile_allowed = False


@functools.cache
def _get_dspark_arch_str(sm_version: int | None = None) -> str | None:
    """Return the CuTe allocator arch for a supported DSpark GPU."""
    if sm_version is None:
        sm_version = get_sm_version()
    return _DSV4_DSPARK_ARCH_BY_SM.get(sm_version)


@functools.cache
def _is_dspark_rmsnorm_rope_sm_supported(sm_version: int | None = None) -> bool:
    """Return whether the fused RMSNorm/RoPE kernels support this GPU."""
    if sm_version is None:
        sm_version = get_sm_version()
    return sm_version in _DSPARK_RMSNORM_ROPE_SM_VERSIONS


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
    if not _is_dspark_rmsnorm_rope_sm_supported() or not all(t.is_cuda for t in operands):
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


def _is_dspark_attention_rope_supported(
    x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor
) -> bool:
    return freqs.shape[1:] == (_DSV4_DSPARK_ROPE_DIM // 2, 2) and (
        is_fused_dspark_rmsnorm_rope_supported(
            x, weight, freqs, num_heads=1, rope_dim=_DSV4_DSPARK_ROPE_DIM
        )
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
        and _is_dspark_attention_rope_supported(x, weight, freqs)
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
        and _is_dspark_attention_rope_supported(x, weight, freqs)
    )


def _is_dspark_page_write_layout_supported(
    x: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    start_pos: torch.Tensor,
    capacities: torch.Tensor,
) -> bool:
    batch = x.shape[0]
    return (
        x.shape[1] == 1
        and kv_cache.ndim == 3
        and kv_cache.shape[1] in (16, 32, 64, 128, 256)
        and kv_cache.shape[2] == _DSV4_DSPARK_HEAD_DIM
        and kv_cache.dtype == x.dtype
        and kv_cache.stride(1) == _DSV4_DSPARK_HEAD_DIM
        and kv_cache.stride(2) == 1
        and block_tables.ndim == 2
        and block_tables.shape[0] == batch
        and block_tables.dtype == torch.int32
        and block_tables.stride(1) == 1
        and all(
            t.is_cuda and t.device == x.device
            for t in (kv_cache, block_tables, start_pos, capacities)
        )
        and all(
            t.shape == (batch,) and t.dtype == torch.int64 and t.is_contiguous()
            for t in (start_pos, capacities)
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
    *,
    block_tables: torch.Tensor | None = None,
    capacities: torch.Tensor | None = None,
) -> bool:
    """Return whether RMSNorm/RoPE can directly prepare the attention inputs."""
    if block_tables is not None:
        return (
            capacities is not None
            and main_x.ndim == 3
            and main_x.shape[-1] == _DSV4_DSPARK_HEAD_DIM
            and main_x.shape[0] == block_x.shape[0]
            and all(t.device == main_x.device for t in (weight, main_freqs, block_x, block_freqs))
            and _is_dspark_attention_rope_supported(main_x, weight, main_freqs)
            and _is_dspark_page_write_layout_supported(
                main_x, kv_cache, block_tables, start_pos, capacities
            )
            and _is_dspark_draft_block_supported(block_x, weight, block_freqs)
        )
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
    norm_dim: int,
):
    rows = cute.sym_int()
    freq_rows = cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (rows, hidden_dim), stride_order=(1, 0)
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
def _compile_dspark_backfill(num_stages: int, eps: float):
    tokens = cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), _DSV4_DSPARK_HEAD_DIM), stride_order=(1, 0)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (num_stages, _DSV4_DSPARK_HEAD_DIM), stride_order=(1, 0)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (cute.sym_int(), _DSV4_DSPARK_ROPE_DIM // 2, 2),
        stride_order=(2, 1, 0),
    )
    mask_fake = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (tokens,), stride_order=(0,))
    positions_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (tokens,), stride_order=(0,)
    )
    slots_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (cute.sym_int(),), stride_order=(0,)
    )
    windows_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (cute.sym_int(), num_stages, _DSV4_DSPARK_WINDOW_SIZE, _DSV4_DSPARK_HEAD_DIM),
        stride=(cute.sym_int64(), cute.sym_int64(), _DSV4_DSPARK_HEAD_DIM, 1),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkBackfillKernel(
        _DSV4_DSPARK_HEAD_DIM, _DSV4_DSPARK_ROPE_DIM, eps, _DSV4_DSPARK_WINDOW_SIZE, num_stages
    )
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        freqs_fake,
        mask_fake,
        positions_fake,
        slots_fake,
        windows_fake,
        cutlass.Int32(1),
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.cache
def _compile_dspark_paged_backfill(num_stages: int, page_size: int, eps: float, device: int):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DSpark paged backfill must be warmed up before CUDA graph capture")
    tokens = cute.sym_int()
    kv = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), _DSV4_DSPARK_HEAD_DIM), stride_order=(1, 0)
    )
    weight = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (num_stages, _DSV4_DSPARK_HEAD_DIM), stride_order=(1, 0)
    )
    freqs = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int(), _DSV4_DSPARK_ROPE_DIM // 2, 2), stride_order=(2, 1, 0)
    )
    mask = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (tokens,), stride_order=(0,))
    positions = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (tokens,), stride_order=(0,))
    tables = cute.runtime.make_fake_tensor(
        cutlass.Int32, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    capacities = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (cute.sym_int(),), stride_order=(0,)
    )
    pools = tuple(
        cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (cute.sym_int(), page_size, _DSV4_DSPARK_HEAD_DIM),
            stride=(cute.sym_int64(), _DSV4_DSPARK_HEAD_DIM, 1),
        )
        for _ in range(num_stages)
    )
    return cute.compile(
        DSparkPagedBackfillKernel(
            _DSV4_DSPARK_HEAD_DIM, _DSV4_DSPARK_ROPE_DIM, eps, page_size, num_stages
        ),
        kv,
        weight,
        freqs,
        mask,
        positions,
        tables,
        capacities,
        pools,
        cutlass.Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
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
    # Fused QKV projections supply regular strided views; this specialization
    # uses compact rows in its compiled signature.
    x_flat = x.view(-1, x.shape[-1]).contiguous()
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
    _validate_dspark_preparation(x, weight, freqs)
    block_size = x.shape[1]
    if block_size not in _DSV4_DSPARK_BLOCK_SIZES:
        raise ValueError(
            f"DSpark draft block size must be 1 to {_DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE}, "
            f"got {block_size}"
        )
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


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_backfill",
    mutates_args=("windows",),
    device_types="cuda",
)
def cute_dsl_dspark_backfill(
    kv: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    windows: torch.Tensor,
    eps: float,
) -> None:
    """Write the RMSNorm/RoPE of each valid ``[G, M]`` interim token's per-stage K/V into its window.

    ``kv`` is ``[G, M, num_stages * head_dim]``, ``weight`` the stacked
    ``[num_stages, head_dim]`` K/V norms, ``freqs`` the ``[positions, rope_dim // 2, 2]``
    RoPE table and ``windows`` ``[N, num_stages, window_size, head_dim]``.
    Rows with a false ``mask`` are left untouched.
    """
    compiled = _compile_dspark_backfill(weight.shape[0], eps)
    compiled(
        kv.view(-1, _DSV4_DSPARK_HEAD_DIM),
        weight,
        freqs,
        mask.view(-1).view(torch.uint8),
        positions.view(-1),
        slots,
        windows,
        mask.shape[1],
    )


@torch.library.register_fake("trtllm::cute_dsl_dspark_backfill")
def _(
    kv: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    windows: torch.Tensor,
    eps: float,
) -> None:
    return None


def _validate_dspark_preparation(
    x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor
) -> None:
    if x.ndim != 3 or x.shape[-1] != _DSV4_DSPARK_HEAD_DIM or x.dtype != torch.bfloat16:
        raise ValueError(
            f"DSpark preparation input must be BF16 [batch, tokens, {_DSV4_DSPARK_HEAD_DIM}], "
            f"got {x.dtype} {tuple(x.shape)}"
        )
    if weight.dtype != x.dtype or weight.shape != (_DSV4_DSPARK_HEAD_DIM,):
        raise ValueError(
            f"DSpark preparation weight must be BF16 [{_DSV4_DSPARK_HEAD_DIM}], "
            f"got {weight.dtype} {tuple(weight.shape)}"
        )
    expected_freqs_shape = (x.shape[0] * x.shape[1], _DSV4_DSPARK_ROPE_DIM // 2, 2)
    if freqs.dtype != torch.float32 or freqs.shape != expected_freqs_shape:
        raise ValueError(
            f"DSpark preparation frequencies must be FP32 {list(expected_freqs_shape)}, "
            f"got {freqs.dtype} {tuple(freqs.shape)}"
        )
    for name, tensor in (("input", x), ("weight", weight), ("frequencies", freqs)):
        if not tensor.is_cuda:
            raise ValueError(
                f"DSpark preparation {name} must be a CUDA tensor, got {tensor.device}"
            )
        if tensor.device != x.device:
            raise ValueError(
                f"DSpark preparation {name} must be on {x.device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(
                f"DSpark preparation {name} must be contiguous, got strides {tensor.stride()}"
            )


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
        cutlass.Uint8, (scale_elems,), stride_order=(0,)
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
        cutlass.Uint8, (scale_elems,), stride_order=(0,)
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
    output_scale = x.new_empty((mxfp8_scale_infer_shape([list(x_flat.shape)]),), dtype=torch.uint8)
    compiled = _compile_dspark_rmsnorm_fp8_quant(hidden_dim, eps)
    compiled(
        x_flat,
        weight,
        output,
        output_scale,
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
        x.new_empty((mxfp8_scale_infer_shape([[rows, x.shape[-1]]]),), dtype=torch.uint8),
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
    output_scale = first.new_empty(
        (mxfp8_scale_infer_shape([[rows, hidden_dim]]),), dtype=torch.uint8
    )
    compiled = _compile_dspark_concat_fp8_quant(hidden_dim)
    compiled(
        first,
        second,
        output,
        output_scale,
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
        first.new_empty((mxfp8_scale_infer_shape([[rows, hidden_dim]]),), dtype=torch.uint8),
    )


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_paged_backfill", mutates_args=("pools",), device_types="cuda"
)
def cute_dsl_dspark_paged_backfill(
    kv: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    capacities: torch.Tensor,
    pools: list[torch.Tensor],
    eps: float,
) -> None:
    """Write valid [request, interim, stage] RMSNorm/RoPE rows through absolute page mappings."""
    if not pools:
        raise ValueError("DSpark paged backfill requires a pool for each draft stage")
    stages = len(pools)
    dim = _DSV4_DSPARK_HEAD_DIM
    page_size = pools[0].shape[1] if pools[0].ndim == 3 else 0
    if (
        positions.ndim != 2
        or positions.dtype != torch.int64
        or mask.shape != positions.shape
        or mask.dtype != torch.bool
        or kv.shape not in ((*positions.shape, stages * dim), (*positions.shape, stages, dim))
        or weight.shape != (stages, dim)
        or kv.dtype != torch.bfloat16
        or weight.dtype != kv.dtype
        or freqs.ndim != 3
        or freqs.shape[1:] != (_DSV4_DSPARK_ROPE_DIM // 2, 2)
        or freqs.dtype != torch.float32
        or not all(
            tensor.is_cuda and tensor.device == kv.device and tensor.is_contiguous()
            for tensor in (kv, weight, freqs, mask, positions)
        )
        or page_size <= 0
        or not all(
            pool.is_cuda
            and pool.device == kv.device
            and pool.dtype == kv.dtype
            and pool.ndim == 3
            and pool.shape[1:] == (page_size, dim)
            and pool.stride()[1:] == (dim, 1)
            for pool in pools
        )
        or block_tables.ndim != 2
        or block_tables.shape[0] != positions.shape[0]
        or block_tables.dtype != torch.int32
        or block_tables.stride(1) != 1
        or block_tables.device != kv.device
        or capacities.shape != (positions.shape[0],)
        or capacities.dtype != torch.int64
        or not capacities.is_contiguous()
        or capacities.device != kv.device
    ):
        raise ValueError("Invalid DSpark paged backfill input layout")
    if not positions.numel():
        return
    with torch.cuda.device(kv.device):
        compiled = _compile_dspark_paged_backfill(stages, page_size, eps, kv.device.index)
        compiled(
            kv.view(-1, _DSV4_DSPARK_HEAD_DIM),
            weight,
            freqs,
            mask.view(-1).view(torch.uint8),
            positions.view(-1),
            block_tables,
            capacities,
            tuple(pools),
            mask.shape[1],
        )


@torch.library.register_fake("trtllm::cute_dsl_dspark_paged_backfill")
def _(
    kv: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    capacities: torch.Tensor,
    pools: list[torch.Tensor],
    eps: float,
) -> None:
    return None


@functools.cache
def _compile_dspark_rmsnorm_rope_page_write(page_size: int, eps: float, device: int):
    # Device identity scopes compilation to the active CUDA target.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSpark RMSNorm/RoPE page-write must be warmed up before CUDA graph capture"
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
        (pages, page_size, _DSV4_DSPARK_HEAD_DIM),
        stride=(cute.sym_int64(), _DSV4_DSPARK_HEAD_DIM, 1),
    )
    tables_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32, (rows, cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    start_pos_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (rows,), stride_order=(0,)
    )
    capacities_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (rows,), stride_order=(0,)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkRMSNormRoPEPageWriteKernel(
        _DSV4_DSPARK_HEAD_DIM,
        _DSV4_DSPARK_ROPE_DIM,
        eps,
        page_size,
    )
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        freqs_fake,
        cache_fake,
        tables_fake,
        start_pos_fake,
        capacities_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_rmsnorm_rope_page_write",
    mutates_args=("kv_cache",),
    device_types="cuda",
)
def cute_dsl_dspark_rmsnorm_rope_page_write(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    start_pos: torch.Tensor,
    capacities: torch.Tensor,
    eps: float,
) -> None:
    """Normalize/rotate [batch, 1, 512] KV into pages; zero capacities mask dummy rows."""
    _validate_dspark_preparation(x, weight, freqs)
    batch = x.shape[0]
    if not _is_dspark_page_write_layout_supported(x, kv_cache, block_tables, start_pos, capacities):
        raise ValueError(
            "DSpark page write requires BF16 pages with contiguous tokens, INT32 "
            "page-table rows and contiguous INT64 positions/capacities on x's device"
        )
    with torch.cuda.device(x.device):
        compiled = _compile_dspark_rmsnorm_rope_page_write(kv_cache.shape[1], eps, x.device.index)
        compiled(
            x.view(batch, _DSV4_DSPARK_HEAD_DIM),
            weight,
            freqs,
            kv_cache,
            block_tables,
            start_pos,
            capacities,
        )


@torch.library.register_fake("trtllm::cute_dsl_dspark_rmsnorm_rope_page_write")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    start_pos: torch.Tensor,
    capacities: torch.Tensor,
    eps: float,
) -> None:
    return None
