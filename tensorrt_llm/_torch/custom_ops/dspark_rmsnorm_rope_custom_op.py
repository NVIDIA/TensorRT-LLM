# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSpark RMSNorm and RoPE."""

import functools

import cutlass
import cutlass.cute as cute
import torch

from ..._utils import get_sm_version
from ..cute_dsl_kernels.blackwell.dspark_rmsnorm_rope import (
    DSparkRMSNormRoPEDraftBlockKernel,
    DSparkRMSNormRoPEKernel,
    DSparkRMSNormRoPEPageWriteKernel,
)

_DSV4_DSPARK_HEAD_DIM = 512
_DSV4_DSPARK_ROPE_DIM = 64
_DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE = 8

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


def _validate_dspark_preparation(
    x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor
) -> None:
    if (
        x.ndim != 3
        or x.shape[-1] != _DSV4_DSPARK_HEAD_DIM
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or weight.shape != (_DSV4_DSPARK_HEAD_DIM,)
        or freqs.dtype != torch.float32
        or freqs.shape != (x.shape[0] * x.shape[1], _DSV4_DSPARK_ROPE_DIM // 2, 2)
        or not all(
            t.is_cuda and t.device == x.device and t.is_contiguous() for t in (x, weight, freqs)
        )
    ):
        raise ValueError(
            "DSpark preparation requires contiguous BF16 [batch, tokens, 512] inputs, "
            "BF16 [512] weights and FP32 [batch * tokens, 32, 2] frequencies"
        )


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
    if (
        x.shape[1] != 1
        or kv_cache.ndim != 3
        or kv_cache.shape[1] not in (16, 32, 64, 128, 256)
        or kv_cache.shape[2] != _DSV4_DSPARK_HEAD_DIM
        or kv_cache.dtype != x.dtype
        or kv_cache.stride(1) != _DSV4_DSPARK_HEAD_DIM
        or kv_cache.stride(2) != 1
        or block_tables.ndim != 2
        or block_tables.shape[0] != batch
        or block_tables.dtype != torch.int32
        or block_tables.stride(1) != 1
        or not all(
            t.is_cuda and t.device == x.device
            for t in (kv_cache, block_tables, start_pos, capacities)
        )
        or not all(
            t.shape == (batch,) and t.dtype == torch.int64 and t.is_contiguous()
            for t in (start_pos, capacities)
        )
    ):
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


@functools.cache
def _compile_dspark_rmsnorm_rope_draft_block(block_size: int, eps: float, device: int):
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
    if block_size not in (5, 6):
        raise ValueError("DSpark draft block size must be 5 or 6")
    with torch.cuda.device(x.device):
        compiled = _compile_dspark_rmsnorm_rope_draft_block(block_size, eps, x.device.index)
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
