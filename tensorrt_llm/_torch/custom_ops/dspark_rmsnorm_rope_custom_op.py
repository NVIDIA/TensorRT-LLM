# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSpark RMSNorm and RoPE."""

import functools

import cutlass
import cutlass.cute as cute
import torch

from ..._utils import get_sm_version
from ..attention.kernels.blackwell.dspark_rmsnorm_rope import (
    DSparkMlaBlockRoPEKernel,
    DSparkMlaCtxNormRoPEKernel,
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
# The RMSNorm/RoPE kernels are plain SIMT CuTe kernels with no architecture-specific
# instructions, so they run on more GPUs than the DSV4 DSpark attention kernel gated
# by _DSV4_DSPARK_ARCH_BY_SM.
_DSPARK_RMSNORM_ROPE_SM_VERSIONS = frozenset((100, 103, 107))


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
    norm_dim: int,
):
    rows = cute.sym_int()
    freq_rows = cute.sym_int()
    # Regular row-strided input, as is_fused_dspark_rmsnorm_rope_supported
    # admits: the MLA drafter normalizes column slices of fused GEMM outputs.
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


@functools.cache
def _compile_dspark_mla_ctx_norm_rope(
    num_layers: int, kv_lora_rank: int, rope_dim: int, eps: float
):
    head_dim = kv_lora_rank + rope_dim
    tokens = cute.sym_int()
    x_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (tokens, num_layers * head_dim), stride=(cute.sym_int64(), 1)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (num_layers, kv_lora_rank), stride_order=(1, 0)
    )
    positions_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (tokens,), stride_order=(0,)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int(), rope_dim // 2, 2), stride_order=(2, 1, 0)
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), head_dim), stride_order=(1, 0)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkMlaCtxNormRoPEKernel(num_layers, kv_lora_rank, rope_dim, eps)
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        positions_fake,
        freqs_fake,
        output_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def dspark_mla_ctx_norm_rope(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm + RoPE of the context latents of every drafter layer, one launch.

    Args:
        latent: ``[N, L, kv_lora_rank + rope_dim]`` BF16 with contiguous
            ``(L, head_dim)`` rows (any token stride).
        norm_weight: ``[L, kv_lora_rank]`` BF16 latent RMSNorm weights.
        positions: ``[N]`` int64 RoPE positions.
        freqs: ``[P, rope_dim // 2, 2]`` FP32 (cos, sin) table indexed by position.
        eps: RMSNorm epsilon.
    Returns:
        ``[L, N, head_dim]``: for each layer the values of
        ``cute_dsl_dspark_rmsnorm_rope(latent[:, l], norm_weight[l], freqs[positions],
        1, rope_dim, eps, True, True, False, kv_lora_rank)``, bitwise.
    """
    num_tokens, num_layers, head_dim = latent.shape
    kv_lora_rank = norm_weight.shape[1]
    rope_dim = head_dim - kv_lora_rank
    assert latent.dtype == torch.bfloat16 and norm_weight.dtype == torch.bfloat16
    assert latent.stride(2) == 1 and latent.stride(1) == head_dim
    assert norm_weight.shape[0] == num_layers and norm_weight.is_contiguous()
    assert positions.dtype == torch.int64 and positions.shape == (num_tokens,)
    assert freqs.dtype == torch.float32 and freqs.is_contiguous()
    assert freqs.shape[1:] == (rope_dim // 2, 2)
    out = torch.empty((num_layers, num_tokens, head_dim), dtype=latent.dtype, device=latent.device)
    if num_tokens == 0:
        return out
    compiled = _compile_dspark_mla_ctx_norm_rope(num_layers, kv_lora_rank, rope_dim, float(eps))
    # detach(): DLPack export refuses tensors that require grad (parameters).
    compiled(
        latent.detach().as_strided((num_tokens, num_layers * head_dim), (latent.stride(0), 1)),
        norm_weight.detach(),
        positions.contiguous(),
        freqs.detach(),
        out.view(num_layers * num_tokens, head_dim),
    )
    return out


@functools.cache
def _compile_dspark_mla_block_rope(
    num_heads: int,
    qk_nope_dim: int,
    kv_lora_rank: int,
    rope_dim: int,
    eps: float,
):
    head_dim = kv_lora_rank + rope_dim
    q_rows = cute.sym_int()
    tokens = cute.sym_int()
    q_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (q_rows, qk_nope_dim + rope_dim), stride=(cute.sym_int64(), 1)
    )
    query_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (q_rows, head_dim), stride=(cute.sym_int64(), 1)
    )
    kv_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (tokens, head_dim), stride=(cute.sym_int64(), 1)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (kv_lora_rank,), stride_order=(0,)
    )
    positions_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (tokens,), stride_order=(0,)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int(), rope_dim // 2, 2), stride_order=(2, 1, 0)
    )
    pool_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (cute.sym_int(), cute.sym_int(), head_dim),
        stride=(cute.sym_int64(), cute.sym_int64(), 1),
    )
    pages_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (tokens,), stride_order=(0,))
    offsets_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (tokens,), stride_order=(0,)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkMlaBlockRoPEKernel(num_heads, qk_nope_dim, kv_lora_rank, rope_dim, eps)
    return cute.compile(
        kernel,
        q_fake,
        query_fake,
        kv_fake,
        weight_fake,
        positions_fake,
        freqs_fake,
        pool_fake,
        pages_fake,
        offsets_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def dspark_mla_block_rope(
    q: torch.Tensor,
    query: torch.Tensor,
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    pool: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    eps: float,
) -> None:
    """The block RoPE work of one MLA drafter layer in one launch.

    Args:
        q: ``[T, H, qk_nope_dim + rope_dim]`` BF16 query rows (unit inner stride,
            ``H`` heads per token).
        query: ``[T, H, kv_lora_rank + rope_dim]`` BF16 absorbed query; only its
            rope columns are written.
        kv: ``[T, kv_lora_rank + rope_dim]`` BF16 block latents (row-strided).
        norm_weight: ``[kv_lora_rank]`` BF16 latent RMSNorm weight.
        positions: ``[T]`` int64 RoPE positions of the block tokens.
        freqs: ``[P, rope_dim // 2, 2]`` FP32 (cos, sin) table indexed by position.
        pool: ``[pages, page_size, kv_lora_rank + rope_dim]`` paged latent pool.
        pages, offsets: ``[T]`` int64 destination page and in-page offset.
        eps: RMSNorm epsilon.

    Writes what ``cute_dsl_dspark_rope_into(q, freqs[positions], query, H, rope_dim,
    kv_lora_rank)`` writes into ``query`` and what
    ``scatter_rows_to_pool(cute_dsl_dspark_rmsnorm_rope(kv, norm_weight,
    freqs[positions], 1, rope_dim, eps, True, True, False, kv_lora_rank), pool,
    pages, offsets)`` writes into ``pool``, bitwise.
    """
    num_tokens, num_heads, q_dim = q.shape
    head_dim = query.shape[-1]
    kv_lora_rank = norm_weight.shape[0]
    rope_dim = head_dim - kv_lora_rank
    assert q.dtype == query.dtype == kv.dtype == pool.dtype == torch.bfloat16
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
    assert query.shape[:2] == (num_tokens, num_heads) and kv.shape == (num_tokens, head_dim)
    assert q.stride(2) == 1 and q.stride(0) == num_heads * q.stride(1)
    assert query.stride(2) == 1 and query.stride(0) == num_heads * query.stride(1)
    assert kv.stride(1) == 1 and pool.stride(2) == 1 and pool.shape[2] == head_dim
    assert positions.dtype == pages.dtype == offsets.dtype == torch.int64
    assert positions.numel() == pages.numel() == offsets.numel() == num_tokens
    assert freqs.dtype == torch.float32 and freqs.is_contiguous()
    assert freqs.shape[1:] == (rope_dim // 2, 2)
    if num_tokens == 0:
        return
    compiled = _compile_dspark_mla_block_rope(
        num_heads, q_dim - rope_dim, kv_lora_rank, rope_dim, float(eps)
    )
    # detach(): DLPack export refuses tensors that require grad (parameters).
    compiled(
        q.detach().as_strided((num_tokens * num_heads, q_dim), (q.stride(1), 1)),
        query.detach().as_strided((num_tokens * num_heads, head_dim), (query.stride(1), 1)),
        kv.detach(),
        norm_weight.detach(),
        positions.reshape(-1).contiguous(),
        freqs.detach(),
        pool.detach(),
        pages.reshape(-1).contiguous(),
        offsets.reshape(-1).contiguous(),
    )
