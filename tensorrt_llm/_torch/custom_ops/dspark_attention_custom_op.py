# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSpark rolling-window attention."""

import functools

import cutlass
import cutlass.cute as cute
import torch

from ..._utils import get_sm_version, is_sm_100f
from ...logger import logger
from ..cute_dsl_kernels.blackwell.dspark_attention import DSparkAttentionKernel

_INDEX_DTYPE_TO_CUTLASS = {
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


def _log_unsupported(reason: str, key: str) -> bool:
    logger.debug_once(
        f"Falling back from fused DSpark attention: {reason}",
        key=("fused_dspark_attention_unsupported", key),
    )
    return False


def is_fused_dspark_attention_supported(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    attn_sink: torch.Tensor,
) -> bool:
    """Return whether the production DSpark shape can use the CuteDSL op."""
    if not is_sm_100f():
        return _log_unsupported(f"SM {get_sm_version()} is not SM100/SM103", "sm_version")
    if not all(t.is_cuda for t in (q, main_kv, block_kv, kv_cache, slots, start_pos, attn_sink)):
        return _log_unsupported("all inputs must be CUDA tensors", "device")
    if q.dtype != torch.bfloat16:
        return _log_unsupported(f"q dtype must be BF16, got {q.dtype}", "q_dtype")
    if main_kv.dtype != q.dtype or block_kv.dtype != q.dtype or kv_cache.dtype != q.dtype:
        return _log_unsupported(
            "main_kv, block_kv, and kv_cache dtypes must match q; "
            f"got {main_kv.dtype}, {block_kv.dtype}, and {kv_cache.dtype}",
            "kv_dtype",
        )
    if attn_sink.dtype != torch.float32:
        return _log_unsupported(
            f"attn_sink dtype must be FP32, got {attn_sink.dtype}", "attn_sink_dtype"
        )
    if slots.dtype not in (torch.int32, torch.int64) or start_pos.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return _log_unsupported(
            f"slots and start_pos must use INT32 or INT64, got {slots.dtype} and {start_pos.dtype}",
            "index_dtype",
        )
    if slots.dtype != start_pos.dtype:
        return _log_unsupported(
            f"slots and start_pos dtypes must match, got {slots.dtype} and {start_pos.dtype}",
            "index_dtype_match",
        )
    if q.ndim != 4 or main_kv.ndim != 2 or block_kv.ndim != 3 or kv_cache.ndim != 3:
        return _log_unsupported(
            "expected q/main_kv/block_kv/kv_cache ranks 4/2/3/3, "
            f"got {q.ndim}/{main_kv.ndim}/{block_kv.ndim}/{kv_cache.ndim}",
            "tensor_ranks",
        )
    head_dim = q.shape[-1]
    if head_dim != 512:
        return _log_unsupported(f"head_dim must be 512, got {head_dim}", "head_dim")
    expected_main_kv_shape = (q.shape[0], head_dim)
    if main_kv.shape != expected_main_kv_shape:
        return _log_unsupported(
            f"main_kv shape must be {expected_main_kv_shape}, got {tuple(main_kv.shape)}",
            "main_kv_shape",
        )
    expected_block_kv_shape = (q.shape[0], q.shape[1], head_dim)
    if block_kv.shape != expected_block_kv_shape:
        return _log_unsupported(
            f"block_kv shape must be {expected_block_kv_shape}, got {tuple(block_kv.shape)}",
            "block_kv_shape",
        )
    if kv_cache.shape[-1] != head_dim:
        return _log_unsupported(
            f"kv_cache trailing dimension must be {head_dim}, got {kv_cache.shape[-1]}",
            "kv_cache_shape",
        )
    expected_sink_shape = (q.shape[2],)
    if attn_sink.shape != expected_sink_shape:
        return _log_unsupported(
            f"attn_sink shape must be {expected_sink_shape}, got {tuple(attn_sink.shape)}",
            "attn_sink_shape",
        )
    expected_batch_shape = (q.shape[0],)
    if slots.shape != expected_batch_shape or start_pos.shape != expected_batch_shape:
        return _log_unsupported(
            "slots and start_pos shapes must match the q batch size; "
            f"expected {expected_batch_shape}, got {tuple(slots.shape)} and {tuple(start_pos.shape)}",
            "index_shape",
        )
    if not q.is_contiguous():
        return _log_unsupported("q must be contiguous", "q_layout")
    if not main_kv.is_contiguous():
        return _log_unsupported("main_kv must be contiguous", "main_kv_layout")
    if not block_kv.is_contiguous():
        return _log_unsupported("block_kv must be contiguous", "block_kv_layout")
    if not slots.is_contiguous():
        return _log_unsupported("slots must be contiguous", "slots_layout")
    if not start_pos.is_contiguous():
        return _log_unsupported("start_pos must be contiguous", "start_pos_layout")
    if not attn_sink.is_contiguous():
        return _log_unsupported("attn_sink must be contiguous", "attn_sink_layout")
    return True


@functools.cache
def _compile_fused_dspark_attention(
    block_size: int,
    num_heads: int,
    head_dim: int,
    window_size: int,
    cache_stride: tuple[int, ...],
    index_dtype: torch.dtype,
    softmax_scale: float,
):
    # Batch is deliberately symbolic: DSpark's block/head geometry is fixed by
    # the model, while the generation batch changes from iteration to iteration.
    # One warmup compile therefore covers every eager and CUDA-graph batch size.
    batch_size = cute.sym_int()
    q_shape = (batch_size, block_size, num_heads, head_dim)
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, q_shape, stride_order=(3, 2, 1, 0)
    )
    main_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (batch_size, head_dim), stride_order=(1, 0)
    )
    block_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch_size, block_size, head_dim),
        stride_order=(2, 1, 0),
    )
    cache_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (cute.sym_int(), window_size, head_dim),
        stride=cache_stride,
    )
    index_cutlass_dtype = _INDEX_DTYPE_TO_CUTLASS[index_dtype]
    slots_fake = cute.runtime.make_fake_compact_tensor(
        index_cutlass_dtype, (batch_size,), stride_order=(0,)
    )
    start_fake = cute.runtime.make_fake_compact_tensor(
        index_cutlass_dtype, (batch_size,), stride_order=(0,)
    )
    sink_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (num_heads,), stride_order=(0,)
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, q_shape, stride_order=(3, 2, 1, 0)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkAttentionKernel(
        window_size=window_size,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        softmax_scale=softmax_scale,
    )
    return cute.compile(
        kernel,
        q_fake,
        main_fake,
        block_fake,
        cache_fake,
        slots_fake,
        start_fake,
        sink_fake,
        output_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_attention",
    mutates_args=("kv_cache",),
    device_types="cuda",
)
def cute_dsl_dspark_attention(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run fused DSpark cache update + sliding-window MQA attention.

    ``slots`` values must be in ``[0, kv_cache.shape[0])``, and ``start_pos``
    values must be nonnegative. The support check does not inspect tensor values,
    so callers are responsible for enforcing these preconditions.
    """
    if not is_fused_dspark_attention_supported(
        q, main_kv, block_kv, kv_cache, slots, start_pos, attn_sink
    ):
        raise ValueError(
            "cute_dsl_dspark_attention requires contiguous BF16 production DSpark tensors "
            f"with head_dim=512 on SM100/SM103; got SM {get_sm_version()}"
        )
    output = torch.empty_like(q)
    compiled = _compile_fused_dspark_attention(
        q.shape[1],
        q.shape[2],
        q.shape[3],
        kv_cache.shape[1],
        tuple(kv_cache.stride()),
        slots.dtype,
        softmax_scale,
    )
    compiled(q, main_kv, block_kv, kv_cache, slots, start_pos, attn_sink, output)
    return output


@torch.library.register_fake("trtllm::cute_dsl_dspark_attention")
def _(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)
