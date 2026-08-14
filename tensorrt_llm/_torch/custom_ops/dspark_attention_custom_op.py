# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSpark rolling-window attention."""

import functools

import cutlass
import cutlass.cute as cute
import torch

from ..._utils import get_sm_version, is_sm_100f
from ..cute_dsl_kernels.blackwell.dspark_attention import DSparkAttentionKernel

_INDEX_DTYPE_TO_CUTLASS = {
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


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
        return False
    if not all(t.is_cuda for t in (q, main_kv, block_kv, kv_cache, slots, start_pos, attn_sink)):
        return False
    if q.dtype != torch.bfloat16:
        return False
    if main_kv.dtype != q.dtype or block_kv.dtype != q.dtype or kv_cache.dtype != q.dtype:
        return False
    if attn_sink.dtype != torch.float32:
        return False
    if slots.dtype not in (torch.int32, torch.int64) or start_pos.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return False
    if slots.dtype != start_pos.dtype:
        return False
    if q.ndim != 4 or main_kv.ndim != 2 or block_kv.ndim != 3 or kv_cache.ndim != 3:
        return False
    head_dim = q.shape[-1]
    return (
        head_dim == 512
        and main_kv.shape == (q.shape[0], head_dim)
        and block_kv.shape == (q.shape[0], q.shape[1], head_dim)
        and kv_cache.shape[-1] == head_dim
        and attn_sink.shape == (q.shape[2],)
        and slots.shape == (q.shape[0],)
        and start_pos.shape == (q.shape[0],)
        and q.is_contiguous()
        and main_kv.is_contiguous()
        and block_kv.is_contiguous()
        and slots.is_contiguous()
        and start_pos.is_contiguous()
        and attn_sink.is_contiguous()
    )


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
