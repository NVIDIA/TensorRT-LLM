# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused Blackwell DSpark attention over KV pages and a draft block."""

from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.typing import Numeric, Pointer, Type

from ...logger import logger
from ..cute_dsl_kernels.blackwell.dspark.attention import DSparkAttention
from ..cute_dsl_kernels.blackwell.utils import make_ptr
from .dspark_rmsnorm_rope_custom_op import _get_dspark_arch_str

_DSV4_DSPARK_NUM_HEADS = 128
_DSV4_DSPARK_HEAD_DIM = 512
_DSV4_DSPARK_WINDOW_SIZE = 128
_DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE = 8
_DSV4_DSPARK_BLOCK_SIZES = (5, 6)
_DSV4_DSPARK_ROPE_DIM = 64


_dspark_attention_kernel_cache: dict[tuple[int, int, str], Callable[..., None]] = {}


def _make_compile_gmem_pointer(dtype: Type[Numeric], assumed_align: int) -> Pointer:
    """Create a compile-only pointer specimen without allocating device memory."""
    return make_ptr(
        dtype,
        0,
        cute.AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def _compile_dspark_attention(
    block_size: int,
    arch_str: str,
    page_size: int,
) -> Callable[..., None]:
    """Compile the pointer host wrapper without runtime tensor specimens."""
    num_heads, head_dim = _DSV4_DSPARK_NUM_HEADS, _DSV4_DSPARK_HEAD_DIM
    batch = cute.sym_int()
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch, block_size, num_heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Real tensor specimens used to make PyTorch's primary CUDA context current
    # implicitly. Fake-only compilation must do so before querying occupancy;
    # the returned stream is not part of the compiled TVM-FFI launch ABI.
    torch.cuda.current_stream()
    hardware_info = cutlass_utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(2)
    kernel = DSparkAttention(
        cutlass.Float32,
        (128, 128),
        (128, 256),
        max_active_clusters,
        _DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE,
        _DSV4_DSPARK_WINDOW_SIZE,
        0.0,
        seq_len_q=block_size,
        mma_qk_tiler_k=128,
        inverse_rope_dim=_DSV4_DSPARK_ROPE_DIM,
        arch_str=arch_str,
        history_page_size=page_size,
    )
    # Scalars and typed, aligned compile-only pointers define the runtime ABI.
    # ``output_fake`` keeps one tensor argument for TVM-FFI environment-stream
    # detection without tying compilation to a real worker buffer.
    compiled = cute.compile(
        kernel.wrapper,
        1,
        1,
        page_size * head_dim,
        1,
        1,
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.BFloat16, 16),
        _make_compile_gmem_pointer(cutlass.Int32, 4),
        _make_compile_gmem_pointer(cutlass.Int64, 8),
        _make_compile_gmem_pointer(cutlass.Int64, 8),
        _make_compile_gmem_pointer(cutlass.Int64, 8),
        _make_compile_gmem_pointer(cutlass.Float32, 4),
        _make_compile_gmem_pointer(cutlass.Float32, 4),
        output_fake,
        cutlass.Float32(1.0),
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    logger.info(
        "DSpark Attention enabled: implementation=dspark_attn, "
        f"block={block_size}, heads={num_heads}, head_dim={head_dim}"
    )
    return compiled


@torch.library.custom_op(
    "trtllm::fused_dsv4_dspark_attention", mutates_args=(), device_types="cuda"
)
def fused_dsv4_dspark_attention(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    valid_lengths: torch.Tensor,
    capacities: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Attend to a 128-token context window and fuse the inverse-RoPE epilogue."""
    arch_str = _get_dspark_arch_str()
    if arch_str is None:
        raise RuntimeError("Embedded DSpark attention requires SM100 or SM103")
    batch, block, heads, dim = q.shape
    page_size = kv_pages.shape[1]
    if block not in (5, 6) or (heads, dim) != (128, 512) or page_size not in (16, 32, 64, 128, 256):
        raise ValueError(
            "DSpark requires block 5/6, 128 heads, head_dim 512 and 16/32/64/128/256-token pages"
        )
    if q.dtype != torch.bfloat16 or draft_block.dtype != q.dtype or kv_pages.dtype != q.dtype:
        raise ValueError("DSpark Q, draft KV and pages must use BF16")
    if draft_block.shape != (batch, 8, dim) or kv_pages.shape[2] != dim:
        raise ValueError("DSpark draft KV must have shape [batch, 8, 512]")
    if (
        block_tables.dtype != torch.int32
        or block_tables.shape[0] != batch
        or block_tables.stride(1) != 1
    ):
        raise ValueError("DSpark page tables must have contiguous INT32 rows")
    for value in (positions, valid_lengths, capacities):
        if value.dtype != torch.int64 or value.shape != (batch,) or not value.is_contiguous():
            raise ValueError(
                "DSpark positions, lengths and capacities must be contiguous INT64 batch vectors"
            )
    if attn_sink.dtype != torch.float32 or attn_sink.shape != (heads,):
        raise ValueError("DSpark attention sink must be FP32 with one entry per head")
    if inverse_rope_freqs.dtype != torch.float32 or inverse_rope_freqs.shape != (
        batch,
        block,
        32,
        2,
    ):
        raise ValueError("DSpark inverse RoPE frequencies must be FP32 [batch, block, 32, 2]")
    if kv_pages.stride(1) != dim or kv_pages.stride(2) != 1 or kv_pages.stride(0) % 8:
        raise ValueError("DSpark pages require contiguous tokens and an aligned page stride")
    for value in (q, draft_block, attn_sink, inverse_rope_freqs):
        if not value.is_contiguous():
            raise ValueError("DSpark query, draft, sink and frequencies must be contiguous")
    if not all(
        value.is_cuda and value.device == q.device
        for value in (
            q,
            draft_block,
            kv_pages,
            block_tables,
            positions,
            valid_lengths,
            capacities,
            attn_sink,
            inverse_rope_freqs,
        )
    ):
        raise ValueError("All DSpark attention inputs must be on the same CUDA device")
    cache_key = (block, page_size, arch_str)
    compiled = _dspark_attention_kernel_cache.get(cache_key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DSpark paged attention must run eagerly before graph capture")
        compiled = _compile_dspark_attention(block, arch_str, page_size)
        _dspark_attention_kernel_cache[cache_key] = compiled
    output = torch.empty_like(q)
    compiled(
        batch,
        kv_pages.shape[0],
        kv_pages.stride(0),
        block_tables.shape[1],
        block_tables.stride(0),
        q.data_ptr(),
        kv_pages.data_ptr(),
        draft_block.data_ptr(),
        block_tables.data_ptr(),
        positions.data_ptr(),
        valid_lengths.data_ptr(),
        capacities.data_ptr(),
        attn_sink.data_ptr(),
        inverse_rope_freqs.data_ptr(),
        output,
        softmax_scale,
    )
    return output


@torch.library.register_fake("trtllm::fused_dsv4_dspark_attention")
def _(
    q: torch.Tensor,
    draft_block: torch.Tensor,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    valid_lengths: torch.Tensor,
    capacities: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)
