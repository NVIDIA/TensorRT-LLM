# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather cached FP4 MLA prefix partitions for explicit-KV context attention."""

from typing import Any

import torch
import triton

from .config import FP4_BLOCK_SIZE, FP4_MLA_K_RESIDUAL_DIM, _fp4_mla_attention_backend
from .fp4_mla_kernels import _fp4_mla_chunked_cache_gather_kernel
from .layout import (
    _get_fp4_mla_global_scale,
    _get_fp4_mla_kv_cache_tensors,
    _validate_fp4_mla_kv_storage_shape,
)
from .metadata import _materialize_fp4_mla_device_page_table_for_forward


def load_fp4_mla_chunked_kv_cache(
    metadata: Any,
    layer_idx: int,
    *,
    num_ctx_cached_tokens: int,
    cu_chunked_seq_len: torch.Tensor,
    chunked_global_offset: torch.Tensor,
    chunked_max_seq_len: int,
    out_dtype: torch.dtype,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one cached-prefix partition from dense FP4 MLA V2 storage."""
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"FP4 MLA chunk gather does not support output dtype {out_dtype}.")
    if num_ctx_cached_tokens < 0:
        raise ValueError("FP4 MLA chunk gather token count must be non-negative.")
    if chunked_max_seq_len < 0:
        raise ValueError("FP4 MLA chunk gather max sequence length must be non-negative.")
    if num_ctx_cached_tokens > 0 and chunked_max_seq_len == 0:
        raise ValueError("A non-empty FP4 MLA chunk gather requires a positive max length.")
    if (
        kv_lora_rank <= 0
        or kv_lora_rank % FP4_BLOCK_SIZE != 0
        or qk_rope_head_dim != FP4_MLA_K_RESIDUAL_DIM
    ):
        raise ValueError(
            "FP4 MLA chunk gather requires a positive kv_lora_rank and "
            f"qk_rope_head_dim={FP4_MLA_K_RESIDUAL_DIM}, got "
            f"{kv_lora_rank} and {qk_rope_head_dim}."
        )

    num_contexts = int(metadata.num_contexts)
    if num_ctx_cached_tokens > 0 and num_contexts <= 0:
        raise ValueError("A non-empty FP4 MLA chunk gather requires context requests.")
    tensors = (cu_chunked_seq_len, chunked_global_offset)
    if any(not isinstance(tensor, torch.Tensor) or not tensor.is_cuda for tensor in tensors):
        raise ValueError("FP4 MLA chunk gather metadata must use CUDA tensors.")
    if (
        cu_chunked_seq_len.dtype != torch.int64
        or cu_chunked_seq_len.ndim != 1
        or cu_chunked_seq_len.numel() < num_contexts + 1
        or not cu_chunked_seq_len.is_contiguous()
    ):
        raise ValueError(
            "FP4 MLA chunk gather requires a contiguous int64 cumulative-length tensor."
        )
    if (
        chunked_global_offset.dtype != torch.int64
        or chunked_global_offset.ndim != 1
        or chunked_global_offset.numel() < num_contexts
        or not chunked_global_offset.is_contiguous()
    ):
        raise ValueError("FP4 MLA chunk gather requires contiguous int64 global offsets.")
    if cu_chunked_seq_len.device != chunked_global_offset.device:
        raise ValueError("FP4 MLA chunk gather metadata tensors must share one device.")

    compressed_kv = torch.empty(
        (num_ctx_cached_tokens, kv_lora_rank),
        dtype=out_dtype,
        device=cu_chunked_seq_len.device,
    )
    k_pe = torch.empty(
        (num_ctx_cached_tokens, qk_rope_head_dim),
        dtype=out_dtype,
        device=cu_chunked_seq_len.device,
    )
    if num_ctx_cached_tokens == 0:
        return compressed_kv, k_pe

    if not bool(getattr(metadata.fp4_mla_state, "device_page_table", False)):
        raise RuntimeError("FP4 MLA chunk gather requires fixed-stride device page metadata.")
    _materialize_fp4_mla_device_page_table_for_forward(metadata)
    page_table_stride = int(metadata.fp4_mla_state.page_table_stride)
    page_ids = metadata.fp4_mla_state._paged_kv_indices
    if (
        page_table_stride <= 0
        or not isinstance(page_ids, torch.Tensor)
        or page_ids.dtype != torch.int32
        or not page_ids.is_cuda
        or page_ids.device != cu_chunked_seq_len.device
        or page_ids.numel() < num_contexts * page_table_stride
    ):
        raise RuntimeError("FP4 MLA chunk gather received invalid device page metadata.")

    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    head_dim = kv_lora_rank + qk_rope_head_dim
    storage_head_dim = _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=_fp4_mla_attention_backend(),
    )
    if storage_head_dim != head_dim + FP4_MLA_K_RESIDUAL_DIM:
        raise RuntimeError(
            "FP4 MLA chunk gather requires K residual storage for BF16 reconstruction."
        )
    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    global_scale = _get_fp4_mla_global_scale(metadata, kv_cache.device)
    token_block = 16
    grid = (
        triton.cdiv(chunked_max_seq_len, token_block),
        num_contexts,
        triton.cdiv(head_dim, FP4_BLOCK_SIZE),
    )
    _fp4_mla_chunked_cache_gather_kernel[grid](
        compressed_kv,
        k_pe,
        kv_cache,
        sf_cache,
        page_ids,
        cu_chunked_seq_len,
        chunked_global_offset,
        global_scale,
        chunked_max_seq_len,
        page_table_stride,
        kv_cache.shape[0],
        metadata.page_size,
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        compressed_kv.stride(0),
        k_pe.stride(0),
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_TOKEN=storage_head_dim // FP4_BLOCK_SIZE,
        TOKEN_BLOCK=token_block,
        num_warps=4,
    )
    return compressed_kv, k_pe


__all__ = ["load_fp4_mla_chunked_kv_cache"]
