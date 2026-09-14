# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA packed-V views, repacking, and cache validity tracking."""

import os
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from .config import (
    FP4_MLA_TOKENS_PER_BLOCK,
    _ceil_div,
    _env_enabled_default,
    _env_int,
    _fp4_mla_attention_backend,
)
from .layout import _ensure_workspace_tensor


def _shared_v_pack_storage_enabled() -> bool:
    return os.getenv("TRTLLM_FP4_MLA_SHARE_V_PACK_STORAGE", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _select_triton_block_v(num_queries: int, *, prefer_prepacked_v: bool = False) -> int:
    env_block_v = _env_int("TRTLLM_FP4_MLA_BLOCK_V")
    if env_block_v is not None:
        return env_block_v
    if prefer_prepacked_v:
        return 128
    return 32 if num_queries <= 32 else 128


def _v_packed_shape(
    kv_cache: torch.Tensor,
    v_head_dim: int,
    page_size: int,
    block_v: int,
) -> tuple[int, int]:
    return (kv_cache.shape[0] * _ceil_div(v_head_dim, block_v) * block_v, page_size // 2)


def _get_fp4_mla_v_packed_pool(metadata: Any, local_layer: int) -> Optional[torch.Tensor]:
    return metadata.kv_cache_manager.get_mla_v_packed_pool(local_layer)


def _get_fp4_mla_v_packed_pool_base(metadata: Any) -> Optional[torch.Tensor]:
    return metadata.kv_cache_manager.get_mla_v_packed_pool_base()


def _get_fp4_mla_v_scale_pool_base(metadata: Any) -> Optional[torch.Tensor]:
    return metadata.kv_cache_manager.get_mla_v_scale_pool_base()


def _get_cutedsl_persistent_v_packed_cache(
    metadata: Any,
    local_layer: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int,
) -> torch.Tensor:
    v_packed = _get_fp4_mla_v_packed_pool(metadata, local_layer)
    if v_packed is None:
        raise RuntimeError(
            "CuTeDSL FP4 MLA requires the manager-owned persistent V-packed "
            "pool; the scratch full-repack fallback has been removed."
        )
    expected_shape = _v_packed_shape(kv_cache, v_head_dim, page_size, block_v)
    if (
        not isinstance(v_packed, torch.Tensor)
        or v_packed.dtype != torch.uint8
        or v_packed.device != kv_cache.device
        or tuple(v_packed.shape) != expected_shape
        or not v_packed.is_contiguous()
    ):
        raise RuntimeError(
            "FP4 MLA persistent V-packed pool must be a contiguous uint8 tensor "
            f"with shape {expected_shape} on {kv_cache.device}; got "
            f"{type(v_packed).__name__}, "
            f"shape={getattr(v_packed, 'shape', None)}, "
            f"dtype={getattr(v_packed, 'dtype', None)}, "
            f"device={getattr(v_packed, 'device', None)}."
        )
    return v_packed


def _repack_cutedsl_v_packed_cache(
    v_packed: torch.Tensor,
    kv_cache: torch.Tensor,
    page_ids: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int,
    page_indptr: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
    generation_lens: Optional[torch.Tensor] = None,
    max_touched_pages: int = 1,
) -> None:
    if page_ids.numel() == 0:
        return
    from .fp4_mla_cutedsl_v_repack import fp4_mla_repack_v_cache

    fp4_mla_repack_v_cache(
        v_packed,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        page_indptr=page_indptr,
        kv_lens=kv_lens,
        generation_lens=generation_lens,
        max_touched_pages=max_touched_pages,
    )


def _v_packed_cache_tag(
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    block_v: int = 128,
) -> tuple[Any, ...]:
    v_sf_tag = (
        None
        if v_sf is None
        else (
            int(v_sf.data_ptr()),
            str(v_sf.device),
            str(v_sf.dtype),
            tuple(int(dim) for dim in v_sf.shape),
            tuple(int(stride) for stride in v_sf.stride()),
        )
    )
    page_ids_tag = (
        None
        if page_ids is None
        else (
            int(page_ids.data_ptr()),
            str(page_ids.device),
            str(page_ids.dtype),
            tuple(int(dim) for dim in page_ids.shape),
            tuple(int(stride) for stride in page_ids.stride()),
        )
    )
    return (
        int(layer_idx),
        None if local_layer is None else int(local_layer),
        int(kv_cache.data_ptr()),
        str(kv_cache.device),
        str(kv_cache.dtype),
        tuple(int(dim) for dim in kv_cache.shape),
        tuple(int(stride) for stride in kv_cache.stride()),
        int(v_head_dim),
        int(page_size),
        int(block_v),
        v_sf_tag,
        page_ids_tag,
    )


def _triton_prepack_v_enabled() -> bool:
    if _fp4_mla_attention_backend() != "triton":
        return False
    default = _env_enabled_default("TRTLLM_FP4_MLA_PREPACK_V", True)
    return _env_enabled_default("TRTLLM_FP4_MLA_TRITON_PREPACK_V", default)


def _triton_can_prepack_v(v_head_dim: int, page_size: int, block_v: int) -> bool:
    return (
        _triton_prepack_v_enabled()
        and hasattr(tl, "make_tensor_descriptor")
        and block_v in (32, 128)
        and v_head_dim % block_v == 0
        and page_size == FP4_MLA_TOKENS_PER_BLOCK
    )


def _triton_v_packed_attr(layer_idx: int) -> str:
    if _shared_v_pack_storage_enabled():
        return "_fp4_mla_triton_attention_v_packed_buf"
    return f"_fp4_mla_triton_attention_v_packed_buf_l{layer_idx}"


def _triton_v_packed_valid_attr(layer_idx: int) -> str:
    return f"_fp4_mla_triton_attention_v_packed_valid_l{layer_idx}"


def _triton_shared_v_packed_valid_attr() -> str:
    return "_fp4_mla_triton_attention_v_packed_valid_tag"


def _triton_v_packed_cache_tag(
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    block_v: int = 128,
) -> tuple[Any, ...]:
    return (
        "triton",
        _v_packed_cache_tag(
            layer_idx,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=page_size,
            block_v=block_v,
            local_layer=local_layer,
            v_sf=v_sf,
            page_ids=page_ids,
        ),
    )


def _set_triton_v_packed_cache_valid(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    block_v: int = 128,
) -> None:
    valid_attr = (
        _triton_shared_v_packed_valid_attr()
        if _shared_v_pack_storage_enabled()
        else _triton_v_packed_valid_attr(layer_idx)
    )
    metadata.fp4_mla_state.v_packed_cache_tags[valid_attr] = _triton_v_packed_cache_tag(
        layer_idx,
        kv_cache,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        page_ids=page_ids,
    )


def _is_triton_v_packed_cache_valid(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    block_v: int = 128,
) -> bool:
    valid_attr = (
        _triton_shared_v_packed_valid_attr()
        if _shared_v_pack_storage_enabled()
        else _triton_v_packed_valid_attr(layer_idx)
    )
    return metadata.fp4_mla_state.v_packed_cache_tags.get(valid_attr) == _triton_v_packed_cache_tag(
        layer_idx,
        kv_cache,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        page_ids=page_ids,
    )


def _get_triton_v_packed_cache(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    block_v: int = 128,
) -> Optional[torch.Tensor]:
    if not _triton_can_prepack_v(v_head_dim, page_size, block_v):
        return None
    if not _is_triton_v_packed_cache_valid(
        metadata,
        layer_idx,
        kv_cache,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        page_ids=page_ids,
    ):
        return None
    v_packed = metadata.fp4_mla_state.workspaces.get(_triton_v_packed_attr(layer_idx))
    expected_shape = _v_packed_shape(kv_cache, v_head_dim, page_size, block_v)
    if (
        v_packed is None
        or v_packed.dtype != torch.uint8
        or v_packed.device != kv_cache.device
        or len(v_packed.shape) != 2
        or v_packed.shape[0] < expected_shape[0]
        or v_packed.shape[1] < expected_shape[1]
    ):
        return None
    return v_packed[: expected_shape[0], : expected_shape[1]]


def _update_triton_v_packed_cache(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    page_ids: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    num_valid_pages: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if not _triton_can_prepack_v(v_head_dim, page_size, block_v):
        return None
    if page_ids.numel() == 0:
        return None
    from .fp4_mla_triton import fp4_mla_repack_v_cache_triton

    def _tma_alloc(size: int, alignment: int, stream):
        return torch.empty(size, device=kv_cache.device, dtype=torch.int8)

    triton.set_allocator(_tma_alloc)
    attr_name = _triton_v_packed_attr(layer_idx)
    v_packed = _ensure_workspace_tensor(
        metadata,
        attr_name,
        _v_packed_shape(kv_cache, v_head_dim, page_size, block_v),
        dtype=torch.uint8,
        device=kv_cache.device,
    )
    fp4_mla_repack_v_cache_triton(
        v_packed,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        num_valid_pages=num_valid_pages,
    )
    _set_triton_v_packed_cache_valid(
        metadata,
        layer_idx,
        kv_cache,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        page_ids=page_ids,
    )
    return v_packed


def _maybe_update_triton_v_packed_cache(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    page_ids: torch.Tensor,
    *,
    num_queries: int,
    v_head_dim: int,
    page_size: int,
    local_layer: Optional[int] = None,
    v_sf: Optional[torch.Tensor] = None,
    num_valid_pages: Optional[torch.Tensor] = None,
) -> None:
    block_v = _select_triton_block_v(num_queries, prefer_prepacked_v=_triton_prepack_v_enabled())
    _update_triton_v_packed_cache(
        metadata,
        layer_idx,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        num_valid_pages=num_valid_pages,
    )
