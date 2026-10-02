# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA tensor layout validation, scale views, and workspace allocation."""

from typing import Any

import torch

from .config import (
    _FP4_MLA_K_RESIDUAL_BACKENDS,
    FP4_BLOCK_SIZE,
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_SCALE_COL_GROUP,
    FP4_MLA_SCALE_ROW_GROUP,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    _ceil_div,
)

# FP4 MLA scale-layout helpers


def get_fp4_mla_v_scale_pool_size(v_head_dim: int, page_size: int) -> int:
    """Return elements per page for the swizzled FP4 MLA V-scale pool.

    The PV matmul treats V as a RHS matrix shaped ``[v_head_dim, kv_tokens]``.
    NVFP4 block scales therefore group along the token/K axis, not along the
    latent dimension as the K-view cache does.  The physical layout matches the
    Triton block-scaled matmul scale layout:
    ``[ceil(v_head_dim / 128), ceil(page_size / 16 / 4), 32, 16]``.
    """
    return _get_fp4_mla_swizzled_scale_size(v_head_dim, page_size)


def _get_fp4_mla_swizzled_scale_size(rows: int, cols: int) -> int:
    scale_cols = _ceil_div(cols, FP4_BLOCK_SIZE)
    row_groups = _ceil_div(rows, FP4_MLA_SCALE_ROW_GROUP)
    col_groups = _ceil_div(scale_cols, FP4_MLA_SCALE_COL_GROUP)
    return row_groups * col_groups * 32 * 16


def get_fp4_mla_v_scale_pool_shape(
    num_layers: int,
    num_pages: int,
    v_head_dim: int,
    page_size: int,
) -> tuple[int, int, int, int, int, int]:
    """Return the logical swizzled V-scale view shape.

    The leading dimensions are ``[layer, physical_page]``.  The remaining
    dimensions are the preshuffled ``[N // 128, K // 16 // 4, 32, 16]`` shape
    consumed by Triton block-scaled matmul for the V/PV RHS operand.
    """
    token_scale_cols = _ceil_div(page_size, FP4_BLOCK_SIZE)
    return (
        num_layers,
        num_pages,
        _ceil_div(v_head_dim, FP4_MLA_SCALE_ROW_GROUP),
        _ceil_div(token_scale_cols, FP4_MLA_SCALE_COL_GROUP),
        32,
        16,
    )


def get_fp4_mla_v_scale_pool_view(
    metadata: Any,
    *,
    v_head_dim: int,
) -> torch.Tensor:
    """View the auxiliary MLA V-scale pool in Triton's block-scaled layout."""
    pool = getattr(metadata.fp4_mla_state, "v_scale_pool", None)
    if pool is None:
        raise RuntimeError("FP4 MLA V scale pool is not allocated.")

    elems_per_page = get_fp4_mla_v_scale_pool_size(v_head_dim, metadata.page_size)
    if pool.shape[-1] < elems_per_page:
        raise RuntimeError(
            f"FP4 MLA V scale pool page stride is too small: got "
            f"{pool.shape[-1]}, need {elems_per_page}."
        )

    token_scale_cols = _ceil_div(metadata.page_size, FP4_BLOCK_SIZE)
    col_groups = _ceil_div(token_scale_cols, FP4_MLA_SCALE_COL_GROUP)
    shape = get_fp4_mla_v_scale_pool_shape(
        pool.shape[0], pool.shape[1], v_head_dim, metadata.page_size
    )
    strides = (
        pool.stride(0),
        pool.stride(1),
        col_groups * 32 * 16,
        32 * 16,
        16,
        1,
    )
    return torch.as_strided(pool, size=shape, stride=strides)


# Python launch helpers


def _get_fp4_mla_global_scale(metadata: Any, device: torch.device) -> torch.Tensor:
    global_scale = getattr(metadata.fp4_mla_state, "kv_global_scale", None)
    if (
        not isinstance(global_scale, torch.Tensor)
        or global_scale.device != device
        or global_scale.dtype != torch.float32
        or global_scale.numel() != 1
    ):
        raise RuntimeError("FP4 MLA requires a preallocated FP32 KV global-scale tensor.")
    return global_scale


def _get_fp4_mla_q_global_scale(metadata: Any, device: torch.device) -> torch.Tensor:
    global_scale = getattr(metadata.fp4_mla_state, "q_global_scale", None)
    if (
        not isinstance(global_scale, torch.Tensor)
        or global_scale.device != device
        or global_scale.dtype != torch.float32
        or global_scale.numel() != 1
    ):
        raise RuntimeError("FP4 MLA requires a preallocated FP32 Q global-scale tensor.")
    return global_scale


def _get_fp4_mla_kv_cache_tensors(
    metadata: Any, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return metadata.kv_cache_manager.get_fp4_mla_cache_buffers(layer_idx)


def _get_fp4_mla_hp_pool_layout(
    metadata: Any,
    pool: torch.Tensor,
) -> tuple[int, int]:
    """Return the manager-owned HP ring size and per-token head dimension."""
    manager = getattr(metadata, "kv_cache_manager", None)
    if manager is None or not hasattr(manager, "fp4_mla_hp_pool_size"):
        raise ValueError("FP4 MLA requires a V2 manager-owned HP ring.")
    hp_pool_size = manager.fp4_mla_hp_pool_size
    if (
        hp_pool_size < HP_BLOCK_SIZE
        or pool.ndim != 4
        or pool.shape[2] < 1
        or pool.shape[-1] % hp_pool_size != 0
    ):
        raise ValueError(
            "FP4 MLA high-precision KV pool does not match its configured "
            f"ring: shape={tuple(pool.shape)}, ring_size={hp_pool_size}."
        )
    return hp_pool_size, pool.shape[-1] // hp_pool_size


def _validate_fp4_mla_hp_generation_width(
    hp_pool_size: int,
    generation_len: int,
) -> None:
    """Ensure one target plus rewindable drafts fit without clobbering the live tail."""
    max_rewind_len = hp_pool_size - HP_BLOCK_SIZE
    if generation_len <= 0 or generation_len - 1 > max_rewind_len:
        raise RuntimeError(
            "FP4 MLA generation exceeds the HP ring's rewind slack: "
            f"generation={generation_len}, max_rewind={max_rewind_len}."
        )


def _validate_fp4_mla_kv_storage_shape(
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    *,
    head_dim: int,
    backend: str,
) -> int:
    """Validate the backend-specific physical KV and scale strides."""
    residual_dim = FP4_MLA_K_RESIDUAL_DIM if backend in _FP4_MLA_K_RESIDUAL_BACKENDS else 0
    expected_storage_head_dim = head_dim + residual_dim
    storage_head_dim = kv_cache.shape[-1] * 2
    if storage_head_dim != expected_storage_head_dim:
        raise RuntimeError(
            "FP4 MLA KV cache storage head dimension does not match the selected backend: "
            f"got {storage_head_dim}, expected {expected_storage_head_dim}. Recreate the engine "
            f"after setting {FP4_MLA_ATTENTION_BACKEND_ENV}."
        )

    expected_scale_columns = expected_storage_head_dim // FP4_BLOCK_SIZE
    if sf_cache.shape[-1] != expected_scale_columns:
        raise RuntimeError(
            "FP4 MLA KV cache scale storage does not match the contiguous data layout: "
            f"got {sf_cache.shape[-1]} columns, expected {expected_scale_columns}."
        )
    return storage_head_dim


def _validate_fp4_mla_cache_shape(page_size: int, head_dim: int) -> None:
    if page_size != FP4_MLA_TOKENS_PER_BLOCK:
        raise ValueError(
            f"FP4 MLA KV cache requires tokens_per_block={FP4_MLA_TOKENS_PER_BLOCK} "
            f"for swizzled block scales, got {page_size}."
        )

    sf_per_token = head_dim // FP4_BLOCK_SIZE
    if head_dim % FP4_BLOCK_SIZE != 0 or sf_per_token % 4 != 0:
        raise ValueError(
            f"FP4 MLA KV head_dim must produce a scale column count divisible by 4; "
            f"got head_dim={head_dim}, scale_columns={sf_per_token}."
        )


def _validate_fp4_mla_attention_q_shape(head_dim: int, q_residual_dim: int) -> None:
    if q_residual_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA Q residual_dim must be divisible by {FP4_BLOCK_SIZE}, got {q_residual_dim}."
        )
    if q_residual_dim <= 0 or q_residual_dim > head_dim:
        raise ValueError(
            f"FP4 MLA Q residual_dim must be in (0, head_dim], got "
            f"residual_dim={q_residual_dim}, head_dim={head_dim}."
        )

    q_head_dim = head_dim + q_residual_dim
    q_sf_per_token = q_head_dim // FP4_BLOCK_SIZE
    if q_head_dim % FP4_BLOCK_SIZE != 0 or q_sf_per_token % FP4_MLA_SCALE_COL_GROUP != 0:
        raise ValueError(
            f"FP4 MLA residual Q must produce a scale column count divisible "
            f"by {FP4_MLA_SCALE_COL_GROUP}; got q_head_dim={q_head_dim}, "
            f"scale_columns={q_sf_per_token}."
        )


def _ensure_workspace_tensor(
    metadata: Any,
    attr_name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    workspaces = metadata.fp4_mla_state.workspaces
    tensor = workspaces.get(attr_name)
    needs_alloc = (
        tensor is None
        or tensor.dtype != dtype
        or tensor.device != device
        or len(tensor.shape) != len(shape)
        or any(tensor.shape[idx] < dim for idx, dim in enumerate(shape))
    )
    if needs_alloc:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError(
                f"Cannot allocate {attr_name} while capturing a CUDA graph. "
                "Run a warmup prepare/forward first."
            )
        tensor = torch.empty(shape, dtype=dtype, device=device)
        workspaces[attr_name] = tensor

    slices = tuple(slice(0, dim) for dim in shape)
    return tensor[slices]
