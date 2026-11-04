# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA tensor layout validation, scale views, and workspace allocation."""

from typing import Any

import torch

from tensorrt_llm.logger import logger

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


# Decode scratch shared by every metadata of the process.
#
# Every CUDA-graph bucket owns its own metadata, hence its own Fp4MlaState and ``workspaces`` dict, and the decode
# scratch (packed split-KV partials, the 4x plan and partials, reduce buffers) used to be allocated per state at
# that bucket's warm-up: ~6 GiB over 26 buckets for the packed split partials alone, four times that once the decode
# graphs are keyed by KV regime, all taken out of the KV cache. The graphs of one rank never run concurrently (one
# metadata drives a whole step on the current stream) and every workspace is per-call scratch or a per-step plan
# whose key is cleared at every prepare, so one set per device serves all of them. The storage comes from the
# process-wide ``Buffers`` pool (``memory_buffer_utils.get_memory_buffers``) as reserved blocks: a reserved block is
# never freed, so a pointer baked into a captured graph stays valid even when a later, larger request adds a block.
# Callers pass ``capacity_numel``, the largest element count any bucket or the eager path can ask for, so the first
# request allocates the one block everybody reuses. Only caches validated by per-state tags (the Triton V-pack
# buffers) stay per state (``shared=False``).
_WORKSPACE_POOL_PREFIX = "fp4_mla_ws:"


def _workspace_pool():
    from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

    return get_memory_buffers()


def _workspace_pool_key(attr_name: str, device: torch.device) -> str:
    if device.type != "cuda":
        return f"{_WORKSPACE_POOL_PREFIX}{attr_name}:{device.type}"
    index = device.index if device.index is not None else torch.cuda.current_device()
    return f"{_WORKSPACE_POOL_PREFIX}{attr_name}:cuda{index}"


def _shared_workspace_flat(
    attr_name: str,
    numel: int,
    capacity_numel: int | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Flat view, at least ``numel`` elements long, of the process-wide block for ``attr_name``: the smallest block
    that fits, or a new block of ``max(numel, capacity_numel)`` elements when none fits or every block is below the
    capacity bound (a bounded caller upgrades a request-sized block once). Never allocates while capturing."""
    pool = _workspace_pool()
    key = _workspace_pool_key(attr_name, device)
    itemsize = dtype.itemsize
    blocks = pool.buffers.get(key, [])
    largest = max((block.buffer.numel() for block in blocks), default=0)
    want = max(numel, int(capacity_numel or 0))
    capturing = torch.cuda.is_current_stream_capturing()
    if largest >= numel * itemsize and (largest >= want * itemsize or capturing):
        return pool.get_buffer([numel], dtype, key, True)
    if capturing:
        raise ValueError(
            f"Cannot allocate {attr_name} while capturing a CUDA graph. Run a warmup prepare/forward first."
        )
    if blocks:
        logger.warning(
            f"FP4 MLA decode scratch {attr_name}: {largest} B held, {want * itemsize} B bound or requested now; "
            "allocating the larger block (the old one stays reserved for the graphs that captured it)."
        )
    # the pool allocates on the current device; blocks may be created on a side stream or inside a graph's memory
    # pool, which is harmless for storage that is never freed
    if device.type == "cuda":
        with torch.cuda.device(device):
            return pool.get_buffer([want], dtype, key, True)
    return pool.get_buffer([want], dtype, key, True)


def _workspace_capacity_numel(
    metadata: Any, attr_name: str, dtype: torch.dtype, device: torch.device
) -> int:
    """Elements ``_ensure_workspace_tensor`` can hand out for ``attr_name`` without allocating: the largest
    process-wide block of that name, or the per-state entry of an unshared workspace. Capture-time probes use it."""
    tensor = metadata.fp4_mla_state.workspaces.get(attr_name)
    held = (
        tensor.numel()
        if tensor is not None and tensor.dtype == dtype and tensor.device == device
        else 0
    )
    blocks = _workspace_pool().buffers.get(_workspace_pool_key(attr_name, device), [])
    largest = max((block.buffer.numel() for block in blocks), default=0) // dtype.itemsize
    return max(held, largest)


def _ensure_workspace_tensor(
    metadata: Any,
    attr_name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    capacity_numel: int | None = None,
    shared: bool = True,
) -> torch.Tensor:
    # Grow-only scratch keyed by element count, handed out as a contiguous view
    # of the stored buffer. Callers that share one workspace across geometries
    # (the q=4 target decode and the q=1 MTP-draft decode of one step both use
    # the packed split-KV partials, with 512 vs 128 rows per task) therefore
    # always get contiguous FP32 [tasks, rows, 512]; slicing the stored shape
    # dimension-wise instead returned a strided view the kernel host rejects
    # ("split-KV buffer must be contiguous ..."), which killed CUDA graph
    # capture of small MTP buckets.
    # ``shared`` workspaces (the default) are views of the process-wide block
    # described above, sized by ``capacity_numel`` on its first allocation;
    # the per-state entry only caches the view of the latest request (the
    # block itself is grow-only, see _workspace_capacity_numel).
    numel = 1
    for dim in shape:
        numel *= int(dim)
    workspaces = metadata.fp4_mla_state.workspaces
    tensor = workspaces.get(attr_name)
    needs_alloc = (
        tensor is None
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.numel() < numel
    )
    if needs_alloc:
        if shared:
            flat = _shared_workspace_flat(attr_name, numel, capacity_numel, dtype, device)
            tensor = flat.reshape(-1)[:numel].view(shape)
        else:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    f"Cannot allocate {attr_name} while capturing a CUDA graph. "
                    "Run a warmup prepare/forward first."
                )
            tensor = torch.empty(shape, dtype=dtype, device=device)
        workspaces[attr_name] = tensor

    if tuple(tensor.shape) == tuple(shape):
        return tensor
    return tensor.reshape(-1)[:numel].view(shape)


def _max_sequences(metadata: Any) -> int:
    """Engine-wide sequence capacity of ``metadata`` (0 when unknown): the batch bound of the shared scratch."""
    value = getattr(metadata, "max_num_sequences", None) or getattr(
        metadata, "max_num_requests", None
    )
    return int(value or 0)
