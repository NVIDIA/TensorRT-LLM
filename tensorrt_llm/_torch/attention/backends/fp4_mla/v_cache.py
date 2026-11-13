# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA CuTeDSL packed-V views and imported cache sidecar rebuilding."""

from typing import Any, Optional, Sequence

import torch
import triton

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings import DataType

from .config import (
    _FP4_MLA_CUTEDSL_BACKEND,
    FP4_BLOCK_SIZE,
    FP4_MLA_SCALE_ROW_GROUP,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    _ceil_div,
    _env_int,
    _fp4_mla_attention_backend,
    _fp4_mla_cutedsl_fused_v_transpose_enabled,
)
from .fp4_mla_kernels import _fp4_mla_rebuild_v_scale_from_k_scale_kernel
from .layout import get_fp4_mla_v_scale_pool_size


def _select_triton_block_v(num_queries: int) -> int:
    env_block_v = _env_int("TRTLLM_FP4_MLA_BLOCK_V")
    if env_block_v is not None:
        return env_block_v
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


def _rebuild_fp4_mla_v_scales_from_k_scales(
    sf_cache: torch.Tensor,
    v_scale_pool: torch.Tensor,
    page_ids: torch.Tensor,
    page_valid_tokens: torch.Tensor,
    *,
    local_layer: int,
    v_head_dim: int,
    page_size: int,
) -> None:
    """Bit-exactly rebuild imported MLA V scales from transferred K scales."""
    if page_ids.numel() == 0:
        return
    if page_size != FP4_MLA_TOKENS_PER_BLOCK:
        raise ValueError(
            "FP4 MLA imported V-scale rebuild requires "
            f"tokens_per_block={FP4_MLA_TOKENS_PER_BLOCK}, got {page_size}."
        )
    for name, tensor in (
        ("sf_cache", sf_cache),
        ("v_scale_pool", v_scale_pool),
        ("page_ids", page_ids),
        ("page_valid_tokens", page_valid_tokens),
    ):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor.")
    if page_ids.dtype != torch.int32 or page_valid_tokens.dtype != torch.int32:
        raise TypeError("FP4 MLA imported page IDs and valid-token counts must use int32.")
    if page_ids.ndim != 1 or page_valid_tokens.ndim != 1:
        raise ValueError("FP4 MLA imported page metadata must be one-dimensional.")
    if page_ids.numel() != page_valid_tokens.numel():
        raise ValueError("FP4 MLA imported page IDs and valid-token counts must have equal length.")
    if not page_ids.is_contiguous() or not page_valid_tokens.is_contiguous():
        raise ValueError("FP4 MLA imported page metadata must be contiguous.")
    if not (sf_cache.device == v_scale_pool.device == page_ids.device == page_valid_tokens.device):
        raise ValueError("FP4 MLA imported cache tensors must be on the same device.")

    k_sf_bytes = sf_cache.view(torch.uint8)
    v_sf_bytes = v_scale_pool.view(torch.uint8)
    if k_sf_bytes.ndim < 2 or v_sf_bytes.ndim < 3:
        raise ValueError(
            "FP4 MLA imported scale pools require per-page K storage and "
            "per-layer/per-page V storage."
        )
    num_layers = int(v_sf_bytes.shape[0])
    num_pages = int(v_sf_bytes.shape[1])
    if not 0 <= local_layer < num_layers:
        raise IndexError(
            f"local_layer={local_layer} is outside the V-scale pool with {num_layers} layers."
        )
    if int(k_sf_bytes.shape[0]) != num_pages:
        raise ValueError(
            "FP4 MLA K/V scale pools disagree on their physical page count: "
            f"{int(k_sf_bytes.shape[0])} != {num_pages}."
        )
    sf_per_token = int(k_sf_bytes.shape[-1])
    required_sf_per_token = _ceil_div(v_head_dim, FP4_BLOCK_SIZE)
    if sf_per_token < required_sf_per_token:
        raise ValueError(
            "FP4 MLA K-scale storage is too narrow for the compressed V head: "
            f"{sf_per_token} < {required_sf_per_token}."
        )
    required_v_page_elems = get_fp4_mla_v_scale_pool_size(v_head_dim, page_size)
    if int(v_sf_bytes.shape[-1]) < required_v_page_elems:
        raise ValueError(
            "FP4 MLA V-scale page storage is too small for import rebuild: "
            f"{int(v_sf_bytes.shape[-1])} < {required_v_page_elems}."
        )

    token_groups = page_size // HP_BLOCK_SIZE
    _fp4_mla_rebuild_v_scale_from_k_scale_kernel[
        (page_ids.numel(), triton.cdiv(v_head_dim, FP4_BLOCK_SIZE))
    ](
        k_sf_bytes,
        v_sf_bytes,
        page_ids,
        page_valid_tokens,
        page_ids.numel(),
        num_pages,
        num_layers,
        local_layer,
        page_size,
        k_sf_bytes.stride(0),
        v_sf_bytes.stride(0),
        v_sf_bytes.stride(1),
        V_HEAD_D=v_head_dim,
        HP_BLOCK=HP_BLOCK_SIZE,
        SF_PER_TOKEN=sf_per_token,
        SF_PER_PAGE=token_groups,
        BLOCK_TOKEN_GROUPS=triton.next_power_of_2(token_groups),
        num_warps=4,
    )


def _stage_fp4_mla_import_page_metadata(
    prompt_block_ids: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    layer_page_offsets: Sequence[int],
    *,
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage the imported page worklist of every FP4 layer without synchronizing.

    Both tensors are row-major over ``(layer, prompt page)``. A page ID is the
    prompt page plus the layer's page offset, which addresses that layer's
    copy in the pool-base views; a valid-token count is the number of prompt
    tokens the page holds.
    """
    if device.type != "cuda":
        raise ValueError("FP4 MLA disaggregated import requires a CUDA device.")
    prompt_pages = []
    prompt_page_valid_tokens = []
    for block_ids, prompt_len in zip(prompt_block_ids, prompt_lens, strict=True):
        prompt_pages.extend(block_ids)
        prompt_page_valid_tokens.extend([page_size] * (len(block_ids) - 1))
        prompt_page_valid_tokens.append(prompt_len - (len(block_ids) - 1) * page_size)
    shape = (len(layer_page_offsets), len(prompt_pages))
    pin_memory = prefer_pinned()
    page_ids_host = torch.empty(shape, dtype=torch.int32, device="cpu", pin_memory=pin_memory)
    torch.add(
        torch.tensor(layer_page_offsets, dtype=torch.int32)[:, None],
        torch.tensor(prompt_pages, dtype=torch.int32)[None, :],
        out=page_ids_host,
    )
    page_valid_tokens_host = torch.empty(
        shape, dtype=torch.int32, device="cpu", pin_memory=pin_memory
    )
    page_valid_tokens_host.copy_(torch.tensor(prompt_page_valid_tokens, dtype=torch.int32))
    # Constructing a CUDA tensor directly from a Python list synchronizes the
    # current stream. During overlap scheduling that stream already waits for
    # the previous forward, exposing the import rebuild as an inter-step gap.
    # Explicit non-blocking copies keep the CPU free to enqueue the rebuild
    # and the next forward while preserving their existing stream order.
    page_ids = torch.empty_like(page_ids_host, device=device)
    page_valid_tokens = torch.empty_like(page_valid_tokens_host, device=device)
    page_ids.copy_(page_ids_host, non_blocking=True)
    page_valid_tokens.copy_(page_valid_tokens_host, non_blocking=True)
    return page_ids.view(-1), page_valid_tokens.view(-1)


def rebuild_fp4_mla_disagg_imported_cache(
    kv_cache_manager: Any,
    request_ids: Sequence[int],
    prompt_lens: Sequence[int],
) -> bool:
    """Rebuild GEN-local FP4 MLA sidecars after a disaggregated KV import.

    The disaggregated payload carries the native V2 K, K-scale, and BF16 HP
    roles. V scales and CuTeDSL's V-packed layout are deterministic
    process-local views, so rebuilding them here avoids transfer bandwidth and
    guarantees they are ready before a first decode step that may execute
    through a pre-captured CUDA graph.

    Every FP4 layer shares one coalesced page geometry, so one launch per
    sidecar rebuilds all imported requests and layers through the pool-base
    views; the import adds no per-layer host work to the generation step.
    """
    if (
        kv_cache_manager is None
        or getattr(kv_cache_manager, "dtype", None) != DataType.NVFP4
        or getattr(kv_cache_manager, "kv_factor", None) != 1
        or getattr(kv_cache_manager, "mla_v_scale_head_dim", None) is None
        or not callable(getattr(kv_cache_manager, "get_fp4_mla_page_table_spec", None))
    ):
        return False
    if len(request_ids) != len(prompt_lens):
        raise ValueError(
            "FP4 MLA disaggregated import needs one prompt_len per request, got "
            f"{len(prompt_lens)} for {len(request_ids)} requests."
        )
    for prompt_len in prompt_lens:
        if not isinstance(prompt_len, int) or prompt_len < 0:
            raise ValueError(
                f"FP4 MLA disaggregated import needs a nonnegative prompt_len, got {prompt_len}."
            )
    # Helix assigns whole pages round-robin, so a rank may own no prompt pages.
    # There are no process-local V sidecars to rebuild for such a request.
    imports = [
        (int(request_id), prompt_len)
        for request_id, prompt_len in zip(request_ids, prompt_lens)
        if prompt_len > 0
    ]
    if not imports:
        return True

    page_size = int(kv_cache_manager.tokens_per_block)
    if page_size != FP4_MLA_TOKENS_PER_BLOCK:
        raise ValueError(
            "FP4 MLA disaggregated import requires "
            f"tokens_per_block={FP4_MLA_TOKENS_PER_BLOCK}, got {page_size}."
        )
    pp_layers = list(getattr(kv_cache_manager, "pp_layers", ()))
    num_local_layers = int(getattr(kv_cache_manager, "num_local_layers", len(pp_layers)))
    if len(pp_layers) != num_local_layers:
        raise RuntimeError(
            "FP4 MLA disaggregated import cannot map local to global layers: "
            f"{len(pp_layers)} PP layers for {num_local_layers} local layers."
        )
    fp4_local_layers = list(
        getattr(kv_cache_manager, "_fp4_mla_compact_to_local", range(num_local_layers))
    )
    if not fp4_local_layers:
        raise RuntimeError("FP4 MLA disaggregated import found no local MLA layers.")
    if any(local_layer < 0 or local_layer >= num_local_layers for local_layer in fp4_local_layers):
        raise RuntimeError(
            "FP4 MLA disaggregated import has invalid compact-to-local layer mapping: "
            f"{fp4_local_layers}."
        )

    first_attention_layer = pp_layers[fp4_local_layers[0]]
    block_ids_per_seq = kv_cache_manager.get_batch_cache_indices(
        [request_id for request_id, _ in imports], layer_idx=first_attention_layer
    )
    if len(block_ids_per_seq) != len(imports):
        raise RuntimeError(
            "FP4 MLA disaggregated import got page lists for "
            f"{len(block_ids_per_seq)} of {len(imports)} requests."
        )
    prompt_block_ids = []
    for (request_id, prompt_len), block_ids in zip(imports, block_ids_per_seq):
        num_prompt_pages = _ceil_div(prompt_len, page_size)
        if len(block_ids) < num_prompt_pages:
            raise RuntimeError(
                "FP4 MLA disaggregated import is missing prompt pages for request "
                f"{request_id}: need {num_prompt_pages}, have {len(block_ids)}."
            )
        prompt_block_ids.append([int(block_id) for block_id in block_ids[:num_prompt_pages]])

    kv_cache, sf_cache = kv_cache_manager.get_fp4_mla_cache_pool_bases()
    v_scale_pool = kv_cache_manager.get_mla_v_scale_pool_base()
    if not isinstance(v_scale_pool, torch.Tensor):
        raise RuntimeError("FP4 MLA disaggregated import requires the manager V-scale pool.")
    page_ids, page_valid_tokens = _stage_fp4_mla_import_page_metadata(
        prompt_block_ids,
        [prompt_len for _, prompt_len in imports],
        [
            int(kv_cache_manager.get_mla_v_scale_page_offset(compact_layer))
            for compact_layer in range(len(fp4_local_layers))
        ],
        page_size=page_size,
        device=v_scale_pool.device,
    )

    # The worklist already carries each layer's page offset, so the pool-base
    # V-scale view is addressed as a single layer.
    _rebuild_fp4_mla_v_scales_from_k_scales(
        sf_cache,
        v_scale_pool.unsqueeze(0),
        page_ids,
        page_valid_tokens,
        local_layer=0,
        v_head_dim=int(kv_cache_manager.mla_v_scale_head_dim),
        page_size=page_size,
    )
    if (
        _fp4_mla_attention_backend() == _FP4_MLA_CUTEDSL_BACKEND
        and not _fp4_mla_cutedsl_fused_v_transpose_enabled()
    ):
        v_head_dim = getattr(kv_cache_manager, "mla_v_head_dim", None)
        if v_head_dim is None:
            raise RuntimeError(
                "CuTeDSL FP4 MLA disaggregated import requires a persistent V head dimension."
            )
        v_packed = kv_cache_manager.get_mla_v_packed_pool_base()
        if not isinstance(v_packed, torch.Tensor):
            raise RuntimeError(
                "CuTeDSL FP4 MLA disaggregated import requires the persistent V-packed pool."
            )
        _repack_cutedsl_v_packed_cache(
            v_packed,
            kv_cache,
            page_ids,
            v_head_dim=int(v_head_dim),
            page_size=page_size,
            block_v=FP4_MLA_SCALE_ROW_GROUP,
        )
    return True
