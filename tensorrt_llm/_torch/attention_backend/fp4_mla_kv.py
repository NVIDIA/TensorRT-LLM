# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MLA FP4 KV-cache helpers.

The high-precision (HP) BF16 KV pool is a standalone circular buffer used
alongside the paged FP4 KV pool when MLA models run with NVFP4 KV cache.
Each sequence slot stores the ``HP_BLOCK_SIZE`` most-recent latent vectors at
BF16, so attention backends can consult BF16 values for the tail tokens that
do not yet fill a complete FP4 quant block of 16 elements along the sequence
dimension.

Used by both ``TrtllmAttention`` (via an internal C++ attention op that reads
both pools) and ``FlashInferAttention`` (via either explicit Python-side
dequant into a BF16 workspace before calling FlashInfer MLA wrappers, or an
env-gated Triton attention path that reads packed FP4 Q, K, and V directly).
"""

import os
from typing import Any, Literal, Optional

import torch
import triton

from .fp4_mla_kernels import (
    _fp4_mla_attention_prob_pack_page_kernel,
    _fp4_mla_attention_prob_store_page_kernel,
    _fp4_mla_attention_pv_kernel,
    _fp4_mla_attention_stats_kernel,
    _fp4_mla_dequant_kernel,
    _fp4_mla_overlay_hp_tail_kernel,
    _fp4_mla_scatter_kernel,
    _fp4_mla_v_scale_store_context_tokens_kernel,
    _fp4_mla_v_scale_store_hp_tail_kernel,
    _hp_kv_store_context_kernel,
    _hp_kv_store_gen_kernel,
)

HP_BLOCK_SIZE: int = 16
FP4_BLOCK_SIZE: int = 16
FP4_MLA_TOKENS_PER_BLOCK: int = 128
FP4_MLA_SCALE_ROW_GROUP: int = 128
FP4_MLA_SCALE_COL_GROUP: int = 4
FP4_MLA_KV_GLOBAL_SCALE: float = 448.0 * 6.0 / 448.0 * 6.0
FP4_MLA_P_GLOBAL_SCALE: float = 448.0 * 6.0
FP4_MLA_Q_RESIDUAL_DIM: int = 64
FLASHINFER_FP4_MLA_ATTENTION_ENV = "TRTLLM_FLASHINFER_FP4_MLA_ATTENTION"
FLASHINFER_FP4_MLA_ATTENTION_BACKEND_ENV = "TRTLLM_FLASHINFER_FP4_MLA_ATTENTION_BACKEND"
FLASHINFER_FP4_MLA_DEBUG_ENV = "TRTLLM_FLASHINFER_FP4_MLA_DEBUG"
_FP4_MLA_CUTE_DSL_BACKEND = "cute_dsl"
_HPUpdatePhase = Literal["all", "context", "generation"]


# Environment and debug helpers


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def is_flashinfer_fp4_mla_attention_enabled() -> bool:
    """Return whether FlashInfer MLA should allocate no-dequant FP4 attention buffers."""
    return _env_enabled(FLASHINFER_FP4_MLA_ATTENTION_ENV)


def _fp4_mla_attention_backend() -> str:
    return os.getenv(FLASHINFER_FP4_MLA_ATTENTION_BACKEND_ENV, "triton").lower()


def _fp4_mla_debug_enabled() -> bool:
    return _env_enabled(FLASHINFER_FP4_MLA_DEBUG_ENV)


def _fp4_mla_debug(message: str) -> None:
    if _fp4_mla_debug_enabled():
        print(f"[fp4_mla_debug] {message}", flush=True)


def _tensor_layout(tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return "None"
    return (
        f"shape={list(tensor.shape)} stride={list(tensor.stride())} "
        f"dtype={tensor.dtype} device={tensor.device}"
    )


def _debug_tensor_range(name: str, tensor: Optional[torch.Tensor]) -> None:
    if not _fp4_mla_debug_enabled():
        return
    if tensor is None:
        _fp4_mla_debug(f"{name}: None")
        return
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        _fp4_mla_debug(f"{name}: empty {_tensor_layout(tensor)}")
        return
    try:
        first = flat[: min(8, flat.numel())].cpu().tolist()
        _fp4_mla_debug(
            f"{name}: {_tensor_layout(tensor)} n={flat.numel()} "
            f"min={flat.min().item()} max={flat.max().item()} first={first}"
        )
    except RuntimeError as exc:
        _fp4_mla_debug(f"{name}: failed to read range: {exc}")


def _debug_sync(label: str) -> None:
    if not _fp4_mla_debug_enabled():
        return
    if torch.cuda.is_current_stream_capturing():
        _fp4_mla_debug(f"{label}: skip sync during CUDA graph capture")
        return
    _fp4_mla_debug(f"{label}: synchronize")
    torch.cuda.synchronize()
    _fp4_mla_debug(f"{label}: sync complete")


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


# FP4 MLA scale-layout helpers


def get_fp4_mla_v_scale_pool_size(v_head_dim: int, page_size: int) -> int:
    """Return elements per page for the swizzled FP4 MLA V-scale pool.

    The PV matmul treats V as a RHS matrix shaped ``[v_head_dim, kv_tokens]``.
    NVFP4 block scales therefore group along the token/K axis, not along the
    latent dimension as the K-view cache does.  The physical layout matches the
    Triton block-scaled matmul scale layout:
    ``[ceil(v_head_dim / 128), ceil(page_size / 16 / 4), 32, 16]``.
    """
    token_scale_cols = _ceil_div(page_size, FP4_BLOCK_SIZE)
    row_groups = _ceil_div(v_head_dim, FP4_MLA_SCALE_ROW_GROUP)
    col_groups = _ceil_div(token_scale_cols, FP4_MLA_SCALE_COL_GROUP)
    return row_groups * col_groups * 32 * 16


def _get_fp4_mla_swizzled_scale_size(rows: int, cols: int) -> int:
    scale_cols = _ceil_div(cols, FP4_BLOCK_SIZE)
    row_groups = _ceil_div(rows, FP4_MLA_SCALE_ROW_GROUP)
    col_groups = _ceil_div(scale_cols, FP4_MLA_SCALE_COL_GROUP)
    return row_groups * col_groups * 32 * 16


def _use_fp4_mla_swizzled_sf() -> bool:
    return is_flashinfer_fp4_mla_attention_enabled()


def _get_fp4_mla_context_start_positions(metadata: Any, num_contexts: int) -> torch.Tensor:
    kv_cache_params = getattr(metadata, "kv_cache_params", None)
    cached_token_lens = getattr(kv_cache_params, "num_cached_tokens_per_seq", None)
    if cached_token_lens is not None:
        return torch.as_tensor(cached_token_lens[:num_contexts], dtype=torch.int64, device="cpu")

    return (
        (
            metadata.kv_lens_cuda_runtime[:num_contexts]
            - metadata.prompt_lens_cuda_runtime[:num_contexts]
        )
        .detach()
        .cpu()
    )


def _validate_fp4_mla_context_start_alignment(metadata: Any, num_contexts: int) -> None:
    context_start_positions = _get_fp4_mla_context_start_positions(metadata, num_contexts)
    bad_start = (context_start_positions < 0) | ((context_start_positions % HP_BLOCK_SIZE) != 0)
    if bool(torch.any(bad_start).item()):
        starts = context_start_positions.detach().cpu().tolist()
        raise ValueError(
            "FP4 MLA shared-tile context update requires every context "
            f"start position to be {HP_BLOCK_SIZE}-token aligned, got "
            f"start positions {starts}."
        )


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
    pool = getattr(metadata, "fp4_mla_v_scale_pool", None)
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
    global_scale = getattr(metadata, "_fp4_mla_global_scale", None)
    if global_scale is None:
        global_scale = torch.ones((1,), dtype=torch.float32, device=device)
    return global_scale


def _get_fp4_mla_cache_tensors(metadata: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    kv_cache = metadata.kv_cache_manager.get_buffers(layer_idx).view(torch.uint8)
    sf_cache = metadata.kv_cache_manager.get_block_scale_buffers(layer_idx)
    if sf_cache is None:
        raise RuntimeError("NVFP4 KV cache scale pool is not available.")
    return kv_cache, sf_cache


def _scatter_fp4_mla_kv_cache_2d_context(
    metadata: Any,
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    token_offset: int,
    local_layer: int,
    v_head_dim: int,
    head_dim: int,
    num_tokens: int,
    num_dim_blocks: int,
    sf_per_token: int,
    sf_per_page: int,
) -> None:
    num_contexts = metadata.num_contexts
    if num_contexts > 0:
        prompt_lens_cpu = metadata.prompt_lens_cpu_runtime[:num_contexts]
        ctx_token_count = int(prompt_lens_cpu.sum().item())
        if num_tokens != ctx_token_count:
            raise RuntimeError(
                f"FP4 MLA 2D context scatter needs {ctx_token_count} context tokens, got "
                f"{num_tokens}."
            )
        _validate_fp4_mla_context_start_alignment(metadata, num_contexts)

    _fp4_mla_v_scale_store_context_tokens_kernel[
        (
            num_tokens,
            num_dim_blocks,
        )
    ](
        kv_cache,
        sf_cache,
        v_sf,
        latent_cache,
        global_scale,
        metadata.batch_indices,
        metadata.positions,
        metadata.paged_kv_indices,
        metadata.paged_kv_indptr,
        metadata.paged_kv_indices.shape[0],
        metadata.paged_kv_indptr.shape[0],
        metadata.batch_indices.shape[0],
        v_sf.shape[1],
        v_sf.shape[0],
        token_offset,
        num_tokens,
        local_layer,
        metadata.page_size,
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        latent_cache.stride(0),
        latent_cache.stride(1),
        v_sf.stride(0),
        v_sf.stride(1),
        HEAD_D=head_dim,
        V_HEAD_D=v_head_dim,
        HP_BLOCK=HP_BLOCK_SIZE,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_TOKEN=sf_per_token,
        SF_PER_PAGE=sf_per_page,
    )
    _debug_sync("scatter_fp4_mla_kv_cache_2d_context")


def _scatter_fp4_mla_kv_cache_2d_generation(
    metadata: Any,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    local_layer: int,
    v_head_dim: int,
    head_dim: int,
    num_tokens: int,
    num_dim_blocks: int,
    sf_per_token: int,
    sf_per_page: int,
) -> None:
    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    num_gen = num_seqs - num_contexts
    if num_gen <= 0:
        return
    if num_tokens != num_gen:
        raise RuntimeError(
            f"FP4 MLA 2D generation scatter expected {num_gen} generation tokens, got {num_tokens}."
        )

    pool = getattr(metadata, "high_precision_kv_pool", None)
    if pool is None:
        raise RuntimeError("FP4 MLA 2D generation scatter requires the HP KV pool.")
    hp_head_dim = pool.shape[-1] // HP_BLOCK_SIZE
    if hp_head_dim < head_dim:
        raise RuntimeError(
            f"FP4 MLA 2D generation scatter needs at least {head_dim} HP channels, got "
            f"{hp_head_dim}."
        )

    page_ids = metadata.paged_kv_indices[metadata.num_context_blocks :]
    _fp4_mla_v_scale_store_hp_tail_kernel[(num_gen, num_dim_blocks)](
        kv_cache,
        sf_cache,
        v_sf,
        pool,
        global_scale,
        metadata.seq_slots[num_contexts:num_seqs],
        metadata.kv_lens_cuda_runtime[num_contexts:num_seqs],
        page_ids,
        metadata.paged_kv_indptr_decode,
        page_ids.shape[0],
        metadata.paged_kv_indptr_decode.shape[0],
        v_sf.shape[1],
        pool.shape[0],
        v_sf.shape[0],
        local_layer,
        metadata.page_size,
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        pool.stride(0),
        pool.stride(1),
        v_sf.stride(0),
        v_sf.stride(1),
        HEAD_D=hp_head_dim,
        V_HEAD_D=v_head_dim,
        HP_BLOCK=HP_BLOCK_SIZE,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_TOKEN=sf_per_token,
        SF_PER_PAGE=sf_per_page,
    )
    _debug_sync("scatter_fp4_mla_kv_cache_2d_generation")


def _scatter_fp4_mla_kv_cache_1d(
    metadata: Any,
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    layer_idx: int,
    token_offset: int,
    num_tokens: int,
    head_dim: int,
    sf_per_token: int,
    use_swizzled_sf: bool,
) -> None:
    q_fp4, q_sf = torch.ops.trtllm.fp4_quantize(
        latent_cache, global_scale, FP4_BLOCK_SIZE, False, False
    )
    q_sf = q_sf.view(num_tokens, head_dim // FP4_BLOCK_SIZE)

    packed_dim = head_dim // 2
    block_packed_dim = triton.next_power_of_2(packed_dim)
    block_sf = triton.next_power_of_2(sf_per_token)

    _fp4_mla_debug(
        "scatter launch: "
        f"num_tokens={num_tokens} token_offset={token_offset} "
        f"page_size={metadata.page_size} layer_idx={layer_idx} "
        f"head_dim={head_dim} packed_dim={packed_dim} "
        f"sf_per_token={sf_per_token} use_swizzled_sf={use_swizzled_sf}"
    )
    _fp4_mla_debug(f"scatter latent_cache: {_tensor_layout(latent_cache)}")
    _fp4_mla_debug(f"scatter kv_cache: {_tensor_layout(kv_cache)}")
    _fp4_mla_debug(f"scatter sf_cache: {_tensor_layout(sf_cache)}")
    _debug_tensor_range(
        "scatter batch_indices",
        metadata.batch_indices[token_offset : token_offset + num_tokens],
    )
    _debug_tensor_range(
        "scatter positions",
        metadata.positions[token_offset : token_offset + num_tokens],
    )
    _debug_tensor_range("scatter paged_kv_indices", metadata.paged_kv_indices)
    _debug_tensor_range("scatter paged_kv_indptr", metadata.paged_kv_indptr)

    _fp4_mla_scatter_kernel[(num_tokens,)](
        kv_cache,
        sf_cache,
        q_fp4,
        q_sf,
        metadata.batch_indices,
        metadata.positions,
        metadata.paged_kv_indices,
        metadata.paged_kv_indptr,
        metadata.paged_kv_indices.shape[0],
        metadata.paged_kv_indptr.shape[0],
        kv_cache.shape[0],
        token_offset,
        metadata.page_size,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        sf_cache.stride(0),
        sf_cache.stride(1),
        sf_cache.stride(2),
        sf_cache.stride(3),
        sf_cache.stride(4),
        q_fp4.stride(0),
        q_fp4.stride(1),
        q_sf.stride(0),
        q_sf.stride(1),
        PACKED_D=packed_dim,
        SF_PER_TOKEN=sf_per_token,
        BLOCK_PACKED_D=block_packed_dim,
        BLOCK_SF=block_sf,
        USE_SWIZZLED_SF=use_swizzled_sf,
    )
    _debug_sync("scatter_fp4_mla_kv_cache")


# Public cache update and decode entry points


def scatter_fp4_mla_kv_cache(
    metadata: Any,
    latent_cache: Optional[torch.Tensor],
    layer_idx: int,
    *,
    token_offset: int,
    phase: Optional[_HPUpdatePhase] = None,
    local_layer: Optional[int] = None,
    v_head_dim: Optional[int] = None,
) -> None:
    """Quantize MLA latent tokens and scatter them into the paged FP4 cache.

    Contract: this helper scatters exactly ``latent_cache.shape[0]`` tokens,
    reading index metadata at ``batch_indices[token_offset : token_offset + N]``
    and ``positions[token_offset : token_offset + N]``. Callers must pass a
    latent_cache pre-sliced to the current phase (context or generation) so
    that ``shape[0]`` matches the number of index entries they intend to
    consume. ``MLA.forward_impl`` (tensorrt_llm/_torch/modules/attention.py)
    slices ``latent_cache[:num_ctx_tokens]`` for context and
    ``latent_cache[num_ctx_tokens:]`` for generation before dispatching.

    When the no-dequant FP4 MLA attention path is enabled, callers should pass
    ``phase``, ``local_layer``, and ``v_head_dim``.  Context scatter then writes
    the final FP4 tile representation directly: dimensions below
    ``v_head_dim`` use one shared 16-token by 16-dim scale written into both
    K's token-major scale layout and V's dim-major scale layout.  Tail K-only
    dimensions use K's per-token 1D scales.  Generation scatter rewrites the
    active 16-token tile from the HP pool, so the caller must update the HP pool
    before invoking this helper.
    """
    if latent_cache is None or latent_cache.numel() == 0:
        return

    latent_cache = latent_cache.reshape(latent_cache.shape[0], -1).contiguous()
    num_tokens = latent_cache.shape[0]
    head_dim = latent_cache.shape[-1]
    if head_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA KV head_dim must be divisible by {FP4_BLOCK_SIZE}, got {head_dim}."
        )
    indices_len = metadata.batch_indices.shape[0]
    positions_len = metadata.positions.shape[0]
    if token_offset + num_tokens > indices_len or token_offset + num_tokens > positions_len:
        raise RuntimeError(
            f"FP4 MLA scatter would read batch_indices[{token_offset}:"
            f"{token_offset + num_tokens}] / positions[{token_offset}:"
            f"{token_offset + num_tokens}], but only {indices_len} / "
            f"{positions_len} entries are available. This indicates "
            "latent_cache was not pre-sliced to the current phase's token "
            "range (see MLA.forward_impl)."
        )

    use_swizzled_sf = _use_fp4_mla_swizzled_sf()
    if use_swizzled_sf:
        _validate_fp4_mla_cache_shape(metadata.page_size, head_dim)

    global_scale = _get_fp4_mla_global_scale(metadata, latent_cache.device)
    kv_cache, sf_cache = _get_fp4_mla_cache_tensors(metadata, layer_idx)
    sf_per_token = head_dim // FP4_BLOCK_SIZE

    use_2d_scatter = (
        use_swizzled_sf
        and phase in ("context", "generation")
        and getattr(metadata, "fp4_mla_v_scale_pool", None) is not None
    )
    if use_2d_scatter:
        assert phase is not None
        if local_layer is None or v_head_dim is None:
            raise ValueError("Real FP4 MLA scatter requires local_layer and v_head_dim.")
        if metadata.page_size % HP_BLOCK_SIZE != 0:
            raise ValueError(
                f"FP4 MLA scatter requires page_size divisible by "
                f"{HP_BLOCK_SIZE}, got {metadata.page_size}."
            )
        if v_head_dim > head_dim:
            raise ValueError(f"FP4 MLA v_head_dim={v_head_dim} cannot exceed head_dim={head_dim}.")
        if v_head_dim % FP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"FP4 MLA v_head_dim must be divisible by {FP4_BLOCK_SIZE}, got {v_head_dim}."
            )

        sf_cache = sf_cache.view(torch.float8_e4m3fn)
        v_sf = get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=v_head_dim)
        num_dim_blocks = triton.cdiv(head_dim, FP4_BLOCK_SIZE)
        sf_per_page = metadata.page_size // HP_BLOCK_SIZE
        _fp4_mla_debug(
            "scatter 2d launch: "
            f"phase={phase} num_tokens={num_tokens} "
            f"token_offset={token_offset} layer_idx={layer_idx} "
            f"local_layer={local_layer} head_dim={head_dim} "
            f"v_head_dim={v_head_dim} num_dim_blocks={num_dim_blocks}"
        )
        _fp4_mla_debug(f"scatter 2d kv_cache: {_tensor_layout(kv_cache)}")
        _fp4_mla_debug(f"scatter 2d sf_cache: {_tensor_layout(sf_cache)}")
        _fp4_mla_debug(f"scatter 2d v_sf: {_tensor_layout(v_sf)}")

        if phase == "context":
            _scatter_fp4_mla_kv_cache_2d_context(
                metadata,
                latent_cache,
                kv_cache,
                sf_cache,
                v_sf,
                global_scale,
                token_offset=token_offset,
                local_layer=local_layer,
                v_head_dim=v_head_dim,
                head_dim=head_dim,
                num_tokens=num_tokens,
                num_dim_blocks=num_dim_blocks,
                sf_per_token=sf_per_token,
                sf_per_page=sf_per_page,
            )
        else:
            _scatter_fp4_mla_kv_cache_2d_generation(
                metadata,
                kv_cache,
                sf_cache,
                v_sf,
                global_scale,
                local_layer=local_layer,
                v_head_dim=v_head_dim,
                head_dim=head_dim,
                num_tokens=num_tokens,
                num_dim_blocks=num_dim_blocks,
                sf_per_token=sf_per_token,
                sf_per_page=sf_per_page,
            )
        return

    _scatter_fp4_mla_kv_cache_1d(
        metadata,
        latent_cache,
        kv_cache,
        sf_cache,
        global_scale,
        layer_idx=layer_idx,
        token_offset=token_offset,
        num_tokens=num_tokens,
        head_dim=head_dim,
        sf_per_token=sf_per_token,
        use_swizzled_sf=use_swizzled_sf,
    )


def _ensure_decode_workspace(
    metadata: Any,
    head_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_blocks = _get_decode_workspace_num_blocks(metadata)
    workspace = getattr(metadata, "_fp4_mla_decode_cache_buf", None)
    needs_alloc = (
        workspace is None
        or workspace.shape[0] < max(num_blocks, 1)
        or workspace.shape[1] != metadata.page_size
        or workspace.shape[2] != head_dim
        or workspace.dtype != dtype
    )
    if needs_alloc:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError(
                "Cannot allocate FlashInfer FP4 MLA decode workspace while "
                "capturing a CUDA graph. Run a warmup prepare/forward first."
            )
        workspace = torch.empty(
            (max(num_blocks, 1), metadata.page_size, head_dim),
            dtype=dtype,
            device=metadata.paged_kv_indices.device,
        )
        metadata._fp4_mla_decode_cache_buf = workspace
    return workspace[:num_blocks]


def _get_decode_workspace_num_blocks(metadata: Any) -> int:
    if metadata.is_cuda_graph:
        max_blocks_per_seq = (
            metadata.kv_cache_manager.max_seq_len + metadata.page_size - 1
        ) // metadata.page_size
        max_graph_blocks = metadata.max_num_requests * max_blocks_per_seq
        return min(
            metadata.kv_cache_manager.blocks_in_primary_pool,
            max_graph_blocks,
        )
    return metadata.num_generation_blocks


def _get_decode_src_page_ids(metadata: Any, num_blocks: int) -> torch.Tensor:
    page_ids = (
        metadata._paged_kv_indices
        if metadata.is_cuda_graph and hasattr(metadata, "_paged_kv_indices")
        else metadata.paged_kv_indices
    )
    src_page_ids = page_ids[metadata.num_context_blocks : metadata.num_context_blocks + num_blocks]
    if src_page_ids.numel() != num_blocks:
        raise RuntimeError(
            f"FP4 MLA dequant needs {num_blocks} decode page ids from "
            f"paged_kv_indices[{metadata.num_context_blocks}:"
            f"{metadata.num_context_blocks + num_blocks}], got "
            f"{src_page_ids.numel()}."
        )
    return src_page_ids


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


def get_fp4_mla_decode_cache(
    metadata: Any,
    layer_idx: int,
    local_layer: int,
    *,
    head_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a compact dequantized MLA cache for FlashInfer decode."""
    # Must match scatter_fp4_mla_kv_cache: the env var picks the SF layout,
    # not the page size. When the dequant fallback is the read path
    # (env disabled), scatter wrote linear scales and we must read linear.
    use_swizzled_sf = _use_fp4_mla_swizzled_sf()
    if use_swizzled_sf:
        _validate_fp4_mla_cache_shape(metadata.page_size, head_dim)
    combined = _ensure_decode_workspace(metadata, head_dim, dtype)
    num_blocks = combined.shape[0]
    if num_blocks == 0:
        return combined

    kv_cache, sf_cache = _get_fp4_mla_cache_tensors(metadata, layer_idx)
    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    global_scale = _get_fp4_mla_global_scale(metadata, combined.device)
    src_page_ids = _get_decode_src_page_ids(metadata, num_blocks)
    block_d = triton.next_power_of_2(head_dim)

    _fp4_mla_dequant_kernel[(num_blocks, metadata.page_size)](
        combined,
        kv_cache,
        sf_cache,
        global_scale,
        src_page_ids,
        src_page_ids.shape[0],
        kv_cache.shape[0],
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        sf_cache.stride(0),
        sf_cache.stride(1),
        sf_cache.stride(2),
        sf_cache.stride(3),
        sf_cache.stride(4),
        combined.stride(0),
        combined.stride(1),
        combined.stride(2),
        D=head_dim,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        BLOCK_D=block_d,
        USE_SWIZZLED_SF=use_swizzled_sf,
    )

    num_gen = metadata.num_seqs - metadata.num_contexts
    if num_gen > 0 and metadata.high_precision_kv_pool is not None:
        pool = metadata.high_precision_kv_pool
        _fp4_mla_overlay_hp_tail_kernel[(num_gen, HP_BLOCK_SIZE)](
            combined,
            pool,
            metadata.seq_slots[metadata.num_contexts : metadata.num_seqs],
            metadata.kv_lens_cuda_runtime[metadata.num_contexts : metadata.num_seqs],
            metadata.paged_kv_indptr_decode,
            pool.shape[0],
            pool.shape[1],
            combined.shape[0],
            local_layer,
            metadata.page_size,
            combined.stride(0),
            combined.stride(1),
            combined.stride(2),
            pool.stride(0),
            pool.stride(1),
            D=head_dim,
            BLOCK_D=block_d,
            HP_BLOCK=HP_BLOCK_SIZE,
        )

    return combined


def _ensure_workspace_tensor(
    metadata: Any,
    attr_name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = getattr(metadata, attr_name, None)
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
        setattr(metadata, attr_name, tensor)

    slices = tuple(slice(0, dim) for dim in shape)
    return tensor[slices]


def _max_generation_pages(metadata: Any) -> int:
    num_gen = metadata.num_seqs - metadata.num_contexts
    if num_gen <= 0:
        return 0
    num_blocks = getattr(metadata, "num_blocks", None)
    if num_blocks is not None:
        return max(num_blocks[metadata.num_contexts : metadata.num_seqs])
    return metadata.num_generation_blocks


def _host_int_list(value: Any, start: int, end: int) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            return None
        return [int(item) for item in value[start:end].tolist()]
    try:
        return [int(item) for item in value[start:end]]
    except (TypeError, ValueError):
        return None


def _infer_cutile_assume_full_pages(metadata: Any, max_pages: int, page_size: int) -> bool:
    if getattr(metadata, "is_cuda_graph", False):
        return False

    start = metadata.num_contexts
    end = metadata.num_seqs
    kv_cache_params = getattr(metadata, "kv_cache_params", None)
    cached_token_lens = _host_int_list(
        getattr(kv_cache_params, "num_cached_tokens_per_seq", None),
        start,
        end,
    )
    seq_lens_kv = _host_int_list(getattr(metadata, "seq_lens_kv", None), start, end)
    if cached_token_lens is not None and seq_lens_kv is not None:
        if len(cached_token_lens) != len(seq_lens_kv):
            return False
        kv_lens = [
            cached_len + seq_len for cached_len, seq_len in zip(cached_token_lens, seq_lens_kv)
        ]
    elif kv_cache_params is None:
        kv_lens = _host_int_list(getattr(metadata, "prompt_lens_cpu_runtime", None), start, end)
    else:
        return False

    return bool(kv_lens) and min(kv_lens) == max(kv_lens) == max_pages * page_size


def run_fp4_mla_attention_decode(
    metadata: Any,
    layer_idx: int,
    local_layer: int,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    output: torch.Tensor,
    *,
    sm_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> None:
    """Run MLA decode with FP4 QK and FP4 PV tensor-core matmuls.

    Q is quantized to FP4, QK reads the packed K-view cache with swizzled
    block scales, softmax probabilities are quantized to FP4 per page, and PV
    repacks V nibbles from the shared KV cache while reading the auxiliary
    V-view scale pool.  No BF16 dequantized KV workspace is materialized on
    this path.
    """
    if not is_flashinfer_fp4_mla_attention_enabled():
        raise RuntimeError(
            f"FP4 MLA attention decode requires {FLASHINFER_FP4_MLA_ATTENTION_ENV}=1."
        )

    head_dim = kv_lora_rank + qk_rope_head_dim
    _validate_fp4_mla_cache_shape(metadata.page_size, head_dim)
    if metadata.page_size != FP4_MLA_TOKENS_PER_BLOCK:
        raise ValueError(
            f"FP4 MLA attention decode requires page_size={FP4_MLA_TOKENS_PER_BLOCK}, "
            f"got {metadata.page_size}."
        )

    num_gen = q_nope.shape[0]
    if num_gen == 0:
        return
    num_gen_seqs = metadata.num_seqs - metadata.num_contexts
    if num_gen != num_gen_seqs:
        raise NotImplementedError(
            "FP4 MLA attention decode currently supports one query token per "
            f"generation sequence, got {num_gen} query tokens for "
            f"{num_gen_seqs} sequences."
        )

    num_heads = q_nope.shape[1]
    if q_pe.shape[:2] != (num_gen, num_heads):
        raise ValueError("FP4 MLA attention q_nope/q_pe batch dimensions do not match.")
    if output.shape[:2] != (num_gen, num_heads):
        raise ValueError("FP4 MLA attention output batch dimensions do not match.")
    if q_nope.shape[-1] != kv_lora_rank:
        raise ValueError(
            f"q_nope last dimension must be kv_lora_rank={kv_lora_rank}, got {q_nope.shape[-1]}."
        )
    if q_pe.shape[-1] != qk_rope_head_dim:
        raise ValueError(
            f"q_pe last dimension must be qk_rope_head_dim={qk_rope_head_dim}, "
            f"got {q_pe.shape[-1]}."
        )

    if getattr(metadata, "fp4_mla_v_scale_pool", None) is None:
        raise RuntimeError(
            "FP4 MLA attention decode requires the auxiliary V scale pool to be allocated."
        )

    global_scale = _get_fp4_mla_global_scale(metadata, q_nope.device)
    q_residual_dim = FP4_MLA_Q_RESIDUAL_DIM
    _validate_fp4_mla_attention_q_shape(head_dim, q_residual_dim)

    q_full = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_q_buf",
        (num_gen, num_heads, head_dim),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )
    q_full[..., :kv_lora_rank].copy_(q_nope)
    q_full[..., kv_lora_rank:].copy_(q_pe)
    q_2d = q_full.reshape(num_gen * num_heads, head_dim)
    if q_2d.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(
            f"FP4 MLA residual Q quantization requires BF16 or FP8 Q; got {q_2d.dtype}."
        )
    q_fp4, q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
        q_2d,
        global_scale,
        q_residual_dim,
        is_act=True,
    )
    q_sf = q_sf.view(torch.float8_e4m3fn)

    kv_cache, sf_cache = _get_fp4_mla_cache_tensors(metadata, layer_idx)
    sf_cache = sf_cache.view(torch.float8_e4m3fn)

    num_gen_blocks = metadata.num_generation_blocks
    v_sf = get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=kv_lora_rank)[local_layer].view(
        torch.float8_e4m3fn
    )

    src_page_ids = metadata.paged_kv_indices[
        metadata.num_context_blocks : metadata.num_context_blocks + num_gen_blocks
    ]
    kv_lens = metadata.kv_lens_cuda_runtime[metadata.num_contexts : metadata.num_seqs]
    max_pages = _max_generation_pages(metadata)
    if max_pages == 0:
        return

    backend = _fp4_mla_attention_backend()
    if backend == "cutile":
        from .fp4_mla_cutile import fp4_mla_paged_attention

        total_p_rows = max(src_page_ids.shape[0] * num_heads, 1)
        p_fp4 = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_p_buf",
            (total_p_rows, metadata.page_size // 2),
            dtype=torch.uint8,
            device=q_nope.device,
        )
        p_sf = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_p_sf_buf",
            (max(_get_fp4_mla_swizzled_scale_size(total_p_rows, metadata.page_size), 1),),
            dtype=torch.float8_e4m3fn,
            device=q_nope.device,
        )
        stats_shape = (num_gen, num_heads)
        max_scores = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_max_buf",
            stats_shape,
            dtype=torch.float32,
            device=q_nope.device,
        )
        denom = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_denom_buf",
            stats_shape,
            dtype=torch.float32,
            device=q_nope.device,
        )
        page_max = None
        page_sum = None
        if max_pages >= 8:
            page_stats_shape = (num_gen, max_pages, num_heads)
            page_max = _ensure_workspace_tensor(
                metadata,
                "_fp4_mla_attention_page_max_buf",
                page_stats_shape,
                dtype=torch.float32,
                device=q_nope.device,
            )
            page_sum = _ensure_workspace_tensor(
                metadata,
                "_fp4_mla_attention_page_sum_buf",
                page_stats_shape,
                dtype=torch.float32,
                device=q_nope.device,
            )
        assume_full_pages = _infer_cutile_assume_full_pages(
            metadata,
            max_pages,
            metadata.page_size,
        )
        assume_valid_pages = False
        _fp4_mla_debug(
            "attention decode cutile launch: "
            f"num_gen={num_gen} num_heads={num_heads} local_layer={local_layer} "
            f"layer_idx={layer_idx} head_dim={head_dim} kv_lora_rank={kv_lora_rank} "
            f"rope_dim={qk_rope_head_dim} max_pages={max_pages} "
            f"assume_full_pages={assume_full_pages} "
            f"assume_valid_pages={assume_valid_pages}"
        )
        fp4_mla_paged_attention(
            q_fp4,
            q_sf,
            kv_cache,
            sf_cache,
            v_sf,
            global_scale,
            src_page_ids,
            metadata.paged_kv_indptr_decode,
            kv_lens,
            output,
            sm_scale=float(sm_scale),
            num_heads=num_heads,
            v_head_dim=kv_lora_rank,
            page_size=metadata.page_size,
            q_residual_dim=q_residual_dim,
            max_pages=max_pages,
            assume_full_pages=assume_full_pages,
            assume_valid_pages=assume_valid_pages,
            p_fp4_workspace=p_fp4,
            p_sf_workspace=p_sf,
            max_scores_workspace=max_scores,
            denom_workspace=denom,
            page_max_workspace=page_max,
            page_sum_workspace=page_sum,
        )
        _debug_sync("attention_cutile")
        return
    total_p_rows = num_gen_blocks * num_heads
    p_fp4 = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_p_buf",
        (max(total_p_rows, 1), metadata.page_size // 2),
        dtype=torch.uint8,
        device=q_nope.device,
    )[:total_p_rows]
    p_sf = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_p_sf_buf",
        (max(_get_fp4_mla_swizzled_scale_size(total_p_rows, metadata.page_size), 1),),
        dtype=torch.float8_e4m3fn,
        device=q_nope.device,
    )
    stats_shape = (num_gen, num_heads)
    max_scores = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_max_buf",
        stats_shape,
        dtype=torch.float32,
        device=q_nope.device,
    )
    denom = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_denom_buf",
        stats_shape,
        dtype=torch.float32,
        device=q_nope.device,
    )

    if backend == _FP4_MLA_CUTE_DSL_BACKEND:
        from .fp4_mla_cute import run_fp4_mla_attention_decode_cute

        page_stats_shape = (num_gen, max_pages, num_heads)
        page_max = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_page_max_buf",
            page_stats_shape,
            dtype=torch.float32,
            device=q_nope.device,
        )
        page_sum = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_page_sum_buf",
            page_stats_shape,
            dtype=torch.float32,
            device=q_nope.device,
        )
        run_fp4_mla_attention_decode_cute(
            output=output,
            max_scores=max_scores,
            denom=denom,
            page_max=page_max,
            page_sum=page_sum,
            p_fp4=p_fp4,
            p_sf=p_sf,
            q_fp4=q_fp4,
            q_sf=q_sf,
            kv_cache=kv_cache,
            sf_cache=sf_cache,
            v_sf=v_sf,
            global_scale=global_scale,
            src_page_ids=src_page_ids,
            paged_kv_indptr_decode=metadata.paged_kv_indptr_decode,
            kv_lens=kv_lens,
            sm_scale=float(sm_scale),
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            q_residual_dim=q_residual_dim,
            page_size=metadata.page_size,
            max_pages=max_pages,
        )
        _debug_sync("attention_cute_dsl")
        return
    if backend != "triton":
        raise ValueError(
            f"Unsupported FP4 MLA attention backend '{backend}'. "
            f"Set {FLASHINFER_FP4_MLA_ATTENTION_BACKEND_ENV} to 'triton', "
            "'cutile', or 'cute_dsl'."
        )

    p_probs = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_p_prob_buf",
        (max(num_gen * num_heads, 1), metadata.page_size),
        dtype=torch.float32,
        device=q_nope.device,
    )[: num_gen * num_heads]

    block_h = 128
    block_t = metadata.page_size
    block_k = 256
    block_v = 128
    q_head_dim = head_dim + q_residual_dim
    q_sf_per_token = q_head_dim // FP4_BLOCK_SIZE
    k_sf_per_token = head_dim // FP4_BLOCK_SIZE
    sf_per_page = metadata.page_size // FP4_BLOCK_SIZE
    num_head_blocks = triton.cdiv(num_heads, block_h)

    _fp4_mla_debug(
        "attention decode: "
        f"num_gen={num_gen} num_heads={num_heads} local_layer={local_layer} "
        f"layer_idx={layer_idx} head_dim={head_dim} q_head_dim={q_head_dim} "
        f"q_residual_dim={q_residual_dim} "
        f"kv_lora_rank={kv_lora_rank} rope_dim={qk_rope_head_dim} "
        f"num_gen_blocks={num_gen_blocks} max_pages={max_pages} "
        f"num_head_blocks={num_head_blocks} sm_scale={sm_scale}"
    )
    _fp4_mla_debug(f"attention q_nope: {_tensor_layout(q_nope)}")
    _fp4_mla_debug(f"attention q_pe: {_tensor_layout(q_pe)}")
    _fp4_mla_debug(f"attention output: {_tensor_layout(output)}")
    _fp4_mla_debug(f"attention q_fp4: {_tensor_layout(q_fp4)}")
    _fp4_mla_debug(f"attention q_sf: {_tensor_layout(q_sf)}")
    _fp4_mla_debug(f"attention kv_cache: {_tensor_layout(kv_cache)}")
    _fp4_mla_debug(f"attention sf_cache: {_tensor_layout(sf_cache)}")
    _fp4_mla_debug(f"attention v_sf: {_tensor_layout(v_sf)}")
    _fp4_mla_debug(f"attention p_fp4: {_tensor_layout(p_fp4)}")
    _fp4_mla_debug(f"attention p_sf: {_tensor_layout(p_sf)}")
    _fp4_mla_debug(f"attention p_probs: {_tensor_layout(p_probs)}")
    _debug_tensor_range("attention src_page_ids", src_page_ids)
    _debug_tensor_range("attention paged_kv_indptr_decode", metadata.paged_kv_indptr_decode)
    _debug_tensor_range("attention kv_lens", kv_lens)

    _fp4_mla_debug(
        "attention stats launch: "
        f"grid=({num_gen}, {num_head_blocks}) "
        f"block_h={block_h} block_t={block_t} block_k={block_k}"
    )
    _fp4_mla_attention_stats_kernel[(num_gen, num_head_blocks)](
        max_scores,
        denom,
        q_fp4,
        q_sf,
        kv_cache,
        sf_cache,
        global_scale,
        src_page_ids,
        metadata.paged_kv_indptr_decode,
        kv_lens,
        src_page_ids.shape[0],
        kv_cache.shape[0],
        q_fp4.stride(0),
        q_fp4.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        max_scores.stride(0),
        sm_scale=sm_scale,
        NUM_HEADS=num_heads,
        Q_HEAD_D=q_head_dim,
        K_HEAD_D=head_dim,
        Q_RESIDUAL_D=q_residual_dim,
        PAGE_SIZE=metadata.page_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        Q_SF_PER_TOKEN=q_sf_per_token,
        K_SF_PER_TOKEN=k_sf_per_token,
        MAX_PAGES=max_pages,
        BLOCK_H=block_h,
        BLOCK_T=block_t,
        BLOCK_K=block_k,
    )
    _debug_sync("attention_stats")

    for page_rel in range(max_pages):
        _fp4_mla_debug(
            "attention prob page store launch: "
            f"page_rel={page_rel} grid=({num_gen}, {num_head_blocks})"
        )
        _fp4_mla_attention_prob_store_page_kernel[(num_gen, num_head_blocks)](
            p_probs,
            max_scores,
            denom,
            q_fp4,
            q_sf,
            kv_cache,
            sf_cache,
            global_scale,
            src_page_ids,
            metadata.paged_kv_indptr_decode,
            kv_lens,
            page_rel,
            src_page_ids.shape[0],
            kv_cache.shape[0],
            p_probs.stride(0),
            p_probs.stride(1),
            q_fp4.stride(0),
            q_fp4.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(4),
            sf_cache.stride(0),
            max_scores.stride(0),
            sm_scale=sm_scale,
            NUM_HEADS=num_heads,
            Q_HEAD_D=q_head_dim,
            K_HEAD_D=head_dim,
            Q_RESIDUAL_D=q_residual_dim,
            PAGE_SIZE=metadata.page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            Q_SF_PER_TOKEN=q_sf_per_token,
            K_SF_PER_TOKEN=k_sf_per_token,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
        )
        _debug_sync(f"attention_prob_page_store_{page_rel}")

        _fp4_mla_debug(
            "attention prob page pack launch: "
            f"page_rel={page_rel} grid=({num_gen}, {sf_per_page}, "
            f"{num_head_blocks})"
        )
        _fp4_mla_attention_prob_pack_page_kernel[
            (
                num_gen,
                sf_per_page,
                num_head_blocks,
            )
        ](
            p_fp4,
            p_sf,
            p_probs,
            metadata.paged_kv_indptr_decode,
            kv_lens,
            page_rel,
            src_page_ids.shape[0],
            p_fp4.stride(0),
            p_fp4.stride(1),
            p_probs.stride(0),
            p_probs.stride(1),
            NUM_HEADS=num_heads,
            PAGE_SIZE=metadata.page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            SF_PER_PAGE=sf_per_page,
            P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
            BLOCK_H=block_h,
        )
        _debug_sync(f"attention_prob_page_pack_{page_rel}")

    _fp4_mla_debug(
        "attention pv launch: "
        f"grid=({num_gen}, {num_head_blocks}, "
        f"{triton.cdiv(kv_lora_rank, block_v)}) block_v={block_v}"
    )
    _fp4_mla_attention_pv_kernel[
        (
            num_gen,
            num_head_blocks,
            triton.cdiv(kv_lora_rank, block_v),
        )
    ](
        output,
        p_fp4,
        p_sf,
        kv_cache,
        v_sf,
        global_scale,
        src_page_ids,
        metadata.paged_kv_indptr_decode,
        kv_lens,
        src_page_ids.shape[0],
        kv_cache.shape[0],
        output.stride(0),
        output.stride(1),
        output.stride(2),
        p_fp4.stride(0),
        p_fp4.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        v_sf.stride(0),
        NUM_HEADS=num_heads,
        V_HEAD_D=kv_lora_rank,
        PAGE_SIZE=metadata.page_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_PAGE=sf_per_page,
        MAX_PAGES=max_pages,
        P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
        BLOCK_H=block_h,
        BLOCK_V=block_v,
    )
    _debug_sync("attention_pv")


def update_hp_kv_for_fp4_mla(
    metadata: Any,
    latent_cache: Optional[torch.Tensor],
    local_layer: int,
    *,
    phase: _HPUpdatePhase = "all",
) -> None:
    """Store recent KV tokens at BF16 into the high-precision pool.

    Called on every layer before the attention kernel. The pool acts as a
    circular buffer of HP_BLOCK_SIZE slots per sequence:

    Context phase stores the last ``kv_len % HP_BLOCK_SIZE`` new tokens of
        each request into buffer positions [0, remainder). These are the
        tail tokens that do not fill a complete FP4 block of 16.

    Generation phase stores the single new token for each request into
        position ``(kv_len - 1) % HP_BLOCK_SIZE``, overwriting the oldest
        entry in the circular buffer.

    The Triton kernels use the GPU ``seq_slots`` tensor for scatter indexing
    and are CUDA-graph-compatible for the generation phase.

    Args:
        metadata: Attention metadata exposing ``num_contexts``, ``num_seqs``,
            ``seq_slots`` / ``seq_slots_cpu``, ``request_ids``,
            ``is_cuda_graph``, ``is_warmup``, ``hp_pool_owners``,
            ``high_precision_kv_pool``, ``prompt_lens_cpu_runtime``,
            ``prompt_lens_cuda_runtime``, ``kv_lens_cuda_runtime``.
        latent_cache: MLA latent cache for the current tokens, shape
            [num_tokens, head_dim]. When ``None``, only ownership tracking
            runs (no data is written to the pool).
        local_layer: Layer index within the local pipeline-parallel slice.
        phase: Which portion of ``latent_cache`` is present. ``"all"`` means
            context tokens followed by generation tokens, ``"context"`` means
            only context tokens, and ``"generation"`` means only generation
            tokens.
    """
    if phase not in ("all", "context", "generation"):
        raise ValueError(f"Unexpected FP4 MLA HP update phase: {phase}")
    if metadata.hp_pool_owners is None:
        return
    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    update_context = phase in ("all", "context")
    update_generation = phase in ("all", "generation")

    # ------------------------------------------------------------------
    # Ownership tracking (layer 0, eager mode only - debug guard).
    # Context phase never uses CUDA graph; decode check is debug-only.
    # ------------------------------------------------------------------
    if local_layer == 0 and not metadata.is_cuda_graph and not metadata.is_warmup:
        # Context: register ownership of each seq_slot.
        if update_context:
            for batch_idx in range(num_contexts):
                seq_slot = metadata.seq_slots_cpu[batch_idx].item()
                request_id = metadata.request_ids[batch_idx]
                metadata.hp_pool_owners[seq_slot] = request_id

        # Decode: verify that the expected request still owns each slot.
        if update_generation:
            for batch_idx in range(num_contexts, num_seqs):
                seq_slot = metadata.seq_slots_cpu[batch_idx].item()
                request_id = metadata.request_ids[batch_idx]
                owner = metadata.hp_pool_owners.get(seq_slot)
                if owner != request_id:
                    raise RuntimeError(
                        f"HP KV pool ownership mismatch: seq_slot={seq_slot} "
                        f"is owned by request {owner} but request "
                        f"{request_id} is attempting to use it"
                    )

    if latent_cache is None:
        return

    # ------------------------------------------------------------------
    # Triton kernel dispatch - runs on every layer, CUDA-graph-safe.
    # ------------------------------------------------------------------
    pool = metadata.high_precision_kv_pool
    head_dim = latent_cache.shape[-1]
    pool_head_dim = pool.shape[-1] // HP_BLOCK_SIZE
    if pool_head_dim < head_dim:
        raise RuntimeError(
            f"FP4 MLA HP pool head dimension is too small: got "
            f"{pool_head_dim}, need at least {head_dim}."
        )
    block_d = triton.next_power_of_2(head_dim)
    pool_s0 = pool.stride(0)  # stride across sequence slots
    pool_s1 = pool.stride(1)  # stride across layers
    lc_stride = latent_cache.stride(0)
    _fp4_mla_debug(
        "hp update: "
        f"phase={phase} local_layer={local_layer} num_contexts={num_contexts} "
        f"num_seqs={num_seqs} head_dim={head_dim} "
        f"pool_head_dim={pool_head_dim} block_d={block_d}"
    )
    _fp4_mla_debug(f"hp latent_cache: {_tensor_layout(latent_cache)}")
    _fp4_mla_debug(f"hp pool: {_tensor_layout(pool)}")
    _debug_tensor_range("hp seq_slots", metadata.seq_slots[:num_seqs])
    _debug_tensor_range("hp kv_lens", metadata.kv_lens_cuda_runtime[:num_seqs])

    # Context phase: store last (kv_len % HP_BLOCK_SIZE) new tokens.
    if update_context and num_contexts > 0:
        prompt_lens_cpu = metadata.prompt_lens_cpu_runtime[:num_contexts]
        # Exclusive prefix sum: token offset in latent_cache for each ctx seq.
        token_offsets_cpu = torch.zeros(num_contexts, dtype=torch.int32, device="cpu")
        if num_contexts > 1:
            token_offsets_cpu[1:].copy_(torch.cumsum(prompt_lens_cpu[:-1].to(torch.int32), dim=0))
        token_offsets_gpu = token_offsets_cpu.to(pool.device, non_blocking=False)
        prompt_lens_gpu = metadata.prompt_lens_cuda_runtime[:num_contexts]

        _fp4_mla_debug(
            "hp context launch: "
            f"grid=({num_contexts}, {HP_BLOCK_SIZE}) "
            f"token_offsets={token_offsets_cpu.tolist()}"
        )
        _debug_tensor_range("hp context prompt_lens", prompt_lens_gpu)
        _hp_kv_store_context_kernel[(num_contexts, HP_BLOCK_SIZE)](
            pool,
            latent_cache,
            metadata.seq_slots,
            metadata.kv_lens_cuda_runtime,
            token_offsets_gpu,
            prompt_lens_gpu,
            pool.shape[0],
            pool.shape[1],
            local_layer,
            pool_s0,
            pool_s1,
            lc_stride,
            D=head_dim,
            POOL_HEAD_D=pool_head_dim,
            BLOCK_D=block_d,
            HP_BLOCK=HP_BLOCK_SIZE,
        )
        _debug_sync("hp_context")

    # Generation phase: store current token at (kv_len - 1) % HP_BLOCK_SIZE.
    num_gen = num_seqs - num_contexts
    if update_generation and num_gen > 0:
        gen_tok_start = 0
        if phase == "all":
            # Scalar offset: number of context tokens packed before gen tokens.
            gen_tok_start = int(metadata.prompt_lens_cpu_runtime[:num_contexts].sum().item())

        _fp4_mla_debug(f"hp generation launch: grid=({num_gen},) gen_tok_start={gen_tok_start}")
        _hp_kv_store_gen_kernel[(num_gen,)](
            pool,
            latent_cache,
            metadata.seq_slots[num_contexts:],
            metadata.kv_lens_cuda_runtime[num_contexts:],
            gen_tok_start,
            pool.shape[0],
            pool.shape[1],
            local_layer,
            pool_s0,
            pool_s1,
            lc_stride,
            D=head_dim,
            POOL_HEAD_D=pool_head_dim,
            BLOCK_D=block_d,
            HP_BLOCK=HP_BLOCK_SIZE,
        )
        _debug_sync("hp_generation")
