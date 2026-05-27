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
import triton.language as tl

from tensorrt_llm.logger import logger

from .fp4_mla_kernels import (
    _fp4_mla_dequant_kernel,
    _fp4_mla_overlay_hp_tail_kernel,
    _fp4_mla_scatter_kernel,
    _fp4_mla_v_scale_store_context_tokens_kernel,
    _fp4_mla_v_scale_store_generation_tiles_kernel,
    _hp_kv_restore_rejected_from_pool_kernel,
    _hp_kv_restore_rejected_from_values_kernel,
    _hp_kv_store_context_kernel,
    _hp_kv_store_gen_kernel,
)

HP_BLOCK_SIZE: int = 16
FP4_BLOCK_SIZE: int = 16
FP4_MLA_TOKENS_PER_BLOCK: int = 128
FP4_MLA_SCALE_ROW_GROUP: int = 128
FP4_MLA_SCALE_COL_GROUP: int = 4
FP4_MLA_KV_GLOBAL_SCALE: float = 448.0 * 6.0 / (448.0 * 6.0)
FP4_MLA_P_GLOBAL_SCALE: float = 448.0 * 6.0
FP4_MLA_Q_RESIDUAL_DIM: int = 64
FLASHINFER_FP4_MLA_ATTENTION_ENV = "TRTLLM_FLASHINFER_FP4_MLA_ATTENTION"
FLASHINFER_FP4_MLA_ATTENTION_BACKEND_ENV = "TRTLLM_FLASHINFER_FP4_MLA_ATTENTION_BACKEND"
FLASHINFER_FP4_MLA_DEBUG_ENV = "TRTLLM_FLASHINFER_FP4_MLA_DEBUG"
# Opt-in (triton only): re-quantize the active decode page each step
# from the BF16 staging buffer with an exact per-page (K/V shared) global scale,
# instead of the static FP4_MLA_KV_GLOBAL_SCALE. Off by default so the existing
# static-scale path is unchanged until validated.
FP4_MLA_PER_PAGE_SCALE_ENV = "TRTLLM_FP4_MLA_PER_PAGE_SCALE"
# Diagnostic (overlap + MTP debugging): assert that the read-side decode paging
# (num_blocks / num_generation_blocks / paged_kv_indptr_decode) and the gen
# token positions are consistent with the (corrected) kv_lens_cuda_runtime.
FP4_MLA_DEBUG_ASSERT_ENV = "TRTLLM_FP4_MLA_DEBUG_ASSERT"
# Fix (opt-in): rebuild the read-side decode paging from the corrected kv_lens
# in _preprocess_inputs after the overlap kv_lens correction.
FP4_MLA_OVERLAP_REPAGE_ENV = "TRTLLM_FP4_MLA_OVERLAP_REPAGE"
_HPUpdatePhase = Literal["all", "context", "generation"]
_FP4_MLA_MTP_HP_SNAPSHOTS = "_fp4_mla_mtp_hp_snapshots"
# Separate MTP snapshot store for the per-page dynamic-scale staging buffer
# (fp4_mla_page_stage_pool). Kept distinct from the HP-pool snapshots so the
# two rollback paths never alias.
_FP4_MLA_MTP_STAGE_SNAPSHOTS = "_fp4_mla_mtp_stage_snapshots"


# Environment and debug helpers


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def is_flashinfer_fp4_mla_attention_enabled() -> bool:
    """Return whether FlashInfer MLA should allocate no-dequant FP4 attention buffers."""
    return _env_enabled(FLASHINFER_FP4_MLA_ATTENTION_ENV)


def _fp4_mla_attention_backend() -> str:
    return os.getenv(FLASHINFER_FP4_MLA_ATTENTION_BACKEND_ENV, "triton").lower()


def fp4_mla_per_page_scale_enabled() -> bool:
    """Return whether the per-page dynamic FP4 global-scale path is on.

    Only the ``triton`` backend implements the matching read side, so the
    per-page store + dynamic Q scale are gated to it; on any other backend the
    flag is ignored (the static FP4_MLA_KV_GLOBAL_SCALE path runs unchanged).
    """
    return _env_enabled(FP4_MLA_PER_PAGE_SCALE_ENV) and _fp4_mla_attention_backend() == "triton"


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


def _host_int_list_during_forward(value: Any, start: int, end: int) -> Optional[list[int]]:
    if torch.cuda.is_current_stream_capturing():
        return None
    return _host_int_list(value, start, end)


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


def _get_fp4_mla_kv_cache_tensors(
    metadata: Any, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
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
    latent_cache: torch.Tensor,
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
    gen_token_lens = _host_int_list_during_forward(
        getattr(metadata, "prompt_lens_cpu_runtime", None), num_contexts, num_seqs
    )
    if gen_token_lens is not None:
        expected_num_tokens = sum(gen_token_lens)
        if num_tokens != expected_num_tokens:
            raise RuntimeError(
                "FP4 MLA 2D generation scatter token count mismatch: "
                f"expected {expected_num_tokens} generation tokens from "
                f"per-sequence lengths {gen_token_lens}, got {num_tokens}."
            )
        if min(gen_token_lens) != max(gen_token_lens):
            raise NotImplementedError(
                "FP4 MLA no-dequant generation scatter currently supports "
                f"uniform linear MTP lengths only, got {gen_token_lens}."
            )
    elif num_tokens < num_gen:
        raise RuntimeError(
            f"FP4 MLA 2D generation scatter needs at least {num_gen} generation "
            f"tokens, got {num_tokens}."
        )
    elif num_tokens % num_gen != 0:
        raise NotImplementedError(
            "FP4 MLA no-dequant generation scatter requires a uniform "
            f"generation length, got {num_tokens} tokens for {num_gen} sequences."
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

    max_gen_len = max(gen_token_lens) if gen_token_lens is not None else num_tokens // num_gen
    max_gen_tiles = _ceil_div(max_gen_len + HP_BLOCK_SIZE - 1, HP_BLOCK_SIZE)
    page_ids = metadata.paged_kv_indices[metadata.num_context_blocks :]
    _fp4_mla_v_scale_store_generation_tiles_kernel[
        (
            num_gen,
            max(max_gen_tiles, 1),
            num_dim_blocks,
        )
    ](
        kv_cache,
        sf_cache,
        v_sf,
        pool,
        latent_cache,
        global_scale,
        metadata.seq_slots[num_contexts:num_seqs],
        metadata.kv_lens_cuda_runtime[num_contexts:num_seqs],
        metadata.prompt_lens_cuda_runtime[num_contexts:num_seqs],
        page_ids,
        metadata.paged_kv_indptr_decode,
        page_ids.shape[0],
        metadata.paged_kv_indptr_decode.shape[0],
        pool.shape[0],
        v_sf.shape[1],
        v_sf.shape[0],
        local_layer,
        metadata.page_size,
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        pool.stride(0),
        pool.stride(1),
        latent_cache.stride(0),
        latent_cache.stride(1),
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


def _check_fp4_mla_page_generation_single_page(
    metadata: Any,
    num_contexts: int,
    num_seqs: int,
    page_size: int,
) -> None:
    """Raise if a step writes more than one generation token per sequence.

    The per-page re-quant v1 re-quantizes a single active page per step from the
    staging buffer (which holds exactly one page). A linear-MTP draft of length
    > 1 can straddle a page boundary, which would require the previous
    (completing) page to be re-quantized too -- not yet supported. 1-token
    decode never crosses, so only MTP drafts are rejected. Detectable only in
    eager mode (host metadata available); under CUDA graph the captured shape
    is assumed to be 1-token decode, so do not enable the per-page path together
    with MTP + CUDA graph until the multi-page store lands.
    """
    gen_token_lens = _host_int_list_during_forward(
        getattr(metadata, "prompt_lens_cpu_runtime", None), num_contexts, num_seqs
    )
    if gen_token_lens is None:
        return
    max_gen_len = max(gen_token_lens) if gen_token_lens else 1
    if max_gen_len > 1:
        raise NotImplementedError(
            "FP4 MLA per-page dynamic scale (v1) supports 1-token decode only; "
            f"got a generation length of {max_gen_len} (linear MTP). Multi-page "
            "re-quant for MTP drafts is a follow-up."
        )


def _store_fp4_mla_page_dynamic_generation(
    metadata: Any,
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    page_scale_pool: torch.Tensor,
    stage_pool: torch.Tensor,
    *,
    local_layer: int,
    v_head_dim: int,
    head_dim: int,
    sf_per_token: int,
    sf_per_page: int,
) -> None:
    """Re-quantize the active decode page with an exact per-page global scale.

    Two passes (see ``fp4_mla_triton``): Pass A computes the shared K/V
    page amax -> ``page_gscale`` into ``page_scale_pool``; Pass B re-quantizes
    every FP4 tile of the active page from ``stage_pool`` (old tokens) +
    ``latent_cache`` (new tokens), baking ``page_gscale`` into the K and V block
    scales. Replaces the static 16-token tile scatter for the ``triton``
    path when the per-page scale is enabled.
    """
    from .fp4_mla_triton import _fp4_mla_page_requant_gen_kernel, _fp4_mla_page_scale_gen_kernel

    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    num_gen = num_seqs - num_contexts
    if num_gen <= 0:
        return

    page_size = metadata.page_size
    _check_fp4_mla_page_generation_single_page(metadata, num_contexts, num_seqs, page_size)

    pool_head_dim = stage_pool.shape[-1] // page_size
    if pool_head_dim < head_dim:
        raise RuntimeError(
            f"FP4 MLA staging pool head dim {pool_head_dim} < latent head_dim {head_dim}."
        )

    page_ids = metadata.paged_kv_indices[metadata.num_context_blocks :]
    seq_slots = metadata.seq_slots[num_contexts:num_seqs]
    kv_lens = metadata.kv_lens_cuda_runtime[num_contexts:num_seqs]
    gen_lens = metadata.prompt_lens_cuda_runtime[num_contexts:num_seqs]
    indptr = metadata.paged_kv_indptr_decode
    num_dim_blocks = triton.cdiv(head_dim, FP4_BLOCK_SIZE)
    tiles_per_page = page_size // FP4_BLOCK_SIZE
    block_d = triton.next_power_of_2(head_dim)

    # Pass A: per-page (K/V shared) amax -> page_gscale.
    _fp4_mla_page_scale_gen_kernel[(num_gen,)](
        page_scale_pool,
        stage_pool,
        latent_cache,
        seq_slots,
        kv_lens,
        gen_lens,
        page_ids,
        indptr,
        page_ids.shape[0],
        indptr.shape[0],
        stage_pool.shape[0],
        kv_cache.shape[0],
        page_scale_pool.shape[0],
        local_layer,
        page_size,
        page_scale_pool.stride(0),
        stage_pool.stride(0),
        stage_pool.stride(1),
        latent_cache.stride(0),
        latent_cache.stride(1),
        HEAD_D=head_dim,
        POOL_HEAD_D=pool_head_dim,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        PAGE_SLOTS=page_size,
        BLOCK_D=block_d,
        P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
    )
    _debug_sync("fp4_mla_page_scale_gen")

    # Pass B: re-quantize every tile of the active page with page_gscale.
    _fp4_mla_page_requant_gen_kernel[(num_gen, tiles_per_page, num_dim_blocks)](
        kv_cache,
        sf_cache,
        v_sf,
        stage_pool,
        latent_cache,
        page_scale_pool,
        seq_slots,
        kv_lens,
        gen_lens,
        page_ids,
        indptr,
        page_ids.shape[0],
        indptr.shape[0],
        stage_pool.shape[0],
        kv_cache.shape[0],
        page_scale_pool.shape[0],
        local_layer,
        page_size,
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        stage_pool.stride(0),
        stage_pool.stride(1),
        latent_cache.stride(0),
        latent_cache.stride(1),
        v_sf.stride(0),
        v_sf.stride(1),
        page_scale_pool.stride(0),
        HEAD_D=head_dim,
        POOL_HEAD_D=pool_head_dim,
        V_HEAD_D=v_head_dim,
        PAGE_SLOTS=page_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_TOKEN=sf_per_token,
        SF_PER_PAGE=sf_per_page,
    )
    _debug_sync("fp4_mla_page_requant_gen")


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
    dimensions use K's per-token 1D scales.  Generation scatter rewrites each
    touched 16-token tile by reading old tokens from the HP pool and new tokens
    from ``latent_cache``; callers then update the HP pool after scatter.
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
    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
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
            page_scale_pool = getattr(metadata, "fp4_mla_page_scale_pool", None)
            stage_pool = getattr(metadata, "fp4_mla_page_stage_pool", None)
            if (
                fp4_mla_per_page_scale_enabled()
                and page_scale_pool is not None
                and stage_pool is not None
            ):
                # Per-page dynamic scale: re-quantize the active page from the
                # BF16 staging buffer with its exact per-page global scale.
                # NOTE: the staging buffer must hold this step's *pre-update*
                # tokens, so update_page_stage_for_fp4_mla must run AFTER this.
                _store_fp4_mla_page_dynamic_generation(
                    metadata,
                    latent_cache,
                    kv_cache,
                    sf_cache,
                    v_sf,
                    page_scale_pool,
                    stage_pool,
                    local_layer=local_layer,
                    v_head_dim=v_head_dim,
                    head_dim=head_dim,
                    sf_per_token=sf_per_token,
                    sf_per_page=sf_per_page,
                )
            else:
                _scatter_fp4_mla_kv_cache_2d_generation(
                    metadata,
                    latent_cache,
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
        if phase == "context":
            v_pack_page_ids = metadata.paged_kv_indices
        else:
            v_pack_page_ids = metadata.paged_kv_indices[metadata.num_context_blocks :]
        _maybe_update_cutile_v_packed_cache(
            metadata,
            layer_idx,
            kv_cache,
            v_pack_page_ids,
            v_head_dim=v_head_dim,
            page_size=metadata.page_size,
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


def _assert_fp4_mla_decode_paging_consistent(
    metadata: Any,
    kv_lens: torch.Tensor,
    num_gen_blocks: int,
    query_len_per_seq: int,
) -> None:
    """Flag read-side decode paging that diverges from the corrected kv_lens.

    With the overlap scheduler the generation metadata is first built from the
    all-draft-accepted over-estimate; ``kv_lens_cuda_runtime`` is then corrected
    in ``_preprocess_inputs`` and ``positions`` / ``batch_indices`` rebuilt. The
    page-table side -- ``num_blocks`` / ``num_generation_blocks`` /
    ``paged_kv_indptr_decode`` and the ``positions`` of the new tokens -- is what
    the decode kernels index with. This check raises on the first decode where
    any of those is inconsistent with the corrected kv_lens, so we can tell
    whether stale paging (rather than masking) corrupts the read. Host-syncs;
    gated by ``TRTLLM_FP4_MLA_DEBUG_ASSERT`` and skipped under CUDA-graph capture.
    """
    if not _env_enabled(FP4_MLA_DEBUG_ASSERT_ENV):
        return
    if torch.cuda.is_current_stream_capturing():
        return
    # Warmup builds synthetic metadata with dummy kv_lens but no real page-table
    # allocation (num_generation_blocks==0), so the consistency check does not
    # apply there.
    if getattr(metadata, "is_warmup", False):
        return
    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    num_gen = num_seqs - num_contexts
    if num_gen <= 0:
        return
    # Skip the benign context->generation reclassification state: the MTP draft
    # loop resets num_contexts to 0 without rebuilding the decode page table
    # (for FlashInfer the reorder is gated on enable_flash_mla), leaving
    # num_context_blocks > 0 / stale num_generation_blocks. That forward's
    # attention is degraded but rejected drafts fall back to the golden token,
    # so it does NOT affect output accuracy (it fires with overlap off too).
    # Only check pure steady-state generation, where a paging bug would
    # genuinely corrupt the committed output.
    if int(getattr(metadata, "num_context_blocks", 0)) != 0:
        return
    # Only the MTP *target* verification forward (query_len_per_seq > 1)
    # determines the accepted/committed tokens, so it is the only forward whose
    # paging staleness can change output accuracy. The draft-model forwards
    # (query_len_per_seq == 1) read a separate KV layer and their bad output is
    # rejected (and they mismatch in both overlap modes), so skip them here to
    # isolate the accuracy-relevant path.
    if query_len_per_seq <= 1:
        return

    page_size = metadata.page_size
    kv_lens_host = kv_lens.detach().to("cpu", torch.int64).tolist()
    expected_blocks = [_ceil_div(kv, page_size) for kv in kv_lens_host]
    expected_gen_blocks = sum(expected_blocks)
    problems: list[str] = []

    num_blocks = getattr(metadata, "num_blocks", None)
    if num_blocks is not None:
        actual_blocks = [int(b) for b in num_blocks[num_contexts:num_seqs]]
        if actual_blocks != expected_blocks:
            problems.append(f"per-seq num_blocks {actual_blocks} != expected {expected_blocks}")
    if num_gen_blocks != expected_gen_blocks:
        problems.append(f"num_generation_blocks={num_gen_blocks} != expected {expected_gen_blocks}")

    indptr = getattr(metadata, "paged_kv_indptr_decode", None)
    if indptr is not None:
        indptr_host = indptr[: num_gen + 1].detach().to("cpu", torch.int64).tolist()
        expected_indptr = [0]
        for b in expected_blocks:
            expected_indptr.append(expected_indptr[-1] + b)
        if indptr_host != expected_indptr:
            problems.append(f"paged_kv_indptr_decode {indptr_host} != expected {expected_indptr}")

    positions = getattr(metadata, "positions", None)
    prompt_lens = getattr(metadata, "prompt_lens_cuda_runtime", None)
    if positions is not None and prompt_lens is not None:
        num_ctx_tokens = int(getattr(metadata, "num_ctx_tokens", 0))
        gen_pos = positions[num_ctx_tokens:].detach().to("cpu", torch.int64).tolist()
        pls = prompt_lens[num_contexts:num_seqs].detach().to("cpu", torch.int64).tolist()
        off = 0
        for s, (kv_len, prompt_len) in enumerate(zip(kv_lens_host, pls)):
            expected = list(range(kv_len - prompt_len, kv_len))
            got = gen_pos[off : off + prompt_len]
            if got != expected:
                problems.append(
                    f"seq{s} gen positions {got} != expected {expected} "
                    f"(kv_len={kv_len}, prompt_len={prompt_len})"
                )
            off += prompt_len

    if problems:
        ctx = (
            f"num_contexts={num_contexts}, num_seqs={num_seqs}, "
            f"num_generations={getattr(metadata, 'num_generations', '?')}, "
            f"query_len_per_seq={query_len_per_seq}, "
            f"num_context_blocks={getattr(metadata, 'num_context_blocks', '?')}, "
            f"num_blocks={getattr(metadata, 'num_blocks', '?')}, "
            f"use_spec_decoding={getattr(metadata, 'use_spec_decoding', '?')}, "
            f"is_spec_dec_mode={getattr(metadata, 'is_spec_dec_mode', '?')}, "
            f"kv_lens={kv_lens_host}"
        )
        msg = (
            "FP4 MLA decode paging inconsistent with corrected kv_lens "
            f"({ctx}):\n  " + "\n  ".join(problems)
        )
        if _env_enabled("TRTLLM_FP4_MLA_DEBUG_ASSERT_WARN"):
            logger.warning(msg)
            return
        raise AssertionError(msg)


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

    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
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


def _cutile_persistent_v_pack_enabled() -> bool:
    if _fp4_mla_attention_backend() != "cutile":
        return False
    return os.getenv("TRTLLM_FP4_MLA_PERSISTENT_V_PACK", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _cutile_v_packed_attr(layer_idx: int) -> str:
    return f"_fp4_mla_attention_v_packed_buf_l{layer_idx}"


def _cutile_v_packed_valid_attr(layer_idx: int) -> str:
    return f"_fp4_mla_attention_v_packed_valid_l{layer_idx}"


def _cutile_v_packed_shape(
    kv_cache: torch.Tensor,
    v_head_dim: int,
    page_size: int,
) -> tuple[int, int]:
    block_v = 128
    return (kv_cache.shape[0] * _ceil_div(v_head_dim, block_v) * block_v, page_size // 2)


def _maybe_update_cutile_v_packed_cache(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    page_ids: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
) -> None:
    if not _cutile_persistent_v_pack_enabled():
        return
    if v_head_dim % 128 != 0 or page_size != FP4_MLA_TOKENS_PER_BLOCK:
        return
    if page_ids.numel() == 0:
        return

    from .fp4_mla_cutile import fp4_mla_repack_v_cache

    attr_name = _cutile_v_packed_attr(layer_idx)
    v_packed = _ensure_workspace_tensor(
        metadata,
        attr_name,
        _cutile_v_packed_shape(kv_cache, v_head_dim, page_size),
        dtype=torch.uint8,
        device=kv_cache.device,
    )
    fp4_mla_repack_v_cache(
        v_packed,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        block_v=128,
    )
    setattr(metadata, _cutile_v_packed_valid_attr(layer_idx), True)


def _get_cutile_v_packed_cache(
    metadata: Any,
    layer_idx: int,
    kv_cache: torch.Tensor,
    *,
    v_head_dim: int,
    page_size: int,
) -> Optional[torch.Tensor]:
    if not _cutile_persistent_v_pack_enabled():
        return None
    if not bool(getattr(metadata, _cutile_v_packed_valid_attr(layer_idx), False)):
        return None
    v_packed = getattr(metadata, _cutile_v_packed_attr(layer_idx), None)
    expected_shape = _cutile_v_packed_shape(kv_cache, v_head_dim, page_size)
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


def _get_linear_mtp_query_len_per_seq(
    metadata: Any,
    *,
    num_queries: int,
    num_gen_seqs: int,
) -> int:
    """Return the uniform generation query length required by linear MTP."""
    if num_gen_seqs <= 0:
        return 1

    start = metadata.num_contexts
    end = metadata.num_seqs
    query_lens = _host_int_list_during_forward(
        getattr(metadata, "prompt_lens_cpu_runtime", None), start, end
    )
    if query_lens is None:
        query_lens = _host_int_list_during_forward(getattr(metadata, "seq_lens", None), start, end)

    if query_lens is None:
        if num_queries % num_gen_seqs != 0:
            raise NotImplementedError(
                "FP4 MLA linear MTP requires a uniform generation query length; "
                f"got {num_queries} query tokens for {num_gen_seqs} sequences."
            )
        return num_queries // num_gen_seqs

    if sum(query_lens) != num_queries and num_queries == num_gen_seqs:
        return 1
    if sum(query_lens) != num_queries:
        raise RuntimeError(
            "FP4 MLA generation query metadata does not match q shape: "
            f"query_lens={query_lens}, total={sum(query_lens)}, "
            f"q_tokens={num_queries}."
        )
    if not query_lens:
        return 1
    if min(query_lens) <= 0:
        raise RuntimeError(f"FP4 MLA generation query lengths must be positive, got {query_lens}.")
    if min(query_lens) != max(query_lens):
        raise NotImplementedError(
            "FP4 MLA no-dequant attention currently supports linear MTP with "
            f"a uniform generation length per sequence, got {query_lens}."
        )
    return query_lens[0]


def _run_triton_attention_decode(
    *,
    metadata: Any,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    kv_lens: torch.Tensor,
    p_fp4: torch.Tensor,
    p_sf: torch.Tensor,
    max_scores: torch.Tensor,
    denom: torch.Tensor,
    output: torch.Tensor,
    num_queries: int,
    num_heads: int,
    head_dim: int,
    kv_lora_rank: int,
    q_residual_dim: int,
    query_len_per_seq: int,
    max_pages: int,
    sm_scale: float,
    q_global_scale: Optional[torch.Tensor] = None,
    page_scale_pool: Optional[torch.Tensor] = None,
    local_layer: int = 0,
    use_per_page_scale: bool = False,
) -> None:
    """Dispatch the ``triton`` FP4 MLA decode pipeline.

    Mirrors the four-stage layout used by ``fp4_mla_cutile.py``
    (page-stats with packed P -> reduce-stats -> prob-scale -> PV) but
    routes through the self-contained kernels in
    ``fp4_mla_triton.py``. Threads through the constexpr assume flags,
    TMA descriptors, occupancy/num-warps launch meta, and pipelined PV loop.
    """
    from .fp4_mla_triton import _fp4_mla_attention_page_stats_kernel as _attn_page_stats_kernel
    from .fp4_mla_triton import _fp4_mla_attention_prob_scale_kernel as _attn_prob_scale_kernel
    from .fp4_mla_triton import _fp4_mla_attention_pv_kernel as _attn_pv_kernel
    from .fp4_mla_triton import _fp4_mla_attention_pv_reduce_kernel as _attn_pv_reduce_kernel
    from .fp4_mla_triton import _fp4_mla_attention_reduce_stats_kernel as _attn_reduce_stats_kernel

    block_h = 128
    block_t = metadata.page_size
    # Adaptive BLOCK_V: small batches need a finer V split to fill enough waves
    # on B200 (~148 SMs). PV grid = num_queries * num_head_blocks(1) *
    # (kv_lora_rank / BLOCK_V). We want >= ~2*num_SMs programs so that >1 CTA
    # lands per SM and hides the L1TEX scoreboard stalls. Empirically (sweep):
    #   bs<=32 -> BLOCK_V=32; bs>=64 -> BLOCK_V=128.
    # (BLOCK_V=16 is rejected by the V TMA descriptor min-stride requirement.)
    block_v = 32 if num_queries <= 32 else 128
    q_head_dim = head_dim + q_residual_dim
    # BLOCK_K = 512 matches cutile's "nvt" backend default and aligns the K-window
    # with the residual-Q boundary (Q_HEAD_D = 640 = 512 + 128 tail). The
    # residual-Q TMA tail path requires Q_RESIDUAL_D == 64 and TAIL_BLOCK_K == 128.
    block_k = 512
    full_block_end = (q_head_dim // block_k) * block_k
    tail_k = q_head_dim - full_block_end
    tail_block_k = 1 << (tail_k - 1).bit_length() if tail_k > 0 else block_k
    q_sf_per_token = q_head_dim // FP4_BLOCK_SIZE
    k_sf_per_token = head_dim // FP4_BLOCK_SIZE
    sf_per_page = metadata.page_size // FP4_BLOCK_SIZE
    num_head_blocks = triton.cdiv(num_heads, block_h)

    assume_full_heads = num_heads % block_h == 0
    assume_full_v = kv_lora_rank % block_v == 0
    # Match the cutile path: only mark pages "full" when we can prove every
    # generation sequence has the same number of cached tokens AND
    # query_len_per_seq == 1 (so the kv_len adjustment is a no-op).
    assume_full_pages = (
        _infer_cutile_assume_full_pages(metadata, max_pages, metadata.page_size)
        and query_len_per_seq == 1
    )
    # Leave validity checks on. Matches cutile's default and is correctness-
    # safe. The perfect-shape PV fast path (tl.ext.make_view + load_view_tko)
    # remains gated off — when measured on the TileIR backend (ENABLE_TILE=1)
    # it was net-slower on the bench, so the cost of enabling it isn't worth
    # the win on the FP4 MLA shapes we care about.
    assume_valid_pages = False
    # cutile checks only `make_tensor_descriptor`; on the nvt backend the
    # presence of TMA descriptors implies `tl.ext.make_view` is available too.
    use_tma_data_load = hasattr(triton.language, "make_tensor_descriptor")

    # Install the device-side scratch allocator on every call. Triton stores
    # the allocator in a ContextVar (triton.runtime._allocation), so a single
    # process-wide install is not visible from worker threads / asyncio tasks
    # that run with a different Context — the kernel launch would then hit the
    # default NullAllocator and raise. Matches the cutile path.
    if use_tma_data_load:

        def _tma_alloc(size: int, alignment: int, stream):
            return torch.empty(size, device=q_fp4.device, dtype=torch.int8)

        triton.set_allocator(_tma_alloc)

    # cutile-equivalent launch meta. occupancy=2 lets two CTAs land per SM
    # which improves wave-tail efficiency at the bs=32 hot point.
    # NOTE: num_stages=2 (instead of the Triton 3.6 default of 3) sidesteps
    # the TritonGPUAutomaticWarpSpecialization + NVWSInsertTmemAref pass that
    # ICEs on the page_stats kernel under Triton 3.6.0 / sm_100.
    launch_meta = {"occupancy": 2}
    # PV loop pipelining. With TMA loads, num_stages>=2 lets the next page's
    # loads overlap with the current MMA via mbarrier. The PV report shows
    # long_scoreboard=4.5 cycles avg on V loads at PV_LOOP_STAGES=2; bumping the
    # depth pays off when the grid is small enough that occupancy can absorb
    # the extra in-flight tile state — i.e. medium batch / large max_pages.
    # Larger pipelines hurt at small batch (more live state, fewer dim blocks).
    if num_queries <= 16 or max_pages <= 4:
        pv_loop_stages = 2
    else:
        pv_loop_stages = 3

    # Page-stats kernel: per (query, head_block, page) program, does QK,
    # softmax stats, and packs probs into FP4 with the per-page local-max
    # scaling trick. The page-max correction is applied later by
    # prob_scale_kernel via p_sf in-place rescaling.
    page_stats_shape = (num_queries, max_pages, num_heads)
    page_max = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_page_max_buf",
        page_stats_shape,
        dtype=torch.float32,
        device=q_fp4.device,
    )
    page_sum = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_page_sum_buf",
        page_stats_shape,
        dtype=torch.float32,
        device=q_fp4.device,
    )

    pack_prob_in_page_stats = True
    # Per-page scale pointers. When the per-page path is off, pass the static
    # global_scale tensor as harmless dummies (the kernel never dereferences
    # them under USE_PER_PAGE_SCALE=False) and a zero layer stride.
    page_stats_q_gscale = q_global_scale if use_per_page_scale else global_scale
    page_stats_page_scale = (
        page_scale_pool if (use_per_page_scale and page_scale_pool is not None) else global_scale
    )
    page_stats_pscale_s0 = (
        page_scale_pool.stride(0) if (use_per_page_scale and page_scale_pool is not None) else 0
    )
    _attn_page_stats_kernel[(num_queries, num_head_blocks, max_pages)](
        page_max,
        page_sum,
        p_fp4,
        p_sf,
        q_fp4,
        q_sf,
        kv_cache,
        sf_cache,
        global_scale,
        page_stats_q_gscale,
        page_stats_page_scale,
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
        page_max.stride(0),
        page_max.stride(1),
        p_fp4.stride(0),
        p_fp4.stride(1),
        p_fp4.shape[0],
        q_fp4.shape[0],
        sm_scale,
        local_layer,
        page_stats_pscale_s0,
        NUM_HEADS=num_heads,
        Q_HEAD_D=q_head_dim,
        K_HEAD_D=head_dim,
        Q_RESIDUAL_D=q_residual_dim,
        PAGE_SIZE=metadata.page_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        Q_SF_PER_TOKEN=q_sf_per_token,
        K_SF_PER_TOKEN=k_sf_per_token,
        SF_PER_PAGE=sf_per_page,
        P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
        QUERY_LEN_PER_SEQ=query_len_per_seq,
        MAX_PAGES=max_pages,
        BLOCK_H=block_h,
        BLOCK_T=block_t,
        BLOCK_K=block_k,
        FULL_BLOCK_END=full_block_end,
        TAIL_BLOCK_K=tail_block_k,
        USE_TMA_DATA_LOAD=use_tma_data_load,
        PACK_PROBS=pack_prob_in_page_stats,
        ASSUME_FULL_HEADS=assume_full_heads,
        ASSUME_FULL_PAGES=assume_full_pages,
        ASSUME_VALID_PAGES=assume_valid_pages,
        USE_PER_PAGE_SCALE=use_per_page_scale,
        **launch_meta,
    )

    _attn_reduce_stats_kernel[(num_queries, num_head_blocks)](
        max_scores,
        denom,
        page_max,
        page_sum,
        max_pages,
        max_scores.stride(0),
        page_max.stride(0),
        page_max.stride(1),
        NUM_HEADS=num_heads,
        MAX_PAGES=max_pages,
        BLOCK_H=block_h,
        **launch_meta,
    )

    _attn_prob_scale_kernel[(num_queries, num_head_blocks, max_pages)](
        p_sf,
        max_scores,
        denom,
        page_max,
        metadata.paged_kv_indptr_decode,
        kv_lens,
        src_page_ids.shape[0],
        max_scores.stride(0),
        page_max.stride(0),
        page_max.stride(1),
        NUM_HEADS=num_heads,
        PAGE_SIZE=metadata.page_size,
        SF_PER_PAGE=sf_per_page,
        QUERY_LEN_PER_SEQ=query_len_per_seq,
        MAX_PAGES=max_pages,
        BLOCK_H=block_h,
        ASSUME_FULL_HEADS=assume_full_heads,
        ASSUME_FULL_PAGES=assume_full_pages,
        ASSUME_VALID_PAGES=assume_valid_pages,
        **launch_meta,
    )

    num_dim_blocks = triton.cdiv(kv_lora_rank, block_v)

    # PV page split: partition the page range across additional programs and
    # reduce in a follow-up kernel. ncu showed PV at waves/SM=0.49 for bs=32 —
    # PV is L1-bandwidth bound, so raising in-flight CTAs is the lever.
    # BLOCK_V is bounded below by the 16-byte TMA descriptor min-stride.
    # PV page split: ncu shows that with the current shape (bs=32, max_pages=256)
    # the PV kernel is L1-cache-throughput bound (long_scoreboard=4.5 cycles
    # avg, L1 global LD hit-rate <40%). Increasing the program count via page
    # splitting reduced waves/SM idle time but did NOT improve wall-time at
    # current shapes — the per-CTA L1 thrash is the limit. Gate the split off
    # by default; re-enable only for very small grids where occupancy is the
    # bottleneck rather than per-CTA L1 pressure.
    page_split = 1
    base_grid = num_queries * num_head_blocks * num_dim_blocks
    if max_pages >= 16 and base_grid < 148:
        for p in (8, 4, 2):
            if max_pages % p == 0 and max_pages // p >= 16 and base_grid * p <= 148 * 4:
                page_split = p
                break
    if page_split > 1:
        pages_per_split = max_pages // page_split
        partial_out = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_pv_partial_buf",
            (num_queries, page_split, num_heads, kv_lora_rank),
            dtype=torch.float32,
            device=q_fp4.device,
        )
        _attn_pv_kernel[(num_queries, num_head_blocks, num_dim_blocks * page_split)](
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
            output.shape[0] * output.shape[1],
            p_fp4.stride(0),
            p_fp4.stride(1),
            p_fp4.shape[0],
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(4),
            v_sf.stride(0),
            NUM_HEADS=num_heads,
            V_HEAD_D=kv_lora_rank,
            PAGE_SIZE=metadata.page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            SF_PER_PAGE=sf_per_page,
            QUERY_LEN_PER_SEQ=query_len_per_seq,
            MAX_PAGES=max_pages,
            P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
            BLOCK_H=block_h,
            BLOCK_V=block_v,
            USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
            USE_TMA_V_LOAD=use_tma_data_load and kv_lora_rank % block_v == 0,
            PV_LOOP_STAGES=pv_loop_stages,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            ASSUME_FULL_V=assume_full_v,
            ASSUME_VALID_PAGES=assume_valid_pages,
            PAGE_SPLIT=page_split,
            PAGES_PER_SPLIT=pages_per_split,
            PARTIAL_OUT=True,
            partial_out_ptr=partial_out,
            partial_s0=partial_out.stride(0),
            partial_s1=partial_out.stride(1),
            partial_s2=partial_out.stride(2),
            partial_s3=partial_out.stride(3),
            **launch_meta,
        )
        _attn_pv_reduce_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
            output,
            partial_out,
            global_scale,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            NUM_HEADS=num_heads,
            V_HEAD_D=kv_lora_rank,
            PAGE_SPLIT=page_split,
            P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
            BLOCK_H=block_h,
            BLOCK_V=block_v,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_V=assume_full_v,
            **launch_meta,
        )
    else:
        _attn_pv_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
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
            output.shape[0] * output.shape[1],
            p_fp4.stride(0),
            p_fp4.stride(1),
            p_fp4.shape[0],
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(4),
            v_sf.stride(0),
            NUM_HEADS=num_heads,
            V_HEAD_D=kv_lora_rank,
            PAGE_SIZE=metadata.page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            SF_PER_PAGE=sf_per_page,
            QUERY_LEN_PER_SEQ=query_len_per_seq,
            MAX_PAGES=max_pages,
            P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
            BLOCK_H=block_h,
            BLOCK_V=block_v,
            USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
            USE_TMA_V_LOAD=use_tma_data_load and kv_lora_rank % block_v == 0,
            PV_LOOP_STAGES=pv_loop_stages,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            ASSUME_FULL_V=assume_full_v,
            ASSUME_VALID_PAGES=assume_valid_pages,
            **launch_meta,
        )


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

    num_queries = q_nope.shape[0]
    if num_queries == 0:
        return
    num_gen_seqs = metadata.num_seqs - metadata.num_contexts
    query_len_per_seq = _get_linear_mtp_query_len_per_seq(
        metadata,
        num_queries=num_queries,
        num_gen_seqs=num_gen_seqs,
    )

    num_heads = q_nope.shape[1]
    if q_pe.shape[:2] != (num_queries, num_heads):
        raise ValueError("FP4 MLA attention q_nope/q_pe batch dimensions do not match.")
    if output.shape[:2] != (num_queries, num_heads):
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
        (num_queries, num_heads, head_dim),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )
    q_full[..., :kv_lora_rank].copy_(q_nope)
    q_full[..., kv_lora_rank:].copy_(q_pe)
    q_2d = q_full.reshape(num_queries * num_heads, head_dim)
    if q_2d.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(
            f"FP4 MLA residual Q quantization requires BF16 or FP8 Q; got {q_2d.dtype}."
        )
    # Per-page path: Q gets its own per-step dynamic global scale (independent of
    # the KV per-page scale); the dev QK kernel divides by q_gscale * page_gscale.
    page_scale_pool = getattr(metadata, "fp4_mla_page_scale_pool", None)
    use_per_page_scale = fp4_mla_per_page_scale_enabled() and page_scale_pool is not None
    if use_per_page_scale:
        q_amax = q_2d.to(torch.float32).abs().amax().reshape(1)
        q_global_scale = torch.where(
            q_amax > 0.0,
            torch.full_like(q_amax, FP4_MLA_P_GLOBAL_SCALE) / q_amax,
            torch.ones_like(q_amax),
        )
    else:
        q_global_scale = global_scale
    q_fp4, q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
        q_2d,
        q_global_scale,
        q_residual_dim,
        is_act=True,
    )
    q_sf = q_sf.view(torch.float8_e4m3fn)

    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
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

    _assert_fp4_mla_decode_paging_consistent(metadata, kv_lens, num_gen_blocks, query_len_per_seq)

    backend = _fp4_mla_attention_backend()
    if backend == "cutile":
        from .fp4_mla_cutile import fp4_mla_paged_attention

        total_p_rows = num_queries * max_pages * num_heads
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
        stats_shape = (num_queries, num_heads)
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
            page_stats_shape = (num_queries, max_pages, num_heads)
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
        assume_full_pages = (
            _infer_cutile_assume_full_pages(
                metadata,
                max_pages,
                metadata.page_size,
            )
            and query_len_per_seq == 1
        )
        assume_valid_pages = False
        cutile_block_h = _env_int("TRTLLM_FP4_MLA_BLOCK_H") or 128
        cutile_block_v = _env_int("TRTLLM_FP4_MLA_BLOCK_V") or 128
        cutile_num_gen_seqs = num_queries // query_len_per_seq
        cutile_assume_valid_pages = assume_valid_pages or (
            assume_full_pages and src_page_ids.numel() == cutile_num_gen_seqs * max_pages
        )
        cutile_auto_prepack_v = (
            hasattr(tl, "make_tensor_descriptor")
            and num_heads % cutile_block_h == 0
            and assume_full_pages
            and cutile_assume_valid_pages
            and kv_lora_rank == 512
            and metadata.page_size == FP4_MLA_TOKENS_PER_BLOCK
            and cutile_block_h in (64, 128)
            and cutile_block_v == 128
            and metadata.page_size // FP4_BLOCK_SIZE == 8
        )
        cutile_prepack_v_env = os.environ.get("TRTLLM_FP4_MLA_PREPACK_V")
        cutile_prepack_v_for_pv = (
            cutile_auto_prepack_v
            if cutile_prepack_v_env is None
            else cutile_prepack_v_env == "1" and cutile_auto_prepack_v
        )
        v_packed = (
            _get_cutile_v_packed_cache(
                metadata,
                layer_idx,
                kv_cache,
                v_head_dim=kv_lora_rank,
                page_size=metadata.page_size,
            )
            if cutile_auto_prepack_v
            else None
        )
        use_cutile_v_packed_cache = v_packed is not None
        if use_cutile_v_packed_cache:
            cutile_prepack_v_for_pv = False
        elif cutile_prepack_v_for_pv:
            v_packed = _ensure_workspace_tensor(
                metadata,
                "_fp4_mla_attention_v_packed_buf",
                (
                    kv_cache.shape[0] * triton.cdiv(kv_lora_rank, cutile_block_v) * cutile_block_v,
                    metadata.page_size // 2,
                ),
                dtype=torch.uint8,
                device=q_nope.device,
            )
        _fp4_mla_debug(
            "attention decode cutile launch: "
            f"num_queries={num_queries} query_len_per_seq={query_len_per_seq} "
            f"num_heads={num_heads} local_layer={local_layer} "
            f"layer_idx={layer_idx} head_dim={head_dim} kv_lora_rank={kv_lora_rank} "
            f"rope_dim={qk_rope_head_dim} max_pages={max_pages} "
            f"assume_full_pages={assume_full_pages} "
            f"assume_valid_pages={assume_valid_pages} "
            f"use_v_packed_cache={use_cutile_v_packed_cache}"
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
            query_len_per_seq=query_len_per_seq,
            assume_full_pages=assume_full_pages,
            assume_valid_pages=assume_valid_pages,
            prepack_v_for_pv=cutile_prepack_v_for_pv,
            use_prepacked_v_for_pv=use_cutile_v_packed_cache,
            p_fp4_workspace=p_fp4,
            p_sf_workspace=p_sf,
            v_packed_workspace=v_packed,
            max_scores_workspace=max_scores,
            denom_workspace=denom,
            page_max_workspace=page_max,
            page_sum_workspace=page_sum,
        )
        _debug_sync("attention_cutile")
        return

    total_p_rows = num_queries * max_pages * num_heads
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
    stats_shape = (num_queries, num_heads)
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

    if backend != "triton":
        raise ValueError(
            f"Unsupported FP4 MLA attention backend '{backend}'. "
            f"Set {FLASHINFER_FP4_MLA_ATTENTION_BACKEND_ENV} to "
            "'triton' or 'cutile'."
        )

    # Self-contained triton path. Uses kernels copied from the cutile reference
    # into fp4_mla_triton.py: TMA-loaded QK + fused page-stats pack,
    # reduce-stats, prob-scale, and a specialized PV with tl.ext views.
    if backend == "triton":
        _run_triton_attention_decode(
            metadata=metadata,
            q_fp4=q_fp4,
            q_sf=q_sf.contiguous().view(-1),
            kv_cache=kv_cache,
            sf_cache=sf_cache,
            v_sf=v_sf,
            global_scale=global_scale,
            src_page_ids=src_page_ids,
            kv_lens=kv_lens,
            p_fp4=p_fp4,
            p_sf=p_sf,
            max_scores=max_scores,
            denom=denom,
            output=output,
            num_queries=num_queries,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_lora_rank=kv_lora_rank,
            q_residual_dim=q_residual_dim,
            query_len_per_seq=query_len_per_seq,
            max_pages=max_pages,
            sm_scale=float(sm_scale),
            q_global_scale=q_global_scale,
            page_scale_pool=page_scale_pool,
            local_layer=local_layer,
            use_per_page_scale=use_per_page_scale,
        )
        _debug_sync("attention_triton")
        return


def _hp_pool_layer_view(
    pool: torch.Tensor,
    local_layer: int,
    pool_head_dim: int,
) -> torch.Tensor:
    return pool[:, local_layer, 0, :].view(pool.shape[0], HP_BLOCK_SIZE, pool_head_dim)


def _snapshot_hp_kv_for_mtp_generation(
    metadata: Any,
    pool: torch.Tensor,
    local_layer: int,
    *,
    num_gen: int,
    num_gen_tokens: int,
    max_gen_len: int,
    metadata_token_offset: int,
    head_dim: int,
    pool_head_dim: int,
) -> None:
    if getattr(metadata, "is_warmup", False):
        return
    if num_gen_tokens <= num_gen:
        return
    if max_gen_len > HP_BLOCK_SIZE:
        raise NotImplementedError(
            "FP4 MLA HP-pool rollback for linear MTP supports at most "
            f"{HP_BLOCK_SIZE} generation tokens per sequence, got {max_gen_len}."
        )

    end_token_offset = metadata_token_offset + num_gen_tokens
    if (
        end_token_offset > metadata.batch_indices.shape[0]
        or end_token_offset > metadata.positions.shape[0]
    ):
        raise RuntimeError(
            "FP4 MLA HP-pool snapshot would read past generation metadata: "
            f"token_offset={metadata_token_offset}, num_gen_tokens={num_gen_tokens}, "
            f"batch_indices={metadata.batch_indices.shape[0]}, "
            f"positions={metadata.positions.shape[0]}."
        )

    snapshots = getattr(metadata, _FP4_MLA_MTP_HP_SNAPSHOTS, None)
    if snapshots is None:
        snapshots = {}
        setattr(metadata, _FP4_MLA_MTP_HP_SNAPSHOTS, snapshots)

    snapshot_pool = getattr(metadata, "fp4_mla_hp_snapshot_pool", None)
    if snapshot_pool is not None:
        snapshot_pool[:, local_layer, :, :].copy_(pool[:, local_layer, :, :])
        snapshots[int(local_layer)] = {
            "mode": "pool",
            "metadata_token_offset": metadata_token_offset,
            "num_gen_tokens": num_gen_tokens,
            "head_dim": head_dim,
            "pool_head_dim": pool_head_dim,
        }
        return

    device = pool.device
    token_indices = torch.arange(
        metadata_token_offset,
        end_token_offset,
        dtype=torch.long,
        device=device,
    )
    batch_indices = metadata.batch_indices[token_indices].to(torch.long)
    positions = metadata.positions[token_indices].to(torch.long)
    seq_slots = metadata.seq_slots[batch_indices].to(torch.long)
    hp_slots = torch.remainder(positions, HP_BLOCK_SIZE).to(torch.long)
    first_new_positions = metadata.kv_lens_cuda_runtime[batch_indices].to(
        torch.long
    ) - metadata.prompt_lens_cuda_runtime[batch_indices].to(torch.long)
    pool_view = _hp_pool_layer_view(pool, local_layer, pool_head_dim)
    values = pool_view[seq_slots, hp_slots, :head_dim].clone()

    snapshots[int(local_layer)] = {
        "mode": "values",
        "batch_indices": batch_indices,
        "seq_slots": seq_slots,
        "hp_slots": hp_slots,
        "positions": positions,
        "first_new_positions": first_new_positions,
        "values": values,
        "head_dim": head_dim,
        "pool_head_dim": pool_head_dim,
    }


def repair_fp4_mla_hp_kv_for_mtp_rejection(
    metadata: Any,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Restore HP-pool slots that belonged to rejected linear-MTP tokens.

    Packed FP4 pages past the accepted logical KV length are harmless because
    later attention ignores them. The BF16 HP pool is a circular tail mirror, so
    rejected speculative writes must be rolled back before the next tile rewrite.
    """
    snapshots = getattr(metadata, _FP4_MLA_MTP_HP_SNAPSHOTS, None)
    if not snapshots:
        return

    keep_snapshots = False
    try:
        pool = getattr(metadata, "high_precision_kv_pool", None)
        if pool is None:
            return
        accepted_tokens = num_accepted_tokens.to(device=pool.device)
        for local_layer, snapshot in snapshots.items():
            head_dim = snapshot["head_dim"]
            pool_head_dim = snapshot["pool_head_dim"]
            block_d = triton.next_power_of_2(head_dim)
            if snapshot.get("mode") == "pool":
                keep_snapshots = True
                _hp_kv_restore_rejected_from_pool_kernel[(snapshot["num_gen_tokens"],)](
                    pool,
                    metadata.fp4_mla_hp_snapshot_pool,
                    metadata.batch_indices,
                    metadata.positions,
                    metadata.seq_slots,
                    metadata.kv_lens_cuda_runtime,
                    metadata.prompt_lens_cuda_runtime,
                    accepted_tokens,
                    snapshot["metadata_token_offset"],
                    snapshot["num_gen_tokens"],
                    metadata.batch_indices.shape[0],
                    metadata.num_seqs,
                    accepted_tokens.shape[0],
                    pool.shape[0],
                    pool.shape[1],
                    int(local_layer),
                    pool.stride(0),
                    pool.stride(1),
                    metadata.fp4_mla_hp_snapshot_pool.stride(0),
                    metadata.fp4_mla_hp_snapshot_pool.stride(1),
                    D=head_dim,
                    POOL_HEAD_D=pool_head_dim,
                    BLOCK_D=block_d,
                    HP_BLOCK=HP_BLOCK_SIZE,
                )
            else:
                positions = snapshot["positions"]
                batch_indices = snapshot["batch_indices"]
                seq_slots = snapshot["seq_slots"]
                hp_slots = snapshot["hp_slots"]
                first_new_positions = snapshot["first_new_positions"]
                if positions.shape[0] == 0:
                    continue
                _hp_kv_restore_rejected_from_values_kernel[(positions.shape[0],)](
                    pool,
                    snapshot["values"],
                    batch_indices,
                    positions,
                    seq_slots,
                    hp_slots,
                    first_new_positions,
                    accepted_tokens,
                    positions.shape[0],
                    accepted_tokens.shape[0],
                    pool.shape[0],
                    pool.shape[1],
                    int(local_layer),
                    pool.stride(0),
                    pool.stride(1),
                    snapshot["values"].stride(0),
                    snapshot["values"].stride(1),
                    D=head_dim,
                    POOL_HEAD_D=pool_head_dim,
                    BLOCK_D=block_d,
                    HP_BLOCK=HP_BLOCK_SIZE,
                )
    finally:
        if not keep_snapshots:
            setattr(metadata, _FP4_MLA_MTP_HP_SNAPSHOTS, None)


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

    Generation phase stores every new token for each request into position
        ``position % HP_BLOCK_SIZE``, overwriting the oldest entries in the
        circular buffer.  This supports linear MTP where a request contributes
        more than one generation token in a forward pass.

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

    # Generation phase: store current tokens at position % HP_BLOCK_SIZE.
    num_gen = num_seqs - num_contexts
    if update_generation and num_gen > 0:
        gen_tok_start = 0
        metadata_token_offset = getattr(metadata, "num_ctx_tokens", 0)
        if phase == "all":
            # Scalar offset: number of context tokens packed before gen tokens.
            gen_tok_start = int(metadata.prompt_lens_cpu_runtime[:num_contexts].sum().item())
            metadata_token_offset = gen_tok_start
        num_gen_tokens = latent_cache.shape[0] - gen_tok_start
        if num_gen_tokens < 0:
            raise RuntimeError(
                "FP4 MLA HP generation update received fewer latent tokens than "
                f"the context prefix: latent_tokens={latent_cache.shape[0]}, "
                f"context_tokens={gen_tok_start}."
            )
        if num_gen_tokens == 0:
            return

        gen_token_lens = _host_int_list_during_forward(
            getattr(metadata, "prompt_lens_cpu_runtime", None), num_contexts, num_seqs
        )
        if gen_token_lens is not None:
            max_gen_len = max(gen_token_lens)
        elif num_gen_tokens % num_gen == 0:
            max_gen_len = num_gen_tokens // num_gen
        else:
            max_gen_len = num_gen_tokens
        _snapshot_hp_kv_for_mtp_generation(
            metadata,
            pool,
            local_layer,
            num_gen=num_gen,
            num_gen_tokens=num_gen_tokens,
            max_gen_len=max_gen_len,
            metadata_token_offset=metadata_token_offset,
            head_dim=head_dim,
            pool_head_dim=pool_head_dim,
        )

        _fp4_mla_debug(
            "hp generation launch: "
            f"grid=({num_gen_tokens},) gen_tok_start={gen_tok_start} "
            f"metadata_token_offset={metadata_token_offset}"
        )
        _hp_kv_store_gen_kernel[(num_gen_tokens,)](
            pool,
            latent_cache,
            metadata.seq_slots,
            metadata.batch_indices,
            metadata.positions,
            gen_tok_start,
            metadata_token_offset,
            num_gen_tokens,
            metadata.batch_indices.shape[0],
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


def _stage_pool_layer_view(
    pool: torch.Tensor,
    local_layer: int,
    pool_head_dim: int,
    slots: int,
) -> torch.Tensor:
    return pool[:, local_layer, 0, :].view(pool.shape[0], slots, pool_head_dim)


def _snapshot_page_stage_for_mtp_generation(
    metadata: Any,
    pool: torch.Tensor,
    local_layer: int,
    *,
    slots: int,
    num_gen: int,
    num_gen_tokens: int,
    max_gen_len: int,
    metadata_token_offset: int,
    head_dim: int,
    pool_head_dim: int,
) -> None:
    """Snapshot staging-pool slots before linear-MTP generation writes.

    Mirror of ``_snapshot_hp_kv_for_mtp_generation`` for the per-page staging
    buffer, keyed under ``_FP4_MLA_MTP_STAGE_SNAPSHOTS`` so it never aliases the
    HP-pool snapshots. ``slots`` is the page size (vs. the HP pool's
    ``HP_BLOCK_SIZE``), so the per-sequence MTP draft length must not exceed it.
    """
    if getattr(metadata, "is_warmup", False):
        return
    if num_gen_tokens <= num_gen:
        return
    if max_gen_len > slots:
        raise NotImplementedError(
            "FP4 MLA staging rollback for linear MTP supports at most "
            f"{slots} generation tokens per sequence, got {max_gen_len}."
        )

    end_token_offset = metadata_token_offset + num_gen_tokens
    if (
        end_token_offset > metadata.batch_indices.shape[0]
        or end_token_offset > metadata.positions.shape[0]
    ):
        raise RuntimeError(
            "FP4 MLA staging snapshot would read past generation metadata: "
            f"token_offset={metadata_token_offset}, num_gen_tokens={num_gen_tokens}, "
            f"batch_indices={metadata.batch_indices.shape[0]}, "
            f"positions={metadata.positions.shape[0]}."
        )

    snapshots = getattr(metadata, _FP4_MLA_MTP_STAGE_SNAPSHOTS, None)
    if snapshots is None:
        snapshots = {}
        setattr(metadata, _FP4_MLA_MTP_STAGE_SNAPSHOTS, snapshots)

    snapshot_pool = getattr(metadata, "fp4_mla_page_stage_snapshot_pool", None)
    if snapshot_pool is not None:
        snapshot_pool[:, local_layer, :, :].copy_(pool[:, local_layer, :, :])
        snapshots[int(local_layer)] = {
            "mode": "pool",
            "metadata_token_offset": metadata_token_offset,
            "num_gen_tokens": num_gen_tokens,
            "head_dim": head_dim,
            "pool_head_dim": pool_head_dim,
            "slots": slots,
        }
        return

    device = pool.device
    token_indices = torch.arange(
        metadata_token_offset,
        end_token_offset,
        dtype=torch.long,
        device=device,
    )
    batch_indices = metadata.batch_indices[token_indices].to(torch.long)
    positions = metadata.positions[token_indices].to(torch.long)
    seq_slots = metadata.seq_slots[batch_indices].to(torch.long)
    stage_slots = torch.remainder(positions, slots).to(torch.long)
    first_new_positions = metadata.kv_lens_cuda_runtime[batch_indices].to(
        torch.long
    ) - metadata.prompt_lens_cuda_runtime[batch_indices].to(torch.long)
    pool_view = _stage_pool_layer_view(pool, local_layer, pool_head_dim, slots)
    values = pool_view[seq_slots, stage_slots, :head_dim].clone()

    snapshots[int(local_layer)] = {
        "mode": "values",
        "batch_indices": batch_indices,
        "seq_slots": seq_slots,
        "hp_slots": stage_slots,
        "positions": positions,
        "first_new_positions": first_new_positions,
        "values": values,
        "head_dim": head_dim,
        "pool_head_dim": pool_head_dim,
        "slots": slots,
    }


def update_page_stage_for_fp4_mla(
    metadata: Any,
    latent_cache: Optional[torch.Tensor],
    local_layer: int,
    *,
    phase: _HPUpdatePhase = "all",
) -> None:
    """Store recent KV tokens at BF16 into the per-page staging buffer.

    Separate twin of ``update_hp_kv_for_fp4_mla`` for the per-page
    dynamic-scale path. Identical circular-buffer semantics, but the buffer
    holds a whole page (``metadata.page_size`` slots) instead of the HP pool's
    ``HP_BLOCK_SIZE`` slots, so the active page can be re-quantized to FP4 with
    its exact per-page amax. Reuses the HP store kernels with ``HP_BLOCK`` set
    to the page size. Ownership tracking is intentionally omitted -- the HP
    update already validates it on this path.
    """
    if phase not in ("all", "context", "generation"):
        raise ValueError(f"Unexpected FP4 MLA staging update phase: {phase}")
    pool = getattr(metadata, "fp4_mla_page_stage_pool", None)
    if pool is None or latent_cache is None:
        return

    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    update_context = phase in ("all", "context")
    update_generation = phase in ("all", "generation")

    slots = metadata.page_size
    head_dim = latent_cache.shape[-1]
    pool_head_dim = pool.shape[-1] // slots
    if pool_head_dim < head_dim:
        raise RuntimeError(
            f"FP4 MLA staging pool head dimension is too small: got "
            f"{pool_head_dim}, need at least {head_dim}."
        )
    block_d = triton.next_power_of_2(head_dim)
    pool_s0 = pool.stride(0)
    pool_s1 = pool.stride(1)
    lc_stride = latent_cache.stride(0)

    # Context phase: store last (kv_len % page_size) new tokens.
    if update_context and num_contexts > 0:
        prompt_lens_cpu = metadata.prompt_lens_cpu_runtime[:num_contexts]
        token_offsets_cpu = torch.zeros(num_contexts, dtype=torch.int32, device="cpu")
        if num_contexts > 1:
            token_offsets_cpu[1:].copy_(torch.cumsum(prompt_lens_cpu[:-1].to(torch.int32), dim=0))
        token_offsets_gpu = token_offsets_cpu.to(pool.device, non_blocking=False)
        prompt_lens_gpu = metadata.prompt_lens_cuda_runtime[:num_contexts]

        _hp_kv_store_context_kernel[(num_contexts, slots)](
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
            HP_BLOCK=slots,
        )
        _debug_sync("stage_context")

    # Generation phase: store current tokens at position % page_size.
    num_gen = num_seqs - num_contexts
    if update_generation and num_gen > 0:
        gen_tok_start = 0
        metadata_token_offset = getattr(metadata, "num_ctx_tokens", 0)
        if phase == "all":
            gen_tok_start = int(metadata.prompt_lens_cpu_runtime[:num_contexts].sum().item())
            metadata_token_offset = gen_tok_start
        num_gen_tokens = latent_cache.shape[0] - gen_tok_start
        if num_gen_tokens < 0:
            raise RuntimeError(
                "FP4 MLA staging generation update received fewer latent tokens "
                f"than the context prefix: latent_tokens={latent_cache.shape[0]}, "
                f"context_tokens={gen_tok_start}."
            )
        if num_gen_tokens == 0:
            return

        gen_token_lens = _host_int_list_during_forward(
            getattr(metadata, "prompt_lens_cpu_runtime", None), num_contexts, num_seqs
        )
        if gen_token_lens is not None:
            max_gen_len = max(gen_token_lens)
        elif num_gen_tokens % num_gen == 0:
            max_gen_len = num_gen_tokens // num_gen
        else:
            max_gen_len = num_gen_tokens
        _snapshot_page_stage_for_mtp_generation(
            metadata,
            pool,
            local_layer,
            slots=slots,
            num_gen=num_gen,
            num_gen_tokens=num_gen_tokens,
            max_gen_len=max_gen_len,
            metadata_token_offset=metadata_token_offset,
            head_dim=head_dim,
            pool_head_dim=pool_head_dim,
        )

        _hp_kv_store_gen_kernel[(num_gen_tokens,)](
            pool,
            latent_cache,
            metadata.seq_slots,
            metadata.batch_indices,
            metadata.positions,
            gen_tok_start,
            metadata_token_offset,
            num_gen_tokens,
            metadata.batch_indices.shape[0],
            pool.shape[0],
            pool.shape[1],
            local_layer,
            pool_s0,
            pool_s1,
            lc_stride,
            D=head_dim,
            POOL_HEAD_D=pool_head_dim,
            BLOCK_D=block_d,
            HP_BLOCK=slots,
        )
        _debug_sync("stage_generation")


def repair_fp4_mla_page_stage_for_mtp_rejection(
    metadata: Any,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Restore staging-pool slots that belonged to rejected linear-MTP tokens.

    Separate twin of ``repair_fp4_mla_hp_kv_for_mtp_rejection`` for the per-page
    staging buffer. Reuses the HP restore kernels with ``HP_BLOCK`` set to the
    snapshot's page size.
    """
    snapshots = getattr(metadata, _FP4_MLA_MTP_STAGE_SNAPSHOTS, None)
    if not snapshots:
        return

    keep_snapshots = False
    try:
        pool = getattr(metadata, "fp4_mla_page_stage_pool", None)
        if pool is None:
            return
        accepted_tokens = num_accepted_tokens.to(device=pool.device)
        for local_layer, snapshot in snapshots.items():
            head_dim = snapshot["head_dim"]
            pool_head_dim = snapshot["pool_head_dim"]
            slots = snapshot["slots"]
            block_d = triton.next_power_of_2(head_dim)
            if snapshot.get("mode") == "pool":
                keep_snapshots = True
                _hp_kv_restore_rejected_from_pool_kernel[(snapshot["num_gen_tokens"],)](
                    pool,
                    metadata.fp4_mla_page_stage_snapshot_pool,
                    metadata.batch_indices,
                    metadata.positions,
                    metadata.seq_slots,
                    metadata.kv_lens_cuda_runtime,
                    metadata.prompt_lens_cuda_runtime,
                    accepted_tokens,
                    snapshot["metadata_token_offset"],
                    snapshot["num_gen_tokens"],
                    metadata.batch_indices.shape[0],
                    metadata.num_seqs,
                    accepted_tokens.shape[0],
                    pool.shape[0],
                    pool.shape[1],
                    int(local_layer),
                    pool.stride(0),
                    pool.stride(1),
                    metadata.fp4_mla_page_stage_snapshot_pool.stride(0),
                    metadata.fp4_mla_page_stage_snapshot_pool.stride(1),
                    D=head_dim,
                    POOL_HEAD_D=pool_head_dim,
                    BLOCK_D=block_d,
                    HP_BLOCK=slots,
                )
            else:
                positions = snapshot["positions"]
                batch_indices = snapshot["batch_indices"]
                seq_slots = snapshot["seq_slots"]
                stage_slots = snapshot["hp_slots"]
                first_new_positions = snapshot["first_new_positions"]
                if positions.shape[0] == 0:
                    continue
                _hp_kv_restore_rejected_from_values_kernel[(positions.shape[0],)](
                    pool,
                    snapshot["values"],
                    batch_indices,
                    positions,
                    seq_slots,
                    stage_slots,
                    first_new_positions,
                    accepted_tokens,
                    positions.shape[0],
                    accepted_tokens.shape[0],
                    pool.shape[0],
                    pool.shape[1],
                    int(local_layer),
                    pool.stride(0),
                    pool.stride(1),
                    snapshot["values"].stride(0),
                    snapshot["values"].stride(1),
                    D=head_dim,
                    POOL_HEAD_D=pool_head_dim,
                    BLOCK_D=block_d,
                    HP_BLOCK=slots,
                )
    finally:
        if not keep_snapshots:
            setattr(metadata, _FP4_MLA_MTP_STAGE_SNAPSHOTS, None)
