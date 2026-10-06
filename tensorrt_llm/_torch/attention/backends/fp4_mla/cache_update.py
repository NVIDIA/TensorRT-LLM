# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA context/generation cache scatter and fused Q/RoPE quantization."""

from typing import Any, Optional

import torch
import triton

from tensorrt_llm._utils import get_sm_version

from .config import (
    _FP4_MLA_CUTEDSL_BACKEND,
    _FP4_MLA_K_RESIDUAL_BACKENDS,
    _FP4_MLA_MAX_GRID_Z,
    FP4_BLOCK_SIZE,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_Q1_PREFIX_BLOCK_DIM,
    FP4_MLA_Q_LOGICAL_DIM,
    FP4_MLA_Q_PACKED_DIM,
    FP4_MLA_Q_PREFIX_BLOCK_DIM,
    FP4_MLA_Q_PREFIX_DIM,
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_Q_SF_GROUPS,
    FP4_MLA_SCALE_ROW_GROUP,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    _ceil_div,
    _fp4_mla_attention_backend,
    _fp4_mla_cutedsl_fused_v_transpose_enabled,
    _fp4_mla_q1_kv_blocks_per_program,
    _fp4_mla_q1_prefix_blocks_per_program,
    _HPUpdatePhase,
)
from .fp4_mla_kernels import (
    _fp4_mla_context_cache_update_kernel,
    _fp4_mla_generation_fused_qk_rope_cache_update_kernel,
)
from .layout import (
    _get_fp4_mla_global_scale,
    _get_fp4_mla_hp_pool_layout,
    _get_fp4_mla_kv_cache_tensors,
    _get_fp4_mla_q_global_scale,
    _get_fp4_mla_swizzled_scale_size,
    _validate_fp4_mla_cache_shape,
    _validate_fp4_mla_hp_generation_width,
    _validate_fp4_mla_kv_storage_shape,
    get_fp4_mla_v_scale_pool_view,
)
from .metadata import (
    _fp4_mla_generation_hp_page_ids,
    _fp4_mla_generation_num_blocks_device,
    _fp4_mla_generation_page_ids,
    _fp4_mla_uniform_generation_lengths,
    _materialize_fp4_mla_device_page_table_for_forward,
)
from .v_cache import (
    _get_cutedsl_persistent_v_packed_cache,
    _get_fp4_mla_v_packed_pool_base,
    _maybe_update_triton_v_packed_cache,
    _repack_cutedsl_v_packed_cache,
)


def _validate_fp4_mla_context_rope(
    latent_cache: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    v_head_dim: int,
) -> int:
    head_dim = latent_cache.shape[-1]
    rope_dim = head_dim - v_head_dim
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(
            "FP4 MLA fused context K-RoPE requires a positive even RoPE dimension, "
            f"got head_dim={head_dim}, v_head_dim={v_head_dim}."
        )
    if rotary_cos_sin.device != latent_cache.device:
        raise ValueError("FP4 MLA context latent cache and RoPE table must use the same device.")
    if rotary_cos_sin.dtype != torch.float32:
        raise TypeError(
            f"FP4 MLA fused context K-RoPE requires a float32 table, got {rotary_cos_sin.dtype}."
        )
    if not rotary_cos_sin.is_contiguous():
        raise ValueError("FP4 MLA fused context K-RoPE requires a contiguous RoPE table.")
    table_row_size = rope_dim * 2
    if rotary_cos_sin.numel() < table_row_size or rotary_cos_sin.numel() % table_row_size != 0:
        raise ValueError(
            "FP4 MLA context RoPE table size must be a positive multiple of "
            f"{table_row_size}, got {rotary_cos_sin.numel()}."
        )
    return rope_dim


def can_fuse_fp4_mla_q_quant(
    metadata: Any,
    q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
) -> bool:
    """Return whether generation can quantize Q in the fused cache update."""
    num_gen = metadata.num_seqs - metadata.num_contexts
    return bool(
        _fp4_mla_attention_backend() in _FP4_MLA_K_RESIDUAL_BACKENDS
        and get_sm_version() == 107
        and num_gen > 0
        and getattr(metadata, "kv_cache_manager", None) is not None
        and q.shape[0] > 0
        and q.shape[0] % num_gen == 0
        and q.is_cuda
        and q_pe.is_cuda
        and latent_cache.is_cuda
        and q.device == q_pe.device == latent_cache.device
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
        and q.ndim == 3
        and 0 < q.shape[1] <= 128
        and q.shape[2] == FP4_MLA_Q_PREFIX_DIM + FP4_MLA_Q_RESIDUAL_DIM
        and q_pe.dtype == torch.bfloat16
        and tuple(q_pe.shape) == (q.shape[0], q.shape[1], FP4_MLA_Q_RESIDUAL_DIM)
        and latent_cache.dtype == torch.bfloat16
        and tuple(latent_cache.shape) == (q.shape[0], FP4_MLA_Q_PREFIX_DIM + FP4_MLA_Q_RESIDUAL_DIM)
        and metadata.page_size == FP4_MLA_TOKENS_PER_BLOCK
    )


def _prepare_fp4_mla_q_buffers(
    metadata: Any,
    num_queries: int,
    num_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return manager-owned, fixed-capacity packed-Q staging buffers."""
    if num_queries <= 0:
        raise ValueError(f"FP4 MLA Q buffer preparation needs queries, got {num_queries}.")
    owner = getattr(metadata, "kv_cache_manager", None)
    if owner is None:
        raise RuntimeError("Fused FP4 MLA Q quantization requires a KV cache manager.")
    if num_heads <= 0 or num_heads > 128:
        raise ValueError(f"FP4 MLA Q buffers require 1-128 local heads, got {num_heads}.")
    buffers = getattr(owner, "_fp4_mla_q_buffers", None)
    if buffers is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Cannot create FP4 MLA Q buffers while capturing a CUDA graph. "
                "Run a warmup forward first."
            )
        buffers = {}
        setattr(owner, "_fp4_mla_q_buffers", buffers)

    max_num_tokens = int(getattr(metadata, "max_num_tokens", num_queries) or num_queries)
    max_num_sequences = int(
        getattr(metadata, "max_num_sequences", None)
        or getattr(metadata, "max_num_requests", num_queries)
        or num_queries
    )
    max_query_width = 1 + int(getattr(metadata, "max_total_draft_tokens", None) or 0)
    capacity = min(
        max_num_tokens,
        max_num_sequences * max_query_width,
        _FP4_MLA_MAX_GRID_Z,
    )
    if num_queries > capacity:
        raise ValueError(
            f"FP4 MLA active queries exceed the configured Q capacity: {num_queries} > {capacity}."
        )

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    canonical_device = torch.device("cuda", device_index)
    expected_q_shape = (capacity * num_heads, FP4_MLA_Q_PACKED_DIM)
    expected_q_sf_shape = (
        _get_fp4_mla_swizzled_scale_size(capacity * num_heads, FP4_MLA_Q_LOGICAL_DIM),
    )
    q_key = f"q_{num_heads}"
    q_sf_key = f"q_sf_{num_heads}"
    q_storage = buffers.get(q_key)
    q_sf_storage = buffers.get(q_sf_key)
    if q_storage is None and q_sf_storage is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Cannot allocate FP4 MLA Q buffers while capturing a CUDA graph. "
                "Run a warmup forward first."
            )
        q_storage = torch.empty(expected_q_shape, dtype=torch.uint8, device=canonical_device)
        q_sf_storage = torch.empty(
            expected_q_sf_shape,
            dtype=torch.float8_e4m3fn,
            device=canonical_device,
        )
        buffers[q_key] = q_storage
        buffers[q_sf_key] = q_sf_storage
    if (
        q_storage is None
        or q_storage.dtype != torch.uint8
        or q_storage.device != canonical_device
        or tuple(q_storage.shape) != expected_q_shape
        or not q_storage.is_contiguous()
        or q_sf_storage is None
        or q_sf_storage.dtype != torch.float8_e4m3fn
        or q_sf_storage.device != canonical_device
        or tuple(q_sf_storage.shape) != expected_q_sf_shape
        or not q_sf_storage.is_contiguous()
    ):
        raise RuntimeError("FP4 MLA packed-Q buffers do not match the configured capacity.")
    return q_storage, q_sf_storage, capacity


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


def _validate_fp4_mla_context_start_alignment(
    metadata: Any,
    num_contexts: int,
    *,
    alignment: int = HP_BLOCK_SIZE,
) -> None:
    context_start_positions = _get_fp4_mla_context_start_positions(metadata, num_contexts)
    bad_start = (context_start_positions < 0) | ((context_start_positions % alignment) != 0)
    if bool(torch.any(bad_start).item()):
        starts = context_start_positions.detach().cpu().tolist()
        raise ValueError(
            "FP4 MLA shared-tile context update requires every context "
            f"start position to be {alignment}-token aligned, got "
            f"start positions {starts}."
        )


def _scatter_fp4_mla_kv_cache_2d_context(
    metadata: Any,
    latent_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    rotary_cos_sin: Optional[torch.Tensor],
    q_context: Optional[torch.Tensor],
    q_nope_head_dim: Optional[int],
    *,
    token_offset: int,
    local_layer: int,
    v_head_dim: int,
    head_dim: int,
    num_tokens: int,
    num_dim_blocks: int,
    sf_per_token: int,
    sf_per_page: int,
    v_packed_base: Optional[torch.Tensor] = None,
    v_page_offset: int = 0,
) -> bool:
    num_contexts = metadata.num_contexts
    if num_contexts > 0:
        prompt_lens_cpu = metadata.prompt_lens_cpu_runtime[:num_contexts]
        ctx_token_count = int(prompt_lens_cpu.sum().item())
        if num_tokens != ctx_token_count:
            raise RuntimeError(
                f"FP4 MLA 2D context scatter needs {ctx_token_count} context tokens, got "
                f"{num_tokens}."
            )
        _validate_fp4_mla_context_start_alignment(metadata, num_contexts, alignment=FP4_BLOCK_SIZE)

    apply_k_rope = rotary_cos_sin is not None
    rope_dim = (
        _validate_fp4_mla_context_rope(latent_cache, rotary_cos_sin, v_head_dim)
        if rotary_cos_sin is not None
        else 0
    )
    rotary_cos_sin_ptr = rotary_cos_sin if rotary_cos_sin is not None else latent_cache

    apply_q_rope = q_context is not None
    block_q_heads = 16
    if apply_q_rope:
        if rotary_cos_sin is None or q_nope_head_dim is None:
            raise ValueError("FP4 MLA fused context Q-RoPE requires a rotary table and Q layout.")
        q_head_dim = q_nope_head_dim + rope_dim
        if (
            q_context.dtype != torch.bfloat16
            or q_context.device != latent_cache.device
            or q_context.ndim != 2
            or q_context.shape[0] != num_tokens
            or q_context.shape[1] <= 0
            or q_nope_head_dim <= 0
            or q_context.shape[1] % q_head_dim != 0
            or not q_context.is_contiguous()
        ):
            raise ValueError(
                "FP4 MLA fused context Q-RoPE requires a contiguous same-device BF16 "
                f"tensor shaped [tokens, heads * ({q_nope_head_dim} + {rope_dim})]."
            )
        num_q_heads = q_context.shape[1] // q_head_dim
        q_context_view = q_context.view(num_tokens, num_q_heads, q_head_dim)
        q_head_blocks = triton.cdiv(num_q_heads, block_q_heads)
    else:
        num_q_heads = 0
        q_context_view = latent_cache
        q_head_blocks = 0

    hp_pool = getattr(metadata.fp4_mla_state, "hp_pool", None)
    if not isinstance(hp_pool, torch.Tensor):
        raise TypeError("FP4 MLA high-precision KV pool must be a tensor.")
    if hp_pool.device != latent_cache.device:
        raise ValueError("FP4 MLA latent cache and high-precision pool must share a device.")
    if hp_pool.dtype != torch.bfloat16:
        raise TypeError(f"FP4 MLA high-precision KV pool must use BF16, got {hp_pool.dtype}.")
    hp_pool_size, pool_head_dim = _get_fp4_mla_hp_pool_layout(metadata, hp_pool)
    if pool_head_dim < head_dim:
        raise RuntimeError(
            f"FP4 MLA HP pool head dimension is too small: got "
            f"{pool_head_dim}, need at least {head_dim}."
        )
    if local_layer < 0 or local_layer >= hp_pool.shape[1]:
        raise ValueError(
            f"FP4 MLA local layer {local_layer} is outside the HP pool's {hp_pool.shape[1]} layers."
        )
    if hp_pool.stride(-1) != 1:
        raise ValueError("FP4 MLA high-precision KV pool must be contiguous in head_dim.")
    hp_page_ids = metadata.fp4_mla_state.hp_page_indices
    if not isinstance(hp_page_ids, torch.Tensor):
        raise RuntimeError("FP4 MLA context cache update requires HP page metadata.")
    store_hp_tail = num_contexts > 0
    num_hp_pages = hp_pool.shape[0]
    pool_s0 = hp_pool.stride(0)
    pool_s1 = hp_pool.stride(1)

    write_v_packed = v_packed_base is not None
    v_packed_output = v_packed_base if write_v_packed else kv_cache
    v_packed_s0 = v_packed_output.stride(0) if write_v_packed else 0
    v_packed_s1 = v_packed_output.stride(1) if write_v_packed else 0

    _fp4_mla_context_cache_update_kernel[
        (
            num_tokens,
            num_dim_blocks + q_head_blocks,
        )
    ](
        kv_cache,
        sf_cache,
        v_sf,
        v_packed_output,
        latent_cache,
        q_context_view,
        global_scale,
        rotary_cos_sin_ptr,
        hp_pool,
        hp_page_ids,
        metadata.fp4_mla_state.batch_indices,
        metadata.fp4_mla_state.positions,
        metadata.fp4_mla_state.paged_kv_indices,
        metadata.fp4_mla_state.paged_kv_indptr,
        metadata.fp4_mla_state.paged_kv_indices.shape[0],
        metadata.fp4_mla_state.paged_kv_indptr.shape[0],
        metadata.fp4_mla_state.batch_indices.shape[0],
        v_sf.shape[1],
        v_sf.shape[0],
        num_contexts,
        num_hp_pages,
        token_offset,
        num_tokens,
        local_layer,
        v_page_offset if write_v_packed else 0,
        metadata.page_size,
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        latent_cache.stride(0),
        latent_cache.stride(1),
        q_context_view.stride(0),
        q_context_view.stride(1) if apply_q_rope else 0,
        q_context_view.stride(2) if apply_q_rope else 0,
        v_sf.stride(0),
        v_sf.stride(1),
        v_packed_s0,
        v_packed_s1,
        pool_s0,
        pool_s1,
        HEAD_D=head_dim,
        V_HEAD_D=v_head_dim,
        HP_BLOCK=FP4_BLOCK_SIZE,
        HP_POOL_SIZE=hp_pool_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_TOKEN=sf_per_token,
        SF_PER_PAGE=sf_per_page,
        K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
        STORE_K_RESIDUAL=(_fp4_mla_attention_backend() in _FP4_MLA_K_RESIDUAL_BACKENDS),
        ROPE_DIM=rope_dim,
        APPLY_K_ROPE=apply_k_rope,
        APPLY_Q_ROPE=apply_q_rope,
        NUM_DIM_BLOCKS=num_dim_blocks,
        NUM_Q_HEADS=num_q_heads,
        Q_NOPE_DIM=q_nope_head_dim if q_nope_head_dim is not None else 0,
        BLOCK_Q_HEADS=block_q_heads,
        POOL_HEAD_D=pool_head_dim,
        STORE_HP_TAIL=store_hp_tail,
        WRITE_V_PACKED=write_v_packed,
    )
    return store_hp_tail


def _scatter_fp4_mla_kv_cache_2d_generation(
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
    rotary_cos_sin: torch.Tensor,
    q_pe: torch.Tensor,
    q_rope_out: torch.Tensor,
    q_quant_input: torch.Tensor,
    q_fp4_out: torch.Tensor,
    q_sf_out: torch.Tensor,
    v_packed_base: Optional[torch.Tensor],
    v_page_offset: int,
    helix_position_offsets: Optional[torch.Tensor],
    helix_is_inactive_rank: Optional[torch.Tensor],
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    num_gen = num_seqs - num_contexts
    if num_gen <= 0:
        return
    if num_tokens < num_gen:
        raise RuntimeError(
            f"FP4 MLA 2D generation scatter needs at least {num_gen} generation "
            f"tokens, got {num_tokens}."
        )
    if num_tokens % num_gen != 0:
        raise NotImplementedError(
            "FP4 MLA no-dequant generation scatter requires a uniform linear MTP "
            f"generation length, got {num_tokens} tokens for {num_gen} sequences."
        )
    # The prompt_lens/kv_lens runtime aliases can lag at the decode anchor
    # (seq_lens == 1) under CUDA graph / one-engine MTP while each generation
    # sequence really appends num_tokens // num_gen tokens this step. Recover the
    # true per-sequence lengths for the no-dequant kernel below (a no-op when the
    # aliases already match).
    kv_lens_gen, gen_lens_gen = _fp4_mla_uniform_generation_lengths(metadata, num_tokens, num_gen)
    _materialize_fp4_mla_device_page_table_for_forward(metadata, kv_lens_gen)

    pool = getattr(metadata.fp4_mla_state, "hp_pool", None)
    if pool is None:
        raise RuntimeError("FP4 MLA 2D generation scatter requires the HP KV pool.")
    try:
        hp_pool_size, hp_head_dim = _get_fp4_mla_hp_pool_layout(metadata, pool)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if hp_head_dim < head_dim:
        raise RuntimeError(
            f"FP4 MLA 2D generation scatter needs at least {head_dim} HP channels, got "
            f"{hp_head_dim}."
        )
    hp_page_ids = _fp4_mla_generation_hp_page_ids(metadata, num_gen)
    if not isinstance(hp_page_ids, torch.Tensor):
        raise RuntimeError("FP4 MLA generation requires HP page metadata.")
    num_hp_pages = pool.shape[0]

    max_gen_len = num_tokens // num_gen
    use_helix = helix_position_offsets is not None or helix_is_inactive_rank is not None
    use_helix_local_slots = use_helix and bool(getattr(metadata, "_helix_spec_tokens_valid", False))
    if use_helix:
        if helix_position_offsets is None or helix_is_inactive_rank is None:
            raise RuntimeError(
                "FP4 MLA Helix requires both position-offset and inactive-rank metadata."
            )
        if max_gen_len != 1 and not use_helix_local_slots:
            raise NotImplementedError(
                "FP4 MLA multi-token Helix requires speculative per-token metadata."
            )
        if (
            helix_position_offsets.dtype != torch.int32
            or helix_position_offsets.device != latent_cache.device
            or helix_position_offsets.ndim != 1
            or helix_position_offsets.numel() < num_tokens
            or not helix_position_offsets.is_contiguous()
        ):
            raise ValueError(
                "FP4 MLA Helix position offsets must be a contiguous same-device "
                "int32 tensor covering every generation token."
            )
        if (
            helix_is_inactive_rank.dtype != torch.bool
            or helix_is_inactive_rank.device != latent_cache.device
            or helix_is_inactive_rank.ndim != 1
            or helix_is_inactive_rank.numel() < num_gen
            or not helix_is_inactive_rank.is_contiguous()
        ):
            raise ValueError(
                "FP4 MLA Helix inactive-rank metadata must be a contiguous "
                "same-device bool tensor covering every generation sequence."
            )
        if use_helix_local_slots:
            helix_local_slots = getattr(metadata, "helix_local_slots", None)
            if (
                not isinstance(helix_local_slots, torch.Tensor)
                or helix_local_slots.dtype != torch.int32
                or helix_local_slots.device != latent_cache.device
                or helix_local_slots.ndim != 1
                or helix_local_slots.numel() < num_tokens
                or not helix_local_slots.is_contiguous()
            ):
                raise ValueError(
                    "FP4 MLA speculative Helix local slots must be a contiguous "
                    "same-device int32 tensor covering every generation token."
                )
        else:
            helix_local_slots = helix_position_offsets
    else:
        helix_position_offsets = kv_lens_gen
        helix_local_slots = kv_lens_gen
        helix_is_inactive_rank = gen_lens_gen
    _validate_fp4_mla_hp_generation_width(hp_pool_size, max_gen_len)
    page_ids = _fp4_mla_generation_page_ids(metadata, num_gen)
    rope_dim = head_dim - v_head_dim
    block_q_heads = 32
    q1_kv_blocks_per_program = 1
    # Grouped Q1 kernels reuse K's packed codes for V. Keep one dimension
    # block per program until their warp-specialized V quantizer is split out.
    if max_gen_len == 1:
        q1_kv_blocks_per_program = _fp4_mla_q1_kv_blocks_per_program(num_gen, v_head_dim)
    q_prefix_block_dim = (
        FP4_MLA_Q1_PREFIX_BLOCK_DIM if max_gen_len == 1 else FP4_MLA_Q_PREFIX_BLOCK_DIM
    )
    q_prefix_blocks = FP4_MLA_Q_PREFIX_DIM // q_prefix_block_dim
    q_prefix_blocks_per_program = _fp4_mla_q1_prefix_blocks_per_program(
        num_gen,
        q1_kv_blocks_per_program,
    )
    q_work_blocks = q_prefix_blocks // q_prefix_blocks_per_program + 1
    if latent_cache.dtype != torch.bfloat16:
        raise TypeError(
            "Fused FP4 MLA Q/K RoPE and cache storage requires BF16 latent KV, "
            f"got {latent_cache.dtype}."
        )
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(
            "Fused FP4 MLA K RoPE requires a positive even K tail, "
            f"got head_dim={head_dim} v_head_dim={v_head_dim}."
        )
    if (
        rotary_cos_sin is None
        or rotary_cos_sin.device != latent_cache.device
        or rotary_cos_sin.dtype != torch.float32
        or rotary_cos_sin.numel() % (rope_dim * 2) != 0
    ):
        raise ValueError(
            "Fused FP4 MLA K RoPE requires a same-device FP32 rotary "
            f"table with rows of {rope_dim * 2} values."
        )
    if (
        q_pe is None
        or q_rope_out is None
        or q_pe.dtype != torch.bfloat16
        or q_rope_out.dtype != torch.bfloat16
        or q_pe.device != latent_cache.device
        or q_rope_out.device != latent_cache.device
        or q_pe.ndim != 3
        or q_rope_out.shape != q_pe.shape
        or q_pe.shape[0] != num_tokens
        or q_pe.shape[1] <= 0
        or q_pe.shape[2] != rope_dim
    ):
        raise ValueError(
            "Fused FP4 MLA Q RoPE requires same-device BF16 q_pe and "
            f"q_rope_out tensors shaped [tokens, heads, {rope_dim}]."
        )
    num_q_heads = q_pe.shape[1]
    q_head_blocks = _ceil_div(num_q_heads, block_q_heads)
    max_gen_tiles = _ceil_div(max_gen_len + FP4_BLOCK_SIZE - 1, FP4_BLOCK_SIZE)
    rotary_table = rotary_cos_sin
    q_global_scale = _get_fp4_mla_q_global_scale(metadata, latent_cache.device)
    q_pe_input = q_pe
    q_rope_output = q_rope_out
    q_full_input = q_quant_input
    q_fp4_output = q_fp4_out
    q_sf_output = q_sf_out
    write_v_packed = v_packed_base is not None
    v_packed_output = v_packed_base if write_v_packed else kv_cache
    v_packed_s0 = v_packed_output.stride(0) if write_v_packed else 0
    v_packed_s1 = v_packed_output.stride(1) if write_v_packed else 0
    kv_work_blocks = (
        v_head_dim // FP4_BLOCK_SIZE // q1_kv_blocks_per_program + 1
        if max_gen_len == 1
        else num_dim_blocks
    )
    launch_grid = (
        num_gen,
        max(
            kv_work_blocks,
            max_gen_len * q_head_blocks * q_work_blocks,
        ),
    )
    store_k_residual = _fp4_mla_attention_backend() in _FP4_MLA_K_RESIDUAL_BACKENDS

    def launch_generation_update(
        grid: tuple[int, ...],
        *,
        page_ids_len: int,
        indptr_len: int,
        max_gen_tiles_variant: int,
        q_prefix_block_dim_variant: int,
        q_prefix_blocks_variant: int,
        q_prefix_blocks_per_program_variant: int,
        q1_kv_blocks_per_program_variant: int,
    ) -> None:
        q_work_blocks_variant = q_prefix_blocks_variant // q_prefix_blocks_per_program_variant + 1
        _fp4_mla_generation_fused_qk_rope_cache_update_kernel[grid](
            kv_cache,
            sf_cache,
            v_sf,
            v_packed_output,
            pool,
            latent_cache,
            global_scale,
            q_global_scale,
            rotary_table,
            q_pe_input,
            q_rope_output,
            q_full_input,
            q_fp4_output,
            q_sf_output,
            kv_lens_gen,
            gen_lens_gen,
            helix_position_offsets,
            helix_local_slots,
            helix_is_inactive_rank,
            page_ids,
            hp_page_ids,
            metadata.fp4_mla_state.paged_kv_indptr_decode,
            page_ids_len,
            hp_page_ids.numel(),
            indptr_len,
            v_sf.shape[1],
            num_hp_pages,
            v_sf.shape[0],
            local_layer,
            v_page_offset if write_v_packed else 0,
            metadata.page_size,
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(4),
            sf_cache.stride(0),
            pool.stride(0),
            pool.stride(1),
            v_sf.stride(0),
            v_sf.stride(1),
            v_packed_s0,
            v_packed_s1,
            q_pe_input.stride(0),
            q_pe_input.stride(1) if q_pe_input.ndim > 1 else 0,
            q_pe_input.stride(2) if q_pe_input.ndim > 2 else 0,
            q_rope_output.stride(0),
            q_rope_output.stride(1) if q_rope_output.ndim > 1 else 0,
            q_rope_output.stride(2) if q_rope_output.ndim > 2 else 0,
            HEAD_D=hp_head_dim,
            V_HEAD_D=v_head_dim,
            HP_BLOCK=FP4_BLOCK_SIZE,
            HP_POOL_SIZE=hp_pool_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            SF_PER_TOKEN=sf_per_token,
            SF_PER_PAGE=sf_per_page,
            K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
            STORE_K_RESIDUAL=store_k_residual,
            FUSE_ROPE_CACHE_STORE=True,
            USE_HELIX=use_helix,
            USE_HELIX_LOCAL_SLOTS=use_helix_local_slots,
            WRITE_V_PACKED=write_v_packed,
            MAX_GEN_TILES=max_gen_tiles_variant,
            ROPE_DIM=rope_dim,
            ROPE_PAIR_BLOCK=triton.next_power_of_2(rope_dim // 2),
            NUM_DIM_BLOCKS=num_dim_blocks,
            NUM_Q_HEADS=num_q_heads,
            Q_HEAD_BLOCKS=max(q_head_blocks, 1),
            BLOCK_Q_HEADS=block_q_heads,
            Q_PREFIX_D=FP4_MLA_Q_PREFIX_DIM,
            Q_PREFIX_BLOCK_D=q_prefix_block_dim_variant,
            Q_PREFIX_BLOCKS=q_prefix_blocks_variant,
            Q_PREFIX_BLOCKS_PER_PROGRAM=q_prefix_blocks_per_program_variant,
            Q_WORK_BLOCKS=q_work_blocks_variant,
            Q_SF_COLS=FP4_MLA_Q_SF_GROUPS,
            WRITE_Q=True,
            Q1_KV_BLOCKS_PER_PROGRAM=q1_kv_blocks_per_program_variant,
            maxnreg=56,
        )

    launch_generation_update(
        launch_grid,
        page_ids_len=page_ids.shape[0],
        indptr_len=metadata.fp4_mla_state.paged_kv_indptr_decode.shape[0],
        max_gen_tiles_variant=max(max_gen_tiles, 1),
        q_prefix_block_dim_variant=q_prefix_block_dim,
        q_prefix_blocks_variant=q_prefix_blocks,
        q_prefix_blocks_per_program_variant=q_prefix_blocks_per_program,
        q1_kv_blocks_per_program_variant=q1_kv_blocks_per_program,
    )
    return kv_lens_gen, gen_lens_gen, page_ids


# Public cache update and decode entry points


def scatter_fp4_mla_kv_cache(
    metadata: Any,
    latent_cache: torch.Tensor,
    layer_idx: int,
    *,
    token_offset: int,
    phase: _HPUpdatePhase,
    local_layer: int,
    v_head_dim: int,
    rotary_cos_sin: Optional[torch.Tensor] = None,
    q_pe: Optional[torch.Tensor] = None,
    q_rope_out: Optional[torch.Tensor] = None,
    q_quant_input: Optional[torch.Tensor] = None,
    helix_position_offsets: Optional[torch.Tensor] = None,
    helix_is_inactive_rank: Optional[torch.Tensor] = None,
    q_context: Optional[torch.Tensor] = None,
    q_nope_head_dim: Optional[int] = None,
) -> bool:
    """Quantize MLA latent tokens and scatter them into the paged FP4 cache.

    Contract: this helper scatters exactly ``latent_cache.shape[0]`` tokens,
    reading index metadata at ``batch_indices[token_offset : token_offset + N]``
    and ``positions[token_offset : token_offset + N]``. Callers must pass a
    latent_cache pre-sliced to the current phase (context or generation) so
    that ``shape[0]`` matches the number of index entries they intend to
    consume. ``MLA.forward_impl`` (tensorrt_llm/_torch/modules/attention.py)
    slices ``latent_cache[:num_ctx_tokens]`` for context and
    ``latent_cache[num_ctx_tokens:]`` for generation before dispatching.

    Callers must pass ``phase``, ``local_layer``, and ``v_head_dim``. Context
    scatter writes the final FP4 tile representation directly. Dimensions
    below ``v_head_dim`` share one 16-token by 16-dim FP4 tile between K
    and V, with the scale written into K's token-major and V's dim-major
    layouts. Tail K-only dimensions use K's per-token 1D scales. For
    exclusively owned CuTeDSL pages, context scatter also writes the
    persistent packed-V sidecar.
    Context scatter can rotate Q in place and rotate the K tail directly from
    the unassembled latent tensor. When context Q is supplied for chunked
    prefill, it also writes rotated K back for current-chunk attention.
    Generation scatter rewrites each touched 16-token tile by reading
    old tokens from the HP pool and new tokens from ``latent_cache``. The
    static-scale generation
    specialization can also rotate Q and new K tails while updating the HP pool.
    The context kernel also stores the final incomplete tile in the BF16 HP
    pool. When ``q_quant_input`` is supplied, the generation kernel also emits
    backend-ready residual FP4 Q. The return value reports whether the current
    phase updated the HP pool.
    """
    if phase == "generation":
        metadata.fp4_mla_state.generation_cache_scattered = False
        metadata.fp4_mla_state.prequantized_q = None
        metadata.fp4_mla_state.prequantized_q_sf = None
        metadata.fp4_mla_state.q_batch_capacity = None
    if latent_cache.numel() == 0:
        raise ValueError("FP4 MLA cache scatter requires at least one latent token.")
    if q_context is not None and not latent_cache.is_contiguous():
        raise ValueError(
            "FP4 MLA fused context Q/K RoPE requires contiguous latent_cache "
            "storage for in-place current-K update."
        )

    latent_cache = latent_cache.reshape(latent_cache.shape[0], -1).contiguous()
    num_tokens = latent_cache.shape[0]
    head_dim = latent_cache.shape[-1]
    if head_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA KV head_dim must be divisible by {FP4_BLOCK_SIZE}, got {head_dim}."
        )
    indices_len = metadata.fp4_mla_state.batch_indices.shape[0]
    positions_len = metadata.fp4_mla_state.positions.shape[0]
    if token_offset + num_tokens > indices_len or token_offset + num_tokens > positions_len:
        raise RuntimeError(
            f"FP4 MLA scatter would read batch_indices[{token_offset}:"
            f"{token_offset + num_tokens}] / positions[{token_offset}:"
            f"{token_offset + num_tokens}], but only {indices_len} / "
            f"{positions_len} entries are available. This indicates "
            "latent_cache was not pre-sliced to the current phase's token "
            "range (see MLA.forward_impl)."
        )

    _validate_fp4_mla_cache_shape(metadata.page_size, head_dim)

    backend = _fp4_mla_attention_backend()
    global_scale = _get_fp4_mla_global_scale(metadata, latent_cache.device)
    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    storage_head_dim = _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=backend,
    )
    sf_per_token = storage_head_dim // FP4_BLOCK_SIZE

    if phase not in ("context", "generation"):
        raise ValueError("FP4 MLA scatter requires phase='context' or 'generation'.")
    if getattr(metadata.fp4_mla_state, "v_scale_pool", None) is None:
        raise RuntimeError("FP4 MLA scatter requires the auxiliary V scale pool.")
    if metadata.page_size % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA scatter requires page_size divisible by "
            f"{FP4_BLOCK_SIZE}, got {metadata.page_size}."
        )
    if v_head_dim > head_dim:
        raise ValueError(f"FP4 MLA v_head_dim={v_head_dim} cannot exceed head_dim={head_dim}.")
    if head_dim - v_head_dim != FP4_MLA_K_RESIDUAL_DIM:
        raise ValueError(
            "FP4 MLA K residual quantization requires the K-only tail to match "
            f"the {FP4_MLA_K_RESIDUAL_DIM}-channel residual, got "
            f"head_dim={head_dim} v_head_dim={v_head_dim}."
        )
    if v_head_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA v_head_dim must be divisible by {FP4_BLOCK_SIZE}, got {v_head_dim}."
        )

    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    v_sf = get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=v_head_dim)
    num_dim_blocks = triton.cdiv(head_dim, FP4_BLOCK_SIZE)
    sf_per_page = metadata.page_size // FP4_BLOCK_SIZE

    generation_state = None
    generation_inputs = (rotary_cos_sin, q_pe, q_rope_out, q_quant_input)
    q_fp4_out = None
    q_sf_out = None
    if phase == "context":
        if any(arg is not None for arg in (q_pe, q_rope_out, q_quant_input)):
            raise ValueError("FP4 MLA context cache update does not accept generation Q tensors.")
        if (q_context is None) != (q_nope_head_dim is None):
            raise ValueError("FP4 MLA context Q and q_nope_head_dim must be provided together.")
        hp_pool_updated = False
    else:
        if q_context is not None or q_nope_head_dim is not None:
            raise ValueError("FP4 MLA generation cache update does not accept context Q tensors.")
        if (helix_position_offsets is None) != (helix_is_inactive_rank is None):
            raise ValueError(
                "FP4 MLA Helix position-offset and inactive-rank metadata "
                "must be provided together."
            )
        if not all(arg is not None for arg in generation_inputs):
            raise ValueError(
                "FP4 MLA generation requires rotary_cos_sin, q_pe, q_rope_out, "
                "and q_quant_input for fused RoPE, cache update, and Q quantization."
            )
        if not can_fuse_fp4_mla_q_quant(metadata, q_quant_input, q_pe, latent_cache):
            raise ValueError(
                "Fused FP4 MLA Q quantization received an unsupported shape, dtype, "
                "scale mode, or backend."
            )
        q_fp4_out, q_sf_out, q_batch_capacity = _prepare_fp4_mla_q_buffers(
            metadata,
            num_tokens,
            q_quant_input.shape[1],
            q_quant_input.device,
        )
        metadata.fp4_mla_state.prequantized_q = q_fp4_out
        metadata.fp4_mla_state.prequantized_q_sf = q_sf_out
        metadata.fp4_mla_state.q_batch_capacity = q_batch_capacity
        hp_pool_updated = True
    cutedsl_backend = _fp4_mla_attention_backend() == _FP4_MLA_CUTEDSL_BACKEND
    fused_v_transpose = cutedsl_backend and _fp4_mla_cutedsl_fused_v_transpose_enabled()
    kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
    persistent_v_packed = None
    v_packed_base = None
    v_page_offset = 0
    direct_v_packed_write = False
    if cutedsl_backend and not fused_v_transpose:
        persistent_v_packed = _get_cutedsl_persistent_v_packed_cache(
            metadata,
            local_layer,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=metadata.page_size,
            block_v=FP4_MLA_SCALE_ROW_GROUP,
        )
        # Reused/imported pages may not carry this process-local sidecar. Write
        # packed V directly only when the cache pages are exclusively owned.
        # The fused generation kernel handles uniform linear-MTP batches and
        # updates every 16-token tile touched by the verification window.
        num_gen = metadata.num_seqs - metadata.num_contexts
        block_reuse = getattr(kv_cache_manager, "enable_block_reuse", True)
        direct_context_v_packed_write = (
            phase == "context"
            and metadata.num_contexts > 0
            and num_tokens > 0
            and block_reuse is False
        )
        direct_generation_v_packed_write = (
            phase == "generation"
            and num_gen > 0
            and num_tokens >= num_gen
            and num_tokens % num_gen == 0
            and block_reuse is False
        )
        direct_v_packed_write = direct_context_v_packed_write or direct_generation_v_packed_write
        if direct_v_packed_write:
            v_packed_base = _get_fp4_mla_v_packed_pool_base(metadata)
            get_v_page_offset = getattr(kv_cache_manager, "get_mla_v_packed_page_offset", None)
            v_page_offset = (
                int(get_v_page_offset(local_layer))
                if callable(get_v_page_offset)
                else local_layer * kv_cache.shape[0]
            )
            expected_row_width = metadata.page_size // 2
            required_base_rows = (v_page_offset + kv_cache.shape[0]) * v_head_dim
            if (
                not isinstance(v_packed_base, torch.Tensor)
                or v_packed_base.dtype != torch.uint8
                or v_packed_base.device != kv_cache.device
                or v_packed_base.ndim != 2
                or v_packed_base.shape[0] < required_base_rows
                or v_packed_base.shape[1] != expected_row_width
                or not v_packed_base.is_contiguous()
            ):
                raise RuntimeError(
                    "FP4 MLA direct V-packed cache update requires the "
                    "stable full-pool base to be a contiguous uint8 tensor "
                    f"with at least {required_base_rows} rows and "
                    f"{expected_row_width} columns on {kv_cache.device}."
                )
            expected_layer_ptr = v_packed_base.data_ptr() + (
                v_page_offset * v_head_dim * expected_row_width
            )
            if expected_layer_ptr != persistent_v_packed.data_ptr():
                raise RuntimeError(
                    "FP4 MLA V-packed layer view does not match its stable "
                    "full-pool base and page offset."
                )
    v_pack_num_valid = None
    if phase == "context":
        _materialize_fp4_mla_device_page_table_for_forward(metadata)
        hp_pool_updated = _scatter_fp4_mla_kv_cache_2d_context(
            metadata,
            latent_cache,
            kv_cache,
            sf_cache,
            v_sf,
            global_scale,
            rotary_cos_sin,
            q_context,
            q_nope_head_dim,
            token_offset=token_offset,
            local_layer=local_layer,
            v_head_dim=v_head_dim,
            head_dim=head_dim,
            num_tokens=num_tokens,
            num_dim_blocks=num_dim_blocks,
            sf_per_token=sf_per_token,
            sf_per_page=sf_per_page,
            v_packed_base=v_packed_base,
            v_page_offset=v_page_offset,
        )
        v_pack_page_ids = metadata.fp4_mla_state.paged_kv_indices
    else:
        generation_state = _scatter_fp4_mla_kv_cache_2d_generation(
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
            rotary_cos_sin=rotary_cos_sin,
            q_pe=q_pe,
            q_rope_out=q_rope_out,
            q_quant_input=q_quant_input,
            q_fp4_out=q_fp4_out,
            q_sf_out=q_sf_out,
            v_packed_base=v_packed_base,
            v_page_offset=v_page_offset,
            helix_position_offsets=helix_position_offsets,
            helix_is_inactive_rank=helix_is_inactive_rank,
        )
        v_pack_page_ids = _fp4_mla_generation_page_ids(
            metadata, metadata.num_seqs - metadata.num_contexts
        )
        if getattr(metadata, "is_cuda_graph", False):
            # Frozen launch grids cannot follow the per-replay page count;
            # the repack kernels stride over this device-side count instead.
            v_pack_num_valid = _fp4_mla_generation_num_blocks_device(metadata)
    cutedsl_repack_page_indptr = None
    cutedsl_repack_kv_lens = None
    cutedsl_repack_generation_lens = None
    cutedsl_repack_max_touched_pages = 1
    if cutedsl_backend and not fused_v_transpose and not direct_v_packed_write:
        if phase == "context":
            num_contexts = metadata.num_contexts
            cutedsl_v_pack_page_ids = metadata.fp4_mla_state.paged_kv_indices[
                : metadata.fp4_mla_state.num_context_blocks
            ]
            cutedsl_repack_page_indptr = metadata.fp4_mla_state.paged_kv_indptr[: num_contexts + 1]
            cutedsl_repack_kv_lens = metadata.kv_lens_cuda_runtime[:num_contexts]
            cutedsl_repack_generation_lens = metadata.prompt_lens_cuda_runtime[:num_contexts]
            cutedsl_repack_max_touched_pages = int(
                metadata.fp4_mla_state.context_repack_max_touched_pages
            )
        elif generation_state is not None:
            kv_lens_gen, gen_lens_gen, generation_page_ids = generation_state
            num_gen = kv_lens_gen.numel()
            cutedsl_v_pack_page_ids = generation_page_ids
            cutedsl_repack_page_indptr = metadata.fp4_mla_state.paged_kv_indptr_decode
            cutedsl_repack_kv_lens = kv_lens_gen
            cutedsl_repack_generation_lens = gen_lens_gen
            cutedsl_repack_max_touched_pages = _ceil_div(
                num_tokens // num_gen + metadata.page_size - 1,
                metadata.page_size,
            )
        else:
            cutedsl_v_pack_page_ids = v_pack_page_ids
        _repack_cutedsl_v_packed_cache(
            persistent_v_packed,
            kv_cache,
            cutedsl_v_pack_page_ids,
            v_head_dim=v_head_dim,
            page_size=metadata.page_size,
            block_v=FP4_MLA_SCALE_ROW_GROUP,
            page_indptr=cutedsl_repack_page_indptr,
            kv_lens=cutedsl_repack_kv_lens,
            generation_lens=cutedsl_repack_generation_lens,
            max_touched_pages=cutedsl_repack_max_touched_pages,
        )
    _maybe_update_triton_v_packed_cache(
        metadata,
        layer_idx,
        kv_cache,
        v_pack_page_ids,
        num_queries=num_tokens,
        v_head_dim=v_head_dim,
        page_size=metadata.page_size,
        local_layer=local_layer,
        v_sf=v_sf[local_layer],
        num_valid_pages=v_pack_num_valid,
    )
    if phase == "generation":
        metadata.fp4_mla_state.generation_cache_scattered = hp_pool_updated
    return hp_pool_updated
