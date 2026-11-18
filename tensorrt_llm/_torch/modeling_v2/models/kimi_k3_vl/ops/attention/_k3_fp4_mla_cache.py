# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 family copy of the FP4 MLA backend code outside the KV-cache-manager contract.

Contents: FP4 MLA cache update (context / generation scatter with the fused Q quantization), chunked gather,
packed-V cache and FP8 context scratch.

Copied verbatim from ``tensorrt_llm/_torch/attention/backends/fp4_mla/`` (only the imports differ):
the KV-cache-manager contract modules (config, layout, metadata, packed_plan, state) stay imported.
``test_modeling_v2_k3_fp4_mla_copies.py`` checks each definition against its source.
"""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import torch
import triton
import triton.language as tl

# The FP4 MLA KV-cache layout contract shared with the KV cache manager (kept, not copied).
from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
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
    _fp4_mla_packed_rows_per_seq,
    _fp4_mla_q1_kv_blocks_per_program,
    _fp4_mla_q1_prefix_blocks_per_program,
    _HPUpdatePhase,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.layout import (
    _ensure_workspace_tensor,
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
from tensorrt_llm._torch.attention.backends.fp4_mla.metadata import (
    _fp4_mla_generation_hp_page_ids,
    _fp4_mla_generation_num_blocks_device,
    _fp4_mla_generation_page_ids,
    _fp4_mla_uniform_generation_lengths,
    _materialize_fp4_mla_device_page_table_for_forward,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.packed_plan import (
    _fp4_mla_helix_plan_bounds,
    _fp4_mla_packed_plan_geometry,
)
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.bindings import DataType

from ._k3_fp4_mla_kernels import (
    _fp4_mla_chunked_cache_gather_kernel,
    _fp4_mla_context_cache_update_kernel,
    _fp4_mla_generation_fused_qk_rope_cache_update_kernel,
    _fp8_mla_context_block_table_kernel,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata


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
    rows_per_seq: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return manager-owned, fixed-capacity packed-Q staging buffers.

    With ``rows_per_seq`` the buffers hold ``max_num_sequences`` request blocks
    of that many rows (zero-filled so padded rows stay finite); otherwise one
    row per (token, head) up to the token capacity.
    """
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
    if rows_per_seq > 0:
        q_rows = max_num_sequences * rows_per_seq
        q_key = f"q_{num_heads}_p{rows_per_seq}"
        q_sf_key = f"q_sf_{num_heads}_p{rows_per_seq}"
    else:
        q_rows = capacity * num_heads
        q_key = f"q_{num_heads}"
        q_sf_key = f"q_sf_{num_heads}"
    expected_q_shape = (q_rows, FP4_MLA_Q_PACKED_DIM)
    expected_q_sf_shape = (_get_fp4_mla_swizzled_scale_size(q_rows, FP4_MLA_Q_LOGICAL_DIM),)
    q_storage = buffers.get(q_key)
    q_sf_storage = buffers.get(q_sf_key)
    if q_storage is None and q_sf_storage is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Cannot allocate FP4 MLA Q buffers while capturing a CUDA graph. "
                "Run a warmup forward first."
            )
        allocate = torch.zeros if rows_per_seq > 0 else torch.empty
        q_storage = allocate(expected_q_shape, dtype=torch.uint8, device=canonical_device)
        q_sf_storage = allocate(
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
    v_sf: Optional[torch.Tensor],
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
    # A context-only cache keeps no V scales: bound pages by the layer's K
    # view and layers by the HP pool instead.
    store_v_scale = v_sf is not None
    num_pages = v_sf.shape[1] if store_v_scale else kv_cache.shape[0]
    num_layers = v_sf.shape[0] if store_v_scale else hp_pool.shape[1]

    _fp4_mla_context_cache_update_kernel[
        (
            num_tokens,
            num_dim_blocks + q_head_blocks,
        )
    ](
        kv_cache,
        sf_cache,
        v_sf if store_v_scale else sf_cache,
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
        num_pages,
        num_layers,
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
        v_sf.stride(0) if store_v_scale else 0,
        v_sf.stride(1) if store_v_scale else 0,
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
        STORE_V_SCALE=store_v_scale,
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
    # aliases already match). As the first FP4 MLA kernel of the step, the same
    # launch also writes the packed split plan the decode layers reuse.
    plan_geometry = None
    if q_pe.ndim == 3 and q_pe.shape[1] > 0:
        plan_geometry = _fp4_mla_packed_plan_geometry(
            metadata,
            num_heads=q_pe.shape[1],
            query_len=num_tokens // num_gen,
            num_requests=num_gen,
            device=latent_cache.device,
            helix_kv_bounds=_fp4_mla_helix_plan_bounds(
                metadata,
                num_tokens,
                use_helix=helix_position_offsets is not None or helix_is_inactive_rank is not None,
            ),
        )
    kv_lens_gen, gen_lens_gen = _fp4_mla_uniform_generation_lengths(
        metadata, num_tokens, num_gen, plan_geometry=plan_geometry
    )
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
    q_rows_per_seq = _fp4_mla_packed_rows_per_seq(num_q_heads, max_gen_len)
    if q_rows_per_seq and q_fp4_out.shape[0] < num_gen * q_rows_per_seq:
        raise RuntimeError(
            "FP4 MLA packed Q buffer is smaller than the generation batch: "
            f"{q_fp4_out.shape[0]} rows < {num_gen} x {q_rows_per_seq}."
        )
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
            Q_ROWS_PER_SEQ=q_rows_per_seq,
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
    consume.

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
        metadata.fp4_mla_state.prequantized_q_rows_per_seq = None
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
    kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
    # A disaggregated context worker keeps a single-copy cache without the
    # generation-only V scales and packed V.
    context_only_cache = getattr(kv_cache_manager, "fp4_mla_context_only_cache", False) is True
    if context_only_cache and phase != "context":
        raise RuntimeError("FP4 MLA context-only cache cannot execute generation.")
    if not context_only_cache and getattr(metadata.fp4_mla_state, "v_scale_pool", None) is None:
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
    v_sf = (
        None
        if context_only_cache
        else get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=v_head_dim)
    )
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
        num_gen_for_q = metadata.num_seqs - metadata.num_contexts
        q_rows_per_seq = 0
        if num_gen_for_q > 0 and num_tokens % num_gen_for_q == 0:
            q_rows_per_seq = _fp4_mla_packed_rows_per_seq(
                q_quant_input.shape[1], num_tokens // num_gen_for_q
            )
        q_fp4_out, q_sf_out, q_batch_capacity = _prepare_fp4_mla_q_buffers(
            metadata,
            num_tokens,
            q_quant_input.shape[1],
            q_quant_input.device,
            rows_per_seq=q_rows_per_seq,
        )
        metadata.fp4_mla_state.prequantized_q = q_fp4_out
        metadata.fp4_mla_state.prequantized_q_sf = q_sf_out
        metadata.fp4_mla_state.q_batch_capacity = q_batch_capacity
        metadata.fp4_mla_state.prequantized_q_rows_per_seq = q_rows_per_seq
        hp_pool_updated = True
    cutedsl_backend = _fp4_mla_attention_backend() == _FP4_MLA_CUTEDSL_BACKEND
    fused_v_transpose = cutedsl_backend and _fp4_mla_cutedsl_fused_v_transpose_enabled()
    persistent_v_packed = None
    v_packed_base = None
    v_page_offset = 0
    direct_v_packed_write = False
    if cutedsl_backend and not fused_v_transpose and not context_only_cache:
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
    if (
        cutedsl_backend
        and not fused_v_transpose
        and not direct_v_packed_write
        and not context_only_cache
    ):
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
    if context_only_cache:
        return hp_pool_updated
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


def _select_triton_block_v(num_queries: int, *, prefer_prepacked_v: bool = False) -> int:
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
    from ._k3_fp4_mla_v_repack import fp4_mla_repack_v_cache

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
    return _fp4_mla_attention_backend() == "triton"


def _triton_can_prepack_v(v_head_dim: int, page_size: int, block_v: int) -> bool:
    return (
        _triton_prepack_v_enabled()
        and hasattr(tl, "make_tensor_descriptor")
        and block_v in (32, 128)
        and v_head_dim % block_v == 0
        and page_size == FP4_MLA_TOKENS_PER_BLOCK
    )


def _triton_v_packed_attr(layer_idx: int) -> str:
    """The V-packed buffer every layer shares (one layer's V is packed at a time)."""
    del layer_idx
    return "_fp4_mla_triton_attention_v_packed_buf"


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
    valid_attr = _triton_shared_v_packed_valid_attr()
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
    from ._k3_fp4_mla_kernels import fp4_mla_repack_v_cache

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
    fp4_mla_repack_v_cache(
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


_FP8_CONTEXT_SUPPORTED_SMS = {90, 100, 103, 107, 120}


_FP8_CONTEXT_SCRATCH_ATTR = "_fp4_mla_fp8_context_scratch"


def require_fp4_mla_fp8_context_support() -> None:
    """Validate support for TRT-LLM's FP8 MLA context FMHA."""
    sm = get_sm_version()
    if sm not in _FP8_CONTEXT_SUPPORTED_SMS:
        supported_sms = ", ".join(
            str(supported_sm) for supported_sm in sorted(_FP8_CONTEXT_SUPPORTED_SMS)
        )
        raise RuntimeError(
            f"FP4 MLA context requires TRT-LLM FP8 FMHA, which is not "
            f"supported on SM{sm}. Supported SM versions: {supported_sms}."
        )


def _execute_fp8_context_with_cache_update(
    attention_fn: Callable[[], None],
    cache_update_fn: Callable[[], None],
    aux_stream: torch.cuda.Stream,
    start_event: torch.cuda.Event,
    done_event: torch.cuda.Event,
) -> None:
    """Overlap FP8 context attention with its independent FP4 cache update."""
    current_stream = torch.cuda.current_stream()
    if aux_stream == current_stream:
        raise RuntimeError(
            "FP4 MLA context cache update requires a distinct auxiliary CUDA stream."
        )

    start_event.record(current_stream)
    # Launch the auxiliary work first so the main stream cannot drain the
    # attention queue before the cache update has been submitted.
    joined = False
    try:
        with torch.cuda.stream(aux_stream):
            aux_stream.wait_event(start_event)
            cache_update_fn()
            done_event.record(aux_stream)
        attention_fn()
        current_stream.wait_event(done_event)
        joined = True
    finally:
        if not joined:
            # Either callback may enqueue work before raising, and done_event
            # may not have been recorded. Drain before scratch or KV pages can
            # be released/reused; the successful path stays asynchronous.
            aux_stream.synchronize()


@dataclass(frozen=True, slots=True)
class _Fp8MlaContextCacheManagerView:
    """Minimal cache-manager view consumed by TRT-LLM FP8 context FMHA."""

    tokens_per_block: int
    max_seq_len: int
    kv_cache_pool_pointers: torch.Tensor
    kv_cache_pool_mapping: torch.Tensor
    layer_offsets: Tuple[int, ...]
    dtype: DataType = DataType.FP8
    # The scratch holds one layer with no sliding window, so TRT-LLM attention
    # falls back to the partition's max_seq_len for the attention window.
    max_attention_window_vec: Tuple[Optional[int], ...] = (None,)


@dataclass
class _Fp8MlaContextScratch:
    pool: torch.Tensor
    block_offsets: torch.Tensor
    block_ids_per_seq: torch.Tensor
    host_pool_pointers: torch.Tensor
    host_pool_mapping: torch.Tensor
    host_total_kv_lens: torch.Tensor
    max_num_sequences: int
    max_blocks_per_seq: int
    capacity_blocks: int
    max_num_tokens: int
    page_size: int
    head_dim: int
    cache_stream: torch.cuda.Stream
    cache_start_event: torch.cuda.Event
    cache_done_event: torch.cuda.Event
    cache_manager_view: _Fp8MlaContextCacheManagerView
    mapping_signature: Optional[Tuple[int, ...]] = None

    @classmethod
    def create(
        cls,
        meta: "TrtllmAttentionMetadata",
        *,
        device: torch.device,
        head_dim: int,
    ) -> "_Fp8MlaContextScratch":
        kv_cache_manager = meta.kv_cache_manager
        if kv_cache_manager is None:
            raise RuntimeError("FP8 MLA context scratch requires a KV cache manager.")

        page_size = int(meta.tokens_per_block)
        max_num_sequences = int(meta.max_num_sequences or meta.max_num_requests)
        max_blocks_per_seq = int(kv_cache_manager.max_blocks_per_seq)
        max_num_tokens = int(meta.max_num_tokens)
        runtime_features = meta.runtime_features
        if runtime_features.chunked_prefill:
            max_num_tokens *= max(
                1,
                int(runtime_features.chunked_prefill_buffer_batch_size),
            )
        max_nonempty_sequences = min(max_num_sequences, max_num_tokens)
        capacity_blocks = max(
            1,
            (max_num_tokens + page_size - 1) // page_size + max(0, max_nonempty_sequences - 1),
        )
        pool = torch.empty(
            capacity_blocks * page_size * head_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        block_offsets = torch.zeros(
            1,
            max_num_sequences,
            2,
            max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )
        block_ids_per_seq = torch.zeros(
            max_num_sequences,
            max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )
        host_pool_pointers = torch.tensor(
            [[pool.data_ptr(), 0]],
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        host_pool_mapping = torch.zeros(
            1,
            2,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        cache_manager_view = _Fp8MlaContextCacheManagerView(
            tokens_per_block=page_size,
            max_seq_len=int(kv_cache_manager.max_seq_len),
            kv_cache_pool_pointers=host_pool_pointers,
            kv_cache_pool_mapping=host_pool_mapping,
            layer_offsets=(0,),
        )
        host_total_kv_lens = torch.zeros(
            2,
            dtype=meta.host_total_kv_lens.dtype,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        return cls(
            pool=pool,
            block_offsets=block_offsets,
            block_ids_per_seq=block_ids_per_seq,
            host_pool_pointers=host_pool_pointers,
            host_pool_mapping=host_pool_mapping,
            host_total_kv_lens=host_total_kv_lens,
            max_num_sequences=max_num_sequences,
            max_blocks_per_seq=max_blocks_per_seq,
            capacity_blocks=capacity_blocks,
            max_num_tokens=max_num_tokens,
            page_size=page_size,
            head_dim=head_dim,
            cache_stream=torch.cuda.Stream(device=device),
            cache_start_event=torch.cuda.Event(),
            cache_done_event=torch.cuda.Event(),
            cache_manager_view=cache_manager_view,
        )

    def matches(
        self,
        meta: "TrtllmAttentionMetadata",
        *,
        device: torch.device,
        head_dim: int,
    ) -> bool:
        kv_cache_manager = meta.kv_cache_manager
        required_max_num_tokens = int(meta.max_num_tokens)
        if meta.runtime_features.chunked_prefill:
            required_max_num_tokens *= max(
                1,
                int(meta.runtime_features.chunked_prefill_buffer_batch_size),
            )
        return (
            kv_cache_manager is not None
            and self.pool.device == device
            and self.head_dim == head_dim
            and self.max_num_tokens >= required_max_num_tokens
            and self.page_size == meta.tokens_per_block
            and self.cache_manager_view.max_seq_len == int(kv_cache_manager.max_seq_len)
            and self.max_num_sequences >= int(meta.max_num_sequences or meta.max_num_requests)
            and self.max_blocks_per_seq >= int(kv_cache_manager.max_blocks_per_seq)
        )

    def prepare(
        self,
        meta: "TrtllmAttentionMetadata",
        *,
        context_lengths_cuda: Optional[torch.Tensor] = None,
        context_lengths_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        if context_lengths_cuda is None:
            context_lengths_cuda = meta.prompt_lens_cuda_runtime
        if context_lengths_cpu is None:
            context_lengths_cpu = meta.prompt_lens_cpu_runtime
        if (
            context_lengths_cpu.device.type != "cpu"
            or context_lengths_cpu.dtype != torch.int32
            or context_lengths_cpu.ndim != 1
            or context_lengths_cpu.numel() < meta.num_contexts
        ):
            raise ValueError("FP8 MLA context metadata requires a CPU int32 context-length tensor.")
        context_lengths = tuple(
            int(length) for length in context_lengths_cpu[: meta.num_contexts].tolist()
        )
        self.host_total_kv_lens[0] = sum(context_lengths)
        self.host_total_kv_lens[1] = 0
        if context_lengths == self.mapping_signature:
            return

        if len(context_lengths) > self.max_num_sequences:
            raise ValueError(
                f"FP8 MLA context scratch supports at most {self.max_num_sequences} sequences, "
                f"got {len(context_lengths)}."
            )
        required_blocks = 0
        for context_length in context_lengths:
            if context_length < 0:
                raise ValueError(f"Context length must be non-negative, got {context_length}.")
            num_blocks = (context_length + self.page_size - 1) // self.page_size
            if num_blocks > self.max_blocks_per_seq:
                raise ValueError(
                    f"Context sequence requires {num_blocks} FP8 scratch blocks, but the "
                    f"page table holds only {self.max_blocks_per_seq}."
                )
            required_blocks += num_blocks
        if required_blocks > self.capacity_blocks:
            raise RuntimeError(
                f"FP8 MLA context scratch requires {required_blocks} blocks, but only "
                f"{self.capacity_blocks} were allocated."
            )

        if (
            context_lengths_cuda.dtype != torch.int32
            or not context_lengths_cuda.is_cuda
            or context_lengths_cuda.ndim != 1
            or context_lengths_cuda.stride(0) != 1
            or context_lengths_cuda.numel() < meta.num_contexts
        ):
            raise ValueError(
                "FP8 MLA context metadata requires a contiguous CUDA int32 context-length tensor."
            )
        # ModelEngine's context warmup reaches this launch before serving.
        # Only engine-lifetime capacities are constexpr; varying request
        # counts and sequence lengths reuse the warmup specialization.
        _fp8_mla_context_block_table_kernel[(self.max_num_sequences,)](
            context_lengths_cuda,
            self.block_offsets,
            self.block_ids_per_seq,
            meta.num_contexts,
            PAGE_SIZE=self.page_size,
            MAX_BLOCKS_PER_SEQUENCE=self.max_blocks_per_seq,
            SEQUENCE_BLOCK=triton.next_power_of_2(self.max_num_sequences),
            PAGE_BLOCK=triton.next_power_of_2(self.max_blocks_per_seq),
            num_warps=8,
        )
        self.mapping_signature = context_lengths


def _build_fp8_mla_context_metadata(
    meta: "TrtllmAttentionMetadata",
    scratch: _Fp8MlaContextScratch,
    *,
    kv_lens_cuda: Optional[torch.Tensor] = None,
    kv_lens_cpu: Optional[torch.Tensor] = None,
) -> "TrtllmAttentionMetadata":
    """Route the mandatory FP8 cache write through a direct metadata view."""
    if kv_lens_cuda is None:
        kv_lens_cuda = meta.prompt_lens_cuda_runtime[: meta.num_contexts]
    if kv_lens_cpu is None:
        kv_lens_cpu = meta.prompt_lens_cpu_runtime[: meta.num_contexts]
    fp8_meta = copy.copy(meta)
    fp8_meta.fp4_mla_state = None
    fp8_meta.kv_cache_manager = scratch.cache_manager_view
    fp8_meta.kv_cache_block_offsets = scratch.block_offsets
    fp8_meta.block_ids_per_seq = scratch.block_ids_per_seq
    fp8_meta.prompt_lens_cuda_runtime = meta.prompt_lens_cuda_runtime[: meta.num_contexts]
    fp8_meta.prompt_lens_cpu_runtime = meta.prompt_lens_cpu_runtime[: meta.num_contexts]
    fp8_meta.host_request_types_runtime = meta.host_request_types_runtime[: meta.num_contexts]
    # The disposable cache represents only this context invocation. Expose
    # exact context-only lengths so cached prefixes, trailing generation
    # metadata, and stale totals cannot extend FP8 K/V quantization.
    fp8_meta.kv_lens_cuda_runtime = kv_lens_cuda
    fp8_meta.kv_lens_runtime = kv_lens_cpu
    fp8_meta.host_total_kv_lens = scratch.host_total_kv_lens
    # Scratch lengths intentionally start from zero. Preserve the actual
    # absolute positions for Q/K RoPE through the native kernel's explicit
    # per-token position-offset input.
    fp8_meta.helix_position_offsets = meta.fp4_mla_state.positions[: meta.num_ctx_tokens]
    return fp8_meta


def _get_fp8_mla_context_metadata(
    meta: "TrtllmAttentionMetadata",
    scratch: _Fp8MlaContextScratch,
) -> "TrtllmAttentionMetadata":
    """Prepare and reuse one direct metadata view for all layers in a step."""
    state = meta.fp4_mla_state.fp8_context_state
    if state is not None and state[0] is scratch:
        return state[1]

    scratch.prepare(meta)
    fp8_meta = _build_fp8_mla_context_metadata(meta, scratch)
    meta.fp4_mla_state.fp8_context_state = (scratch, fp8_meta)
    return fp8_meta
