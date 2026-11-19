# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP4 MLA generation cache update of an FP4-KV MLA layer: one Triton launch per layer and step.

The served update (6904cd9672, ``attention/backends/fp4_mla/cache_update.py``
``scatter_fp4_mla_kv_cache``, generation phase, and ``TrtllmAttention._fp4_mla_rope_generation`` /
``fp4_mla_kv_append_generation``) for the deployment's configuration: the CuTe DSL backend with the
packed query layout, exclusively owned cache pages (block reuse off: packed V is written directly),
no fused V transpose. ``_fp4_mla_generation_fused_qk_rope_cache_update_kernel`` (``kernels/
fp4_mla_kernels``, the served file) appends the step's latent rows to the FP4 cache, the HP ring,
the V scales and packed V, and

* :func:`rope_generation` (batches the fused query does not take, 24 / 32 rows): also quantizes the
  query ``[T, H, 512 | 64]`` into the packed FP4 query rows the decode reads (``WRITE_Q_ROWS``);
* :func:`kv_append_generation` (the fused query, at most 16 rows): leaves the query rows to
  ``k3_mla_q_fused``, which writes them on the main stream while this runs on the aux stream.

Both prepare the packed query buffers on ``metadata.fp4_mla_state`` for the decode
(``_fp4_mla_decode``). As the first FP4 MLA kernel of a step without the early step metadata, the
update also launches the generation lengths fused with the decode's task plan and the device page
table (``_fp4_mla_decode.uniform_generation_lengths`` / ``materialize_page_table``).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import triton

from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
    FP4_BLOCK_SIZE,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_Q1_PREFIX_BLOCK_DIM,
    FP4_MLA_Q_PREFIX_BLOCK_DIM,
    FP4_MLA_Q_PREFIX_DIM,
    FP4_MLA_Q_SF_GROUPS,
    FP4_MLA_SCALE_ROW_GROUP,
    _fp4_mla_q1_kv_blocks_per_program,
    _fp4_mla_q1_prefix_blocks_per_program,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.layout import (
    _get_fp4_mla_global_scale,
    _get_fp4_mla_hp_pool_layout,
    _get_fp4_mla_kv_cache_tensors,
    _get_fp4_mla_q_global_scale,
    _validate_fp4_mla_cache_shape,
    _validate_fp4_mla_hp_generation_width,
    _validate_fp4_mla_kv_storage_shape,
    get_fp4_mla_v_scale_pool_view,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.metadata import (
    _fp4_mla_generation_hp_page_ids,
    _fp4_mla_generation_num_blocks_device,
    _fp4_mla_generation_page_ids,
)

from ...kernels.k3r_fp4_mla.fp4_mla_kernels import (
    _fp4_mla_generation_fused_qk_rope_cache_update_kernel,
)
from . import _fp4_mla_decode_ull as _decode
from ._k3_fp4_mla_cache import (
    _get_cutedsl_persistent_v_packed_cache,
    _get_fp4_mla_v_packed_pool_base,
    _maybe_update_triton_v_packed_cache,
    _prepare_fp4_mla_q_buffers,
    _repack_cutedsl_v_packed_cache,
    can_fuse_fp4_mla_q_quant,
)


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


# The CuTe DSL backend stores the K residual (served ``_FP4_MLA_K_RESIDUAL_BACKENDS``) and runs
# without the fused V transpose (TRTLLM_FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE, unset in 1007804).
_STORE_K_RESIDUAL = True
_FUSED_V_TRANSPOSE = False


def rope_generation(
    mqa: Any,
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    metadata: Any,
) -> None:
    """The generation cache update of the FP4 MLA layer behind ``mqa`` (its generation
    ``K3MlaAttnState``) with the query rows: identity RoPE of ``q_pe`` into ``fused_q[..., 512:]``
    and the packed FP4 query of ``fused_q`` (``[T, H, 576]``, W_UK written) for the decode. Served
    ``TrtllmAttention.mla_rope_generation(..., fuse_fp4_q_quant=True)`` on an FP4 cache."""
    kv_lora_rank, rope_dim = mqa.kv_lora_rank, mqa.qk_rope_head_dim
    mqa._ensure_rope_table_size(metadata.max_seq_len)
    if q_pe.shape[-1] != rope_dim:
        raise RuntimeError(f"FP4 MLA q_pe last dimension must be {rope_dim}, got {q_pe.shape[-1]}.")
    if latent_cache.shape[-1] != kv_lora_rank + rope_dim:
        raise RuntimeError(
            f"FP4 MLA latent_cache last dimension must be {kv_lora_rank + rope_dim}, got "
            f"{latent_cache.shape[-1]}."
        )
    if not mqa.rope_duplicate_data:
        raise RuntimeError("FP4 MLA requires RopeParams.duplicate_data=True.")
    if not (
        mqa.uses_fp4_mla_attention
        and can_fuse_fp4_mla_q_quant(metadata, fused_q, q_pe, latent_cache)
    ):
        raise RuntimeError("FP4 MLA fused Q quantization eligibility changed before launch.")
    if (
        mqa.rotary_cos_sin is None
        or q_pe.dtype != torch.bfloat16
        or fused_q.dtype != torch.bfloat16
        or latent_cache.dtype != torch.bfloat16
        or mqa.rotary_cos_sin.dtype != torch.float32
    ):
        raise RuntimeError(
            "FP4 MLA generation requires fused BF16 RoPE/cache update with an FP32 rotary table."
        )
    if not _scatter_generation(
        metadata,
        latent_cache,
        mqa,
        q_pe=q_pe,
        q_rope_out=fused_q[..., kv_lora_rank:],
        q_quant_input=fused_q,
        q_rows_external=False,
    ):
        raise RuntimeError("Fused FP4 MLA RoPE/cache scatter did not update the HP pool.")


def kv_append_generation(
    mqa: Any, latent_cache: torch.Tensor, metadata: Any, num_q_heads: int
) -> None:
    """The KV half of :func:`rope_generation`: append this step's latent rows of the FP4 MLA layer
    behind ``mqa`` and prepare (not write) the packed query buffers of ``num_q_heads`` heads.
    Served ``TrtllmAttention.fp4_mla_kv_append_generation``."""
    kv_lora_rank, rope_dim = mqa.kv_lora_rank, mqa.qk_rope_head_dim
    if not mqa.uses_fp4_mla_attention:
        raise RuntimeError("The FP4 MLA KV append requires FP4 MLA attention.")
    if latent_cache.shape[-1] != kv_lora_rank + rope_dim:
        raise RuntimeError(
            f"FP4 MLA latent_cache last dimension must be {kv_lora_rank + rope_dim}, got "
            f"{latent_cache.shape[-1]}."
        )
    if (
        mqa.rotary_cos_sin is None
        or latent_cache.dtype != torch.bfloat16
        or mqa.rotary_cos_sin.dtype != torch.float32
    ):
        raise RuntimeError(
            "FP4 MLA generation requires a BF16 latent cache and an FP32 rotary table."
        )
    if metadata.helix_position_offsets is not None and metadata.helix_is_inactive_rank is None:
        raise RuntimeError("FP4 MLA Helix requires inactive-rank metadata.")
    mqa._ensure_rope_table_size(metadata.max_seq_len)
    num_tokens = latent_cache.shape[0]
    # Shape-only stand-ins for the query tensors the KV-only update never reads.
    stub = latent_cache.new_empty(1)
    q_pe = stub.expand(num_tokens, num_q_heads, rope_dim)
    q_input = stub.expand(num_tokens, num_q_heads, kv_lora_rank + rope_dim)
    if not _scatter_generation(
        metadata,
        latent_cache,
        mqa,
        q_pe=q_pe,
        q_rope_out=q_pe,
        q_quant_input=q_input,
        q_rows_external=True,
    ):
        raise RuntimeError("FP4 MLA generation KV append did not update the HP pool.")


def _scatter_generation(
    metadata: Any,
    latent_cache: torch.Tensor,
    mqa: Any,
    *,
    q_pe: torch.Tensor,
    q_rope_out: torch.Tensor,
    q_quant_input: torch.Tensor,
    q_rows_external: bool,
) -> bool:
    """Served ``scatter_fp4_mla_kv_cache(phase="generation")``; returns whether it updated the HP
    pool."""
    state = metadata.fp4_mla_state
    state.generation_cache_scattered = False
    state.prequantized_q = None
    state.prequantized_q_sf = None
    state.q_batch_capacity = None
    state.prequantized_q_rows_per_seq = None
    if latent_cache.numel() == 0:
        raise ValueError("FP4 MLA cache scatter requires at least one latent token.")
    layer_idx = mqa.layer_idx
    local_layer = mqa.get_fp4_mla_local_layer_idx(metadata)
    v_head_dim = mqa.kv_lora_rank
    rotary_cos_sin = mqa.rotary_cos_sin
    helix_position_offsets = metadata.helix_position_offsets
    helix_is_inactive_rank = metadata.helix_is_inactive_rank
    token_offset = getattr(metadata, "num_ctx_tokens", 0)

    latent_cache = latent_cache.reshape(latent_cache.shape[0], -1).contiguous()
    num_tokens = latent_cache.shape[0]
    head_dim = latent_cache.shape[-1]
    if head_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA KV head_dim must be divisible by {FP4_BLOCK_SIZE}, got {head_dim}."
        )
    indices_len = state.batch_indices.shape[0]
    positions_len = state.positions.shape[0]
    if token_offset + num_tokens > indices_len or token_offset + num_tokens > positions_len:
        raise RuntimeError(
            f"FP4 MLA scatter would read batch_indices[{token_offset}:"
            f"{token_offset + num_tokens}] / positions[{token_offset}:"
            f"{token_offset + num_tokens}], but only {indices_len} / {positions_len} entries are "
            "available: latent_cache must hold the generation tokens only."
        )
    _validate_fp4_mla_cache_shape(metadata.page_size, head_dim)
    global_scale = _get_fp4_mla_global_scale(metadata, latent_cache.device)
    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    storage_head_dim = _validate_fp4_mla_kv_storage_shape(
        kv_cache, sf_cache, head_dim=head_dim, backend=_decode.CUTEDSL_BACKEND
    )
    sf_per_token = storage_head_dim // FP4_BLOCK_SIZE
    kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
    if getattr(kv_cache_manager, "fp4_mla_context_only_cache", False) is True:
        raise RuntimeError("FP4 MLA context-only cache cannot execute generation.")
    if getattr(state, "v_scale_pool", None) is None:
        raise RuntimeError("FP4 MLA scatter requires the auxiliary V scale pool.")
    if metadata.page_size % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA scatter requires page_size divisible by {FP4_BLOCK_SIZE}, got "
            f"{metadata.page_size}."
        )
    if head_dim - v_head_dim != FP4_MLA_K_RESIDUAL_DIM or v_head_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP4 MLA K residual quantization requires a {FP4_MLA_K_RESIDUAL_DIM}-channel K-only "
            f"tail and a V width divisible by {FP4_BLOCK_SIZE}, got head_dim={head_dim} "
            f"v_head_dim={v_head_dim}."
        )
    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    v_sf = get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=v_head_dim)
    num_dim_blocks = triton.cdiv(head_dim, FP4_BLOCK_SIZE)
    sf_per_page = metadata.page_size // FP4_BLOCK_SIZE

    if (helix_position_offsets is None) != (helix_is_inactive_rank is None):
        raise ValueError(
            "FP4 MLA Helix position-offset and inactive-rank metadata must be provided together."
        )
    if rotary_cos_sin is None:
        raise ValueError("FP4 MLA generation requires rotary_cos_sin for fused RoPE.")
    if not q_rows_external and not can_fuse_fp4_mla_q_quant(
        metadata, q_quant_input, q_pe, latent_cache
    ):
        raise ValueError(
            "Fused FP4 MLA Q quantization received an unsupported shape, dtype, scale mode, or "
            "backend."
        )
    num_gen_for_q = metadata.num_seqs - metadata.num_contexts
    q_rows_per_seq = 0
    if num_gen_for_q > 0 and num_tokens % num_gen_for_q == 0:
        q_rows_per_seq = _decode.packed_rows_per_seq(
            q_quant_input.shape[1], num_tokens // num_gen_for_q
        )
    q_fp4_out, q_sf_out, q_batch_capacity = _prepare_fp4_mla_q_buffers(
        metadata,
        num_tokens,
        q_quant_input.shape[1],
        q_quant_input.device,
        rows_per_seq=q_rows_per_seq,
    )
    state.prequantized_q = q_fp4_out
    state.prequantized_q_sf = q_sf_out
    state.q_batch_capacity = q_batch_capacity
    state.prequantized_q_rows_per_seq = q_rows_per_seq
    # Whether another kernel writes this step's query rows (features.ull2_mla_decode_pdl).
    _decode._step(metadata).q_rows_external = bool(q_rows_external)

    persistent_v_packed = None
    v_packed_base = None
    v_page_offset = 0
    direct_v_packed_write = False
    if not _FUSED_V_TRANSPOSE:
        persistent_v_packed = _get_cutedsl_persistent_v_packed_cache(
            metadata,
            local_layer,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=metadata.page_size,
            block_v=FP4_MLA_SCALE_ROW_GROUP,
        )
        # Reused/imported pages may not carry this process-local sidecar. Write packed V directly
        # only when the cache pages are exclusively owned.
        num_gen = metadata.num_seqs - metadata.num_contexts
        block_reuse = getattr(kv_cache_manager, "enable_block_reuse", True)
        direct_v_packed_write = (
            num_gen > 0
            and num_tokens >= num_gen
            and num_tokens % num_gen == 0
            and block_reuse is False
        )
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
                    "FP4 MLA direct V-packed cache update requires the stable full-pool base to "
                    f"be a contiguous uint8 tensor with at least {required_base_rows} rows and "
                    f"{expected_row_width} columns on {kv_cache.device}."
                )
            expected_layer_ptr = v_packed_base.data_ptr() + (
                v_page_offset * v_head_dim * expected_row_width
            )
            if expected_layer_ptr != persistent_v_packed.data_ptr():
                raise RuntimeError(
                    "FP4 MLA V-packed layer view does not match its stable full-pool base and "
                    "page offset."
                )
    generation_state = _scatter_2d_generation(
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
        write_q_rows=not q_rows_external,
    )
    v_pack_page_ids = _fp4_mla_generation_page_ids(
        metadata, metadata.num_seqs - metadata.num_contexts
    )
    v_pack_num_valid = None
    if getattr(metadata, "is_cuda_graph", False):
        # Frozen launch grids cannot follow the per-replay page count; the repack kernels stride
        # over this device-side count instead.
        v_pack_num_valid = _fp4_mla_generation_num_blocks_device(metadata)
    if not _FUSED_V_TRANSPOSE and not direct_v_packed_write:
        # Pages shared through block reuse: repack V after the update (not the deployment's).
        repack_page_indptr = None
        repack_kv_lens = None
        repack_generation_lens = None
        repack_max_touched_pages = 1
        if generation_state is not None:
            kv_lens_gen, gen_lens_gen, generation_page_ids = generation_state
            num_gen = kv_lens_gen.numel()
            repack_page_ids = generation_page_ids
            repack_page_indptr = state.paged_kv_indptr_decode
            repack_kv_lens = kv_lens_gen
            repack_generation_lens = gen_lens_gen
            repack_max_touched_pages = _ceil_div(
                num_tokens // num_gen + metadata.page_size - 1, metadata.page_size
            )
        else:
            repack_page_ids = v_pack_page_ids
        _repack_cutedsl_v_packed_cache(
            persistent_v_packed,
            kv_cache,
            repack_page_ids,
            v_head_dim=v_head_dim,
            page_size=metadata.page_size,
            block_v=FP4_MLA_SCALE_ROW_GROUP,
            page_indptr=repack_page_indptr,
            kv_lens=repack_kv_lens,
            generation_lens=repack_generation_lens,
            max_touched_pages=repack_max_touched_pages,
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
    state.generation_cache_scattered = True
    return True


def _scatter_2d_generation(
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
    write_q_rows: bool,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Served ``_scatter_fp4_mla_kv_cache_2d_generation``: the step metadata (lengths fused with
    the decode's plan, page table) where still pending, then the one update kernel."""
    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    num_gen = num_seqs - num_contexts
    if num_gen <= 0:
        return None
    if num_tokens < num_gen or num_tokens % num_gen != 0:
        raise NotImplementedError(
            "FP4 MLA generation scatter requires a uniform linear MTP generation length, got "
            f"{num_tokens} tokens for {num_gen} sequences."
        )
    # The prompt_lens/kv_lens runtime aliases can lag at the decode anchor (seq_lens == 1) under
    # CUDA graph / one-engine MTP while each generation sequence really appends
    # num_tokens // num_gen tokens this step. As the first FP4 MLA kernel of the step, the same
    # launch also writes the packed split plan the decode layers reuse.
    use_helix = helix_position_offsets is not None or helix_is_inactive_rank is not None
    plan_geometry = None
    if q_pe.ndim == 3 and q_pe.shape[1] > 0:
        plan_geometry = _decode.packed_plan_geometry(
            metadata,
            num_heads=q_pe.shape[1],
            query_len=num_tokens // num_gen,
            num_requests=num_gen,
            device=latent_cache.device,
            helix_kv_bounds=_decode.helix_plan_bounds(metadata, num_tokens, use_helix=use_helix),
        )
    kv_lens_gen, gen_lens_gen = _decode.uniform_generation_lengths(
        metadata, num_tokens, num_gen, plan_geometry=plan_geometry
    )
    _decode.materialize_page_table(metadata, kv_lens_gen)

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
                "FP4 MLA Helix position offsets must be a contiguous same-device int32 tensor "
                "covering every generation token."
            )
        if (
            helix_is_inactive_rank.dtype != torch.bool
            or helix_is_inactive_rank.device != latent_cache.device
            or helix_is_inactive_rank.ndim != 1
            or helix_is_inactive_rank.numel() < num_gen
            or not helix_is_inactive_rank.is_contiguous()
        ):
            raise ValueError(
                "FP4 MLA Helix inactive-rank metadata must be a contiguous same-device bool "
                "tensor covering every generation sequence."
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
                    "FP4 MLA speculative Helix local slots must be a contiguous same-device "
                    "int32 tensor covering every generation token."
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
    # Grouped Q1 kernels reuse K's packed codes for V. Keep one dimension block per program until
    # their warp-specialized V quantizer is split out.
    if max_gen_len == 1:
        q1_kv_blocks_per_program = _fp4_mla_q1_kv_blocks_per_program(num_gen, v_head_dim)
    q_prefix_block_dim = (
        FP4_MLA_Q1_PREFIX_BLOCK_DIM if max_gen_len == 1 else FP4_MLA_Q_PREFIX_BLOCK_DIM
    )
    q_prefix_blocks = FP4_MLA_Q_PREFIX_DIM // q_prefix_block_dim
    q_prefix_blocks_per_program = _fp4_mla_q1_prefix_blocks_per_program(
        num_gen, q1_kv_blocks_per_program
    )
    q_work_blocks = q_prefix_blocks // q_prefix_blocks_per_program + 1
    if latent_cache.dtype != torch.bfloat16:
        raise TypeError(
            "Fused FP4 MLA Q/K RoPE and cache storage requires BF16 latent KV, got "
            f"{latent_cache.dtype}."
        )
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(
            "Fused FP4 MLA K RoPE requires a positive even K tail, got "
            f"head_dim={head_dim} v_head_dim={v_head_dim}."
        )
    if (
        rotary_cos_sin.device != latent_cache.device
        or rotary_cos_sin.dtype != torch.float32
        or rotary_cos_sin.numel() % (rope_dim * 2) != 0
    ):
        raise ValueError(
            "Fused FP4 MLA K RoPE requires a same-device FP32 rotary table with rows of "
            f"{rope_dim * 2} values."
        )
    if (
        q_pe.dtype != torch.bfloat16
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
            "Fused FP4 MLA Q RoPE requires same-device BF16 q_pe and q_rope_out tensors shaped "
            f"[tokens, heads, {rope_dim}]."
        )
    num_q_heads = q_pe.shape[1]
    q_head_blocks = _ceil_div(num_q_heads, block_q_heads)
    q_rows_per_seq = _decode.packed_rows_per_seq(num_q_heads, max_gen_len)
    if q_rows_per_seq and q_fp4_out.shape[0] < num_gen * q_rows_per_seq:
        raise RuntimeError(
            "FP4 MLA packed Q buffer is smaller than the generation batch: "
            f"{q_fp4_out.shape[0]} rows < {num_gen} x {q_rows_per_seq}."
        )
    max_gen_tiles = _ceil_div(max_gen_len + FP4_BLOCK_SIZE - 1, FP4_BLOCK_SIZE)
    q_global_scale = _get_fp4_mla_q_global_scale(metadata, latent_cache.device)
    write_v_packed = v_packed_base is not None
    v_packed_output = v_packed_base if write_v_packed else kv_cache
    v_packed_s0 = v_packed_output.stride(0) if write_v_packed else 0
    v_packed_s1 = v_packed_output.stride(1) if write_v_packed else 0
    kv_work_blocks = (
        v_head_dim // FP4_BLOCK_SIZE // q1_kv_blocks_per_program + 1
        if max_gen_len == 1
        else num_dim_blocks
    )
    grid = (
        num_gen,
        max(kv_work_blocks, max_gen_len * q_head_blocks * q_work_blocks if write_q_rows else 0),
    )
    _fp4_mla_generation_fused_qk_rope_cache_update_kernel[grid](
        kv_cache,
        sf_cache,
        v_sf,
        v_packed_output,
        pool,
        latent_cache,
        global_scale,
        q_global_scale,
        rotary_cos_sin,
        q_pe,
        q_rope_out,
        q_quant_input,
        q_fp4_out,
        q_sf_out,
        kv_lens_gen,
        gen_lens_gen,
        helix_position_offsets,
        helix_local_slots,
        helix_is_inactive_rank,
        page_ids,
        hp_page_ids,
        metadata.fp4_mla_state.paged_kv_indptr_decode,
        page_ids.shape[0],
        hp_page_ids.numel(),
        metadata.fp4_mla_state.paged_kv_indptr_decode.shape[0],
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
        q_pe.stride(0),
        q_pe.stride(1) if q_pe.ndim > 1 else 0,
        q_pe.stride(2) if q_pe.ndim > 2 else 0,
        q_rope_out.stride(0),
        q_rope_out.stride(1) if q_rope_out.ndim > 1 else 0,
        q_rope_out.stride(2) if q_rope_out.ndim > 2 else 0,
        HEAD_D=hp_head_dim,
        V_HEAD_D=v_head_dim,
        HP_BLOCK=FP4_BLOCK_SIZE,
        HP_POOL_SIZE=hp_pool_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_TOKEN=sf_per_token,
        SF_PER_PAGE=sf_per_page,
        K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
        STORE_K_RESIDUAL=_STORE_K_RESIDUAL,
        FUSE_ROPE_CACHE_STORE=True,
        USE_HELIX=use_helix,
        USE_HELIX_LOCAL_SLOTS=use_helix_local_slots,
        WRITE_V_PACKED=write_v_packed,
        MAX_GEN_TILES=max(max_gen_tiles, 1),
        ROPE_DIM=rope_dim,
        ROPE_PAIR_BLOCK=triton.next_power_of_2(rope_dim // 2),
        NUM_DIM_BLOCKS=num_dim_blocks,
        NUM_Q_HEADS=num_q_heads,
        Q_HEAD_BLOCKS=max(q_head_blocks, 1),
        BLOCK_Q_HEADS=block_q_heads,
        Q_PREFIX_D=FP4_MLA_Q_PREFIX_DIM,
        Q_PREFIX_BLOCK_D=q_prefix_block_dim,
        Q_PREFIX_BLOCKS=q_prefix_blocks,
        Q_PREFIX_BLOCKS_PER_PROGRAM=q_prefix_blocks_per_program,
        Q_WORK_BLOCKS=q_work_blocks,
        Q_SF_COLS=FP4_MLA_Q_SF_GROUPS,
        WRITE_Q=True,
        Q1_KV_BLOCKS_PER_PROGRAM=q1_kv_blocks_per_program,
        Q_ROWS_PER_SEQ=q_rows_per_seq,
        WRITE_Q_ROWS=write_q_rows,
        maxnreg=56,
    )
    return kv_lens_gen, gen_lens_gen, page_ids
