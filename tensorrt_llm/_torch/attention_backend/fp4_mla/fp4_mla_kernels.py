# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for MLA FP4 KV-cache helpers."""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["num_contexts"],
    do_not_specialize_on_alignment=["num_contexts"],
)
def _fp8_mla_context_block_table_kernel(
    context_lengths_ptr,
    block_offsets_ptr,
    block_ids_ptr,
    num_contexts,
    PAGE_SIZE: tl.constexpr,
    MAX_BLOCKS_PER_SEQUENCE: tl.constexpr,
    SEQUENCE_BLOCK: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
):
    """Materialize compact scratch-cache page IDs directly on the GPU."""
    sequence_idx = tl.program_id(0)
    sequence_offsets = tl.arange(0, SEQUENCE_BLOCK)
    context_lengths = tl.load(
        context_lengths_ptr + sequence_offsets,
        mask=sequence_offsets < num_contexts,
        other=0,
    )
    context_lengths = tl.maximum(context_lengths, 0)
    sequence_blocks = (context_lengths + PAGE_SIZE - 1) // PAGE_SIZE
    block_start = tl.sum(
        tl.where(sequence_offsets < sequence_idx, sequence_blocks, 0),
        axis=0,
    )

    page_offsets = tl.arange(0, PAGE_BLOCK)
    active_sequence = sequence_idx < num_contexts
    context_length = tl.load(
        context_lengths_ptr + sequence_idx,
        mask=active_sequence,
        other=0,
    )
    num_blocks = (tl.maximum(context_length, 0) + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = tl.minimum(num_blocks, MAX_BLOCKS_PER_SEQUENCE)
    active_page = active_sequence & (page_offsets < num_blocks)
    page_ids = tl.where(active_page, block_start + page_offsets, 0)
    page_mask = page_offsets < MAX_BLOCKS_PER_SEQUENCE

    block_ids_offset = sequence_idx * MAX_BLOCKS_PER_SEQUENCE + page_offsets
    tl.store(block_ids_ptr + block_ids_offset, page_ids, mask=page_mask)

    block_offsets_offset = sequence_idx * 2 * MAX_BLOCKS_PER_SEQUENCE + page_offsets
    tl.store(block_offsets_ptr + block_offsets_offset, page_ids, mask=page_mask)
    tl.store(
        block_offsets_ptr + block_offsets_offset + MAX_BLOCKS_PER_SEQUENCE,
        page_ids,
        mask=page_mask,
    )


# Q activation quantization helpers. These mirror the conversion sequence in
# arcquantFP4.cu; the KV-cache quantizer below uses a different midpoint mapping.


@triton.jit
def _fp4_mla_q_sf_offset(row_idx, col_idx, sf_cols: tl.constexpr):
    padded_cols: tl.constexpr = ((sf_cols + 3) // 4) * 4
    return (
        col_idx % 4
        + (col_idx // 4) * (4 * 128)
        + (row_idx % 32) * 16
        + ((row_idx % 128) // 32) * 4
        + (row_idx // 128) * (128 * padded_cols)
    )


@triton.jit
def _fp4_mla_q_sf_offset_h128(q_token, q_head_block, head_lane, col_idx, sf_cols: tl.constexpr):
    """Compute the Q-scale swizzle from the fixed H128 row decomposition."""
    padded_cols: tl.constexpr = ((sf_cols + 3) // 4) * 4
    return (
        col_idx % 4
        + (col_idx // 4) * (4 * 128)
        + head_lane * 16
        + q_head_block * 4
        + q_token.to(tl.int64) * (128 * padded_cols)
    )


@triton.jit
def _fp4_mla_pack_e2m1x2_rne(even, odd):
    """Match cvt.rn.satfinite.e2m1x2.f32 used by arcquantFP4.cu."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 packed0, packed1, packed2, packed3;
            cvt.rn.satfinite.e2m1x2.f32 packed0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 packed1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 packed2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 packed3, $8, $4;
            mov.b32 $0, {packed0, packed1, packed2, packed3};
        }
        """,
        constraints="=r,f,f,f,f,f,f,f,f",
        args=[even.to(tl.float32), odd.to(tl.float32)],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _fp4_mla_pack_e2m1x8_rne(even0, even1, even2, even3, odd0, odd1, odd2, odd3):
    """Pack four consecutive E2M1x2 bytes into one word."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 packed0, packed1, packed2, packed3;
            cvt.rn.satfinite.e2m1x2.f32 packed0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 packed1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 packed2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 packed3, $8, $4;
            mov.b32 $0, {packed0, packed1, packed2, packed3};
        }
        """,
        constraints="=r,f,f,f,f,f,f,f,f",
        args=[even0, even1, even2, even3, odd0, odd1, odd2, odd3],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_mla_unpack_e2m1x2(packed):
    """Decode packed E2M1 through the native GR100 conversion path."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 packed0, packed1, packed2, packed3;
            .reg .b16 low, high;
            .reg .b32 half2;
            mov.b32 {packed0, packed1, packed2, packed3}, $2;
            cvt.rn.f16x2.e2m1x2 half2, packed0;
            mov.b32 {low, high}, half2;
            cvt.f32.f16 $0, low;
            cvt.f32.f16 $1, high;
        }
        """,
        constraints="=f,=f,r",
        args=[packed.to(tl.uint32)],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_mla_rcp_approx_ftz(value):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_mla_rope_fp32(even, odd, cos, sin):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 even_cos, odd_sin, odd_cos, even_sin;
            mul.rn.f32 even_cos, $2, $4;
            mul.rn.f32 odd_sin, $3, $5;
            sub.rn.f32 $0, even_cos, odd_sin;
            mul.rn.f32 odd_cos, $3, $4;
            mul.rn.f32 even_sin, $2, $5;
            add.rn.f32 $1, odd_cos, even_sin;
        }
        """,
        constraints="=f,=f,f,f,f,f",
        args=[even, odd, cos, sin],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_mla_quantize_q_groups(values_even, values_odd):
    amax = tl.max(
        tl.maximum(tl.abs(values_even), tl.abs(values_odd)),
        axis=2,
    )
    scale = tl.maximum(amax / 6.0, 0.001953125)
    scale_fp8 = scale.to(tl.float8e4nv)
    inv_scale = _fp4_mla_rcp_approx_ftz(scale_fp8.to(tl.float32))
    packed = _fp4_mla_pack_e2m1x2_rne(
        values_even * inv_scale[:, :, None],
        values_odd * inv_scale[:, :, None],
    )
    return packed, scale_fp8


@triton.jit
def _fp4_mla_quantize_q_groups_words(
    values_even,
    values_odd,
    BLOCK_Q_HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """Quantize FP4 groups while retaining native packed-word ownership."""
    tl.static_assert(values_even.shape[2] == 8)
    tl.static_assert(values_odd.shape[2] == 8)
    amax = tl.max(
        tl.maximum(tl.abs(values_even), tl.abs(values_odd)),
        axis=2,
    )
    scale = tl.maximum(amax / 6.0, 0.001953125)
    scale_fp8 = scale.to(tl.float8e4nv)
    inv_scale = _fp4_mla_rcp_approx_ftz(scale_fp8.to(tl.float32))
    scaled_even = tl.reshape(
        values_even * inv_scale[:, :, None],
        (BLOCK_Q_HEADS, GROUPS, 2, 2, 2),
    )
    scaled_odd = tl.reshape(
        values_odd * inv_scale[:, :, None],
        (BLOCK_Q_HEADS, GROUPS, 2, 2, 2),
    )
    even_02, even_13 = tl.split(scaled_even)
    odd_02, odd_13 = tl.split(scaled_odd)
    even0, even2 = tl.split(even_02)
    even1, even3 = tl.split(even_13)
    odd0, odd2 = tl.split(odd_02)
    odd1, odd3 = tl.split(odd_13)
    packed_words = _fp4_mla_pack_e2m1x8_rne(
        even0,
        even1,
        even2,
        even3,
        odd0,
        odd1,
        odd2,
        odd3,
    )
    return packed_words, scale_fp8


@triton.jit
def _fp4_mla_quantize_q_residual_groups(values_even, values_odd):
    packed, scale_fp8 = _fp4_mla_quantize_q_groups(values_even, values_odd)
    rounded_scale = scale_fp8.to(tl.float32)
    main_even, main_odd = _fp4_mla_unpack_e2m1x2(packed)
    residual_even = values_even - main_even * rounded_scale[:, :, None]
    residual_odd = values_odd - main_odd * rounded_scale[:, :, None]
    residual_packed, residual_scale_fp8 = _fp4_mla_quantize_q_groups(residual_even, residual_odd)
    return packed, scale_fp8, residual_packed, residual_scale_fp8


@triton.jit
def _fp4_mla_swizzled_sf_offset(
    row_idx,
    col_idx,
    SF_PER_TOKEN: tl.constexpr,
):
    padded_cols = ((SF_PER_TOKEN + 3) // 4) * 4
    col_in_group = col_idx % 4
    col_group = col_idx // 4
    row_in_group0 = row_idx % 32
    row_in_group1 = (row_idx % 128) // 32
    row_group = row_idx // 128
    return (
        col_in_group
        + col_group * (4 * 128)
        + row_in_group0 * 16
        + row_in_group1 * 4
        + row_group * (128 * padded_cols)
    )


# FP4 conversion and cache kernels


@triton.jit
def _fp4_e2m1_to_f32(nibble):
    magnitude = nibble & 0x7
    value = tl.where(
        magnitude == 0,
        0.0,
        tl.where(
            magnitude == 1,
            0.5,
            tl.where(
                magnitude == 2,
                1.0,
                tl.where(
                    magnitude == 3,
                    1.5,
                    tl.where(
                        magnitude == 4,
                        2.0,
                        tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    sign = (nibble & 0x8) != 0
    return tl.where(sign, -value, value)


@triton.jit
def _fp4_mla_floor_half_up(abs_value, multiplier, bias):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 rounded;
            fma.rz.f32 rounded, $1, $2, $3;
            cvt.rzi.s32.f32 $0, rounded;
        }
        """,
        constraints="=r,f,f,f",
        args=[abs_value, multiplier, bias],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_mla_warp_max(value):
    return tl.inline_asm_elementwise(
        """
        .reg .f32 other;
        mov.f32 $0, $1;
        shfl.sync.bfly.b32 other, $0, 16, 0x1f, 0xffffffff;
        max.f32 $0, $0, other;
        shfl.sync.bfly.b32 other, $0, 8, 0x1f, 0xffffffff;
        max.f32 $0, $0, other;
        shfl.sync.bfly.b32 other, $0, 4, 0x1f, 0xffffffff;
        max.f32 $0, $0, other;
        shfl.sync.bfly.b32 other, $0, 2, 0x1f, 0xffffffff;
        max.f32 $0, $0, other;
        shfl.sync.bfly.b32 other, $0, 1, 0x1f, 0xffffffff;
        max.f32 $0, $0, other;
        """,
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_e2m1_quantize(x):
    abs_x = tl.abs(x)
    bounded = tl.where(abs_x < 6.0, abs_x, 6.0)
    small = bounded < 2.0
    multiplier = tl.where(small, 2.0, 1.0)
    bias = tl.where(small, 0.5, 2.5)
    magnitude = _fp4_mla_floor_half_up(bounded, multiplier, bias)
    magnitude = tl.minimum(magnitude, 6) + (~(abs_x < 5.0)).to(tl.int32)
    sign = tl.where(x < 0.0, 8, 0)
    return (magnitude | sign).to(tl.uint8)


@triton.jit(
    do_not_specialize=[
        "page_ids_len",
        "indptr_len",
        "metadata_num_tokens",
        "num_pages",
        "num_layers",
        "num_contexts",
        "num_hp_pages",
        "token_offset",
        "num_tokens",
        "local_layer",
        "v_page_offset",
    ],
    do_not_specialize_on_alignment=[
        "page_ids_len",
        "indptr_len",
        "metadata_num_tokens",
        "num_pages",
        "num_layers",
        "num_contexts",
        "num_hp_pages",
        "token_offset",
        "num_tokens",
        "local_layer",
        "v_page_offset",
    ],
)
def _fp4_mla_context_cache_update_kernel(
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    v_packed_ptr,
    latent_cache_ptr,
    global_scale_ptr,
    rotary_cos_sin_ptr,
    hp_pool_ptr,
    hp_page_ids_ptr,
    batch_indices_ptr,
    positions_ptr,
    paged_kv_indices_ptr,
    paged_kv_indptr_ptr,
    page_ids_len,
    indptr_len,
    metadata_num_tokens,
    num_pages,
    num_layers,
    num_contexts,
    num_hp_pages,
    token_offset,
    num_tokens,
    local_layer,
    v_page_offset,
    page_size,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    lc_s0,
    lc_s1,
    vsf_s0,
    vsf_s1,
    v_packed_s0,
    v_packed_s1,
    pool_s0,
    pool_s1,
    HEAD_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
    HP_POOL_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    K_RESIDUAL_D: tl.constexpr,
    STORE_K_RESIDUAL: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    APPLY_K_ROPE: tl.constexpr,
    POOL_HEAD_D: tl.constexpr,
    STORE_HP_TAIL: tl.constexpr,
    WRITE_V_PACKED: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dim_block = tl.program_id(1)
    if (local_layer < 0) | (local_layer >= num_layers):
        return
    if token_idx >= num_tokens:
        return

    metadata_token_idx = token_offset + token_idx
    if metadata_token_idx >= metadata_num_tokens:
        return

    batch_idx = tl.load(batch_indices_ptr + metadata_token_idx).to(tl.int64)
    position = tl.load(positions_ptr + metadata_token_idx).to(tl.int64)
    if (batch_idx < 0) | (batch_idx + 1 >= indptr_len) | (position < 0):
        return
    if position % HP_BLOCK != 0:
        return

    page_size_i64 = tl.cast(page_size, tl.int64)
    page_idx = position // page_size_i64
    page_pos = position - page_idx * page_size_i64
    page_start = tl.load(paged_kv_indptr_ptr + batch_idx).to(tl.int64)
    page_end = tl.load(paged_kv_indptr_ptr + batch_idx + 1).to(tl.int64)
    page_table_offset = page_start + page_idx
    if (
        (page_pos < 0)
        | (page_pos >= page_size_i64)
        | (page_table_offset < page_start)
        | (page_table_offset >= page_end)
        | (page_table_offset < 0)
        | (page_table_offset >= page_ids_len)
    ):
        return
    physical_page = tl.load(paged_kv_indices_ptr + page_table_offset).to(tl.int64)
    if (physical_page < 0) | (physical_page >= num_pages):
        return

    byte_offsets = tl.arange(0, HP_BLOCK // 2)
    token_offsets = tl.arange(0, HP_BLOCK)
    even_d = dim_block * FP4_BLOCK + byte_offsets * 2
    odd_d = even_d + 1
    all_d = dim_block * FP4_BLOCK + tl.arange(0, FP4_BLOCK)
    mask_even_d = even_d < HEAD_D
    mask_odd_d = odd_d < HEAD_D
    mask_all_d = all_d < HEAD_D
    safe_even_d = tl.where(mask_even_d, even_d, 0)
    safe_odd_d = tl.where(mask_odd_d, odd_d, 0)
    safe_all_d = tl.where(mask_all_d, all_d, 0)

    token_candidates = token_idx + token_offsets
    valid_tokens = token_candidates < num_tokens
    candidate_metadata = token_offset + token_candidates
    valid_tokens = valid_tokens & (candidate_metadata < metadata_num_tokens)
    safe_candidate_metadata = tl.where(valid_tokens, candidate_metadata, 0)

    candidate_batch = tl.load(
        batch_indices_ptr + safe_candidate_metadata, mask=valid_tokens, other=-1
    ).to(tl.int64)
    candidate_pos = tl.load(
        positions_ptr + safe_candidate_metadata, mask=valid_tokens, other=-1
    ).to(tl.int64)
    valid_tokens = valid_tokens & (candidate_batch == batch_idx)
    valid_tokens = valid_tokens & (candidate_pos == position + token_offsets)
    # int64 so safe_token_candidates * lc_s0 doesn't overflow when num_tokens * head_dim > 2^31.
    safe_token_candidates = tl.where(valid_tokens, token_candidates, 0).to(tl.int64)

    even_values = tl.load(
        latent_cache_ptr + safe_token_candidates[:, None] * lc_s0 + safe_even_d[None, :] * lc_s1,
        mask=valid_tokens[:, None] & mask_even_d[None, :],
        other=0.0,
    ).to(tl.float32)
    odd_values = tl.load(
        latent_cache_ptr + safe_token_candidates[:, None] * lc_s0 + safe_odd_d[None, :] * lc_s1,
        mask=valid_tokens[:, None] & mask_odd_d[None, :],
        other=0.0,
    ).to(tl.float32)
    if APPLY_K_ROPE:
        tl.static_assert(V_HEAD_D + ROPE_DIM == HEAD_D)
        tl.static_assert(ROPE_DIM % 2 == 0)
        if dim_block * FP4_BLOCK >= V_HEAD_D:
            rope_dim_mask = mask_even_d & (even_d >= V_HEAD_D) & (odd_d < V_HEAD_D + ROPE_DIM)
            rope_pair_offsets = tl.where(rope_dim_mask, (safe_even_d - V_HEAD_D) // 2, 0)
            valid_position = valid_tokens & (candidate_pos >= 0)
            rotary_offsets = (
                candidate_pos[:, None] * (ROPE_DIM * 2) + rope_pair_offsets[None, :] * 2
            )
            rotary_mask = valid_position[:, None] & rope_dim_mask[None, :]
            cos = tl.load(
                rotary_cos_sin_ptr + rotary_offsets,
                mask=rotary_mask,
                other=0.0,
            ).to(tl.float32)
            sin = tl.load(
                rotary_cos_sin_ptr + rotary_offsets + 1,
                mask=rotary_mask,
                other=0.0,
            ).to(tl.float32)
            roped_even, roped_odd = _fp4_mla_rope_fp32(even_values, odd_values, cos, sin)
            roped_even = roped_even.to(tl.bfloat16).to(tl.float32)
            roped_odd = roped_odd.to(tl.bfloat16).to(tl.float32)
            even_values = tl.where(rotary_mask, roped_even, even_values)
            odd_values = tl.where(rotary_mask, roped_odd, odd_values)

    if STORE_HP_TAIL:
        tl.static_assert(POOL_HEAD_D >= HEAD_D)
        hp_batch_valid = (batch_idx >= 0) & (batch_idx < num_contexts)
        # Context starts are tile-aligned, so only the final incomplete tile
        # has fewer than HP_BLOCK valid candidates for this sequence.
        tile_is_partial = tl.sum(valid_tokens.to(tl.int32), axis=0) < HP_BLOCK
        hp_index = tl.load(
            hp_page_ids_ptr + page_table_offset,
            mask=hp_batch_valid & tile_is_partial,
            other=-1,
        ).to(tl.int64)
        hp_batch_valid = hp_batch_valid & (hp_index >= 0) & (hp_index < num_hp_pages)
        hp_tail_tokens = valid_tokens & hp_batch_valid & tile_is_partial
        hp_slots = tl.where(hp_tail_tokens, candidate_pos % HP_POOL_SIZE, 0)
        safe_hp_index = tl.where(hp_batch_valid, hp_index, 0)
        hp_base = safe_hp_index * pool_s0 + local_layer * pool_s1 + hp_slots[:, None] * POOL_HEAD_D
        tl.store(
            hp_pool_ptr + hp_base + safe_even_d[None, :],
            even_values,
            mask=hp_tail_tokens[:, None] & mask_even_d[None, :],
        )
        tl.store(
            hp_pool_ptr + hp_base + safe_odd_d[None, :],
            odd_values,
            mask=hp_tail_tokens[:, None] & mask_odd_d[None, :],
        )

    amax_per_token = tl.maximum(
        tl.max(tl.abs(even_values), axis=1),
        tl.max(tl.abs(odd_values), axis=1),
    )
    tile_amax = tl.max(amax_per_token, axis=0)
    kv_global_scale = tl.load(global_scale_ptr)
    # K consumes scales as [token, dim-block], while V consumes scales as
    # [dim, token-block].  Only the compressed-KV prefix has both views, so
    # tail K-only dims keep K's per-token scale.
    shared_tile = dim_block * FP4_BLOCK < V_HEAD_D
    tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)
    token_scale = tl.where(amax_per_token > 0.0, amax_per_token / 6.0, 1.0)
    local_scale = tl.where(shared_tile, tile_scale, token_scale)
    # A capped page amax (or a block above the static reference amax) can push an
    # outlier block's e4m3 scale above the e4m3 ceiling (448); clamp so it clips
    # gracefully instead of overflowing the scale's e4m3 representation.
    stored_scale = tl.minimum(local_scale * kv_global_scale, 448.0)

    quant_scale = local_scale
    if STORE_K_RESIDUAL:
        if dim_block * FP4_BLOCK >= HEAD_D - K_RESIDUAL_D:
            quant_scale = stored_scale.to(tl.float8e4nv).to(tl.float32) / kv_global_scale
    low = _fp4_e2m1_quantize(even_values / quant_scale[:, None])
    high = _fp4_e2m1_quantize(odd_values / quant_scale[:, None])
    packed = low | (high << 4)
    v_dim_packed = packed
    v_stored_scale = tl.minimum(tile_scale * kv_global_scale, 448.0)

    packed_cols = dim_block * (FP4_BLOCK // 2) + byte_offsets
    page_positions = page_pos + token_offsets
    kv_base = physical_page * kv_s0
    tl.store(
        kv_cache_ptr + kv_base + page_positions[:, None] * kv_s2 + packed_cols[None, :] * kv_s4,
        packed,
        mask=valid_tokens[:, None] & mask_even_d[None, :],
    )

    if WRITE_V_PACKED:
        tl.static_assert(V_HEAD_D % FP4_BLOCK == 0)
        # The canonical cache packs dimensions; PV packs tokens. Reuse the
        # selected FP4 codes and transpose this 16x16 tile in registers.
        v_pairs = tl.reshape(
            tl.permute(v_dim_packed, 1, 0),
            (FP4_BLOCK // 2, HP_BLOCK // 2, 2),
        )
        even_token_packed, odd_token_packed = tl.split(v_pairs)
        v_low = (even_token_packed & 0x0F) | ((odd_token_packed & 0x0F) << 4)
        v_high = ((even_token_packed >> 4) & 0x0F) | (odd_token_packed & 0xF0)
        logical_v_dims = dim_block * FP4_BLOCK + byte_offsets * 2
        valid_v_dims = logical_v_dims < V_HEAD_D
        v_even_rows = ((v_page_offset + physical_page) * V_HEAD_D + logical_v_dims).to(tl.int64)
        v_cols = page_pos // 2 + tl.arange(0, HP_BLOCK // 2)
        v_even_offsets = v_even_rows[:, None] * v_packed_s0 + v_cols[None, :] * v_packed_s1
        v_odd_offsets = v_even_offsets + v_packed_s0
        token_pair_validity = tl.reshape(valid_tokens, (HP_BLOCK // 2, 2))
        token0_valid, token1_valid = tl.split(token_pair_validity)
        partial_pair = token0_valid & (~token1_valid)
        v_store_mask = valid_v_dims[:, None] & token0_valid[None, :]
        partial_pair_mask = valid_v_dims[:, None] & partial_pair[None, :]
        old_v_low = tl.load(
            v_packed_ptr + v_even_offsets,
            mask=partial_pair_mask,
            other=0,
        )
        old_v_high = tl.load(
            v_packed_ptr + v_odd_offsets,
            mask=partial_pair_mask,
            other=0,
        )
        v_low = tl.where(
            partial_pair[None, :],
            (old_v_low & 0xF0) | (v_low & 0x0F),
            v_low,
        ).to(tl.uint8)
        v_high = tl.where(
            partial_pair[None, :],
            (old_v_high & 0xF0) | (v_high & 0x0F),
            v_high,
        ).to(tl.uint8)
        tl.store(v_packed_ptr + v_even_offsets, v_low, mask=v_store_mask)
        tl.store(v_packed_ptr + v_odd_offsets, v_high, mask=v_store_mask)

    k_sf_offsets = _fp4_mla_swizzled_sf_offset(page_positions, dim_block, SF_PER_TOKEN)
    tl.store(sf_cache_ptr + physical_page * sf_s0 + k_sf_offsets, stored_scale, mask=valid_tokens)

    if STORE_K_RESIDUAL:
        residual_start = HEAD_D - K_RESIDUAL_D
        if dim_block * FP4_BLOCK >= residual_start:
            main_even = _fp4_e2m1_to_f32(low) * quant_scale[:, None]
            main_odd = _fp4_e2m1_to_f32(high) * quant_scale[:, None]
            residual_even = even_values - main_even
            residual_odd = odd_values - main_odd
            residual_amax = tl.maximum(
                tl.max(tl.abs(residual_even), axis=1),
                tl.max(tl.abs(residual_odd), axis=1),
            )
            residual_scale = tl.where(residual_amax > 0.0, residual_amax / 6.0, 1.0)
            residual_stored_scale = tl.minimum(residual_scale * kv_global_scale, 448.0)
            residual_quant_scale = (
                residual_stored_scale.to(tl.float8e4nv).to(tl.float32) / kv_global_scale
            )
            residual_low = _fp4_e2m1_quantize(residual_even / residual_quant_scale[:, None])
            residual_high = _fp4_e2m1_quantize(residual_odd / residual_quant_scale[:, None])
            residual_packed = residual_low | (residual_high << 4)
            residual_group = dim_block - residual_start // FP4_BLOCK
            residual_packed_cols = HEAD_D // 2 + residual_group * (FP4_BLOCK // 2) + byte_offsets
            tl.store(
                kv_cache_ptr
                + kv_base
                + page_positions[:, None] * kv_s2
                + residual_packed_cols[None, :] * kv_s4,
                residual_packed,
                mask=valid_tokens[:, None] & mask_even_d[None, :],
            )
            residual_sf_offsets = _fp4_mla_swizzled_sf_offset(
                page_positions,
                HEAD_D // FP4_BLOCK + residual_group,
                SF_PER_TOKEN,
            )
            tl.store(
                sf_cache_ptr + physical_page * sf_s0 + residual_sf_offsets,
                residual_stored_scale,
                mask=valid_tokens,
            )

    v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
        vsf_s0, tl.int64
    ) + physical_page * tl.cast(vsf_s1, tl.int64)
    token_scale_col = page_pos // HP_BLOCK
    sf_offsets = _fp4_mla_swizzled_sf_offset(
        safe_all_d,
        token_scale_col,
        SF_PER_PAGE,
    )
    tl.store(
        v_sf_ptr + v_sf_base + sf_offsets.to(tl.int64),
        v_stored_scale,
        mask=mask_all_d & (all_d < V_HEAD_D),
    )


@triton.jit
def _fp4_mla_generation_q1_kv_tile_update(
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    v_packed_ptr,
    hp_pool_ptr,
    hp_index,
    kv_len,
    block_base_pos,
    page_pos,
    physical_page,
    dim_blocks,
    global_scale,
    local_layer,
    v_page_offset,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    pool_s0,
    pool_s1,
    vsf_s0,
    vsf_s1,
    v_packed_s0,
    v_packed_s1,
    HEAD_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
    HP_POOL_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    WRITE_V_PACKED: tl.constexpr,
):
    """Update two 16-channel Q1 KV tiles inside the fused generation operator."""
    tl.static_assert(HP_BLOCK == FP4_BLOCK)
    tl.static_assert(HP_POOL_SIZE >= HP_BLOCK)
    tl.static_assert(V_HEAD_D % FP4_BLOCK == 0)
    tl.static_assert(HEAD_D >= V_HEAD_D)

    byte_offsets = tl.arange(0, HP_BLOCK // 2)
    token_offsets = tl.arange(0, HP_BLOCK)
    even_d = dim_blocks[:, None] * FP4_BLOCK + byte_offsets[None, :] * 2
    all_d = dim_blocks[:, None] * FP4_BLOCK + tl.arange(0, FP4_BLOCK)[None, :]

    abs_positions = block_base_pos + token_offsets
    valid_tokens = abs_positions < kv_len
    hp_slots = abs_positions % HP_POOL_SIZE

    hp_even_ptrs = (
        hp_pool_ptr
        + hp_index * pool_s0
        + local_layer * pool_s1
        + hp_slots[None, :, None] * HEAD_D
        + even_d[:, None, :]
    )
    hp_pair_ptrs = hp_even_ptrs.to(tl.pointer_type(tl.uint32))
    packed_bf16 = tl.load(
        hp_pair_ptrs,
        mask=valid_tokens[None, :, None],
        other=0,
    )
    even_bits = (packed_bf16 & 0xFFFF).to(tl.uint16)
    odd_bits = (packed_bf16 >> 16).to(tl.uint16)
    even_values = even_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    odd_values = odd_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)

    pair_amax = tl.maximum(tl.abs(even_values), tl.abs(odd_values))
    tile_amax = tl.max(
        tl.reshape(
            pair_amax,
            (2, HP_BLOCK * (FP4_BLOCK // 2)),
        ),
        axis=1,
    )
    tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)

    token_scale_col = page_pos // HP_BLOCK
    sf_offsets = _fp4_mla_swizzled_sf_offset(all_d, token_scale_col, SF_PER_PAGE)
    v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
        vsf_s0, tl.int64
    ) + physical_page * tl.cast(vsf_s1, tl.int64)
    tl.store(
        v_sf_ptr + v_sf_base + sf_offsets.to(tl.int64),
        tile_scale[:, None] * global_scale,
    )

    low = _fp4_e2m1_quantize(even_values / tile_scale[:, None, None])
    high = _fp4_e2m1_quantize(odd_values / tile_scale[:, None, None])
    packed = low | (high << 4)
    packed_cols = dim_blocks[:, None] * (FP4_BLOCK // 2) + byte_offsets[None, :]
    page_positions = page_pos + token_offsets
    kv_base = physical_page * kv_s0
    tl.store(
        kv_cache_ptr
        + kv_base
        + page_positions[None, :, None] * kv_s2
        + packed_cols[:, None, :] * kv_s4,
        packed,
        mask=valid_tokens[None, :, None],
    )

    if WRITE_V_PACKED:
        v_pairs = tl.reshape(
            tl.permute(packed, 0, 2, 1),
            (2, FP4_BLOCK // 2, HP_BLOCK // 2, 2),
        )
        even_token_packed, odd_token_packed = tl.split(v_pairs)
        v_low = (even_token_packed & 0x0F) | ((odd_token_packed & 0x0F) << 4)
        v_high = ((even_token_packed >> 4) & 0x0F) | (odd_token_packed & 0xF0)
        v_even_rows = (
            (v_page_offset + physical_page) * V_HEAD_D
            + dim_blocks[:, None] * FP4_BLOCK
            + tl.arange(0, FP4_BLOCK // 2)[None, :] * 2
        ).to(tl.int64)
        pair_offsets = tl.arange(0, HP_BLOCK // 2)
        v_cols = page_pos // 2 + pair_offsets
        v_even_offsets = v_even_rows[:, :, None] * v_packed_s0 + v_cols[None, None, :] * v_packed_s1
        v_odd_offsets = v_even_offsets + v_packed_s0
        token0_valid = block_base_pos + pair_offsets * 2 < kv_len
        token1_valid = block_base_pos + pair_offsets * 2 + 1 < kv_len
        partial_pair = token0_valid & (~token1_valid)
        old_v_low = tl.load(
            v_packed_ptr + v_even_offsets,
            mask=partial_pair[None, None, :],
            other=0,
        )
        old_v_high = tl.load(
            v_packed_ptr + v_odd_offsets,
            mask=partial_pair[None, None, :],
            other=0,
        )
        v_low = tl.where(
            partial_pair[None, None, :],
            (old_v_low & 0xF0) | (v_low & 0x0F),
            v_low,
        ).to(tl.uint8)
        v_high = tl.where(
            partial_pair[None, None, :],
            (old_v_high & 0xF0) | (v_high & 0x0F),
            v_high,
        ).to(tl.uint8)
        tl.store(
            v_packed_ptr + v_even_offsets,
            v_low,
            mask=token0_valid[None, None, :],
        )
        tl.store(
            v_packed_ptr + v_odd_offsets,
            v_high,
            mask=token0_valid[None, None, :],
        )

    k_sf_offsets = _fp4_mla_swizzled_sf_offset(
        page_positions[None, :], dim_blocks[:, None], SF_PER_TOKEN
    )
    tl.store(
        sf_cache_ptr + physical_page * sf_s0 + k_sf_offsets,
        tile_scale[:, None] * global_scale,
        mask=valid_tokens[None, :],
    )


@triton.jit
def _fp4_mla_generation_q1_kv_tail8_update(
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    v_packed_ptr,
    hp_pool_ptr,
    hp_index,
    kv_len,
    block_base_pos,
    page_pos,
    physical_page,
    dim_block_base,
    global_scale,
    local_layer,
    v_page_offset,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    pool_s0,
    pool_s1,
    vsf_s0,
    vsf_s1,
    v_packed_s0,
    v_packed_s1,
    HEAD_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
    HP_POOL_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    WRITE_V_PACKED: tl.constexpr,
    DIM_BLOCK_STRIDE: tl.constexpr,
):
    """Update four Q1 KV tiles with one warp per tile for tails up to 8 tokens."""
    tl.static_assert(HP_BLOCK == 16)
    tl.static_assert(HP_POOL_SIZE >= HP_BLOCK)
    tl.static_assert(FP4_BLOCK == HP_BLOCK)
    tl.static_assert(V_HEAD_D % FP4_BLOCK == 0)

    threads = tl.arange(0, 128)
    warp = threads // 32
    lane = threads % 32
    dim_block = dim_block_base + warp * DIM_BLOCK_STRIDE
    byte_offset = lane % (FP4_BLOCK // 2)
    token_pair = lane // (FP4_BLOCK // 2)
    token0 = token_pair * 2
    token1 = token0 + 1
    tail_count = kv_len - block_base_pos
    token0_valid = token0 < tail_count
    token1_valid = token1 < tail_count
    abs_position0 = block_base_pos + token0
    abs_position1 = block_base_pos + token1
    hp_slot0 = abs_position0 % HP_POOL_SIZE
    hp_slot1 = abs_position1 % HP_POOL_SIZE
    dim_even = dim_block * FP4_BLOCK + byte_offset * 2
    hp_base = hp_index * pool_s0 + local_layer * pool_s1
    pair0_ptr = (hp_pool_ptr + hp_base + hp_slot0 * HEAD_D + dim_even).to(
        tl.pointer_type(tl.uint32)
    )
    pair1_ptr = (hp_pool_ptr + hp_base + hp_slot1 * HEAD_D + dim_even).to(
        tl.pointer_type(tl.uint32)
    )
    pair0_bits = tl.load(pair0_ptr, mask=token0_valid, other=0)
    pair1_bits = tl.load(pair1_ptr, mask=token1_valid, other=0)
    even0 = (pair0_bits & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    odd0 = (pair0_bits >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    even1 = (pair1_bits & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    odd1 = (pair1_bits >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)

    local_amax = tl.maximum(
        tl.maximum(tl.abs(even0), tl.abs(odd0)),
        tl.maximum(tl.abs(even1), tl.abs(odd1)),
    )
    tile_amax = _fp4_mla_warp_max(local_amax)
    tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)

    low0 = _fp4_e2m1_quantize(even0 / tile_scale)
    high0 = _fp4_e2m1_quantize(odd0 / tile_scale)
    low1 = _fp4_e2m1_quantize(even1 / tile_scale)
    high1 = _fp4_e2m1_quantize(odd1 / tile_scale)
    packed0 = low0 | (high0 << 4)
    packed1 = low1 | (high1 << 4)

    packed_col = dim_block * (FP4_BLOCK // 2) + byte_offset
    kv_base = physical_page * kv_s0
    tl.store(
        kv_cache_ptr + kv_base + (page_pos + token0) * kv_s2 + packed_col * kv_s4,
        packed0,
        mask=token0_valid,
    )
    tl.store(
        kv_cache_ptr + kv_base + (page_pos + token1) * kv_s2 + packed_col * kv_s4,
        packed1,
        mask=token1_valid,
    )

    stored_scale = tile_scale * global_scale
    token_scale_col = page_pos // HP_BLOCK
    v_sf_dim = dim_block * FP4_BLOCK + lane
    v_sf_offset = _fp4_mla_swizzled_sf_offset(
        v_sf_dim,
        token_scale_col,
        SF_PER_PAGE,
    )
    v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
        vsf_s0, tl.int64
    ) + physical_page * tl.cast(vsf_s1, tl.int64)
    tl.store(
        v_sf_ptr + v_sf_base + v_sf_offset.to(tl.int64),
        stored_scale,
        mask=lane < FP4_BLOCK,
    )

    k_sf_position = page_pos + lane
    k_sf_offset = _fp4_mla_swizzled_sf_offset(
        k_sf_position,
        dim_block,
        SF_PER_TOKEN,
    )
    tl.store(
        sf_cache_ptr + physical_page * sf_s0 + k_sf_offset,
        stored_scale,
        mask=lane < tail_count,
    )

    if WRITE_V_PACKED:
        v_low = (packed0 & 0x0F) | ((packed1 & 0x0F) << 4)
        v_high = ((packed0 >> 4) & 0x0F) | (packed1 & 0xF0)
        v_row_base = ((v_page_offset + physical_page) * V_HEAD_D + dim_block * FP4_BLOCK).to(
            tl.int64
        )
        v_even_row = v_row_base + byte_offset * 2
        v_odd_row = v_even_row + 1
        v_col = page_pos // 2 + token_pair
        v_even_offset = v_even_row * v_packed_s0 + v_col * v_packed_s1
        v_odd_offset = v_odd_row * v_packed_s0 + v_col * v_packed_s1
        partial_pair = token0_valid & (~token1_valid)
        old_v_low = tl.load(
            v_packed_ptr + v_even_offset,
            mask=partial_pair,
            other=0,
        )
        old_v_high = tl.load(
            v_packed_ptr + v_odd_offset,
            mask=partial_pair,
            other=0,
        )
        v_low = tl.where(
            partial_pair,
            (old_v_low & 0xF0) | (v_low & 0x0F),
            v_low,
        ).to(tl.uint8)
        v_high = tl.where(
            partial_pair,
            (old_v_high & 0xF0) | (v_high & 0x0F),
            v_high,
        ).to(tl.uint8)
        tl.store(
            v_packed_ptr + v_even_offset,
            v_low,
            mask=token0_valid,
        )
        tl.store(
            v_packed_ptr + v_odd_offset,
            v_high,
            mask=token0_valid,
        )


@triton.jit(
    do_not_specialize=[
        "page_ids_len",
        "hp_page_ids_len",
        "indptr_len",
        "num_pages",
        "num_hp_pages",
        "num_layers",
        "local_layer",
        "v_page_offset",
    ],
    do_not_specialize_on_alignment=[
        "page_ids_len",
        "hp_page_ids_len",
        "indptr_len",
        "num_pages",
        "num_hp_pages",
        "num_layers",
        "local_layer",
        "v_page_offset",
    ],
)
def _fp4_mla_generation_fused_qk_rope_cache_update_kernel(
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    v_packed_ptr,
    hp_pool_ptr,
    latent_cache_ptr,
    global_scale_ptr,
    q_global_scale_ptr,
    rotary_cos_sin_ptr,
    q_pe_ptr,
    q_rope_out_ptr,
    q_full_ptr,
    q_fp4_out_ptr,
    q_sf_out_ptr,
    kv_lens_ptr,
    prompt_lens_ptr,
    page_ids_ptr,
    hp_page_ids_ptr,
    paged_kv_indptr_ptr,
    page_ids_len,
    hp_page_ids_len,
    indptr_len,
    num_pages,
    num_hp_pages,
    num_layers,
    local_layer,
    v_page_offset,
    page_size: tl.constexpr,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    pool_s0,
    pool_s1,
    vsf_s0,
    vsf_s1,
    v_packed_s0,
    v_packed_s1,
    q_pe_s0,
    q_pe_s1,
    q_pe_s2,
    q_out_s0,
    q_out_s1,
    q_out_s2,
    HEAD_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
    HP_POOL_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    K_RESIDUAL_D: tl.constexpr,
    STORE_K_RESIDUAL: tl.constexpr,
    FUSE_ROPE_CACHE_STORE: tl.constexpr,
    WRITE_V_PACKED: tl.constexpr,
    MAX_GEN_TILES: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    ROPE_PAIR_BLOCK: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    Q_HEAD_BLOCKS: tl.constexpr,
    BLOCK_Q_HEADS: tl.constexpr,
    Q_PREFIX_D: tl.constexpr,
    Q_PREFIX_BLOCK_D: tl.constexpr,
    Q_PREFIX_BLOCKS: tl.constexpr,
    Q_PREFIX_BLOCKS_PER_PROGRAM: tl.constexpr,
    Q_WORK_BLOCKS: tl.constexpr,
    Q_SF_COLS: tl.constexpr,
    WRITE_Q: tl.constexpr,
    Q1_KV_BLOCKS_PER_PROGRAM: tl.constexpr,
):
    q1_shared_main: tl.constexpr = FUSE_ROPE_CACHE_STORE and MAX_GEN_TILES == 1
    q1_grouped_kv: tl.constexpr = q1_shared_main and Q1_KV_BLOCKS_PER_PROGRAM > 1
    q1_main_kv_programs: tl.constexpr = V_HEAD_D // FP4_BLOCK // Q1_KV_BLOCKS_PER_PROGRAM
    q_prefix_programs: tl.constexpr = Q_PREFIX_BLOCKS // Q_PREFIX_BLOCKS_PER_PROGRAM
    q1_kv_blocks_per_iteration: tl.constexpr = 2
    tl.static_assert(Q1_KV_BLOCKS_PER_PROGRAM == 1 or q1_grouped_kv)
    tl.static_assert(
        Q1_KV_BLOCKS_PER_PROGRAM == 1 or Q1_KV_BLOCKS_PER_PROGRAM % q1_kv_blocks_per_iteration == 0
    )
    tl.static_assert(V_HEAD_D % (FP4_BLOCK * Q1_KV_BLOCKS_PER_PROGRAM) == 0)
    tl.static_assert(Q_PREFIX_BLOCKS % Q_PREFIX_BLOCKS_PER_PROGRAM == 0)
    seq_idx = tl.program_id(0)
    if (local_layer < 0) | (local_layer >= num_layers):
        return
    if seq_idx + 1 >= indptr_len:
        return
    gen_len = tl.load(prompt_lens_ptr + seq_idx)
    if gen_len <= 0:
        return
    kv_len = tl.load(kv_lens_ptr + seq_idx)
    # Generation tokens are request-major. Deriving their absolute position
    # here avoids materializing per-token metadata in a CUDA graph.
    first_new_pos = kv_len - gen_len
    if FUSE_ROPE_CACHE_STORE:
        work_idx = tl.program_id(1)
        dim_block = work_idx
        if work_idx >= 0:
            q_program = work_idx
            if (
                q1_grouped_kv
                and Q_WORK_BLOCKS == 2
                and Q_HEAD_BLOCKS >= 3
                and q_prefix_programs == 1
            ):
                # Move the final prefix program ahead of the last tail
                # programs to shorten the grid drain without changing Q work.
                penultimate_tail: tl.constexpr = Q_HEAD_BLOCKS * Q_WORK_BLOCKS - 3
                final_prefix: tl.constexpr = penultimate_tail + 1
                q_program = tl.where(
                    work_idx == penultimate_tail,
                    final_prefix,
                    tl.where(
                        work_idx == final_prefix,
                        penultimate_tail,
                        work_idx,
                    ),
                )
            q_token_head_program = q_program // Q_WORK_BLOCKS
            q_work_block = q_program - q_token_head_program * Q_WORK_BLOCKS
            q_token_idx = q_token_head_program // Q_HEAD_BLOCKS
            q_head_block = q_token_head_program - q_token_idx * Q_HEAD_BLOCKS
            if q_token_idx < gen_len:
                q_token = seq_idx * gen_len + q_token_idx
                head_lanes = tl.arange(0, BLOCK_Q_HEADS)
                head_offsets = q_head_block * BLOCK_Q_HEADS + head_lanes
                q_rows = q_token * NUM_Q_HEADS + head_offsets
                q_input_row_stride: tl.constexpr = Q_PREFIX_D + ROPE_DIM
                q_packed_row_stride: tl.constexpr = (Q_PREFIX_D + 2 * ROPE_DIM) // 2
                # Every lane is valid only when the launch covers the head
                # dimension with an exact number of full head blocks.
                if NUM_Q_HEADS == Q_HEAD_BLOCKS * BLOCK_Q_HEADS:
                    row_mask = tl.full(
                        [BLOCK_Q_HEADS],
                        True,
                        dtype=tl.int1,
                    )
                else:
                    row_mask = head_offsets < NUM_Q_HEADS
                if WRITE_Q and q_work_block < q_prefix_programs:
                    if Q_PREFIX_BLOCKS_PER_PROGRAM >= 1:
                        tl.static_assert(Q_PREFIX_D % FP4_BLOCK == 0)
                        tl.static_assert(Q_PREFIX_D % Q_PREFIX_BLOCK_D == 0)
                        tl.static_assert(Q_PREFIX_BLOCK_D % FP4_BLOCK == 0)
                        q_global_scale = tl.load(q_global_scale_ptr).to(tl.float32)
                        prefix_dims = tl.arange(0, Q_PREFIX_BLOCK_D)
                        prefix_base = q_rows[:, None].to(tl.int64) * q_input_row_stride
                        prefix_groups: tl.constexpr = Q_PREFIX_BLOCK_D // FP4_BLOCK
                        q_prefix_iterations: tl.constexpr = Q_PREFIX_BLOCKS_PER_PROGRAM
                        for q_prefix_iteration in tl.range(
                            0,
                            q_prefix_iterations,
                            num_stages=1,
                            loop_unroll_factor=1,
                            disable_licm=True,
                        ):
                            prefix_work_block = (
                                q_work_block * Q_PREFIX_BLOCKS_PER_PROGRAM + q_prefix_iteration
                            )
                            prefix_dim_offset = prefix_work_block * Q_PREFIX_BLOCK_D
                            prefix_offsets = prefix_base + (
                                prefix_dim_offset + prefix_dims[None, :]
                            ).to(tl.int64)
                            prefix_values = tl.load(
                                q_full_ptr + prefix_offsets,
                                mask=row_mask[:, None],
                                other=0.0,
                            ).to(tl.float32)
                            prefix_pairs = tl.reshape(
                                prefix_values * q_global_scale,
                                (
                                    BLOCK_Q_HEADS,
                                    prefix_groups,
                                    FP4_BLOCK // 2,
                                    2,
                                ),
                            )
                            prefix_even, prefix_odd = tl.split(prefix_pairs)
                            prefix_packed_words, prefix_scales = _fp4_mla_quantize_q_groups_words(
                                prefix_even,
                                prefix_odd,
                                BLOCK_Q_HEADS,
                                prefix_groups,
                            )
                            prefix_group_offsets = prefix_work_block * prefix_groups + tl.arange(
                                0, prefix_groups
                            )
                            prefix_word_offsets = q_rows[:, None, None].to(
                                tl.int64
                            ) * q_packed_row_stride + (
                                prefix_group_offsets[None, :, None] * (FP4_BLOCK // 2)
                                + tl.arange(0, FP4_BLOCK // 8)[None, None, :] * 4
                            ).to(tl.int64)
                            tl.store(
                                (q_fp4_out_ptr + prefix_word_offsets).to(
                                    tl.pointer_type(tl.uint32)
                                ),
                                prefix_packed_words,
                                mask=row_mask[:, None, None],
                            )
                            if NUM_Q_HEADS == 128 and BLOCK_Q_HEADS == 32:
                                tl.static_assert(NUM_Q_HEADS == 128)
                                tl.static_assert(BLOCK_Q_HEADS == 32)
                                prefix_sf_offsets = _fp4_mla_q_sf_offset_h128(
                                    q_token,
                                    q_head_block,
                                    head_lanes[:, None],
                                    prefix_group_offsets[None, :],
                                    Q_SF_COLS,
                                )
                            else:
                                prefix_sf_offsets = _fp4_mla_q_sf_offset(
                                    q_rows[:, None],
                                    prefix_group_offsets[None, :],
                                    Q_SF_COLS,
                                )
                            tl.store(
                                q_sf_out_ptr + prefix_sf_offsets,
                                prefix_scales,
                                mask=row_mask[:, None],
                            )
                else:
                    position = (first_new_pos + q_token_idx).to(tl.int64)
                    valid_position = position >= 0
                    pair_offsets = tl.arange(0, ROPE_PAIR_BLOCK)
                    pair_mask = pair_offsets < ROPE_DIM // 2
                    rotary_offsets = position * (ROPE_DIM * 2) + pair_offsets * 2
                    cos = tl.load(
                        rotary_cos_sin_ptr + rotary_offsets,
                        mask=valid_position & pair_mask,
                        other=0.0,
                    ).to(tl.float32)
                    sin = tl.load(
                        rotary_cos_sin_ptr + rotary_offsets + 1,
                        mask=valid_position & pair_mask,
                        other=0.0,
                    ).to(tl.float32)
                    q_base = q_token * q_pe_s0 + head_offsets[:, None].to(tl.int64) * q_pe_s1
                    q_dims = tl.arange(0, ROPE_DIM)
                    q_dim_mask = valid_position & row_mask[:, None] & (q_dims[None, :] < ROPE_DIM)
                    q_offsets = q_base + q_dims[None, :].to(tl.int64) * q_pe_s2
                    q_values = tl.load(
                        q_pe_ptr + q_offsets,
                        mask=q_dim_mask,
                        other=0.0,
                    ).to(tl.float32)
                    q_pairs = tl.reshape(q_values, (BLOCK_Q_HEADS, ROPE_DIM // 2, 2))
                    q_even, q_odd = tl.split(q_pairs)
                    q_roped_even, q_roped_odd = _fp4_mla_rope_fp32(
                        q_even, q_odd, cos[None, :], sin[None, :]
                    )
                    q_roped_even = q_roped_even.to(tl.bfloat16)
                    q_roped_odd = q_roped_odd.to(tl.bfloat16)
                    q_out_base = q_token * q_out_s0 + head_offsets[:, None].to(tl.int64) * q_out_s1
                    q_out_offsets = q_out_base + q_dims[None, :].to(tl.int64) * q_out_s2
                    q_roped = tl.reshape(
                        tl.join(q_roped_even, q_roped_odd),
                        (BLOCK_Q_HEADS, ROPE_DIM),
                    )
                    tl.store(
                        q_rope_out_ptr + q_out_offsets,
                        q_roped,
                        mask=q_dim_mask,
                    )
                    if WRITE_Q:
                        tl.static_assert(ROPE_DIM == 64)
                        tl.static_assert(ROPE_PAIR_BLOCK == 32)
                        q_global_scale = tl.load(q_global_scale_ptr).to(tl.float32)
                        tail_even = tl.reshape(
                            q_roped_even.to(tl.float32) * q_global_scale,
                            (
                                BLOCK_Q_HEADS,
                                ROPE_DIM // FP4_BLOCK,
                                FP4_BLOCK // 2,
                            ),
                        )
                        tail_odd = tl.reshape(
                            q_roped_odd.to(tl.float32) * q_global_scale,
                            (
                                BLOCK_Q_HEADS,
                                ROPE_DIM // FP4_BLOCK,
                                FP4_BLOCK // 2,
                            ),
                        )
                        packed, scales, residual_packed, residual_scales = (
                            _fp4_mla_quantize_q_residual_groups(tail_even, tail_odd)
                        )
                        prefix_groups: tl.constexpr = Q_PREFIX_D // FP4_BLOCK
                        tail_group_count: tl.constexpr = ROPE_DIM // FP4_BLOCK
                        output_groups = prefix_groups + tl.arange(0, tail_group_count * 2)
                        bytes_in_group = tl.arange(0, FP4_BLOCK // 2)
                        packed_tail = tl.reshape(
                            tl.join(packed, residual_packed).permute(0, 1, 3, 2),
                            (
                                BLOCK_Q_HEADS,
                                tail_group_count * 2,
                                FP4_BLOCK // 2,
                            ),
                        )
                        output_offsets = q_rows[:, None, None].to(
                            tl.int64
                        ) * q_packed_row_stride + (
                            output_groups[None, :, None] * (FP4_BLOCK // 2)
                            + bytes_in_group[None, None, :]
                        ).to(tl.int64)
                        tail_row_mask = valid_position & row_mask
                        tl.store(
                            q_fp4_out_ptr + output_offsets,
                            packed_tail,
                            mask=tail_row_mask[:, None, None],
                        )
                        tail_scales = tl.reshape(
                            tl.join(scales, residual_scales),
                            (BLOCK_Q_HEADS, tail_group_count * 2),
                        )
                        if NUM_Q_HEADS == 128 and BLOCK_Q_HEADS == 32:
                            sf_offsets = _fp4_mla_q_sf_offset_h128(
                                q_token,
                                q_head_block,
                                head_lanes[:, None],
                                output_groups[None, :],
                                Q_SF_COLS,
                            )
                        else:
                            sf_offsets = _fp4_mla_q_sf_offset(
                                q_rows[:, None],
                                output_groups[None, :],
                                Q_SF_COLS,
                            )
                        tl.store(
                            q_sf_out_ptr + sf_offsets,
                            tail_scales,
                            mask=tail_row_mask[:, None],
                        )
            if q1_grouped_kv:
                if work_idx > q1_main_kv_programs:
                    return
                dim_block = tl.where(
                    work_idx == q1_main_kv_programs,
                    V_HEAD_D // FP4_BLOCK,
                    work_idx,
                )
            else:
                if work_idx >= NUM_DIM_BLOCKS:
                    return
    else:
        dim_block = tl.program_id(2)
    if FUSE_ROPE_CACHE_STORE and MAX_GEN_TILES == 1:
        if dim_block * FP4_BLOCK > V_HEAD_D:
            return
    first_tile_pos = (first_new_pos // HP_BLOCK) * HP_BLOCK
    page_start = tl.load(paged_kv_indptr_ptr + seq_idx).to(tl.int64)
    page_end = tl.load(paged_kv_indptr_ptr + seq_idx + 1).to(tl.int64)
    hp_index = tl.full((), 0, tl.int64)

    if q1_grouped_kv:
        # A single V owner can share the K-residual program.  Multi-owner
        # specializations retain their interleaved dimension ownership.
        if q1_main_kv_programs == 1:
            kv_owner = work_idx == q1_main_kv_programs
            kv_work_idx = work_idx - q1_main_kv_programs
        else:
            kv_owner = work_idx < q1_main_kv_programs
            kv_work_idx = work_idx
        if kv_owner:
            block_base_pos = first_tile_pos
            if block_base_pos >= kv_len:
                return
            if block_base_pos + HP_BLOCK <= first_new_pos:
                return

            page_idx = block_base_pos // page_size
            page_pos = block_base_pos - page_idx * page_size
            physical_page_offset = page_start + page_idx
            if (
                (page_pos < 0)
                | (page_pos >= page_size)
                | (physical_page_offset < page_start)
                | (physical_page_offset >= page_end)
                | (physical_page_offset < 0)
                | (physical_page_offset >= page_ids_len)
            ):
                return
            physical_page = tl.load(page_ids_ptr + physical_page_offset).to(tl.int64)
            if (physical_page < 0) | (physical_page >= num_pages):
                return
            if physical_page_offset >= hp_page_ids_len:
                return
            hp_index = tl.load(hp_page_ids_ptr + physical_page_offset).to(tl.int64)
            if (hp_index < 0) | (hp_index >= num_hp_pages):
                return

            latent_token = seq_idx * gen_len
            global_scale = tl.load(global_scale_ptr)
            # Each grouped CTA stages only the dimensions it owns.  The barrier
            # then makes the new BF16 token visible to that CTA's rolled loop.
            prestage_linear = tl.arange(0, Q1_KV_BLOCKS_PER_PROGRAM * FP4_BLOCK)
            prestage_iteration = prestage_linear // FP4_BLOCK
            prestage_lane = prestage_linear % FP4_BLOCK
            prestage_dim = (
                kv_work_idx + prestage_iteration * q1_main_kv_programs
            ) * FP4_BLOCK + prestage_lane
            prestage_values = tl.load(
                latent_cache_ptr
                + latent_token.to(tl.int64) * (NUM_DIM_BLOCKS * FP4_BLOCK)
                + prestage_dim.to(tl.int64)
            )
            prestage_slot = first_new_pos % HP_POOL_SIZE
            tl.store(
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + prestage_slot * HEAD_D
                + prestage_dim,
                prestage_values,
            )
            tl.debug_barrier()
            tail_count = kv_len - block_base_pos
            if tail_count <= HP_BLOCK // 2:
                tl.static_assert(Q1_KV_BLOCKS_PER_PROGRAM % 4 == 0)
                for dim_group in tl.range(
                    0,
                    Q1_KV_BLOCKS_PER_PROGRAM // 4,
                    num_stages=1,
                    loop_unroll_factor=1,
                    disable_licm=True,
                ):
                    dim_block_base = kv_work_idx + dim_group * 4 * q1_main_kv_programs
                    _fp4_mla_generation_q1_kv_tail8_update(
                        kv_cache_ptr,
                        sf_cache_ptr,
                        v_sf_ptr,
                        v_packed_ptr,
                        hp_pool_ptr,
                        hp_index,
                        kv_len,
                        block_base_pos,
                        page_pos,
                        physical_page,
                        dim_block_base,
                        global_scale,
                        local_layer,
                        v_page_offset,
                        kv_s0,
                        kv_s2,
                        kv_s4,
                        sf_s0,
                        pool_s0,
                        pool_s1,
                        vsf_s0,
                        vsf_s1,
                        v_packed_s0,
                        v_packed_s1,
                        HEAD_D,
                        V_HEAD_D,
                        HP_BLOCK,
                        HP_POOL_SIZE,
                        FP4_BLOCK,
                        SF_PER_TOKEN,
                        SF_PER_PAGE,
                        WRITE_V_PACKED,
                        q1_main_kv_programs,
                    )
            else:
                for dim_group in tl.range(
                    0,
                    Q1_KV_BLOCKS_PER_PROGRAM // q1_kv_blocks_per_iteration,
                    num_stages=1,
                    loop_unroll_factor=1,
                    disable_licm=True,
                ):
                    dim_iterations = dim_group * q1_kv_blocks_per_iteration + tl.arange(
                        0, q1_kv_blocks_per_iteration
                    )
                    pair_dim_blocks = kv_work_idx + dim_iterations * q1_main_kv_programs
                    _fp4_mla_generation_q1_kv_tile_update(
                        kv_cache_ptr,
                        sf_cache_ptr,
                        v_sf_ptr,
                        v_packed_ptr,
                        hp_pool_ptr,
                        hp_index,
                        kv_len,
                        block_base_pos,
                        page_pos,
                        physical_page,
                        pair_dim_blocks,
                        global_scale,
                        local_layer,
                        v_page_offset,
                        kv_s0,
                        kv_s2,
                        kv_s4,
                        sf_s0,
                        pool_s0,
                        pool_s1,
                        vsf_s0,
                        vsf_s1,
                        v_packed_s0,
                        v_packed_s1,
                        HEAD_D,
                        V_HEAD_D,
                        HP_BLOCK,
                        HP_POOL_SIZE,
                        FP4_BLOCK,
                        SF_PER_TOKEN,
                        SF_PER_PAGE,
                        WRITE_V_PACKED,
                    )
        if work_idx < q1_main_kv_programs:
            return

    byte_offsets = tl.arange(0, HP_BLOCK // 2)
    token_offsets = tl.arange(0, HP_BLOCK)
    even_d = dim_block * FP4_BLOCK + byte_offsets * 2
    odd_d = even_d + 1
    all_d = dim_block * FP4_BLOCK + tl.arange(0, FP4_BLOCK)
    mask_even_d = even_d < HEAD_D
    mask_odd_d = odd_d < HEAD_D
    mask_all_d = all_d < HEAD_D
    safe_even_d = tl.where(mask_even_d, even_d, 0)
    safe_odd_d = tl.where(mask_odd_d, odd_d, 0)
    safe_all_d = tl.where(mask_all_d, all_d, 0)

    if FUSE_ROPE_CACHE_STORE and MAX_GEN_TILES == 1:
        if dim_block * FP4_BLOCK >= V_HEAD_D:
            tl.static_assert(STORE_K_RESIDUAL)
            tl.static_assert(HEAD_D - K_RESIDUAL_D == V_HEAD_D)
            tl.static_assert(K_RESIDUAL_D % FP4_BLOCK == 0)
            position = first_new_pos.to(tl.int64)
            if position < 0:
                return
            page_idx = position // page_size
            page_pos = position - page_idx * page_size
            physical_page_offset = page_start + page_idx
            if (
                (page_pos < 0)
                | (page_pos >= page_size)
                | (physical_page_offset < page_start)
                | (physical_page_offset >= page_end)
                | (physical_page_offset < 0)
                | (physical_page_offset >= page_ids_len)
            ):
                return
            physical_page = tl.load(page_ids_ptr + physical_page_offset).to(tl.int64)
            if (physical_page < 0) | (physical_page >= num_pages):
                return
            if physical_page_offset >= hp_page_ids_len:
                return
            hp_index = tl.load(hp_page_ids_ptr + physical_page_offset).to(tl.int64)
            if (hp_index < 0) | (hp_index >= num_hp_pages):
                return

            latent_token = seq_idx * gen_len
            tail_byte_offsets = tl.arange(0, K_RESIDUAL_D // 2)
            tail_even_d = V_HEAD_D + tail_byte_offsets * 2
            tail_odd_d = tail_even_d + 1
            tail_even = tl.load(
                latent_cache_ptr + latent_token * (NUM_DIM_BLOCKS * FP4_BLOCK) + tail_even_d,
            ).to(tl.float32)
            tail_odd = tl.load(
                latent_cache_ptr + latent_token * (NUM_DIM_BLOCKS * FP4_BLOCK) + tail_odd_d,
            ).to(tl.float32)
            rope_pair_offsets = tail_byte_offsets
            rotary_offsets = position * (ROPE_DIM * 2) + rope_pair_offsets * 2
            cos = tl.load(rotary_cos_sin_ptr + rotary_offsets).to(tl.float32)
            sin = tl.load(rotary_cos_sin_ptr + rotary_offsets + 1).to(tl.float32)
            tail_even, tail_odd = _fp4_mla_rope_fp32(tail_even, tail_odd, cos, sin)
            tail_even = tail_even.to(tl.bfloat16).to(tl.float32)
            tail_odd = tail_odd.to(tl.bfloat16).to(tl.float32)

            residual_groups: tl.constexpr = K_RESIDUAL_D // FP4_BLOCK
            tail_even_groups = tl.reshape(
                tail_even,
                (residual_groups, FP4_BLOCK // 2),
            )
            tail_odd_groups = tl.reshape(
                tail_odd,
                (residual_groups, FP4_BLOCK // 2),
            )
            token_amax = tl.maximum(
                tl.max(tl.abs(tail_even_groups), axis=1),
                tl.max(tl.abs(tail_odd_groups), axis=1),
            )
            token_scale = tl.where(token_amax > 0.0, token_amax / 6.0, 1.0)
            global_scale = tl.load(global_scale_ptr)
            stored_scale = token_scale * global_scale
            quant_scale = stored_scale.to(tl.float8e4nv).to(tl.float32) / global_scale
            low = _fp4_e2m1_quantize(tail_even_groups / quant_scale[:, None])
            high = _fp4_e2m1_quantize(tail_odd_groups / quant_scale[:, None])
            packed = low | (high << 4)

            group_offsets = tl.arange(0, residual_groups)
            bytes_in_group = tl.arange(0, FP4_BLOCK // 2)
            packed_cols = (
                V_HEAD_D // 2 + group_offsets[:, None] * (FP4_BLOCK // 2) + bytes_in_group[None, :]
            )
            kv_base = physical_page * kv_s0
            tl.store(
                kv_cache_ptr + kv_base + page_pos * kv_s2 + packed_cols * kv_s4,
                packed,
            )
            k_sf_cols = V_HEAD_D // FP4_BLOCK + group_offsets
            k_sf_offset = _fp4_mla_swizzled_sf_offset(page_pos, k_sf_cols, SF_PER_TOKEN)
            tl.store(
                sf_cache_ptr + physical_page * sf_s0 + k_sf_offset,
                stored_scale,
            )

            main_even, main_odd = _fp4_mla_unpack_e2m1x2(packed)
            residual_even = tail_even_groups - main_even * quant_scale[:, None]
            residual_odd = tail_odd_groups - main_odd * quant_scale[:, None]
            residual_amax = tl.maximum(
                tl.max(tl.abs(residual_even), axis=1),
                tl.max(tl.abs(residual_odd), axis=1),
            )
            residual_scale = tl.where(residual_amax > 0.0, residual_amax / 6.0, 1.0)
            residual_stored_scale = residual_scale * global_scale
            residual_quant_scale = (
                residual_stored_scale.to(tl.float8e4nv).to(tl.float32) / global_scale
            )
            residual_low = _fp4_e2m1_quantize(residual_even / residual_quant_scale[:, None])
            residual_high = _fp4_e2m1_quantize(residual_odd / residual_quant_scale[:, None])
            residual_packed = residual_low | (residual_high << 4)
            residual_packed_cols = (
                HEAD_D // 2 + group_offsets[:, None] * (FP4_BLOCK // 2) + bytes_in_group[None, :]
            )
            tl.store(
                kv_cache_ptr + kv_base + page_pos * kv_s2 + residual_packed_cols * kv_s4,
                residual_packed,
            )
            residual_sf_offset = _fp4_mla_swizzled_sf_offset(
                page_pos,
                HEAD_D // FP4_BLOCK + group_offsets,
                SF_PER_TOKEN,
            )
            tl.store(
                sf_cache_ptr + physical_page * sf_s0 + residual_sf_offset,
                residual_stored_scale,
            )

            hp_slot = position % HP_POOL_SIZE
            hp_store_base = (
                hp_pool_ptr + hp_index * pool_s0 + local_layer * pool_s1 + hp_slot * HEAD_D
            )
            tl.store(hp_store_base + tail_even_d, tail_even)
            tl.store(hp_store_base + tail_odd_d, tail_odd)
            return

    for tile_iter in tl.static_range(0, MAX_GEN_TILES):
        if FUSE_ROPE_CACHE_STORE:
            tile_idx = tile_iter
        else:
            tile_idx = tl.program_id(1)
        block_base_pos = first_tile_pos + tile_idx * HP_BLOCK
        if block_base_pos >= kv_len:
            return
        if block_base_pos + HP_BLOCK <= first_new_pos:
            return

        page_idx = block_base_pos // page_size
        page_pos = block_base_pos - page_idx * page_size
        physical_page_offset = page_start + page_idx
        if (
            (page_pos < 0)
            | (page_pos >= page_size)
            | (physical_page_offset < page_start)
            | (physical_page_offset >= page_end)
            | (physical_page_offset < 0)
            | (physical_page_offset >= page_ids_len)
        ):
            return
        physical_page = tl.load(page_ids_ptr + physical_page_offset).to(tl.int64)
        if (physical_page < 0) | (physical_page >= num_pages):
            return
        if physical_page_offset >= hp_page_ids_len:
            return
        hp_index = tl.load(hp_page_ids_ptr + physical_page_offset).to(tl.int64)
        if (hp_index < 0) | (hp_index >= num_hp_pages):
            return

        abs_positions = block_base_pos + token_offsets
        valid_tokens = abs_positions < kv_len
        from_latent = abs_positions >= first_new_pos
        hp_slots = abs_positions % HP_POOL_SIZE
        new_token_offsets = abs_positions - first_new_pos
        # Linear MTP uses a uniform generation length, so each sequence
        # occupies one contiguous gen_len slice in latent_cache.
        latent_tokens = seq_idx * gen_len + new_token_offsets
        safe_latent_tokens = tl.where(valid_tokens & from_latent, latent_tokens, 0).to(tl.int64)

        if q1_shared_main:
            # Q1 reads each token from exactly one source. Selecting the pointer
            # first removes two masked loads and additions without changing MTP.
            hp_even_ptrs = (
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + hp_slots[:, None] * HEAD_D
                + safe_even_d[None, :]
            )
            hp_odd_ptrs = (
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + hp_slots[:, None] * HEAD_D
                + safe_odd_d[None, :]
            )
            latent_even_ptrs = (
                latent_cache_ptr
                + safe_latent_tokens[:, None] * (NUM_DIM_BLOCKS * FP4_BLOCK)
                + safe_even_d[None, :]
            )
            latent_odd_ptrs = (
                latent_cache_ptr
                + safe_latent_tokens[:, None] * (NUM_DIM_BLOCKS * FP4_BLOCK)
                + safe_odd_d[None, :]
            )
            even_values = tl.load(
                tl.where(from_latent[:, None], latent_even_ptrs, hp_even_ptrs),
                mask=valid_tokens[:, None] & mask_even_d[None, :],
                other=0.0,
            ).to(tl.float32)
            odd_values = tl.load(
                tl.where(from_latent[:, None], latent_odd_ptrs, hp_odd_ptrs),
                mask=valid_tokens[:, None] & mask_odd_d[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            hp_even = tl.load(
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + hp_slots[:, None] * HEAD_D
                + safe_even_d[None, :],
                mask=valid_tokens[:, None] & (~from_latent)[:, None] & mask_even_d[None, :],
                other=0.0,
            ).to(tl.float32)
            hp_odd = tl.load(
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + hp_slots[:, None] * HEAD_D
                + safe_odd_d[None, :],
                mask=valid_tokens[:, None] & (~from_latent)[:, None] & mask_odd_d[None, :],
                other=0.0,
            ).to(tl.float32)
            latent_even = tl.load(
                latent_cache_ptr
                + safe_latent_tokens[:, None] * (NUM_DIM_BLOCKS * FP4_BLOCK)
                + safe_even_d[None, :],
                mask=valid_tokens[:, None] & from_latent[:, None] & mask_even_d[None, :],
                other=0.0,
            ).to(tl.float32)
            latent_odd = tl.load(
                latent_cache_ptr
                + safe_latent_tokens[:, None] * (NUM_DIM_BLOCKS * FP4_BLOCK)
                + safe_odd_d[None, :],
                mask=valid_tokens[:, None] & from_latent[:, None] & mask_odd_d[None, :],
                other=0.0,
            ).to(tl.float32)
            even_values = hp_even + latent_even
            odd_values = hp_odd + latent_odd

        if FUSE_ROPE_CACHE_STORE and not q1_shared_main:
            if dim_block * FP4_BLOCK >= V_HEAD_D:
                positions = abs_positions.to(tl.int64)
                valid_position = valid_tokens & from_latent & (positions >= 0)
                rope_pair_offsets = (safe_even_d - V_HEAD_D) // 2
                rope_dim_mask = mask_even_d & (even_d >= V_HEAD_D) & (odd_d < V_HEAD_D + ROPE_DIM)
                rotary_offsets = (
                    positions[:, None] * (ROPE_DIM * 2) + rope_pair_offsets[None, :] * 2
                )
                rotary_mask = valid_position[:, None] & rope_dim_mask[None, :]
                cos = tl.load(
                    rotary_cos_sin_ptr + rotary_offsets,
                    mask=rotary_mask,
                    other=0.0,
                ).to(tl.float32)
                sin = tl.load(
                    rotary_cos_sin_ptr + rotary_offsets + 1,
                    mask=rotary_mask,
                    other=0.0,
                ).to(tl.float32)
                roped_even, roped_odd = _fp4_mla_rope_fp32(even_values, odd_values, cos, sin)
                roped_even = roped_even.to(tl.bfloat16).to(tl.float32)
                roped_odd = roped_odd.to(tl.bfloat16).to(tl.float32)
                even_values = tl.where(rotary_mask, roped_even, even_values)
                odd_values = tl.where(rotary_mask, roped_odd, odd_values)

        amax_per_token = tl.maximum(
            tl.max(tl.abs(even_values), axis=1),
            tl.max(tl.abs(odd_values), axis=1),
        )
        global_scale = tl.load(global_scale_ptr)
        token_scale = tl.where(amax_per_token > 0.0, amax_per_token / 6.0, 1.0)
        if q1_shared_main:
            tile_amax = tl.max(amax_per_token, axis=0)
            tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)
            local_scale = tl.full((HP_BLOCK,), tile_scale, tl.float32)
            token_scale_col = page_pos // HP_BLOCK
            sf_offsets = _fp4_mla_swizzled_sf_offset(safe_all_d, token_scale_col, SF_PER_PAGE)
            v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
                vsf_s0, tl.int64
            ) + physical_page * tl.cast(vsf_s1, tl.int64)
            tl.store(
                v_sf_ptr + v_sf_base + sf_offsets.to(tl.int64),
                tile_scale * global_scale,
                mask=mask_all_d & (all_d < V_HEAD_D),
            )
        else:
            shared_tile = dim_block * FP4_BLOCK < V_HEAD_D
            if shared_tile:
                tile_amax = tl.max(amax_per_token, axis=0)
                tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)
                local_scale = tl.full((HP_BLOCK,), tile_scale, tl.float32)
                token_scale_col = page_pos // HP_BLOCK
                sf_offsets = _fp4_mla_swizzled_sf_offset(
                    safe_all_d,
                    token_scale_col,
                    SF_PER_PAGE,
                )
                v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
                    vsf_s0, tl.int64
                ) + physical_page * tl.cast(vsf_s1, tl.int64)
                tl.store(
                    v_sf_ptr + v_sf_base + sf_offsets.to(tl.int64),
                    tile_scale * global_scale,
                    mask=mask_all_d & (all_d < V_HEAD_D),
                )
            else:
                local_scale = token_scale
        stored_scale = local_scale * global_scale

        quant_scale = local_scale
        if STORE_K_RESIDUAL and not q1_shared_main:
            if dim_block * FP4_BLOCK >= HEAD_D - K_RESIDUAL_D:
                quant_scale = stored_scale.to(tl.float8e4nv).to(tl.float32) / global_scale
        low = _fp4_e2m1_quantize(even_values / quant_scale[:, None])
        high = _fp4_e2m1_quantize(odd_values / quant_scale[:, None])
        packed = low | (high << 4)
        v_dim_packed = packed

        packed_cols = dim_block * (FP4_BLOCK // 2) + byte_offsets
        page_positions = page_pos + token_offsets
        kv_base = physical_page * kv_s0
        tl.store(
            kv_cache_ptr + kv_base + page_positions[:, None] * kv_s2 + packed_cols[None, :] * kv_s4,
            packed,
            mask=valid_tokens[:, None] & mask_even_d[None, :],
        )

        if WRITE_V_PACKED:
            if q1_shared_main or dim_block * FP4_BLOCK < V_HEAD_D:
                # The canonical cache packs dimensions; PV packs tokens.
                # Transpose the selected V codes in registers.
                v_pairs = tl.reshape(
                    v_dim_packed.T,
                    (FP4_BLOCK // 2, HP_BLOCK // 2, 2),
                )
                even_token_packed, odd_token_packed = tl.split(v_pairs)
                v_low = (even_token_packed & 0x0F) | ((odd_token_packed & 0x0F) << 4)
                v_high = ((even_token_packed >> 4) & 0x0F) | (odd_token_packed & 0xF0)
                # Keep the two dimension parities in the 8x8 layout produced
                # by the token-pair transpose.  Interleaving them into a
                # 16x8 tensor forces another cross-warp layout conversion;
                # two stores address the exact same coalesced rows without
                # changing any packed bytes.
                v_even_rows = (
                    (v_page_offset + physical_page) * V_HEAD_D
                    + dim_block * FP4_BLOCK
                    + tl.arange(0, FP4_BLOCK // 2) * 2
                ).to(tl.int64)
                pair_offsets = tl.arange(0, HP_BLOCK // 2)
                v_cols = page_pos // 2 + pair_offsets
                v_even_offsets = v_even_rows[:, None] * v_packed_s0 + v_cols[None, :] * v_packed_s1
                v_odd_offsets = v_even_offsets + v_packed_s0
                token0_valid = block_base_pos + pair_offsets * 2 < kv_len
                token1_valid = block_base_pos + pair_offsets * 2 + 1 < kv_len
                partial_pair = token0_valid & (~token1_valid)
                old_v_low = tl.load(
                    v_packed_ptr + v_even_offsets,
                    mask=partial_pair[None, :],
                    other=0,
                )
                old_v_high = tl.load(
                    v_packed_ptr + v_odd_offsets,
                    mask=partial_pair[None, :],
                    other=0,
                )
                v_low = tl.where(
                    partial_pair[None, :],
                    (old_v_low & 0xF0) | (v_low & 0x0F),
                    v_low,
                ).to(tl.uint8)
                v_high = tl.where(
                    partial_pair[None, :],
                    (old_v_high & 0xF0) | (v_high & 0x0F),
                    v_high,
                ).to(tl.uint8)
                tl.store(
                    v_packed_ptr + v_even_offsets,
                    v_low,
                    mask=token0_valid[None, :],
                )
                tl.store(
                    v_packed_ptr + v_odd_offsets,
                    v_high,
                    mask=token0_valid[None, :],
                )

        k_sf_offsets = _fp4_mla_swizzled_sf_offset(page_positions, dim_block, SF_PER_TOKEN)
        tl.store(
            sf_cache_ptr + physical_page * sf_s0 + k_sf_offsets,
            stored_scale,
            mask=valid_tokens,
        )

        if STORE_K_RESIDUAL and not q1_shared_main:
            residual_start = HEAD_D - K_RESIDUAL_D
            if dim_block * FP4_BLOCK >= residual_start:
                if MAX_GEN_TILES == 1:
                    main_even, main_odd = _fp4_mla_unpack_e2m1x2(packed)
                    main_even *= quant_scale[:, None]
                    main_odd *= quant_scale[:, None]
                else:
                    main_even = _fp4_e2m1_to_f32(low) * quant_scale[:, None]
                    main_odd = _fp4_e2m1_to_f32(high) * quant_scale[:, None]
                residual_even = even_values - main_even
                residual_odd = odd_values - main_odd
                residual_amax = tl.maximum(
                    tl.max(tl.abs(residual_even), axis=1),
                    tl.max(tl.abs(residual_odd), axis=1),
                )
                residual_scale = tl.where(residual_amax > 0.0, residual_amax / 6.0, 1.0)
                residual_stored_scale = residual_scale * global_scale
                residual_quant_scale = (
                    residual_stored_scale.to(tl.float8e4nv).to(tl.float32) / global_scale
                )
                residual_low = _fp4_e2m1_quantize(residual_even / residual_quant_scale[:, None])
                residual_high = _fp4_e2m1_quantize(residual_odd / residual_quant_scale[:, None])
                residual_packed = residual_low | (residual_high << 4)
                residual_group = dim_block - residual_start // FP4_BLOCK
                residual_packed_cols = (
                    HEAD_D // 2 + residual_group * (FP4_BLOCK // 2) + byte_offsets
                )
                tl.store(
                    kv_cache_ptr
                    + kv_base
                    + page_positions[:, None] * kv_s2
                    + residual_packed_cols[None, :] * kv_s4,
                    residual_packed,
                    mask=valid_tokens[:, None] & mask_even_d[None, :],
                )
                residual_sf_offsets = _fp4_mla_swizzled_sf_offset(
                    page_positions,
                    HEAD_D // FP4_BLOCK + residual_group,
                    SF_PER_TOKEN,
                )
                tl.store(
                    sf_cache_ptr + physical_page * sf_s0 + residual_sf_offsets,
                    residual_stored_scale,
                    mask=valid_tokens,
                )

        if FUSE_ROPE_CACHE_STORE:
            hp_store_mask = valid_tokens[:, None] & from_latent[:, None]
            hp_store_base = (
                hp_pool_ptr
                + hp_index * pool_s0
                + local_layer * pool_s1
                + hp_slots[:, None] * HEAD_D
            )
            tl.store(
                hp_store_base + safe_even_d[None, :],
                even_values,
                mask=hp_store_mask & mask_even_d[None, :],
            )
            tl.store(
                hp_store_base + safe_odd_d[None, :],
                odd_values,
                mask=hp_store_mask & mask_odd_d[None, :],
            )
            if tile_iter + 1 < MAX_GEN_TILES:
                # The next tile can wrap onto HP slots read by this tile.
                # Keep each dimension block's read-before-overwrite order.
                tl.debug_barrier()
