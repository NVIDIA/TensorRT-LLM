# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for MLA FP4 KV-cache helpers."""

import triton
import triton.language as tl

# HP pool maintenance kernels


@triton.jit
def _hp_kv_store_context_kernel(
    pool_ptr,
    latent_cache_ptr,
    seq_slots_ptr,  # int32 [num_contexts], seq_slot for each ctx seq
    kv_lens_ptr,  # int32 [num_contexts], total KV length after prefill
    token_offsets_ptr,  # int32 [num_contexts], exclusive prompt prefix sum
    prompt_lens_ptr,  # int32 [num_contexts], new tokens per ctx seq
    num_seq_slots,
    num_layers,
    layer_idx,
    pool_stride_seq,  # pool.stride(0): elements between adjacent seq slots
    pool_stride_layer,  # pool.stride(1): elements between adjacent layers
    lc_stride,  # latent_cache.stride(0): elements between adjacent tokens
    D: tl.constexpr,  # head_dim (runtime dimension, = latent_cache.shape[-1])
    POOL_HEAD_D: tl.constexpr,
    BLOCK_D: tl.constexpr,  # next_power_of_2(D), used for vectorised load/store
    HP_BLOCK: tl.constexpr,  # = HP_BLOCK_SIZE (16)
):
    """Store the tail tokens of each context sequence into the HP KV pool.

    Grid: (num_contexts, HP_BLOCK_SIZE).
    Only programs for tail tokens present in latent_cache actually write.

    For a context sequence with total KV length L = num_cached + prompt_len:
      - remainder = L % HP_BLOCK
      - The last ``remainder`` new tokens (latent_cache positions
        [offset + prompt_len - remainder, offset + prompt_len)) are stored
        into pool slots [0, remainder), which correspond to the absolute token
        positions [L - remainder, L) in the circular buffer.
    """
    ctx_idx = tl.program_id(0)
    buf_pos = tl.program_id(1)
    if (layer_idx < 0) | (layer_idx >= num_layers):
        return

    kv_len = tl.load(kv_lens_ptr + ctx_idx)
    remainder = kv_len % HP_BLOCK
    prompt_len = tl.load(prompt_lens_ptr + ctx_idx)
    store_count = tl.minimum(remainder, prompt_len)
    first_buf_pos = remainder - store_count
    if buf_pos < first_buf_pos:
        return
    if buf_pos >= remainder:
        return

    seq_slot = tl.load(seq_slots_ptr + ctx_idx).to(tl.int64)
    if (seq_slot < 0) | (seq_slot >= num_seq_slots):
        return
    tok_offset = tl.load(token_offsets_ptr + ctx_idx).to(tl.int64)

    # Index of this token within latent_cache: last `remainder` new tokens,
    # buf_pos-th of them (0-indexed from the start of the tail).
    token_idx = tok_offset + prompt_len.to(tl.int64) - remainder.to(tl.int64) + buf_pos.to(tl.int64)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    safe_offs_d = tl.where(mask_d, offs_d, 0)
    src = tl.load(
        latent_cache_ptr + token_idx * lc_stride + safe_offs_d,
        mask=mask_d,
        other=0.0,
    )

    # Destination: pool[seq_slot, layer_idx, 0, buf_pos, :D].
    dst_base = seq_slot * pool_stride_seq + layer_idx * pool_stride_layer + buf_pos * POOL_HEAD_D
    tl.store(pool_ptr + dst_base + safe_offs_d, src, mask=mask_d)


@triton.jit
def _hp_kv_store_gen_kernel(
    pool_ptr,
    latent_cache_ptr,
    seq_slots_ptr,  # int32 [num_seqs], seq_slot for each sequence
    batch_indices_ptr,  # int32 [metadata_num_tokens], sequence index per token
    positions_ptr,  # int32 [metadata_num_tokens], absolute KV position per token
    gen_tok_start,  # int, offset in latent_cache where generation tokens begin
    token_offset,  # int, offset in metadata token arrays where generation tokens begin
    num_tokens,  # int, number of generation tokens in latent_cache
    metadata_num_tokens,
    num_seq_slots,
    num_layers,
    layer_idx,
    pool_stride_seq,
    pool_stride_layer,
    lc_stride,
    D: tl.constexpr,
    POOL_HEAD_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
):
    """Store generation tokens into the HP KV pool.

    Grid: (num_generation_tokens,).
    Each program stores one token into the circular buffer position
    ``position % HP_BLOCK``, overwriting the oldest entry.  The token metadata
    is read from the same batch_indices/positions arrays used by the KV scatter
    path, so this supports linear MTP where each sequence contributes multiple
    generation tokens.
    """
    gen_token_idx = tl.program_id(0)
    if (layer_idx < 0) | (layer_idx >= num_layers):
        return
    if gen_token_idx >= num_tokens:
        return

    metadata_token_idx = token_offset + gen_token_idx
    if metadata_token_idx >= metadata_num_tokens:
        return
    batch_idx = tl.load(batch_indices_ptr + metadata_token_idx).to(tl.int64)
    position = tl.load(positions_ptr + metadata_token_idx).to(tl.int64)
    if (batch_idx < 0) | (position < 0):
        return

    seq_slot = tl.load(seq_slots_ptr + batch_idx).to(tl.int64)
    if (seq_slot < 0) | (seq_slot >= num_seq_slots):
        return
    buf_pos = position % HP_BLOCK

    token_idx = tl.cast(gen_tok_start, tl.int64) + gen_token_idx.to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    safe_offs_d = tl.where(mask_d, offs_d, 0)
    src = tl.load(
        latent_cache_ptr + token_idx * lc_stride + safe_offs_d,
        mask=mask_d,
        other=0.0,
    )

    dst_base = seq_slot * pool_stride_seq + layer_idx * pool_stride_layer + buf_pos * POOL_HEAD_D
    tl.store(pool_ptr + dst_base + safe_offs_d, src, mask=mask_d)


@triton.jit
def _hp_kv_restore_rejected_from_pool_kernel(
    pool_ptr,
    snapshot_pool_ptr,
    batch_indices_ptr,
    positions_ptr,
    seq_slots_ptr,
    kv_lens_ptr,
    prompt_lens_ptr,
    accepted_tokens_ptr,
    token_offset,
    num_tokens,
    metadata_num_tokens,
    num_seqs,
    num_accepted_tokens,
    num_seq_slots,
    num_layers,
    local_layer,
    pool_stride_seq,
    pool_stride_layer,
    snapshot_stride_seq,
    snapshot_stride_layer,
    D: tl.constexpr,
    POOL_HEAD_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    metadata_token_idx = token_offset + token_idx
    token_valid = (
        (token_idx < num_tokens)
        & (metadata_token_idx >= 0)
        & (metadata_token_idx < metadata_num_tokens)
        & (local_layer >= 0)
        & (local_layer < num_layers)
    )

    batch_idx = tl.load(batch_indices_ptr + metadata_token_idx, mask=token_valid, other=-1).to(
        tl.int64
    )
    position = tl.load(positions_ptr + metadata_token_idx, mask=token_valid, other=-1).to(tl.int64)
    batch_valid = (
        token_valid
        & (batch_idx >= 0)
        & (batch_idx < num_seqs)
        & (batch_idx < num_accepted_tokens)
        & (position >= 0)
    )

    seq_slot = tl.load(seq_slots_ptr + batch_idx, mask=batch_valid, other=-1).to(tl.int64)
    kv_len = tl.load(kv_lens_ptr + batch_idx, mask=batch_valid, other=0).to(tl.int64)
    prompt_len = tl.load(prompt_lens_ptr + batch_idx, mask=batch_valid, other=0).to(tl.int64)
    accepted = tl.load(accepted_tokens_ptr + batch_idx, mask=batch_valid, other=0).to(tl.int64)

    hp_slot = position % HP_BLOCK
    first_new_position = kv_len - prompt_len
    should_restore = (
        batch_valid
        & (seq_slot >= 0)
        & (seq_slot < num_seq_slots)
        & (hp_slot >= 0)
        & (hp_slot < HP_BLOCK)
        & (position >= first_new_position + accepted)
    )

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    safe_offs_d = tl.where(mask_d, offs_d, 0)
    src_base = (
        seq_slot * snapshot_stride_seq + local_layer * snapshot_stride_layer + hp_slot * POOL_HEAD_D
    )
    dst_base = seq_slot * pool_stride_seq + local_layer * pool_stride_layer + hp_slot * POOL_HEAD_D
    values = tl.load(
        snapshot_pool_ptr + src_base + safe_offs_d, mask=should_restore & mask_d, other=0.0
    )
    tl.store(pool_ptr + dst_base + safe_offs_d, values, mask=should_restore & mask_d)


@triton.jit
def _hp_kv_restore_rejected_from_values_kernel(
    pool_ptr,
    values_ptr,
    batch_indices_ptr,
    positions_ptr,
    seq_slots_ptr,
    hp_slots_ptr,
    first_new_positions_ptr,
    accepted_tokens_ptr,
    num_tokens,
    num_accepted_tokens,
    num_seq_slots,
    num_layers,
    local_layer,
    pool_stride_seq,
    pool_stride_layer,
    values_stride_token,
    values_stride_dim,
    D: tl.constexpr,
    POOL_HEAD_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token_valid = (token_idx < num_tokens) & (local_layer >= 0) & (local_layer < num_layers)

    batch_idx = tl.load(batch_indices_ptr + token_idx, mask=token_valid, other=-1).to(tl.int64)
    position = tl.load(positions_ptr + token_idx, mask=token_valid, other=-1).to(tl.int64)
    seq_slot = tl.load(seq_slots_ptr + token_idx, mask=token_valid, other=-1).to(tl.int64)
    hp_slot = tl.load(hp_slots_ptr + token_idx, mask=token_valid, other=-1).to(tl.int64)
    first_new_position = tl.load(first_new_positions_ptr + token_idx, mask=token_valid, other=0).to(
        tl.int64
    )
    batch_valid = token_valid & (batch_idx >= 0) & (batch_idx < num_accepted_tokens)
    accepted = tl.load(accepted_tokens_ptr + batch_idx, mask=batch_valid, other=0).to(tl.int64)
    should_restore = (
        batch_valid
        & (seq_slot >= 0)
        & (seq_slot < num_seq_slots)
        & (hp_slot >= 0)
        & (hp_slot < HP_BLOCK)
        & (position >= first_new_position + accepted)
    )

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    safe_offs_d = tl.where(mask_d, offs_d, 0)
    values = tl.load(
        values_ptr + token_idx * values_stride_token + safe_offs_d * values_stride_dim,
        mask=should_restore & mask_d,
        other=0.0,
    )
    dst_base = seq_slot * pool_stride_seq + local_layer * pool_stride_layer + hp_slot * POOL_HEAD_D
    tl.store(pool_ptr + dst_base + safe_offs_d, values, mask=should_restore & mask_d)


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


@triton.jit
def _fp4_mla_scatter_kernel(
    kv_cache_ptr,
    sf_cache_ptr,
    q_fp4_ptr,
    q_sf_ptr,
    batch_indices_ptr,
    positions_ptr,
    paged_kv_indices_ptr,
    paged_kv_indptr_ptr,
    page_ids_len,
    indptr_len,
    num_pages,
    token_offset,
    page_size,
    kv_s0,
    kv_s1,
    kv_s2,
    kv_s3,
    kv_s4,
    sf_s0,
    sf_s1,
    sf_s2,
    sf_s3,
    sf_s4,
    q_fp4_s0,
    q_fp4_s1,
    q_sf_s0,
    q_sf_s1,
    PACKED_D: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    BLOCK_PACKED_D: tl.constexpr,
    BLOCK_SF: tl.constexpr,
    USE_SWIZZLED_SF: tl.constexpr,
):
    token_idx = tl.program_id(0)
    metadata_token_idx = token_offset + token_idx

    # Keep page address math in int64; 128-token FP4 pages can overflow
    # int32 offsets in large KV pools.
    batch_idx = tl.load(batch_indices_ptr + metadata_token_idx).to(tl.int64)
    position = tl.load(positions_ptr + metadata_token_idx).to(tl.int64)
    if (batch_idx < 0) | (batch_idx + 1 >= indptr_len) | (position < 0):
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

    offs_packed = tl.arange(0, BLOCK_PACKED_D)
    mask_packed = offs_packed < PACKED_D
    safe_offs_packed = tl.where(mask_packed, offs_packed, 0)
    q_vals = tl.load(
        q_fp4_ptr + token_idx * q_fp4_s0 + safe_offs_packed * q_fp4_s1,
        mask=mask_packed,
        other=0,
    )
    kv_dst = physical_page * kv_s0 + page_pos * kv_s2
    tl.store(kv_cache_ptr + kv_dst + safe_offs_packed * kv_s4, q_vals, mask=mask_packed)

    offs_sf = tl.arange(0, BLOCK_SF)
    mask_sf = offs_sf < SF_PER_TOKEN
    # Masked lanes are predicated off, but the address arithmetic still runs.
    # In the swizzled layout, out-of-range cols land past the per-page stride;
    # in the linear layout, they spill ~(BLOCK_SF - SF_PER_TOKEN) bytes past
    # each page row. For the last physical page either case can fall outside
    # the sf_cache allocation. Pin masked lanes to col 0 so all computed
    # addresses stay in-bounds regardless of allocator slack.
    safe_offs_sf = tl.where(mask_sf, offs_sf, 0)
    sf_vals = tl.load(
        q_sf_ptr + token_idx * q_sf_s0 + safe_offs_sf * q_sf_s1,
        mask=mask_sf,
        other=0,
    )
    if USE_SWIZZLED_SF:
        sf_offsets = _fp4_mla_swizzled_sf_offset(page_pos, safe_offs_sf, SF_PER_TOKEN)
        sf_dst = physical_page * sf_s0
        tl.store(sf_cache_ptr + sf_dst + sf_offsets, sf_vals, mask=mask_sf)
    else:
        sf_dst = physical_page * sf_s0 + page_pos * sf_s2
        tl.store(sf_cache_ptr + sf_dst + safe_offs_sf * sf_s4, sf_vals, mask=mask_sf)


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
def _fp4_e2m1_quantize(x):
    abs_x = tl.abs(x)
    magnitude = tl.where(
        abs_x < 0.25,
        0,
        tl.where(
            abs_x < 0.75,
            1,
            tl.where(
                abs_x < 1.25,
                2,
                tl.where(
                    abs_x < 1.75,
                    3,
                    tl.where(abs_x < 2.5, 4, tl.where(abs_x < 3.5, 5, tl.where(abs_x < 5.0, 6, 7))),
                ),
            ),
        ),
    )
    sign = tl.where(x < 0.0, 8, 0)
    return (magnitude | sign).to(tl.uint8)


@triton.jit
def _fp4_mla_v_scale_store_context_tokens_kernel(
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    latent_cache_ptr,
    global_scale_ptr,
    batch_indices_ptr,
    positions_ptr,
    paged_kv_indices_ptr,
    paged_kv_indptr_ptr,
    page_ids_len,
    indptr_len,
    metadata_num_tokens,
    num_pages,
    num_layers,
    token_offset,
    num_tokens,
    local_layer,
    page_size,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    lc_s0,
    lc_s1,
    vsf_s0,
    vsf_s1,
    HEAD_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
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
    amax_per_token = tl.maximum(
        tl.max(tl.abs(even_values), axis=1),
        tl.max(tl.abs(odd_values), axis=1),
    )
    tile_amax = tl.max(amax_per_token, axis=0)
    global_scale = tl.load(global_scale_ptr)
    # K consumes scales as [token, dim-block], while V consumes scales as
    # [dim, token-block].  Only the compressed-KV prefix has both views, so
    # tail K-only dims keep K's per-token scale.
    shared_tile = dim_block * FP4_BLOCK < V_HEAD_D
    tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)
    token_scale = tl.where(amax_per_token > 0.0, amax_per_token / 6.0, 1.0)
    local_scale = tl.where(shared_tile, tile_scale, token_scale)
    stored_scale = local_scale * global_scale
    v_stored_scale = tile_scale * global_scale

    low = _fp4_e2m1_quantize(even_values / local_scale[:, None])
    high = _fp4_e2m1_quantize(odd_values / local_scale[:, None])
    packed = low | (high << 4)

    packed_cols = dim_block * (FP4_BLOCK // 2) + byte_offsets
    page_positions = page_pos + token_offsets
    kv_base = physical_page * kv_s0
    tl.store(
        kv_cache_ptr + kv_base + page_positions[:, None] * kv_s2 + packed_cols[None, :] * kv_s4,
        packed,
        mask=valid_tokens[:, None] & mask_even_d[None, :],
    )

    k_sf_offsets = _fp4_mla_swizzled_sf_offset(page_positions, dim_block, SF_PER_TOKEN)
    tl.store(sf_cache_ptr + physical_page * sf_s0 + k_sf_offsets, stored_scale, mask=valid_tokens)

    token_scale_col = page_pos // HP_BLOCK
    sf_offsets = _fp4_mla_swizzled_sf_offset(safe_all_d, token_scale_col, SF_PER_PAGE)
    v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
        vsf_s0, tl.int64
    ) + physical_page * tl.cast(vsf_s1, tl.int64)
    tl.store(
        v_sf_ptr + v_sf_base + sf_offsets.to(tl.int64),
        v_stored_scale,
        mask=mask_all_d & (all_d < V_HEAD_D),
    )


@triton.jit
def _fp4_mla_v_scale_store_hp_tail_kernel(
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    hp_pool_ptr,
    global_scale_ptr,
    seq_slots_ptr,
    kv_lens_ptr,
    page_ids_ptr,
    paged_kv_indptr_ptr,
    page_ids_len,
    indptr_len,
    num_pages,
    num_seq_slots,
    num_layers,
    local_layer,
    page_size,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    pool_s0,
    pool_s1,
    vsf_s0,
    vsf_s1,
    HEAD_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    dim_block = tl.program_id(1)
    if (local_layer < 0) | (local_layer >= num_layers):
        return
    if seq_idx + 1 >= indptr_len:
        return

    kv_len = tl.load(kv_lens_ptr + seq_idx)
    remainder = kv_len % HP_BLOCK
    if kv_len == 0:
        return

    tail_count = tl.where(remainder == 0, HP_BLOCK, remainder)
    block_base_pos = kv_len - tail_count
    page_idx = block_base_pos // page_size
    page_pos = block_base_pos - page_idx * page_size
    page_start = tl.load(paged_kv_indptr_ptr + seq_idx).to(tl.int64)
    page_end = tl.load(paged_kv_indptr_ptr + seq_idx + 1).to(tl.int64)
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
    seq_slot = tl.load(seq_slots_ptr + seq_idx).to(tl.int64)
    if (seq_slot < 0) | (seq_slot >= num_seq_slots):
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

    hp_slots = (block_base_pos + token_offsets) % HP_BLOCK
    valid_tokens = token_offsets < tail_count
    even_values = tl.load(
        hp_pool_ptr
        + seq_slot * pool_s0
        + local_layer * pool_s1
        + hp_slots[:, None] * HEAD_D
        + safe_even_d[None, :],
        mask=valid_tokens[:, None] & mask_even_d[None, :],
        other=0.0,
    ).to(tl.float32)
    odd_values = tl.load(
        hp_pool_ptr
        + seq_slot * pool_s0
        + local_layer * pool_s1
        + hp_slots[:, None] * HEAD_D
        + safe_odd_d[None, :],
        mask=valid_tokens[:, None] & mask_odd_d[None, :],
        other=0.0,
    ).to(tl.float32)
    amax_per_token = tl.maximum(
        tl.max(tl.abs(even_values), axis=1),
        tl.max(tl.abs(odd_values), axis=1),
    )
    tile_amax = tl.max(amax_per_token, axis=0)
    global_scale = tl.load(global_scale_ptr)
    # K consumes scales as [token, dim-block], while V consumes scales as
    # [dim, token-block].  Only the compressed-KV prefix has both views, so
    # tail K-only dims keep K's per-token scale.
    shared_tile = dim_block * FP4_BLOCK < V_HEAD_D
    tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)
    token_scale = tl.where(amax_per_token > 0.0, amax_per_token / 6.0, 1.0)
    local_scale = tl.where(shared_tile, tile_scale, token_scale)
    stored_scale = local_scale * global_scale
    v_stored_scale = tile_scale * global_scale

    low = _fp4_e2m1_quantize(even_values / local_scale[:, None])
    high = _fp4_e2m1_quantize(odd_values / local_scale[:, None])
    packed = low | (high << 4)

    packed_cols = dim_block * (FP4_BLOCK // 2) + byte_offsets
    page_positions = page_pos + token_offsets
    kv_base = physical_page * kv_s0
    tl.store(
        kv_cache_ptr + kv_base + page_positions[:, None] * kv_s2 + packed_cols[None, :] * kv_s4,
        packed,
        mask=valid_tokens[:, None] & mask_even_d[None, :],
    )

    k_sf_offsets = _fp4_mla_swizzled_sf_offset(page_positions, dim_block, SF_PER_TOKEN)
    tl.store(sf_cache_ptr + physical_page * sf_s0 + k_sf_offsets, stored_scale, mask=valid_tokens)

    token_scale_col = page_pos // HP_BLOCK
    sf_offsets = _fp4_mla_swizzled_sf_offset(safe_all_d, token_scale_col, SF_PER_PAGE)
    v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
        vsf_s0, tl.int64
    ) + physical_page * tl.cast(vsf_s1, tl.int64)
    tl.store(
        v_sf_ptr + v_sf_base + sf_offsets.to(tl.int64),
        v_stored_scale,
        mask=mask_all_d & (all_d < V_HEAD_D),
    )


@triton.jit
def _fp4_mla_v_scale_store_generation_tiles_kernel(
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    hp_pool_ptr,
    latent_cache_ptr,
    global_scale_ptr,
    seq_slots_ptr,
    kv_lens_ptr,
    prompt_lens_ptr,
    page_ids_ptr,
    paged_kv_indptr_ptr,
    page_ids_len,
    indptr_len,
    num_seq_slots,
    num_pages,
    num_layers,
    local_layer,
    page_size,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    pool_s0,
    pool_s1,
    lc_s0,
    lc_s1,
    vsf_s0,
    vsf_s1,
    HEAD_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    dim_block = tl.program_id(2)
    if (local_layer < 0) | (local_layer >= num_layers):
        return
    if seq_idx + 1 >= indptr_len:
        return

    kv_len = tl.load(kv_lens_ptr + seq_idx)
    gen_len = tl.load(prompt_lens_ptr + seq_idx)
    if gen_len <= 0:
        return
    first_new_pos = kv_len - gen_len
    first_tile_pos = (first_new_pos // HP_BLOCK) * HP_BLOCK
    block_base_pos = first_tile_pos + tile_idx * HP_BLOCK
    if block_base_pos >= kv_len:
        return
    if block_base_pos + HP_BLOCK <= first_new_pos:
        return

    page_idx = block_base_pos // page_size
    page_pos = block_base_pos - page_idx * page_size
    page_start = tl.load(paged_kv_indptr_ptr + seq_idx).to(tl.int64)
    page_end = tl.load(paged_kv_indptr_ptr + seq_idx + 1).to(tl.int64)
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
    seq_slot = tl.load(seq_slots_ptr + seq_idx).to(tl.int64)
    if (seq_slot < 0) | (seq_slot >= num_seq_slots):
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

    abs_positions = block_base_pos + token_offsets
    valid_tokens = abs_positions < kv_len
    from_latent = abs_positions >= first_new_pos
    hp_slots = abs_positions % HP_BLOCK
    new_token_offsets = abs_positions - first_new_pos
    # Linear MTP uses a uniform generation length, so each sequence occupies a
    # contiguous gen_len slice in latent_cache.
    latent_tokens = seq_idx * gen_len + new_token_offsets
    safe_latent_tokens = tl.where(valid_tokens & from_latent, latent_tokens, 0).to(tl.int64)

    hp_even = tl.load(
        hp_pool_ptr
        + seq_slot * pool_s0
        + local_layer * pool_s1
        + hp_slots[:, None] * HEAD_D
        + safe_even_d[None, :],
        mask=valid_tokens[:, None] & (~from_latent)[:, None] & mask_even_d[None, :],
        other=0.0,
    ).to(tl.float32)
    hp_odd = tl.load(
        hp_pool_ptr
        + seq_slot * pool_s0
        + local_layer * pool_s1
        + hp_slots[:, None] * HEAD_D
        + safe_odd_d[None, :],
        mask=valid_tokens[:, None] & (~from_latent)[:, None] & mask_odd_d[None, :],
        other=0.0,
    ).to(tl.float32)
    latent_even = tl.load(
        latent_cache_ptr + safe_latent_tokens[:, None] * lc_s0 + safe_even_d[None, :] * lc_s1,
        mask=valid_tokens[:, None] & from_latent[:, None] & mask_even_d[None, :],
        other=0.0,
    ).to(tl.float32)
    latent_odd = tl.load(
        latent_cache_ptr + safe_latent_tokens[:, None] * lc_s0 + safe_odd_d[None, :] * lc_s1,
        mask=valid_tokens[:, None] & from_latent[:, None] & mask_odd_d[None, :],
        other=0.0,
    ).to(tl.float32)
    even_values = hp_even + latent_even
    odd_values = hp_odd + latent_odd

    amax_per_token = tl.maximum(
        tl.max(tl.abs(even_values), axis=1),
        tl.max(tl.abs(odd_values), axis=1),
    )
    tile_amax = tl.max(amax_per_token, axis=0)
    global_scale = tl.load(global_scale_ptr)
    shared_tile = dim_block * FP4_BLOCK < V_HEAD_D
    tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)
    token_scale = tl.where(amax_per_token > 0.0, amax_per_token / 6.0, 1.0)
    local_scale = tl.where(shared_tile, tile_scale, token_scale)
    stored_scale = local_scale * global_scale
    v_stored_scale = tile_scale * global_scale

    low = _fp4_e2m1_quantize(even_values / local_scale[:, None])
    high = _fp4_e2m1_quantize(odd_values / local_scale[:, None])
    packed = low | (high << 4)

    packed_cols = dim_block * (FP4_BLOCK // 2) + byte_offsets
    page_positions = page_pos + token_offsets
    kv_base = physical_page * kv_s0
    tl.store(
        kv_cache_ptr + kv_base + page_positions[:, None] * kv_s2 + packed_cols[None, :] * kv_s4,
        packed,
        mask=valid_tokens[:, None] & mask_even_d[None, :],
    )

    k_sf_offsets = _fp4_mla_swizzled_sf_offset(page_positions, dim_block, SF_PER_TOKEN)
    tl.store(sf_cache_ptr + physical_page * sf_s0 + k_sf_offsets, stored_scale, mask=valid_tokens)

    token_scale_col = page_pos // HP_BLOCK
    sf_offsets = _fp4_mla_swizzled_sf_offset(safe_all_d, token_scale_col, SF_PER_PAGE)
    v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
        vsf_s0, tl.int64
    ) + physical_page * tl.cast(vsf_s1, tl.int64)
    tl.store(
        v_sf_ptr + v_sf_base + sf_offsets.to(tl.int64),
        v_stored_scale,
        mask=mask_all_d & (all_d < V_HEAD_D),
    )


@triton.jit
def _fp4_mla_load_values(
    kv_cache_ptr,
    sf_cache_ptr,
    physical_page,
    page_pos,
    offs_d,
    mask_d,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    sf_s2,
    sf_s4,
    D: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    USE_SWIZZLED_SF: tl.constexpr,
):
    packed_offsets = offs_d // 2
    packed = tl.load(
        kv_cache_ptr + physical_page * kv_s0 + page_pos * kv_s2 + packed_offsets * kv_s4,
        mask=mask_d,
        other=0,
    )
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibble = tl.where((offs_d & 1) == 0, low, high)

    scale_offsets = offs_d // FP4_BLOCK
    # See _fp4_mla_scatter_kernel for the rationale: pin masked lanes to col 0
    # so the address arithmetic stays inside the per-page stride for the last
    # physical page, regardless of which SF layout is in use.
    safe_scale_offsets = tl.where(mask_d, scale_offsets, 0)
    if USE_SWIZZLED_SF:
        swizzled_sf_offsets = _fp4_mla_swizzled_sf_offset(
            page_pos, safe_scale_offsets, SF_PER_TOKEN
        )
        scale = tl.load(
            sf_cache_ptr + physical_page * sf_s0 + swizzled_sf_offsets,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
    else:
        scale = tl.load(
            sf_cache_ptr + physical_page * sf_s0 + page_pos * sf_s2 + safe_scale_offsets * sf_s4,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
    return _fp4_e2m1_to_f32(nibble) * scale


@triton.jit
def _fp4_mla_dequant_kernel(
    out_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    page_ids_len,
    num_pages,
    kv_s0,
    kv_s1,
    kv_s2,
    kv_s3,
    kv_s4,
    sf_s0,
    sf_s1,
    sf_s2,
    sf_s3,
    sf_s4,
    out_s0,
    out_s1,
    out_s2,
    D: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_SWIZZLED_SF: tl.constexpr,
):
    compact_page = tl.program_id(0).to(tl.int64)
    page_pos = tl.program_id(1).to(tl.int64)
    valid_compact_page = (compact_page >= 0) & (compact_page < page_ids_len)
    safe_compact_page = tl.where(valid_compact_page, compact_page, 0)
    physical_page = tl.load(
        src_page_ids_ptr + safe_compact_page,
        mask=valid_compact_page,
        other=-1,
    ).to(tl.int64)
    valid_physical_page = valid_compact_page & (physical_page >= 0) & (physical_page < num_pages)
    safe_physical_page = tl.where(valid_physical_page, physical_page, 0)
    global_scale = tl.load(global_scale_ptr)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    safe_offs_d = tl.where(mask_d, offs_d, 0)
    value = _fp4_mla_load_values(
        kv_cache_ptr,
        sf_cache_ptr,
        safe_physical_page,
        page_pos,
        safe_offs_d,
        mask_d & valid_physical_page,
        kv_s0,
        kv_s2,
        kv_s4,
        sf_s0,
        sf_s2,
        sf_s4,
        D,
        FP4_BLOCK,
        D // FP4_BLOCK,
        USE_SWIZZLED_SF,
    )

    tl.store(
        out_ptr + compact_page * out_s0 + page_pos * out_s1 + safe_offs_d * out_s2,
        value / global_scale,
        mask=mask_d,
    )


@triton.jit
def _fp4_mla_qk_scores_tile(
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    src_page_ids_ptr,
    compact_page,
    q_row_base,
    head_offsets,
    token_offsets,
    q_fp4_s0,
    q_fp4_s1,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    page_ids_len,
    num_pages,
    Q_HEAD_D: tl.constexpr,
    K_HEAD_D: tl.constexpr,
    Q_RESIDUAL_D: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    Q_SF_PER_TOKEN: tl.constexpr,
    K_SF_PER_TOKEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    valid_compact_page = (compact_page >= 0) & (compact_page < page_ids_len)
    safe_compact_page = tl.where(valid_compact_page, compact_page, 0)
    physical_page = tl.load(
        src_page_ids_ptr + safe_compact_page, mask=valid_compact_page, other=-1
    ).to(tl.int64)
    valid_physical_page = valid_compact_page & (physical_page >= 0) & (physical_page < num_pages)
    safe_physical_page = tl.where(valid_physical_page, physical_page, 0)
    q_rows = q_row_base + head_offsets
    mask_h = head_offsets < NUM_HEADS
    safe_q_rows = tl.where(mask_h, q_rows, q_row_base)
    scores = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)

    packed_k_offsets = tl.arange(0, BLOCK_K // 2)
    scale_offsets = tl.arange(0, BLOCK_K // FP4_BLOCK)
    # fp4_quantize_with_residual lays out the last Q groups as
    # [main_0, residual_0, main_1, residual_1, ...].  The KV cache stores
    # each tail K group once, so QK maps both logical Q tail groups to the
    # same physical K group.
    residual_groups = Q_RESIDUAL_D // FP4_BLOCK
    non_residual_groups = K_HEAD_D // FP4_BLOCK - residual_groups
    for q_start in tl.range(0, Q_HEAD_D, BLOCK_K):
        q_elem_offsets = q_start + packed_k_offsets * 2
        q_group_offsets = q_elem_offsets // FP4_BLOCK
        k_group_offsets = tl.where(
            q_group_offsets < non_residual_groups,
            q_group_offsets,
            non_residual_groups + (q_group_offsets - non_residual_groups) // 2,
        )
        byte_offsets_in_group = (q_elem_offsets % FP4_BLOCK) // 2
        packed_q_cols = q_start // 2 + packed_k_offsets
        packed_k_cols = k_group_offsets * (FP4_BLOCK // 2) + byte_offsets_in_group
        mask_k = q_elem_offsets < Q_HEAD_D
        safe_packed_q_cols = tl.where(mask_k, packed_q_cols, 0)
        safe_packed_k_cols = tl.where(mask_k, packed_k_cols, 0)
        q_vals = tl.load(
            q_fp4_ptr + safe_q_rows[:, None] * q_fp4_s0 + safe_packed_q_cols[None, :] * q_fp4_s1,
            mask=mask_h[:, None] & mask_k[None, :],
            other=0,
        )
        k_vals = tl.load(
            kv_cache_ptr
            + safe_physical_page * kv_s0
            + token_offsets[:, None].to(tl.int64) * kv_s2
            + safe_packed_k_cols[None, :] * kv_s4,
            mask=valid_physical_page & mask_k[None, :],
            other=0,
        )

        q_sf_cols = q_start // FP4_BLOCK + scale_offsets
        k_sf_cols = tl.where(
            q_sf_cols < non_residual_groups,
            q_sf_cols,
            non_residual_groups + (q_sf_cols - non_residual_groups) // 2,
        )
        mask_sf = q_sf_cols < Q_SF_PER_TOKEN
        safe_q_sf_cols = tl.where(mask_sf, q_sf_cols, 0)
        safe_k_sf_cols = tl.where(mask_sf, k_sf_cols, 0)
        q_sf_offsets = _fp4_mla_swizzled_sf_offset(
            safe_q_rows[:, None], safe_q_sf_cols[None, :], Q_SF_PER_TOKEN
        )
        k_sf_offsets = _fp4_mla_swizzled_sf_offset(
            token_offsets[:, None], safe_k_sf_cols[None, :], K_SF_PER_TOKEN
        )
        q_scales = tl.load(q_sf_ptr + q_sf_offsets)
        k_scales = tl.load(sf_cache_ptr + safe_physical_page * sf_s0 + k_sf_offsets)
        scores = tl.dot_scaled(
            q_vals,
            q_scales,
            "e2m1",
            k_vals.T,
            k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
        )

    return scores


@triton.jit
def _fp4_mla_attention_stats_kernel(
    max_ptr,
    denom_ptr,
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_ids_len,
    num_pages,
    q_fp4_s0,
    q_fp4_s1,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    stats_s0,
    sm_scale: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    Q_HEAD_D: tl.constexpr,
    K_HEAD_D: tl.constexpr,
    Q_RESIDUAL_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    Q_SF_PER_TOKEN: tl.constexpr,
    K_SF_PER_TOKEN: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < NUM_HEADS
    safe_offs_h = tl.where(mask_h, offs_h, 0)
    offs_t = tl.arange(0, BLOCK_T)
    q_row_base = query_idx * NUM_HEADS
    kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    kv_len = tl.maximum(kv_len, 0)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)

    max_score = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    denom = tl.zeros((BLOCK_H,), dtype=tl.float32)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)
    for page_rel in tl.range(0, MAX_PAGES):
        page_start = page_rel * PAGE_SIZE
        if page_start < kv_len:
            compact_page = page_table_start + page_rel
            scores = _fp4_mla_qk_scores_tile(
                q_fp4_ptr,
                q_sf_ptr,
                kv_cache_ptr,
                sf_cache_ptr,
                src_page_ids_ptr,
                compact_page,
                q_row_base,
                offs_h,
                offs_t,
                q_fp4_s0,
                q_fp4_s1,
                kv_s0,
                kv_s2,
                kv_s4,
                sf_s0,
                page_ids_len,
                num_pages,
                Q_HEAD_D,
                K_HEAD_D,
                Q_RESIDUAL_D,
                FP4_BLOCK,
                Q_SF_PER_TOKEN,
                K_SF_PER_TOKEN,
                BLOCK_H,
                BLOCK_T,
                BLOCK_K,
                NUM_HEADS,
            )
            valid_t = page_start + offs_t < kv_len
            scores = tl.where(mask_h[:, None] & valid_t[None, :], scores * qk_scale, -float("inf"))
            page_max = tl.max(scores, axis=1)
            new_max = tl.maximum(max_score, page_max)
            denom = denom * tl.exp(max_score - new_max) + tl.sum(
                tl.exp(scores - new_max[:, None]), axis=1
            )
            max_score = new_max

    tl.store(max_ptr + query_idx * stats_s0 + safe_offs_h, max_score, mask=mask_h)
    tl.store(denom_ptr + query_idx * stats_s0 + safe_offs_h, denom, mask=mask_h)


@triton.jit
def _fp4_mla_attention_prob_store_page_kernel(
    probs_ptr,
    max_ptr,
    denom_ptr,
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_rel,
    page_ids_len,
    num_pages,
    probs_s0,
    probs_s1,
    q_fp4_s0,
    q_fp4_s1,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    stats_s0,
    sm_scale: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    Q_HEAD_D: tl.constexpr,
    K_HEAD_D: tl.constexpr,
    Q_RESIDUAL_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    Q_SF_PER_TOKEN: tl.constexpr,
    K_SF_PER_TOKEN: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    kv_len = tl.maximum(kv_len, 0)
    page_start = page_rel * PAGE_SIZE
    if page_start >= kv_len:
        return

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < NUM_HEADS
    safe_offs_h = tl.where(mask_h, offs_h, 0)
    offs_t = tl.arange(0, PAGE_SIZE)
    valid_t = page_start + offs_t < kv_len
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    compact_page = page_table_start + page_rel
    if (compact_page < 0) | (compact_page >= page_ids_len):
        return
    q_row_base = query_idx * NUM_HEADS

    scores = _fp4_mla_qk_scores_tile(
        q_fp4_ptr,
        q_sf_ptr,
        kv_cache_ptr,
        sf_cache_ptr,
        src_page_ids_ptr,
        compact_page,
        q_row_base,
        offs_h,
        offs_t,
        q_fp4_s0,
        q_fp4_s1,
        kv_s0,
        kv_s2,
        kv_s4,
        sf_s0,
        page_ids_len,
        num_pages,
        Q_HEAD_D,
        K_HEAD_D,
        Q_RESIDUAL_D,
        FP4_BLOCK,
        Q_SF_PER_TOKEN,
        K_SF_PER_TOKEN,
        BLOCK_H,
        PAGE_SIZE,
        BLOCK_K,
        NUM_HEADS,
    )
    max_score = tl.load(max_ptr + query_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    denom = tl.load(denom_ptr + query_idx * stats_s0 + safe_offs_h, mask=mask_h, other=1.0)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)
    denom_valid = denom > 0.0
    safe_denom = tl.where(denom_valid, denom, 1.0)
    safe_max = tl.where(denom_valid, max_score, 0.0)
    scores = tl.where(mask_h[:, None] & valid_t[None, :], scores * qk_scale, -float("inf"))
    probs = tl.exp(scores - safe_max[:, None]) / safe_denom[:, None]
    probs = tl.where(mask_h[:, None] & valid_t[None, :] & denom_valid[:, None], probs, 0.0)

    prob_rows = query_idx * NUM_HEADS + offs_h
    safe_prob_rows = tl.where(mask_h, prob_rows, query_idx * NUM_HEADS)
    tl.store(
        probs_ptr + safe_prob_rows[:, None] * probs_s0 + offs_t[None, :] * probs_s1,
        probs,
        mask=mask_h[:, None],
    )


@triton.jit
def _fp4_mla_attention_prob_pack_page_kernel(
    p_fp4_ptr,
    p_sf_ptr,
    probs_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_rel,
    page_ids_len,
    p_s0,
    p_s1,
    probs_s0,
    probs_s1,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    query_idx = tl.program_id(0)
    token_group = tl.program_id(1)
    head_block = tl.program_id(2)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    kv_len = tl.maximum(kv_len, 0)
    page_start = page_rel * PAGE_SIZE
    if page_start >= kv_len:
        return

    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    compact_page = page_table_start + page_rel
    if (compact_page < 0) | (compact_page >= page_ids_len):
        return

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < NUM_HEADS
    byte_offsets = tl.arange(0, FP4_BLOCK // 2)
    token_base = token_group * FP4_BLOCK
    even_t = token_base + byte_offsets * 2
    odd_t = even_t + 1
    valid_even = page_start + even_t < kv_len
    valid_odd = page_start + odd_t < kv_len

    prob_rows = query_idx * NUM_HEADS + offs_h
    safe_prob_rows = tl.where(mask_h, prob_rows, query_idx * NUM_HEADS)
    even_probs = tl.load(
        probs_ptr + safe_prob_rows[:, None] * probs_s0 + even_t[None, :] * probs_s1,
        mask=mask_h[:, None] & valid_even[None, :],
        other=0.0,
    )
    odd_probs = tl.load(
        probs_ptr + safe_prob_rows[:, None] * probs_s0 + odd_t[None, :] * probs_s1,
        mask=mask_h[:, None] & valid_odd[None, :],
        other=0.0,
    )
    amax = tl.maximum(tl.max(tl.abs(even_probs), axis=1), tl.max(tl.abs(odd_probs), axis=1))
    local_scale = tl.where(amax > 0.0, amax / 6.0, 1.0)
    stored_scale = tl.where(amax > 0.0, tl.minimum(local_scale * P_GLOBAL_SCALE, 448.0), 1.0)

    p_page = query_idx * MAX_PAGES + page_rel
    p_rows = p_page * NUM_HEADS + offs_h
    safe_p_rows = tl.where(mask_h, p_rows, p_page * NUM_HEADS)
    sf_offsets = _fp4_mla_swizzled_sf_offset(safe_p_rows, token_group, SF_PER_PAGE)
    tl.store(p_sf_ptr + sf_offsets, stored_scale, mask=mask_h)

    even_quant = _fp4_e2m1_quantize(even_probs / local_scale[:, None])
    odd_quant = _fp4_e2m1_quantize(odd_probs / local_scale[:, None])
    packed = even_quant | (odd_quant << 4)
    byte_cols = token_group * (FP4_BLOCK // 2) + byte_offsets
    tl.store(
        p_fp4_ptr + safe_p_rows[:, None] * p_s0 + byte_cols[None, :] * p_s1,
        packed,
        mask=mask_h[:, None],
    )


@triton.jit
def _fp4_mla_attention_pv_kernel(
    out_ptr,
    p_fp4_ptr,
    p_sf_ptr,
    kv_cache_ptr,
    v_sf_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_ids_len,
    num_pages,
    out_s0,
    out_s1,
    out_s2,
    p_s0,
    p_s1,
    kv_s0,
    kv_s2,
    kv_s4,
    vsf_s0,
    NUM_HEADS: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    dim_block = tl.program_id(2)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_h = offs_h < NUM_HEADS
    mask_v = offs_v < V_HEAD_D
    safe_offs_h = tl.where(mask_h, offs_h, 0)
    safe_offs_v = tl.where(mask_v, offs_v, 0)
    packed_t = tl.arange(0, PAGE_SIZE // 2)
    scale_cols = tl.arange(0, PAGE_SIZE // FP4_BLOCK)
    even_t = packed_t * 2
    odd_t = even_t + 1
    v_packed_offsets = safe_offs_v // 2
    v_use_high_nibble = (safe_offs_v & 1) != 0

    kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    kv_len = tl.maximum(kv_len, 0)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    global_scale = tl.load(global_scale_ptr)
    acc = tl.zeros((BLOCK_H, BLOCK_V), dtype=tl.float32)
    for page_rel in tl.range(0, MAX_PAGES):
        page_start = page_rel * PAGE_SIZE
        if page_start < kv_len:
            compact_page = page_table_start + page_rel
            valid_compact_page = (compact_page >= 0) & (compact_page < page_ids_len)
            safe_compact_page = tl.where(valid_compact_page, compact_page, 0)
            physical_page = tl.load(
                src_page_ids_ptr + safe_compact_page, mask=valid_compact_page, other=-1
            ).to(tl.int64)
            valid_physical_page = (
                valid_compact_page & (physical_page >= 0) & (physical_page < num_pages)
            )
            safe_physical_page = tl.where(valid_physical_page, physical_page, 0)

            p_page = query_idx * MAX_PAGES + page_rel
            p_rows = p_page * NUM_HEADS + offs_h
            safe_p_rows = tl.where(mask_h, p_rows, p_page * NUM_HEADS)
            p_vals = tl.load(
                p_fp4_ptr + safe_p_rows[:, None] * p_s0 + packed_t[None, :] * p_s1,
                mask=valid_compact_page & mask_h[:, None],
                other=0,
            )
            p_sf_offsets = _fp4_mla_swizzled_sf_offset(
                safe_p_rows[:, None], scale_cols[None, :], SF_PER_PAGE
            )
            p_scales = tl.load(p_sf_ptr + p_sf_offsets)

            valid_even_t = page_start + even_t < kv_len
            valid_odd_t = page_start + odd_t < kv_len
            even_packed = tl.load(
                kv_cache_ptr
                + safe_physical_page * kv_s0
                + even_t[None, :].to(tl.int64) * kv_s2
                + v_packed_offsets[:, None] * kv_s4,
                mask=(valid_physical_page & mask_v[:, None] & valid_even_t[None, :]),
                other=0,
            )
            odd_packed = tl.load(
                kv_cache_ptr
                + safe_physical_page * kv_s0
                + odd_t[None, :].to(tl.int64) * kv_s2
                + v_packed_offsets[:, None] * kv_s4,
                mask=(valid_physical_page & mask_v[:, None] & valid_odd_t[None, :]),
                other=0,
            )
            even_low = even_packed & 0x0F
            even_high = (even_packed >> 4) & 0x0F
            odd_low = odd_packed & 0x0F
            odd_high = (odd_packed >> 4) & 0x0F
            even_nibble = tl.where(v_use_high_nibble[:, None], even_high, even_low)
            odd_nibble = tl.where(v_use_high_nibble[:, None], odd_high, odd_low)
            v_vals = even_nibble | (odd_nibble << 4)
            v_sf_offsets = _fp4_mla_swizzled_sf_offset(
                safe_offs_v[:, None], scale_cols[None, :], SF_PER_PAGE
            )
            v_scales = tl.load(v_sf_ptr + safe_physical_page * vsf_s0 + v_sf_offsets)
            acc = tl.dot_scaled(
                p_vals,
                p_scales,
                "e2m1",
                v_vals.T,
                v_scales,
                "e2m1",
                acc=acc,
                fast_math=True,
            )

    tl.store(
        out_ptr
        + query_idx * out_s0
        + safe_offs_h[:, None] * out_s1
        + safe_offs_v[None, :] * out_s2,
        acc / (global_scale * P_GLOBAL_SCALE),
        mask=mask_h[:, None] & mask_v[None, :],
    )


@triton.jit
def _fp4_mla_overlay_hp_tail_kernel(
    out_ptr,
    pool_ptr,
    seq_slots_ptr,
    kv_lens_ptr,
    paged_kv_indptr_decode_ptr,
    num_seq_slots,
    num_layers,
    num_pages,
    layer_idx,
    page_size,
    out_s0,
    out_s1,
    out_s2,
    pool_stride_seq,
    pool_stride_layer,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HP_BLOCK: tl.constexpr,
):
    gen_idx = tl.program_id(0)
    tail_idx = tl.program_id(1)
    if (layer_idx < 0) | (layer_idx >= num_layers):
        return

    kv_len = tl.load(kv_lens_ptr + gen_idx)
    tail_count = kv_len % HP_BLOCK
    if tail_idx >= tail_count:
        return

    abs_pos = (kv_len - tail_count + tail_idx).to(tl.int64)
    page_size_i64 = tl.cast(page_size, tl.int64)
    rel_page = abs_pos // page_size_i64
    page_pos = abs_pos - rel_page * page_size_i64
    compact_page = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64) + rel_page
    if (compact_page < 0) | (compact_page >= num_pages):
        return
    hp_slot = abs_pos % HP_BLOCK
    seq_slot = tl.load(seq_slots_ptr + gen_idx).to(tl.int64)
    if (seq_slot < 0) | (seq_slot >= num_seq_slots):
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    safe_offs_d = tl.where(mask_d, offs_d, 0)
    src_base = seq_slot * pool_stride_seq + layer_idx * pool_stride_layer + hp_slot * D
    value = tl.load(pool_ptr + src_base + safe_offs_d, mask=mask_d, other=0.0)
    tl.store(
        out_ptr + compact_page * out_s0 + page_pos * out_s1 + safe_offs_d * out_s2,
        value,
        mask=mask_d,
    )
