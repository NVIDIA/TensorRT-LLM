# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the DSA indexer and sparse MLA: the K3.1 Rubin family copy.

Based on ``tensorrt_llm/_torch/attention/backends/sparse/dsa/kernels.py``, with changes the engine file does not
carry: the Top-K remap returns -1 for a negative token index or an unmapped page, and
``fused_dsa_helix_decode_metadata`` and ``helix_zero_or_empty_rows`` are added. The engine file's other kernels are
not copied. No torch op is registered; the family ops call these launchers directly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

########################################################
# Index gather kernel
########################################################


@triton.jit
def _convert_req_index_to_global_index_kernel_with_stride_factor(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_kv_heads, num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_kv_heads, num_tokens, kv_factor, NUM_TOPK_TOKENS]
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    stride_factor: tl.constexpr,  # elements per physical page in pool
    layer_id: tl.constexpr,  # for multi-layer KV cache
    num_kv_heads: tl.constexpr,
    kv_factor: tl.constexpr,
    bt_stride0,  # stride for batch dim
    bt_stride1,  # stride for page dim
    ti_stride0,
    ti_stride1,
    ti_stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
):
    """
    Convert request-local token indices to global KV cache pool indices.

    KV cache pool layout: [max_num_pages, num_layers, kv_factor, num_kv_heads, tokens_per_block, head_dim]
    block_table: [num_requests, max_pages_per_req]
      - stores memPoolBlockIdx (physical page index, first dim of pool)
      - same block_table for K and V (K/V share the same physical page)

    stride_factor = num_layers * kv_factor * num_kv_heads * BLOCK_SIZE
      (elements per physical page, excluding head_dim)

    Global index:
      memPoolBlockIdx * stride_factor
        + layer_id * kv_factor * num_kv_heads * BLOCK_SIZE
        + kv_factor_idx * num_kv_heads * BLOCK_SIZE
        + kv_head_idx * BLOCK_SIZE
        + token_in_page
    """
    token_id = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    tile_id = tl.program_id(2)

    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req = tl.load(req_id_ptr + token_id)

    # Load local token indices: token_indices[kv_head_idx, token_id, indice_id]
    ti_ptr = (
        token_indices_ptr
        + kv_head_idx * ti_stride0
        + token_id * ti_stride1
        + indice_id * ti_stride2
    )
    tok = tl.load(ti_ptr)

    is_invalid_tok = tok < 0
    page_idx = tok // BLOCK_SIZE
    token_in_page = tok % BLOCK_SIZE
    # Like the C++ remap: no block-table read for a negative index, -1 past the table or on a padding entry.
    valid_page = (page_idx < max_num_blocks_per_req) & (~is_invalid_tok)

    # block_table[req, page_idx] → memPoolBlockIdx (physical page index)
    bt_ptr = block_table_ptr + req * bt_stride0 + page_idx * bt_stride1
    mem_pool_idx = tl.load(bt_ptr, mask=valid_page, other=-1)
    valid_page = valid_page & (mem_pool_idx >= 0)

    # Base offset within physical page (invariant across kv_factor loop)
    base_off = (
        layer_id * kv_factor * num_kv_heads * BLOCK_SIZE + kv_head_idx * BLOCK_SIZE + token_in_page
    )

    for kv_factor_idx in tl.static_range(kv_factor):
        # Within-page offset: layer → kv_factor → kv_head → token
        inpage_off = base_off + kv_factor_idx * num_kv_heads * BLOCK_SIZE

        global_idx = mem_pool_idx * stride_factor + inpage_off

        out_val = tl.where(is_invalid_tok | (~valid_page), -1, global_idx)

        out_ptr_ij = (
            out_ptr
            + kv_head_idx * out_stride0
            + token_id * out_stride1
            + kv_factor_idx * out_stride2
            + indice_id * out_stride3
        )
        tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, kv_factor, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_kv_heads, num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    stride_factor: int = None,  # elements per block in pool
    layer_id: int = 0,  # for multi-layer KV cache
    num_kv_heads: int = 1,
    kv_factor: int = 1,
):
    """
    Convert request-local token indices to global KV cache pool indices.

    Accepts both 2D and 3D token_indices:
      - 2D [num_tokens, topk]: MLA path (num_kv_heads=1 implicit)
      - 3D [num_kv_heads, num_tokens, topk]: MQA/GQA path

    Output shape: [num_kv_heads, num_tokens, kv_factor, topk]

    Args:
        block_table: [num_requests, max_pages] — physical page indices.
        stride_factor: Elements per physical page
                       (num_layers * kv_factor * num_kv_heads * BLOCK_SIZE).
        num_kv_heads: Number of KV heads.
        kv_factor: 2 for MQA/GQA, 1 for MLA.
    """
    if stride_factor is None:
        stride_factor = kv_factor * num_kv_heads * BLOCK_SIZE
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.ndim in (2, 3), (
        f"Expected 2D [num_tokens, topk] or 3D [num_kv_heads, num_tokens, topk], got {token_indices.ndim}D"
    )
    assert block_table.ndim == 2, f"Expected 2D [batch, max_pages], got {block_table.ndim}D"
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty(
        (num_kv_heads, num_tokens, kv_factor, NUM_TOPK_TOKENS),
        dtype=torch.int32,
        device=token_indices.device,
    )

    bt_stride0, bt_stride1 = block_table_c.stride()
    # 2D input: ti_stride0=0 (kv_head_idx is always 0, term vanishes)
    if token_indices_c.ndim == 2:
        ti_stride0 = 0
        ti_stride1, ti_stride2 = token_indices_c.stride()
    else:
        ti_stride0, ti_stride1, ti_stride2 = token_indices_c.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

    grid = (num_tokens, num_kv_heads, tiles_per_row)

    _convert_req_index_to_global_index_kernel_with_stride_factor[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        stride_factor,
        layer_id,
        num_kv_heads,
        kv_factor,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        ti_stride2,
        out_stride0,
        out_stride1,
        out_stride2,
        out_stride3,
    )
    return out


########################################################
# Fused Helix decode metadata kernel
########################################################


@triton.jit(do_not_specialize=["num_seqs", "num_tokens", "width", "max_blocks"])
def _fused_dsa_helix_decode_metadata_kernel(
    seq_lens_ptr,  # int32 [num_seqs]
    kv_lens_ptr,  # int32 [num_seqs], rank-local
    block_offsets_ptr,  # int32 [rows, max_blocks], rank-local pages
    helix_local_slots_ptr,  # int32 [num_tokens], -1 where another rank owns the token
    helix_kv_bounds_ptr,  # int32 [num_tokens]
    req_idx_per_token_ptr,  # int32 [num_tokens] (out)
    slot_fp8_ptr,  # int64 [num_tokens] (out)
    slot_scale_ptr,  # int64 [num_tokens] (out)
    gen_kv_indptr_ptr,  # int64 [num_seqs + 1] (out, entries 1..num_seqs)
    gen_cached_indptr_ptr,  # int64 [num_seqs + 1] (out, entries 1..num_seqs)
    kv_lens_2d_ptr,  # int32 [num_seqs, >= width] (out)
    kv_lens_expanded_ptr,  # int32 [num_tokens] (out)
    block_offsets_stride_0,
    block_offsets_stride_1,
    kv_lens_2d_stride_0,
    num_seqs,
    num_tokens,
    width,
    max_blocks,
    tokens_per_block: tl.constexpr,
    data_bytes_per_token: tl.constexpr,
    scale_size: tl.constexpr,
    block_stride: tl.constexpr,
    scale_base_offset: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < num_seqs
    seq = tl.load(seq_lens_ptr + offs_s, mask=mask_s, other=0)

    # ----- program 0: per-sequence outputs -------------------------------------------------
    if pid == 0:
        kv = tl.load(kv_lens_ptr + offs_s, mask=mask_s, other=0)
        tl.store(gen_kv_indptr_ptr + 1 + offs_s, tl.cumsum(kv.to(tl.int64), 0), mask=mask_s)
        tl.store(
            gen_cached_indptr_ptr + 1 + offs_s, tl.cumsum((kv - seq).to(tl.int64), 0), mask=mask_s
        )
        offs_w = tl.arange(0, BLOCK_W)
        kv_2d = kv[:, None] + tl.zeros((BLOCK_S, BLOCK_W), dtype=tl.int32)
        tl.store(
            kv_lens_2d_ptr + offs_s[:, None] * kv_lens_2d_stride_0 + offs_w[None, :],
            kv_2d,
            mask=mask_s[:, None] & (offs_w[None, :] < width),
        )
        return

    # ----- programs 1..: BLOCK_T tokens each ------------------------------------------------
    t = (pid - 1) * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t < num_tokens
    # searchsorted(cumsum(seq_lens), t, right=True): the number of sequence ends <= t.
    ends = tl.cumsum(seq, 0)
    before = (ends[None, :] <= t[:, None]) & mask_s[None, :]
    req = tl.sum(before.to(tl.int32), axis=1)
    tl.store(req_idx_per_token_ptr + t, req, mask=mask_t)

    # The token's rank-local slot, or no write (-1) where another CP rank owns it.
    local = tl.load(helix_local_slots_ptr + t, mask=mask_t, other=0).to(tl.int64)
    owned = local >= 0
    pos = tl.maximum(local, 0)
    blk = tl.minimum(pos // tokens_per_block, max_blocks - 1)
    in_blk = pos % tokens_per_block
    bid = tl.load(
        block_offsets_ptr
        + req.to(tl.int64) * block_offsets_stride_0
        + blk * block_offsets_stride_1,
        mask=mask_t,
        other=0,
    ).to(tl.int64)
    fp8 = bid * block_stride + in_blk * data_bytes_per_token
    scale = bid * block_stride + scale_base_offset + in_blk * scale_size
    tl.store(slot_fp8_ptr + t, tl.where(owned, fp8, -1), mask=mask_t)
    tl.store(slot_scale_ptr + t, tl.where(owned, scale, -1), mask=mask_t)

    # Each verify row scores at least one key (the indexer drops the rows of empty bounds).
    bounds = tl.load(helix_kv_bounds_ptr + t, mask=mask_t, other=1)
    tl.store(kv_lens_expanded_ptr + t, tl.maximum(bounds, 1), mask=mask_t)


def fused_dsa_helix_decode_metadata(
    seq_lens: torch.Tensor,  # int32 [num_seqs]
    kv_lens: torch.Tensor,  # int32 [num_seqs]
    block_offsets: torch.Tensor,  # int32 [rows >= num_seqs, max_blocks]
    helix_local_slots: torch.Tensor,  # int32 [>= num_tokens]
    helix_kv_bounds: torch.Tensor,  # int32 [>= num_tokens]
    req_idx_per_token: torch.Tensor,  # int32 [>= num_tokens] (out)
    slot_mapping_fp8: torch.Tensor,  # int64 [>= num_tokens] (out)
    slot_mapping_scale: torch.Tensor,  # int64 [>= num_tokens] (out)
    gen_kv_indptr: torch.Tensor,  # int64 [>= num_seqs + 1] (out)
    gen_cached_token_indptr: torch.Tensor,  # int64 [>= num_seqs + 1] (out)
    kv_lens_2d: torch.Tensor,  # int32 [>= num_seqs, >= width] (out)
    kv_lens_expanded: torch.Tensor,  # int32 [>= num_tokens] (out)
    num_tokens: int,
    width: int,
    tokens_per_block: int,
    index_head_dim: int,
    quant_block_size: int,
    data_bytes_per_token: int | None = None,
) -> None:
    """The DSA decode metadata of a Helix CP generation step in one Triton launch.

    Writes, for a step of ``num_seqs`` generation requests and ``num_tokens`` verify tokens
    (no context requests), exactly what the eager chain of ``on_update_kv_lens`` leaves:

    * ``req_idx_per_token[:num_tokens]``: ``searchsorted(cumsum(seq_lens), t, right=True)``;
    * ``slot_mapping_{fp8,scale}[:num_tokens]``: the token's indexer-K slot at its rank-local
      position ``helix_local_slots[t]`` in ``block_offsets[req]``, or -1 where another CP rank
      owns the token (the Helix ownership rewrite of the generic slots);
    * ``gen_kv_indptr[1:]`` / ``gen_cached_token_indptr[1:]``: the int64 prefix sums of
      ``kv_lens`` and ``kv_lens - seq_lens`` (entry 0 is not written, as in the eager chain);
    * ``kv_lens_2d[:num_seqs, :width]``: ``kv_lens`` broadcast over the verify width;
    * ``kv_lens_expanded[:num_tokens]``: ``max(helix_kv_bounds, 1)``.

    The DeepGEMM schedules are left to the caller. Integer arithmetic only, so the outputs are
    bitwise those of the eager chain. Graph-capturable (no allocation, no host sync).
    """
    num_seqs = seq_lens.shape[0]
    assert num_seqs > 0 and num_tokens > 0
    assert block_offsets.shape[0] >= num_seqs
    if data_bytes_per_token is None:
        data_bytes_per_token = index_head_dim
    scale_size = index_head_dim // quant_block_size * 4  # float32 = 4 bytes
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token
    block_s = triton.next_power_of_2(num_seqs)
    block_t = min(128, triton.next_power_of_2(num_tokens))
    block_w = triton.next_power_of_2(max(width, 1))
    grid = (1 + triton.cdiv(num_tokens, block_t),)
    _fused_dsa_helix_decode_metadata_kernel[grid](
        seq_lens,
        kv_lens,
        block_offsets,
        helix_local_slots,
        helix_kv_bounds,
        req_idx_per_token,
        slot_mapping_fp8,
        slot_mapping_scale,
        gen_kv_indptr,
        gen_cached_token_indptr,
        kv_lens_2d,
        kv_lens_expanded,
        block_offsets.stride(0),
        block_offsets.stride(1),
        kv_lens_2d.stride(0),
        num_seqs,
        num_tokens,
        width,
        block_offsets.shape[1],
        tokens_per_block,
        data_bytes_per_token,
        scale_size,
        block_stride,
        scale_base_offset,
        BLOCK_S=block_s,
        BLOCK_T=block_t,
        BLOCK_W=block_w,
    )


########################################################
# Helix combine mask of a token-sparse generation batch
########################################################


@triton.jit(do_not_specialize=["num_rows"])
def _helix_zero_or_empty_rows_kernel(
    bounds_ptr,  # int [num_rows]: per-token rank-local KV bound (helix_kv_bounds)
    row_lens_ptr,  # int [num_rows]: kept Top-K entries per row on this rank
    out_ptr,  # bool [num_rows]
    num_rows,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < num_rows
    bounds = tl.load(bounds_ptr + offs, mask=valid, other=1)
    row_lens = tl.load(row_lens_ptr + offs, mask=valid, other=1)
    tl.store(out_ptr + offs, (bounds == 0) | (row_lens == 0), mask=valid)


def helix_zero_or_empty_rows(bounds: torch.Tensor, row_lens: torch.Tensor) -> torch.Tensor:
    """``(bounds == 0) | (row_lens == 0)`` as one bool tensor, in one launch.

    The Helix combine mask of a token-sparse generation batch: rows this CP rank owns no KV for
    (zero rank-local bound) or whose Top-K kept nothing on it. The same values as the two compares
    and the OR of ``MLA._attn_forward_gen``. CUDA-graph safe: no host sync, the launch depends on
    the static row count only.
    """
    num_rows = bounds.shape[0]
    if (
        bounds.dim() != 1
        or row_lens.dim() != 1
        or row_lens.shape[0] != num_rows
        or bounds.stride(0) != 1
        or row_lens.stride(0) != 1
    ):
        raise ValueError(
            f"helix_zero_or_empty_rows: bounds {tuple(bounds.shape)} and row lengths "
            f"{tuple(row_lens.shape)} must be contiguous 1-D tensors of one length"
        )
    out = torch.empty((num_rows,), dtype=torch.bool, device=bounds.device)
    if num_rows == 0:
        return out
    block = min(1024, max(32, triton.next_power_of_2(num_rows)))
    _helix_zero_or_empty_rows_kernel[(triton.cdiv(num_rows, block),)](
        bounds, row_lens, out, num_rows, BLOCK=block, num_warps=1 if block <= 256 else 4
    )
    return out
