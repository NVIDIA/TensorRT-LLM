# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    kv_head_idx = tl.program_id(0)
    token_id = tl.program_id(1)
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
    valid_page = page_idx < max_num_blocks_per_req

    # block_table[req, page_idx] → memPoolBlockIdx (physical page index)
    bt_ptr = block_table_ptr + req * bt_stride0 + page_idx * bt_stride1
    mem_pool_idx = tl.load(bt_ptr, mask=valid_page, other=0)

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

    grid = (num_kv_heads, num_tokens, tiles_per_row)

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


@triton.jit
def _triton_gather_k_cache_kernel(
    k_cache_ptr,
    slot_fp8_ptr,
    slot_scale_ptr,
    out_fp8_ptr,
    out_scale_ptr,
    k_token_start,
    num_k_tokens,
    HEAD_DIM: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offsets = (pid * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)).to(tl.int64)
    token_mask = token_offsets < num_k_tokens

    fp8_base = tl.load(slot_fp8_ptr + k_token_start + token_offsets, mask=token_mask, other=0)
    scale_base = tl.load(slot_scale_ptr + k_token_start + token_offsets, mask=token_mask, other=0)

    byte_offsets = tl.arange(0, HEAD_DIM).to(tl.int64)
    src_fp8 = fp8_base[:, None] + byte_offsets[None, :]
    dst_fp8 = token_offsets[:, None] * HEAD_DIM + byte_offsets[None, :]
    gather_mask = token_mask[:, None]

    fp8_data = tl.load(k_cache_ptr + src_fp8, mask=gather_mask, other=0)
    tl.store(out_fp8_ptr + dst_fp8, fp8_data, mask=gather_mask)

    scale_byte_offsets = tl.arange(0, SCALE_BYTES).to(tl.int64)
    src_scale = scale_base[:, None] + scale_byte_offsets[None, :]
    dst_scale = token_offsets[:, None] * SCALE_BYTES + scale_byte_offsets[None, :]

    scale_data = tl.load(k_cache_ptr + src_scale, mask=gather_mask, other=0)
    tl.store(out_scale_ptr + dst_scale, scale_data, mask=gather_mask)


def triton_gather_k_cache(
    k_cache: torch.Tensor,
    slot_mapping_fp8: torch.Tensor,
    slot_mapping_scale: torch.Tensor,
    k_token_start: int,
    k_token_end: int,
    head_dim: int,
):
    """Gather K FP8 values and scales from the indexer K cache for a chunk.

    Replaces ``_gather_k_cache_for_chunk``, fusing ~8-12 small PyTorch ops
    (arange, unsqueeze, broadcast add, _unravel_indices, advanced indexing)
    into a single Triton kernel that directly gathers from flat byte offsets.
    This is purely data movement — bit-exact with the original.

    Args:
        k_cache: Indexer K cache pool data (2D contiguous), uint8.
        slot_mapping_fp8: Flat byte indices for FP8 data
            ``[total_kv_len]``, int64.
        slot_mapping_scale: Flat byte indices for scale data
            ``[total_kv_len]``, int64.
        k_token_start: Start index into slot mapping arrays.
        k_token_end: End index into slot mapping arrays.
        head_dim: FP8 head dimension (typically 128).

    Returns:
        Tuple of (k_fp8, k_scale):
            k_fp8: ``[num_k_tokens, head_dim]``, float8_e4m3fn.
            k_scale: ``[num_k_tokens, 1]``, float32.
    """
    num_k_tokens = k_token_end - k_token_start
    device = k_cache.device

    if num_k_tokens == 0:
        return (
            torch.empty(0, head_dim, dtype=torch.float8_e4m3fn, device=device),
            torch.empty(0, 1, dtype=torch.float32, device=device),
        )

    SCALE_BYTES = 4
    BLOCK_TOKENS = 32

    k_cache_flat = k_cache.reshape(-1)
    out_fp8 = torch.empty(num_k_tokens, head_dim, dtype=torch.uint8, device=device)
    out_scale = torch.empty(num_k_tokens, SCALE_BYTES, dtype=torch.uint8, device=device)

    grid = (triton.cdiv(num_k_tokens, BLOCK_TOKENS),)
    _triton_gather_k_cache_kernel[grid](
        k_cache_flat,
        slot_mapping_fp8,
        slot_mapping_scale,
        out_fp8.view(-1),
        out_scale.view(-1),
        k_token_start,
        num_k_tokens,
        HEAD_DIM=head_dim,
        SCALE_BYTES=SCALE_BYTES,
        BLOCK_TOKENS=BLOCK_TOKENS,
    )

    k_fp8 = out_fp8.view(torch.float8_e4m3fn)
    k_scale = out_scale.view(torch.float32).view(num_k_tokens, 1)
    return k_fp8, k_scale


########################################################
# Fused decode metadata kernel
########################################################


@triton.jit(do_not_specialize=["num_seqs", "num_tokens"])
def _fused_dsa_decode_metadata_kernel(
    seq_lens_ptr,  # int32 [num_seqs]
    kv_lens_ptr,  # int32 [num_seqs]
    block_offsets_ptr,  # int32 [num_seqs, max_blocks]
    req_idx_per_token_ptr,  # int32 [num_tokens] (out)
    slot_fp8_ptr,  # int64 [num_tokens] (out)
    slot_scale_ptr,  # int64 [num_tokens] (out)
    gen_kv_indptr_ptr,  # int64 [num_seqs + 1] (out)
    gen_cached_indptr_ptr,  # int64 [num_seqs + 1] (out)
    block_offsets_stride_0,
    block_offsets_stride_1: tl.constexpr,
    num_seqs,
    num_tokens,
    max_blocks: tl.constexpr,
    tokens_per_block: tl.constexpr,
    data_bytes_per_token: tl.constexpr,
    scale_size: tl.constexpr,
    block_stride: tl.constexpr,
    scale_base_offset: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)

    # ----- header program: the two int64 generation cumsums ------------------
    if pid == num_seqs:
        offs_b = tl.arange(0, BLOCK_S)
        mask_b = offs_b < num_seqs
        seq = tl.load(seq_lens_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        kv = tl.load(kv_lens_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        cached = kv - seq
        cu_kv = tl.cumsum(kv, 0)
        cu_cached = tl.cumsum(cached, 0)
        tl.store(gen_kv_indptr_ptr, tl.full((), 0, tl.int64))
        tl.store(gen_cached_indptr_ptr, tl.full((), 0, tl.int64))
        tl.store(gen_kv_indptr_ptr + 1 + offs_b, cu_kv, mask=mask_b)
        tl.store(gen_cached_indptr_ptr + 1 + offs_b, cu_cached, mask=mask_b)
        return

    # ----- per-request program: req_idx + slot mappings ----------------------
    r = pid
    # flat_start = sum(seq_lens[0:r]) computed in registers (num_seqs tiny)
    offs_s = tl.arange(0, BLOCK_S)
    seq_all = tl.load(seq_lens_ptr + offs_s, mask=offs_s < num_seqs, other=0)
    flat_start = tl.sum(tl.where(offs_s < r, seq_all, 0)).to(tl.int64)

    seq_r = tl.load(seq_lens_ptr + r).to(tl.int64)
    kv_r = tl.load(kv_lens_ptr + r).to(tl.int64)
    start_pos = kv_r - seq_r

    offs_j = tl.arange(0, BLOCK_T).to(tl.int64)
    g = flat_start + offs_j  # global (flattened) token idx
    # Bound every store by the host-known num_tokens in addition to the
    # per-request length. g is derived from a device-side prefix sum of
    # seq_lens, so a stale/corrupt seq_lens_cuda must never let a write escape
    # the [0, num_tokens) live extent -- the eager chain writes host-sliced
    # buffers and is bounded by construction, and this restores that guarantee.
    mask_j = (offs_j < seq_r) & (g < num_tokens)

    # req_idx_per_token[g] = r
    tl.store(req_idx_per_token_ptr + g, tl.full((BLOCK_T,), r, tl.int32), mask=mask_j)

    gpos = start_pos + offs_j  # global KV position
    blk = gpos // tokens_per_block
    pos = gpos % tokens_per_block
    # Match torch's floor-division / non-negative remainder for gpos < 0: Triton
    # lowers signed //,% to truncate-toward-zero with a sign-following
    # remainder, so normalize pos into [0, tokens_per_block). The lower clamp
    # below already erases the corresponding quotient difference. gpos < 0 only
    # arises from stale token-to-seq mappings (the eager _compute_slot_mappings
    # cuda path defends against the same case).
    pos = tl.where(pos < 0, pos + tokens_per_block, pos)
    # clamp to prevent OOB from stale token-to-seq mappings under CUDA graph
    # replay with MTP + DSA (matches _compute_slot_mappings cuda path).
    blk = tl.minimum(tl.maximum(blk, 0), max_blocks - 1)
    bid = tl.load(
        block_offsets_ptr + r * block_offsets_stride_0 + blk * block_offsets_stride_1,
        mask=mask_j,
        other=0,
    ).to(tl.int64)

    fp8 = bid * block_stride + pos * data_bytes_per_token
    scale = bid * block_stride + scale_base_offset + pos * scale_size
    tl.store(slot_fp8_ptr + g, fp8, mask=mask_j)
    tl.store(slot_scale_ptr + g, scale, mask=mask_j)


def fused_dsa_decode_metadata(
    seq_lens: torch.Tensor,  # int32 [num_seqs]
    kv_lens: torch.Tensor,  # int32 [num_seqs]
    block_offsets: torch.Tensor,  # int32 [num_seqs, max_blocks]
    req_idx_per_token: torch.Tensor,  # int32 [>= num_tokens] (out, sliced view)
    slot_mapping_fp8: torch.Tensor,  # int64 [>= num_tokens] (out, sliced view)
    slot_mapping_scale: torch.Tensor,  # int64 [>= num_tokens] (out, sliced view)
    gen_kv_indptr: torch.Tensor,  # int64 [>= num_seqs + 1] (out, sliced view)
    gen_cached_token_indptr: torch.Tensor,  # int64 [>= num_seqs + 1] (out, sliced)
    num_tokens: int,
    max_query_len: int,
    tokens_per_block: int,
    index_head_dim: int,
    quant_block_size: int,
    data_bytes_per_token: int | None = None,
) -> None:
    """Fill the DSA decode metadata (req_idx + slot mappings + gen indptrs) in one
    Triton launch.  Bit-identical to the eager chain in on_update_kv_lens.

    ``max_query_len`` is the per-request upper bound on query tokens at decode
    (== next_n == 1 + max_draft_tokens). It sizes the per-request lane count
    (BLOCK_T); it must be >= every ``seq_lens[r]``.

    Views must already be sliced to their live extents:
    ``req_idx_per_token[:num_tokens]``, ``slot_mapping_*[:num_tokens]``,
    ``gen_*_indptr[:num_seqs + 1]``.
    """
    num_seqs = seq_lens.shape[0]
    assert num_seqs > 0 and num_tokens > 0
    assert block_offsets.shape[0] == num_seqs
    # Coarse aggregate sanity check. It does NOT prove per-request coverage
    # (each seq_len <= max_query_len): e.g. seq_lens=[10, 1, 1, 1], next_n=8
    # passes yet request 0 needs 10 lanes. The pure-decode contract guarantees
    # every gen request carries exactly next_n query tokens (so sum == num_tokens
    # and each seq_len == next_n <= max_query_len); the per-store g < num_tokens
    # mask in the kernel keeps any invariant violation in-bounds rather than
    # corrupting neighboring buffers.
    assert num_tokens <= num_seqs * max_query_len, (
        f"fused DSA metadata: num_tokens={num_tokens} exceeds "
        f"num_seqs={num_seqs} * max_query_len={max_query_len}"
    )

    if data_bytes_per_token is None:
        data_bytes_per_token = index_head_dim
    scale_size = index_head_dim // quant_block_size * 4  # float32 = 4 bytes
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token
    max_blocks = block_offsets.shape[1]

    # Per-request lane count: bound by the per-request query length (next_n),
    # NOT num_tokens. Sizing by num_tokens would make each of the num_seqs
    # programs iterate ~num_tokens lanes (O(num_seqs^2) work) and blow up the
    # constexpr BLOCK_T at high concurrency (e.g. bs=256 -> 2048: register
    # spill + a distinct compiled signature per graph bucket). next_n is a
    # batch-independent host constant (typically 8), keeping work O(num_tokens).
    block_t = triton.next_power_of_2(max_query_len)
    block_s = triton.next_power_of_2(num_seqs)
    grid = (num_seqs + 1,)

    _fused_dsa_decode_metadata_kernel[grid](
        seq_lens,
        kv_lens,
        block_offsets,
        req_idx_per_token,
        slot_mapping_fp8,
        slot_mapping_scale,
        gen_kv_indptr,
        gen_cached_token_indptr,
        block_offsets.stride(0),
        block_offsets.stride(1),
        num_seqs,
        num_tokens,
        max_blocks,
        tokens_per_block,
        data_bytes_per_token,
        scale_size,
        block_stride,
        scale_base_offset,
        BLOCK_S=block_s,
        BLOCK_T=block_t,
    )


########################################################
# Fused Helix decode metadata kernel
########################################################

# DeepGEMM's SM100 paged-MQA-logits schedule (``sm100_paged_mqa_logits_metadata``): every request is
# split into ``ceil(context_len / 256)`` work units, independent of the ``block_kv`` argument.
_DG_SM100_SPLIT_KV = 256
# Rows per step of the schedule program's scan.
_DG_SCHEDULE_BLOCK_R = 32


@triton.jit
def _dg_sm100_schedule(
    lens_ptr,  # int32 [num_rows]: context length of each row
    out_ptr,  # int32 [num_slots + 1, 2] (out)
    num_rows,
    num_slots,
    AT_LEAST_ONE: tl.constexpr,  # read max(lens, 1)
    SPLIT_KV: tl.constexpr,
    BLOCK_SM: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """``get_paged_mqa_logits_metadata(lens.view(-1, 1), block_kv, num_slots)`` of DeepGEMM's SM100 path.

    Row ``r`` holds ``ceil(lens[r] / SPLIT_KV)`` work units; slot ``s`` starts at unit
    ``w = s * (total // num_slots) + min(s, total % num_slots)``, written as ``(r, w - prefix[r - 1])``
    for the row ``r`` whose inclusive prefix first exceeds ``w``, or ``(num_rows, 0)`` past the last
    row. Lengths are non-negative and the total stays below 2**31, so the int32 arithmetic is
    DeepGEMM's uint32 arithmetic.
    """
    offs_r = tl.arange(0, BLOCK_R)
    total = tl.sum(tl.zeros([BLOCK_R], dtype=tl.int32), axis=0)
    for start in range(0, num_rows, BLOCK_R):
        mask_r = start + offs_r < num_rows
        lens = tl.load(lens_ptr + start + offs_r, mask=mask_r, other=0)
        if AT_LEAST_ONE:
            lens = tl.maximum(lens, 1)
        total += tl.sum(tl.where(mask_r, (lens + SPLIT_KV - 1) // SPLIT_KV, 0), axis=0)
    per_slot = total // num_slots
    rem = total - per_slot * num_slots
    slot = tl.arange(0, BLOCK_SM)
    w = slot * per_slot + tl.minimum(slot, rem)
    # The row of w: the number of inclusive prefixes <= w (prefixes are non-decreasing); the
    # prefix before it is the largest of those (0 when there is none).
    row = tl.zeros([BLOCK_SM], dtype=tl.int32)
    before = tl.zeros([BLOCK_SM], dtype=tl.int32)
    carry = tl.sum(tl.zeros([BLOCK_R], dtype=tl.int32), axis=0)
    for start in range(0, num_rows, BLOCK_R):
        mask_r = start + offs_r < num_rows
        lens = tl.load(lens_ptr + start + offs_r, mask=mask_r, other=0)
        if AT_LEAST_ONE:
            lens = tl.maximum(lens, 1)
        work = tl.where(mask_r, (lens + SPLIT_KV - 1) // SPLIT_KV, 0)
        prefix = carry + tl.cumsum(work, 0)
        le = (prefix[None, :] <= w[:, None]) & mask_r[None, :]
        row += tl.sum(le.to(tl.int32), axis=1)
        before = tl.maximum(before, tl.max(tl.where(le, prefix[None, :], 0), axis=1))
        carry += tl.sum(work, axis=0)
    tail = row >= num_rows
    q_idx = tl.where(tail, num_rows, row)
    split = tl.where(tail, 0, w - before)
    mask_sm = slot <= num_slots
    tl.store(out_ptr + slot * 2, q_idx, mask=mask_sm)
    tl.store(out_ptr + slot * 2 + 1, split, mask=mask_sm)


@triton.jit(do_not_specialize=["num_seqs", "num_tokens", "width", "max_blocks", "num_slots"])
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
    schedule_ptr,  # int32 [num_slots + 1, 2] (out, with SCHEDULE): DeepGEMM schedule of kv_lens
    schedule_expanded_ptr,  # int32 [num_slots + 1, 2] (out, with SCHEDULE): of kv_lens_expanded
    block_offsets_stride_0,
    block_offsets_stride_1,
    kv_lens_2d_stride_0,
    num_seqs,
    num_tokens,
    width,
    max_blocks,
    num_slots,
    tokens_per_block: tl.constexpr,
    data_bytes_per_token: tl.constexpr,
    scale_size: tl.constexpr,
    block_stride: tl.constexpr,
    scale_base_offset: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
    SCHEDULE: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    BLOCK_SM: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    num_token_programs = tl.cdiv(num_tokens, BLOCK_T)
    # ----- with SCHEDULE, the last two programs: the DeepGEMM schedules --------------------
    if SCHEDULE:
        if pid == 1 + num_token_programs:
            _dg_sm100_schedule(
                kv_lens_ptr, schedule_ptr, num_seqs, num_slots, False, SPLIT_KV, BLOCK_SM, BLOCK_R
            )
            return
        if pid == 2 + num_token_programs:
            # kv_lens_expanded = max(helix_kv_bounds, 1), read from the bounds directly.
            _dg_sm100_schedule(
                helix_kv_bounds_ptr,
                schedule_expanded_ptr,
                num_tokens,
                num_slots,
                True,
                SPLIT_KV,
                BLOCK_SM,
                BLOCK_R,
            )
            return
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
    schedule: torch.Tensor | None = None,  # int32 [num_slots + 1, 2] (out)
    schedule_expanded: torch.Tensor | None = None,  # int32 [num_slots + 1, 2] (out)
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

    With ``schedule`` and ``schedule_expanded`` (both or neither), the same launch also writes the
    DeepGEMM SM100 paged-MQA-logits schedules ``get_paged_mqa_logits_metadata`` builds over
    ``num_slots = schedule.shape[0] - 1`` slots: of ``kv_lens.view(-1, 1)`` into ``schedule`` and of
    ``kv_lens_expanded[:num_tokens].view(-1, 1)`` into ``schedule_expanded``
    (:func:`_dg_sm100_schedule`). Without them the schedules are left to the caller.

    Integer arithmetic only, so the outputs are bitwise those of the eager chain.
    Graph-capturable (no allocation, no host sync).
    """
    num_seqs = seq_lens.shape[0]
    assert num_seqs > 0 and num_tokens > 0
    assert block_offsets.shape[0] >= num_seqs
    with_schedule = schedule is not None
    assert with_schedule == (schedule_expanded is not None)
    if with_schedule:
        num_slots = schedule.shape[0] - 1
        for buf in (schedule, schedule_expanded):
            assert buf.dtype == torch.int32 and buf.is_contiguous()
            assert buf.shape == (num_slots + 1, 2)
        assert num_slots > 0
    else:
        num_slots = 1
        schedule = schedule_expanded = kv_lens_expanded
    if data_bytes_per_token is None:
        data_bytes_per_token = index_head_dim
    scale_size = index_head_dim // quant_block_size * 4  # float32 = 4 bytes
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token
    block_s = triton.next_power_of_2(num_seqs)
    block_t = min(128, triton.next_power_of_2(num_tokens))
    block_w = triton.next_power_of_2(max(width, 1))
    grid = (1 + triton.cdiv(num_tokens, block_t) + (2 if with_schedule else 0),)
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
        schedule,
        schedule_expanded,
        block_offsets.stride(0),
        block_offsets.stride(1),
        kv_lens_2d.stride(0),
        num_seqs,
        num_tokens,
        width,
        block_offsets.shape[1],
        num_slots,
        tokens_per_block,
        data_bytes_per_token,
        scale_size,
        block_stride,
        scale_base_offset,
        BLOCK_S=block_s,
        BLOCK_T=block_t,
        BLOCK_W=block_w,
        SCHEDULE=with_schedule,
        SPLIT_KV=_DG_SM100_SPLIT_KV,
        BLOCK_SM=triton.next_power_of_2(num_slots + 1),
        BLOCK_R=_DG_SCHEDULE_BLOCK_R,
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
