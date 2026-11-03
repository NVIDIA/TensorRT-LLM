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


@triton.jit
def _gather_k_cache_paged_kernel(
    k_cache_ptr,  # uint8 base of the [pages, page_bytes] layer view (page stride PAGE_STRIDE)
    slot_fp8_ptr,  # int64 flat byte index of each token's data (< 0: skip the token)
    slot_scale_ptr,  # int64 flat byte index of each token's scale
    out_fp8_ptr,
    out_scale_ptr,
    k_token_start,
    num_k_tokens,
    page_bytes,
    page_stride,
    HEAD_DIM: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    t = (tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)).to(tl.int64)
    m = t < num_k_tokens
    f = tl.load(slot_fp8_ptr + k_token_start + t, mask=m, other=-1)
    fs = tl.load(slot_scale_ptr + k_token_start + t, mask=m, other=-1)
    ok = m & (f >= 0) & (fs >= 0)
    f = tl.where(ok, f, 0)
    fs = tl.where(ok, fs, 0)
    src = (f // page_bytes) * page_stride + f % page_bytes
    src_s = (fs // page_bytes) * page_stride + fs % page_bytes
    b = tl.arange(0, HEAD_DIM).to(tl.int64)
    x = tl.load(k_cache_ptr + src[:, None] + b[None, :], mask=ok[:, None])
    tl.store(out_fp8_ptr + t[:, None] * HEAD_DIM + b[None, :], x, mask=ok[:, None])
    bs = tl.arange(0, SCALE_BYTES).to(tl.int64)
    xs = tl.load(k_cache_ptr + src_s[:, None] + bs[None, :], mask=ok[:, None])
    tl.store(out_scale_ptr + t[:, None] * SCALE_BYTES + bs[None, :], xs, mask=ok[:, None])


def gather_k_cache_paged(
    k_cache_4d: torch.Tensor,
    slot_mapping_fp8: torch.Tensor,
    slot_mapping_scale: torch.Tensor,
    k_token_start: int,
    num_k_tokens: int,
    head_dim: int,
):
    """``torch.ops.trtllm.indexer_k_cache_gather_op`` for a page-strided indexer K-cache view
    ``[pages, tokens_per_block, 1, bytes_per_token]`` whose pages are contiguous: the same bytes (tokens with
    a negative slot are skipped, as there), many tokens per program instead of one 32-thread block per token.
    Returns ``None`` for other layouts (the caller keeps the C++ op)."""
    pages, tpb, heads, per_token = k_cache_4d.shape
    if not (
        heads == 1
        and k_cache_4d.dtype == torch.uint8
        and k_cache_4d.stride(3) == 1
        and k_cache_4d.stride(1) == per_token
        and head_dim in (64, 128, 256)
    ):
        return None
    device = k_cache_4d.device
    scale_bytes = 4
    out_fp8 = torch.empty(num_k_tokens, head_dim, dtype=torch.uint8, device=device)
    out_scale = torch.empty(num_k_tokens, scale_bytes, dtype=torch.uint8, device=device)
    if num_k_tokens > 0:
        block_tokens = 32
        _gather_k_cache_paged_kernel[(triton.cdiv(num_k_tokens, block_tokens),)](
            k_cache_4d,
            slot_mapping_fp8,
            slot_mapping_scale,
            out_fp8,
            out_scale,
            k_token_start,
            num_k_tokens,
            tpb * per_token,
            k_cache_4d.stride(0),
            HEAD_DIM=head_dim,
            SCALE_BYTES=scale_bytes,
            BLOCK_TOKENS=block_tokens,
        )
    return out_fp8.view(torch.float8_e4m3fn), out_scale.view(torch.float32).view(num_k_tokens, 1)


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
# BF16 MQA logits
########################################################
# Index scores sum_h w[t, h] * relu(q[t, h] . k[s]) from BF16 queries and keys
# with FP32 accumulation, for checkpoints whose indexer is not quantized.


@triton.jit(do_not_specialize=["num_q", "num_k"])
def _bf16_mqa_logits_kernel(
    q_ptr,  # bf16 [num_q, num_heads, head_dim]
    k_ptr,  # bf16 [num_k, head_dim]
    w_ptr,  # fp32 [num_q, num_heads]
    ks_ptr,  # int32 [num_q], first key of each query's window
    ke_ptr,  # int32 [num_q], one past the last key of each query's window
    out_ptr,  # fp32 [num_q, num_k]
    num_q,
    num_k,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_wt,
    stride_ot,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    col_start = tl.program_id(1) * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    row_ok = rows < num_q
    col_ok = cols < num_k
    # Padded rows get an empty window.
    ks = tl.load(ks_ptr + rows, mask=row_ok, other=num_k)
    ke = tl.load(ke_ptr + rows, mask=row_ok, other=0)
    scores = tl.full((BLOCK_Q, BLOCK_N), float("-inf"), tl.float32)
    if (tl.max(ke, axis=0) > col_start) & (tl.min(ks, axis=0) < col_start + BLOCK_N):
        d = tl.arange(0, HEAD_DIM)
        k = tl.load(
            k_ptr + cols[:, None].to(tl.int64) * stride_kt + d[None, :],
            mask=col_ok[:, None],
            other=0.0,
        )
        k_t = tl.trans(k)
        q_rows = q_ptr + rows[:, None].to(tl.int64) * stride_qt + d[None, :]
        w_rows = w_ptr + rows.to(tl.int64) * stride_wt
        acc = tl.zeros((BLOCK_Q, BLOCK_N), tl.float32)
        for h in range(NUM_HEADS):
            q = tl.load(q_rows + h * stride_qh, mask=row_ok[:, None], other=0.0)
            w = tl.load(w_rows + h, mask=row_ok, other=0.0)
            acc += tl.maximum(tl.dot(q, k_t), 0.0) * w[:, None]
        in_window = (cols[None, :] >= ks[:, None]) & (cols[None, :] < ke[:, None])
        scores = tl.where(in_window, acc, float("-inf"))
    tl.store(
        out_ptr + rows[:, None].to(tl.int64) * stride_ot + cols[None, :],
        scores,
        mask=row_ok[:, None] & col_ok[None, :],
    )


def bf16_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Index scores of ragged prefill queries against one shared key matrix.

    Args:
        q: BF16 queries, [num_q, num_heads, head_dim].
        k: BF16 keys, [num_k, head_dim].
        weights: FP32 per-head weights, [num_q, num_heads].
        cu_seqlen_ks: int32 [num_q], first key each query sees.
        cu_seqlen_ke: int32 [num_q], one past the last key each query sees.

    Returns:
        FP32 [num_q, num_k] scores, -inf outside each row's [ks, ke) window
        (the cleaned output of DeepGEMM's fp8_mqa_logits).
    """
    num_q, num_heads, head_dim = q.shape
    num_k = k.shape[0]
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and weights.stride(-1) == 1
    logits = torch.empty((num_q, num_k), dtype=torch.float32, device=q.device)
    if num_q == 0 or num_k == 0:
        return logits
    block_q, block_n = 64, 64
    grid = (triton.cdiv(num_q, block_q), triton.cdiv(num_k, block_n))
    _bf16_mqa_logits_kernel[grid](
        q,
        k,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        num_q,
        num_k,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        weights.stride(0),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_Q=block_q,
        BLOCK_N=block_n,
    )
    return logits


@triton.jit(do_not_specialize=["max_seq_len", "lens_col"])
def _bf16_paged_mqa_logits_kernel(
    q_ptr,  # bf16 [batch, next_n, num_heads, head_dim]
    k_pages_ptr,  # bf16 view of the indexer pages
    w_ptr,  # fp32 [batch * next_n, num_heads]
    context_lens_ptr,  # int32 [batch, lens_cols]; lens_col is the last column's offset
    block_table_ptr,  # int32 [batch, max_blocks]
    out_ptr,  # fp32 [batch * next_n, max_seq_len]
    max_seq_len,
    lens_col,
    stride_qb,
    stride_qn,
    stride_qh,
    stride_page,
    stride_wt,
    stride_cb,
    stride_btb,
    stride_ot,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    PAGES_PER_PROGRAM: tl.constexpr,
):
    b = tl.program_id(0)
    # DeepGEMM's convention: the request's last query sees context_len keys,
    # query n of NEXT_N sees context_len - NEXT_N + n + 1.
    context_len = tl.load(context_lens_ptr + b * stride_cb + lens_col)
    h = tl.arange(0, BLOCK_H)
    h_ok = h < NUM_HEADS
    d = tl.arange(0, HEAD_DIM)
    t = tl.arange(0, TOKENS_PER_BLOCK)
    for p in tl.static_range(PAGES_PER_PROGRAM):
        page = tl.program_id(1) * PAGES_PER_PROGRAM + p
        cols = page * TOKENS_PER_BLOCK + t
        col_ok = cols < max_seq_len
        if page * TOKENS_PER_BLOCK < context_len:
            block = tl.load(block_table_ptr + b * stride_btb + page).to(tl.int64)
            # Keys fill the front of each page; the per-token scale slots that
            # follow them are unused for BF16.
            k = tl.load(k_pages_ptr + block * stride_page + t[:, None] * HEAD_DIM + d[None, :])
            k_t = tl.trans(k)
            for n in tl.static_range(NEXT_N):
                row = b * NEXT_N + n
                q = tl.load(
                    q_ptr + b * stride_qb + n * stride_qn + h[:, None] * stride_qh + d[None, :],
                    mask=h_ok[:, None],
                    other=0.0,
                )
                w = tl.load(w_ptr + row * stride_wt + h, mask=h_ok, other=0.0)
                dots = tl.maximum(tl.dot(q, k_t), 0.0) * w[:, None]
                scores = tl.sum(dots, axis=0)
                scores = tl.where(cols < context_len - NEXT_N + n + 1, scores, float("-inf"))
                tl.store(out_ptr + row.to(tl.int64) * stride_ot + cols, scores, mask=col_ok)
        else:
            for n in tl.static_range(NEXT_N):
                row = b * NEXT_N + n
                tl.store(
                    out_ptr + row.to(tl.int64) * stride_ot + cols,
                    tl.full((TOKENS_PER_BLOCK,), float("-inf"), tl.float32),
                    mask=col_ok,
                )


def bf16_paged_mqa_logits(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Index scores of decode queries against their paged BF16 indexer keys.

    Args:
        q: BF16 queries, [batch, next_n, num_heads, head_dim].
        k_cache: uint8 indexer pages, [num_blocks, tokens_per_block, 1,
            bytes_per_token]. Each page holds tokens_per_block BF16 keys
            followed by tokens_per_block 4-byte scale slots.
        weights: FP32 per-head weights, [batch * next_n, num_heads].
        context_lens: int32 [batch] or [batch, next_n]; the last entry of a
            row is the request's KV length.
        block_table: int32 [batch, max_blocks] page indices into k_cache.
        max_seq_len: Width of the output rows.

    Returns:
        FP32 [batch * next_n, max_seq_len] scores, -inf past each query's
        causal end (the cleaned output of DeepGEMM's fp8_paged_mqa_logits).
    """
    batch, next_n, num_heads, head_dim = q.shape
    tokens_per_block = k_cache.shape[1]
    assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.uint8
    assert q.stride(-1) == 1 and weights.stride(-1) == 1
    k_pages = k_cache.view(torch.bfloat16)
    assert k_pages.shape[-1] * 2 == k_cache.shape[-1]
    if context_lens.dim() == 1:
        context_lens = context_lens.unsqueeze(1)
    lens_col = (context_lens.shape[1] - 1) * context_lens.stride(1)
    logits = torch.empty((batch * next_n, max_seq_len), dtype=torch.float32, device=q.device)
    if batch == 0 or max_seq_len == 0:
        return logits
    pages_per_program = 4
    grid = (batch, triton.cdiv(triton.cdiv(max_seq_len, tokens_per_block), pages_per_program))
    _bf16_paged_mqa_logits_kernel[grid](
        q,
        k_pages,
        weights,
        context_lens,
        block_table,
        logits,
        max_seq_len,
        lens_col,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_pages.stride(0),
        weights.stride(0),
        context_lens.stride(0),
        block_table.stride(0),
        logits.stride(0),
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        BLOCK_H=max(16, triton.next_power_of_2(num_heads)),
        HEAD_DIM=head_dim,
        TOKENS_PER_BLOCK=tokens_per_block,
        PAGES_PER_PROGRAM=pages_per_program,
    )
    return logits
