# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.fused_ops.gelu_tanh_mul_fp4_quant import _rcp_approx

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
# Fused DSA indexer prologue (TRTLLM_DSA_INDEXER_FUSED_PROLOGUE)
########################################################
#
# ``dsa_indexer_prologue_fused`` runs the FP8 indexer's per-token prologue -- the
# work between the fused ``wk | weights_proj`` GEMM and the paged MQA-logits
# kernel -- as one launch:
#
#   1. ``k_norm`` LayerNorm over the indexer head dim
#   2. RoPE on the Q and K rope slices (flashinfer cos/sin cache layout)
#   3. [pe | nope] FP8 E4M3 quantization of Q and K, one scale per row
#   4. ``weights * q_scale * softmax_scale * n_heads**-0.5``
#
# Numerics contract (bit-identical to the per-kernel chain,
# ``Indexer._prologue_unfused``): fp32 math on bf16-rounded intermediates -- the
# LayerNorm output and the RoPE output are rounded to bf16 exactly where the
# chain materializes bf16 tensors; the flashinfer RoPE expression
# ``fma(x, cos, (+-partner) * sin)``; fused_cat_fp8's per-row scale rule (amax
# clamped at 1e-12, / 448, UE8M0 power-of-two round-up when requested), its
# ``rcp.approx.ftz`` reciprocal and the RN-satfinite E4M3 conversion.
#
# The indexer K-cache scatter is not part of this kernel: it needs the batch's
# slot mappings and stays in ``Indexer._update_k_cache``.

_PDL_ENABLED = get_env_enable_pdl()


@triton.jit
def _fp8_row_scale(amax, UE8M0: tl.constexpr):
    """fused_cat_fp8 per-row scale: amax clamped at 1e-12, / 448, optional UE8M0 round-up."""
    amax = tl.maximum(amax, 1.0e-12)
    ratio = amax * 0.002232142857142857  # 1.0f / 448.0f
    if UE8M0:
        bits = ratio.to(tl.uint32, bitcast=True)
        mant = bits & 0x007FFFFF
        exp_bits = bits & 0x7F800000
        exp_bits = tl.where(mant != 0, exp_bits + 0x00800000, exp_bits)
        scale = exp_bits.to(tl.float32, bitcast=True)
    else:
        scale = ratio
    return scale


@triton.jit
def _rope_terms(d, ROPE_DIM: tl.constexpr, INTERLEAVE: tl.constexpr):
    """Partner column, cos/sin column and sign for column ``d`` (flashinfer layout)."""
    HALF: tl.constexpr = ROPE_DIM // 2
    is_rope = d < ROPE_DIM
    if INTERLEAVE:
        # GPT-J style: pairs (2i, 2i + 1) share cos/sin column i; even lanes subtract.
        partner = tl.where(is_rope, d ^ 1, d)
        cs_col = d // 2
        neg = (d % 2) == 0
    else:
        # NeoX style: halves [0, HALF) and [HALF, ROPE_DIM); the first half subtracts.
        partner = tl.where(d < HALF, d + HALF, tl.where(is_rope, d - HALF, d))
        cs_col = d % HALF
        neg = d < HALF
    cs_col = tl.where(is_rope, cs_col, 0)
    return is_rope, partner, cs_col, neg


@triton.jit
def _apply_rope(x, xp, c, s, neg):
    # flashinfer: vec[i] = vec[i] * cos[i] + (neg ? -perm[i] : perm[i]) * sin[i];
    # nvcc contracts this to fma(vec, cos, (+-perm) * sin).
    t = tl.where(neg, -xp, xp) * s
    return tl.fma(x, c, t)


@triton.jit
def _dsa_indexer_prologue_fused_kernel(
    # inputs
    k_ptr,  # [N, D] pre-norm indexer K (bf16 or fp32; rounded to bf16 first)
    k_stride,
    ln_w_ptr,  # [D] LayerNorm weight
    ln_b_ptr,  # [D] LayerNorm bias
    q_ptr,  # [N, H, D] bf16 indexer Q (unit inner stride)
    q_stride_tok,
    q_stride_head,
    w_ptr,  # [N, H] weights_proj output (bf16 or fp32)
    w_stride,
    pos_ptr,  # [N] positions
    cos_sin_ptr,  # [max_pos, ROPE_DIM] fp32: [cos(0..HALF) | sin(0..HALF)]
    cos_sin_stride,
    # outputs
    q_fp8_ptr,  # [N, H, D] fp8 e4m3 (contiguous)
    q_scale_ptr,  # [N, H] fp32
    k_fp8_ptr,  # [N, D] fp8 e4m3 (contiguous)
    k_scale_ptr,  # [N] fp32
    w_out_ptr,  # [N, H] fp32
    # scalars
    eps,
    weight_scale,
    H: tl.constexpr,
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    UE8M0: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    tok = tl.program_id(0)
    hblk = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    tok64 = tok.to(tl.int64)
    d = tl.arange(0, D)
    is_rope, partner, cs_col, neg = _rope_terms(d, ROPE_DIM, INTERLEAVE)
    HALF: tl.constexpr = ROPE_DIM // 2

    pos = tl.load(pos_ptr + tok).to(tl.int64)
    cs_row = cos_sin_ptr + pos * cos_sin_stride
    cos_v = tl.load(cs_row + cs_col, mask=is_rope, other=1.0)
    sin_v = tl.load(cs_row + HALF + cs_col, mask=is_rope, other=0.0)

    # ---------------- Q: HEADS_PER_PROG heads of this token ----------------
    h = hblk * HEADS_PER_PROG + tl.arange(0, HEADS_PER_PROG)
    q_row = q_ptr + tok64 * q_stride_tok + h[:, None].to(tl.int64) * q_stride_head
    qx = tl.load(q_row + d[None, :]).to(tl.float32)
    qp = tl.load(q_row + partner[None, :]).to(tl.float32)
    q_rot = _apply_rope(qx, qp, cos_v[None, :], sin_v[None, :], neg[None, :])
    qv = tl.where(is_rope[None, :], q_rot, qx).to(tl.bfloat16).to(tl.float32)
    q_amax = tl.max(tl.abs(qv), axis=1)
    q_scale = _fp8_row_scale(q_amax, UE8M0)
    q_inv = _rcp_approx(q_scale)
    q8 = (qv * q_inv[:, None]).to(tl.float8e4nv)
    qo = (tok64 * H + h[:, None]) * D + d[None, :]
    tl.store(q_fp8_ptr + qo, q8)
    tl.store(q_scale_ptr + tok64 * H + h, q_scale)

    # weights * q_scale * (softmax_scale * n_heads^-0.5), same op order as _scale().
    w = tl.load(w_ptr + tok64 * w_stride + h).to(tl.float32)
    tl.store(w_out_ptr + tok64 * H + h, (w * q_scale) * weight_scale)

    # ---------------- K: once per token ----------------
    if hblk == 0:
        k_row = k_ptr + tok64 * k_stride
        kx = tl.load(k_row + d).to(tl.bfloat16).to(tl.float32)
        kxp = tl.load(k_row + partner).to(tl.bfloat16).to(tl.float32)
        mean = tl.sum(kx, axis=0) / D
        kc = kx - mean
        var = tl.sum(kc * kc, axis=0) / D
        rstd = tl.math.rsqrt(var + eps)
        lw = tl.load(ln_w_ptr + d).to(tl.float32)
        lb = tl.load(ln_b_ptr + d).to(tl.float32)
        lwp = tl.load(ln_w_ptr + partner).to(tl.float32)
        lbp = tl.load(ln_b_ptr + partner).to(tl.float32)
        kn = (kc * rstd * lw + lb).to(tl.bfloat16).to(tl.float32)
        knp = ((kxp - mean) * rstd * lwp + lbp).to(tl.bfloat16).to(tl.float32)
        k_rot = _apply_rope(kn, knp, cos_v, sin_v, neg)
        kv = tl.where(is_rope, k_rot, kn).to(tl.bfloat16).to(tl.float32)
        k_amax = tl.max(tl.abs(kv), axis=0)
        k_scale = _fp8_row_scale(k_amax, UE8M0)
        # inline asm needs a block operand: broadcast the row scale over D.
        k_inv = _rcp_approx(k_scale + tl.zeros([D], dtype=tl.float32))
        k8 = (kv * k_inv).to(tl.float8e4nv)
        tl.store(k_fp8_ptr + tok64 * D + d, k8)
        tl.store(k_scale_ptr + tok64, k_scale)


def prologue_heads_per_prog(n_heads: int) -> int:
    """Q heads per program: 8 when it divides ``n_heads``, else the largest power of two that does."""
    return 8 if n_heads % 8 == 0 else (n_heads & -n_heads)


def dsa_indexer_prologue_fused(
    indexer_k: torch.Tensor,
    weights: torch.Tensor,
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
    weight_scale: float,
    rope_dim: int,
    interleave: bool,
    use_ue8m0: bool,
    heads_per_prog: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-launch FP8 indexer prologue: k_norm + RoPE + FP8 Q/K quantization + weight scale.

    Args:
        indexer_k: [N, D] pre-norm indexer K (bf16, or fp32 which is rounded to bf16).
        weights: [N, H] weights_proj output (bf16 or fp32), unit inner stride.
        q: [N, H, D] bf16 indexer Q, unit inner stride (token/head strides are honored).
        positions: [N] (any int dtype) rope positions.
        cos_sin: [max_pos, rope_dim] fp32 flashinfer cos/sin cache ([cos | sin]).
        heads_per_prog: Q heads per program; ``prologue_heads_per_prog(H)`` when None.

    Returns (q_fp8 [N, H, D], k_fp8 [N, D], k_scale [N, 1], weights [N, H] fp32,
    q_scale [N, H, 1]) -- the ``pre_indexer_proj`` FP8 contract.
    """
    n_tok, n_heads, head_dim = q.shape
    assert head_dim == 128 and indexer_k.shape == (n_tok, head_dim)
    assert q.stride(-1) == 1 and indexer_k.stride(-1) == 1 and weights.stride(-1) == 1
    assert cos_sin.dtype == torch.float32 and cos_sin.stride(-1) == 1
    dev = q.device
    q_fp8 = torch.empty((n_tok, n_heads, head_dim), dtype=torch.float8_e4m3fn, device=dev)
    q_scale = torch.empty((n_tok, n_heads, 1), dtype=torch.float32, device=dev)
    k_fp8 = torch.empty((n_tok, head_dim), dtype=torch.float8_e4m3fn, device=dev)
    k_scale = torch.empty((n_tok, 1), dtype=torch.float32, device=dev)
    w_out = torch.empty((n_tok, n_heads), dtype=torch.float32, device=dev)
    if n_tok == 0:
        return q_fp8, k_fp8, k_scale, w_out, q_scale

    hp = prologue_heads_per_prog(n_heads) if heads_per_prog is None else heads_per_prog
    grid = (n_tok, n_heads // hp)
    _dsa_indexer_prologue_fused_kernel[grid](
        indexer_k,
        indexer_k.stride(0),
        ln_weight,
        ln_bias,
        q,
        q.stride(0),
        q.stride(1),
        weights,
        weights.stride(0),
        positions,
        cos_sin,
        cos_sin.stride(0),
        q_fp8,
        q_scale,
        k_fp8,
        k_scale,
        w_out,
        float(eps),
        float(weight_scale),
        H=n_heads,
        D=head_dim,
        ROPE_DIM=rope_dim,
        HEADS_PER_PROG=hp,
        INTERLEAVE=bool(interleave),
        UE8M0=bool(use_ue8m0),
        USE_PDL=_PDL_ENABLED,
        num_warps=4,
        launch_pdl=_PDL_ENABLED,
    )
    return q_fp8, k_fp8, k_scale, w_out, q_scale
