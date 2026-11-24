# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Torch mirrors of NVFP4 sparse attention and its MXFP8 handoff."""

from __future__ import annotations

import torch

HEAD_DIM = 128
PAGE = 128
TOPK = 16
SF_VEC = 16
MX_BLOCK = 32

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequant_nvfp4_pages(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    pages: torch.Tensor,
    quad_scales: bool,
    kv_head: int = 0,
) -> torch.Tensor:
    """Dequantize KV head ``kv_head`` of the given pages of a packed NVFP4 cache to float64 [n, PAGE, HEAD_DIM].

    ``packed`` is the [pages, kv_heads, PAGE, HEAD_DIM // 2] byte view (low
    nibble = even element), ``block_scale`` the [pages, kv_heads, PAGE,
    HEAD_DIM // 16] E4M3 bytes ``fused_write_layer_caches_nvfp4`` writes: token
    ``t``'s column ``c`` at byte ``t * 8 + c`` of the page's region for K, and at
    ``(t // 4) * 32 + c * 4 + t % 4`` for V (``quad_scales``). The per-tensor
    global scale is not applied. Every product is exact in float64.
    """
    table = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    rows = packed[pages, kv_head].to(torch.int64)  # [n, PAGE, HEAD_DIM // 2]
    nibbles = torch.stack([rows & 0x0F, rows >> 4], dim=-1).reshape(rows.shape[0], PAGE, HEAD_DIM)
    cols = HEAD_DIM // SF_VEC
    region = block_scale[pages, kv_head].reshape(rows.shape[0], PAGE * cols)
    t = torch.arange(PAGE, device=packed.device)[:, None]
    c = torch.arange(cols, device=packed.device)[None, :]
    index = (t // 4) * 32 + c * 4 + t % 4 if quad_scales else t * cols + c
    scales = region[:, index.reshape(-1)].reshape(rows.shape[0], PAGE, cols)
    scales = scales.view(torch.float8_e4m3fn).to(torch.float64)
    return table[nibbles] * scales.repeat_interleave(SF_VEC, dim=-1)


def sparse_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_block_scale: torch.Tensor,
    v_block_scale: torch.Tensor,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    sm_scale: float,
    kv_head: int = 0,
) -> torch.Tensor:
    """Block-sparse decode attention over KV head ``kv_head`` of a packed NVFP4 cache, float64 [T, heads, HEAD_DIM].

    Query token ``tok`` belongs to request ``tok // decode_query_len`` at offset
    ``tok % decode_query_len`` and attends the first ``min(16, ceil(kv_len /
    PAGE))`` entries of ``topk[tok]`` that are not -1, each block truncated at
    ``kv_len = max(seq_len - decode_query_len + offset + 1, 0)``. A token that
    attends nothing (a CUDA-graph padding row) is all zeros. ``q`` is E4M3 (or
    its byte view); softmax and both products are float64.
    """
    q64 = (
        q.view(torch.float8_e4m3fn).to(torch.float64)
        if q.element_size() == 1
        else q.to(torch.float64)
    )
    total_q, num_heads, _ = q64.shape
    kgs = float(k_global_scale.reshape(-1)[0])
    vgs = float(v_global_scale.reshape(-1)[0])
    out = torch.zeros(total_q, num_heads, HEAD_DIM, dtype=torch.float64, device=q.device)
    seq = seq_lens.tolist()
    positions = torch.arange(PAGE, device=q.device)
    for tok in range(total_q):
        req, offset = divmod(tok, decode_query_len)
        kv_len = max(seq[req] - decode_query_len + offset + 1, 0)
        real_topk = min(TOPK, (kv_len + PAGE - 1) // PAGE)
        blocks = topk[tok, :real_topk]
        blocks = blocks[blocks >= 0].to(torch.int64)
        if blocks.numel() == 0:
            continue
        pages = block_table[req, blocks].to(torch.int64)
        keys = (
            dequant_nvfp4_pages(
                k, k_block_scale, pages, quad_scales=False, kv_head=kv_head
            ).reshape(-1, HEAD_DIM)
            * kgs
        )
        values = (
            dequant_nvfp4_pages(v, v_block_scale, pages, quad_scales=True, kv_head=kv_head).reshape(
                -1, HEAD_DIM
            )
            * vgs
        )
        valid = ((blocks[:, None] * PAGE + positions[None, :]) < kv_len).reshape(-1)
        logits = (q64[tok] @ keys.T) * sm_scale
        logits = logits.masked_fill(~valid[None, :], float("-inf"))
        values = values.masked_fill(~valid[:, None], 0)
        out[tok] = torch.softmax(logits, dim=-1) @ values
    return out


def mxfp8_handoff_reference(
    out: torch.Tensor, num_rows_alloc: int, *, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The MXFP8 o_proj hand-off the merge writes for BF16 rows ``out`` [T, width]: ``(q8 [T, width] uint8,
    sf [num_rows_alloc * width // 32] uint8)``.

    Per 32-element block: ``e`` is the biased exponent of ``amax * rcp(448)``
    rounded up when its mantissa is non-zero (capped at 254), the elements are
    scaled by ``2^(127 - e)`` and rounded to E4M3 (saturating), and ``e`` lands
    in the 128x4-swizzled scale layout the MXFP8 GEMM reads. Entries of ``sf``
    for rows past ``T`` are zero. Computed from the op's own BF16 rows, so a
    comparison against it is exact.
    """
    total_q = out.shape[0]
    x = out.float().reshape(total_q, width // MX_BLOCK, MX_BLOCK)
    amax = x.abs().amax(dim=-1)
    sfv = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=out.device)
    bits = sfv.view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    e = e.clamp(max=254)
    qs = torch.where(amax != 0, ((254 - e) << 23).view(torch.float32), torch.zeros_like(amax))
    xq = (x * qs[..., None]).clamp(-448.0, 448.0)
    q8 = xq.to(torch.float8_e4m3fn).view(torch.uint8).reshape(total_q, width)
    num_sf_k_tiles = width // 128
    sf = torch.zeros(num_rows_alloc * (width // MX_BLOCK), dtype=torch.uint8, device=out.device)
    tok = torch.arange(total_q, device=out.device)[:, None]
    k_sf = torch.arange(width // MX_BLOCK, device=out.device)[None, :]
    offset = (
        (tok // 128) * num_sf_k_tiles * 512
        + (k_sf // 4) * 512
        + (tok % 32) * 16
        + ((tok % 128) // 32) * 4
        + k_sf % 4
    )
    sf[offset.reshape(-1)] = e.to(torch.uint8).reshape(-1)
    return q8, sf


def validate_sparse_selection(
    q: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    identity_blocks: bool,
    page: int,
) -> None:
    """The inputs this call takes and answers wrongly.

    Every producer walks a token's first ``min(topk, ceil(kv_len / 128))``
    table entries and indexes the block table with them unchecked, so they
    must be real blocks of the request. ``identity_blocks`` is a promise the
    multi-query producer relies on: every token of a request reads the same
    blocks, all of them. Reads ``topk_idx`` and ``seq_lens`` back to the
    host: a test-only cost.
    """
    total_q = q.shape[0]
    seq = seq_lens.tolist()
    max_topk = topk_idx.shape[-1]
    for tok in range(total_q):
        req, offset = divmod(tok, decode_query_len)
        kv_len = max(seq[req] - decode_query_len + offset + 1, 0)
        nblocks = (kv_len + page - 1) // page
        ids = topk_idx[:, tok, : min(max_topk, nblocks)]
        assert bool(((ids >= 0) & (ids < min(nblocks, block_table.shape[1]))).all()), (
            f"token {tok}: the first min(topk, block count) selected ids must be blocks of the request; "
            "the block table is indexed with them unchecked"
        )
        if identity_blocks:
            assert nblocks <= max_topk and bool(
                (ids == torch.arange(nblocks, device=ids.device, dtype=ids.dtype)).all()
            ), (
                f"token {tok}: identity_blocks promises the table holds every block of the request in order"
            )
