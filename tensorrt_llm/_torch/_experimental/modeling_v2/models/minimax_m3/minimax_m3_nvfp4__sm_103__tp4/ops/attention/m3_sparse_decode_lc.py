# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch long-context sparse decode over the selected NVFP4 blocks (16-CTA clusters).

Also the plain-torch block-sparse attention and MXFP8 hand-off mirrors the
Triton sparse decode entry (``m3_sparse_decode``) reuses: both compute the same
thing over the same cache.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The TP4 geometry the kernel is compiled for.
NUM_HEADS = 16
HEAD_DIM = 128
PAGE = 128
TOPK = 16
#: NVFP4 elements per E4M3 block scale.
SF_VEC = 16
#: MXFP8 elements per UE8M0 scale, and the o_proj width the hand-off is laid out for.
MX_BLOCK = 32
WIDTH = NUM_HEADS * HEAD_DIM

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequant_nvfp4_pages(
    packed: torch.Tensor, block_scale: torch.Tensor, pages: torch.Tensor, quad_scales: bool
) -> torch.Tensor:
    """Dequantize KV head 0 of the given pages of a packed NVFP4 cache to float64 [n, PAGE, HEAD_DIM].

    ``packed`` is the [pages, kv_heads, PAGE, HEAD_DIM // 2] byte view (low
    nibble = even element), ``block_scale`` the [pages, kv_heads, PAGE,
    HEAD_DIM // 16] E4M3 bytes ``fused_write_layer_caches_nvfp4`` writes: token
    ``t``'s column ``c`` at byte ``t * 8 + c`` of the page's region for K, and at
    ``(t // 4) * 32 + c * 4 + t % 4`` for V (``quad_scales``). The per-tensor
    global scale is not applied. Every product is exact in float64.
    """
    table = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    rows = packed[pages, 0].to(torch.int64)  # [n, PAGE, HEAD_DIM // 2]
    nibbles = torch.stack([rows & 0x0F, rows >> 4], dim=-1).reshape(rows.shape[0], PAGE, HEAD_DIM)
    cols = HEAD_DIM // SF_VEC
    region = block_scale[pages, 0].reshape(rows.shape[0], PAGE * cols)
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
) -> torch.Tensor:
    """Block-sparse decode attention over one KV head of a packed NVFP4 cache, float64 [T, heads, HEAD_DIM].

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
            dequant_nvfp4_pages(k, k_block_scale, pages, quad_scales=False).reshape(-1, HEAD_DIM)
            * kgs
        )
        values = (
            dequant_nvfp4_pages(v, v_block_scale, pages, quad_scales=True).reshape(-1, HEAD_DIM)
            * vgs
        )
        valid = ((blocks[:, None] * PAGE + positions[None, :]) < kv_len).reshape(-1)
        logits = (q64[tok] @ keys.T) * sm_scale
        logits = logits.masked_fill(~valid[None, :], float("-inf"))
        out[tok] = torch.softmax(logits, dim=-1) @ values
    return out


def mxfp8_handoff_reference(
    out: torch.Tensor, num_rows_alloc: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The MXFP8 o_proj hand-off the merge writes for BF16 rows ``out`` [T, WIDTH]: ``(q8 [T, WIDTH] uint8,
    sf [num_rows_alloc * WIDTH // 32] uint8)``.

    Per 32-element block: ``e`` is the biased exponent of ``amax * rcp(448)``
    rounded up when its mantissa is non-zero (capped at 254), the elements are
    scaled by ``2^(127 - e)`` and rounded to E4M3 (saturating), and ``e`` lands
    in the 128x4-swizzled scale layout the MXFP8 GEMM reads. Entries of ``sf``
    for rows past ``T`` are zero. Computed from the op's own BF16 rows, so a
    comparison against it is exact.
    """
    total_q = out.shape[0]
    x = out.float().reshape(total_q, WIDTH // MX_BLOCK, MX_BLOCK)
    amax = x.abs().amax(dim=-1)
    sfv = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=out.device)
    bits = sfv.view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    e = e.clamp(max=254)
    qs = torch.where(amax != 0, ((254 - e) << 23).view(torch.float32), torch.zeros_like(amax))
    xq = (x * qs[..., None]).clamp(-448.0, 448.0)
    q8 = xq.to(torch.float8_e4m3fn).view(torch.uint8).reshape(total_q, WIDTH)
    num_sf_k_tiles = WIDTH // 128
    sf = torch.zeros(num_rows_alloc * (WIDTH // MX_BLOCK), dtype=torch.uint8, device=out.device)
    tok = torch.arange(total_q, device=out.device)[:, None]
    k_sf = torch.arange(WIDTH // MX_BLOCK, device=out.device)[None, :]
    offset = (
        (tok // 128) * num_sf_k_tiles * 512
        + (k_sf // 4) * 512
        + (tok % 32) * 16
        + ((tok % 128) // 32) * 4
        + k_sf % 4
    )
    sf[offset.reshape(-1)] = e.to(torch.uint8).reshape(-1)
    return q8, sf


class M3SparseDecodeLc(OpWrapper):
    """Sparse decode attention, split-K merge and MXFP8 hand-off of one TP4 decode step in one launch.

    One cluster of 16 CTAs per query token: CTA ``r`` attends the token's
    ``r``-th selected block (``topk[tok, r]``), and the cluster merges the 16
    partials over distributed shared memory into the BF16 row ``out[tok]``. With
    ``q8`` / ``sf`` it also writes the row's MXFP8 copy and swizzled UE8M0
    scales for the o_proj GEMM. All operands are the byte views
    ``m3_attn_lc.m3_lc_attn`` hands over (E4M3 ``q``, packed E2M1 pages, E4M3
    block scales). Returns ``out``.

    Per block: NVFP4 K/V dequantized to FP16 (exact), FP8 q widened to FP16
    (exact), QK and PV on FP16 tensor cores with FP32 accumulation, the softmax
    probabilities rounded to FP16 for PV, log2-domain softmax with ``ex2.approx``.
    ``linear_softmax`` picks the denominator recurrence (the two agree to
    rounding); the target picks it per batch shape.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode without speculation, one request: T=1, log-domain recurrence, MXFP8 hand-off",
            spec=dict(seq_lens=(9000,), decode_query_len=1, linear_softmax=False, mxfp8=True),
        ),
        Cell(
            why="Eagle3 verify at one request: T=4, the log-domain recurrence the target keeps for local batch 1",
            spec=dict(seq_lens=(20000,), decode_query_len=4, linear_softmax=False, mxfp8=True),
        ),
        Cell(
            why="Eagle3 verify at two requests: T=8 on the linear denominator",
            spec=dict(seq_lens=(5000, 12345), decode_query_len=4, linear_softmax=True, mxfp8=True),
        ),
        Cell(
            why="Eagle3 verify at batch 8 (T=32): short requests with -1 tails and zero-length padding rows",
            spec=dict(
                seq_lens=(16384, 2049, 130, 0, 7777, 4, 30000, 0),
                decode_query_len=4,
                linear_softmax=True,
                mxfp8=True,
            ),
        ),
        Cell(
            why="decode without speculation at batch 8: T=8 rows of distinct requests",
            spec=dict(
                seq_lens=(3000, 2100, 9999, 2048, 2049, 4500, 16000, 2200),
                decode_query_len=1,
                linear_softmax=False,
                mxfp8=True,
            ),
        ),
        Cell(
            why="no hand-off buffers (the variant the call site takes past 128 rows), driven at T=8",
            spec=dict(seq_lens=(5000, 12345), decode_query_len=4, linear_softmax=True, mxfp8=False),
        ),
    )

    note = """
    Certified: the four instantiated variants (`linear_softmax` x hand-off on /
    off), all with `early_trigger=True` -- the op refuses `False`. The
    hand-off cells are certified bit-exactly against `mxfp8_handoff_reference`
    applied to the op's own BF16 rows; the BF16 rows themselves within the band
    the test derives.

    `topk` is read for the first `min(16, ceil(kv_len / 128))` entries of each
    token; a -1 among them skips that CTA's block, entries past them are never
    read. `use_pdl` only changes the launch attribute. The hand-off assumes
    `num_sf_k_tiles = 16` (a 2048-wide o_proj input) and buffers with at least
    as many rows as the step; `MERGE_MXFP8` sizes them at 128 rows.
    """

    def __call__(
        self,
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
        out: torch.Tensor,
        q8: Optional[torch.Tensor],
        sf: Optional[torch.Tensor],
        num_sf_k_tiles: int,
        decode_query_len: int,
        sm_scale: float,
        linear_softmax: bool,
        use_pdl: bool,
        early_trigger: bool,
    ) -> torch.Tensor:
        torch.ops.trtllm.minimax_m3_sparse_decode_lc(
            q,
            k,
            v,
            k_block_scale,
            v_block_scale,
            k_global_scale,
            v_global_scale,
            topk,
            block_table,
            seq_lens,
            out,
            q8,
            sf,
            num_sf_k_tiles,
            decode_query_len,
            sm_scale,
            linear_softmax,
            use_pdl,
            early_trigger,
        )
        return out

    def reference(
        self,
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
        out: torch.Tensor,
        q8: Optional[torch.Tensor],
        sf: Optional[torch.Tensor],
        num_sf_k_tiles: int,
        decode_query_len: int,
        sm_scale: float,
        linear_softmax: bool,
        use_pdl: bool,
        early_trigger: bool,
    ) -> torch.Tensor:
        """The BF16 ``out`` [T, 2048] the op is supposed to write: ``sparse_attention_reference`` in float64,
        rounded once. ``linear_softmax``, ``use_pdl`` and ``early_trigger`` do not change what is computed; the
        hand-off buffers are checked separately with ``mxfp8_handoff_reference``.
        """
        attn = sparse_attention_reference(
            q.view(-1, NUM_HEADS, HEAD_DIM),
            k,
            v,
            k_block_scale,
            v_block_scale,
            k_global_scale,
            v_global_scale,
            topk,
            block_table,
            seq_lens,
            decode_query_len,
            sm_scale,
        )
        return attn.reshape(out.shape).to(out.dtype)

    def is_valid(
        self,
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
        out: torch.Tensor,
        q8: Optional[torch.Tensor],
        sf: Optional[torch.Tensor],
        num_sf_k_tiles: int,
        decode_query_len: int,
        sm_scale: float,
        linear_softmax: bool,
        use_pdl: bool,
        early_trigger: bool,
    ) -> None:
        """The inputs this op takes and answers wrongly (``m3_attn_lc.lc_applies`` is the target's form of the
        layout half).

        The kernel addresses a page's token rows at a fixed 64-byte (data) and
        8-byte (scale) pitch and loads them 16 / 8 bytes at a time, so only the
        page stride is read from the views. It reads the selected block ids,
        the block table and the hand-off buffers unchecked. Reads ``topk`` and
        ``seq_lens`` back to the host: a test-only cost.
        """
        for name, t in (("k", k), ("v", v)):
            assert t.dim() == 4 and t.shape[1] == 1 and t.shape[2:] == (PAGE, HEAD_DIM // 2), (
                f"{name} must be the [pages, 1, 128, 64] byte view of one KV head"
            )
            assert t.stride(3) == 1 and t.stride(2) == HEAD_DIM // 2, (
                f"{name} rows must be dense 64-byte rows: the kernel ignores the row stride"
            )
            assert t.data_ptr() % 16 == 0 and t.stride(0) % 16 == 0, (
                f"{name} pages must be 16-byte aligned"
            )
        for name, t in (("k_block_scale", k_block_scale), ("v_block_scale", v_block_scale)):
            assert t.dim() == 4 and t.stride(3) == 1 and t.stride(2) == HEAD_DIM // SF_VEC, (
                f"{name} rows must be dense 8-byte rows: the kernel ignores the row stride"
            )
            assert t.data_ptr() % 8 == 0 and t.stride(0) % 8 == 0, (
                f"{name} pages must be 8-byte aligned"
            )
        assert q.data_ptr() % 16 == 0 and q.stride(0) % 16 == 0 and q.stride(1) % 16 == 0, (
            "q rows are loaded 16 bytes at a time and must be 16-byte aligned"
        )
        assert k_global_scale.numel() == 1 and v_global_scale.numel() == 1, (
            "global scales are single values"
        )
        total_q = q.shape[0]
        assert (
            total_q % decode_query_len == 0 and seq_lens.numel() >= total_q // decode_query_len
        ), (
            "every query token must belong to a request: seq_lens is read at tok // decode_query_len unchecked"
        )
        seq = seq_lens.tolist()
        num_cols = block_table.shape[1]
        for tok in range(total_q):
            req, offset = divmod(tok, decode_query_len)
            kv_len = max(seq[req] - decode_query_len + offset + 1, 0)
            nblocks = (kv_len + PAGE - 1) // PAGE
            ids = topk[tok, : min(TOPK, nblocks)]
            assert bool(((ids >= -1) & (ids < min(nblocks, num_cols))).all()), (
                f"token {tok}: selected block ids must be -1 or below the request's block count; the block "
                "table is indexed with them unchecked"
            )
        if q8 is not None:
            assert num_sf_k_tiles == WIDTH // 128, (
                f"num_sf_k_tiles must be {WIDTH // 128}: the hand-off rows are {WIDTH} wide and a smaller "
                "value overlaps them silently"
            )
            assert (
                q8.numel() >= total_q * WIDTH
                and sf is not None
                and sf.numel() >= ((total_q + 127) // 128 * 128 * (WIDTH // MX_BLOCK))
            ), "the hand-off buffers must hold every row of the step: they are written unchecked"


m3_sparse_decode_lc = M3SparseDecodeLc()
