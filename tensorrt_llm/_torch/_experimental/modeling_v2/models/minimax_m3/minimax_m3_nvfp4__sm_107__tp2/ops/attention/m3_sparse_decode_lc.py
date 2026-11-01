# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch long-context sparse decode over the selected NVFP4 blocks (16-CTA clusters), SM107 TP2 geometry.

Also the plain-torch block-sparse attention and MXFP8 hand-off mirrors the
Triton sparse decode entry (``m3_sparse_decode``) reuses: both compute the same
thing over the same cache.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ....ops.attention.sparse_reference import _E2M1 as _E2M1
from ....ops.attention.sparse_reference import dequant_nvfp4_pages as _dequant_nvfp4_pages
from ....ops.attention.sparse_reference import mxfp8_handoff_reference as _mxfp8_handoff_reference
from ....ops.attention.sparse_reference import (
    sparse_attention_reference as _sparse_attention_reference,
)

#: The TP2 geometry the kernel is compiled for: 16 query heads on each of 2 KV heads per rank.
HEADS_PER_KV = 16
KV_HEADS = 2
NUM_HEADS = HEADS_PER_KV * KV_HEADS
HEAD_DIM = 128
PAGE = 128
TOPK = 16
#: NVFP4 elements per E4M3 block scale.
SF_VEC = 16
#: MXFP8 elements per UE8M0 scale, and the o_proj width the hand-off is laid out for.
MX_BLOCK = 32
WIDTH = NUM_HEADS * HEAD_DIM


def dequant_nvfp4_pages(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    pages: torch.Tensor,
    quad_scales: bool,
    kv_head: int = 0,
) -> torch.Tensor:
    """Use the shared reference with this target's geometry."""
    return _dequant_nvfp4_pages(packed, block_scale, pages, quad_scales, kv_head)


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
    """Use the shared reference with this target's geometry."""
    return _sparse_attention_reference(
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
        decode_query_len,
        sm_scale,
        kv_head,
    )


def mxfp8_handoff_reference(
    out: torch.Tensor, num_rows_alloc: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use the shared reference with this target's geometry."""
    return _mxfp8_handoff_reference(out, num_rows_alloc, width=WIDTH)


class M3SparseDecodeLc(OpWrapper):
    """Sparse decode attention, split-K merge and MXFP8 hand-off of one TP2 decode step in one launch.

    One cluster of 16 CTAs per (query token, KV head): CTA ``r`` attends the
    token's ``r``-th selected block of that head (``topk[h, tok, r]``), and the
    cluster merges the 16 partials over distributed shared memory into the
    head's 16 BF16 rows of ``out[tok]``. With ``q8`` / ``sf`` it also writes
    the row's MXFP8 copy and swizzled UE8M0 scales for the o_proj, bitwise
    ``torch.ops.trtllm.mxfp8_quantize(out, True)``. All operands are the byte
    views ``m3_attn_lc.m3_lc_attn`` hands over (E4M3 ``q``, packed E2M1 pages,
    E4M3 block scales). Returns ``out``.

    Per block: NVFP4 K/V dequantized to FP16 (exact), FP8 q widened to FP16
    (exact), QK and PV on FP16 tensor cores with FP32 accumulation, the softmax
    probabilities rounded to FP16 for PV, log2-domain softmax with
    ``ex2.approx``. ``linear_softmax`` picks the denominator recurrence (the two
    agree to rounding); SM107 keeps the log-domain recurrence of the stock
    Triton producer. ``prewait`` stages the request's block-table row in shared
    memory before the grid dependency wait; it does not change a value.
    ``n1_ids`` (int32 [2, rows, 16], dense rows, initialized to -1) holds the
    layer's selected block ids one decode step back: each CTA prefetches its
    slot's block into L2 before the grid dependency wait and overwrites the slot
    with this step's id (-1 past the token's block count); rows at or past
    ``rows`` skip both. A hint: it does not change a value.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Eagle3 verify at one request, 165k context: T=4, log-domain recurrence, MXFP8 hand-off",
            spec=dict(
                seq_lens=(165521,),
                decode_query_len=4,
                linear_softmax=False,
                mxfp8=True,
                prewait=False,
            ),
        ),
        Cell(
            why="Eagle3 verify at two requests, long and medium context: T=8, no hand-off",
            spec=dict(
                seq_lens=(165521, 98765),
                decode_query_len=4,
                linear_softmax=False,
                mxfp8=False,
                prewait=False,
            ),
        ),
        Cell(
            why="Eagle3 verify at two requests: T=8 with the MXFP8 hand-off",
            spec=dict(
                seq_lens=(165521, 12345),
                decode_query_len=4,
                linear_softmax=False,
                mxfp8=True,
                prewait=False,
            ),
        ),
        Cell(
            why="two short requests with -1 tails and a zero-length padding request: T=8",
            spec=dict(
                seq_lens=(130, 0),
                decode_query_len=4,
                linear_softmax=False,
                mxfp8=True,
                prewait=False,
            ),
        ),
        Cell(
            why="decode without speculation at two requests: T=2",
            spec=dict(
                seq_lens=(165521, 5000),
                decode_query_len=1,
                linear_softmax=False,
                mxfp8=True,
                prewait=False,
            ),
        ),
        Cell(
            why="the wide target's largest hand-off step: 32 requests, T=128 (one full scale tile), mixed lengths "
            "with tile edges and a zero-length padding request, MXFP8 hand-off",
            spec=dict(
                seq_lens=tuple((165521, 20000, 1281, 129)[i % 4] for i in range(31)) + (0,),
                decode_query_len=4,
                linear_softmax=False,
                mxfp8=True,
                prewait=False,
            ),
        ),
        Cell(
            why="the linear denominator and pre-wait staging variants, driven at T=8",
            spec=dict(
                seq_lens=(165521, 20000),
                decode_query_len=4,
                linear_softmax=True,
                mxfp8=True,
                prewait=True,
            ),
        ),
    )

    note = """
    Certified: the eight instantiated variants (`linear_softmax` x `prewait` x
    hand-off on / off). The hand-off cells are certified bit-exactly against
    `mxfp8_handoff_reference` applied to the op's own BF16 rows, which equals
    `torch.ops.trtllm.mxfp8_quantize(out, True)`; the BF16 rows themselves within
    the band the test derives. `prewait` and the hand-off do not change the BF16
    rows.

    `topk` is the head-major [2, T, 16] table; it is read for the first
    `min(16, ceil(kv_len / 128))` entries of each token; a -1 among them skips
    that CTA's block, entries past them are never read. `use_pdl` only changes
    the launch attribute. The hand-off assumes `num_sf_k_tiles = 32` (a
    4096-wide o_proj input) and buffers with at least as many rows as the step;
    `MERGE_MXFP8` sizes them at 128 rows.
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
        prewait: bool,
        n1_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch.ops.trtllm.minimax_m3_sm107_tp2_sparse_decode_lc(
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
            prewait,
            n1_ids,
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
        prewait: bool,
        n1_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The BF16 ``out`` [T, 4096] the op is supposed to write: ``sparse_attention_reference`` per KV head in
        float64, rounded once. ``linear_softmax``, ``use_pdl``, ``prewait`` and ``n1_ids`` do not change what is
        computed; the hand-off buffers are checked separately with ``mxfp8_handoff_reference``.
        """
        q3 = q.view(-1, NUM_HEADS, HEAD_DIM)
        attn = torch.cat(
            [
                sparse_attention_reference(
                    q3[:, h * HEADS_PER_KV : (h + 1) * HEADS_PER_KV],
                    k,
                    v,
                    k_block_scale,
                    v_block_scale,
                    k_global_scale,
                    v_global_scale,
                    topk[h],
                    block_table,
                    seq_lens,
                    decode_query_len,
                    sm_scale,
                    kv_head=h,
                )
                for h in range(KV_HEADS)
            ],
            dim=1,
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
        prewait: bool,
        n1_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """The inputs this op takes and answers wrongly (``m3_attn_lc.lc_applies`` is the target's form of the
        layout half).

        The kernel addresses a page's token rows at a fixed 64-byte (data) and
        8-byte (scale) pitch and loads them 16 / 8 bytes at a time, so only the
        page and KV-head strides are read from the views. It reads the selected
        block ids, the block table and the hand-off buffers unchecked. Reads
        ``topk`` and ``seq_lens`` back to the host: a test-only cost.
        """
        for name, t in (("k", k), ("v", v)):
            assert (
                t.dim() == 4 and t.shape[1] == KV_HEADS and t.shape[2:] == (PAGE, HEAD_DIM // 2)
            ), f"{name} must be the [pages, 2, 128, 64] byte view of the rank's KV heads"
            assert t.stride(3) == 1 and t.stride(2) == HEAD_DIM // 2, (
                f"{name} rows must be dense 64-byte rows: the kernel ignores the row stride"
            )
            assert t.data_ptr() % 16 == 0 and t.stride(0) % 16 == 0 and t.stride(1) % 16 == 0, (
                f"{name} pages and heads must be 16-byte aligned"
            )
        assert v.stride() == k.stride(), "k and v share their strides"
        for name, t in (("k_block_scale", k_block_scale), ("v_block_scale", v_block_scale)):
            assert (
                t.dim() == 4
                and t.shape[1] == KV_HEADS
                and t.stride(3) == 1
                and t.stride(2) == HEAD_DIM // SF_VEC
            ), f"{name} rows must be dense 8-byte rows: the kernel ignores the row stride"
            assert t.data_ptr() % 8 == 0 and t.stride(0) % 8 == 0 and t.stride(1) % 8 == 0, (
                f"{name} pages and heads must be 8-byte aligned"
            )
        assert q.dim() == 3 and q.shape[1:] == (NUM_HEADS, HEAD_DIM), "q must be [T, 32, 128]"
        assert q.data_ptr() % 16 == 0 and q.stride(0) % 16 == 0 and q.stride(1) % 16 == 0, (
            "q rows are loaded 16 bytes at a time and must be 16-byte aligned"
        )
        assert k_global_scale.numel() == 1 and v_global_scale.numel() == 1, (
            "global scales are single values"
        )
        total_q = q.shape[0]
        assert topk.dim() == 3 and topk.shape[0] == KV_HEADS and topk.shape[1] == total_q, (
            "topk is the head-major [2, T, 16] table"
        )
        assert (
            total_q % decode_query_len == 0 and seq_lens.numel() >= total_q // decode_query_len
        ), (
            "every query token must belong to a request: seq_lens is read at tok // decode_query_len unchecked"
        )
        seq = seq_lens.tolist()
        num_cols = block_table.shape[1]
        for h in range(KV_HEADS):
            for tok in range(total_q):
                req, offset = divmod(tok, decode_query_len)
                kv_len = max(seq[req] - decode_query_len + offset + 1, 0)
                nblocks = (kv_len + PAGE - 1) // PAGE
                ids = topk[h, tok, : min(TOPK, nblocks)]
                assert bool(((ids >= -1) & (ids < min(nblocks, num_cols))).all()), (
                    f"head {h} token {tok}: selected block ids must be -1 or below the request's block count; "
                    "the block table is indexed with them unchecked"
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
