# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA index scoring and top-16 block selection of a pure decode span of at most 2 requests, in one launch."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .m3_index_score import m3_index_score
from .m3_select_long import m3_select_long

#: Requests per launch the kernel takes.
MAX_REQUESTS = 2
#: Query rows per request it takes (decode, or a verify of up to 3 drafts).
MAX_DECODE_QUERY_LEN = 4
#: (KV head x query row) score rows of one request its stage ring holds.
MAX_ROWS = 8
#: Bytes of the stage ring the select scratch shares.
RING_BYTES = 12 * 16384

# M3's checkpoint: no forced initial blocks, the one block holding the query forced in.
_M3 = dict(num_kv_heads=1, init_blocks=0, local_blocks=1)


def ring_fits(rows: int, max_tiles: int) -> bool:
    """Whether a request's select scratch of ``rows`` score rows over ``max_tiles`` blocks fits the stage ring."""
    return rows * ((max_tiles + 3) // 4 * 4) * 4 + rows * 1024 <= RING_BYTES


def scratch_shape(max_tiles: int) -> tuple[int, int, int]:
    """The scratch [requests, rows, L] that serves every request count and geometry the op takes."""
    return (MAX_REQUESTS, MAX_ROWS, (max_tiles + 3) // 4 * 4)


class M3FusedScoreSelect(OpWrapper):
    """Fill ``out`` [total_q, num_kv_heads, 16] with each query row's top-16 KV blocks, straight from the FP8 index Q
    and the paged FP8 index-K cache.

    The table is ``m3_select_long`` (ascending ids, -1 tail, ties to the smaller block, the last
    ``local_blocks`` valid blocks and the first ``init_blocks`` forced in) applied to the per-KV-head maxima over its
    index heads of the ``m3_index_score`` scores, with each row's valid-block count ``n_valid`` clamped to
    ``max_tiles``. Grid (splits, requests): each CTA scores a share of its request's blocks through a 12-stage TMA
    ring with ``mma.sync`` (the products and the 128-term sums of the CuTe DSL scorer, so the same fp32 scores); the
    last CTA of a request (an arrival counter it returns to zero) selects. The request's block-table row and length
    are read and its older index-K pages are issued before the grid wait; only the last CTA releases the PDL
    dependents. ``scratch`` holds the per-request scores between the two phases and needs no initialization;
    ``counters`` must be zero before the first call and stays zero between calls. Returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Eagle3 verify at one request (the TP8 operating point): 4 rows over 20000 tokens",
            spec=dict(batch=1, qlen=4, seq_lens=(20000,), max_tiles=8192, **_M3),
        ),
        Cell(
            why="one request decoding one row over 65536 tokens: 512 blocks across every split",
            spec=dict(batch=1, qlen=1, seq_lens=(65536,), max_tiles=8192, **_M3),
        ),
        Cell(
            why="one request of 129 tokens: two blocks, most splits empty",
            spec=dict(batch=1, qlen=4, seq_lens=(129,), max_tiles=8192, **_M3),
        ),
        Cell(
            why="one request of 200000 tokens: more blocks than one split holds in its 128 local slots",
            spec=dict(batch=1, qlen=4, seq_lens=(200000,), max_tiles=8192, **_M3),
        ),
        Cell(
            why="two requests verifying 4 rows each, of different lengths",
            spec=dict(batch=2, qlen=4, seq_lens=(20000, 3000), max_tiles=8192, **_M3),
        ),
        Cell(
            why="two requests, one of them shorter than 16 blocks: its whole row is selected and -1 padded",
            spec=dict(batch=2, qlen=4, seq_lens=(1500, 70000), max_tiles=8192, **_M3),
        ),
    )

    note = """
    Certified at one index head and one KV head per rank (TP8) with M3's forcing (`init_blocks=0`,
    `local_blocks=1`), FP8 index Q and index-K, a 1M-token engine (`max_tiles=8192`). The selection is checked bit
    for bit against `reference`; scores are compared through the table only. Rows with more than 1024 blocks tied at
    the 16th score are not driven.
    """

    def __call__(
        self,
        idx_q: torch.Tensor,
        idx_k_pages: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        n_valid: torch.Tensor,
        out: torch.Tensor,
        scratch: torch.Tensor,
        counters: torch.Tensor,
        decode_query_len: int,
        num_kv_heads: int,
        init_blocks: int,
        local_blocks: int,
        max_tiles: int,
        use_pdl: bool,
    ) -> torch.Tensor:
        torch.ops.trtllm.minimax_m3_tp8_fused_score_select(
            idx_q,
            idx_k_pages,
            block_table,
            seq_lens,
            n_valid,
            out,
            scratch,
            counters,
            decode_query_len,
            num_kv_heads,
            init_blocks,
            local_blocks,
            max_tiles,
            use_pdl,
        )
        return out

    def reference(
        self,
        idx_q: torch.Tensor,
        idx_k_pages: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        n_valid: torch.Tensor,
        out: torch.Tensor,
        scratch: torch.Tensor,
        counters: torch.Tensor,
        decode_query_len: int,
        num_kv_heads: int,
        init_blocks: int,
        local_blocks: int,
        max_tiles: int,
        use_pdl: bool,
    ) -> torch.Tensor:
        """The int32 [total_q, num_kv_heads, 16] table: ``m3_select_long.reference`` over the per-KV-head maxima of
        ``m3_index_score.reference``. ``out``, ``scratch``, ``counters`` and ``use_pdl`` only say where and how the op
        works."""
        total_q, heads, _ = idx_q.shape
        placeholder = torch.empty(heads, total_q, max_tiles, device=idx_q.device)
        scores = m3_index_score.reference(
            idx_q, idx_k_pages, block_table, seq_lens, placeholder, decode_query_len, 1, False
        )  # [heads, total_q, max_tiles]
        per_kv = scores.view(num_kv_heads, heads // num_kv_heads, total_q, max_tiles).amax(dim=1)
        table = torch.empty(total_q, num_kv_heads, 16, dtype=torch.int32, device=idx_q.device)
        return m3_select_long.reference(
            per_kv.transpose(1, 2), n_valid, table, init_blocks, local_blocks
        )

    def is_valid(
        self,
        idx_q: torch.Tensor,
        idx_k_pages: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        n_valid: torch.Tensor,
        out: torch.Tensor,
        scratch: torch.Tensor,
        counters: torch.Tensor,
        decode_query_len: int,
        num_kv_heads: int,
        init_blocks: int,
        local_blocks: int,
        max_tiles: int,
        use_pdl: bool,
    ) -> None:
        """The inputs this op takes and answers wrongly.

        A non-zero counter makes a request's selection run early, on partial scores. The block-table row is read up
        to ``ceil(seq_len / 128)`` entries unchecked. Reads ``counters`` and ``seq_lens`` back to the host: a test-only
        cost.
        """
        assert bool((counters[: seq_lens.shape[0]] == 0).all()), (
            "counters must be zero on entry: the kernel selects once a request's arrivals reach its split count"
        )
        nblocks = (seq_lens + 127) // 128
        assert bool((nblocks <= block_table.shape[1]).all()), (
            "a request runs past its block-table row"
        )
        assert bool((nblocks <= max_tiles).all()), "a request has more blocks than max_tiles"
        assert init_blocks >= 0 and local_blocks >= 0, "forcing counts must be non-negative"


m3_fused_score_select = M3FusedScoreSelect()
