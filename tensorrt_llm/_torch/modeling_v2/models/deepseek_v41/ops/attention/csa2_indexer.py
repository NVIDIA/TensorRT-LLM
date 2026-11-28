# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexer top-k over the paged compressed index keys, with the two-level candidate-block filter.

Plain torch: the reference the fused indexers (`csa2_indexer_topk_decode`, `csa2_indexer_candidates`) are checked
against. Static shapes, no host sync (CUDA-graph safe).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_pack_rows import csa2_unpack_rows
from .csa2_rows_io import csa2_gather_rows_footer, csa2_row_addr

_TOKENS = 128  # query tokens scored at once


def csa2_candidate_blocks(
    score: torch.Tensor, c_len: torch.Tensor, topk_blocks: int, block: int
) -> torch.Tensor:
    """Level one: keep the `topk_blocks` best blocks of `block` positions per query; bool mask like `score`.

    A block scores as its best position (`-inf` = not reachable). The block holding the newest reachable
    position is pinned, so the most recent tokens always survive. A plain helper of the two indexer
    entries; its test compares it with the reference's `select_candidate_blocks`.
    """
    width = score.shape[-1]
    padded = torch.nn.functional.pad(score, (0, -width % block), value=float("-inf"))
    blocks = padded.unflatten(-1, (-1, block)).amax(dim=-1)
    nb = blocks.shape[-1]
    last = (c_len - 1) // block
    blocks = blocks.masked_fill(torch.arange(nb, device=score.device) == last, float("inf"))
    top = blocks.topk(min(topk_blocks, nb), dim=-1)
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > float("-inf")
    )
    return keep.repeat_interleave(block, dim=-1)[..., :width]


def _picks(s: torch.Tensor, k: int, j: int) -> torch.Tensor:
    """The `k` best positions of scores `s` `[m, j]`, ascending, as int32; a pick of a -inf score (a position past the
    row's compressed length, or outside a consumer's candidates) is -1, placed after the real picks."""
    top = s.topk(k, dim=-1, sorted=False)
    idx = torch.where(top.values > float("-inf"), top.indices, j).sort(dim=-1).values
    return torch.where(idx < j, idx, -1).to(torch.int32)


def _is_valid_common(q, w, pos, index_pool, table, ratio, topk, cand_block, cand_topk, want_cand):
    assert ratio in (1, 2), (
        "the indexer layers have compression ratio 1 or 2 (128 // ratio rows per page)"
    )
    assert q.dim() == 3 and q.shape[-1] == 128 and w.shape == q.shape[:2]
    assert w.dtype == torch.float32, "w is the fp32, already scaled head weights"
    assert index_pool.dtype == torch.uint8 and index_pool.shape[1:] == (128 // ratio, 68), (
        "the index pool is [pages, 128 // ratio, 68] uint8"
    )
    assert index_pool.is_contiguous()
    assert table.dim() == 2 and pos.dim() == 1 and pos.shape[0] == q.shape[0]
    assert topk > 0
    if want_cand:
        assert cand_block > 0 and cand_topk > 0, "want_cand needs cand_block and cand_topk"


class Csa2IndexerTopk(OpWrapper):
    """Per token, the `topk` best compressed positions, ascending; `-1` marks an unreachable pick.

    `q` `[n, h, 128]` bf16 (rotated, quantize-dequantized), `w` `[n, h]` fp32 (already scaled),
    `pos` `[n]`. `segs` lists `(t0, t1, row, span)` per request: its tokens `[t0, t1)`, its table
    row, and an upper bound of its position + 1 (sets the number of scored keys). `index_pool` is
    `[pages, rpp, 68]` uint8 with `rpp = 128 // ratio`; `table` the request's pages. Score of key `j`
    for a query: `sum_h relu(q_h . k_j) * w_h`, reachable when `j < (pos + 1) // ratio`. `cand_in`:
    per-seg candidate masks from the candidate source layer (second level); `want_cand`: also return
    this layer's own candidate masks (`cand_topk` blocks of `cand_block` keys).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="ratio 1: queries from early to the newest position, 100 keys, top-16",
            spec=dict(ratio=1, nkeys=100, qpos=(3, 40, 99, 99, 99), heads=4, topk=16),
        ),
        Cell(
            why="ratio 2: the same with positions halved into compressed rows",
            spec=dict(ratio=2, nkeys=100, qpos=(3, 40, 199, 199, 199), heads=4, topk=16),
        ),
        Cell(
            why="candidate source layer: level-one block mask (8-key blocks, 5 kept) and top-12",
            spec=dict(ratio=1, nkeys=200, qpos=(150, 199, 199), heads=4, topk=12, cand=(8, 5)),
        ),
        Cell(
            why="candidate consumer layer: level two masked by the source layer's blocks",
            spec=dict(
                ratio=1, nkeys=200, qpos=(150, 199, 199), heads=4, topk=12, cand=(8, 5), second=True
            ),
        ),
        Cell(
            why="64 index heads, a 130-token chunk over 2 pages of 64-wide keys (prefill shape)",
            spec=dict(ratio=2, nkeys=140, qpos=tuple(range(150, 280)), heads=64, topk=32),
        ),
        Cell(
            why="fewer reachable keys than top-k: the surplus picks are -1",
            spec=dict(ratio=1, nkeys=30, qpos=(5, 12), heads=4, topk=16),
        ),
    )
    note = """
    One request per call in every cell; `segs` with several requests loop over the same body.
    The scoring einsum runs in the caller's matmul precision: the test pins IEEE fp32 (see its
    band); under torch's TF32 default scores carry ~1e-3 relative noise and a pick near the
    top-k boundary can flip, which only changes which of two near-equal keys is attended.
    """

    def __call__(
        self,
        q: torch.Tensor,
        w: torch.Tensor,
        pos: torch.Tensor,
        segs: Sequence[Tuple[int, int, int, int]],
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        cand_block: int = 0,
        cand_topk: int = 0,
        cand_in: Optional[torch.Tensor] = None,
        want_cand: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        rpp = 128 // ratio
        spans = [-(-s[3] // ratio) for s in segs]
        kmax = min(topk, max(spans))
        n = q.shape[0]
        out = torch.full((n, kmax), -1, dtype=torch.int32, device=q.device)
        cands: List[torch.Tensor] = []
        for si, (t0, t1, row, _) in enumerate(segs):
            j = spans[si]
            keys = csa2_unpack_rows(
                csa2_gather_rows_footer(
                    index_pool,
                    csa2_row_addr(
                        table,
                        torch.full((j,), row, device=q.device),
                        torch.arange(j, device=q.device),
                        rpp,
                    ),
                ),
                "index",
                torch.float32,
            )
            cand_parts = []
            for a in range(t0, t1, _TOKENS):
                b = min(a + _TOKENS, t1)
                s = torch.einsum("nhd,td->nht", q[a:b].float(), keys).relu_() * w[a:b].unsqueeze(-1)
                s = s.sum(dim=1)
                c_len = ((pos[a:b].long() + 1) // ratio).unsqueeze(-1)
                s = s.masked_fill(torch.arange(j, device=q.device) >= c_len, float("-inf"))
                if want_cand:
                    cand_parts.append(csa2_candidate_blocks(s, c_len, cand_topk, cand_block))
                elif cand_in is not None:
                    s = s.masked_fill(~cand_in[si][a - t0 : b - t0], float("-inf"))
                k = min(topk, j)
                out[a:b, :k] = _picks(s, k, j)
            if want_cand:
                cands.append(torch.cat(cand_parts, dim=0))
        return out, (cands if want_cand else None)

    def reference(
        self,
        q: torch.Tensor,
        w: torch.Tensor,
        pos: torch.Tensor,
        segs: Sequence[Tuple[int, int, int, int]],
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        cand_block: int = 0,
        cand_topk: int = 0,
        cand_in: Optional[torch.Tensor] = None,
        want_cand: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """The checkpoint Indexer + `select_candidate_blocks`, dense: every key of the request is
        decoded from the pool by Python address arithmetic, every query scored against all of
        them, and the level-one mask built with the reference's own formulation."""
        rpp = 128 // ratio
        n = q.shape[0]
        kmax = min(topk, max(-(-s[3] // ratio) for s in segs))
        out = torch.full((n, kmax), -1, dtype=torch.int32, device=q.device)
        cands = []
        flat = index_pool.reshape(-1)
        tl = table.tolist()
        for si, (t0, t1, row, span) in enumerate(segs):
            j = -(-span // ratio)
            rows = []
            for key in range(j):
                blk, off = divmod(key, rpp)
                a = tl[row][blk] * rpp + off
                page, i = divmod(a, 64)
                base = page * 64 * 68
                data = flat[base + i * 64 : base + (i + 1) * 64]
                sc = flat[base + 64 * 64 + i * 4 : base + 64 * 64 + (i + 1) * 4]
                rows.append(
                    csa2_unpack_rows.reference(torch.cat([data, sc]), "index", torch.float32)
                )
            keys = torch.stack(rows)
            s = torch.einsum("nhd,td->nht", q[t0:t1].float(), keys).relu() * w[t0:t1].unsqueeze(-1)
            s = s.sum(1)
            c_len = ((pos[t0:t1].long() + 1) // ratio).unsqueeze(-1)
            s = s.masked_fill(torch.arange(j, device=q.device) >= c_len, float("-inf"))
            if want_cand:
                width = s.size(-1)
                sc = torch.nn.functional.pad(s, (0, -width % cand_block), value=-torch.inf)
                sc = sc.unflatten(-1, (-1, cand_block)).amax(dim=-1)
                last = (c_len - 1) // cand_block
                sc = sc.masked_fill(torch.arange(sc.size(-1), device=s.device) == last, torch.inf)
                top = sc.topk(min(cand_topk, sc.size(-1)), dim=-1)
                keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(
                    -1, top.indices, top.values > -torch.inf
                )
                cands.append(keep.repeat_interleave(cand_block, dim=-1)[..., :width])
            elif cand_in is not None:
                s = s.masked_fill(~cand_in[si], float("-inf"))
            k = min(topk, j)
            top = s.topk(k, dim=-1)
            # a pick of a -inf score (past `c_len`, or outside a consumer's candidates) is no pick: -1, after the others
            idx = torch.where(top.values > float("-inf"), top.indices, j).sort(dim=-1).values
            out[t0:t1, :k] = torch.where(idx < j, idx, -1).to(torch.int32)
        return out, (cands if want_cand else None)

    def is_valid(
        self,
        q: torch.Tensor,
        w: torch.Tensor,
        pos: torch.Tensor,
        segs: Sequence[Tuple[int, int, int, int]],
        index_pool: torch.Tensor,
        table: torch.Tensor,
        ratio: int,
        topk: int,
        cand_block: int = 0,
        cand_topk: int = 0,
        cand_in: Optional[torch.Tensor] = None,
        want_cand: bool = False,
    ) -> None:
        _is_valid_common(
            q, w, pos, index_pool, table, ratio, topk, cand_block, cand_topk, want_cand
        )
        assert not (want_cand and cand_in is not None), (
            "a layer is a candidate source or a consumer"
        )
        for t0, t1, row, span in segs:
            assert int(pos[t0:t1].max()) < span, (
                "span must exceed the request's last position: keys past it are never scored"
            )
        if cand_in is not None:
            assert len(cand_in) == len(segs)


csa2_indexer_topk = Csa2IndexerTopk()
