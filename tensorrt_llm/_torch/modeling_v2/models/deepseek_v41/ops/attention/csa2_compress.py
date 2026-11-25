# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ratio-2 softmax-gated pooling of a token pair, with the pair-straddling state kept in a paged pool.

Plain torch. Static shapes, no host sync: safe under CUDA-graph capture.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_rows_io import csa2_gather_rows, csa2_row_addr, csa2_scatter_rows

_ROWS = 128  # rows per page of the compressor-state pools (one row per token position)


class Csa2CompressPairs(OpWrapper):
    """Pool every token's pair `(2j, 2j+1)` (j = pos // 2) into one fp32 latent row `[n, d]`.

    `kv`/`score` are the token's value and gate projections (fp32 `[n, d]`). `cs`/`ce` are the first
    and one-past-last position of the token's request in this step. The latent of the pair is
    `sum_i kv_i * softmax_i(score)` per channel. The partner token comes from this step when it is
    inside `[cs, ce)`, from the state pool when it precedes `cs`, and is absent (weight 0) when it
    lies at or past `ce` -- that latent is incomplete and must not be read until the pair completes.
    Every token of the step is written to the state pools (in place) for later steps, as the engine's
    compressor keeps every source row: a speculative step may restart at an odd position inside the
    previous step (rejected drafts), and its partner must be the accepted row written there.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="one request prefilled whole, odd length (the last pair is incomplete)",
            spec=dict(d=512, reqs=((11,),)),
        ),
        Cell(
            why="decode: one token per step, so every pair straddles two steps",
            spec=dict(d=512, reqs=((1,) * 9,)),
        ),
        Cell(
            why="chunked prefill with odd chunk sizes: pairs straddle chunk edges",
            spec=dict(d=512, reqs=((3, 1, 4, 1, 2),)),
        ),
        Cell(
            why="speculative verify: a 6-token step (1 + 5 drafts) after a prefill",
            spec=dict(d=512, reqs=((5, 6, 6, 6),)),
        ),
        Cell(
            why="two requests in one step with different lengths, one with an odd start",
            spec=dict(d=512, reqs=((6, 1), (4, 3))),
        ),
        Cell(
            why="a request long enough to cross the 128-row state page",
            spec=dict(d=64, reqs=((130, 5),)),
        ),
    )
    note = """
    Tokens of one request must be adjacent and in ascending position inside the step (the
    partner of a token is read from the row next to it); `is_valid` states it. The ratio is
    fixed at 2; the ratio-1 layers use a plain projection and no entry.
    """

    def __call__(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        cs: torch.Tensor,
        ce: torch.Tensor,
        state_kv: torch.Tensor,
        state_score: torch.Tensor,
        table_kv: torch.Tensor,
        table_score: torch.Tensor,
    ) -> torch.Tensor:
        n = kv.shape[0]
        keep = torch.ones_like(pos, dtype=torch.bool)
        for pool, tbl, val in ((state_kv, table_kv, kv), (state_score, table_score, score)):
            csa2_scatter_rows(pool, csa2_row_addr(tbl, rid, pos, _ROWS), val, keep)
        partner = pos ^ 1
        delta = (partner - pos).long()  # +1 for even positions, -1 for odd
        inflight = (partner >= cs) & (partner < ce)
        row = (torch.arange(n, device=kv.device) + delta).clamp(0, n - 1)
        old = partner < cs
        paddr = csa2_row_addr(table_kv, rid, partner, _ROWS)
        saddr = csa2_row_addr(table_score, rid, partner, _ROWS)
        pk = torch.where(
            inflight.unsqueeze(-1), kv[row], csa2_gather_rows(state_kv, torch.where(old, paddr, -1))
        )
        ps = torch.where(
            inflight.unsqueeze(-1),
            score[row],
            csa2_gather_rows(state_score, torch.where(old, saddr, -1)),
        )
        ps = torch.where((inflight | old).unsqueeze(-1), ps, torch.full_like(ps, float("-inf")))
        own_first = (pos & 1) == 0
        a_kv = torch.where(own_first.unsqueeze(-1), kv, pk)
        b_kv = torch.where(own_first.unsqueeze(-1), pk, kv)
        a_s = torch.where(own_first.unsqueeze(-1), score, ps)
        b_s = torch.where(own_first.unsqueeze(-1), ps, score)
        m = torch.maximum(a_s, b_s)
        ea, eb = torch.exp(a_s - m), torch.exp(b_s - m)
        return (a_kv * ea + b_kv * eb) / (ea + eb)

    def reference(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        cs: torch.Tensor,
        ce: torch.Tensor,
        state_kv: torch.Tensor,
        state_score: torch.Tensor,
        table_kv: torch.Tensor,
        table_score: torch.Tensor,
    ) -> torch.Tensor:
        """Token by token: look the partner up by (request, position) in this step, else in the
        state pool, else drop it; pool the two with `softmax` over the stacked pair in float64.
        Writes every token of the step to the state pools (pass copies)."""
        n = kv.shape[0]
        pl, rl, csl, cel = pos.tolist(), rid.tolist(), cs.tolist(), ce.tolist()
        step_row = {(rl[i], pl[i]): i for i in range(n)}
        tk, ts = table_kv.tolist(), table_score.tolist()

        def state_row(pool, table, r, p):
            blk, off = divmod(p, _ROWS)
            page = table[r][blk]
            return pool.view(-1, pool.shape[-1])[page * _ROWS + off]

        for i in range(n):  # every token of the step
            blk, off = divmod(pl[i], _ROWS)
            state_kv.view(-1, kv.shape[-1])[tk[rl[i]][blk] * _ROWS + off] = kv[i]
            state_score.view(-1, kv.shape[-1])[ts[rl[i]][blk] * _ROWS + off] = score[i]
        out = torch.empty(n, kv.shape[-1], dtype=torch.float32, device=kv.device)
        for i in range(n):
            p, r = pl[i], rl[i]
            q = p ^ 1
            if csl[i] <= q < cel[i]:
                qk, qs = kv[step_row[(r, q)]], score[step_row[(r, q)]]
            elif q < csl[i]:
                qk, qs = state_row(state_kv, tk, r, q), state_row(state_score, ts, r, q)
            else:
                qk, qs = torch.zeros_like(kv[i]), torch.full_like(score[i], float("-inf"))
            pair_kv = torch.stack([kv[i], qk] if p % 2 == 0 else [qk, kv[i]]).double()
            pair_s = torch.stack([score[i], qs] if p % 2 == 0 else [qs, score[i]]).double()
            out[i] = (pair_kv * pair_s.softmax(dim=0)).sum(dim=0).float()
        return out

    def is_valid(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        cs: torch.Tensor,
        ce: torch.Tensor,
        state_kv: torch.Tensor,
        state_score: torch.Tensor,
        table_kv: torch.Tensor,
        table_score: torch.Tensor,
    ) -> None:
        assert kv.dtype == torch.float32 and score.dtype == torch.float32, (
            "kv / score must be fp32: the state pools are fp32 and a lower precision pairs "
            "a rounded token with an exact partner"
        )
        assert state_kv.dtype == torch.float32 and state_score.dtype == torch.float32
        assert pos.dtype in (torch.int32, torch.int64)
        assert bool(((pos >= cs) & (pos < ce)).all()), (
            "every token must lie in its request's [cs, ce)"
        )
        runs = 1 + int((rid[1:] != rid[:-1]).sum()) if rid.numel() else 0
        assert runs == rid.unique().numel(), (
            "the tokens of one request must form one run (request-major rows): a recurring request id pairs a row "
            "with another request's"
        )
        same = rid[1:] == rid[:-1]
        assert bool((pos[1:][same] == pos[:-1][same] + 1).all()), (
            "the tokens of one request must be adjacent and ascending in the step"
        )


csa2_compress_pairs = Csa2CompressPairs()
