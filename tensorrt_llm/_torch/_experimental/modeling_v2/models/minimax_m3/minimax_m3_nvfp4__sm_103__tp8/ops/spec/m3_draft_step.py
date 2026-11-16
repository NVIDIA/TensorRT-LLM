# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One pass of the fused greedy Eagle3 draft loop: the drafter's vocab-sharded BF16 head (streamed as BF16 fragments),
the cross-rank argmax and the loop's per-pass bookkeeping in one kernel (``m3_draft_step.draft_step``)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from .m3_shard_accept import exchange_tokens, local_argmax, make_keys

#: Requests (head rows) per pass, and exchange rows per call half of the workspace.
MAX_TOKENS = 8
#: 16-row weight tiles one CTA can own (the kernel's shared-memory accumulator).
MAX_TILES_PER_CTA = 32
#: Draft length of the fused loop, and the drafter's TP8 head shard (200064-id vocabulary) and hidden size.
DRAFT_LEN = 3
DRAFT_SHARD = 25008
#: Ranks of the TP group the argmax is exchanged over.
WORLD = 8
HIDDEN = 6144


def bf16_fragments(weight: torch.Tensor) -> torch.Tensor:
    """BF16 [N, K] -> the same values in A-fragment order [N / 16, K / 16, 256].

    One lane's 16 bytes of a 16-row tile's k16 step are the m16n8k16 A fragment: element (row h * 8 + g, k ks * 16 +
    hk * 8 + jj * 2 + e) goes to [tile][ks][g][jj][hk][h][e] (lane = g * 4 + jj).
    """
    n, k = weight.shape
    assert n % 16 == 0 and k % 16 == 0 and weight.dtype == torch.bfloat16, (n, k, weight.dtype)
    q = weight.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()
    return q.view(n // 16, k // 16, 256)


def from_fragments(wf: torch.Tensor) -> torch.Tensor:
    """The inverse of ``bf16_fragments``: [N / 16, K / 16, 256] -> [N, K]."""
    tiles, ks, _ = wf.shape
    return (
        wf.view(tiles, ks, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(tiles * 16, ks * 16)
    )


def head_weight(
    wf: torch.Tensor,
    k: int,
    exc_off: Optional[torch.Tensor] = None,
    exc_pos: Optional[torch.Tensor] = None,
    exc_w: Optional[torch.Tensor] = None,
    exp_base: int = 0,
) -> torch.Tensor:
    """The BF16 head shard [N, K] the kernel multiplies by: the fragment-ordered copy. The op also streams a 12-bit
    coded head (``exc_off`` given); this target never builds one, and nothing here certifies it."""
    if exc_off is not None:
        raise NotImplementedError("the tp8 target streams the BF16 fragments only")
    return from_fragments(wf)


def shard_logits(
    x: torch.Tensor,
    wf: torch.Tensor,
    exc_off: Optional[torch.Tensor] = None,
    exc_pos: Optional[torch.Tensor] = None,
    exc_w: Optional[torch.Tensor] = None,
    exp_base: int = 0,
) -> torch.Tensor:
    """The in-tree head's BF16 logits of the rows x [M, K] against the shard (``head_weight``): the fp32 product,
    rounded once to BF16 (the BF16 GEMM's output), as fp32."""
    w = head_weight(wf, x.shape[1], exc_off, exc_pos, exc_w, exp_base)
    with true_fp32_matmul():
        logits = x.float() @ w.float().t()
    return logits.to(torch.bfloat16).float()


class M3DraftStep(OpWrapper):
    """One fused greedy draft pass over ``rows <= 8`` requests (one head row each).

    Head input ``x[x_rows[b]]`` (pass 0: the last accepted verify row of each request) or ``x[b]``. The head is this
    rank's BF16 lm_head shard in fragment order (``m3_draft_step.bf16_fragments``, global ids ``vocab_offset ..``),
    (the op also takes a 12-bit coded stream with ``exc_off`` / ``exc_pos`` / ``exc_w`` / ``exp_base``, which this
    target never builds); each logit is the fp32 dot product
    rounded to BF16 and the row's token is the global argmax over all ranks'
    shards (ties, -0 / +0 included, to the lowest global id), exchanged through the multicast slots
    ``[2][8][world]``. The last CTA then writes ``draft_out[b, step]`` and ``ids_next[b]`` (the token), ``pos_next[b]
    = pos_src[x_rows[b] or b] + 1``, and advances the draft KV lengths as the in-tree draft loop does between passes
    (pass 0 with ``num_acc``: ``kv_lens -= draft_len - num_acc`` and ``seq_lens = 1``; later passes: ``kv_lens +=
    1``). ``out_keys`` (optional) receives each row's winning key.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Every pass of the fused draft loop (dispatch.DRAFT_STEP): draft_len 3 at 1..8 requests, the drafter's
    # full-vocabulary BF16 head (25008 rows per rank of 8, hidden 6144). Pass 0 reads the verify rows' hidden (4 per
    # request) through x_rows and rolls the KV back; the later passes read one row per request.
    _PASSES: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"pass 0 at {n} request(s): head rows gathered from {4 * n} verify rows, KV rollback, seq_lens = 1",
                spec=dict(
                    nreq=n,
                    draft_len=DRAFT_LEN,
                    step=0,
                    rank=n % WORLD,
                    world=WORLD,
                    vs=DRAFT_SHARD,
                    k=HIDDEN,
                ),
            )
            for n in range(1, MAX_TOKENS + 1)
        ),
        Cell(
            why="pass 1 at two requests: one row each, kv_lens + 1, rank 7's ids",
            spec=dict(
                nreq=2, draft_len=DRAFT_LEN, step=1, rank=7, world=WORLD, vs=DRAFT_SHARD, k=HIDDEN
            ),
        ),
        Cell(
            why="last pass (step 2) at 8 requests, rank 0",
            spec=dict(
                nreq=8, draft_len=DRAFT_LEN, step=2, rank=0, world=WORLD, vs=DRAFT_SHARD, k=HIDDEN
            ),
        ),
        Cell(
            why="pass 2 at 5 requests with exact cross-rank ties (the lower global id wins)",
            spec=dict(
                nreq=5,
                draft_len=DRAFT_LEN,
                step=2,
                rank=5,
                world=WORLD,
                vs=DRAFT_SHARD,
                k=HIDDEN,
                ties=True,
            ),
        ),
    )
    # The BF16 fragments at every pass shape.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"bf16: {c.why}", spec=dict(c.spec, coded=False)) for c in _PASSES
    )

    note = """
    The head is the in-tree BF16 head's arithmetic (BF16 weights and activations, fp32 accumulation, a BF16 logit)
    and the argmax is the in-tree one over BF16 logits, so a token differs from the in-tree path's only where the fp32
    sum, accumulated in another order, rounds to the other side of a BF16 boundary and that decides the row's maximum.
    The cells' rows are decidable and the peers' keys sit 0.25 away (or tie exactly); the edge tests (forced BF16
    ties, -0 / +0, top-2 one fp32 step apart that round to one BF16 value, adjacent BF16 values) use integer data
    whose fp32 sums are exact in any order and hold the kernel to the in-tree path bit for bit. The cross-rank
    exchange is certified on one device with the peers' keys written into the slots (see m3_shard_accept).
    """

    def __call__(
        self,
        x: torch.Tensor,
        x_rows: Optional[torch.Tensor],
        rows: int,
        wf: torch.Tensor,
        vocab_offset: int,
        keys: torch.Tensor,
        counters: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        rank: int,
        world: int,
        draft_out: torch.Tensor,
        step: int,
        ids_next: torch.Tensor,
        pos_src: torch.Tensor,
        pos_next: torch.Tensor,
        kv_lens: torch.Tensor,
        num_acc: Optional[torch.Tensor],
        seq_lens: Optional[torch.Tensor],
        draft_len: int,
        num_ctas: int,
        pdl: bool = True,
        out_keys: Optional[torch.Tensor] = None,
        exc_off: Optional[torch.Tensor] = None,
        exc_pos: Optional[torch.Tensor] = None,
        exc_w: Optional[torch.Tensor] = None,
        exp_base: int = 0,
    ) -> None:
        torch.ops.trtllm.minimax_m3_draft_step(
            x,
            x_rows,
            rows,
            wf,
            vocab_offset,
            keys,
            counters,
            ws_uc,
            ws_mc_ptr,
            rank,
            world,
            draft_out,
            step,
            ids_next,
            pos_src,
            pos_next,
            kv_lens,
            num_acc,
            seq_lens,
            draft_len,
            num_ctas,
            pdl,
            out_keys,
            exc_off,
            exc_pos,
            exc_w,
            exp_base,
        )

    def reference(
        self,
        x: torch.Tensor,
        x_rows: Optional[torch.Tensor],
        rows: int,
        wf: torch.Tensor,
        vocab_offset: int,
        keys: torch.Tensor,
        counters: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        rank: int,
        world: int,
        draft_out: torch.Tensor,
        step: int,
        ids_next: torch.Tensor,
        pos_src: torch.Tensor,
        pos_next: torch.Tensor,
        kv_lens: torch.Tensor,
        num_acc: Optional[torch.Tensor],
        seq_lens: Optional[torch.Tensor],
        draft_len: int,
        num_ctas: int,
        pdl: bool = True,
        out_keys: Optional[torch.Tensor] = None,
        exc_off: Optional[torch.Tensor] = None,
        exc_pos: Optional[torch.Tensor] = None,
        exc_w: Optional[torch.Tensor] = None,
        exp_base: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Every buffer the pass writes, computed on copies: "logits" (this rank's BF16 logits as fp32), "local_keys"
        (this rank's key per row), "keys" (the winning key), "tokens", "draft_out", "ids_next", "pos_next", "kv_lens"
        and "seq_lens".

        Reads ``ws_uc``, ``counters`` and ``kv_lens`` before the call. ``keys``, ``ws_mc_ptr``, ``num_ctas``, ``pdl``
        and ``out_keys`` do not change the result.
        """
        h = x[x_rows.long()] if x_rows is not None else x[:rows]
        logits = shard_logits(h, wf, exc_off, exc_pos, exc_w, exp_base)
        best, idx = local_argmax(logits, vocab_offset)
        local_keys = make_keys(best, idx)
        tokens = exchange_tokens(local_keys, ws_uc, counters, rank, world, MAX_TOKENS)
        half = int(counters[1].item()) & 1
        slots = ws_uc.view(torch.int64)[
            half * MAX_TOKENS * world : (half * MAX_TOKENS + rows) * world
        ]
        slots = slots.view(rows, world).clone()
        slots[:, rank] = local_keys
        sign = torch.iinfo(torch.int64).min
        win = (slots ^ sign).max(dim=1).values ^ sign
        d_out = draft_out.clone()
        d_out[:rows, step] = tokens
        src = x_rows.long() if x_rows is not None else torch.arange(rows, device=x.device)
        lens = kv_lens.clone()
        sl = seq_lens.clone() if seq_lens is not None else None
        if step == 0 and num_acc is not None:
            lens[:rows] -= draft_len - num_acc[:rows]
            sl[:rows] = 1
        elif step > 0:
            lens[:rows] += 1
        out = {
            "logits": logits,
            "local_keys": local_keys,
            "keys": win,
            "tokens": tokens,
            "draft_out": d_out,
            "ids_next": tokens.clone(),
            "pos_next": pos_src[src] + 1,
            "kv_lens": lens,
        }
        if sl is not None:
            out["seq_lens"] = sl
        return out

    def is_valid(
        self,
        x: torch.Tensor,
        x_rows: Optional[torch.Tensor],
        rows: int,
        wf: torch.Tensor,
        vocab_offset: int,
        keys: torch.Tensor,
        counters: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        rank: int,
        world: int,
        draft_out: torch.Tensor,
        step: int,
        ids_next: torch.Tensor,
        pos_src: torch.Tensor,
        pos_next: torch.Tensor,
        kv_lens: torch.Tensor,
        num_acc: Optional[torch.Tensor],
        seq_lens: Optional[torch.Tensor],
        draft_len: int,
        num_ctas: int,
        pdl: bool = True,
        out_keys: Optional[torch.Tensor] = None,
        exc_off: Optional[torch.Tensor] = None,
        exc_pos: Optional[torch.Tensor] = None,
        exc_w: Optional[torch.Tensor] = None,
        exp_base: int = 0,
    ) -> None:
        assert not bool(keys[:rows].any()), "keys must start zeroed"
        assert int(counters[0].item()) == 0, (
            "counters[0] (finished CTAs) must be zero between calls"
        )
        # The kernel reads the head row x + x_rows[b] * K and the position pos_src[x_rows[b]] with no bound.
        if x_rows is not None:
            assert bool(((x_rows[:rows] >= 0) & (x_rows[:rows] < x.shape[0])).all()), (
                "x_rows must index x"
            )
            assert bool((x_rows[:rows] < pos_src.shape[0]).all()), (
                "pos_src must cover the head rows"
            )
        else:
            assert pos_src.shape[0] >= rows, "pos_src must cover the head rows"
        # Each CTA accumulates at most MAX_TILES_PER_CTA tiles in shared memory.
        tiles = wf.shape[0] if exc_off is None else exc_off.numel() - 1
        assert -(-tiles // num_ctas) <= MAX_TILES_PER_CTA, "too many weight tiles per CTA"
        if exc_off is not None:
            # The exceptions are read per tile in list order with no bound against the coded stream.
            assert int(exc_off[-1].item()) == exc_pos.numel(), (
                "exc_off must end at the exception count"
            )
            assert bool((exc_off[1:] >= exc_off[:-1]).all()), "exc_off must be non-decreasing"
        # A NaN logit maps to a key above every finite one and wins the row silently.
        h = x[x_rows.long()] if x_rows is not None else x[:rows]
        assert not bool(torch.isnan(h).any()), "NaN head inputs win the argmax"
        if num_acc is not None:
            na = num_acc[:rows]
            assert bool(((na >= 1) & (na <= draft_len + 1)).all()), (
                "num_acc counts must lie in [1, draft_len + 1]"
            )


m3_draft_step = M3DraftStep()
