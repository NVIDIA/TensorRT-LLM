# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One pass of the fused greedy Eagle3 draft loop: the drafter's vocab-sharded BF16 head (streamed from its lossless
12-bit coding, ``ops.gemm.m3_target_head``, or as BF16 fragments), the cross-rank argmax and the loop's per-pass
bookkeeping in one kernel (``m3_draft_step.draft_step``)."""

from __future__ import annotations

import functools
from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ..gemm.m3_target_head import decode as decode_coded
from .m3_shard_accept import exchange_tokens, local_argmax, make_keys

#: Requests (head rows) per pass, and exchange rows per call half of the workspace.
MAX_TOKENS = 8
#: 16-row weight tiles one CTA can own (the kernel's shared-memory accumulator).
MAX_TILES_PER_CTA = 32
KERNEL_WARPS = 16  # warps per CTA (draftStep.h kWarps)


@functools.lru_cache(maxsize=64)
def max_tile_pieces(n_tiles: int, num_ctas: int, chunks_per_tile: int) -> int:
    """The most warps that stream one weight tile under the kernel's whole-tile split: CTA c owns tiles
    [c * n / grid, (c + 1) * n / grid), and its (tile, chunk) range is cut into KERNEL_WARPS equal contiguous
    segments. The shared-memory atomics add a tile's pieces in arrival order: deterministic for at most 2 pieces (two
    addends from 0 commute), not for more (fixed_order then sums them in warp order)."""
    worst = 1
    for c in range(num_ctas):
        lo, hi = c * n_tiles // num_ctas, (c + 1) * n_tiles // num_ctas
        span = (hi - lo) * chunks_per_tile
        bounds = [w * span // KERNEL_WARPS for w in range(KERNEL_WARPS + 1)]
        for t in range(hi - lo):
            a, b = t * chunks_per_tile, (t + 1) * chunks_per_tile
            worst = max(
                worst, sum(1 for w in range(KERNEL_WARPS) if bounds[w] < b and bounds[w + 1] > a)
            )
    return worst


#: Draft length of the fused loop, and the drafter's TP8 head shard (200064-id vocabulary) and hidden size.
DRAFT_LEN = 3
DRAFT_SHARD = 25008
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
    """The BF16 head shard [N, K] the kernel multiplies by: the fragment-ordered copy, or (``exc_off`` given) the
    12-bit coded stream with its exception lists decoded."""
    if exc_off is None:
        return from_fragments(wf)
    n = (exc_off.numel() - 1) * 16
    return decode_coded(wf, exp_base, exc_off, exc_pos, exc_w, n, k)


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
    or with ``exc_off`` / ``exc_pos`` / ``exc_w`` / ``exp_base`` its lossless 12-bit coding (``wf`` = the uint8 stream
    of ``m3_target_head.encode_chunked``, which must round-trip to the BF16 shard); each logit is the fp32 dot product
    rounded to BF16 and the row's token is the global argmax over all ranks'
    shards (ties, -0 / +0 included, to the lowest global id), exchanged through the multicast slots
    ``[2][8][world]``. The last CTA then writes ``draft_out[b, step]`` and ``ids_next[b]`` (the token), ``pos_next[b]
    = pos_src[x_rows[b] or b] + 1``, and advances the draft KV lengths as the in-tree draft loop does between passes
    (pass 0 with ``num_acc``: ``kv_lens -= draft_len - num_acc`` and ``seq_lens = 1``; later passes: ``kv_lens +=
    1``). ``out_keys`` (optional) receives each row's winning key.
    """

    ARCHS = frozenset({Arch.SM_107})

    # Every pass of the fused draft loop (modeling's DRAFT_STEP): draft_len 3 at the TP8 GEN worker's 1..2 requests,
    # the drafter's full-vocabulary BF16 head (25008 rows per rank of 8, hidden 6144; 1563 tiles, 8 per CTA on 212
    # SMs). Pass 0 reads the verify rows' hidden (4 per request) through x_rows and rolls the KV back; the later passes
    # read one row per request. BF16 fragments only: on sm_107 the 12-bit coded head is slower than the BF16 stream
    # (modeling's DRAFT_HEAD_CODED is False).
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"pass 0 at {n} request(s): head rows gathered from {4 * n} verify rows, KV rollback, seq_lens = 1",
                spec=dict(
                    nreq=n,
                    draft_len=DRAFT_LEN,
                    step=0,
                    rank=7 * (n - 1),
                    world=8,
                    vs=DRAFT_SHARD,
                    k=HIDDEN,
                    coded=False,
                ),
            )
            for n in (1, 2)
        ),
        Cell(
            why="pass 1 at two requests: one row each, kv_lens + 1, rank 6's ids",
            spec=dict(
                nreq=2,
                draft_len=DRAFT_LEN,
                step=1,
                rank=6,
                world=8,
                vs=DRAFT_SHARD,
                k=HIDDEN,
                coded=False,
            ),
        ),
        Cell(
            why="last pass (step 2) at one request, rank 0",
            spec=dict(
                nreq=1,
                draft_len=DRAFT_LEN,
                step=2,
                rank=0,
                world=8,
                vs=DRAFT_SHARD,
                k=HIDDEN,
                coded=False,
            ),
        ),
        Cell(
            why="pass 2 at 2 requests with exact cross-rank ties (the lower global id wins)",
            spec=dict(
                nreq=2,
                draft_len=DRAFT_LEN,
                step=2,
                rank=3,
                world=8,
                vs=DRAFT_SHARD,
                k=HIDDEN,
                ties=True,
                coded=False,
            ),
        ),
    )

    note = """
    The head is the in-tree BF16 head's arithmetic (BF16 weights and activations, fp32 accumulation, a BF16 logit;
    the coded head's weight is the BF16 shard bit for bit; its exceptions decode to +0 in the MMAs and their exact
    products are added at the tile's flush)
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
        tile_lo: int = 0,
        tile_hi: int = 0,
        total_ctas: int = 0,
        fold: Optional[tuple] = None,
        fixed_order: bool = False,
    ) -> None:
        # fold: (the 14 fold-tail tensors in buildParams' order, rows, tokens per step) -- the step's forward-tail store
        # and next-step staging, run by the pass's last CTA after its tokens (modeling's DRAFT_STEP_FOLD_TAIL).
        tensors, fold_rows, fold_t = (
            (list(fold[0]), int(fold[1]), int(fold[2])) if fold is not None else (None, 0, 0)
        )
        torch.ops.trtllm.minimax_m3_sm107tp8_draft_step(
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
            tile_lo,
            tile_hi,
            total_ctas,
            None,
            None,
            tensors,
            fold_rows,
            fold_t,
            fixed_order,
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
        tile_lo: int = 0,
        tile_hi: int = 0,
        total_ctas: int = 0,
        fold: Optional[tuple] = None,
        fixed_order: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Every buffer the pass writes, computed on copies: "logits" (this rank's BF16 logits as fp32), "local_keys"
        (this rank's key per row), "keys" (the winning key), "tokens", "draft_out", "ids_next", "pos_next", "kv_lens"
        and "seq_lens".

        Reads ``ws_uc``, ``counters`` and ``kv_lens`` before the call. ``keys``, ``ws_mc_ptr``, ``num_ctas``, ``pdl``,
        ``out_keys`` and ``fixed_order`` do not change the result; ``fold`` adds the fold tail's writes (m3_fold_tail),
        not these.
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
        tile_lo: int = 0,
        tile_hi: int = 0,
        total_ctas: int = 0,
        fold: Optional[tuple] = None,
        fixed_order: bool = False,
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
        # D-FIX: the atomics' arrival-order sum is deterministic only while no tile takes more than 2 warp pieces.
        assert (
            fixed_order
            or max_tile_pieces(tiles, num_ctas, x.shape[1] // (16 if exc_off is None else 64)) <= 2
        ), (
            "a tile takes more than 2 warp pieces on this grid: the atomics' sum is order-dependent (fixed_order)"
        )
        if exc_off is not None:
            # The exceptions are read per tile in list order with no bound against the coded stream.
            assert int(exc_off[-1].item()) == exc_pos.numel(), (
                "exc_off must end at the exception count"
            )
            assert bool((exc_off[1:] >= exc_off[:-1]).all()), "exc_off must be non-decreasing"
        # A NaN logit maps to a key above every finite one and wins the row silently.
        h = x[x_rows.long()] if x_rows is not None else x[:rows]
        assert not bool(torch.isnan(h).any()), "NaN head inputs win the argmax"
        if fold is not None:
            assert len(fold) == 3 and len(fold[0]) == 14, (
                "fold is (the 14 fold-tail tensors, rows, tokens per step)"
            )
        if num_acc is not None:
            na = num_acc[:rows]
            assert bool(((na >= 1) & (na <= draft_len + 1)).all()), (
                "num_acc counts must lie in [1, draft_len + 1]"
            )


m3_draft_step = M3DraftStep()
