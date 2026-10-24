# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampled (temperature + top-p) Eagle3 verification over the TP-sharded vocabulary in one kernel
(``m3_sampled_accept.sampled_accept``)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .m3_shard_accept import strict_accept

#: Verify rows per call.
MAX_ROWS = 48
#: The fused sampling kernel's 1 / T guard.
TEMP_EPSILON = 1e-6


def nucleus_probs(
    logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor
) -> torch.Tensor:
    """Per-row sampling distribution [rows, Vs] (fp32): w = exp(x / (T + 1e-6) - max), kept iff the mass strictly
    above its weight is at most top_p * sum(w) (ties included), renormalized over the kept tokens."""
    inv = 1.0 / (temperatures.float() + TEMP_EPSILON)
    x = logits.float() * inv[:, None]
    w = torch.exp(x - x.max(dim=1, keepdim=True).values).double()
    z = w.sum(dim=1, keepdim=True)
    order = torch.sort(w, dim=1, descending=True)
    sw = order.values
    # mass strictly above each sorted weight: the cumulative sum before the first entry of its tie group
    csum = torch.cumsum(sw, dim=1) - sw
    first = torch.ones_like(sw, dtype=torch.bool)
    first[:, 1:] = sw[:, 1:] != sw[:, :-1]
    group = torch.cumsum(first.long(), dim=1) - 1
    starts = torch.zeros_like(csum)
    starts.scatter_reduce_(1, group, csum, reduce="amin", include_self=False)
    above = torch.gather(starts, 1, group)
    keep_sorted = above <= top_ps.double()[:, None] * z
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(1, order.indices, keep_sorted)
    p = torch.where(keep, w, torch.zeros_like(w))
    return (p / p.sum(dim=1, keepdim=True)).float()


class M3SampledAccept(OpWrapper):
    """Sampled verification of ``rows = nreq * (D + 1) <= 48`` verify rows from this rank's BF16 logits shard.

    Each row draws one token from its top-p nucleus of softmax(logits / T) over the full (all-rank) vocabulary by
    distributed rejection sampling (Philox keyed by the row's seed / offset, the row, the attempt and the CTA), then
    the strict acceptance runs against ``draft`` as in ``m3_shard_accept``: ``acc_tokens``, ``num_acc``,
    ``gather_ids``, optional ``next_new[:, 0]``. ``tokens`` receives the sampled token per row; ``extra_attempts``
    accumulates exchange rounds beyond the second. The exchange goes through the multicast workspace (two call
    halves of ``minimax_m3_sampled_accept_ws_words(world)`` words) with the (finished rows, call) ``counters``.
    """

    ARCHS = frozenset({Arch.SM_103})

    # draft_len 3 (T = 4 rows per request) at the TP4 caps: 1 / 2 / 8 requests (4 / 8 / 32 rows, 8 CTAs per row)
    # and the 48-row maximum (4 CTAs per row); the 50016-column target shard. Single rank (see note).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="cap 1 (4 rows), top_p 0.8 over a 12-token head: the draws follow the nucleus distribution",
            spec=dict(
                nreq=1, vs=50016, vocab_offset=0, check="distribution", top_p=0.8, temperature=0.7
            ),
        ),
        Cell(
            why="cap 8 (32 rows, 256 co-resident CTAs): top_p ~ 0 keeps only the argmax, token exact",
            spec=dict(
                nreq=8,
                vs=50016,
                vocab_offset=2 * 50016,
                check="argmax",
                top_p=1e-7,
                temperature=1.0,
            ),
        ),
        Cell(
            why="cap 2 (8 rows) with top_p = 1 (no cut: the first draw stands), next_new and forced counts",
            spec=dict(
                nreq=2,
                vs=50016,
                vocab_offset=50016,
                check="distribution",
                top_p=1.0,
                temperature=1.0,
                force="vec",
            ),
        ),
        Cell(
            why="48 rows (4 CTAs per row): the second launch geometry, nucleus distribution at top_p 0.5",
            spec=dict(
                nreq=12,
                vs=50016,
                vocab_offset=3 * 50016,
                check="distribution",
                top_p=0.5,
                temperature=1.3,
                force="total",
            ),
        ),
    )

    note = """
    What is sampled is gated statistically, the rest exactly. Every drawn token must lie in the reference nucleus;
    the empirical distribution over many calls (distinct Philox offsets) must sit within a stated total-variation
    band of the reference's; the acceptance outputs must equal the strict acceptance of the drawn tokens exactly.
    The Philox stream itself is not reproduced in torch: the kernel's draw uses the hardware exp / log
    approximations (__expf, __logf) and a CTA-order inverse CDF, which torch does not reproduce bit for bit.

    Certified at world 1 only. The decision is replicated from every rank's per-round records, which another rank
    computes from its own shard on the fly, so peers cannot be stood in for on one device; the cross-rank path
    needs the multi-device job and is not covered here. Rows with top-k or min-p must not reach this op (it applies
    neither); is_valid states it.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        vocab_offset: int,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        counters: torch.Tensor,
        tokens: torch.Tensor,
        rank: int,
        world: int,
        draft: torch.Tensor,
        batch_idx: torch.Tensor,
        acc_tokens: torch.Tensor,
        num_acc: torch.Tensor,
        gather_ids: torch.Tensor,
        force_total: int,
        next_new: Optional[torch.Tensor],
        force_vec: Optional[torch.Tensor],
        extra_attempts: Optional[torch.Tensor],
        use_pdl: bool = True,
        top_ks: Optional[torch.Tensor] = None,
        min_ps: Optional[torch.Tensor] = None,
    ) -> None:
        torch.ops.trtllm.minimax_m3_sampled_accept(
            logits,
            vocab_offset,
            temperatures,
            top_ps,
            seeds,
            offsets,
            ws_uc,
            ws_mc_ptr,
            counters,
            tokens,
            rank,
            world,
            draft,
            batch_idx,
            acc_tokens,
            num_acc,
            gather_ids,
            force_total,
            next_new,
            force_vec,
            extra_attempts,
            use_pdl,
        )

    def reference(
        self,
        logits: torch.Tensor,
        vocab_offset: int,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        counters: torch.Tensor,
        tokens: torch.Tensor,
        rank: int,
        world: int,
        draft: torch.Tensor,
        batch_idx: torch.Tensor,
        acc_tokens: torch.Tensor,
        num_acc: torch.Tensor,
        gather_ids: torch.Tensor,
        force_total: int,
        next_new: Optional[torch.Tensor],
        force_vec: Optional[torch.Tensor],
        extra_attempts: Optional[torch.Tensor],
        use_pdl: bool = True,
        top_ks: Optional[torch.Tensor] = None,
        min_ps: Optional[torch.Tensor] = None,
    ) -> dict:
        """{"probs": the per-row nucleus distribution over this shard [rows, Vs] (global id = vocab_offset +
        column), "accept": tokens -> the strict acceptance outputs}. The draw itself is random; ``seeds`` /
        ``offsets`` select it and do not change the distribution. World 1: this shard is the whole vocabulary."""
        rows = logits.shape[0]
        probs = nucleus_probs(logits, temperatures[:rows], top_ps[:rows])

        def accept(drawn: torch.Tensor) -> dict[str, torch.Tensor]:
            return strict_accept(drawn, draft, batch_idx, force_total, next_new, force_vec)

        return {"probs": probs, "accept": accept}

    def is_valid(
        self,
        logits: torch.Tensor,
        vocab_offset: int,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        counters: torch.Tensor,
        tokens: torch.Tensor,
        rank: int,
        world: int,
        draft: torch.Tensor,
        batch_idx: torch.Tensor,
        acc_tokens: torch.Tensor,
        num_acc: torch.Tensor,
        gather_ids: torch.Tensor,
        force_total: int,
        next_new: Optional[torch.Tensor],
        force_vec: Optional[torch.Tensor],
        extra_attempts: Optional[torch.Tensor],
        use_pdl: bool = True,
        top_ks: Optional[torch.Tensor] = None,
        min_ps: Optional[torch.Tensor] = None,
    ) -> None:
        rows = logits.shape[0]
        # top-k and min-p are not applied: a row that has them gets a token outside its real sampling set.
        if top_ks is not None:
            k = top_ks[:rows]
            assert bool(((k <= 0) | (k >= logits.shape[1] * world)).all()), (
                "top-k rows must take the generic path"
            )
        if min_ps is not None:
            assert not bool((min_ps[:rows] > 0).any()), "min-p rows must take the generic path"
        # Rows are request-major verify rows: tokens[b * (D + 1) + k]; a mismatched layout pairs drafts with the
        # wrong rows.
        assert rows == draft.shape[0] * (draft.shape[1] + 1), (
            "logits rows must be nreq * (draft_len + 1)"
        )
        # Per-row parameters are read at the row index, unchecked.
        for t in (temperatures, top_ps, seeds, offsets):
            assert t.numel() >= rows and t.is_contiguous(), (
                "per-row parameters must cover every row"
            )
        assert temperatures.dtype == torch.float32 and top_ps.dtype == torch.float32, (
            "fp32 sampling parameters"
        )
        assert seeds.dtype == torch.int64 and offsets.dtype == torch.int64, (
            "int64 Philox seeds / offsets"
        )
        # The finished-row counter elects the CTA that runs the acceptance.
        assert int(counters[0].item()) == 0, (
            "counters[0] (finished rows) must be zero between calls"
        )


m3_sampled_accept = M3SampledAccept()
