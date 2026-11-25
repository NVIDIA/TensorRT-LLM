# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampled (temperature + top-p) Eagle3 verification over the TP-sharded vocabulary in one kernel
(``m3_sampled_accept.sampled_accept``)."""

from __future__ import annotations

import sys
from typing import Optional, cast

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.spec import (
    sampling_reference as _sampling_shared,
)

from .m3_shard_accept import strict_accept as strict_accept

#: Verify rows per call.
MAX_ROWS = 48
#: The fused sampling kernel's 1 / T guard.
TEMP_EPSILON = 1e-6


def nucleus_probs(
    logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor
) -> torch.Tensor:
    """Per-row sampling distribution [rows, Vs] (fp32): w = exp(x / (T + 1e-6) - max), kept iff the mass strictly
    above its weight is at most top_p * sum(w) (ties included), renormalized over the kept tokens."""
    return _sampling_shared.nucleus_probs(_REFERENCE_TARGET, logits, temperatures, top_ps)


class M3SampledAccept(OpWrapper):
    """Sampled verification of ``rows = nreq * (D + 1) <= 48`` verify rows from this rank's BF16 logits shard.

    Each row draws one token from its top-p nucleus of softmax(logits / T) over the full (all-rank) vocabulary by
    distributed rejection sampling (Philox keyed by the row's seed / offset, the row, the attempt and the CTA), then
    the strict acceptance runs against ``draft`` as in ``m3_shard_accept``: ``acc_tokens``, ``num_acc``,
    ``gather_ids``, optional ``next_new[:, 0]``. ``tokens`` receives the sampled token per row; ``extra_attempts``
    accumulates exchange rounds beyond the second. The exchange goes through the multicast workspace (two call
    halves of ``minimax_m3_sampled_accept_ws_words(world)`` words) with the (finished rows, call) ``counters``.
    """

    ARCHS = frozenset({Arch.SM_107})

    # draft_len 3 (T = 4 rows per request) at the TP8 GEN worker's batch: 1 / 2 requests (4 / 8 rows, 8 CTAs per
    # row), and the 48-row maximum (4 CTAs per row, 12.5 KB of staged logits per CTA); the 25008-column target shard
    # (vocab 200064 / 8). Single rank (see note).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="b1 (4 rows), top_p 0.8 over a 12-token head: the draws follow the nucleus distribution",
            spec=dict(
                nreq=1, vs=25008, vocab_offset=0, check="distribution", top_p=0.8, temperature=0.7
            ),
        ),
        Cell(
            why="b2 (8 rows): top_p ~ 0 keeps only the argmax, token exact",
            spec=dict(
                nreq=2,
                vs=25008,
                vocab_offset=7 * 25008,
                check="argmax",
                top_p=1e-7,
                temperature=1.0,
            ),
        ),
        Cell(
            why="b2 (8 rows) with top_p = 1 (no cut: the first draw stands), next_new and forced counts",
            spec=dict(
                nreq=2,
                vs=25008,
                vocab_offset=7 * 25008,
                check="distribution",
                top_p=1.0,
                temperature=1.0,
                force="vec",
            ),
        ),
        Cell(
            why="AgentX sampling (T 1.0, top_p 0.95) at b1 with forced counts: the draws follow the nucleus",
            spec=dict(
                nreq=1,
                vs=25008,
                vocab_offset=0,
                check="distribution",
                top_p=0.95,
                temperature=1.0,
                force="total",
            ),
        ),
        Cell(
            why="48 rows (4 CTAs per row): the second launch geometry, nucleus distribution at top_p 0.5",
            spec=dict(
                nreq=12,
                vs=25008,
                vocab_offset=0,
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
        torch.ops.trtllm.minimax_m3_sm107tp8_sampled_accept(
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
        return _sampling_shared.sampled_reference(
            self,
            _REFERENCE_TARGET,
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
            top_ks,
            min_ps,
        )

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
        return _sampling_shared.validate_sampling(
            self,
            _REFERENCE_TARGET,
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
            top_ks,
            min_ps,
        )


m3_sampled_accept = M3SampledAccept()

_REFERENCE_TARGET = cast(_sampling_shared.SamplingTarget, sys.modules[__name__])
