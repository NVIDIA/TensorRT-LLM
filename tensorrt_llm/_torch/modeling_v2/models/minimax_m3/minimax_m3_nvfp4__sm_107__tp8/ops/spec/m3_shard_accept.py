# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Greedy Eagle3 verification from this rank's logits shard: per-row argmax, the cross-rank (value, index) exchange
and the strict acceptance in one kernel (``m3_shard_accept.shard_accept``)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ....ops.spec.acceptance_reference import _LOW32 as _LOW32
from ....ops.spec.acceptance_reference import exchange_tokens as exchange_tokens
from ....ops.spec.acceptance_reference import key_token as key_token
from ....ops.spec.acceptance_reference import local_argmax as local_argmax
from ....ops.spec.acceptance_reference import make_keys as make_keys
from ....ops.spec.acceptance_reference import ordered_bits as ordered_bits
from ....ops.spec.acceptance_reference import strict_accept as strict_accept
from ....ops.spec.acceptance_reference import unsigned_max as unsigned_max

#: Verify rows per call, and exchange rows per call half of its workspace.
SHARD_ROWS = 48


class M3ShardAccept(OpWrapper):
    """Greedy verification of ``nreq * (D + 1) <= 48`` verify rows from this rank's BF16 logits shard ``[rows, Vs]``
    (global ids ``vocab_offset ..``).

    Each row's token is its global argmax over all ranks' shards: keys ``ordered(value) << 32 | ~index`` (ties take
    the lowest index, as ``torch.argmax``) are max-reduced over the shard, pushed into every rank's exchange slot
    through the multicast slots ``[2][48][world]`` (Lamport: 0 = empty) and max-reduced over the ranks by the last
    CTA, which then runs the strict acceptance: ``acc_tokens [nreq, D + 1]``, ``num_acc [nreq]`` (``force_vec`` /
    ``force_total`` override the counts), ``gather_ids [nreq]`` (the first draft pass's row per request) and, when
    given, ``next_new[:, 0]`` (the last accepted token).
    """

    ARCHS = frozenset({Arch.SM_107})

    # Every greedy verify step of the target (modeling's SHARD_ACCEPT): draft_len 3 at the TP8 GEN worker's 1..2
    # requests (4..8 rows), plus the 48-row maximum; the target lm_head's BF16 shard logits (25008 columns per rank
    # at TP8, 8 exchange slots per row). The logits come from the head GEMM, so the op only reduces.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="b1: 4 verify rows (1 request x T = 4), rank 0 of 8, next_new staged",
            spec=dict(nreq=1, draft_len=3, vs=25008, world=8, rank=0, force="none", strided=False),
        ),
        Cell(
            why="b2: 8 verify rows, rank 7, global ids offset by 7 x 25008",
            spec=dict(nreq=2, draft_len=3, vs=25008, world=8, rank=7, force="none", strided=False),
        ),
        Cell(
            why="8 rows (2 requests) read from a row-strided view of a wider logits buffer, forced counts",
            spec=dict(nreq=2, draft_len=3, vs=25008, world=8, rank=3, force="vec", strided=True),
        ),
        Cell(
            why="the 48-row maximum (12 requests at draft 3), force_total, rows with exact cross-rank ties",
            spec=dict(
                nreq=12,
                draft_len=3,
                vs=25008,
                world=8,
                rank=5,
                force="total",
                strided=False,
                ties=True,
            ),
        ),
    )

    note = """
    The op reduces BF16 logits exactly, so the reference is bit-exact, including cross-rank ties (lowest global
    index wins). The cross-rank exchange is certified on one device: the other ranks' keys are written into this
    rank's slots before the call, which is what their multicast pushes would have put there; the multicast mapping
    itself (ws_mc_ptr = this rank's slots, as at world 1) is not exercised across devices here.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        vocab_offset: int,
        keys: torch.Tensor,
        counters: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        rank: int,
        world: int,
        draft: torch.Tensor,
        batch_idx: torch.Tensor,
        acc_tokens: torch.Tensor,
        num_acc: torch.Tensor,
        gather_ids: torch.Tensor,
        force_total: int,
        next_new: Optional[torch.Tensor],
        num_ctas: int,
        force_vec: Optional[torch.Tensor] = None,
    ) -> None:
        torch.ops.trtllm.minimax_m3_shard_accept(
            logits,
            vocab_offset,
            keys,
            counters,
            ws_uc,
            ws_mc_ptr,
            rank,
            world,
            draft,
            batch_idx,
            acc_tokens,
            num_acc,
            gather_ids,
            force_total,
            next_new,
            num_ctas,
            force_vec,
        )

    def reference(
        self,
        logits: torch.Tensor,
        vocab_offset: int,
        keys: torch.Tensor,
        counters: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        rank: int,
        world: int,
        draft: torch.Tensor,
        batch_idx: torch.Tensor,
        acc_tokens: torch.Tensor,
        num_acc: torch.Tensor,
        gather_ids: torch.Tensor,
        force_total: int,
        next_new: Optional[torch.Tensor],
        num_ctas: int,
        force_vec: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Lowest-index maximum of each row's BF16 logits, the largest key among it and the peers' keys in
        ``ws_uc``, then the strict acceptance. Reads ``ws_uc`` / ``counters`` before the call; the scratch and
        output buffers and ``num_ctas`` do not change the result."""
        best, idx = local_argmax(logits.float(), vocab_offset)
        tokens = exchange_tokens(make_keys(best, idx), ws_uc, counters, rank, world, SHARD_ROWS)
        out = strict_accept(tokens, draft, batch_idx, force_total, next_new, force_vec)
        out["tokens"] = tokens
        return out

    def is_valid(
        self,
        logits: torch.Tensor,
        vocab_offset: int,
        keys: torch.Tensor,
        counters: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        rank: int,
        world: int,
        draft: torch.Tensor,
        batch_idx: torch.Tensor,
        acc_tokens: torch.Tensor,
        num_acc: torch.Tensor,
        gather_ids: torch.Tensor,
        force_total: int,
        next_new: Optional[torch.Tensor],
        num_ctas: int,
        force_vec: Optional[torch.Tensor] = None,
    ) -> None:
        assert not bool(keys[: logits.shape[0]].any()), "keys must start zeroed"
        assert int(counters[0].item()) == 0, (
            "counters[0] (finished CTAs) must be zero between calls"
        )
        assert ws_uc.numel() >= 2 * SHARD_ROWS * world, "ws_uc must hold [2][48][world] slots"
        # A NaN logit maps to a key above every finite one and wins the row silently.
        assert not bool(torch.isnan(logits).any()), "NaN logits win the argmax"
        nreq = draft.shape[0]
        assert bool(((batch_idx[:nreq] >= 0) & (batch_idx[:nreq] < nreq)).all()), (
            "batch_idx must index the nreq requests"
        )
        if force_vec is not None:
            fv = force_vec[:nreq]
            assert bool(((fv >= 1) & (fv <= draft.shape[1] + 1)).all()), (
                "force_vec counts must lie in [1, D + 1]"
            )


m3_shard_accept = M3ShardAccept()
