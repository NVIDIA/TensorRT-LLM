# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Greedy Eagle3 verification from this rank's logits shard: per-row argmax, the cross-rank (value, index) exchange
and the strict acceptance in one kernel (``m3_shard_accept.shard_accept``)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Verify rows per call, and exchange rows per call half of its workspace.
SHARD_ROWS = 48

_LOW32 = 0xFFFFFFFF


def ordered_bits(v: torch.Tensor) -> torch.Tensor:
    """The kernel's order-preserving uint32 image of fp32 values, as int64 (-0.0 maps to +0.0's image: they tie)."""
    b = v.float().contiguous().view(torch.int32).to(torch.int64) & _LOW32
    b = torch.where(b == 0x80000000, torch.zeros_like(b), b)
    return torch.where(b >= 0x80000000, _LOW32 - b, b | 0x80000000)


def make_keys(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Exchange keys of (fp32 value, global token index) pairs: larger value wins, ties take the lower index."""
    return (ordered_bits(values) << 32) | (_LOW32 - index.to(torch.int64))


def unsigned_max(keys: torch.Tensor, dim: int) -> torch.Tensor:
    """Max of int64-stored uint64 keys (the kernel compares them unsigned)."""
    sign = torch.iinfo(torch.int64).min
    return (keys ^ sign).max(dim=dim).values ^ sign


def key_token(key: torch.Tensor) -> torch.Tensor:
    return (_LOW32 - (key & _LOW32)).to(torch.int32)


def local_argmax(logits: torch.Tensor, vocab_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(max value, lowest global index reaching it) per row of a [M, Vs] fp32 shard."""
    best = logits.max(dim=1).values
    idx = (logits == best[:, None]).float().argmax(dim=1)  # first (lowest) index at the max
    return best, idx.to(torch.int64) + vocab_offset


def exchange_tokens(
    own_keys: torch.Tensor,
    ws_uc: torch.Tensor,
    counters: torch.Tensor,
    rank: int,
    world: int,
    slot_rows: int,
) -> torch.Tensor:
    """Global tokens of the cross-rank pick: this rank's keys against the other ranks' keys already in its slots.

    Slot (half * slot_rows + t) * world + r of ``ws_uc`` holds rank r's key of row t; half = call counter & 1. A
    single-device run stands in for the other ranks by writing their keys there before the call.
    """
    m = own_keys.shape[0]
    half = int(counters[1].item()) & 1
    slots = (
        ws_uc.view(torch.int64)[half * slot_rows * world : (half * slot_rows + m) * world]
        .view(m, world)
        .clone()
    )
    slots[:, rank] = own_keys
    return key_token(unsigned_max(slots, dim=1))


def strict_accept(
    tokens: torch.Tensor,
    draft: torch.Tensor,
    batch_idx: torch.Tensor,
    force_total: int,
    next_new: Optional[torch.Tensor],
    force_vec: Optional[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The strict acceptance epilogue from the verify-row tokens [nreq * (D + 1)] (request-major).

    accepted[b] = tokens of request b's rows; num_accepted[b] = 1 + the matching prefix of draft[b] (overridden by
    force_vec[b], else by force_total > 0); gather_ids[b] = batch_idx[b] * (D + 1) + num_accepted[b] - 1, and
    next_new[b, 0] (when given) = the token at that row. Returns the outputs plus ``next_new`` as written.
    """
    nreq, d = draft.shape
    k1 = d + 1
    tokens = tokens.to(torch.int32)
    accepted = tokens.view(nreq, k1).clone()
    match = (draft.to(torch.int32) == accepted[:, :d]).to(torch.int32)
    num = 1 + match.cumprod(dim=1).sum(dim=1)
    if force_vec is not None:
        num = force_vec[:nreq].to(num.dtype)
    elif force_total > 0:
        num = torch.full_like(num, force_total)
    gather = batch_idx.to(torch.int64) * k1 + num.to(torch.int64) - 1
    out = {"accepted": accepted, "num_accepted": num.to(torch.int32), "gather_ids": gather}
    if next_new is not None:
        nn = next_new.clone()
        nn[:, 0] = tokens[gather]
        out["next_new"] = nn
    return out


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

    ARCHS = frozenset({Arch.SM_103})

    # Every greedy verify step of the target (dispatch.SHARD_ACCEPT): draft_len 3 at 1..12 requests (4..48 rows), the
    # target lm_head's BF16 shard logits (25008 columns per rank at TP8). The logits come from the head GEMM, so the
    # op only reduces.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="cap 1: 4 verify rows (1 request x T = 4), rank 0 of 8, next_new staged",
            spec=dict(nreq=1, draft_len=3, vs=25008, world=8, rank=0, force="none", strided=False),
        ),
        Cell(
            why="cap 2: 8 verify rows, rank 2, global ids offset by 2 x 25008",
            spec=dict(nreq=2, draft_len=3, vs=25008, world=8, rank=2, force="none", strided=False),
        ),
        Cell(
            why="cap 4: 16 verify rows (4 requests x T = 4), rank 1 of 8, next_new staged",
            spec=dict(nreq=4, draft_len=3, vs=25008, world=8, rank=1, force="none", strided=False),
        ),
        Cell(
            why="cap 8: 32 verify rows, rank 7, global ids offset by 7 x 25008",
            spec=dict(nreq=8, draft_len=3, vs=25008, world=8, rank=7, force="none", strided=False),
        ),
        Cell(
            why="12 rows (3 requests) read from a row-strided view of a wider logits buffer, forced counts",
            spec=dict(nreq=3, draft_len=3, vs=25008, world=8, rank=0, force="vec", strided=True),
        ),
        Cell(
            why="the 48-row maximum (12 requests at draft 3), force_total, rows with exact cross-rank ties",
            spec=dict(
                nreq=12,
                draft_len=3,
                vs=25008,
                world=8,
                rank=2,
                force="total",
                strided=False,
                ties=True,
            ),
        ),
        # Every other request count of the verify range, on the 8-rank 25008-column shards.
        *(
            Cell(
                why=f"{n} requests ({4 * n} verify rows): a request count between the ones above",
                spec=dict(
                    nreq=n, draft_len=3, vs=25008, world=8, rank=n % 8, force="none", strided=False
                ),
            )
            for n in (5, 6, 7, 9, 10, 11)
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
