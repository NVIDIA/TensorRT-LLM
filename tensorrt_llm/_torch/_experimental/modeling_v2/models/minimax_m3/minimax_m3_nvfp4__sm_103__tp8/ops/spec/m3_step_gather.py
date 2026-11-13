# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The overlap scheduler's per-step input gathers from the previous step's device outputs, in one launch
(``M3SpecStepFold.gather`` -> ``m3_spec_tail.step_gather``)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class M3StepGather(OpWrapper):
    """For the ``P`` previous requests (sequence slots ``slots``, ``T`` tokens and ``D`` draft tokens each), all
    int32::

        ids[ids0 + p*T + t]  = new_tok[t, slots[p]]            (new_tok [T', slots, (1)])
        draft[d0 + p*D + d]  = next_draft[slots[p], d]
        pos_off[i]           = lens[ppi[i - b0*T]]   for i in [b0*T, (b0+P)*T), else 0   (every i < pos_off.numel())
        kv_off[i]            = lens[slots[i - b0]] - T for i in [b0, b0+P), else 0         (every i < kv_off.numel())

    Every other element of ``ids`` / ``draft`` is left as it was.
    """

    ARCHS = frozenset({Arch.SM_103})

    # draft_len 3: T = 4, D = 3; the store's [T, slots, 1] new tokens and [slots, D] next drafts (8 slots, the
    # max batch); the engine's flat buffers sized for the max batch.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="an unchanged batch of 8 previous requests in shuffled slots, nothing before them (b0 = 0)",
            spec=dict(p=8, t=4, d=3, slots=8, b0=0, ids0=0, d0=0, extra_rows=0),
        ),
        Cell(
            why="2 previous requests after 3 new ones: offsets ids0 / d0 / b0 nonzero, zero-filled offsets outside",
            spec=dict(p=2, t=4, d=3, slots=8, b0=3, ids0=12, d0=9, extra_rows=3),
        ),
        Cell(
            why="one request, the store's 2-D view of new tokens (beam dim squeezed)",
            spec=dict(p=1, t=4, d=3, slots=8, b0=0, ids0=0, d0=0, extra_rows=0, two_d=True),
        ),
    )

    note = """
    Pure index copies: the reference is exact. The op does not bound slots, ppi or the destination offsets; is_valid
    states them.
    """

    def __call__(
        self,
        new_tok,
        next_draft,
        lens,
        slots,
        ppi,
        P,
        T,
        D,
        ids,
        ids0,
        draft,
        d0,
        pos_off,
        kv_off,
        b0,
    ):
        torch.ops.trtllm.minimax_m3_step_gather(
            new_tok,
            next_draft,
            lens,
            slots,
            ppi,
            P,
            T,
            D,
            ids,
            ids0,
            draft,
            d0,
            pos_off,
            kv_off,
            b0,
        )

    def reference(
        self,
        new_tok,
        next_draft,
        lens,
        slots,
        ppi,
        P,
        T,
        D,
        ids,
        ids0,
        draft,
        d0,
        pos_off,
        kv_off,
        b0,
    ):
        """The four destinations after the call, on copies (``ids`` / ``draft`` keep their other elements)."""
        nt = new_tok.reshape(new_tok.shape[0], new_tok.shape[1])
        s = slots[:P].long()
        ids_r, draft_r = ids.clone(), draft.clone()
        ids_r[ids0 : ids0 + P * T] = nt[:T, s].t().reshape(-1)
        draft_r[d0 : d0 + P * D] = next_draft[s, :D].reshape(-1)
        pos_r = torch.zeros_like(pos_off)
        pos_r[b0 * T : (b0 + P) * T] = lens[ppi[: P * T].long()]
        kv_r = torch.zeros_like(kv_off)
        kv_r[b0 : b0 + P] = lens[s] - T
        return {"ids": ids_r, "draft": draft_r, "pos_off": pos_r, "kv_off": kv_r}

    def is_valid(
        self,
        new_tok,
        next_draft,
        lens,
        slots,
        ppi,
        P,
        T,
        D,
        ids,
        ids0,
        draft,
        d0,
        pos_off,
        kv_off,
        b0,
    ):
        # Reads through slots / ppi and writes at the offsets are unchecked: any of them out of range reads or
        # writes another buffer.
        s = slots[:P]
        assert bool(
            (
                (s >= 0) & (s < next_draft.shape[0]) & (s < new_tok.shape[1]) & (s < lens.numel())
            ).all()
        ), "slots must index the store"
        assert bool(((ppi[: P * T] >= 0) & (ppi[: P * T] < lens.numel())).all()), (
            "ppi must index lens"
        )
        assert new_tok.shape[0] >= T and next_draft.shape[1] >= D, "store narrower than T / D"
        assert ids.numel() >= ids0 + P * T and draft.numel() >= d0 + P * D, "destinations too short"
        assert pos_off.numel() >= (b0 + P) * T and kv_off.numel() >= b0 + P, (
            "offset buffers too short"
        )


m3_step_gather = M3StepGather()
