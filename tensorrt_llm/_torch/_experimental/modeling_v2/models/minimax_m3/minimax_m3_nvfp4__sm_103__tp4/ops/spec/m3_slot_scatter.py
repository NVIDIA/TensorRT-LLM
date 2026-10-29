# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The spec sampler's per-slot stores of one step's outputs, in one launch (``M3SpecStepFold.scatter`` ->
``m3_spec_tail.slot_scatter``)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def padded_rows(src: torch.Tensor, width: int) -> torch.Tensor:
    """src rows zero-padded (or cut) to ``width`` columns."""
    out = torch.zeros(src.shape[0], width, dtype=src.dtype, device=src.device)
    w = min(width, src.shape[1])
    out[:, :w] = src[:, :w]
    return out


class M3SlotScatter(OpWrapper):
    """For i < len(slots), all int32 2-D strided views: ``da[slots[i]] = sa[i]``, ``db[slots[i]] = sb[i]``,
    ``dd[slots[i]] = sd[i]`` (each zero-padded / cut to the destination width) and ``dl[slots[i]] = sl[i]``. The
    destinations are the store's transposed [slots, width] views."""

    ARCHS = frozenset({Arch.SM_103})

    # The sampler store at draft_len 3 (8 slots): new tokens / next new tokens [slots, 4] (transposed views of
    # [4, slots, 1]), next draft tokens [slots, 3]; a worker output narrower than the store (dynamic draft length)
    # is zero-padded.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="8 rows into shuffled slots, widths equal to the store's",
            spec=dict(n=8, slots=8, src_w=4, src_d=3, store_w=4, store_d=3),
        ),
        Cell(
            why="3 rows of a narrower step (runtime draft 2): zero padding to the store width",
            spec=dict(n=3, slots=8, src_w=3, src_d=2, store_w=4, store_d=3),
        ),
    )

    note = """
    Pure index copies: the reference is exact. Slots are not bounds-checked; is_valid states it.
    """

    def __call__(self, slots, sa, da, sb, db, sl, dl, sd, dd):
        torch.ops.trtllm.minimax_m3_slot_scatter(slots, sa, da, sb, db, sl, dl, sd, dd)

    def reference(self, slots, sa, da, sb, db, sl, dl, sd, dd):
        """The four destinations after the call, on copies."""
        s = slots.long()
        out = {}
        for name, src, dst in (("da", sa, da), ("db", sb, db), ("dd", sd, dd)):
            r = dst.clone()
            r[s] = padded_rows(src[: s.numel()], dst.shape[1])
            out[name] = r
        lr = dl.clone()
        lr[s] = sl[: s.numel()]
        out["dl"] = lr
        return out

    def is_valid(self, slots, sa, da, sb, db, sl, dl, sd, dd):
        assert bool(((slots >= 0) & (slots < da.shape[0]) & (slots < dl.numel())).all()), (
            "slots must index the store"
        )
        # Two rows into one slot race (which one lands is unordered).
        assert slots.unique().numel() == slots.numel(), "slots must be distinct"
        assert sa.shape[0] >= slots.numel() and sl.numel() >= slots.numel(), "a source row per slot"


m3_slot_scatter = M3SlotScatter()
