# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The q = 1 KDA core: MR8's ``PreparedKda.run`` (``KdaSplit``, one token per request) over a
plain decode step's rows, on the dense q1 scratch slots."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_ull_kda_reference import kda_core_reference

HEADS = 6  # 96 KDA heads over TP16


def _cell(why: str, requests: int, padded: tuple[int, ...] = ()) -> Cell:
    """``padded``: rows whose slot is 0 (requests without a checkpoint page)."""
    return Cell(why=why, spec=dict(requests=requests, padded=padded, heads=HEADS))


class KdaDecode(OpWrapper):
    """The KDA core of one token per request, in place.

    ``core`` is MR8's ``PreparedKda`` prepared for ``rows`` q = 1 rows over ``caches``;
    ``projected`` the BF16 [rows, pitch] packed in-projection rows; ``out`` BF16
    [rows, H * 128]; ``caches`` (conv, state): BF16 [slots, 3 H * 128, 3] conv windows and
    FP32 [slots, H, 128, 128] states; ``slots`` int32 [rows] (row ``r`` reads and writes slot
    ``slots[r]``; <= 0: padding). Writes ``out`` and the slots' state and conv window; returns
    ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("q1 decode, one request", 1),
        _cell("q1 decode, two requests", 2),
        _cell("q1 decode, three requests", 3),
        _cell("q1 decode, four requests (the largest served q1 batch at b4)", 4),
        _cell("q1 decode with a padding row (slot 0): a zero row, nothing written", 4, (2,)),
    )

    note = """
    The plain decode step of the K3.1 ULL target's KDA layers (no drafts: a request's first
    generation step or a step without the drafter), which runs on dense scratch slots 1..rows
    that ``kda_decode_checkpoint`` fills from and publishes to the verify checkpoints. Active
    rows must own distinct slots. Compiled before graph capture (``PreparedKda.prepare_caches``);
    launched with PDL on the current stream.
    """

    def __call__(
        self,
        core,
        projected: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
    ) -> torch.Tensor:
        return core.run(projected, out, caches=caches, slots=slots)

    def reference(
        self,
        core,
        projected: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
    ):
        """``(out, states, pages)`` in FP64 (``_kda_reference.kda_core_reference`` at q = 1);
        ``out`` is not read."""
        return kda_core_reference(core, projected, caches, slots, None, 1)

    def is_valid(
        self,
        core,
        projected: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
    ) -> None:
        """Raise where the kernel would misplace a read or write without an error: slots that
        do not cover the rows, two active rows on one slot, conv windows other than 3 columns,
        a state view without contiguous key rows, or operand rows narrower than the layout."""
        rows, dim, heads = projected.shape[0], core.dim, core.heads
        conv, state = caches[:2]
        if slots.ndim != 1 or slots.numel() < rows:
            raise ValueError("kda_decode: slots must be one int32 index per row")
        if projected.shape[1] < 4 * dim + 128 + heads:
            raise ValueError("kda_decode: projected rows narrower than the packed layout")
        if out.dtype != torch.bfloat16 or out.shape != (rows, dim):
            raise ValueError("kda_decode: out must be BF16 [rows, dim]")
        if state.stride()[-2:] != (128, 1) or conv.stride(-1) != 1 or conv.shape[-1] != 3:
            raise ValueError("kda_decode: contiguous state rows and 3-column conv windows")
        active = [s for s in slots[:rows].tolist() if s > 0]
        if len(set(active)) != len(active):
            raise ValueError("kda_decode: active rows share a slot")


kda_decode = KdaDecode()
