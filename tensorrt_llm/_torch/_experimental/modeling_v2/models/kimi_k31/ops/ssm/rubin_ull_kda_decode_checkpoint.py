# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The q = 1 KDA step's checkpoint copies: ``kernels/kda_io.py``'s Triton gather of each request's
accepted checkpoint into dense scratch rows, and the scatter of its result to checkpoint zero."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.ssm.rubin_ull_kda_io import copy_decode_checkpoint

TOKENS = 8
HEADS = 6


def _cell(why: str, accepted: tuple[int, ...], scatter: bool, padded: tuple[int, ...] = ()) -> Cell:
    """``accepted[r]``: request ``r``'s accepted count (1..8); ``padded``: rows with slot 0."""
    return Cell(
        why=why,
        spec=dict(accepted=accepted, scatter=scatter, padded=padded, tokens=TOKENS, heads=HEADS),
    )


class KdaDecodeCheckpoint(OpWrapper):
    """Copy between the verify checkpoints and the q1 scratch, bit for bit.

    ``caches`` (conv BF16 [rows, C, q + 2], state FP32 [rows, H, 128, 128]) are the verify
    checkpoint views; ``scratch`` (conv BF16 [R + 1, C, 3], state FP32 [R + 1, H, 128, 128])
    the dense q1 rows (row 0 padding, request ``r`` row ``r + 1``); ``slots`` int32
    [R, q] (``slots[r, 0]`` <= 0: padding), ``accepted`` int32 [>= R]. Gather
    (``scatter=False``): scratch row ``r + 1`` gets state ``slots[r, accepted[r] - 1]`` and the
    conv columns ``accepted[r] - 1 ..+3`` of page ``slots[r, 0]`` (zeros for padding).
    Scatter: state ``slots[r, 0]`` and conv columns 0..3 of page ``slots[r, 0]`` get scratch
    row ``r + 1`` (padding rows untouched). Returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("gather, one request, no draft accepted", (1,), False),
        _cell("gather, four requests at accepted 2, 5, 7, 8", (2, 5, 7, 8), False),
        _cell("gather with a padding row: zero scratch row", (3, 1, 6), False, (1,)),
        _cell("scatter, one request", (1,), True),
        _cell("scatter, four requests with a padding row left untouched", (4, 1, 8, 3), True, (2,)),
    )

    note = """
    Around ``kda_decode`` in the K3.1 ULL target's plain decode step: gather before, scatter
    after (checkpoint zero; the cache manager then records zero accepted drafts). Indexed
    copies through the flat checkpoint views never touch reserved index 0. Gather checks the
    accepted range asynchronously (``torch._assert_async``). No arithmetic or conversion.
    """

    def __call__(
        self,
        caches: tuple[torch.Tensor, torch.Tensor],
        scratch: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
        scatter: bool,
    ) -> None:
        copy_decode_checkpoint(caches, scratch, slots, accepted, scatter=scatter)

    def reference(
        self,
        caches: tuple[torch.Tensor, torch.Tensor],
        scratch: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
        scatter: bool,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """``(caches, scratch)`` as the copy leaves them, as new tensors (inputs not written)."""
        conv, state = (t.clone() for t in caches)
        conv_tmp, state_tmp = (t.clone() for t in scratch)
        table, counts = slots.tolist(), accepted.tolist()
        for r, row in enumerate(table):
            destination = row[0]
            if scatter:
                if destination > 0:
                    state[destination] = state_tmp[r + 1]
                    conv[destination, :, :3] = conv_tmp[r + 1]
                continue
            if destination <= 0:
                state_tmp[r + 1] = 0
                conv_tmp[r + 1] = 0
                continue
            position = max(counts[r] - 1, 0)
            state_tmp[r + 1] = caches[1][row[position]]
            conv_tmp[r + 1] = caches[0][destination, :, position : position + 3]
        return (conv, state), (conv_tmp, state_tmp)

    def is_valid(
        self,
        caches: tuple[torch.Tensor, torch.Tensor],
        scratch: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
        scatter: bool,
    ) -> None:
        """Raise where the copy would misplace data without an error: two active requests on
        one page (scatter), or a conv page too narrow for the gathered columns."""
        conv = caches[0]
        table, counts = slots.tolist(), accepted.tolist()
        active = [row[0] for row in table if row[0] > 0]
        if len(set(active)) != len(active):
            raise ValueError("kda_decode_checkpoint: active requests share a page")
        if conv.shape[-1] < slots.shape[1] + 2 or conv.stride(-1) != 1:
            raise ValueError("kda_decode_checkpoint: conv pages narrower than q + 2 columns")
        if not scatter and any(
            row[0] > 0 and not 1 <= counts[r] <= slots.shape[1] for r, row in enumerate(table)
        ):
            raise ValueError("kda_decode_checkpoint: an accepted count outside 1..q")


kda_decode_checkpoint = KdaDecodeCheckpoint()
