# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The verify logits rows of CUDA-graph padding requests, zeroed in place."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.zero_padding_rows import zero_padding_rows_

#: Kimi K3.1 vocabulary (the LM head's out_features).
VOCAB = 163840


def _spec(n, rows_per_request, padding, *, vocab=VOCAB, ids=torch.int64, special=False) -> dict:
    return dict(
        n=n,
        rows_per_request=rows_per_request,
        padding=padding,
        vocab=vocab,
        ids=ids,
        special=special,
    )


class K31ZeroPaddingRows(OpWrapper):
    """``SpecWorkerBase._zero_padding_rows(logits, ...)`` of a uniform verify (``rows_per_request`` rows a request),
    in place: the rows of every request whose slot is the dummy slot row (a CUDA-graph padding request) set to zero,
    every other row untouched; returns ``logits``.

    The engine method returns ``logits.masked_fill(mask, 0.0)``, a new tensor (a copy of every row, then a pass
    over every element); this writes the padding requests' rows only, so the readers of the result (the target
    probabilities, the greedy verification) see the same values. The caller's ``logits`` tensor itself changes in
    those rows: zeros instead of what the padding requests' uninitialized state produced.

    Args:
        logits: ``[n * rows_per_request, vocab]`` contiguous FP32, ``vocab`` a multiple of 4, 16-byte aligned.
        slot_ids: ``[n]`` int32 / int64 sequence slots of the requests (``spec_metadata.batch_slot_ids`` of the
            generation requests).
        dummy_slot_row: the padding requests' slot (``spec_metadata.dummy_slot_row``, positive).
        rows_per_request: logits rows a request contributes.
    Returns:
        ``logits``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="TEP8 B7 width 5 (35 rows), no padding", spec=_spec(7, 5, "none")),
        Cell(
            why="TEP8 B7 width 5, one padding request (B7 in the B8 graph)",
            spec=_spec(8, 5, "last"),
        ),
        Cell(why="TEP16 B16 width 5 (80 rows), three padding requests", spec=_spec(16, 5, "some")),
        Cell(
            why="width 8, every request padding, NaN / inf in the padding rows",
            spec=_spec(4, 8, "all", special=True),
        ),
        Cell(
            why="B1 width 6, int32 slot ids, the request is padding",
            spec=_spec(1, 6, "all", ids=torch.int32),
        ),
        Cell(why="64 requests width 5, first and last padding", spec=_spec(64, 5, "ends")),
        Cell(
            why="a vocab of 1,028 values (not a multiple of 16)",
            spec=_spec(5, 3, "some", vocab=1028),
        ),
    )

    note = """
    Kernel: ``kernels/spec/zero_padding_rows.py``, one launch of one wave of CTAs that read the slot ids and store
    zeros over the padding requests' rows, split evenly over the CTAs; rows of real requests are neither read nor
    written. Bitwise: every row equals the engine method's result (+0.0 in padding rows, the input elsewhere,
    NaN payloads included).
    """

    def __call__(
        self,
        logits: torch.Tensor,
        slot_ids: torch.Tensor,
        dummy_slot_row: int,
        rows_per_request: int,
    ) -> torch.Tensor:
        return zero_padding_rows_(logits, slot_ids, dummy_slot_row, rows_per_request)

    def reference(
        self,
        logits: torch.Tensor,
        slot_ids: torch.Tensor,
        dummy_slot_row: int,
        rows_per_request: int,
    ) -> torch.Tensor:
        """The engine method's result (``masked_fill`` of the padding requests' rows), a new tensor."""
        is_padding = (slot_ids == dummy_slot_row).repeat_interleave(rows_per_request)
        return logits.masked_fill(is_padding.unsqueeze(-1), 0.0)

    def is_valid(
        self,
        logits: torch.Tensor,
        slot_ids: torch.Tensor,
        dummy_slot_row: int,
        rows_per_request: int,
    ) -> None:
        assert logits.dtype == torch.float32 and logits.dim() == 2, "FP32 [rows, vocab] logits"
        assert logits.is_contiguous() and logits.data_ptr() % 16 == 0, (
            "contiguous, 16-byte aligned rows"
        )
        assert logits.shape[1] % 4 == 0, "rows of whole 16-byte vectors"
        assert logits.shape[0] == slot_ids.shape[0] * rows_per_request, (
            "rows_per_request rows a request"
        )
        assert slot_ids.dtype in (torch.int32, torch.int64) and slot_ids.is_contiguous(), (
            "int slot ids"
        )


k31_zero_padding_rows = K31ZeroPaddingRows()

__all__ = ["K31ZeroPaddingRows", "k31_zero_padding_rows"]
