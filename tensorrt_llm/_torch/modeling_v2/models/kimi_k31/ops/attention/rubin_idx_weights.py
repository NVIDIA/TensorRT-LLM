# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's logits weights, ``(w * qs) * scale`` in FP32, in one kernel (Triton).

The decode indexer folds each head's query scale and the softmax x heads^-0.5 constant into the
weights its logits kernel reads: ``w * qs.squeeze(-1) * scale``, two torch elementwise kernels.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention.rubin_idx_weights import _idx_weights


class RubinIdxWeights(OpWrapper):
    """A new dense FP32 ``[rows, heads]`` tensor equal bit for bit to ``w * qs * scale``: the
    product rounded to FP32, then times ``scale`` rounded to FP32, as torch's two kernels. ``w``
    FP32 ``[rows, heads]`` with unit inner stride (a column slice of the projection's output);
    ``qs`` FP32 ``[rows, heads]``, dense (the query quantization's scales, one per row and head).
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 / helix8 decode: 96 rows x 32 heads, w a slice of the 160-wide projection",
            spec=dict(rows=96, heads=32, row_stride=160),
        ),
        Cell(why="TEP8 decode: 8 rows", spec=dict(rows=8, heads=32, row_stride=160)),
        Cell(why="a ragged row count, dense w", spec=dict(rows=37, heads=32, row_stride=32)),
    )
    note = """
    Exact for every input the two torch kernels see: the same two FP32 roundings in the same order
    (no fused multiply-add: nothing is added). The query scales are powers of two (UE8M0), so the
    product is exact unless it leaves the normal range.
    """

    def __call__(self, w: torch.Tensor, qs: torch.Tensor, scale: float) -> torch.Tensor:
        return _idx_weights(w, qs, scale)

    def reference(self, w: torch.Tensor, qs: torch.Tensor, scale: float) -> torch.Tensor:
        return w * qs * scale

    def is_valid(self, w: torch.Tensor, qs: torch.Tensor, scale: float) -> None:
        # The kernel walks qs and its output as dense rows of `heads` and w with its row stride.
        assert w.dtype == qs.dtype == torch.float32, "w and qs must be FP32"
        assert w.dim() == 2 and qs.shape == w.shape, "w and qs must both be [rows, heads]"
        assert w.stride(1) == 1, "w must have a unit inner stride"
        assert qs.is_contiguous(), "qs must be contiguous"
        heads = w.shape[1]
        assert heads > 0 and heads & (heads - 1) == 0, "heads must be a power of two"


rubin_idx_weights = RubinIdxWeights()
