# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode-graph MXFP8 activation quantize at a fixed tactic (QPIN), ahead of the tuned ``m3_mxfp8_gemm``."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8QuantizeRunner

_RUNNERS: Dict[torch.dtype, object] = {}


def _runner(dtype: torch.dtype):
    runner = _RUNNERS.get(dtype)
    if runner is None:
        runner = _RUNNERS[dtype] = MXFP8QuantizeRunner(dtype)
    return runner


class M3Mxfp8PinnedQuantize(OpWrapper):
    """A bf16 activation [M, K] to MXFP8: (e4m3 [M, K], swizzled UE8M0 scales), on the runner's ``tactic``."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="a 16-row decode verify on a K 6144 input (c4)", spec=dict(m=16, k=6144)),
        Cell(why="dense-MLP down projection input, 32 rows", spec=dict(m=32, k=3072)),
    )

    note = """
    Bit-exact: `reference` is the stock MXFP8 quantize (``torch.ops.trtllm.mxfp8_quantize``, swizzled scales) of the
    same input; every tactic computes the same block maxima and roundings.
    """

    def __call__(self, x: torch.Tensor, tactic: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return _runner(x.dtype).forward([x], tactic=tactic)

    def reference(self, x: torch.Tensor, tactic: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The stock quantize; ``tactic`` is ignored."""
        return torch.ops.trtllm.mxfp8_quantize(x, True)


m3_mxfp8_pinned_quantize = M3Mxfp8PinnedQuantize()
