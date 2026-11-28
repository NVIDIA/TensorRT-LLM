# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 rows widened to FP32, bit for bit what ``x.float()`` returns, with full-width accesses (Triton).

The FP32 operand of a TF32 GEMM fed by BF16 activations: the DSA indexer's ``[wk | weights_proj]`` projection on a
DEP16 context step widens its ``[16384, 7168]`` input with it.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.gemm.rubin_upcast_fp32 import _upcast


class RubinUpcastFp32(OpWrapper):
    """A new contiguous FP32 tensor equal bit for bit to ``x.float()``; ``x`` contiguous BF16. The FP32 bits are
    the BF16 bits followed by 16 zero bits, so a NaN keeps its payload (torch's GPU cast may make it canonical)."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 prefill, a 16k chunk: the indexer's [wk | weights_proj] input",
            spec=dict(rows=16384, cols=7168),
        ),
        Cell(why="a ragged chunk", spec=dict(rows=37, cols=7168)),
        Cell(why="one row", spec=dict(rows=1, cols=7168)),
    )
    note = """
    Exact for every input: a BF16 value is an FP32 value. Only the access width differs from torch's casting
    copy.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _upcast(x)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()

    def is_valid(self, x: torch.Tensor) -> None:
        # The kernel walks the storage as one flat run of int16 words.
        assert x.dtype == torch.bfloat16, "x must be BF16"
        assert x.is_contiguous(), "x must be contiguous"


rubin_upcast_fp32 = RubinUpcastFp32()
