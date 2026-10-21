# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-gated exact (erf) GELU -- what `torch.nn.GELU()` is -- the vision patch
merger's activation, which no config key names.

Tier-3 composed: one `torch.nn.functional.gelu(x)` call, `approximate="none"`.
`torch.ops.trtllm.moe_gelu` computes this function bitwise but is not a
substitute: its `tile_idx_to_mn_limit` / `num_non_exiting_tiles` operands must
be int32 *device* tensors (on the host it raises nothing and returns garbage,
nan, or an illegal memory access), so wrapping it means launching two fills
before the call; it also rejects fp32 and non-2-D inputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: `merger.linear_fc1` is `[4608, 4608]`: spatial merge 2 over a 1152-wide tower.
MERGER_WIDTH = 4608

#: Rows the merger is asked for, `sum(t*h*w) / 4`: 1 the smallest, 140 a
#: 336x448 image (grid [1, 20, 28]) -- not a multiple of any MoE tile size --
#: and 8192 the most one tower call carries (32768 patch rows).
MERGER_ROW_COUNTS = (1, 140, 8192)

_SQRT_2 = math.sqrt(2.0)


class GeluErf(OpWrapper):
    """`0.5 * x * (1 + erf(x / sqrt(2)))`, elementwise, any shape, new tensor
    out in `x`'s dtype.

    Fusion boundary: the activation only. `merger.linear_fc1` before it and
    `merger.linear_fc2` after it are the caller's; there is no gate half and no
    multiply. This is the erf GELU and not the tanh one (`GeluTanh`, the block
    MLPs' `hidden_act`); both appear in this one tower.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: shape, dtype; `dist` "uniform" over [lo, hi] (default) or "merger"
    # (a tower hidden state scaled by the merger norm's measured gamma, grouped
    # 4-to-1, through an fc1 at the checkpoint's measured weight/bias scale).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"merger activand, {rows} merged rows x 4608 bf16",
            spec=dict(shape=(rows, MERGER_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
        )
        for rows in MERGER_ROW_COUNTS
    ) + (
        Cell(
            why="activand built the way the merger builds it",
            spec=dict(shape=(512, MERGER_WIDTH), dtype=torch.bfloat16, dist="merger"),
        ),
        Cell(
            why="float32 over [-3, 3], the erf/tanh divergence peak",
            spec=dict(shape=(256, MERGER_WIDTH), dtype=torch.float32, lo=-3.0, hi=3.0),
        ),
    )

    note = """
    Metadata consumed: none. Out-of-place; `x` is not mutated.

    Certified on contiguous bf16 `[rows, 4608]` activands (what the merger's fc1
    returns) plus one fp32 cell.

    Identity is pinned by fp32 only (the fp32 cell and the identity test).
    Max |erf GELU - tanh GELU| is 4.73e-4 at x = -2.699; in fp32 the tanh form
    lands ~1400x the correct path's deviation away, 31x outside the default
    gate. In bf16 one ulp at the output scale exceeds that gap (tanh variant
    8.24e-3 vs correct 7.77e-3 max abs), so a bf16 match does not say which GELU
    was computed.

    Values: `-inf -> nan` (the closed form does the same); `+inf -> +inf`;
    large negatives give `-0.0`; `finfo.max` comes back unchanged.
    CUDA-graph capturable; the output is a fresh allocation.

    Not certified: non-contiguous inputs, fp16, other widths, float8
    operands. No index-typed argument; numel past 2^31
    not probed.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """The closed form in fp64 from primitive ops, rounded once to `x.dtype`."""
        x64 = x.double()
        return (0.5 * x64 * (1.0 + torch.erf(x64 / _SQRT_2))).to(x.dtype)


gelu_erf = GeluErf()
