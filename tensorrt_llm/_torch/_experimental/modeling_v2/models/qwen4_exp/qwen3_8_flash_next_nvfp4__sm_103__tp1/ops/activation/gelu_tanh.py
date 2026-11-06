# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-gated tanh GELU -- HuggingFace's `gelu_pytorch_tanh` -- the vision
tower's block-MLP activation.

Tier-3 composed: one `torch.nn.functional.gelu(x, approximate="tanh")` call.
No registered op computes a plain non-gated GELU over an unquantized tensor:
`flashinfer_gelu_tanh_and_mul` is the gated form (and silently returns a
half-width `gelu(a) * b` when handed this tower's non-gated activand),
`moe_gelu` is the erf GELU behind MoE tile metadata, and the
`cute_dsl_nvfp4_dense_gemm_gelu*` ops are NVFP4 GEMM epilogues.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The block MLP's `linear_fc1` is `[4304, 1152]`: one width, nothing split.
INTERMEDIATE_SIZE = 4304

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


class GeluTanh(OpWrapper):
    """`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`, elementwise,
    any shape, new tensor out in `x`'s dtype.

    Fusion boundary: the activation only. `linear_fc1` before it and
    `linear_fc2` after it are the caller's; there is no gate half and no
    multiply. This is the tanh GELU and not the erf one (`GeluErf`, the patch
    merger's); which one a module wants is a property of the checkpoint.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: shape, dtype; `dist` "uniform" over [lo, hi] (default) or "fc1" (a
    # unit-normal hidden state through an fc1 at the checkpoint's measured
    # weight std 0.0197 / bias std 1.676).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="activand built through fc1 at the checkpoint's measured scale",
            spec=dict(shape=(512, INTERMEDIATE_SIZE), dtype=torch.bfloat16, dist="fc1"),
        ),
        Cell(
            why="block MLP activand at the tower call ceiling, 32768 patch rows x 4304 bf16",
            spec=dict(shape=(32768, INTERMEDIATE_SIZE), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
        ),
        Cell(
            why="block MLP activand, 1 patch row x 4304 bf16",
            spec=dict(shape=(1, INTERMEDIATE_SIZE), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
        ),
    )

    note = """
    Metadata consumed: none. Out-of-place; `x` is not mutated.

    Certified on contiguous bf16 `[rows, 4304]` activands (what the block MLP's
    fc1 returns) at 1, 512 and 32768 rows (32768, the tower call ceiling).

    Identity is pinned by fp32 only (the identity test).
    Max |tanh GELU - erf GELU| is 4.73e-4 (|x| ~ 2.7); in fp32 the erf form
    lands ~1600x the correct path's deviation away, 31x outside the default
    gate. In bf16 one ulp at the output scale exceeds that gap (erf variant
    8.19e-3 vs correct 7.75e-3 max abs), so a bf16 match does not say which GELU
    was computed.

    Measured, not certified: `-inf -> nan` (the closed form does the same: -inf
    times a factor saturated to 0); `+inf -> +inf`; large negatives give
    `-0.0`; `finfo.max` comes back unchanged in bf16/fp16/fp32 (the cube is not
    taken in the input dtype); CUDA-graph capturable; the output is a fresh
    allocation.

    Not certified: fp32 beyond the identity test, non-contiguous inputs, fp16,
    other widths, float8 operands. No index-typed argument; numel past 2^31
    not probed.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """The closed form in fp64 from primitive ops, rounded once to `x.dtype`."""
        x64 = x.double()
        inner = _SQRT_2_OVER_PI * (x64 + 0.044715 * x64 * x64 * x64)
        return (0.5 * x64 * (1.0 + torch.tanh(inner))).to(x.dtype)


gelu_tanh = GeluTanh()
