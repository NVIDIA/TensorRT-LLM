# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-gated tanh GELU -- HuggingFace's `gelu_pytorch_tanh` -- the vision
tower's block-MLP activation.

Tier 4 (native torch), a macro entry of one call: `torch.nn.functional.gelu(x,
approximate="tanh")`, the aten elementwise kernel. Written for the tp1 target
(`qwen3_8_flash_next_nvfp4__sm_103__tp1`, `ops/activation/gelu_tanh`) and kept
unchanged here. No registered op computes a plain non-gated GELU over an
unquantized tensor: `flashinfer_gelu_tanh_and_mul` (and FlashInfer's
`gelu_tanh_and_mul` / `gelu_and_mul`) is the gated form, and silently returns a
half-width `gelu(a) * b` when handed this tower's non-gated activand.
`moe_gelu` is the erf GELU behind MoE tile metadata, and the
`cute_dsl_nvfp4_dense_gemm_gelu*` ops are NVFP4 GEMM epilogues (this tower's
fc1 is bf16).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .._op import Arch, Cell, OpWrapper

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
            why="fc1-scaled activand at the tower call ceiling, 32768 patch rows",
            spec=dict(shape=(32768, INTERMEDIATE_SIZE), dtype=torch.bfloat16, dist="fc1"),
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
    Metadata consumed: none. Out-of-place; `x` is not mutated. Call site: the
    vision tower's block MLP, `gelu_tanh(cublas_mm(layer_norm(h), fc1, fc1_b))`,
    eager only, at most 32768 patch rows a call (`_VISION_CALL_MAX_PATCHES`).
    The tower is rank-invariant, so every member (tp4, tep4, dep4) calls it at
    the unsplit width 4304.

    Source: the aten CUDA kernel compiled into torch (no JIT path). One launch a
    call at every row count measured (1, 512, 32768): `vectorized_elementwise_kernel<8,
    GeluCUDAKernelImpl ...>`, computing in fp32 and rounding once to bf16.

    Certified on contiguous bf16 `[rows, 4304]` activands (what the block MLP's
    fc1 returns) at 1, 512 and 32768 rows (32768, the tower call ceiling): fc1-scaled
    activands (x std ~1.8, |x| <= 8.8) at 512 and 32768 rows, uniform [-16, 16]
    at 1 and 32768. Gate: torch's bf16 default against the fp64 reference.

    Error, measured on GB300 at every cell: within one bf16 ulp of the reference
    everywhere except x in [-7.19, -4.56]. There fp32 cancellation in `1 + tanh`
    leaves the output up to 254 ulp of its own size off, at |ref| <= 3.6e-6 and
    abs error <= 7.6e-8 (8% of the uniform cells' elements, 0.5% of the fc1
    ones). HF's `gelu_pytorch_tanh` is this same call. x >= 0 is bitwise the
    reference (a 0..10 sweep). Against the unrounded fp64 form, the fc1 cells
    measure cosine 0.999999 and relative L2 1.4e-3 (bf16 output rounding), max
    abs 7.755e-3 (half an ulp at |y| in [2, 4)).

    Time a call (eager, 100 calls): 8.2 / 7.9 / 125 us at 1 / 512 / 32768 rows.
    Peak memory of a 32768-row cell (call plus fp64 reference): 5.5 GiB.

    Identity is pinned by fp32 only (the identity test).
    Max |tanh GELU - erf GELU| is 4.73e-4 (|x| ~ 2.7); in fp32 the erf form
    lands ~1600x the correct path's deviation away, 31x outside the default
    gate. In bf16 one ulp at the output scale exceeds that gap (erf variant
    8.19e-3 vs correct 7.75e-3 max abs), so a bf16 match does not say which GELU
    was computed.

    Measured, not certified: `-inf -> nan` (the closed form does the same: -inf
    times a factor saturated to 0); `+inf -> +inf`; large negatives give
    `-0.0`; `finfo.max` comes back unchanged in bf16/fp16/fp32 (the cube is not
    taken in the input dtype); CUDA-graph capturable, replay bitwise eager at 1,
    512 and 32768 rows (the call site is eager); the output is a fresh
    allocation.

    Not certified: fp32 beyond the identity test, non-contiguous inputs, fp16,
    other widths, float8 operands, row counts between the cells. No index-typed
    argument; numel past 2^31 (498951 rows of 4304) not probed, 15x past the
    tower's 32768-row call.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """The closed form in fp64 from primitive ops, rounded once to `x.dtype`."""
        x64 = x.double()
        inner = _SQRT_2_OVER_PI * (x64 + 0.044715 * x64 * x64 * x64)
        return (0.5 * x64 * (1.0 + torch.tanh(inner))).to(x.dtype)


gelu_tanh = GeluTanh()
