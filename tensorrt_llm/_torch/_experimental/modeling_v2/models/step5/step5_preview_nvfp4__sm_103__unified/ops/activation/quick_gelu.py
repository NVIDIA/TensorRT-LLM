# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QuickGELU, `x * sigmoid(1.702 x)`: the activation of every MLP block of
Step-5's vision tower (`vision_config.hidden_act = "quick_gelu"`).

Tier 4, composed from torch elementwise ops: no engine kernel computes this
function as a standalone activation. Rejected at the cheaper rungs:

- No `torch.ops.trtllm` op is QuickGELU. `trtllm::moe_gelu` is the erf GELU
  over MoE tiles, and the qwen4_exp `gelu_erf` / `gelu_tanh` entries are the
  erf and tanh GELUs, different functions (the test shows the gate refusing
  both).
- FlashInfer's activation module has only gated forms (`silu_and_mul`,
  `gelu_and_mul`, `gelu_tanh_and_mul`).
- `trtllm::silu_and_mul_2in(gate=x, up=zeros, swiglu_alpha=1.702,
  swiglu_beta=1.0)` computes `x * sigmoid(1.702 x) * (0 + 1)` in one launch,
  but only given a contiguous all-zero `up` of `x`'s full shape (its
  contiguity check refuses a broadcast view; read, not driven): a caller-held
  234 MB buffer at 13072 rows, or a fill launch per call. It also ties the
  activation to a SwiGLU op's `(up + beta)` offset. Measured on sm_103 at
  13072 x 8960 bf16: 101.8 us plus 59.4 us for the zero fill, against this
  entry's 850.4 us, and correctly rounded on every finite bf16 input with a
  normal-range result outside the flush tail the `note` describes. Recorded
  for the tuning phase.
"""

from __future__ import annotations

import math

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The gain inside the sigmoid (transformers' `QuickGELUActivation`).
ALPHA = 1.702

#: `vision_config.width` 1536 times `mlp_ratio` 8960 / 1536: the c_fc output
#: width (`resblocks.*.mlp.c_fc.weight` is `[8960, 1536]`). The tower is not
#: tensor-parallel, so every rank sees the full width.
WIDTH = 8960

#: Rows per tower call: one 728 px tile is a 52 x 52 patch grid, a 504 px crop
#: 36 x 36; the largest call the envelope names is an overview plus 8 crops.
OVERVIEW_ROWS = 52 * 52
CROP_ROWS = 36 * 36
LARGEST_ROWS = OVERVIEW_ROWS + 8 * CROP_ROWS

#: Below this `x`, fp32 `exp(-ALPHA * x)` overflows, the sigmoid is 0 and the
#: output is a signed zero (see `note`, Hard limits).
FLUSH_BELOW = -math.log(torch.finfo(torch.float32).max) / ALPHA


def _cell(why: str, rows: int, dist: str = "activand") -> Cell:
    return Cell(why=why, spec=dict(rows=rows, width=WIDTH, dtype=torch.bfloat16, dist=dist))


class QuickGelu(OpWrapper):
    """`out = x * sigmoid(1.702 * x)`, elementwise over any shape; a new tensor
    in `x`'s dtype.

    `x` is widened to fp32, and the scaled argument, the sigmoid and the
    product are evaluated there; only the product is rounded back, once.
    transformers' `QuickGELUActivation` (the reference implementation's
    `ACT2FN["quick_gelu"]`) instead rounds to bf16 after each of its three ops;
    this entry is the more accurate of the two (numbers in `note`).

    Fusion boundary: the activation only. The c_fc projection before it and
    c_proj after it are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, width, dtype, dist. "activand": bulk N(0, 3^2) with every
    # eighth column uniform over [-48, 48] (both saturation tails, short of the
    # flush tail). "every_bf16": all 65536 bf16 bit patterns once (NaNs, +-inf,
    # subnormals, the flush tail), then repeated to fill the rows.
    CELLS: tuple[Cell, ...] = (
        _cell("crop tile, 1296 rows x 8960", CROP_ROWS),
        _cell("overview tile, 2704 rows x 8960", OVERVIEW_ROWS),
        _cell("overview + 8 crops, 13072 rows: the envelope's largest call", LARGEST_ROWS),
        _cell("every bf16 bit pattern once: the whole input domain", 8, dist="every_bf16"),
    )

    note = """
    Metadata consumed: none. Out-of-place; `x` is not mutated (asserted per
    cell). Five elementwise launches (widen, scale, sigmoid, multiply, narrow)
    with two fp32 temporaries of `x`'s shape live. Measured on sm_103 at
    1296 / 2704 / 13072 rows x 8960: 70.7 / 181.5 / 850.4 us, and 1.091 GiB
    peak above the input (output included) at 13072 rows. The tuning phase
    replaces it with one fused kernel.

    Numerics, measured over all 65280 finite bf16 inputs: outside the flush
    tail every output is the correctly rounded exact value except x = -11.6875,
    one ulp off (its exact value lies 1.1e-6 relative from a bf16 rounding
    midpoint). The fp32 evaluation's relative error is at most 4.75e-6 (at
    most 0.812 of the bound the test derives). The cells' activands are off
    by one ulp on at most 9964 of 117125114 elements (13072 rows), all at such
    midpoints. transformers' bf16 chain is off on 1305 of the 49111 finite
    inputs with a normal-range result, up to 45 ulp on the cells' activand;
    the test's gate refuses it, as it refuses the erf and tanh GELUs, SiLU and
    a 1.7 gain.

    Specials (certified by the every-bf16 cell, op and reference agree): NaN
    -> NaN; +inf -> +inf; -inf -> NaN (0 * -inf); bf16 max -> unchanged;
    a subnormal result -> within one bf16 subnormal step of the reference.

    Hard limits
      flush tail  for x < FLUSH_BELOW (-52.13; bf16 -52.25 and below) fp32
      `exp(-1.702 x)` overflows and the output is a signed zero; the exact
      value there is below 1.25e-37 in magnitude, nonzero in bf16 down to
      x = -56.75 (19 inputs, -56.75 through -52.25, measured).
      numel of `x`  torch's elementwise kernels split or widen past 2^31
      elements; the largest cell is 13072 x 8960 = 1.17e8, and 2^31 elements
      at width 8960 is 239674 rows, far past the envelope.

    Certified: contiguous 2-D bf16 `[rows, 8960]` at 1296, 2704 and 13072
    rows, and every bf16 value through the every-bf16 cell.

    Not certified: other row counts (the op is shape-generic elementwise, but
    only these were driven), 3-D or non-contiguous inputs, fp16 / fp32 / fp8 /
    integer inputs (an integer `x` returns the activation truncated to that
    dtype, as the reference does), CUDA-graph capture (the call site is eager),
    torch.compile.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        gate = xf.mul(ALPHA).sigmoid_()
        return gate.mul_(xf).to(x.dtype)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """The closed form `x / (1 + exp(-1.702 x))` in fp64 from primitive ops,
        rounded once to `x.dtype` (so an fp64 `x` gives the exact value).

        The sigmoid is written out with `exp` rather than taken from
        `torch.sigmoid`, the call `__call__` makes.
        """
        x64 = x.double()
        return (x64 / (1.0 + torch.exp(-ALPHA * x64))).to(x.dtype)


quick_gelu = QuickGelu()
