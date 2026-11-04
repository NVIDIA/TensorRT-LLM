# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-gated exact (erf) GELU -- what `torch.nn.GELU()` is.

Two call sites, both in a vision tower: the qwen4_exp patch merger's
activation (`merger.linear_fc1` 4608 -> 4608, this GELU, `linear_fc2`), which
no config key names, and the Mistral Large 4 vision-language adapter's (`w_in`
2048 -> 6144, this GELU, `w_out` 6144 -> 6144, no biases).

Tier 4, torch composed: one `torch.nn.functional.gelu(x)` call,
`approximate="none"`, which launches ATen's `GeluCUDAKernelImpl` elementwise
kernel once. The cheaper rungs were rejected:

- tier 1: `torch.ops.trtllm.moe_gelu` computes this function bitwise but is
  not a substitute. Its `tile_idx_to_mn_limit` / `num_non_exiting_tiles`
  operands must be int32 *device* tensors (on the host it raises nothing and
  returns garbage, nan, or an illegal memory access), so wrapping it means
  launching two fills before the call. It also rejects fp32 and non-2-D
  inputs. `cute_dsl_nvfp4_dense_gemm_gelu_blackwell` fuses a GELU into an
  NVFP4 GEMM epilogue, which `_torch/modules/mlp.py` calls GELU(tanh); the
  adapter's `w_in` is a BF16 weight. `flashinfer_gelu_tanh_and_mul` is gated,
  and is the tanh form.
- tier 2: no module callable under `_torch/modules` computes a non-gated erf
  GELU. `gelu_tanh` in `_torch/utils.py` is `F.gelu(approximate="tanh")`, the
  other GELU.
- tier 3: `flashinfer.activation` offers only gated forms (`gelu_and_mul`,
  `gelu_tanh_and_mul`).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .._op import Arch, Cell, OpWrapper

#: qwen4_exp: `merger.linear_fc1` is `[4608, 4608]`, spatial merge 2 over a
#: 1152-wide tower.
MERGER_WIDTH = 4608

#: qwen4_exp: rows the merger is asked for, `sum(t*h*w) / 4`. 64 is a 224x224
#: image (grid [1, 16, 16]) and 140 a 336x448 one ([1, 20, 28]); neither is a
#: multiple of any MoE tile size.
MERGER_ROW_COUNTS = (1, 7, 64, 65, 140, 1024, 4096)

#: Mistral Large 4: the vision-language adapter's `w_in` is `[6144, 2048]`, so the
#: GELU runs at width 6144 on `w_in`'s bf16 output.
ADAPTER_WIDTH = 6144

#: Mistral Large 4: rows the adapter is asked for, one per merged image token.
#: 256 is a 448x448 image (32x32 patches of 14, merged 2x2). 3025 is one image
#: at the 1540-pixel maximum (110x110 patches, merged to 55x55). 6050 is two
#: of those. 30250 is ten maximum-size images, leaving room for the native
#: row delimiters, text and output within the 32768-token envelope.
ADAPTER_ROW_COUNTS = (1, 48, 60, 108, 256, 3025, 6050, 8192, 30250)

_SQRT_2 = math.sqrt(2.0)


class GeluErf(OpWrapper):
    """`0.5 * x * (1 + erf(x / sqrt(2)))`, elementwise, any shape, new tensor
    out in `x`'s dtype.

    Fusion boundary: the activation only. The projection before it (qwen4_exp
    `merger.linear_fc1`, Mistral Large 4 `vision_language_adapter.w_in`) and the
    one after it (`linear_fc2`, `w_out`) are the caller's; there is no gate
    half and no multiply. This is the erf GELU and not the tanh one: qwen4_exp's
    block MLPs use the tanh one, in the same tower.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: shape, dtype; `dist` "uniform" over [lo, hi] (default) or "merger"
    # (a tower hidden state scaled by the merger norm's measured gamma, grouped
    # 4-to-1, through an fc1 at the checkpoint's measured weight/bias scale);
    # `layout` "contiguous" (default), "column_slice" or "transposed".
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"merger activand, {rows} merged rows x 4608 bf16",
                spec=dict(shape=(rows, MERGER_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
            )
            for rows in MERGER_ROW_COUNTS
        )
        + (
            Cell(
                why="activand built the way the merger builds it",
                spec=dict(shape=(512, MERGER_WIDTH), dtype=torch.bfloat16, dist="merger"),
            ),
            Cell(
                why="fp16 over [-16, 16]",
                spec=dict(shape=(256, MERGER_WIDTH), dtype=torch.float16, lo=-16.0, hi=16.0),
            ),
            Cell(
                why="fp32 over [-16, 16]",
                spec=dict(shape=(256, MERGER_WIDTH), dtype=torch.float32, lo=-16.0, hi=16.0),
            ),
            Cell(
                why="bfloat16 over [-3, 3], the erf/tanh divergence peak",
                spec=dict(shape=(256, MERGER_WIDTH), dtype=torch.bfloat16, lo=-3.0, hi=3.0),
            ),
            Cell(
                why="float16 over [-3, 3], the erf/tanh divergence peak",
                spec=dict(shape=(256, MERGER_WIDTH), dtype=torch.float16, lo=-3.0, hi=3.0),
            ),
            Cell(
                why="float32 over [-3, 3], the erf/tanh divergence peak",
                spec=dict(shape=(256, MERGER_WIDTH), dtype=torch.float32, lo=-3.0, hi=3.0),
            ),
            Cell(
                why="column slice of a wider buffer: row stride free",
                spec=dict(
                    shape=(64, MERGER_WIDTH),
                    dtype=torch.bfloat16,
                    lo=-16.0,
                    hi=16.0,
                    layout="column_slice",
                ),
            ),
            Cell(
                why="transposed buffer: output keeps the input's strides",
                spec=dict(
                    shape=(64, MERGER_WIDTH),
                    dtype=torch.bfloat16,
                    lo=-16.0,
                    hi=16.0,
                    layout="transposed",
                ),
            ),
            Cell(
                why="1-D input",
                spec=dict(shape=(MERGER_WIDTH,), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
            ),
            Cell(
                why="3-D input",
                spec=dict(shape=(2, 37, MERGER_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
            ),
            Cell(
                why="zero-row batch",
                spec=dict(shape=(0, MERGER_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
            ),
        )
        + tuple(
            Cell(
                why=f"Mistral Large 4 adapter, {rows} image tokens x 6144 bf16",
                spec=dict(shape=(rows, ADAPTER_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
            )
            for rows in ADAPTER_ROW_COUNTS
        )
    )

    note = """
    Metadata consumed: none. Out-of-place; `x` is not mutated. CUDA-graph
    capturable; the output is a fresh allocation.

    Call sites: the qwen4_exp vision merger at width 4608, and the Mistral Large 4
    vision-language adapter at width 6144 (`w_in`'s contiguous bf16 output,
    eager, one row per merged image token).

    The output inherits `x`'s memory format: a transposed `x` returns a
    non-contiguous result (bitwise equal to the call on a contiguous copy).
    Rows are independent: two images' rows concatenated give, bitwise, the two
    calls' outputs concatenated.

    Numerics. The kernel evaluates `1 + erf` in fp32 and rounds once to the
    output dtype.
    - bf16: every finite bf16 input with |x| <= 4 (33026 values, driven
      exhaustively) returns the correctly rounded value, bitwise. Below -4 the
      fp32 `1 + erf` cancels, and 198 of the bf16 inputs there are not
      correctly rounded: the error stays under |x| * 2^-21 in absolute terms
      but is many ulp of the tiny result. From x = -5.5625 down (exact value
      -7.4e-8 there) the output is -0.0.
    - fp16: 59 of the 34818 fp16 inputs with |x| <= 4 are not correctly
      rounded; near the cancellation the output is up to 1.94 ulp off.
    - fp32: up to 143 ulp relative error near x = -3, and a zero output deep
      in the negative tail, from the same cancellation.
    Every finite bf16 and fp16 input lies within one ulp + |x| * 2^-21 of the
    exact value, at worst half of that allowance.

    Identity. Max |erf GELU - tanh GELU| is 4.73e-4 at x = -2.699. In fp32 the
    tanh form lands about 1400x the correct path's deviation away, 31x outside
    torch's default gate. In bf16 it lands 15x outside the test's per-element
    ulp gate over [-3, 3] and 5.7x outside torch's default bf16 band, so a bf16
    match does tell the two apart on inputs that reach about [-3.5, -2]. The max-abs deviation alone cannot: at
    the output's scale one bf16 ulp exceeds the gap. An erf scale of 1/1.414
    instead of 1/sqrt(2) stays inside one bf16 ulp. Only the bf16
    correctly-rounded count refuses it, at 3.87% of elements over [-3, 3].

    Values: `-inf -> nan` (the closed form does the same); `+inf -> +inf`;
    large negatives give `-0.0`; `finfo.max` comes back unchanged.

    Not certified: float8 operands; rows above 8192 at width 6144.

    Hard limits
      numel  no index-typed argument; the largest certified call is
             8192 x 6144 = 50331648 elements  boundary 2^31  not probed
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """The closed form in fp64, rounded once to `x.dtype`. Written with
        `erfc(-x / sqrt(2))`: `1 + erf(x / sqrt(2))` cancels in fp64 too, and
        returns 0 instead of the tiny negative value below about x = -8.3. An
        fp64 `x` comes back unrounded, the exact value a ulp gate needs."""
        x64 = x.double()
        return (0.5 * x64 * torch.erfc(-x64 / _SQRT_2)).to(x.dtype)


gelu_erf = GeluErf()
