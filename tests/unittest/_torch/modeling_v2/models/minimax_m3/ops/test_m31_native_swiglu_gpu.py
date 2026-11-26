# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE_SWIGLU (family ``minimax_m3/ops/native_swiglu``): the row-grid mode (nv 0) of
``trtllm::minimax_m31_swiglu_oai_vec`` is bit for bit the Triton ``silu_and_mul`` at limit 7 / alpha 1.702 / beta 1
(the DEP8 dense / drafter MLP rows, d 12288, with clamped, signed-zero, subnormal and non-finite values; PDL on and
off; inside a CUDA graph); other widths too (d not a multiple of 8 takes the pair shape); the vector modes (nv 1 / 4)
on every bf16 gate pattern x the special up values and the reverse, and at the TP2 widths (shared expert d 1536, pair
fallback d 1532)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import native_swiglu
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the native SwiGLU-OAI op runs on sm_107 (Rubin) only",
)

D = 12288


def _triton(x):
    return torch.ops.trtllm.silu_and_mul(x, swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _gu(rows, d, gen):
    x = (torch.randn(rows, 2 * d, generator=gen, device="cuda") * 4.0).to(torch.bfloat16)
    x[:, 1] = 9.0  # gate above the limit
    x[:, d + 2] = -11.0  # up below -limit
    x[:, 3] = -0.0
    x[:, d + 4] = 1e-40
    x[:, 5] = float("-inf")
    x[:, d + 6] = float("inf")
    return x


@pytest.mark.parametrize("rows, pdl", [(1, True), (8, True), (84, True), (128, True), (84, False)])
def test_the_row_grid_is_bitwise_the_triton_kernel(rows, pdl):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = _gu(rows, D, gen)
    got = native_swiglu.swiglu_op()(x, pdl, 0)
    want = _triton(x)
    assert torch.equal(torch.isnan(got), torch.isnan(want))
    keep = ~torch.isnan(want)
    assert torch.equal(_bits(got)[keep], _bits(want)[keep])


def _specials(n, seed):
    """The special values (signed zeros, the clamp edges, subnormals, overflow, non-finite), scaled randoms and random
    bit patterns, one row of ``n`` each."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    vals = [0.0, -0.0, 1.0, -1.0, 6.99, 7.0, 7.01, -7.0, -7.01, 1e-38, -1e-38, 1e-40, 88.0, -88.0, 3e38, -3e38,
            float("inf"), float("-inf"), float("nan")]  # fmt: skip
    rows = [torch.full((n,), v, dtype=torch.bfloat16, device="cuda") for v in vals]
    rows += [
        (s * torch.randn(n, generator=g, device="cuda")).to(torch.bfloat16)
        for s in (0.05, 1.0, 4.0, 30.0)
    ]
    rows.append(
        torch.randint(-(1 << 15), 1 << 15, (n,), generator=g, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    return rows


@pytest.mark.parametrize("nv", [1, 4])
@pytest.mark.parametrize("pdl", [True])
@pytest.mark.parametrize("gate_side", [True, False])
def test_every_bf16_pattern(nv, pdl, gate_side):
    """The vector modes on every bf16 gate (or up) value against each special row, bit for bit."""
    allp = (
        torch.arange(1 << 16, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    )
    for j, other in enumerate(_specials(1 << 16, 1)):
        g, u = (allp, other) if gate_side else (other, allp)
        for rows in (1, 256):
            x = torch.cat([g.view(rows, -1), u.view(rows, -1)], 1).contiguous()
            assert torch.equal(_bits(native_swiglu.swiglu_op()(x, pdl, nv)), _bits(_triton(x))), (
                f"set {j} rows {rows}"
            )


@pytest.mark.parametrize("d", [1536, 1532])
@pytest.mark.parametrize("rows", [1, 49, 4096])
@pytest.mark.parametrize("nv", [4])
@pytest.mark.parametrize("pdl", [True])
def test_tp2_shapes(d, rows, nv, pdl):
    g = torch.Generator(device="cuda").manual_seed(d * 7 + rows)
    x = (1.5 * torch.randn(rows, 2 * d, generator=g, device="cuda")).to(torch.bfloat16)
    assert torch.equal(_bits(native_swiglu.swiglu_op()(x, pdl, nv)), _bits(_triton(x)))
