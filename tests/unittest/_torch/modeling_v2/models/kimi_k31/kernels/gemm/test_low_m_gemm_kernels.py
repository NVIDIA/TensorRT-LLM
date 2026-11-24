# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's low-M GEMM kernels (``k31_low_m_gemm``): every kernel kind against FP32."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm.low_m_gemm import (
    encode_choice,
    k31_low_m_gemm,
    plan_choice,
    validate_choice,
)

assert torch.cuda.is_available(), "the low-M GEMM kernels require a CUDA device"


def _operands(m, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    return a, w


def _check(out, a, w):
    with true_fp32_matmul():
        ref = a.float() @ w.float().t()
    # FP32 accumulation in another order, one BF16 rounding: within two BF16 steps of the row's scale.
    scale = ref.abs().amax(dim=1, keepdim=True)
    assert torch.all((out.float() - ref).abs() <= scale * 2.0**-7 + 1e-6)


@pytest.mark.parametrize(
    "choice,m,n,k",
    (
        (("direct", 128, 2), 1, 3072, 7168),
        (("direct", 32, 4), 2, 4672, 1536),
        (("splitk", 64, 8, 4, 10), 8, 1536, 7168),
        (("splitk", 64, 32, 2, 9), 32, 3072, 7168),
        # wide token tiles of the chunked split-K: several token tiles, a weight tile past N, front-heavy chunks
        (("splitkc", 64, 24, 4, 8), 80, 136, 7168),
        (("splitkc", 64, 32, 4, 8), 128, 136, 7168),
        (("splitkc", 64, 48, 3, 6), 100, 768, 7168),
        (("tgv", 64, 16, 12, True), 8, 4672, 1536),
        (("tgv", 64, 32, 11, True), 72, 3072, 7168),
        (("tgv", 128, 16, 6, True), 5, 7168, 768),
        (("tgvchunk", 16, 10, 24, 48, 72), 3, 7168, 12288),
        (("tgvchunk", 80, 8, 48), 80, 7168, 12288),
        (("tgvchunk", 32, 10, 8, 16, 24, 32, 40, 48), 2, 3072, 7168),
        (("tgvchunkm", 128, 16, 6, 8, 16, 24, 32, 40, 48), 5, 12288, 7168),
        (("tgvchunkm", 128, 32, 6, 19, 38), 25, 12288, 7168),
        (("tgvchunkm", 64, 32, 10, 24, 48, 72), 20, 7168, 12288),
    ),
)
@pytest.mark.parametrize("pdl", (False, True))
def test_kernel_kinds(choice, m, n, k, pdl) -> None:
    validate_choice(choice, m, n, k)
    a, w = _operands(m, n, k, seed=m * 131 + n)
    kind, params = encode_choice(choice)
    out = k31_low_m_gemm(a, w, kind, params, pdl)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    _check(out, a, w)


def test_parameter_weight() -> None:
    """A module parameter (requires_grad) is read through its detached view."""
    a, w = _operands(8, 1536, 7168, seed=3)
    param = torch.nn.Parameter(w)
    kind, params = encode_choice(("splitk", 64, 8, 4, 10))
    _check(k31_low_m_gemm(a, param, kind, params, True), a, w)


def test_plan_lookup() -> None:
    plan = {
        (3072, 7168): ((2, ("direct", 128, 2)), (8, ("default",)), (32, ("splitk", 64, 32, 2, 9)))
    }
    assert plan_choice(plan, 1, 3072, 7168) == ("direct", 128, 2)
    assert plan_choice(plan, 2, 3072, 7168) == ("direct", 128, 2)
    assert plan_choice(plan, 5, 3072, 7168) is None  # a "default" run stays on the caller's GEMM
    assert plan_choice(plan, 32, 3072, 7168) == ("splitk", 64, 32, 2, 9)
    assert plan_choice(plan, 33, 3072, 7168) is None  # above the last bound
    assert plan_choice(plan, 8, 3072, 1536) is None  # a shape the plan does not list
    assert plan_choice(None, 8, 3072, 7168) is None


def test_validate_refuses() -> None:
    with pytest.raises(ValueError):
        validate_choice(("splitk", 64, 8, 4, 10), 33, 1536, 7168)  # split-K serves M <= 32
    with pytest.raises(ValueError):
        validate_choice(("tgv", 64, 16, 99, True), 8, 1536, 7168)  # no such TGV config
    with pytest.raises(ValueError):
        validate_choice(("tgv", 64, 16, 12, True), 8, 7168, 192)  # K not a multiple of 128
    with pytest.raises(ValueError):
        validate_choice(
            ("tgvchunk", 64, 8, 24, 48, 72, 84), 8, 7168, 12288
        )  # 5 chunks x 64 columns: no TMEM
    with pytest.raises(ValueError):
        validate_choice(("tgvchunk", 16, 8, 48, 24), 8, 7168, 12288)  # bounds not increasing
    with pytest.raises(ValueError):
        validate_choice(("tgvchunk", 16, 8, 96), 8, 7168, 12288)  # a bound at the last K tile
    with pytest.raises(ValueError):
        validate_choice(
            ("tgvchunkm", 96, 16, 6, 19, 38), 8, 12288, 7168
        )  # 64- or 128-row tiles only
    with pytest.raises(ValueError):
        validate_choice(
            ("tgvchunkm", 128, 16, 6, 19, 38), 8, 3200, 7168
        )  # N not whole 256-row clusters
