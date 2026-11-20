# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL target's single-GPU FP8-weight GEMM entries over their certified cells:
``k3_fp8w_lowm_gemm_ull.CELLS``, ``k3_fp8w_mid_gemm.CELLS`` and
``k3_fp8w_lowm_mxfp8_gemm_ull.CELLS``, each driven through ``validating`` and checked against the
entry's own ``reference`` with the band of ``test_k3_fp8w_gemm_cells.py``.

The weights are random E4M3 codes with power-of-two 128 x 128 block scales, and the activation rows
have a different magnitude per 32-column group, so every UE8M0 exponent path is exercised.
"""

import importlib

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402

K3 = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (103, 107),
    reason="the ULL low-M FP8-weight entries are certified on SM103 / SM107",
)

lowm = importlib.import_module(f"{K3}.kernels.k3r_gemm.k3_fp8w_lowm_gemm")
mx = importlib.import_module(f"{K3}.kernels.k3r_gemm.k3_fp8w_lowm_mxfp8_ull")
k3_fp8w_lowm_gemm_ull = importlib.import_module(
    f"{K3}.ops.gemm.k3_fp8w_lowm_gemm_ull"
).k3_fp8w_lowm_gemm
k3_fp8w_mid_gemm = importlib.import_module(f"{K3}.ops.gemm.k3_fp8w_mid_gemm").k3_fp8w_mid_gemm
k3_fp8w_lowm_mxfp8_gemm_ull = importlib.import_module(
    f"{K3}.ops.gemm.k3_fp8w_lowm_mxfp8_gemm_ull"
).k3_fp8w_lowm_mxfp8_gemm

#: The band of every BF16 GEMM here. Kernel and reference multiply the same E4M3 operands with
#: power-of-two scales, so every product is exact; they differ only in how the products are summed
#: (FP32 split-K partials against FP64), which leaves the sums within ~1e-6 of each other, far
#: below a BF16 ulp. The single rounding to BF16 can then fall on the other side of a boundary:
#: one BF16 ulp of the element, which is at most 2 units of ``ULP`` (2^-8) of the row's largest
#: element. RMS: such flips are rare, well under 1 unit.
ELEMENT_ULP, RMS_ULP = 2, 1


def _ids(cells):
    return [c.why[:48] for c in cells]


def _weight(n: int, k: int, seed: int):
    """UE8M0-resmoothed E4M3 ``[n, k]`` weight and its DeepGEMM-packed scale."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    blocks = (-(-n // 128), k // 128)
    scale = torch.exp2(torch.randint(-8, -2, blocks, device="cuda", generator=gen).float())
    return lowm.prepare_k3_fp8w_lowm_weight(w, scale)


def _rows(m: int, k: int, seed: int) -> torch.Tensor:
    """BF16 ``[m, k]`` with one magnitude per 32-column group (2^-6 .. 2^2)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=gen)
    mag = torch.randint(-6, 3, (m, k // 32), device="cuda", generator=gen).float()
    return (x * torch.exp2(mag).repeat_interleave(32, dim=1)).to(torch.bfloat16)


@pytest.mark.parametrize("cell", k3_fp8w_lowm_gemm_ull.CELLS, ids=_ids(k3_fp8w_lowm_gemm_ull.CELLS))
def test_k3_fp8w_lowm_gemm_ull_cells(cell):
    op = k3_fp8w_lowm_gemm_ull
    s = cell.spec
    n, k, m, variant = s["n"], s["k"], s["m"], s["variant"]
    w, w_scale = _weight(n, k, 1)
    x = _rows(m, k, 2)
    kwargs = {}
    if variant in ("out", "gated"):
        kwargs["out"] = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    if variant == "gated":
        gen = torch.Generator(device="cuda").manual_seed(3)
        kwargs["gate"] = torch.rand(m, k, device="cuda", generator=gen).to(torch.bfloat16)
    if variant == "sigmoid":
        kwargs["sigmoid"] = True
    with validating(op):
        got = op(x, w, w_scale, **kwargs)
    if "out" in kwargs:
        assert got.data_ptr() == kwargs["out"].data_ptr()
    want = op.reference(x, w, w_scale, **kwargs)
    assert_within_ulp(got, want, ELEMENT_ULP, RMS_ULP)


@pytest.mark.parametrize("cell", k3_fp8w_mid_gemm.CELLS, ids=_ids(k3_fp8w_mid_gemm.CELLS))
def test_k3_fp8w_mid_gemm_ull_cells(cell):
    op = k3_fp8w_mid_gemm
    n, k, m = cell.spec["n"], cell.spec["k"], cell.spec["m"]
    plan = lowm.mid_prequant_plan(m, n, k)
    assert plan is not None, "a mid-M cell must be above the low-M cut-over"
    tactic = plan[0]
    w, w_scale = _weight(n, k, 4)
    xq, xs = lowm.quantize_mid_operand(_rows(m, k, 5), tactic)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(op):
        op(xq, xs, w, w_scale, out, tactic)
    want = op.reference(xq, xs, w, w_scale, out, tactic)
    assert_within_ulp(out, want, ELEMENT_ULP, RMS_ULP)


@pytest.mark.parametrize(
    "cell", k3_fp8w_lowm_mxfp8_gemm_ull.CELLS, ids=_ids(k3_fp8w_lowm_mxfp8_gemm_ull.CELLS)
)
def test_k3_fp8w_lowm_mxfp8_gemm_ull_cells(cell):
    op = k3_fp8w_lowm_mxfp8_gemm_ull
    s = cell.spec
    n, k, m = s["n"], s["k"], s["m"]
    w, w_scale = _weight(n, k, 6)
    xq, xs = mx.mxfp8_quantize_reference(_rows(m, k, 7))
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(op):
        op(xq, xs, w, w_scale, out, stage_weights=s["stage"])
    want = op.reference(xq, xs, w, w_scale, out, stage_weights=s["stage"])
    assert_within_ulp(out, want, ELEMENT_ULP, RMS_ULP)
