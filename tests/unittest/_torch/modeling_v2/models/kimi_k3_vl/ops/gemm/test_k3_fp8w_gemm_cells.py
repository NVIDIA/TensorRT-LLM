# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The single-GPU FP8-weight GEMM entries of the K3 default target over their certified cells:
``k3_fp8w_lowm_gemm.CELLS``, ``k3_fp8w_mid_gemm.CELLS``, ``k3_fp8w_lowm_mxfp8_gemm.CELLS`` and
``k3_latent_down_fp4.CELLS``, each driven through ``validating`` and checked against the entry's
own ``reference``. (``k3_mxfp8_gemm.CELLS`` run in ``ops/gemm/test_core_mxfp8_gemm.py``, on SM107.)

The weights are random E4M3 codes with power-of-two 128 x 128 block scales, and the activation rows
have a different magnitude per 32-column group, so every UE8M0 exponent path is exercised.
"""

import importlib

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402

K3 = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the low-M FP8-weight kernels run on SM100 / SM103 / SM107",
)

lowm = importlib.import_module(f"{K3}.kernels.k3r_gemm.k3_fp8w_lowm_gemm")
mx = importlib.import_module(f"{K3}.kernels.k3r_gemm.k3_fp8w_lowm_mxfp8")
ref = importlib.import_module(f"{K3}.ops.gemm._fp8_reference")
k3_fp8w_lowm_gemm = importlib.import_module(f"{K3}.ops.gemm.k3_fp8w_lowm_gemm").k3_fp8w_lowm_gemm
k3_fp8w_mid_gemm = importlib.import_module(f"{K3}.ops.gemm.k3_fp8w_mid_gemm").k3_fp8w_mid_gemm
k3_fp8w_lowm_mxfp8_gemm = importlib.import_module(
    f"{K3}.ops.gemm.k3_fp8w_lowm_mxfp8_gemm"
).k3_fp8w_lowm_mxfp8_gemm
k3_latent_down_fp4 = importlib.import_module(f"{K3}.ops.gemm.k3_latent_down_fp4").k3_latent_down_fp4

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


@pytest.mark.parametrize("cell", k3_fp8w_lowm_gemm.CELLS, ids=_ids(k3_fp8w_lowm_gemm.CELLS))
def test_k3_fp8w_lowm_gemm_cells(cell):
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
    with validating(k3_fp8w_lowm_gemm):
        got = k3_fp8w_lowm_gemm(x, w, w_scale, **kwargs)
    if "out" in kwargs:
        assert got.data_ptr() == kwargs["out"].data_ptr()
    want = k3_fp8w_lowm_gemm.reference(x, w, w_scale, **kwargs)
    assert_within_ulp(got, want, ELEMENT_ULP, RMS_ULP)


@pytest.mark.parametrize("cell", k3_fp8w_mid_gemm.CELLS, ids=_ids(k3_fp8w_mid_gemm.CELLS))
def test_k3_fp8w_mid_gemm_cells(cell):
    n, k, m = cell.spec["n"], cell.spec["k"], cell.spec["m"]
    plan = lowm.mid_prequant_plan(m, n, k)
    assert plan is not None, "a mid-M cell must be above the low-M cut-over"
    tactic = plan[0]
    w, w_scale = _weight(n, k, 4)
    xq, xs = lowm.quantize_mid_operand(_rows(m, k, 5), tactic)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(k3_fp8w_mid_gemm):
        k3_fp8w_mid_gemm(xq, xs, w, w_scale, out, tactic)
    want = k3_fp8w_mid_gemm.reference(xq, xs, w, w_scale, out, tactic)
    assert_within_ulp(out, want, ELEMENT_ULP, RMS_ULP)


@pytest.mark.parametrize(
    "cell", k3_fp8w_lowm_mxfp8_gemm.CELLS, ids=_ids(k3_fp8w_lowm_mxfp8_gemm.CELLS)
)
def test_k3_fp8w_lowm_mxfp8_gemm_cells(cell):
    s = cell.spec
    n, k, m = s["n"], s["k"], s["m"]
    w, w_scale = _weight(n, k, 6)
    xq, xs = mx.mxfp8_quantize_reference(_rows(m, k, 7))
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(k3_fp8w_lowm_mxfp8_gemm):
        k3_fp8w_lowm_mxfp8_gemm(xq, xs, w, w_scale, out, stage_weights=s["stage"])
    want = k3_fp8w_lowm_mxfp8_gemm.reference(xq, xs, w, w_scale, out, stage_weights=s["stage"])
    assert_within_ulp(out, want, ELEMENT_ULP, RMS_ULP)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _nvfp4_values(codes, scales):
    """FP64 values of NVFP4 ``codes [M, N / 2]`` / linear E4M3 ``scales [M, N / 16]`` in the scaled
    domain (``global_sf`` not divided out, so every value is exact) and the per-element scale."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    c = codes.long()
    e2m1 = torch.stack([lut[c & 0xF], lut[c >> 4]], dim=-1).flatten(-2)
    s = scales.view(torch.float8_e4m3fn).to(torch.float64).repeat_interleave(16, dim=1)
    return e2m1 * s, s


@pytest.mark.parametrize("cell", k3_latent_down_fp4.CELLS, ids=_ids(k3_latent_down_fp4.CELLS))
def test_k3_latent_down_fp4_cells(cell):
    """NVFP4 of the BF16 projection. Where the kernel's BF16 value is one ulp off the reference's
    (see ``ELEMENT_ULP``), the block maximum or an element near an E2M1 midpoint can quantize the
    other way, so codes and scales are compared by value: every element within one E2M1 step of
    its block (``|a - b| <= 2 * max(scale_a, scale_b)``: each quantization is within half a step,
    at most one unit of its scale), and at most 1% of the codes / scales differing at all."""
    m = cell.spec["m"]
    n, k = 3584, 7168
    w, w_scale = _weight(n, k, 8)
    x = _rows(m, k, 9)
    amax = float(ref.lowm_gemm(x, w, w_scale).abs().amax())
    global_sf = torch.tensor([448.0 * 6.0 / amax], dtype=torch.float32, device="cuda")
    with validating(k3_latent_down_fp4):
        codes, scales = k3_latent_down_fp4(x, w, w_scale, global_sf)
    want_codes, want_scales = k3_latent_down_fp4.reference(x, w, w_scale, global_sf)
    assert codes.shape == want_codes.shape and scales.shape == want_scales.shape
    assert codes.dtype == want_codes.dtype and scales.dtype == want_scales.dtype
    got, got_s = _nvfp4_values(codes, scales)
    want, want_s = _nvfp4_values(want_codes, want_scales)
    bound = 2.0 * torch.maximum(got_s, want_s)
    excess = float(((got - want).abs() - bound).amax())
    assert excess <= 0, f"an element is more than one E2M1 step off ({excess:.3e})"
    code_diff = float((codes != want_codes).float().mean())
    scale_diff = float((scales != want_scales).float().mean())
    assert code_diff <= 0.01 and scale_diff <= 0.01, (code_diff, scale_diff)
