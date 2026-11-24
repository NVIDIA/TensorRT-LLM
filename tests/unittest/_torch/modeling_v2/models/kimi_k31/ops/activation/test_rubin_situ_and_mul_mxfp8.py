# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_situ_and_mul_mxfp8, SiTU handing the shared experts' down GEMM its MXFP8 operand.

CPU: the reference is situ_and_mul's reference quantized by the MXFP8 reference, in R128c4 order. GPU (sm_107):
every cell bit for bit against the two-kernel path it replaces (the family's situ_and_mul, then the catalog
mxfp8_quantize of its BF16 rows): the E4M3 codes and the whole scale buffer, padding rows included.
Each cell also against the entry's own reference (torch.tanh, within situ_and_mul's tolerance of
the kernel's libdevice tanh): within two E4M3 steps once both operands are decoded.
GPU, edges: the epilogue's helpers alone on rows holding every finite BF16 group maximum and the
edge groups (zeros, subnormals, the subnormal-scale band, 448 * 2^m maxima, infinities, NaNs),
bit for bit against the catalog mxfp8_quantize of those rows; and the entry on gates that put its
BF16 values in that band, in the subnormal range, at infinity and at NaN.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.activation.rubin_situ_and_mul_mxfp8 import (  # noqa: E501
    rubin_situ_and_mul_mxfp8 as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.activation.situ_and_mul import (
    situ_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (
    dequantize_mxfp8,
    mxfp8_scales_from_r128c4,
    quantize_mxfp8_reference,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)


def _inputs(spec, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, n = spec["shape"]
    x = torch.randn(rows, n, generator=gen, device=device) * 3
    if spec.get("saturated"):
        x = torch.sign(x) * 1.0e4
        x[:, :: n // 8] = 0
    x[0, : n // 2] = 0  # a zero gate row: all-zero groups
    return x.to(torch.bfloat16), spec["beta"], spec["linear_beta"]


def test_reference_is_the_two_references():
    spec = dict(shape=(7, 512), beta=4.0, linear_beta=25.0)
    x, beta, linear_beta = _inputs(spec, "cpu")
    codes, scales = op.reference(x, beta, linear_beta)
    want_codes, want_bytes = quantize_mxfp8_reference(situ_and_mul.reference(x, beta, linear_beta))
    assert torch.equal(codes.view(torch.uint8), want_codes.view(torch.uint8))
    assert torch.equal(mxfp8_scales_from_r128c4(scales, 7, 256), want_bytes)


def _assert_mxfp8_close(codes, scales, ref_codes, ref_scales):
    """Two MXFP8 operands of the same rows decode to values within two E4M3 steps of each other.

    A BF16 value one ulp off (situ_and_mul's tolerance) can move its code by one E4M3 step and its
    group's scale byte by one: two steps of the larger value, or two subnormal steps of the larger
    scale, bound the difference.
    """
    m, k = codes.shape
    got, ref = dequantize_mxfp8(codes, scales), dequantize_mxfp8(ref_codes, ref_scales)
    byte = torch.maximum(
        mxfp8_scales_from_r128c4(scales, m, k), mxfp8_scales_from_r128c4(ref_scales, m, k)
    )
    step = torch.exp2(byte.float() - (127 + 8)).repeat_interleave(32, dim=1)
    tol = 2.0**-2 * torch.maximum(got.abs(), ref.abs()) + step + 1e-5
    bad = int(((got - ref).abs() > tol).sum())
    assert bad == 0, f"{bad} values beyond two E4M3 steps of the entry's reference"


@on_sm107
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
@torch.inference_mode()
def test_cells_bitwise_against_the_two_kernel_path(cell, certifying):  # noqa: F811
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
        mxfp8_quantize,
    )

    certifying(op)
    x, beta, linear_beta = _inputs(cell.spec, "cuda", seed=cell.spec["shape"][0])
    rows = situ_and_mul(x, beta, linear_beta)
    want_codes, want_scales = mxfp8_quantize(rows, True, 32)
    codes, scales = op(x, beta, linear_beta)
    assert codes.shape == want_codes.shape and scales.shape == want_scales.shape
    assert torch.equal(codes.view(torch.uint8), want_codes.view(torch.uint8)), "codes differ"
    assert torch.equal(scales, want_scales), "scale bytes differ"
    # The certification: the entry's own reference, the one its claim is checked against.
    ref_codes, ref_scales = op.reference(x, beta, linear_beta)
    assert ref_codes.shape == codes.shape and ref_scales.numel() == scales.numel()
    _assert_mxfp8_close(codes, scales, ref_codes, ref_scales)


@on_sm107
@torch.inference_mode()
def test_epilogue_bitwise_on_every_group_max_and_edge_groups():
    from _mxfp8_epilogue_rows import mx_rows, sweep_rows

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
        mxfp8_quantize,
    )

    x = sweep_rows().cuda()
    want_codes, want_scales = mxfp8_quantize(x, True, 32)
    codes, scales = mx_rows(x)
    assert torch.equal(codes.view(torch.uint8), want_codes.view(torch.uint8)), "codes differ"
    assert torch.equal(scales, want_scales.view(torch.uint8)), "scale bytes differ"


def _edge_gates(rows: int, d: int, gen: torch.Generator) -> torch.Tensor:
    """[gate | up] BF16 rows whose SiTU values are tiny (the subnormal-scale band and subnormal
    values under small scale bytes), plus NaN gates; the rest ordinary."""
    gate = torch.randn(rows, d, generator=gen, device="cuda") * 3
    up = torch.randn(rows, d, generator=gen, device="cuda")
    sign = torch.where(torch.rand(rows, d, generator=gen, device="cuda") < 0.5, -1.0, 1.0)
    # 4 * sigmoid(g) * |u| for g in -84..-82 and |u| in [0.5, 1]: group maxima around 448 * 2^-127.
    for r in range(0, 64):
        gate[r] = -84.0 + 0.5 * (r % 5)
        up[r] = sign[r] * (0.5 + 0.5 * torch.rand(d, generator=gen, device="cuda"))
    # Group maxima near 7e-35 (scale byte ~5) with values of ~1e-39 in each group: |u| ~ 1e-5.
    for r in range(64, 96):
        gate[r] = -80.0
        up[r] = sign[r]
        up[r, 3::32] = sign[r, 3::32] * 1.0e-5
        up[r, 17::32] = sign[r, 17::32] * 3.0e-5
    gate[96, :32] = float("nan")  # a group of NaN
    gate[97, ::2] = float("nan")  # NaN in every even element: the odd maximum
    gate[98, 5] = float("nan")  # one NaN in a group
    gate[101, 1::2] = float("nan")  # NaN in every odd element: a NaN maximum, byte 255
    return torch.cat([gate, up], dim=1).to(torch.bfloat16)


def _in_band(rows_bf16: torch.Tensor) -> int:
    amax = rows_bf16.float().abs().view(rows_bf16.shape[0], -1, 32).amax(-1)
    sf = amax * (1.0 / 448.0)
    return int(((sf > 2.0**-127) & (sf < 2.0**-126)).sum())


@on_sm107
@pytest.mark.parametrize("linear_beta", [25.0, None], ids=["linear_beta", "no_linear_beta"])
@torch.inference_mode()
def test_tiny_and_special_values_bitwise_against_the_two_kernel_path(linear_beta):
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
        mxfp8_quantize,
    )

    gen = torch.Generator(device="cuda").manual_seed(1112)
    rows, d = 128, 1024
    x = _edge_gates(rows, d, gen)
    if linear_beta is None:  # unbounded up: infinite SiTU values
        x[99, d : d + 32] = float("inf")
        x[99, :32] = 10.0
        x[100, d + 7] = float("inf")
        x[100, 7] = 10.0
    want_rows = situ_and_mul(x, 4.0, linear_beta)
    assert _in_band(want_rows) > 0, "no group of the subnormal-scale band"
    want_codes, want_scales = mxfp8_quantize(want_rows, True, 32)
    codes, scales = op(x, 4.0, linear_beta)
    assert torch.equal(codes.view(torch.uint8), want_codes.view(torch.uint8)), "codes differ"
    assert torch.equal(scales, want_scales), "scale bytes differ"
