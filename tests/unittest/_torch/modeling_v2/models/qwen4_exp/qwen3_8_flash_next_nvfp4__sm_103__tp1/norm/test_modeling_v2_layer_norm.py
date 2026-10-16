# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp layer_norm entry.

Weight and bias are drawn over this checkpoint's own 110 vision LayerNorm
tensors' ranges: the affine is wide enough (weights to 17.5, biases to -25.6)
that a wrong normalization is not masked by a near-unit scale.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.norm.layer_norm import (  # noqa: E501
    layer_norm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "layer_norm requires a CUDA device"

DEVICE = torch.device("cuda")
WIDTH = 1152
EPS = 1e-6

#: Over all of model.visual.*.norm{1,2} and model.visual.merger.norm (bf16).
WEIGHT_LO, WEIGHT_HI = -1.210938, 17.5
BIAS_LO, BIAS_HI = -25.625, 14.875

#: torch.testing's untouched defaults, restated for the discrimination test.
DEFAULT_TOL = {
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float16: (1e-3, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
}


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _affine(gen, width=WIDTH, dtype=torch.bfloat16):
    weight = torch.rand(width, generator=gen, device=DEVICE) * (WEIGHT_HI - WEIGHT_LO) + WEIGHT_LO
    bias = torch.rand(width, generator=gen, device=DEVICE) * (BIAS_HI - BIAS_LO) + BIAS_LO
    return weight.to(dtype), bias.to(dtype)


def _build(spec, seed):
    gen = _gen(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    if spec.get("layout", "contiguous") == "column_slice":
        rows, width = shape
        wide = torch.randn(rows, 2 * width, generator=gen, device=DEVICE).to(dtype)
        x = wide[:, width:]
        assert not x.is_contiguous() and x.stride(-1) == 1
    else:
        x = torch.randn(shape, generator=gen, device=DEVICE).to(dtype)
    weight, bias = _affine(gen, width=shape[-1], dtype=spec.get("param_dtype", dtype))
    return x, weight, (bias if spec.get("bias", True) else None), EPS


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x, weight, bias, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = [t.clone() for t in (x, weight, bias) if t is not None]
    out = op(x, weight, bias, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.is_contiguous() and out.data_ptr() != x.data_ptr()
    after = [t for t in (x, weight, bias) if t is not None]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), "inputs must not be mutated"
    if x.dim() == 3:
        assert torch.equal(out.reshape(-1, WIDTH), op(x.reshape(-1, WIDTH), weight, bias, eps))
    if not x.is_contiguous():
        assert torch.equal(out, op(x.contiguous(), weight, bias, eps))
    assert torch.equal(out, op(x, weight, bias, eps)), "two launches must agree bitwise"
    torch.testing.assert_close(out, op.reference(x, weight, bias, eps))


def _fp64(x, weight, bias, eps):
    xd = x.double()
    mean = xd.mean(dim=-1, keepdim=True)
    var = (xd - mean).square().mean(dim=-1, keepdim=True)
    out = (xd - mean) / torch.sqrt(var + eps) * weight.double()
    return out + bias.double() if bias is not None else out


def _worst_ratio(got, expected_fp64, dtype):
    """max |got - expected| / (atol + rtol * |expected|) at `dtype`'s default gate."""
    rtol, atol = DEFAULT_TOL[dtype]
    return ((got.double() - expected_fp64).abs() / (atol + rtol * expected_fp64.abs())).max().item()


def test_bf16_result_is_the_correctly_rounded_reference():
    """The bf16 gate is set by the output dtype, not by the kernel.

    Where the two roundings differ, the bound is against the summed terms'
    magnitude: `x_hat * weight` and `bias` cancel outright on some elements at
    this checkpoint's affine, and a bound relative to that residue bounds
    nothing. fp32 evaluation costs <= ~2^-24 of the term magnitude and each
    bf16 store <= 2^-8 of the result, so 2^-7 covers both twice over.
    """
    gen = _gen(99)
    x = torch.randn(4096, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)

    out = op(x, weight, bias, EPS)
    rounded = op.reference(x, weight, bias, EPS)
    agree = (out == rounded).double().mean().item()
    assert agree > 0.999, f"only {agree:.6%} of elements are the correctly rounded value"

    xd = x.double()
    mean = xd.mean(dim=-1, keepdim=True)
    var = (xd - mean).square().mean(dim=-1, keepdim=True)
    terms = ((xd - mean) / torch.sqrt(var + EPS) * weight.double()).abs() + bias.double().abs()
    deviation = (out.double() - rounded.double()).abs()
    assert (deviation <= 2.0**-7 * terms).all(), (deviation / terms).max().item()


def test_wrong_variants_land_outside_the_same_gate():
    """The default bf16 gate separates this computation from its neighbours."""
    gen = _gen(3)
    x = torch.randn(4096, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)

    expected = _fp64(x, weight, bias, EPS)
    xd, wd, bd = x.double(), weight.double(), bias.double()
    mean = xd.mean(dim=-1, keepdim=True)
    var = (xd - mean).square().mean(dim=-1, keepdim=True)
    wrong = {
        "rmsnorm": xd / torch.sqrt(xd.square().mean(dim=-1, keepdim=True) + EPS) * wd + bd,
        "bias_dropped": _fp64(x, weight, None, EPS),
        "weight_as_delta": (xd - mean) / torch.sqrt(var + EPS) * (wd + 1.0) + bd,
        "unbiased_variance": (xd - mean)
        / torch.sqrt((xd - mean).square().sum(dim=-1, keepdim=True) / (WIDTH - 1) + EPS)
        * wd
        + bd,
    }

    out = op(x, weight, bias, EPS)
    correct_ratio = _worst_ratio(out, expected, torch.bfloat16)
    print(f"\n  correct error/allowance {correct_ratio:.3e}")
    assert correct_ratio < 1.0, correct_ratio
    for name, value in wrong.items():
        ratio = _worst_ratio(value.to(torch.bfloat16), expected, torch.bfloat16)
        print(f"  {name:18s} error/allowance {ratio:.3e}")
        assert ratio > 1.0, f"{name} is inside the gate at error/allowance = {ratio:.6e}"
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out, value.to(torch.bfloat16))


def test_eps_is_inside_the_square_root():
    """Only observable where `var` is comparable to `eps`: a near-constant fp32
    row. The kernel must sit orders of magnitude nearer `1/sqrt(var + eps)`
    than `1/(sqrt(var) + eps)`."""
    gen = _gen(17)
    x = torch.full((4, WIDTH), 1.0, device=DEVICE, dtype=torch.float32)
    x[:, ::2] += 1e-3
    weight, bias = _affine(gen, dtype=torch.float32)

    out = op(x, weight, bias, EPS).double()
    xd, wd, bd = x.double(), weight.double(), bias.double()
    mean = xd.mean(dim=-1, keepdim=True)
    var = (xd - mean).square().mean(dim=-1, keepdim=True)
    inside = (xd - mean) / torch.sqrt(var + EPS) * wd + bd
    outside = (xd - mean) / (torch.sqrt(var) + EPS) * wd + bd
    to_inside = (out - inside).abs().max().item()
    to_outside = (out - outside).abs().max().item()
    assert to_outside > 100.0 * to_inside, (to_inside, to_outside)


def test_zero_variance_row_returns_the_bias_exactly():
    gen = _gen(31)
    x = torch.full((4, WIDTH), 2.5, device=DEVICE, dtype=torch.bfloat16)
    weight, bias = _affine(gen)
    out = op(x, weight, bias, EPS)
    assert torch.isfinite(out.float()).all()
    assert torch.equal(out, bias.expand_as(out).contiguous())


def test_runs_unchanged_under_inference_mode():
    gen = _gen(55)
    x = torch.randn(256, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    eager = op(x, weight, bias, EPS)
    with torch.inference_mode():
        inferred = op(x, weight, bias, EPS).clone()
    assert torch.equal(inferred, eager)


def test_exactly_one_kernel_is_launched():
    """The atomicity claim: one `_layer_norm_fwd_1pass_kernel`, no repacking copy."""
    from torch.profiler import ProfilerActivity, profile

    gen = _gen(66)
    x = torch.randn(1024, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    op(x, weight, bias, EPS)  # warm the Triton JIT
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        op(x, weight, bias, EPS)
        torch.cuda.synchronize()

    launched = [
        event.name
        for event in prof.events()
        if str(event.device_type) == "DeviceType.CUDA" and event.device_time_total > 0
    ]
    assert launched == ["_layer_norm_fwd_1pass_kernel"], launched


def test_capturable_in_a_cuda_graph():
    gen = _gen(77)
    x = torch.randn(512, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    static_x = x.clone()
    op(static_x, weight, bias, EPS)  # compile before capture
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = op(static_x, weight, bias, EPS)

    static_x.copy_(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, op(x, weight, bias, EPS))


def test_rejects_a_weight_whose_length_is_not_the_normalized_width():
    """Loud: `_layer_norm_fwd` asserts the shapes, with no message."""
    gen = _gen(88)
    x = torch.randn(8, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    with pytest.raises(AssertionError):
        op(x, weight[:-1].contiguous(), bias, EPS)
    with pytest.raises(AssertionError):
        op(x, weight, bias[:-1].contiguous(), EPS)


def test_rejects_a_feature_dim_that_does_not_fit_the_fused_kernel():
    width = 65536
    x = torch.randn(2, width, device=DEVICE).to(torch.bfloat16)
    weight = torch.ones(width, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.zeros(width, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="doesn't support feature dim"):
        op(x, weight, bias, EPS)


def test_accepts_an_fp8_activation_and_returns_fp8():
    """Measured, not certified: `empty_like(x)` makes the output e4m3, the
    arithmetic is still fp32, and the store lands within one e4m3 ulp (2^-3
    relative, 2^-9 smallest subnormal) of the exact value."""
    gen = _gen(111)
    x = torch.randn(64, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    x8 = x.to(torch.float8_e4m3fn)

    out = op(x8, weight, bias, EPS)

    assert out.dtype == torch.float8_e4m3fn
    assert torch.isfinite(out.float()).all()
    expected = _fp64(x8.float(), weight, bias, EPS)
    deviation = (out.float().double() - expected).abs()
    assert (deviation <= 2.0**-3 * expected.abs() + 2.0**-9).all(), deviation.max().item()
