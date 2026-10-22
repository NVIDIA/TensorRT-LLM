# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm::fused_residual_norm_quant against a torch reference with the same rounding order."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  -- triggers libth_common.so load (registers trtllm ops)

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
EPS = 1e-6

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _inputs(batch, seq, hidden, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=g) * scale

    return dict(
        x=randn(batch, seq, hidden).bfloat16(),
        y=randn(batch, seq, hidden, scale=0.5).bfloat16(),
        y_bias=randn(hidden, scale=0.1).bfloat16(),
        gate=randn(batch, hidden, scale=0.3),
        ln_weight=1.0 + randn(hidden, scale=0.1),
        ln_bias=randn(hidden, scale=0.1),
        scale=randn(batch, hidden, scale=0.1),
        shift=randn(batch, hidden, scale=0.1),
    )


def _reference(x, y, y_bias, gate, mode, weight, bias, inv_scale):
    """Residual in fp32 with the product rounded before the add, then LayerNorm and FP8 quant."""
    if y is not None:
        y = y + y_bias if y_bias is not None else y  # bf16 add, as the kernel rounds y + y_bias
        prod = y.float() * gate[:, None, :] if gate is not None else y.float()
        x = (x.float() + prod).bfloat16()
    if mode == 0:
        return x, None
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = (xf * xf).mean(-1, keepdim=True) - mean * mean
    xn = (xf - mean) * torch.rsqrt(var.clamp_min(0) + EPS)
    if mode == 1:
        out = xn * weight + bias
    else:
        out = xn * (1.0 + weight[:, None, :]) + bias[:, None, :]
    normed = (out.bfloat16().float() * inv_scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)
    return x, normed


def _run(x, y, y_bias, gate, mode, weight, bias, inv_scale, seq):
    hidden = x.shape[-1]
    x_out, normed = torch.ops.trtllm.fused_residual_norm_quant(
        x.reshape(-1, hidden),
        None if y is None else y.reshape(-1, hidden),
        y_bias,
        gate,
        mode,
        weight,
        bias,
        seq,
        EPS,
        inv_scale,
    )
    return (
        x_out.view(x.shape) if y is not None else None,
        normed.view(x.shape) if mode != 0 else None,
    )


@pytest.mark.parametrize("hidden", [3072, 5120])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize(
    "has_y,has_gate,has_bias",
    [(True, True, True), (True, False, True), (True, True, False), (False, False, False)],
)
def test_matches_reference(hidden, batch, mode, has_y, has_gate, has_bias):
    if mode == 0 and not has_y:
        pytest.skip("mode 0 needs y")
    seq = 333
    t = _inputs(batch, seq, hidden, seed=hidden + batch + mode)
    y = t["y"] if has_y else None
    y_bias = t["y_bias"] if has_bias and has_y else None
    gate = t["gate"] if has_gate and has_y else None
    weight, bias = {
        0: (None, None),
        1: (t["ln_weight"], t["ln_bias"]),
        2: (t["scale"], t["shift"]),
    }[mode]
    inv_scale = torch.tensor([E4M3_MAX / 6.0], device="cuda") if mode != 0 else None

    x_out, normed = _run(t["x"], y, y_bias, gate, mode, weight, bias, inv_scale, seq)
    ref_x, ref_normed = _reference(
        t["x"], y, y_bias, gate, mode, weight, bias, None if inv_scale is None else inv_scale.item()
    )
    if has_y:
        assert torch.equal(x_out, ref_x)
    else:
        assert x_out is None
    if mode != 0:
        # Statistics accumulate in a different order; codes agree up to one e4m3 step.
        torch.testing.assert_close(normed.float(), ref_normed.float(), rtol=0.125, atol=2**-6)
        assert (normed.view(torch.uint8) == ref_normed.view(torch.uint8)).float().mean() > 0.995


def test_rejects_unsupported_hidden():
    assert torch.ops.trtllm.fused_residual_norm_quant_supported(5120)
    assert torch.ops.trtllm.fused_residual_norm_quant_supported(3072)
    assert not torch.ops.trtllm.fused_residual_norm_quant_supported(4096)


def test_fake_under_compile():
    t = _inputs(1, 64, 3072, seed=7)
    inv_scale = torch.tensor([E4M3_MAX / 6.0], device="cuda")

    def fn(x, y):
        return _run(x, y, t["y_bias"], t["gate"], 2, t["scale"], t["shift"], inv_scale, 64)

    eager = fn(t["x"], t["y"])
    compiled = torch.compile(fn, fullgraph=True)(t["x"], t["y"])
    for a, b in zip(compiled, eager):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
