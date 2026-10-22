# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wan FP8 kernels: trtllm::fused_dit_qk_norm_rope_fp8 and trtllm::dit_v_amax against the bf16
fused op, and trtllm::fused_residual_norm_quant against a torch reference with the same rounding
order."""

import math

import pytest
import torch
from utils.util import skip_pre_hopper

import tensorrt_llm  # noqa: F401  -- triggers libth_common.so load (registers trtllm ops)

pytestmark = skip_pre_hopper

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
HEAD_DIM = 128
EPS = 1e-6
QK_MUL = math.sqrt(HEAD_DIM**-0.5 / math.log(2.0))


def _inputs(tokens, num_heads, cos_rows, per_head_cos=False, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = num_heads * HEAD_DIM
    qkv = torch.randn(tokens, 3 * n, device="cuda", generator=g).bfloat16()
    q_w, k_w = (
        (1.0 + 0.1 * torch.randn(n, device="cuda", generator=g)).bfloat16() for _ in range(2)
    )
    cos_cols = n if per_head_cos else HEAD_DIM
    angle = torch.rand(cos_rows, cos_cols, device="cuda", generator=g) * 2 * math.pi
    cos, sin = angle.cos(), angle.sin()
    if per_head_cos:
        cos, sin = (t.view(cos_rows, num_heads, HEAD_DIM) for t in (cos, sin))
    return qkv, q_w, k_w, cos, sin


def _bf16_reference(qkv, num_heads, q_w, k_w, cos, sin, interleave):
    """Normalized and rotated bf16 Q, K and untouched V from the in-place bf16 op."""
    out = qkv.clone()
    torch.ops.trtllm.fused_dit_qk_norm_rope(
        out, num_heads, num_heads, num_heads, HEAD_DIM, EPS, q_w, k_w, None, None,
        cos, sin, -1, interleave, 0,
    )  # fmt: skip
    return out.view(qkv.shape[0], 3, num_heads, HEAD_DIM).unbind(1)


def _quant(x, mul):
    return (x.float() * mul).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)


def _assert_fp8_close(actual, expected):
    """Equal up to one e4m3 rounding step.

    The fp32 RMS sum can be accumulated in a different order than in the bf16 op (it depends on
    the compiled instruction sequence per architecture), which moves the bf16 rounding of some
    elements and so their e4m3 code by one step. A systematic error, such as a wrong rounding
    mode or scale, changes far more than 2% of the codes.
    """
    assert actual.dtype == E4M3 and actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.125, atol=2**-6)
    assert (actual.view(torch.uint8) == expected.view(torch.uint8)).float().mean() > 0.98


def _run(qkv, num_heads, q_w, k_w, cos, sin, interleave, v_amax=None):
    return torch.ops.trtllm.fused_dit_qk_norm_rope_fp8(
        qkv, num_heads, HEAD_DIM, EPS, q_w, k_w, cos, sin, interleave, QK_MUL, v_amax
    )


def _v_amax(qkv, num_heads):
    return qkv[:, 2 * num_heads * HEAD_DIM :].float().abs().amax()


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("num_heads", [4, 10, 24, 40, 64])
@pytest.mark.parametrize("batch", [1, 2])
def test_matches_bf16_op_shared_cos(interleave, num_heads, batch):
    """Shared [S, D] cos table, broadcast over the batch (pipelined kernel)."""
    seq = 1560
    qkv, q_w, k_w, cos, sin = _inputs(batch * seq, num_heads, seq)
    q8, k8, v8, v_descale = _run(qkv, num_heads, q_w, k_w, cos, sin, interleave)

    q_ref, k_ref, v_ref = _bf16_reference(qkv, num_heads, q_w, k_w, cos, sin, interleave)
    scale_v = _v_amax(qkv, num_heads) / E4M3_MAX
    torch.testing.assert_close(v_descale, scale_v.reshape(1), rtol=1e-6, atol=0)
    _assert_fp8_close(q8, _quant(q_ref, QK_MUL))
    _assert_fp8_close(k8, _quant(k_ref, QK_MUL))
    _assert_fp8_close(v8, _quant(v_ref, 1.0 / scale_v))


@pytest.mark.parametrize("interleave", [False, True])
def test_matches_bf16_op_per_head_cos(interleave):
    """Per-head [T, H, D] cos table (general kernel)."""
    tokens, num_heads = 777, 40
    qkv, q_w, k_w, cos, sin = _inputs(tokens, num_heads, tokens, per_head_cos=True, seed=1)
    q8, k8, v8, v_descale = _run(qkv, num_heads, q_w, k_w, cos, sin, interleave)

    q_ref, k_ref, v_ref = _bf16_reference(qkv, num_heads, q_w, k_w, cos, sin, interleave)
    _assert_fp8_close(q8, _quant(q_ref, QK_MUL))
    _assert_fp8_close(k8, _quant(k_ref, QK_MUL))
    _assert_fp8_close(v8, _quant(v_ref, 1.0 / v_descale))


def test_given_v_amax_sets_v_scale():
    """A V amax reduced elsewhere (e.g. across sequence-parallel ranks) sets the V scale."""
    tokens, num_heads = 1560, 40
    qkv, q_w, k_w, cos, sin = _inputs(tokens, num_heads, tokens, seed=2)
    group_amax = (_v_amax(qkv, num_heads) * 3).reshape(1)
    _, _, v8, v_descale = _run(qkv, num_heads, q_w, k_w, cos, sin, False, group_amax)

    scale_v = group_amax / E4M3_MAX
    torch.testing.assert_close(v_descale, scale_v, rtol=1e-6, atol=0)
    v_ref = qkv.view(tokens, 3, num_heads, HEAD_DIM)[:, 2]
    _assert_fp8_close(v8, _quant(v_ref, 1.0 / scale_v))


def test_small_v_amax_saturates():
    """A V amax below the local |V| maximum clamps V to the finite e4m3 range."""
    tokens, num_heads = 1560, 40
    qkv, q_w, k_w, cos, sin = _inputs(tokens, num_heads, tokens, seed=5)
    small_amax = (_v_amax(qkv, num_heads) / 4).reshape(1)
    _, _, v8, v_descale = _run(qkv, num_heads, q_w, k_w, cos, sin, False, small_amax)

    v_ref = qkv.view(tokens, 3, num_heads, HEAD_DIM)[:, 2]
    expected = _quant(v_ref, 1.0 / v_descale)
    assert (expected.float().abs() == E4M3_MAX).any()
    assert v8.float().abs().max().item() == E4M3_MAX
    _assert_fp8_close(v8, expected)


@pytest.mark.parametrize("tokens", [1, 33, 1560 * 2])
def test_dit_v_amax(tokens):
    num_heads = 40
    qkv, *_ = _inputs(tokens, num_heads, tokens, seed=3)
    amax = torch.ops.trtllm.dit_v_amax(qkv, num_heads, HEAD_DIM)
    assert amax.dtype == torch.float32 and amax.shape == (1,)
    assert amax.item() == _v_amax(qkv, num_heads).item()


def test_rejects_unsupported_layouts():
    tokens, num_heads = 64, 40
    qkv, q_w, k_w, cos, sin = _inputs(tokens, num_heads, tokens)
    with pytest.raises(RuntimeError, match="full-dim"):
        _run(qkv, num_heads, q_w[:HEAD_DIM], k_w[:HEAD_DIM], cos, sin, False)
    misaligned = torch.empty(tokens * qkv.shape[1] + 1, device="cuda", dtype=torch.bfloat16)
    misaligned = misaligned[1:].view_as(qkv).copy_(qkv)
    with pytest.raises(RuntimeError, match="aligned"):
        _run(misaligned, num_heads, q_w, k_w, cos, sin, False)
    with pytest.raises(RuntimeError, match="device of qkv"):
        _run(qkv, num_heads, q_w, k_w, cos.cpu(), sin.cpu(), False)


def test_fake_under_compile():
    tokens, num_heads = 256, 10
    qkv, q_w, k_w, cos, sin = _inputs(tokens, num_heads, tokens, seed=4)

    def fn(x):
        amax = torch.ops.trtllm.dit_v_amax(x, num_heads, HEAD_DIM)
        return _run(x, num_heads, q_w, k_w, cos, sin, False, amax)

    eager = fn(qkv)
    compiled = torch.compile(fn, fullgraph=True)(qkv)
    for a, b in zip(compiled, eager):
        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# trtllm::fused_residual_norm_quant: gated residual + LayerNorm / AdaLN + static FP8 quantization.
# ---------------------------------------------------------------------------


def _rnq_inputs(batch, seq, hidden, seed=0):
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


def _rnq_reference(x, y, y_bias, gate, mode, weight, bias, inv_scale):
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


def _rnq_run(x, y, y_bias, gate, mode, weight, bias, inv_scale, seq):
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


@pytest.mark.parametrize(
    "hidden,batch,mode,has_y,has_gate,has_bias",
    [
        # Every mode with all inputs, across widths.
        (3072, 1, 0, True, True, True),
        (5120, 2, 1, True, True, True),
        (4096, 2, 2, True, True, True),
        # Optional inputs left out.
        (3072, 1, 1, True, False, True),
        (5120, 2, 2, True, True, False),
        (5120, 2, 2, False, False, False),
    ],
)
def test_residual_norm_quant_matches_reference(hidden, batch, mode, has_y, has_gate, has_bias):
    seq = 333
    t = _rnq_inputs(batch, seq, hidden, seed=hidden + batch + mode)
    y = t["y"] if has_y else None
    y_bias = t["y_bias"] if has_bias and has_y else None
    gate = t["gate"] if has_gate and has_y else None
    weight, bias = {
        0: (None, None),
        1: (t["ln_weight"], t["ln_bias"]),
        2: (t["scale"], t["shift"]),
    }[mode]
    inv_scale = torch.tensor([E4M3_MAX / 6.0], device="cuda") if mode != 0 else None

    x_out, normed = _rnq_run(t["x"], y, y_bias, gate, mode, weight, bias, inv_scale, seq)
    ref_x, ref_normed = _rnq_reference(
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


def test_residual_norm_quant_supported_hidden_sizes():
    supported = torch.ops.trtllm.fused_residual_norm_quant_supported
    assert all(supported(h) for h in (1024, 3072, 5120, 8192))
    assert not any(supported(h) for h in (0, 1000, 3000, 9216))


def test_residual_norm_quant_rejects_cpu_inv_out_scale():
    t = _rnq_inputs(1, 64, 3072)
    inv_scale = torch.tensor([E4M3_MAX / 6.0])
    with pytest.raises(RuntimeError, match="inv_out_scale must be a CUDA tensor"):
        _rnq_run(t["x"], None, None, None, 1, t["ln_weight"], t["ln_bias"], inv_scale, 64)


def test_residual_norm_quant_fake_under_compile():
    t = _rnq_inputs(1, 64, 3072, seed=7)
    inv_scale = torch.tensor([E4M3_MAX / 6.0], device="cuda")

    def fn(x, y):
        return _rnq_run(x, y, t["y_bias"], t["gate"], 2, t["scale"], t["shift"], inv_scale, 64)

    eager = fn(t["x"], t["y"])
    compiled = torch.compile(fn, fullgraph=True)(t["x"], t["y"])
    for a, b in zip(compiled, eager):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
