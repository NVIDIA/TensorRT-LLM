# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the layer_norm entry this target owns.

The entry owns the contract: `CELLS` is the certified range, `reference` is what
the kernel is compared against, and `is_valid` is the guard (this op has none:
every input it takes is either computed correctly or refused loudly). This file
owns what a cell cannot carry -- turning a spec into real tensors, the
comparison band and its derivation, and the wrong normalizations the band must
refuse.

Weight and bias are drawn over this checkpoint's own 54 vision LayerNorm
tensor pairs' ranges (`vit.layers.*.{input,post_attention}_layernorm`, bf16):
weights [-0.78125, 14.3125], biases [-19.5, 10.9375]. The affine is wide enough
that a wrong normalization is not masked by a near-unit scale.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.norm.layer_norm import (  # noqa: E501
    layer_norm as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "layer_norm requires a CUDA device"

DEVICE = torch.device("cuda")
WIDTH = 1152
EPS = 1e-5
#: One unit in the last place of the bf16 mantissa.
ULP = 2.0**-8

#: Over all of vit.layers.*.input_layernorm and post_attention_layernorm (bf16).
WEIGHT_LO, WEIGHT_HI = -0.78125, 14.3125
BIAS_LO, BIAS_HI = -19.5, 10.9375


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _affine(gen, width=WIDTH, dtype=torch.bfloat16):
    weight = torch.rand(width, generator=gen, device=DEVICE) * (WEIGHT_HI - WEIGHT_LO) + WEIGHT_LO
    bias = torch.rand(width, generator=gen, device=DEVICE) * (BIAS_HI - BIAS_LO) + BIAS_LO
    return weight.to(dtype), bias.to(dtype)


def _build(spec, seed):
    gen = _gen(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    x = torch.randn(shape, generator=gen, device=DEVICE).to(dtype)
    weight, bias = _affine(gen, width=shape[-1], dtype=dtype)
    return x, weight, bias, spec["eps"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, weight, bias, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = [t.clone() for t in (x, weight, bias)]
    out = op(x, weight, bias, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.is_contiguous() and out.data_ptr() != x.data_ptr()
    after = [x, weight, bias]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), "inputs must not be mutated"
    assert torch.equal(out, op(x, weight, bias, eps)), "two launches must agree bitwise"
    ref = op.reference(x, weight, bias, eps)
    torch.testing.assert_close(out, ref)
    # Distribution, as well as the element band. Measured against this
    # reference: relative RMS 0 (1 row), 0.0022 (1024), 0.0032 (16384) ulp of
    # the reference's RMS; the band is 16x above the worst, and the nearest
    # wrong normalization below (unbiased variance) reaches 0.28 ulp.
    of, rf = out.float(), ref.float()
    rel_rms = (of - rf).pow(2).mean().sqrt() / rf.pow(2).mean().sqrt().clamp_min(1e-9)
    assert rel_rms <= 0.05 * ULP, f"relative RMS {rel_rms / ULP:.4f} ulp"
    cosine = torch.nn.functional.cosine_similarity(of.flatten(), rf.flatten(), dim=0)
    assert cosine >= 1 - 1e-5, f"cosine {cosine.item():.9f}"


def _wrong_reference(x, weight, bias, eps, *, ddof=0, center=True, with_bias=True):
    """The entry's fp64 formula with one deliberate mistake, rounded to `x.dtype`."""
    xd = x.double()
    mean = xd.mean(dim=-1, keepdim=True) if center else torch.zeros_like(xd[..., :1])
    var = (xd - mean).square().sum(dim=-1, keepdim=True) / (xd.shape[-1] - ddof)
    out = (xd - mean) / torch.sqrt(var + eps) * weight.double()
    if with_bias:
        out = out + bias.double()
    return out.to(x.dtype)


@pytest.mark.parametrize(
    "mistake",
    [dict(ddof=1), dict(center=False), dict(with_bias=False)],
    ids=["unbiased_variance", "no_mean_subtraction", "no_bias"],
)
def test_the_gate_discriminates_a_wrong_normalization(mistake):
    """The cells' gate must refuse a reference that normalizes wrongly.

    Measured at the 1024-row cell: the op itself differs from the correctly
    rounded fp64 value on 32 of 1179648 elements and fails none of the band;
    these mistakes fail 7703 (unbiased variance), 508538 (no mean) and
    1173164 (no bias) elements of the same output.
    """
    cell = next(c for c in op.CELLS if c.spec["shape"][0] == 1024)
    x, weight, bias, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x, weight, bias, eps)
    torch.testing.assert_close(out, op.reference(x, weight, bias, eps))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out, _wrong_reference(x, weight, bias, eps, **mistake))


def test_capturable_in_a_cuda_graph():
    """A 1024-row capture, replayed on refilled input, is bitwise the eager call.

    Not a call-site requirement -- the vision tower runs this eagerly in
    prefill -- but the property a future capture would rely on.
    """
    gen = _gen(77)
    static_x = torch.randn(1024, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    op(static_x, weight, bias, EPS)  # compile before capture
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = op(static_x, weight, bias, EPS)

    static_x.copy_(torch.randn(1024, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, op(static_x, weight, bias, EPS))


def test_rejects_a_weight_whose_length_is_not_the_normalized_width():
    """Loud: `_layer_norm_fwd` asserts the shapes, with no message."""
    gen = _gen(88)
    x = torch.randn(8, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    with pytest.raises(AssertionError):
        op(x, weight[:-1].contiguous(), bias, EPS)
