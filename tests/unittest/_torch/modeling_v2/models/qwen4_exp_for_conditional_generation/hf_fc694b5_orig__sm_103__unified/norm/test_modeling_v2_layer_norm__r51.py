# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the unified target's layer_norm entry (local variant r51).

Weight and bias are drawn over this checkpoint's own 110 vision LayerNorm
tensors' ranges: the affine is wide enough (weights to 17.5, biases to -25.6)
that a wrong normalization is not masked by a near-unit scale. The default
bf16 gate against the entry's fp64 reference is the band; the negative
controls below show it refuses the wrong normalizations a LayerNorm is
usually confused with.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.norm.layer_norm__r51 import (  # noqa: E501
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
    return x, weight, bias, EPS


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x, weight, bias, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = [t.clone() for t in (x, weight, bias)]
    out = op(x, weight, bias, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.is_contiguous() and out.data_ptr() != x.data_ptr()
    after = [x, weight, bias]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), "inputs must not be mutated"
    assert torch.equal(out, op(x, weight, bias, eps)), "two launches must agree bitwise"
    torch.testing.assert_close(out, op.reference(x, weight, bias, eps))


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

    Measured on GB300 at every cell: the op's output differs from the correctly
    rounded fp64 value on <= 3.1e-5 of the elements, against 8.6e-2 (N - 1
    variance), 0.79 (no mean subtraction) and 0.999 (no bias) for these
    mistakes, and each of them fails the default bf16 `assert_close`.
    """
    cell = next(c for c in op.CELLS if c.spec["shape"][0] == 1024)
    x, weight, bias, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    out = op(x, weight, bias, eps)
    torch.testing.assert_close(out, op.reference(x, weight, bias, eps))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out, _wrong_reference(x, weight, bias, eps, **mistake))


def test_rejects_a_weight_whose_length_is_not_the_normalized_width():
    """Loud: `_layer_norm_fwd` asserts the shapes, with no message."""
    gen = _gen(88)
    x = torch.randn(8, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    with pytest.raises(AssertionError):
        op(x, weight[:-1].contiguous(), bias, EPS)
