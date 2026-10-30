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


def test_rejects_a_weight_whose_length_is_not_the_normalized_width():
    """Loud: `_layer_norm_fwd` asserts the shapes, with no message."""
    gen = _gen(88)
    x = torch.randn(8, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    with pytest.raises(AssertionError):
        op(x, weight[:-1].contiguous(), bias, EPS)
