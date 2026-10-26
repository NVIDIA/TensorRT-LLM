# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the layer_norm catalog entry.

Weight and bias are drawn over the ranges of the checkpoint tensors each cell's
call site normalizes with, so a wrong normalization is not masked by a
near-unit affine:

- `vision_1152`: a 1152-wide vision tower's 110 LayerNorm tensors (weights to
  17.5, biases to -25.6), qwen4_exp's vision tower;
- `indexer_k_norm`: Step-5-Preview's 23 `sparse_indexer_k_norm` weight and
  bias tensors (fp32 on disk, rounded to bf16 here as the caller does);
- `vision_ln_1536`: Step-5-Preview's 95 `vision_model` ln_pre / ln_1 / ln_2
  weight and bias tensors (bf16 on disk).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import (  # noqa: E501
    layer_norm as op,
)

__extra_import_path__ = [".."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "layer_norm requires a CUDA device"

DEVICE = torch.device("cuda")
WIDTH = 1152
EPS = 1e-6

#: (weight lo, weight hi, bias lo, bias hi) over each checkpoint tensor family.
AFFINE_RANGES = {
    # model.visual.*.norm{1,2} and model.visual.merger.norm (bf16).
    "vision_1152": (-1.210938, 17.5, -25.625, 14.875),
    # model.layers.{3,7,...,91}.self_attn.sparse_indexer_k_norm.{weight,bias} (fp32).
    "indexer_k_norm": (0.00147371, 1.20922, -0.922049, 0.883452),
    # vision_model.ln_pre and vision_model.*.ln_{1,2} (bf16).
    "vision_ln_1536": (-0.12793, 14.375, -11.125, 11.375),
}


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _affine(gen, width=WIDTH, dtype=torch.bfloat16, affine="vision_1152"):
    w_lo, w_hi, b_lo, b_hi = AFFINE_RANGES[affine]
    weight = torch.rand(width, generator=gen, device=DEVICE) * (w_hi - w_lo) + w_lo
    bias = torch.rand(width, generator=gen, device=DEVICE) * (b_hi - b_lo) + b_lo
    return weight.to(dtype), bias.to(dtype)


def _build(spec, seed):
    gen = _gen(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    x = torch.randn(shape, generator=gen, device=DEVICE)
    row_offset = spec.get("row_offset", 0.0)
    if row_offset:
        offsets = (torch.rand(shape[0], 1, generator=gen, device=DEVICE) * 2 - 1) * row_offset
        x = x + offsets
    x = x.to(dtype)
    weight, bias = _affine(
        gen, width=shape[-1], dtype=dtype, affine=spec.get("affine", "vision_1152")
    )
    return x, weight, bias, spec.get("eps", EPS)


def _one_pass(x, weight, bias, eps):
    """fp32 var = E[x^2] - mean^2, rsqrt, affine, one rounding: the indexer-kernel form."""
    xf = x.float()
    n = x.shape[-1]
    mean = xf.sum(dim=-1, keepdim=True) / n
    var = xf.square().sum(dim=-1, keepdim=True) / n - mean * mean
    out = (xf - mean) * torch.rsqrt(var + eps) * weight.float() + bias.float()
    return out.to(x.dtype)


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


INDEXER_CELLS = [c for c in op.CELLS if c.spec.get("affine") == "indexer_k_norm"]


@pytest.mark.parametrize("cell", INDEXER_CELLS, ids=[c.why[:44] for c in INDEXER_CELLS])
def test_one_pass_variance_formula_agrees_within_one_ulp(cell):
    """The two-pass kernel against the one-pass variance form, at the key norm's cells.

    Both round an fp32 value once to bf16, and the two fp32 values differ by a
    few fp32 ulps of the variance (|mean| <= 4 + a few sigma here, so the
    cancellation in E[x^2] - mean^2 costs at most ~2^-19 relative). A bf16
    result can therefore move by at most its own spacing, which is <= 2 ULP
    (ULP = 2^-8, half the spacing of [1, 2)) of the row's largest magnitude;
    the relative RMS of such rare flips stays well under one ULP.
    """
    x, weight, bias, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    assert_within_ulp(
        op(x, weight, bias, eps), _one_pass(x, weight, bias, eps), element_ulp=2, rms_ulp=1
    )


def test_cuda_graph_capture_and_replay():
    """The decode graph captures the key norm: replay on refilled inputs equals eager."""
    spec = next(c.spec for c in INDEXER_CELLS if c.spec["shape"][0] == 128)
    x, weight, bias, eps = _build(spec, seed=7)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x, weight, bias, eps)  # warm-up compiles the Triton kernel outside capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, weight, bias, eps)
    x2, weight2, bias2, _ = _build(spec, seed=8)
    x.copy_(x2)
    weight.copy_(weight2)
    bias.copy_(bias2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, op(x, weight, bias, eps))
    torch.testing.assert_close(out, op.reference(x, weight, bias, eps))


def test_rejects_a_weight_whose_length_is_not_the_normalized_width():
    """Loud: `_layer_norm_fwd` asserts the shapes, with no message."""
    gen = _gen(88)
    x = torch.randn(8, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    with pytest.raises(AssertionError):
        op(x, weight[:-1].contiguous(), bias, EPS)
