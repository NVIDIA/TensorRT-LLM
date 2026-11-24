# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_idx_weights, the DSA indexer's logits weights ``(w * qs) * scale`` in one kernel.

CPU: the reference is the indexer's torch expression ``w * qs * scale`` bit for bit (two FP32
roundings in that order, not ``w * (qs * scale)``), on a strided ``w`` and power-of-two scales as
the query quantization writes them; the guards. GPU (sm_107): every cell bit for bit against the
reference, and against torch's two kernels on the GPU.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_weights import (  # noqa: E501
    rubin_idx_weights as op,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)

# The indexer's constant: index_head_dim^-0.5 x index_num_heads^-0.5.
_SCALE = 128**-0.5 * 32**-0.5


def _inputs(rows, heads, row_stride, seed, device="cpu", tiny=False):
    """``w``: a column slice of a ``[rows, row_stride]`` FP32 projection; ``qs``: UE8M0 scales."""
    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(rows, row_stride, generator=gen) * 0.3
    w = proj.to(device)[:, row_stride - heads :]
    exp = torch.randint(-12, -2, (rows, heads), generator=gen)
    if tiny:
        exp = exp - 120  # products near and below FP32's normal range
    qs = torch.ldexp(torch.ones(rows, heads), exp).to(device)
    return w, qs


def test_reference_is_the_indexers_expression_bit_for_bit():
    for rows, row_stride, seed in ((96, 160, 0), (37, 32, 1), (1, 160, 2)):
        w, qs = _inputs(rows, 32, row_stride, seed)
        ref = op.reference(w, qs, _SCALE)
        # The indexer's own expression: qs as [rows * heads, 1] viewed back to [rows, heads, 1].
        flat = qs.reshape(-1, 1).view(-1, 32, 1)
        today = w * flat.squeeze(-1) * _SCALE
        assert ref.dtype == torch.float32 and ref.is_contiguous()
        assert torch.equal(ref.view(torch.int32), today.view(torch.int32))


def test_the_rounding_order_is_pinned():
    """``(w * qs) * scale`` differs from ``w * (qs * scale)`` once a product leaves the normal
    range: the reference must be the former, the indexer's order."""
    w, qs = _inputs(96, 32, 160, 3, tiny=True)
    ref = op.reference(w, qs, _SCALE)
    assert torch.equal(ref.view(torch.int32), ((w * qs) * _SCALE).view(torch.int32))
    assert not torch.equal(ref.view(torch.int32), (w * (qs * _SCALE)).view(torch.int32))


def test_guards():
    w, qs = _inputs(8, 32, 160, 4)
    op.is_valid(w, qs, _SCALE)
    with pytest.raises(AssertionError, match="FP32"):
        op.is_valid(w.half(), qs, _SCALE)
    with pytest.raises(AssertionError, match=r"\[rows, heads\]"):
        op.is_valid(w, qs[:, :16], _SCALE)
    with pytest.raises(AssertionError, match="unit inner stride"):
        op.is_valid(torch.zeros(32, 8).t(), torch.zeros(8, 32), _SCALE)
    with pytest.raises(AssertionError, match="contiguous"):
        op.is_valid(w, torch.zeros(32, 8).t(), _SCALE)
    with pytest.raises(AssertionError, match="power of two"):
        op.is_valid(torch.zeros(8, 24), torch.zeros(8, 24), _SCALE)


# --- GPU -------------------------------------------------------------------------------------


@on_sm107
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    w, qs = _inputs(s["rows"], s["heads"], s["row_stride"], s["rows"], "cuda")
    out = op(w, qs, _SCALE)
    assert out.dtype == torch.float32 and out.shape == w.shape and out.is_contiguous()
    assert torch.equal(out.view(torch.int32), op.reference(w, qs, _SCALE).view(torch.int32))


@on_sm107
@pytest.mark.parametrize("tiny", [False, True], ids=["normal", "subnormal products"])
def test_equals_torchs_two_kernels_on_the_gpu(tiny):
    w, qs = _inputs(96, 32, 160, 5, "cuda", tiny=tiny)
    with validating(op):
        out = op(w, qs, _SCALE)
    today = w * qs.reshape(-1, 1).view(-1, 32, 1).squeeze(-1) * _SCALE
    assert torch.equal(out.view(torch.int32), today.view(torch.int32))
