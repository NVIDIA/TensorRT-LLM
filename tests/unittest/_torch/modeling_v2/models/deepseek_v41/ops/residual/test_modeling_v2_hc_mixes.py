# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_mixes entry: Hyper-Connections pre / post / Sinkhorn comb mixes.

Band: the reference is the whole batch in float64. Under IEEE fp32 the op's only error is the
fp32 accumulation of a K = hc * d product (cuBLAS blocked accumulation, ~1e-6 relative at
K = 20480) carried through a sigmoid / softmax / 20 normalizations, which contract errors: the
gate is atol 2e-5 + rtol 2e-4. Under torch's TF32 default each product carries 2^-11 relative
error; the separate test bounds the mixes at atol 5e-3.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_mixes import (
    hc_mixes,  # noqa: E501
)

NORM_EPS, HC_EPS = 1e-6, 1e-6


def _inputs(n, hc, d, cell):
    gen = gen_for(cell)
    x = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    mix = (2 + hc) * hc
    fn = torch.randn(mix, hc * d, generator=gen, device=DEV) * (
        2.0 / (hc * d) ** 0.5
    )  # mixes of order 2
    scale = torch.tensor([0.5, 0.5, 0.5], device=DEV)
    base = torch.randn(mix, generator=gen, device=DEV) * 0.1
    return x, fn, scale, base


@pytest.mark.parametrize("cell", hc_mixes.CELLS, ids=[c.why[:44] for c in hc_mixes.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_mixes)
    s = cell.spec
    x, fn, scale, base = _inputs(s["n"], s["hc"], s["d"], cell)
    args = (x, fn, scale, base, s["hc"], s["iters"], NORM_EPS, HC_EPS)
    with true_fp32_matmul():
        got = hc_mixes(*args)
        ref = hc_mixes.reference(*args)
    n, hc = s["n"], s["hc"]
    assert [t.shape for t in got] == [(n, hc), (n, hc), (n, hc, hc)]
    assert all(t.dtype == torch.float32 for t in got)
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r, atol=2e-5, rtol=2e-4)


def test_ranges_and_stochasticity():
    cell = hc_mixes.CELLS[1]
    x, fn, scale, base = _inputs(5, 4, 5120, cell)
    with validating(hc_mixes):
        pre, post, comb = hc_mixes(x, fn, scale, base, 4, 20, NORM_EPS, HC_EPS)
    assert (pre > 0).all() and (pre < 1 + 1e-5).all()
    assert (post > 0).all() and (post < 2).all()
    torch.testing.assert_close(comb.sum(-2), torch.ones(5, 4, device=DEV), atol=1e-4, rtol=0)
    torch.testing.assert_close(comb.sum(-1), torch.ones(5, 4, device=DEV), atol=1e-2, rtol=0)


def test_default_matmul_policy_is_bounded():
    cell = hc_mixes.CELLS[1]
    x, fn, scale, base = _inputs(5, 4, 5120, cell)
    args = (x, fn, scale, base, 4, 20, NORM_EPS, HC_EPS)
    ref = hc_mixes.reference(*args)
    matmul = torch.backends.cuda.matmul
    prev = matmul.fp32_precision
    try:
        matmul.fp32_precision = "tf32"
        got = hc_mixes(*args)
    finally:
        matmul.fp32_precision = prev
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r, atol=5e-3, rtol=0)


def test_tiling_changes_nothing():
    import tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_mixes as mod

    x, fn, scale, base = _inputs(11, 4, 64, hc_mixes.CELLS[0])
    args = (x, fn, scale, base, 4, 20, NORM_EPS, HC_EPS)
    with true_fp32_matmul():
        whole = hc_mixes(*args)
        old = mod._TILE
        try:
            mod._TILE = 3
            tiled = hc_mixes(*args)
        finally:
            mod._TILE = old
    for a, b in zip(whole, tiled):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_rejections():
    x, fn, scale, base = _inputs(2, 4, 16, hc_mixes.CELLS[0])
    ok = (x, fn, scale, base, 4, 20, NORM_EPS, HC_EPS)
    hc_mixes.is_valid(*ok)
    with pytest.raises(AssertionError, match="hc_mult"):
        hc_mixes.is_valid(x, fn, scale, base, 2, 20, NORM_EPS, HC_EPS)
    with pytest.raises(AssertionError):
        hc_mixes.is_valid(x, fn.bfloat16(), scale, base, 4, 20, NORM_EPS, HC_EPS)
    with pytest.raises(AssertionError, match="iterations"):
        hc_mixes.is_valid(x, fn, scale, base, 4, 0, NORM_EPS, HC_EPS)
