# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the engram_gate entry: gated n-gram value added into every residual copy.

Band: fp32 reductions over d vs float64, one cast. The gate is a sigmoid of a sqrt of a dot product
that is itself O(1), so errors stay at fp32 level; 1 bf16 ulp of the row's largest value
elementwise, 0.2 ulp RMS. Masked rows pass through bit for bit.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.engram_gate import (  # noqa: E501
    engram_gate,
    engram_gate_kv,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the gate's kernels (Triton, stock's) need CUDA"
)

EPS = 1e-6


def _inputs(spec, cell):
    n, hc, d = spec["n"], spec["hc"], spec["d"]
    gen = gen_for(cell)
    h = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    key = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    value = torch.randn(n, d, generator=gen, device=DEV).bfloat16()
    qw = torch.rand(hc, d, generator=gen, device=DEV) + 0.5
    kw = torch.rand(hc, d, generator=gen, device=DEV) + 0.5
    return h, key, value, qw, kw


@pytest.mark.parametrize("cell", engram_gate.CELLS, ids=[c.why[:44] for c in engram_gate.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(engram_gate)
    s = cell.spec
    h, key, value, qw, kw = _inputs(s, cell)
    mask = None
    if s["masked"]:
        mask = torch.ones(s["n"], dtype=torch.bool, device=DEV)
        mask[[0, s["n"] - 1]] = False
    got = engram_gate(h, key, value, qw, kw, EPS, mask)
    ref = engram_gate.reference(h, key, value, qw, kw, EPS, mask)
    assert got.dtype == h.dtype and got.shape == h.shape
    # element_ulp is in units of ULP[bf16] = 2**-8, the spacing of values at the top of a binade. A
    # value at the bottom of a binade has spacing 2**-7, and the row maximum that scales the band can
    # sit there, so one bf16 rounding step of that element is up to 2 units. The 4097-row cell (~16M
    # outputs) had 1-2 elements land one step apart (0.0043-0.005 of the row max against 1 unit =
    # 0.0039); 2 units = 2**-7 is the band for exactly one bf16 step, no more. rms_ulp is unchanged.
    assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.2)
    if mask is not None:
        assert torch.equal(got[~mask], h[~mask])
        assert not torch.equal(got[mask], h[mask])


def test_rejections():
    h = torch.zeros(3, 4, 8, device=DEV)
    ok = (
        h,
        h,
        torch.zeros(3, 8, device=DEV),
        torch.ones(4, 8, device=DEV),
        torch.ones(4, 8, device=DEV),
        EPS,
    )
    engram_gate.is_valid(*ok)
    with pytest.raises(AssertionError, match="bool"):
        engram_gate.is_valid(*ok, torch.ones(3, dtype=torch.int32, device=DEV))
    with pytest.raises(AssertionError):
        engram_gate.is_valid(h, h, torch.zeros(3, 7, device=DEV), *ok[3:])


def test_strided_key_value():
    """The model passes key and value as column slices of one `[n, hc*d + d]` projection (no copies): the kernel reads
    them through their row strides."""
    gen = torch.Generator(device=DEV).manual_seed(11)
    n, hc, d = 300, 4, 5120
    h = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    kv = torch.randn(n, hc * d + d, generator=gen, device=DEV).bfloat16()
    key, value = kv[:, : hc * d].unflatten(-1, (hc, d)), kv[:, hc * d :]
    qw = torch.randn(hc, d, generator=gen, device=DEV).bfloat16()
    kw = torch.randn(hc, d, generator=gen, device=DEV).bfloat16()
    assert not key.is_contiguous() and not value.is_contiguous()
    got = engram_gate(h, key, value, qw, kw, EPS, None)
    ref = engram_gate.reference(h, key, value, qw, kw, EPS, None)
    assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.2)


@pytest.mark.parametrize(
    "cell", engram_gate_kv.CELLS, ids=[c.why[:44] for c in engram_gate_kv.CELLS]
)
def test_kv_cells(cell, certifying):  # noqa: F811
    """Stock's gate kernel over the fused k | v projection, with the cached norm-weight product."""
    certifying(engram_gate_kv)
    s = cell.spec
    gen = gen_for(cell)
    n, hc, d = s["n"], s["hc"], s["d"]
    h = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    kv = torch.randn(n, hc * d + d, generator=gen, device=DEV).bfloat16()
    qw = (torch.rand(hc, d, generator=gen, device=DEV) + 0.5).bfloat16()
    kw = (torch.rand(hc, d, generator=gen, device=DEV) + 0.5).bfloat16()
    got = engram_gate_kv(h, kv, qw, kw, (qw.float() * kw.float()).contiguous(), EPS)
    ref = engram_gate_kv.reference(h, kv, qw, kw, None, EPS)
    assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.2)


def test_zero_dot_keeps_the_clamp_floor():
    """A zero key makes the dot exactly +0: stock's gate (`modules.engram.functional`, copysign) takes the positive
    floor, sigmoid(sqrt(1e-6)), in the kernel and in the reference alike (sign(0) would drop the floor)."""
    n, hc, d = 2, 4, 64
    h = torch.ones(n, hc, d, device=DEV)  # fp32: the gate's 5e-4 offset is visible
    key = torch.zeros(n, hc, d, device=DEV)
    value = torch.full((n, d), -2.0, device=DEV)
    w = torch.ones(hc, d, device=DEV)
    expect = 1.0 - 2.0 * torch.sigmoid(torch.tensor(1e-3, dtype=torch.float64)).item()
    got = engram_gate(h, key, value, w, w, EPS, None)
    ref = engram_gate.reference(h, key, value, w, w, EPS, None)
    torch.testing.assert_close(ref, torch.full_like(ref, expect), rtol=0, atol=1e-6)
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-6)


def test_masked_rows_pass_a_non_finite_value_through():
    """A masked-out token keeps its residual even when its value is not finite (no 0 * inf)."""
    gen = torch.Generator(device=DEV).manual_seed(5)
    n, hc, d = 3, 4, 64
    h = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    key = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    value = torch.randn(n, d, generator=gen, device=DEV).bfloat16()
    value[0, :8] = float("inf")
    value[0, 8:16] = float("nan")
    w = torch.ones(hc, d, device=DEV)
    mask = torch.tensor([False, True, True], device=DEV)
    got = engram_gate(h, key, value, w, w, EPS, mask)
    ref = engram_gate.reference(h, key, value, w, w, EPS, mask)
    assert torch.equal(got[0], h[0]) and torch.equal(ref[0], h[0])
    assert_within_ulp(got[1:], ref[1:], element_ulp=2.0, rms_ulp=0.2)
