# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's `engram_gate_fused` (the engine's one-launch Engram gate over the WKV output), as the dep
targets call it. Band: fp32 reductions over d against float64 with one bf16 cast: one bf16 rounding step of the row's
largest element, 0.2 ulp RMS (the band of the family's `engram_gate`, which computes the same function); and it must
agree with the plain-torch `engram_gate` on the split key / value to the same band, since it replaces it."""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_gate_fused import (  # noqa: E501
    engram_gate_fused,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.engram_gate import (
    engram_gate,  # noqa: E501
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)
EPS = 1e-6


def _gate_inputs(spec, cell):
    n, hc, d = spec["n"], spec["hc"], spec["d"]
    gen = gen_for(cell)
    h = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    kv = torch.randn(n, (hc + 1) * d, generator=gen, device=DEV).bfloat16()
    qw = (
        (torch.rand(hc, d, generator=gen, device=DEV) + 0.5).bfloat16().float()
    )  # bf16 values, as the checkpoint's
    kw = (torch.rand(hc, d, generator=gen, device=DEV) + 0.5).bfloat16().float()
    return h, kv, qw, kw, (qw * kw).contiguous()


@pytest.mark.parametrize(
    "cell", engram_gate_fused.CELLS, ids=[c.why[:48] for c in engram_gate_fused.CELLS]
)
def test_gate_fused_cells(cell, certifying):  # noqa: F811
    certifying(engram_gate_fused)
    s = cell.spec
    h, kv, qw, kw, prod = _gate_inputs(s, cell)
    got = engram_gate_fused(h, kv, qw, kw, prod, EPS)
    assert got.dtype == h.dtype and got.shape == h.shape
    assert_within_ulp(
        got, engram_gate_fused.reference(h, kv, qw, kw, prod, EPS), element_ulp=2.0, rms_ulp=0.2
    )
    hc, d = s["hc"], s["d"]
    old = engram_gate(h, kv[:, : hc * d].reshape(-1, hc, d), kv[:, hc * d :], qw, kw, EPS, None)
    assert_within_ulp(got, old, element_ulp=2.0, rms_ulp=0.2)


def test_gate_fused_rejections():
    h = torch.zeros(2, 4, 64, dtype=torch.bfloat16, device=DEV)
    w = torch.ones(4, 64, device=DEV)
    kv = torch.zeros(2, 5 * 64, dtype=torch.bfloat16, device=DEV)
    engram_gate_fused.is_valid(h, kv, w, w, w, EPS)
    with pytest.raises(AssertionError):
        engram_gate_fused.is_valid(h, kv[:, :256], w, w, w, EPS)
    with pytest.raises(AssertionError):
        engram_gate_fused.is_valid(h, kv, w, w, w.double(), EPS)
