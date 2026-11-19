# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 low_m_linear entry (decode / verify BF16 projections by a per-layout plan): a sample
of its certified cells against its reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import (
    k31_low_m_linear as op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import low_m_choice
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import K31_LOW_M_PLANS

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "low_m_linear requires a CUDA device"
if torch.cuda.get_device_capability() != (10, 3):
    pytest.skip(
        "low_m_linear is certified on SM 10.3, where the MXFP4 targets' low-M plans run",
        allow_module_level=True,
    )

CELLS = sample_cells(op.CELLS, size_key="m")[:2]


def _build(m, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    return x, w


def _kind(choice):
    return "cublas" if choice is None else choice[0]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    s = cell.spec
    plan = K31_LOW_M_PLANS[s["layout"]]
    x, w = _build(s["m"], s["n"], s["k"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    assert _kind(low_m_choice(plan, x, w)) == s["kind"], "the plan's branch for this cell moved"
    with validating(op):
        out = op(x, w, plan, True)
    ref = op.reference(x, w, plan, True)
    # One BF16 rounding of an FP32 sum taken in another order (see test_modeling_v2_fused_a_gemm).
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)
