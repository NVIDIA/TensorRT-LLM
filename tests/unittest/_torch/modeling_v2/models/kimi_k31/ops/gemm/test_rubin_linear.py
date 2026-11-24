# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_linear entry (``F.linear``, optionally with TF32).

CPU: the reference on small shapes, the TF32 switch is restored, the guard. GPU (sm_107): every
cell against ``F.linear`` itself bit for bit (the entry is that call) and against the reference
within the band of the accumulation: one output ulp (BF16), or TF32's 2^-10 relative input
rounding for the FP32 cells.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_linear import (
    rubin_linear as op,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)


def test_reference_is_the_product_rounded_once():
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(-4, 5, (5, 33), generator=gen).to(torch.bfloat16)
    w = torch.randint(-4, 5, (7, 33), generator=gen).to(torch.bfloat16)
    ref = op.reference(x, w, False)
    assert ref.dtype == torch.bfloat16 and ref.shape == (5, 7)
    # Small integers: every product and sum is exact in BF16's range here.
    assert torch.equal(ref, (x.float() @ w.float().t()).to(torch.bfloat16))


def test_cpu_call_matches_f_linear_and_restores_tf32():
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(9, 64, generator=gen)
    w = torch.randn(16, 64, generator=gen)
    before = torch.backends.cuda.matmul.allow_tf32
    for tf32 in (False, True):
        assert torch.equal(op(x, w, tf32), F.linear(x, w))
        assert torch.backends.cuda.matmul.allow_tf32 == before


def test_guard_refuses_mixed_dtypes():
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    w = torch.zeros(4, 8, dtype=torch.float32)
    with pytest.raises(AssertionError, match="must match"):
        op.is_valid(x, w, False)


@on_sm107
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    gen = torch.Generator().manual_seed(spec["m"] * 7 + spec["n"])
    x = torch.randn(spec["m"], spec["k"], generator=gen).to(spec["dtype"]).cuda()
    w = (
        (torch.randn(spec["n"], spec["k"], generator=gen) / spec["k"] ** 0.5)
        .to(spec["dtype"])
        .cuda()
    )
    out = op(x, w, spec["tf32"])
    if spec["tf32"]:
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        same = F.linear(x, w)
        torch.backends.cuda.matmul.allow_tf32 = prev
    else:
        same = F.linear(x, w)
    assert torch.equal(out, same), "the entry is F.linear"
    ref = op.reference(x, w, spec["tf32"])
    scale = (x.float().abs() @ w.float().abs().t()).clamp_min(1e-6)
    err = ((out.float() - ref.float()).abs() / scale).max().item()
    # BF16: one output rounding (2^-8) plus the reference's. FP32: TF32's 2^-10 per input, both
    # inputs (torch's default FP32 matmul may run TF32 on this GPU without the switch too).
    bound = 2.0**-7 if spec["dtype"] == torch.bfloat16 else 2.0**-9
    assert err <= bound, f"error {err:.3e} past {bound:.3e}"
