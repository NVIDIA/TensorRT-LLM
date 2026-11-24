# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_layer_norm entry (the DSA indexer's key LayerNorm, FP32 math, BF16 out).

CPU: the reference against the function body run eagerly (FP32), the guard. GPU (sm_107): every
cell against the reference within one BF16 rounding of the output (the kernel accumulates in
FP32 and rounds once, the reference in FP64), and against the eager body bit for bit or within
one output ulp (inductor may reduce in another order).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_layer_norm import (  # noqa: E501
    _layer_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_layer_norm import (  # noqa: E501
    rubin_layer_norm as op,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)


def _inputs(m, d, dtype, seed, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, d, generator=gen) * 3 + 0.5).to(dtype).to(device)
    w = (torch.rand(d, generator=gen) + 0.5).to(device)
    b = (torch.randn(d, generator=gen) * 0.1).to(device)
    return x, w, b


def test_reference_matches_the_eager_body_on_cpu():
    x, w, b = _inputs(37, 128, torch.bfloat16, 0)
    ref = op.reference(x, w, b, 1e-5)
    eager = _layer_norm(x, w, b, 1e-5)
    assert ref.dtype == eager.dtype == torch.bfloat16
    # FP64 vs FP32 statistics: at most one BF16 rounding apart.
    diff = (ref.float() - eager.float()).abs()
    assert (diff <= eager.float().abs() * 2.0**-7 + 1e-6).all()


def test_guard_refuses_a_short_affine():
    x, w, b = _inputs(4, 128, torch.bfloat16, 1)
    with pytest.raises(AssertionError, match="one value per normalized channel"):
        op.is_valid(x, w[:1], b, 1e-5)


@on_sm107
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    x, w, b = _inputs(s["m"], s["d"], s["dtype"], s["m"], "cuda")
    out = op(x, w, b, s["eps"])
    ref = op.reference(x, w, b, s["eps"])
    eager = _layer_norm(x, w, b, s["eps"])
    tol = out.float().abs() * 2.0**-7 + 1e-6
    assert ((out.float() - ref.float()).abs() <= tol).all()
    assert ((out.float() - eager.float()).abs() <= tol).all()
