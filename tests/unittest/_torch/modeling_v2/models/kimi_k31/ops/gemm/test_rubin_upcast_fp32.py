# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_upcast_fp32, BF16 rows widened to FP32 bit for bit.

CPU: the reference is ``x.float()``, which places the BF16 bits above 16 zero bits for every one of the 65,536
patterns (NaN payloads included); the guards. GPU (sm_107): every cell bit for bit against the reference, and all
65,536 patterns: the BF16 bits above 16 zero bits, which is ``x.float()`` on the GPU for every pattern but a NaN
(there, a NaN either way).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_upcast_fp32 import (  # noqa: E501
    rubin_upcast_fp32 as op,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)


def _all_patterns(device="cpu"):
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32, device=device).to(torch.int16)
    return bits, bits.view(torch.bfloat16).view(256, 256)


def _shifted(bits):
    return (bits.to(torch.int32) & 0xFFFF) << 16


# --- CPU -------------------------------------------------------------------------------------


def test_reference_places_the_bits_above_sixteen_zeros():
    bits, x = _all_patterns()
    assert torch.equal(op.reference(x).view(torch.int32).reshape(-1), _shifted(bits))


def test_guards():
    op.is_valid(torch.zeros(3, 8, dtype=torch.bfloat16))
    with pytest.raises(AssertionError, match="BF16"):
        op.is_valid(torch.zeros(3, 8, dtype=torch.float16))
    with pytest.raises(AssertionError, match="contiguous"):
        op.is_valid(torch.zeros(8, 3, dtype=torch.bfloat16).t())


# --- GPU -------------------------------------------------------------------------------------


@on_sm107
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(s["rows"])
    x = torch.randint(-(2**15), 2**15, (s["rows"], s["cols"]), generator=gen, device="cuda")
    x = x.to(torch.int16).view(torch.bfloat16)
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    out = op(x)
    assert out.dtype == torch.float32 and out.shape == x.shape and out.is_contiguous()
    assert torch.equal(out.view(torch.int32), op.reference(x).view(torch.int32))


@on_sm107
def test_every_bf16_pattern():
    bits, x = _all_patterns("cuda")
    with validating(op):
        out = op(x)
    assert torch.equal(out.view(torch.int32).reshape(-1), _shifted(bits))
    nan = torch.isnan(x)
    ref = x.float()
    assert torch.isnan(ref[nan]).all() and torch.isnan(out[nan]).all()
    assert torch.equal(out[~nan].view(torch.int32), ref[~nan].view(torch.int32))
