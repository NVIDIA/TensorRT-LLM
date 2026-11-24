# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's FP32 router takes its input from rubin_upcast_fp32: the same router logits, bit for bit.

A dep16 generation server widens the MoE input ``x`` (BF16 ``[rows, 7168]``) for its FP32 router
GEMM in ``_moe_router_input``: ``rubin_upcast_fp32`` on a contiguous ``x`` instead of
``x.float()``. A BF16 value is an FP32 value (its FP32 bits are its 16 bits above 16 zero bits),
so the two agree on every element but a NaN, where both give a NaN (the entry keeps the payload;
torch's GPU cast may make it canonical). ``xf`` has one reader, the router GEMM.

GPU (sm_107; the file collects and skips elsewhere), at an L GEN step's rows and lighter steps'
(``[96 | 64 | 32 | 8, 7168]``), with planted signed zeros, subnormals, infinities, the largest
finite values and NaN payloads of both signs, quiet and signaling:

* the widening: the entry's bits are the BF16 bits above 16 zero bits on every element, and
  ``x.float()``'s on every element that is not a NaN; a NaN stays a NaN;
* the router logits ``rubin_linear(xf, router, False)`` of the two widenings, bit for bit, on
  every row, including the rows that hold a NaN;
* the target's ``_moe_router_input`` and ``_moe_router_fp32`` against ``x.float()``'s bits and
  logits, on a contiguous ``x`` (the entry) and a non-contiguous one (``x.float()``).
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_DEP16 = _FAMILY + ".kimi_k31_pro_nvfp4__sm_107__dep16.modeling"
_HIDDEN, _EXPERTS = 7168, 896
_ROWS = (96, 64, 32, 8)
# BF16 bit patterns planted in row 0 (no NaN) and row 1 (NaNs): signed zeros, the smallest,
# largest and inner subnormals, the smallest normals, the largest finite values, infinities.
_SPECIAL = (0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0040, 0x8035, 0x0080, 0x8080)
_SPECIAL += (0x7F7F, 0xFF7F, 0x7F80, 0xFF80)
# NaN payloads of both signs, quiet (bit 6 of the mantissa set) and signaling.
_NANS = (0x7FC0, 0xFFC0, 0x7F81, 0xFF81, 0x7FFF, 0xFFFF, 0x7FA5, 0xFFEE, 0x7FC1, 0x7F80 | 0x2A)

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the dep16 target and its entries run on sm_107",
)


def _ops():
    linear = importlib.import_module(_FAMILY + ".ops.gemm.rubin_linear").rubin_linear
    upcast = importlib.import_module(_FAMILY + ".ops.gemm.rubin_upcast_fp32").rubin_upcast_fp32
    return linear, upcast


def _bits16(patterns) -> torch.Tensor:
    signed = [p - 0x10000 if p >= 0x8000 else p for p in patterns]
    return torch.tensor(signed, dtype=torch.int16, device="cuda")


def _planted(rows: int, seed: int) -> torch.Tensor:
    """``[rows, 7168]`` BF16 of a unit normal, the specials in row 0 and the NaNs in row 1, each
    pattern twice (near the start and near the end of the row)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, _HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
    bits = x.view(torch.int16)
    for row, patterns in ((0, _SPECIAL), (1, _NANS)):
        p = _bits16(patterns)
        bits[row, 3 : 3 + p.numel()] = p
        bits[row, _HIDDEN - 5 - p.numel() : _HIDDEN - 5] = p
    return x


def _shifted(x: torch.Tensor) -> torch.Tensor:
    return (x.view(torch.int16).to(torch.int32) & 0xFFFF) << 16


def _router(seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(_EXPERTS, _HIDDEN, generator=gen, device="cuda") / _HIDDEN**0.5


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.view(torch.int32), b.view(torch.int32))
    )


@on_sm107
@pytest.mark.parametrize("rows", _ROWS)
def test_the_entry_widens_every_planted_value_as_x_float_does(rows):
    _, upcast = _ops()
    x = _planted(rows, rows)
    out, ref = upcast(x), x.float()
    assert out.is_contiguous() and out.shape == x.shape and out.dtype == torch.float32
    assert torch.equal(out.view(torch.int32), _shifted(x)), "the BF16 bits above 16 zero bits"
    nan = torch.isnan(x)
    assert int(nan.sum()) == 2 * len(_NANS) and bool(nan[1].any()) and not bool(nan[0].any())
    assert torch.equal(out[~nan].view(torch.int32), ref[~nan].view(torch.int32))
    assert torch.isnan(ref[nan]).all() and torch.isnan(out[nan]).all()


@on_sm107
@pytest.mark.parametrize("rows", _ROWS)
def test_the_router_logits_of_both_widenings_are_the_same_bits(rows):
    linear, upcast = _ops()
    x, w = _planted(rows, 100 + rows), _router(200 + rows)
    a = linear(x.float(), w, False)
    b = linear(upcast(x), w, False)
    assert a.shape == (rows, _EXPERTS) and a.dtype == torch.float32
    nan_rows = torch.isnan(x).any(-1)
    assert _same(a[~nan_rows], b[~nan_rows]), "logits of the rows without a NaN"
    assert torch.isnan(a[nan_rows]).all() and torch.isnan(b[nan_rows]).all()
    assert _same(a[nan_rows], b[nan_rows]), "logits of the rows that hold a NaN"


@on_sm107
def test_the_target_router_takes_the_widened_input():
    linear, _ = _ops()
    M = importlib.import_module(_DEP16)
    w = _router(301)
    layer = SimpleNamespace(router=w)
    x = _planted(96, 300)
    xf = M._moe_router_input(x)
    assert torch.equal(xf.view(torch.int32), _shifted(x))
    assert _same(M._moe_router_fp32(layer, x), linear(x.float(), w, False))
    xt = _planted(96, 302).t().contiguous().t()  # the same values, not contiguous
    assert not xt.is_contiguous()
    assert _same(M._moe_router_input(xt), xt.float())
    assert _same(M._moe_router_fp32(layer, xt), linear(xt.float(), w, False))
