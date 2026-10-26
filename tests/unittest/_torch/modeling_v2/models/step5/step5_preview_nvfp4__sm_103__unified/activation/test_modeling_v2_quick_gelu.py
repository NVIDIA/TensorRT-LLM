# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the step5 quick_gelu entry.

A bf16 elementwise op has only 65536 inputs, so one cell drives all of them and
the gate is stated per element: the correctly rounded exact value, or one ulp
off only where the exact value lies within the fp32 evaluation's own error
bound of a bf16 rounding midpoint. That gate also refuses transformers' bf16
chain, the erf and tanh GELUs, SiLU and a 1.7 gain.
"""

import math
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.activation.quick_gelu import (  # noqa: E501
    ALPHA,
    FLUSH_BELOW,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.activation.quick_gelu import (  # noqa: E501
    quick_gelu as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "quick_gelu requires a CUDA device"

#: Relative error of the fp32 argument `fl(x * fl32(1.702))`: fp32(1.702) is
#: 1.29e-8 relative from 1.702 and the product rounds by at most 2^-24
#: (5.96e-8). The sigmoid's relative sensitivity to its argument z is
#: `1 - sigmoid(z)`, so this costs `(1 - s) * |z| * ARG_REL` of the result.
ARG_REL = 7.3e-8
#: expf (at most 2 ulp, 2^-22), the `1 + e` add, the division and the final
#: product (2^-24 each): at most 1.75 * 2^-22, bounded by 2^-21.
EVAL_REL = 2.0**-21

#: Smallest normal bf16 (and fp32) magnitude, and bf16's subnormal step.
MIN_NORMAL = 2.0**-126
SUBNORMAL_STEP = 2.0**-133
#: Bound on the exact magnitude in the flush tail: |x| e^{-1.702 |x|} at
#: x = FLUSH_BELOW is 52.13 * e^-88.72 = 1.6e-37 < 2^-122.
FLUSH_MAGNITUDE = 2.0**-122

BULK_STD = 3.0
TAIL = 48.0
TAIL_EVERY = 8


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _every_bf16(rows: int, width: int) -> torch.Tensor:
    """All 65536 bf16 bit patterns once, then repeated to fill `[rows, width]`."""
    assert rows * width >= 2**16
    bits = torch.arange(rows * width, dtype=torch.int32, device="cuda") % 2**16 - 2**15
    return bits.to(torch.int16).view(torch.bfloat16).view(rows, width)


def _activand(rows: int, width: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Bulk N(0, BULK_STD^2) with every TAIL_EVERY-th column uniform over
    [-TAIL, TAIL]: both saturation tails (out ~ x above 5, out ~ 0 below -5),
    short of the flush tail below -52.13.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = BULK_STD * torch.randn(rows, width, generator=g, device="cuda")
    tail = torch.empty(rows, width, device="cuda").uniform_(-TAIL, TAIL, generator=g)
    x[:, ::TAIL_EVERY] = tail[:, ::TAIL_EVERY]
    return x.to(dtype)


def _build(spec: dict, seed: int) -> torch.Tensor:
    if spec["dist"] == "every_bf16":
        assert spec["dtype"] == torch.bfloat16
        return _every_bf16(spec["rows"], spec["width"])
    return _activand(spec["rows"], spec["width"], spec["dtype"], seed)


def _ordered(t: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns as integers ordered like the values they encode."""
    bits = t.contiguous().view(torch.int16).int()
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _from_ordered(o: torch.Tensor) -> torch.Tensor:
    bits = torch.where(o < 0, (-o) | 0x8000, o)
    bits = torch.where(bits >= 2**15, bits - 2**16, bits)
    return bits.to(torch.int16).view(torch.bfloat16)


def _midpoint_distance(exact: torch.Tensor, rounded: torch.Tensor) -> torch.Tensor:
    """Relative distance of each exact value from the nearer bf16 rounding
    midpoint around its correctly rounded value.
    """
    o = _ordered(rounded)
    r = rounded.double()
    above = (r + _from_ordered(o + 1).double()) / 2
    below = (r + _from_ordered(o - 1).double()) / 2
    return torch.minimum((exact - above).abs(), (exact - below).abs()) / exact.abs()


def _bound(x: torch.Tensor) -> torch.Tensor:
    """The fp32 evaluation's relative error bound at each `x` (see ARG_REL)."""
    z = ALPHA * x.double()
    return (1.0 - torch.sigmoid(z)) * z.abs() * ARG_REL + EVAL_REL


def _compare(x: torch.Tensor, out: torch.Tensor) -> dict:
    """Gate `out` against the entry's reference, element by element:

    - NaN exactly where the reference is NaN; infinities equal;
    - below FLUSH_BELOW (fp32 `exp` overflow): a signed zero, and the exact
      value is below FLUSH_MAGNITUDE;
    - a subnormal exact result: within one bf16 subnormal step;
    - otherwise the correctly rounded value, or one ulp off where the exact
      value lies within `_bound(x)` of a rounding midpoint.
    """
    assert out.shape == x.shape and out.dtype == x.dtype == torch.bfloat16
    rounded = op.reference(x)
    exact = op.reference(x.double())
    nan = torch.isnan(rounded)
    assert torch.equal(torch.isnan(out), nan), "NaN pattern differs from the reference"
    inf = torch.isinf(rounded)
    assert torch.equal(out[inf], rounded[inf]), "infinities differ from the reference"
    xf = x.float()
    flush = (xf < FLUSH_BELOW) & ~nan
    assert (out[flush] == 0).all(), "the flush tail returned a nonzero value"
    assert (exact[flush].abs() < FLUSH_MAGNITUDE).all(), "exact value too large to flush"
    rest = ~nan & ~inf & ~flush
    tiny = rest & (exact.abs() < MIN_NORMAL)
    if tiny.any():
        sub = (out[tiny].double() - rounded[tiny].double()).abs().max().item()
        assert sub <= SUBNORMAL_STEP, f"subnormal result {sub:.3e} away"
    normal = rest & ~tiny
    dist = (_ordered(out[normal]) - _ordered(rounded[normal])).abs()
    near = _midpoint_distance(exact[normal], rounded[normal]) <= _bound(x[normal])
    bad = (dist > 1) | ((dist == 1) & ~near)
    worst = dist.max().item() if dist.numel() else 0
    assert not bad.any(), (
        f"{int(bad.sum())} of {dist.numel()} elements outside the band "
        f"(max {worst} ulp; first inputs {xf[normal][bad][:6].tolist()})"
    )
    return dict(
        normal=dist.numel(),
        off=int((dist > 0).sum()),
        worst=worst,
        flushed=int(flush.sum()),
        subnormal=int(tiny.sum()),
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    x = _build(spec, _seed(cell.why))
    before = x.clone()
    out = op(x)
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    assert torch.equal(before.view(torch.int16), x.view(torch.int16)), "input mutated"
    stats = _compare(x, out)
    print(f"\n  {cell.why}: {stats}")


def test_every_bf16_finite_input_is_correctly_rounded_but_two_classes():
    """Pins the `note`'s numerics: across the finite bf16 inputs, the only
    outputs that are not the correctly rounded exact value are the flush tail
    (x in [-56.75, -52.25] returns zero for a nonzero bf16 result) and the
    midpoint cases the band admits.
    """
    x = _every_bf16(8, 8960).reshape(-1)[: 2**16]
    x = x[torch.isfinite(x)]
    out = op(x)
    rounded = op.reference(x)
    xf = x.float()
    wrong = out.view(torch.int16) != rounded.view(torch.int16)
    wrong &= ~((out == 0) & (rounded == 0))
    flush_wrong = wrong & (xf < FLUSH_BELOW)
    other = wrong & (xf >= FLUSH_BELOW)
    print(
        f"\n  finite inputs {x.numel()}: {int(flush_wrong.sum())} flushed to zero "
        f"(x in [{xf[flush_wrong].min().item()}, {xf[flush_wrong].max().item()}]); "
        f"{int(other.sum())} other off-by-ulp inputs {xf[other].tolist()}"
    )
    assert xf[flush_wrong].max().item() < FLUSH_BELOW
    assert xf[flush_wrong].min().item() >= -57.0
    assert int(other.sum()) <= 8, "more midpoint misses than the note records"


def _quick_gelu_bf16_chain(x: torch.Tensor) -> torch.Tensor:
    """transformers' `QuickGELUActivation` on a bf16 tensor: three roundings."""
    return x * torch.sigmoid(1.702 * x)


def _once(fn):
    def rounded_once(x: torch.Tensor) -> torch.Tensor:
        return fn(x.float()).to(x.dtype)

    return rounded_once


WRONG = {
    "transformers bf16 chain": _quick_gelu_bf16_chain,
    "erf GELU": _once(F.gelu),
    "tanh GELU": _once(lambda t: F.gelu(t, approximate="tanh")),
    "SiLU (gain 1)": _once(F.silu),
    "gain 1.7": _once(lambda t: t * torch.sigmoid(1.7 * t)),
}


@pytest.mark.parametrize("name", list(WRONG), ids=list(WRONG))
def test_gate_refuses_other_activations(name):
    """The band is tight enough to tell QuickGELU from its neighbours, and from
    the bf16 evaluation of itself.
    """
    x = _every_bf16(8, 8960)
    wrong = WRONG[name](x)
    rounded = op.reference(x)
    finite = torch.isfinite(rounded) & torch.isfinite(wrong)
    dist = (_ordered(wrong[finite]) - _ordered(rounded[finite])).abs()
    print(f"\n  {name}: {int((dist > 0).sum())} elements off, max {dist.max().item()} ulp")
    with pytest.raises(AssertionError):
        _compare(x, wrong)


def test_band_derivation_holds_in_fp32():
    """The premise of `_bound`: the entry's fp32 result stays within it of the
    exact value on every input with a normal result, short of the flush tail.
    """
    x = _every_bf16(8, 8960).reshape(-1)[: 2**16]
    x = x[torch.isfinite(x)]
    x32 = x.float()
    out32 = op(x32).double()
    exact = op.reference(x.double())
    keep = (x32 >= FLUSH_BELOW) & (exact.abs() >= MIN_NORMAL)
    rel = (out32[keep] - exact[keep]).abs() / exact[keep].abs()
    ratio = (rel / _bound(x[keep])).max().item()
    print(f"\n  max fp32 relative error {rel.max().item():.3e}, {ratio:.3f} of the bound")
    assert ratio <= 1.0, ratio
    assert math.isclose(FLUSH_BELOW, -52.129, abs_tol=1e-3), FLUSH_BELOW


def test_is_valid_admits_the_call_site():
    """No input is refused: the composition has no metadata it can misread."""
    x = _activand(4, 8960, torch.bfloat16, 7)
    with validating(op):
        op(x)
