# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the tanh_softcap catalog entry.

Every cell is gated twice against the entry's fp64 reference: torch's fp32
default band, and a per-element ulp bound derived at `MAX_ULP`. The second is
the one that bites near zero, where the default atol of 1e-5 admits any
relative error at all. Cells are compared `_CHUNK_ROWS` rows at a time, so the
context-logits cells check every element without an fp64 copy of the whole.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.tanh_softcap import (
    tanh_softcap as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "tanh_softcap requires a CUDA device"

CAP = 30.0
VOCAB = 204800

#: The smallest fp32 whose cap is exactly 30.0: the fp32 tanh rounds to 1 there.
SATURATION = 270.327392578125

#: Per-element bound, in fp32 ulps of the entry's reference (the fp64 closed
#: form rounded once to fp32). The op computes q = fl(x * fl(1 / cap)), two
#: roundings, so q = (x / cap)(1 + e) with |e| <= 2u, u = 2^-24; t = tanhf(q),
#: within the CUDA math library's documented 2 ulp, at most 4u relative;
#: o = fl(t * cap), at most u. tanh's relative condition number 2y / sinh(2y)
#: is at most 1, so e passes through no larger. That is 7u relative, and one
#: ulp of an fp32 value v exceeds u * |v|: 7 ulp from the fp64 value, plus 0.5
#: for rounding the reference to fp32.
MAX_ULP = 7.5

#: torch.testing's untouched fp32 defaults, restated only so the
#: discrimination test can say how far outside that gate each variant lands.
_RTOL, _ATOL = 1.3e-6, 1e-5

#: Rows compared at once: 1024 x 204800 is 0.8 GiB in fp32, 1.6 GiB in fp64.
_CHUNK_ROWS = 1024


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _uniform(shape, lo, hi, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty(shape, device="cuda", dtype=torch.float32).uniform_(lo, hi, generator=g)
    return x.to(dtype)


def _ulps(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """|out - ref| in units of one fp32 ulp of `ref`."""
    mag = ref.abs()
    ulp = torch.nextafter(mag, torch.full_like(mag, math.inf)) - mag
    return (out.double() - ref.double()).abs() / ulp.double()


def _compare(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Torch's fp32 default band, then the derived `MAX_ULP` bound; returns the worst ulp."""
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    torch.testing.assert_close(out, ref)
    worst = _ulps(out, ref).max().item()
    assert worst <= MAX_ULP, f"{worst:.3f} ulp from the reference exceeds {MAX_ULP}"
    return worst


def _cell_params():
    return [
        pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS
    ]


@pytest.mark.parametrize("cell", _cell_params(), ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    shape, cap, seed = (spec["rows"], spec["vocab"]), spec["cap"], _seed(cell.why)
    lo, hi, dtype = spec["lo"], spec["hi"], spec["dtype"]
    x = _uniform(shape, lo, hi, dtype, seed)
    peak = ""
    if spec["graph"]:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                op(x, cap)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op(x, cap)
        seed += 1
        x.copy_(_uniform(shape, lo, hi, dtype, seed))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, op(x, cap)), "a replay is bitwise the eager call on its logits"
    else:
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = op(x, cap)
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - held
        size = x.numel() * x.element_size()
        peak = f"; peak {extra / 2**30:.3f} GiB beyond the input ({extra / size:.2f}x its size)"

    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    assert out.data_ptr() != x.data_ptr(), "the entry is out-of-place"
    assert torch.equal(x, _uniform(shape, lo, hi, dtype, seed)), "the entry must not mutate x"

    for a, b in spec.get("windows", ()):
        assert torch.equal(op(x[a:b], cap), out[a:b]), (
            f"rows [{a}, {b}) alone differ from the same rows of the {shape[0]}-row call"
        )

    worst, saturated = 0.0, 0
    for a in range(0, shape[0], _CHUNK_ROWS):
        xs, outs = x[a : a + _CHUNK_ROWS], out[a : a + _CHUNK_ROWS]
        worst = max(worst, _compare(outs, op.reference(xs, cap)))
        sat = xs.abs() >= SATURATION
        assert torch.equal(outs[sat], torch.sign(xs[sat]) * cap), "saturates to exactly cap"
        saturated += sat.sum().item()
    if not spec["graph"]:
        peak += f"; the cell's peak {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB"
    print(f"\n  {cell.why}: worst {worst:.3f} ulp; {saturated} saturated{peak}")


def test_band_rejects_wrong_variants() -> None:
    """Every plausible wrong soft-cap lands far outside the gate, the bf16 path included."""
    x = _uniform((16, VOCAB), -60.0, 60.0, torch.float32, 31)
    out = op(x, CAP)
    ref = op.reference(x, CAP)
    correct = (out - ref).abs().max().item()
    gate = _ATOL + _RTOL * ref.abs().max().item()
    xb = x.to(torch.bfloat16)
    variants = {
        "bf16 logits path, each step rounded": (torch.tanh(xb / CAP) * CAP).float(),
        "hard clamp to [-cap, cap]": x.clamp(-CAP, CAP),
        "no cap": x.clone(),
        "tanh(x) * cap": torch.tanh(x) * CAP,
        "cap 30.5": torch.tanh(x / 30.5) * 30.5,
    }
    print(f"\n  gate {gate:.3e} at the output's scale; correct path {correct:.3e}")
    for name, wrong in variants.items():
        dev = (wrong - ref).abs().max().item()
        print(f"  {name:38s} dev {dev:.4e} ({dev / gate:10.1f}x gate)")
        assert dev > 100.0 * gate, (name, dev, gate)
        with pytest.raises(AssertionError):
            _compare(wrong, ref)


def test_saturation_and_non_finite() -> None:
    finfo = torch.finfo(torch.float32)
    below = torch.nextafter(torch.tensor(SATURATION), torch.tensor(0.0)).item()
    probe = torch.tensor(
        [0.0, -0.0, SATURATION, -SATURATION, below, -below, finfo.max, -finfo.max],
        device="cuda",
    )
    out = op(probe, CAP)
    assert out[0].item() == 0.0 and not torch.signbit(out[0])
    assert out[1].item() == 0.0 and torch.signbit(out[1])
    assert out[2].item() == CAP and out[3].item() == -CAP
    assert out[4].item() < CAP and out[5].item() > -CAP, "SATURATION is the first fp32 at the cap"
    assert out[6].item() == CAP and out[7].item() == -CAP
    _compare(out, op.reference(probe, CAP))

    nonfinite = torch.tensor([math.inf, -math.inf, math.nan], device="cuda")
    got = op(nonfinite, CAP)
    assert got[0].item() == CAP and got[1].item() == -CAP
    assert math.isnan(got[2].item())

    # A subnormal quotient: relative precision goes, the absolute error stays tiny.
    tiny = torch.tensor([1e-40, -1e-40, finfo.tiny], device="cuda")
    got, ref = op(tiny, CAP), op.reference(tiny, CAP)
    assert torch.equal(torch.sign(got), torch.sign(ref))
    assert (got.double() - ref.double()).abs().max().item() < 1e-43


def test_python_float_cap_is_a_reciprocal_multiply() -> None:
    """The class docstring's rounding sequence, and why `cap` stays a Python float."""
    x = _uniform((16, VOCAB), -400.0, 400.0, torch.float32, 32)
    inv = torch.tensor(1.0 / CAP, dtype=torch.float32).item()
    assert torch.equal(x / CAP, x * inv)
    true_div = x / torch.tensor(CAP, device="cuda")
    assert not torch.equal(x / CAP, true_div)
    assert _ulps(x / CAP, true_div).max().item() <= 1.0


def test_padded_rows_never_reach_live_rows() -> None:
    """A padded decode capture runs dummy rows beside the live ones."""
    x = _uniform((16, VOCAB), -400.0, 400.0, torch.float32, 33)
    live = x[:3].clone()
    x[3:] = math.nan
    x[3:, ::2] = math.inf
    assert torch.equal(op(x, CAP)[:3], op(live, CAP))


@pytest.mark.parametrize(
    "cap", [0.0, -0.0, math.inf, -math.inf, math.nan], ids=["0", "-0", "inf", "-inf", "nan"]
)
def test_guard_refuses_degenerate_cap(cap) -> None:
    x = _uniform((2, VOCAB), -60.0, 60.0, torch.float32, 34)
    x[0, 0] = 0.0
    # Unguarded, the op accepts the cap and answers with zeros and nan.
    got = op(x, cap)
    if cap == 0.0:
        assert math.isnan(got[0, 0].item())
        finite = got[~torch.isnan(got)]
        assert finite.numel() == got.numel() - 1 and (finite == 0.0).all()
    else:
        assert torch.isnan(got).all()
    with validating(op):
        with pytest.raises(ValueError, match="cap must be finite and non-zero"):
            op(x, cap)


def test_guard_admits_a_negative_cap_which_computes_its_magnitude() -> None:
    x = _uniform((16, VOCAB), -400.0, 400.0, torch.float32, 35)
    with validating(op):
        assert torch.equal(op(x, -CAP), op(x, CAP))
