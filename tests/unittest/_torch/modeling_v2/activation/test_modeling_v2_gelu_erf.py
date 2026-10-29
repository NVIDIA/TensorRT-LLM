# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gelu_erf catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` is the
gate's other side. This file owns turning a spec into real tensors, the band,
and the measurements the entry's note states.

The entry's identity is *which* GELU it is. The wrong answer is the tanh
GELU, which qwen4_exp's block MLPs use in the same tower as its merger. It
stays within about one bf16 ulp of the right answer at the output's scale,
so the gate here is per element, in ulp of each element's own value. That is
what lets bf16 tell the two apart.

bf16 and fp16 have 65536 values each, so `test_every_finite_bf16_and_fp16_input`
drives the whole input domain of the two half types through the op once. The
cells then certify the shapes, layouts and launch sizes.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.gelu_erf import gelu_erf as op

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "gelu_erf requires a CUDA device"

MERGER_WIDTH = 4608
TOWER_HIDDEN = 1152
#: qwen4_exp model.visual.merger, measured from that checkpoint:
#: linear_fc1.weight std 0.010927, linear_fc1.bias std 0.002407, norm.weight
#: mean 1.174581 / std 0.140536. The real tensors give an activand at std
#: 0.875 / absmax 4.23.
FC1_WEIGHT_STD = 0.010927
FC1_BIAS_STD = 0.002407
NORM_GAMMA_MEAN = 1.174581
NORM_GAMMA_STD = 0.140536

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

#: torch.testing's untouched per-dtype defaults, restated only so the
#: discrimination tests can say how far outside that gate each variant lands.
_RTOL = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}
_ATOL = 1e-5

#: Explicit mantissa bits, for the ulp of a value in each dtype.
_MANTISSA = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}

#: Absolute allowance per unit |x| for the fp32 `1 + erf` term (see `_gate`).
_CANCEL = 2.0**-21

#: bf16 only: the most elements with |x| <= 4 that may differ from the
#: correctly rounded value (see `_gate`).
_BF16_NOT_CR_MAX = 1e-3
_CR_REGION = 4.0


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _uniform(shape, lo, hi, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty(shape, device="cuda", dtype=torch.float32).uniform_(lo, hi, generator=g)
    return x.to(dtype)


def _build(spec, seed):
    shape, dtype = spec["shape"], spec["dtype"]
    if spec.get("dist", "uniform") == "merger":
        g = torch.Generator(device="cuda").manual_seed(seed)
        rows = shape[0]
        gamma = NORM_GAMMA_MEAN + NORM_GAMMA_STD * torch.randn(
            TOWER_HIDDEN, generator=g, device="cuda"
        )
        hidden = torch.randn(4 * rows, TOWER_HIDDEN, generator=g, device="cuda")
        grouped = (hidden * gamma.to(dtype).float()).reshape(rows, MERGER_WIDTH)
        weight = FC1_WEIGHT_STD * torch.randn(
            MERGER_WIDTH, MERGER_WIDTH, generator=g, device="cuda"
        )
        bias = FC1_BIAS_STD * torch.randn(MERGER_WIDTH, generator=g, device="cuda")
        return (grouped @ weight.to(dtype).float().t() + bias.to(dtype).float()).to(dtype)
    layout = spec.get("layout", "contiguous")
    lo, hi = spec["lo"], spec["hi"]
    if layout == "column_slice":
        rows, width = shape
        x = _uniform((rows, 2 * width), lo, hi, dtype, seed)[:, :width]
        assert not x.is_contiguous() and x.stride(-1) == 1
        return x
    if layout == "transposed":
        rows, width = shape
        x = _uniform((width, rows), lo, hi, dtype, seed).t()
        assert x.stride(-1) != 1
        return x
    return _uniform(shape, lo, hi, dtype, seed)


def _ulp(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """One ulp of `dtype` at |value| (fp64 in, fp64 out); subnormals share the
    smallest normal's ulp."""
    magnitude = value.abs().clamp_min(torch.finfo(dtype).tiny)
    _, exponent = torch.frexp(magnitude)
    return torch.ldexp(torch.ones_like(magnitude), exponent - 1 - _MANTISSA[dtype])


def _gate(out: torch.Tensor, x: torch.Tensor) -> dict:
    """How far `out` is from the exact value, against the band. Does not assert.

    `default` is the worst ratio to torch's own dtype-aware band, against the
    reference rounded to `x.dtype`.

    `ulp` is the worst ratio to the per-element allowance, which is one ulp of
    the exact value plus |x| * 2^-21. The kernel evaluates
    `0.5 * x * (1 + erf(x / sqrt(2)))` in fp32 and rounds once, so:
    - the final rounding contributes half an ulp;
    - the fp32 sum `1 + erf` carries an absolute error of a few 2^-24 (half an
      fp32 ulp at 1, plus erf's own error at |erf| near 1), and `0.5 * x`
      scales that.
    Once `1 + erf` cancels (x below about -3 in fp32, -4 in bf16), that second
    term is all of the result: many ulp of a tiny value, but under |x| * 2^-23
    in absolute terms. The allowance takes 2x on the rounding and 4x on the
    cancellation. Measured at the worst element: 0.500 in bf16 and fp16 (the
    rounding), 0.19 in fp32.

    `not_cr` is the fraction, among elements with |x| <= 4, that differ from
    the correctly rounded value. It is gated in bf16 only. There the fp32
    evaluation is about 2^16 times finer than a bf16 ulp outside the
    cancellation, so a misrounding needs the exact value that close to a
    rounding tie. Measured: 0 of the 33026 bf16 values in [-4, 4]. fp16 has 3
    more mantissa bits, and 59 of its 34818 values in [-4, 4] misround, so the
    count is reported for fp16 and not gated. The ulp gate cannot see an erf
    scale of 1/1.414 for 1/sqrt(2) in bf16; it moves 3.87% of elements over
    [-3, 3], which is 39x this bound.
    """
    exact = op.reference(x.double())
    rounded = op.reference(x)
    dtype = x.dtype
    err = (out.double() - exact).abs()
    allowance = _ulp(exact, dtype) + x.double().abs() * _CANCEL
    default_band = _ATOL + _RTOL[dtype] * rounded.double().abs()
    region = x.double().abs() <= _CR_REGION
    differs = (out != rounded) & region
    in_region = int(region.sum())
    worst = (err / allowance).flatten()
    return dict(
        ulp=worst.max().item() if worst.numel() else 0.0,
        worst_x=x.flatten()[int(worst.argmax())].item() if worst.numel() else 0.0,
        default=((out.double() - rounded.double()).abs() / default_band).max().item()
        if out.numel()
        else 0.0,
        not_cr=int(differs.sum()) / in_region if in_region else 0.0,
        not_cr_count=int(differs.sum()),
        in_region=in_region,
    )


def _compare(out: torch.Tensor, x: torch.Tensor) -> dict:
    """The band: torch's default against the rounded reference, the per-element
    ulp allowance against the exact value, and in bf16 the correctly-rounded
    count. Each is derived in `_gate`."""
    torch.testing.assert_close(out, op.reference(x))
    stats = _gate(out, x)
    assert stats["ulp"] <= 1.0, stats
    if x.dtype == torch.bfloat16:
        assert stats["not_cr"] <= _BF16_NOT_CR_MAX, stats
    return stats


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x = _build(cell.spec, seed=_seed(cell.why))
    before = x.clone()
    out = op(x)
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    if x.numel():
        assert out.data_ptr() != x.data_ptr(), "the entry is out-of-place"
    assert torch.equal(before, x), "the entry must not mutate its input"
    if cell.spec.get("layout") == "transposed":
        assert out.stride() == x.stride(), "the output keeps the input's memory format"
        assert torch.equal(out, op(x.contiguous()))
    if cell.spec.get("layout") == "column_slice":
        assert out.is_contiguous()
        assert torch.equal(out, op(x.contiguous()))
    stats = _compare(out, x)
    print(f"\n  {cell.why}: {stats}")


def _all_finite(dtype: torch.dtype) -> torch.Tensor:
    bits = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda").to(torch.int16)
    values = bits.view(dtype)
    return values[torch.isfinite(values)].contiguous()


def test_every_finite_bf16_and_fp16_input() -> None:
    """The whole input domain of each half type, once: every finite value is
    inside the band, and bf16 is correctly rounded on all of [-4, 4]. Also
    pins where the fp32 `1 + erf` cancellation turns the bf16 output into
    -0.0, which the note states."""
    for dtype, count, in_region in (
        (torch.bfloat16, 65280, 33026),
        (torch.float16, 63488, 34818),
    ):
        x = _all_finite(dtype)
        assert x.numel() == count
        out = op(x)
        stats = _compare(out, x)
        assert stats["in_region"] == in_region
        print(f"\n  exhaustive {dtype}: {stats}")
        if dtype == torch.bfloat16:
            assert stats["not_cr_count"] == 0, "every bf16 input in [-4, 4] rounds correctly"

    x = _all_finite(torch.bfloat16)
    out = op(x)
    deep = x.double() < -1.0
    negative_zero = (out == 0) & torch.signbit(out)
    onset = x[deep & negative_zero].double().max().item()
    exact_at_onset = op.reference(x.new_tensor([onset]).double()).item()
    print(f"  bf16: -0.0 from x = {onset} down; the exact value there is {exact_at_onset:.3e}")
    assert -5.7 < onset < -5.4, onset
    below = x.double() <= onset
    assert bool(negative_zero[below].all()), "every bf16 input from the onset down returns -0.0"
    tail = int(((out != op.reference(x)) & (x.double() < -_CR_REGION)).sum())
    print(f"  bf16: {tail} inputs below -4 are not correctly rounded")


def _erf64(x):
    return op.reference(x.double())


def _tanh64(x):
    """The *other* GELU -- `gelu_pytorch_tanh`, qwen4_exp's block-MLP activation."""
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.tanh(SQRT_2_OVER_PI * (x64 + 0.044715 * x64 * x64 * x64)))


def test_identity_is_the_erf_form_and_not_tanh() -> None:
    """Measured in fp32, where the deviation from the exact value is far below
    the gap between the two GELUs."""
    x32 = _uniform((512, MERGER_WIDTH), -4.0, 4.0, torch.float32, 24)
    out = op(x32).double()
    ref = _erf64(x32)
    correct = (out - ref).abs().max().item()
    gate = _ATOL + _RTOL[torch.float32] * ref.abs().max().item()

    # The entry's own gate refuses the tanh form in fp32.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out.float(), _tanh64(x32).float())

    x64 = x32.double()
    variants = {
        "tanh gelu (gelu_pytorch_tanh)": _tanh64(x32),
        "sigmoid approx x*sigmoid(1.702x)": x64 * torch.sigmoid(1.702 * x64),
        "silu": x64 * torch.sigmoid(x64),
        "relu": torch.clamp(x64, min=0.0),
    }
    print(f"\n  fp32: gate {gate:.3e}; correct-path deviation {correct:.3e}")
    for name, wrong in variants.items():
        dev = (out - wrong).abs().max().item()
        print(
            f"  {name:36s} dev {dev:.6e} ({dev / correct:9.1f}x correct, {dev / gate:7.1f}x gate)"
        )
        assert dev > 100.0 * correct, (name, dev, correct)
        assert dev > 10.0 * gate, (name, dev, gate)

    # erf's scale truncated from 1/sqrt(2) to 1/1.414 -- a plausible
    # transcription slip, weak by construction, so a lower multiple.
    perturbed = 0.5 * x64 * (1.0 + torch.erf(x64 / 1.414))
    dev = (out - perturbed).abs().max().item()
    print(f"  {'erf scale 1/sqrt(2) -> 1/1.414':36s} dev {dev:.6e} ({dev / gate:7.1f}x gate)")
    assert dev > 20.0 * correct, (dev, correct)
    assert dev > 1.5 * gate, (dev, gate)


def test_extremes_and_non_finite() -> None:
    """Saturation, signed-zero underflow, no overflow at finfo.max, -inf -> nan."""
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        finfo = torch.finfo(dtype)
        probe = torch.tensor(
            [0.0, -0.0, 1e-3, -1e-3, 20.0, -20.0, 1000.0, -1000.0, finfo.max, -finfo.max],
            device="cuda",
            dtype=dtype,
        )
        out = op(probe)
        torch.testing.assert_close(out, op.reference(probe))
        assert out[4].item() == 20.0 and out[6].item() == 1000.0
        assert out[8].item() == finfo.max, "no overflow at the top of the dtype's range"
        for index in (5, 7, 9):
            assert out[index].item() == 0.0 and math.copysign(1.0, out[index].item()) < 0

        nonfinite = torch.tensor(
            [float("inf"), float("-inf"), float("nan")], device="cuda", dtype=dtype
        )
        got = op(nonfinite)
        assert got[0].item() == float("inf")
        assert math.isnan(got[1].item()), "gelu_erf(-inf) is nan"
        assert math.isnan(got[2].item())
        ref = _erf64(nonfinite)
        assert ref[0].item() == float("inf"), "the closed form agrees: the function, not the kernel"
        assert math.isnan(ref[1].item()) and math.isnan(ref[2].item())
