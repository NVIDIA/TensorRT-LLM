# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp gelu_erf entry.

The entry's identity is *which* GELU it is, and the wrong answer -- the tanh
GELU this same tower's block MLPs use -- rounds to within about one bf16 ulp of
it. The discrimination test is the point of this file; only fp32 tells the two
apart.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.activation.gelu_erf import (  # noqa: E501
    gelu_erf as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "gelu_erf requires a CUDA device"

MERGER_WIDTH = 4608
TOWER_HIDDEN = 1152
#: model.visual.merger, measured from the checkpoint: linear_fc1.weight std
#: 0.010927, linear_fc1.bias std 0.002407, norm.weight mean 1.174581 / std
#: 0.140536. The real tensors give an activand at std 0.875 / absmax 4.23.
FC1_WEIGHT_STD = 0.010927
FC1_BIAS_STD = 0.002407
NORM_GAMMA_MEAN = 1.174581
NORM_GAMMA_STD = 0.140536

SQRT_2 = math.sqrt(2.0)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

#: torch.testing's untouched per-dtype defaults, restated only so the
#: discrimination test can say how far outside that gate each variant lands.
_RTOL = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}
_ATOL = 1e-5


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
    torch.testing.assert_close(out, op.reference(x))


def _erf64(x):
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.erf(x64 / SQRT_2))


def _tanh64(x):
    """The *other* GELU -- `gelu_pytorch_tanh`, this tower's block-MLP activation."""
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.tanh(SQRT_2_OVER_PI * (x64 + 0.044715 * x64 * x64 * x64)))


def test_identity_is_the_erf_form_and_not_tanh() -> None:
    """Measured in fp32, where the question has an answer; and shown to have
    none in bf16, so a bf16 match cannot quietly be read as confirmation."""
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

    # bf16 cannot see the difference.
    xb = x32.to(torch.bfloat16)
    outb = op(xb).double()
    correct_b = (outb - _erf64(xb)).abs().max().item()
    tanh_b = (outb - _tanh64(xb)).abs().max().item()
    print(f"  bf16: correct {correct_b:.6e}; tanh variant {tanh_b:.6e} ({tanh_b / correct_b:.3f}x)")
    assert tanh_b < 4.0 * correct_b, (tanh_b, correct_b)
    torch.testing.assert_close(outb.to(torch.bfloat16), op.reference(xb))


def test_peak_divergence_between_the_two_gelus() -> None:
    """Where and how far apart the two GELUs are: the gap bounds every claim
    about which one an implementation is."""
    xs = torch.linspace(-8.0, 8.0, 2_000_001, device="cuda", dtype=torch.float64)
    gap = (_erf64(xs) - _tanh64(xs)).abs()
    i = int(gap.argmax().item())
    print(f"\n  max|erf - tanh| = {gap[i].item():.6e} at x = {xs[i].item():.4f}")
    assert 4.0e-4 < gap[i].item() < 1e-3


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


def test_rank_does_not_change_the_arithmetic() -> None:
    """The one engine op computing this function rejects non-2-D input; this one does not."""
    x = _uniform((2, 37, MERGER_WIDTH), -16.0, 16.0, torch.bfloat16, 26)
    assert torch.equal(op(x).reshape(-1, MERGER_WIDTH), op(x.reshape(-1, MERGER_WIDTH)))


def test_rows_are_independent() -> None:
    """The 4-to-1 grouping in front of the merger reorders rows; the activation
    must not care. Permuting rows permutes the result, bitwise."""
    x = _uniform((140, MERGER_WIDTH), -16.0, 16.0, torch.bfloat16, 27)
    order = torch.randperm(140, generator=torch.Generator().manual_seed(0)).to("cuda")
    assert torch.equal(op(x[order].contiguous()), op(x)[order])


def test_cuda_graph_capture_and_replay() -> None:
    x = _uniform((64, MERGER_WIDTH), -16.0, 16.0, torch.bfloat16, 28)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(x)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x)

    x.copy_(_uniform((64, MERGER_WIDTH), -16.0, 16.0, torch.bfloat16, 29))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, op.reference(x))
