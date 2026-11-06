# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp gelu_tanh entry.

The entry's identity is *which* GELU it is, so the discrimination test is the
point of this file: in bf16 the tanh and erf GELUs round to within about one
ulp of each other, and only fp32 tells them apart.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.activation.gelu_tanh import (  # noqa: E501
    gelu_tanh as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "gelu_tanh requires a CUDA device"

INTERMEDIATE_SIZE = 4304
HIDDEN_SIZE = 1152
#: model.visual.blocks.*.mlp.linear_fc1, measured from the checkpoint: weight
#: std 0.0197 (block 0), bias std 1.676 / absmax 9.44 -- the bias dominates.
FC1_WEIGHT_STD = 0.0197
FC1_BIAS_STD = 1.676

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
    if spec.get("dist", "uniform") == "fc1":
        g = torch.Generator(device="cuda").manual_seed(seed)
        rows = shape[0]
        hidden = torch.randn(rows, HIDDEN_SIZE, generator=g, device="cuda").to(dtype)
        weight = (
            FC1_WEIGHT_STD * torch.randn(shape[-1], HIDDEN_SIZE, generator=g, device="cuda")
        ).to(dtype)
        bias = (FC1_BIAS_STD * torch.randn(shape[-1], generator=g, device="cuda")).to(dtype)
        return (hidden.float() @ weight.float().t() + bias.float()).to(dtype)
    lo, hi = spec["lo"], spec["hi"]
    return _uniform(shape, lo, hi, dtype, seed)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x = _build(cell.spec, seed=_seed(cell.why))
    before = x.clone()
    out = op(x)
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    assert out.data_ptr() != x.data_ptr(), "the entry is out-of-place"
    assert torch.equal(before, x), "the entry must not mutate its input"
    torch.testing.assert_close(out, op.reference(x))


def _tanh64(x):
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.tanh(SQRT_2_OVER_PI * (x64 + 0.044715 * x64 * x64 * x64)))


def _erf64(x):
    """The *other* GELU -- exact, erf-based; `torch.nn.GELU()` and `moe_gelu`."""
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.erf(x64 / math.sqrt(2.0)))


def test_identity_is_the_tanh_form_and_not_erf() -> None:
    """Measured in fp32, where the question has an answer; and shown to have
    none in bf16, so a bf16 match cannot quietly be read as confirmation."""
    x32 = _uniform((512, INTERMEDIATE_SIZE), -4.0, 4.0, torch.float32, 24)
    out = op(x32).double()
    ref = _tanh64(x32)
    correct = (out - ref).abs().max().item()
    gate = _ATOL + _RTOL[torch.float32] * ref.abs().max().item()

    # The entry's own gate refuses the erf form in fp32.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out.float(), _erf64(x32).float())

    x64 = x32.double()
    variants = {
        "erf gelu (nn.GELU, and moe_gelu)": _erf64(x32),
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

    # A perturbation of this same formula: weak by construction (bounded by the
    # coefficient change times the tanh derivative), so a lower multiple.
    perturbed = 0.5 * x64 * (1.0 + torch.tanh(SQRT_2_OVER_PI * (x64 + 0.045 * x64**3)))
    dev = (out - perturbed).abs().max().item()
    print(f"  {'coefficient 0.044715 -> 0.045':36s} dev {dev:.6e} ({dev / gate:7.1f}x gate)")
    assert dev > 100.0 * correct, (dev, correct)
    assert dev > 5.0 * gate, (dev, gate)

    # bf16 cannot see the difference: the erf variant's max deviation is within
    # a small multiple of the correct path's own.
    xb = x32.to(torch.bfloat16)
    outb = op(xb).double()
    correct_b = (outb - _tanh64(xb)).abs().max().item()
    erf_b = (outb - _erf64(xb)).abs().max().item()
    print(f"  bf16: correct {correct_b:.6e}; erf variant {erf_b:.6e} ({erf_b / correct_b:.3f}x)")
    assert erf_b < 4.0 * correct_b, (erf_b, correct_b)
    torch.testing.assert_close(outb.to(torch.bfloat16), op.reference(xb))
