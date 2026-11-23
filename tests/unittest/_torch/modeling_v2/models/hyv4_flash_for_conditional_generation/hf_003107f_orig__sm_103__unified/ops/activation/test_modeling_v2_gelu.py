# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's gelu entry.

The entry's identity is *which* GELU it is, and the wrong answer -- the tanh
approximation this family's other block MLPs use -- rounds to within about one
bf16 ulp of it. The discrimination test is the point of this file; only fp32
tells the two apart. The cells certify the tower's two widths and row envelope,
two of them on the distribution the checkpoint actually builds.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.activation.gelu import (  # noqa: E501
    gelu as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "gelu requires a CUDA device"

MLP_WIDTH = 4304
MERGER_WIDTH = 4608
TOWER_HIDDEN = 1152
SPATIAL_MERGE = 4  # 2 x 2

#: Measured from the checkpoint's bf16 tensors (first 128 rows of each weight,
#: full bias and gamma vectors): `vit.layers.0.mlp.dense_h_to_4h.weight` std,
#: `.bias` mean/std, `vit.layers.0.post_attention_layernorm.weight` mean/std,
#: `vit.perceive.proj.0.weight` std, `.bias` mean/std, and
#: `vit.perceive.norm.weight` mean/std. Layer 0 is the widest-spread block; the
#: later blocks' biases sit between -1.0 and -1.8 with smaller spreads.
FC1_WEIGHT_STD = 0.018955
FC1_BIAS_MEAN = -2.416808
FC1_BIAS_STD = 1.668815
LN_GAMMA_MEAN = 1.511802
LN_GAMMA_STD = 2.353820
PROJ_WEIGHT_STD = 0.008602
PROJ_BIAS_MEAN = -0.000401
PROJ_BIAS_STD = 0.003034
PERCEIVE_GAMMA_MEAN = 1.084490
PERCEIVE_GAMMA_STD = 0.029593

SQRT_2 = math.sqrt(2.0)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

#: torch.testing's untouched per-dtype defaults, restated only so the
#: discrimination test can say how far outside that gate each variant lands.
_RTOL = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}
_ATOL = 1e-5

#: 2 ulp of the row's largest element and 0.1 ulp of relative RMS. The op
#: computes in fp32 and rounds once to bf16; the reference computes in fp64 and
#: rounds once to bf16, so a difference is a bf16 rounding tie that flipped --
#: one step of the element's own scale, read against the row's, with the second
#: ulp covering the comparison's own division. Wrong GELUs (tanh, sigmoid,
#: relu) land far outside it in fp32; in bf16 the identity test is what
#: separates them.
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.1


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _uniform(shape, lo, hi, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty(shape, device="cuda", dtype=torch.float32).uniform_(lo, hi, generator=g)
    return x.to(dtype)


def _tower(rows, dtype, seed):
    """The block MLP's fc1 output: a LayerNorm'd hidden at the checkpoint's
    gamma scale through a synthetic `dense_h_to_4h` at its measured stats."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    gamma = LN_GAMMA_MEAN + LN_GAMMA_STD * torch.randn(TOWER_HIDDEN, generator=g, device="cuda")
    hidden = torch.randn(rows, TOWER_HIDDEN, generator=g, device="cuda")
    weight = FC1_WEIGHT_STD * torch.randn(MLP_WIDTH, TOWER_HIDDEN, generator=g, device="cuda")
    bias = FC1_BIAS_MEAN + FC1_BIAS_STD * torch.randn(MLP_WIDTH, generator=g, device="cuda")
    return ((hidden * gamma) @ weight.t() + bias).to(dtype)


def _merger(rows, dtype, seed):
    """The merger's `proj.0` output: `perceive.norm`'s gamma tiled over the 2x2
    window through a synthetic 4608-wide projection at its measured stats."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    gamma = PERCEIVE_GAMMA_MEAN + PERCEIVE_GAMMA_STD * torch.randn(
        TOWER_HIDDEN, generator=g, device="cuda"
    )
    hidden = torch.randn(rows, SPATIAL_MERGE, TOWER_HIDDEN, generator=g, device="cuda")
    hidden = (hidden * gamma).reshape(rows, MERGER_WIDTH)
    weight = PROJ_WEIGHT_STD * torch.randn(MERGER_WIDTH, MERGER_WIDTH, generator=g, device="cuda")
    bias = PROJ_BIAS_MEAN + PROJ_BIAS_STD * torch.randn(MERGER_WIDTH, generator=g, device="cuda")
    return (hidden @ weight.t() + bias).to(dtype)


def _build(spec, seed):
    shape, dtype = spec["shape"], spec["dtype"]
    dist = spec.get("dist", "uniform")
    if dist == "tower":
        return _tower(shape[0], dtype, seed)
    if dist == "merger":
        return _merger(shape[0], dtype, seed)
    return _uniform(shape, spec["lo"], spec["hi"], dtype, seed)


def _cosine(out: torch.Tensor, ref: torch.Tensor) -> float:
    a, b = out.float().reshape(-1), ref.float().reshape(-1)
    return (a @ b / (a.norm() * b.norm())).item()


def _measured(out: torch.Tensor, ref: torch.Tensor) -> str:
    """The numbers quoted in the entry's note, per cell."""
    diff = (out.float() - ref.float()).abs()
    equal = (out == ref).float().mean().item()
    rel_l2 = (diff.norm() / ref.float().norm()).item()
    row_scale = (
        ref.float().reshape(-1, ref.shape[-1]).abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    )
    worst = (diff.reshape(-1, ref.shape[-1]) / row_scale).max().item() / (2.0**-8)
    return (
        f"bitwise equal {equal * 100:.3f}%, rel L2 {rel_l2:.2e} "
        f"({rel_l2 / 2.0**-8:.3f} ulp), worst {worst:.3f} ulp of the row scale"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x = _build(cell.spec, seed=_seed(cell.why))
    before = x.clone()
    out = op(x)
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    assert out.data_ptr() != x.data_ptr(), "the entry is out-of-place"
    assert torch.equal(before, x), "the entry must not mutate its input"
    ref = op.reference(x)
    # Distribution first: a cell whose distribution matches but whose worst
    # element sits past a strict allclose is certified; a diverged one is not.
    cos = _cosine(out, ref)
    assert cos > 0.999999, f"cosine {cos:.9f}: the output diverged from the reference"
    print(f"{cell.why}: {_measured(out, ref)}")
    assert_within_ulp(out, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def _erf64(x):
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.erf(x64 / SQRT_2))


def _tanh64(x):
    """The *other* GELU -- `gelu_pytorch_tanh`, the tanh approximation."""
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.tanh(SQRT_2_OVER_PI * (x64 + 0.044715 * x64 * x64 * x64)))


def test_identity_is_the_erf_form_and_not_tanh() -> None:
    """Measured in fp32, where the question has an answer; and shown to have
    none in bf16, so a bf16 match cannot quietly be read as confirmation."""
    x32 = _uniform((512, MLP_WIDTH), -4.0, 4.0, torch.float32, 24)
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
    torch.testing.assert_close(op(xb), op.reference(xb))


def test_four_d_operand_matches_the_two_d_view() -> None:
    """The reference tower applies the activation to the merger's
    `[B, 4608, H, W]` conv output; this target's call site is the 2-D row form.
    Measured, not a cell: the elementwise kernel reads the same values either
    way, and the output keeps the operand's shape."""
    x = _uniform((256, MERGER_WIDTH), -16.0, 16.0, torch.bfloat16, 41)
    four_d = x.view(4, MERGER_WIDTH, 8, 8)
    assert four_d.is_contiguous()
    out_4d = op(four_d)
    assert out_4d.shape == four_d.shape and out_4d.dtype == four_d.dtype
    torch.testing.assert_close(out_4d.view(256, MERGER_WIDTH), op(x), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1024, MLP_WIDTH), (4096, MERGER_WIDTH)])
def test_cuda_graph_capture_and_replay(shape) -> None:
    spec = dict(shape=shape, dtype=torch.bfloat16, lo=-16.0, hi=16.0)
    x = _build(spec, seed=12)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(x)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x)

    x.copy_(_build(spec, seed=13))
    graph.replay()
    torch.cuda.synchronize()
    assert_within_ulp(captured, op.reference(x), element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)
