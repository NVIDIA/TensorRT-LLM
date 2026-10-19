# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the static_quantize_e4m3_per_tensor entry.

The entry owns the contract. This file turns a cell into operands and holds
the claims no cell carries: that the two kernels behind the op really do round
apart (so `reference` choosing between them is load-bearing), the direction of
the scale, saturation, the scale being read at graph replay, the guard, and the
loud rejections the note relies on.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.static_quantize_e4m3_per_tensor import (  # noqa: E501
    E4M3_MAX,
    OUT_PROJ_SCALES,
    VEC_ALIGN_BYTES,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.static_quantize_e4m3_per_tensor import (  # noqa: E501
    static_quantize_e4m3_per_tensor as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "static_quantize_e4m3_per_tensor requires a CUDA device"

DEV = "cuda"


def _codes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bit-exact on the e4m3 codes.

    Kernel and reference perform the same single fp32 operation on the same
    exact bf16 value and round it once into e4m3, so any differing code is a
    defect. Measured: zero differing codes on every cell, including all 65,282
    non-NaN bf16 values at four scales and both kernels.
    """
    assert out.dtype == ref.dtype == torch.float8_e4m3fn, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    wrong = _codes(out) != _codes(ref)
    assert not wrong.any(), f"{wrong.sum().item()} of {wrong.numel()} e4m3 codes differ"


def _all_bf16() -> torch.Tensor:
    """Every bf16 bit pattern except the NaNs, as one row of 65,282."""
    x = torch.arange(65536, dtype=torch.int32, device=DEV).to(torch.int16).view(torch.bfloat16)
    return x[~torch.isnan(x)].reshape(1, -1).contiguous()


def _mixed(shape, scale: float, g: torch.Generator) -> torch.Tensor:
    """Half gaussian at 224 * scale (about 5% saturate), half log-uniform.

    The log-uniform half puts |x / scale| on every binade from 2^-12 to 2^10:
    below the smallest e4m3 subnormal, through the subnormals and every normal
    exponent, and past 448 into saturation.
    """
    n = torch.randn(shape, generator=g, device=DEV)
    lu = torch.exp2(torch.rand(shape, generator=g, device=DEV) * 22 - 12) * torch.sign(n)
    pick = torch.rand(shape, generator=g, device=DEV) < 0.5
    return (torch.where(pick, n * (E4M3_MAX / 2), lu) * scale).to(torch.bfloat16)


def _relu2(shape, scale: float, g: torch.Generator) -> torch.Tensor:
    """relu(z)^2: half exact zeros, a heavy right tail reaching saturation."""
    z = torch.randn(shape, generator=g, device=DEV)
    return (torch.relu(z).square() * (E4M3_MAX * scale / 16.0)).to(torch.bfloat16)


def _misaligned_copy(x: torch.Tensor) -> torch.Tensor:
    buf = torch.empty(x.numel() + 8, dtype=x.dtype, device=DEV)
    view = buf[1 : 1 + x.numel()].view(x.shape)
    view.copy_(x)
    assert view.is_contiguous() and view.data_ptr() % VEC_ALIGN_BYTES != 0
    return view


def _build(spec, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    if spec["source"] == "all_bf16":
        return _all_bf16(), spec["scales"]
    shape, scale = spec["shape"], spec["scale"]
    x = _relu2(shape, scale, g) if spec["source"] == "relu2" else _mixed(shape, scale, g)
    if spec["misaligned"]:
        x = _misaligned_copy(x)
    return x, (scale,)


def _scale(value: float, shape=(1,)) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32, device=DEV)


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, values = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    for value in values:
        # The checkpoint stores input_scale as [1]; a loader that squeezes it
        # passes 0-d. Both must give the same codes.
        for shape in ((1,), ()):
            s = _scale(value, shape)
            out = op(x, s)
            assert out.shape == x.shape and out.is_contiguous()
            _compare(out, op.reference(x, s))


# ── what a cell cannot carry ──────────────────────────────────────────────


def test_the_two_kernels_round_apart_and_the_reference_follows_each():
    """Why `reference` branches on dtype and address.

    The same values, once at an aligned address and once at a misaligned one,
    come back with different codes: the vectorized kernel multiplies by a
    rounded reciprocal, the scalar kernel divides. Measured at out_proj's
    largest scale on [8192, 4096]: 262519 codes (0.78%) differ, every one by
    exactly one step. A reference that computed either formula for both paths
    would fail one of the two cells that cover them.
    """
    g = torch.Generator(device=DEV).manual_seed(7)
    x = _mixed((8192, 4096), OUT_PROJ_SCALES[1], g)
    s = _scale(OUT_PROJ_SCALES[1])
    vec, scalar = op(x, s), op(_misaligned_copy(x), s)
    differ = _codes(vec) != _codes(scalar)
    assert differ.float().mean().item() > 0.005, differ.float().mean().item()
    step = (_codes(vec)[differ].int() - _codes(scalar)[differ].int()).abs()
    assert torch.all(step == 1), "the two kernels differ by more than one e4m3 step"
    multiply = (x.float() * (1.0 / s[0])).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    divide = (x.float() / s[0]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    assert torch.equal(_codes(vec), _codes(multiply))
    assert torch.equal(_codes(scalar), _codes(divide))


def test_divides_by_the_scale_and_saturates():
    """`input_scale` goes in as stored: x = 448 * s quantizes to 448, x = -s to -1.

    Beyond +-448 -- including infinity -- the op saturates. torch's own cast
    would return NaN there, which is why the reference clamps first.
    """
    s = 0.0166015625
    x = torch.tensor(
        [[448 * s, -s, 0.5 * s, 500 * s, -1e6 * s, float("inf"), float("-inf"), -0.0]],
        device=DEV,
    ).to(torch.bfloat16)
    out = op(x, _scale(s)).float().flatten()
    assert out[:3].tolist() == [448.0, -1.0, 0.5]
    assert out[3:7].tolist() == [448.0, -448.0, 448.0, -448.0]
    assert out[7].item() == 0.0 and torch.signbit(out[7]).item(), "-0.0 must stay -0.0"


def test_a_graph_replay_reads_the_scale_and_the_input_it_is_given():
    """The scale is a device pointer, not a captured constant."""
    g = torch.Generator(device=DEV).manual_seed(3)
    x = _mixed((32, 4096), OUT_PROJ_SCALES[0], g)
    s = _scale(OUT_PROJ_SCALES[0])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x, s)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, s)
    for value in (OUT_PROJ_SCALES[1], 0.25, OUT_PROJ_SCALES[0]):
        x.copy_(_mixed((32, 4096), value, g))
        s.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        _compare(captured, op.reference(x, s))


def test_guard_refuses_a_scale_vector():
    """Accepted by the op, which applies scales[0] to every element."""
    g = torch.Generator(device=DEV).manual_seed(5)
    x = _mixed((4, 4096), 0.05, g)
    vec = torch.linspace(0.01, 1.0, 4, device=DEV)
    assert torch.equal(_codes(op(x, vec)), _codes(op(x, vec[:1]))), (
        "the op stopped ignoring scales[1:]"
    )
    with pytest.raises(AssertionError, match="exactly one value"):
        with validating(op):
            op(x, vec)


@pytest.mark.parametrize(
    "case",
    ("bf16 scale", "strided input", "1-D input", "empty input"),
)
def test_loud_rejections(case):
    """The preconditions the note calls loud, raised by the op itself."""
    x = torch.ones(4, 64, dtype=torch.bfloat16, device=DEV)
    s = _scale(0.05)
    if case == "bf16 scale":
        s = s.to(torch.bfloat16)
    elif case == "strided input":
        x = torch.ones(4, 128, dtype=torch.bfloat16, device=DEV)[:, :64]
    elif case == "1-D input":
        x = x.reshape(-1)
    else:
        x = x[:0]
    with pytest.raises(RuntimeError):
        op(x, s)
