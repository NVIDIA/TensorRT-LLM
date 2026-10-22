# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the dequantize_e4m3_per_tensor catalog entry.

The entry owns the contract. This file turns a cell into operands and holds
the claims no cell carries: the direction, the round trip through the
quantizer, degenerate and overflowing scales, graph capture, and the loud
rejections.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.quantization.dequantize_e4m3_per_tensor import (  # noqa: E501
    PLE_SCALE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.quantization.dequantize_e4m3_per_tensor import (  # noqa: E501
    dequantize_e4m3_per_tensor as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "dequantize_e4m3_per_tensor requires a CUDA device"


_BIT_VIEW = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
}


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bit-exact, NaN for NaN.

    Both sides form the same exact fp32 product and round it once, so any
    difference is a defect. Measured bit-exact on every case, including all
    256 codes x three output dtypes x five scales, with -0.0 and the
    subnormals compared on bit patterns. NaN payloads are not compared.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(out), nan), "NaN lanes differ"
    view = _BIT_VIEW[ref.dtype]
    assert torch.equal(out.contiguous().view(view)[~nan], ref.contiguous().view(view)[~nan]), (
        "dequantization is exact; a bit difference is a defect"
    )


DEV = "cuda"
E4M3_MAX = 448.0
# e4m3: a normal value's relative half-step and the smallest subnormal's
# absolute half-step, in units of the scale.
E4M3_REL_HALF_STEP = 2.0**-4
E4M3_SUBNORMAL_HALF_STEP = 2.0**-10

_TABLE = {}


def _fp8(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn)


def _scale(v: float, rank: int, dtype=torch.bfloat16) -> torch.Tensor:
    """The only shape the op accepts: `input`'s rank, every dimension 1."""
    return torch.full((1,) * rank, v, dtype=dtype, device=DEV)


def _input(spec, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    source, shape = spec["source"], spec["shape"]
    if source == "gather":
        if "table" not in _TABLE:
            tg = torch.Generator(device=DEV).manual_seed(11)
            _TABLE["table"] = _fp8(torch.randn(1 << 20, 160, device=DEV, generator=tg) * 40.0)
        idx = torch.randint(0, 1 << 20, shape[:2], device=DEV, generator=g)
        x = _TABLE["table"][idx]
    elif source == "all_codes":
        x = torch.arange(256, dtype=torch.uint8, device=DEV).view(shape).view(torch.float8_e4m3fn)
    elif source == "unaligned":
        n = shape[0] * shape[1]
        base = _fp8(torch.randn(n + 16, device=DEV, generator=g) * 8.0)
        x = base[1 : 1 + n].view(shape)
        assert x.data_ptr() % 16 != 0
    else:
        x = _fp8(torch.randn(*shape, device=DEV, generator=g) * 8.0)
    assert x.is_contiguous() and x.shape == shape
    return x


def _scales(spec, value: float, rank: int) -> torch.Tensor:
    if not spec.get("scale_slice"):
        return _scale(value, rank, spec["dtype"])
    buf = torch.tensor([[9.0, value, 5.0]], dtype=spec["dtype"], device=DEV)
    sliced = buf[:, 1:2]
    assert sliced.shape == (1, 1) and sliced.storage_offset() == 1
    return sliced


def _bits(t: torch.Tensor) -> torch.Tensor:
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}
    return t.view(view[t.dtype])


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x = _input(cell.spec, zlib.crc32(cell.why.encode()))
    for value in cell.spec["scales"]:
        scales = _scales(cell.spec, value, x.dim())
        out = op(x, scales)
        assert out.shape == x.shape and out.dtype == scales.dtype and out.is_contiguous()
        _compare(out, op.reference(x, scales))
    if cell.spec["source"] == "gather":
        flat = out.reshape(x.shape[0], -1)
        assert flat.shape[1] == 2560 and flat.data_ptr() == out.data_ptr(), "flatten is a view"


# ── what a cell cannot carry ──────────────────────────────────────────────


def test_multiplies_rather_than_divides():
    x = _fp8(torch.tensor([[1.0, 2.0, -3.0, 0.5]], device=DEV))
    for value in (0.5, PLE_SCALE):
        out = op(x, _scale(value, 2, torch.float32))
        assert torch.equal(_bits(out), _bits(x.float() * value))
        assert not torch.allclose(out, x.float() / value)


def test_top_of_the_range_at_the_checkpoint_scale():
    top = op(_fp8(torch.tensor([[E4M3_MAX, -E4M3_MAX]], device=DEV)), _scale(PLE_SCALE, 2))
    # 448 * PLE_SCALE = 0.08929443359375 exactly; bf16's nearest is this
    assert top.float().flatten().tolist() == [0.08935546875, -0.08935546875]


def test_round_trip_through_the_static_quantizer():
    """The error floor is e4m3 itself; a reciprocated scale lands >100x out."""
    g = torch.Generator(device=DEV).manual_seed(15)
    x = (torch.randn(256, 5120, device=DEV, generator=g) * 2.0).to(torch.bfloat16)
    value = 0.05
    scales = _scale(value, 2, torch.float32)
    q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scales)
    back = op(q, scales)
    _compare(back, op.reference(q, scales))
    bound = E4M3_REL_HALF_STEP * x.float().abs() + E4M3_SUBNORMAL_HALF_STEP * value
    assert bool(((back - x.float()).abs() <= bound).all())
    wrong = op(q, _scale(1.0 / value, 2, torch.float32))
    assert float(((wrong - x.float()).abs() / bound)[x.float().abs() > 0.1].min()) > 100.0


def test_degenerate_scale_values_are_accepted_and_silent():
    y = _fp8(torch.tensor([[1.0, -1.0, 0.0, E4M3_MAX]], device=DEV))
    out = op(y, _scale(0.0, 2))
    assert out.float().flatten().tolist() == [0.0, -0.0, 0.0, 0.0]
    assert not bool(torch.isnan(out.float()).any())
    assert op(y, _scale(-2.0, 2)).float().flatten().tolist() == [-2.0, 2.0, -0.0, -896.0]
    out = op(y, _scale(float("inf"), 2)).float().flatten()
    assert out[0].item() == float("inf") and out[1].item() == float("-inf")
    assert torch.isnan(out[2])
    assert bool(torch.isnan(op(y, _scale(float("nan"), 2)).float()).all())


def test_overflow_goes_to_infinity_not_to_a_clamp():
    y = _fp8(torch.tensor([[1.0, E4M3_MAX]], device=DEV))
    out = op(y, _scale(1000.0, 2, torch.float16)).float().flatten()
    assert out[0].item() == 1000.0 and out[1].item() == float("inf")
    out = op(y, _scale(3.0e38, 2, torch.bfloat16)).float().flatten()
    assert torch.isfinite(out[0]) and out[1].item() == float("inf")


def test_cuda_graph_capture_and_replay():
    """Both operands stay live inputs: the scale is read at replay."""
    g = torch.Generator(device=DEV).manual_seed(16)
    static_in = _fp8(torch.randn(256, 16, 160, device=DEV, generator=g) * 8.0)
    static_scales = _scale(PLE_SCALE, 3)
    eager = op(static_in, static_scales).clone()
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            op(static_in, static_scales)
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = op(static_in, static_scales)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(static_out), _bits(eager))
    fresh = _fp8(torch.randn(256, 16, 160, device=DEV, generator=g) * 8.0)
    static_in.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(static_out), _bits(op(fresh, static_scales)))
    static_scales.copy_(_scale(0.5, 3))
    graph.replay()
    torch.cuda.synchronize()
    _compare(static_out, op.reference(fresh, _scale(0.5, 3)))


def test_deterministic_and_operands_untouched():
    g = torch.Generator(device=DEV).manual_seed(17)
    x = _fp8(torch.randn(64, 16, 160, device=DEV, generator=g) * 8.0)
    scales = _scale(PLE_SCALE, 3)
    x_before, s_before = x.clone(), scales.clone()
    out1, out2 = op(x, scales), op(x, scales)
    assert torch.equal(_bits(out1), _bits(out2))
    assert torch.equal(x.view(torch.uint8), x_before.view(torch.uint8))
    assert torch.equal(_bits(scales), _bits(s_before))
    assert out1.data_ptr() not in (x.data_ptr(), scales.data_ptr())


def test_rejects_unsupported():
    """Every metadata violation is loud, which is why the entry has no guard."""
    x = _fp8(torch.randn(4, 128, device=DEV))
    scales = _scale(PLE_SCALE, 2)
    bf = dict(dtype=torch.bfloat16, device=DEV)
    cases = {
        "1-D input": lambda: op(x.reshape(-1), _scale(0.5, 1)),
        "0-D input": lambda: op(_fp8(torch.tensor(1.0, device=DEV)), _scale(0.5, 0)),
        "zero rows": lambda: op(_fp8(torch.randn(0, 128, device=DEV)), scales),
        "zero cols": lambda: op(_fp8(torch.randn(4, 0, device=DEV)), scales),
        "column slice": lambda: op(_fp8(torch.randn(8, 256, device=DEV))[:, :128], scales),
        "transpose": lambda: op(_fp8(torch.randn(128, 8, device=DEV)).t(), scales),
        "bfloat16 input": lambda: op(torch.randn(4, 128, **bf), scales),
        "float8_e5m2 input": lambda: op(
            torch.randn(4, 128, device=DEV).to(torch.float8_e5m2), scales
        ),
        "int8 input": lambda: op(
            torch.randint(-8, 8, (4, 128), dtype=torch.int8, device=DEV), scales
        ),
        "float64 scales": lambda: op(x, _scale(0.5, 2, torch.float64)),
        "cpu scales": lambda: op(x, scales.cpu()),
        "cpu input": lambda: op(x.cpu(), scales),
        "cpu input and scales": lambda: op(x.cpu(), scales.cpu()),
        "scales rank below": lambda: op(x, _scale(0.5, 1)),
        "scales rank above": lambda: op(x, _scale(0.5, 3)),
        "0-D scales": lambda: op(x, _scale(0.5, 0)),
        "scales dim not 1": lambda: op(x, torch.full((1, 2), 0.5, **bf)),
        "scales leading dim not 1": lambda: op(x, torch.full((4, 1), 0.5, **bf)),
        "empty scales": lambda: op(x, torch.empty((1, 0), **bf)),
    }
    for name, fn in cases.items():
        with pytest.raises((RuntimeError, NotImplementedError)):
            fn()
