# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the dequantize_e4m3_per_tensor catalog entry.

The entry owns the contract. This file turns a cell into operands and holds
the claims no cell carries: graph capture with the scale read at replay, and
the loud rejection of an empty batch.
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
    else:
        assert source == "all_codes", source
        x = torch.arange(256, dtype=torch.uint8, device=DEV).view(shape).view(torch.float8_e4m3fn)
    assert x.is_contiguous() and x.shape == shape
    return x


def _bits(t: torch.Tensor) -> torch.Tensor:
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}
    return t.view(view[t.dtype])


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x = _input(cell.spec, zlib.crc32(cell.why.encode()))
    for value in cell.spec["scales"]:
        scales = _scale(value, x.dim(), cell.spec["dtype"])
        out = op(x, scales)
        assert out.shape == x.shape and out.dtype == scales.dtype and out.is_contiguous()
        _compare(out, op.reference(x, scales))
    if cell.spec["source"] == "gather":
        flat = out.reshape(x.shape[0], -1)
        assert flat.shape[1] == 2560 and flat.data_ptr() == out.data_ptr(), "flatten is a view"


# ── what a cell cannot carry ──────────────────────────────────────────────


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


def test_rejects_an_empty_batch():
    """The op raises on zero rows rather than launching; the caller branches first."""
    with pytest.raises((RuntimeError, NotImplementedError)):
        op(_fp8(torch.randn(0, 128, device=DEV)), _scale(PLE_SCALE, 2))
