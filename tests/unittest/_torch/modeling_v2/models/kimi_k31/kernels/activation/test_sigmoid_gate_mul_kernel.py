# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's output-gate kernel (CuTe DSL): bit-identical to ``x * g.sigmoid()`` in
PyTorch (the op's reference) on every gate value and launch variant."""

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.activation.sigmoid_gate_mul as kernel
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.activation.sigmoid_gate_mul import (  # noqa: E501
    k31_sigmoid_gate_mul as op,
)

assert torch.cuda.is_available(), "the output-gate kernel requires a CUDA device"


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _every_bf16() -> torch.Tensor:
    """All 65536 BF16 bit patterns (NaNs, infinities, denormals and signed zeros included)."""
    return (
        torch.arange(-32768, 32768, dtype=torch.int32, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )


@pytest.mark.parametrize(
    "rows,cols,extra",
    ((5, 1536, 2112), (8, 384, 2272), (300, 12288, 64), (70000, 8, 8), (9, 100, 3)),
)
def test_column_slice_gate(rows: int, cols: int, extra: int) -> None:
    """The gate as a column slice of a wider row-major output takes the strided launch."""
    g = torch.Generator(device="cuda").manual_seed(rows + cols)
    x = torch.randn(rows, cols, generator=g, device="cuda").to(torch.bfloat16)
    wide = (torch.randn(rows, cols + extra, generator=g, device="cuda") * 3).to(torch.bfloat16)
    gate = wide[:, extra:]
    assert not gate.is_contiguous()
    out = kernel.k31_sigmoid_gate_mul(x, gate)
    assert torch.equal(_bits(out), _bits(x * gate.sigmoid()))


@pytest.mark.parametrize("layout", ("vector", "element", "slice"))
def test_every_gate_value(layout: str) -> None:
    """The gate is a function of one BF16 value and the product is exact before its rounding, so
    every gate value against a few multiplicands covers the map: all 65536 patterns, the scaled
    reciprocal range (g in about [-88.7, -87.3]), denormal sigmoids, infinities and NaNs."""
    values = _every_bf16()
    for xv in (1.0, -1.5, 3.0e38, 1.0e-38):
        if layout == "element":
            # 2-byte aligned views: one element per step.
            xbuf = torch.full((65537,), xv, device="cuda", dtype=torch.bfloat16)
            gbuf = torch.empty(65537, device="cuda", dtype=torch.bfloat16)
            gbuf[1:] = values
            x, gate = xbuf[1:].view(512, 128), gbuf[1:].view(512, 128)
        elif layout == "slice":
            x = torch.full((512, 128), xv, device="cuda", dtype=torch.bfloat16)
            wide = torch.zeros(512, 168, device="cuda", dtype=torch.bfloat16)
            wide[:, 40:] = values.view(512, 128)
            gate = wide[:, 40:]
        else:
            x = torch.full((512, 128), xv, device="cuda", dtype=torch.bfloat16)
            gate = values.view(512, 128)
        out = kernel.k31_sigmoid_gate_mul(x, gate)
        ref = x * gate.sigmoid()
        nan = torch.isnan(ref)
        assert torch.equal(torch.isnan(out), nan)
        assert torch.equal(_bits(out)[~nan], _bits(ref)[~nan])


def test_extremes_and_denormals() -> None:
    """Saturated gates, the denormal-sigmoid range (no flush to zero), signed zeros, infinities."""
    vals = torch.tensor(
        [
            -110.0,
            -103.0,
            -95.0,
            -90.0,
            -88.5,
            -88.0,
            -87.5,
            -20.0,
            -0.0,
            0.0,
            1e-3,
            20.0,
            90.0,
            float("inf"),
            float("-inf"),
        ],
        device="cuda",
    )
    gate = vals.repeat(64).reshape(8, -1).to(torch.bfloat16)
    x = torch.full_like(gate, 3.0e38).to(torch.bfloat16)
    x[::2] = -1.5
    out = kernel.k31_sigmoid_gate_mul(x, gate)
    assert torch.equal(_bits(out), _bits(x * gate.sigmoid()))


@pytest.mark.parametrize("rows,cols", ((8, 768), (256, 12288)))
@pytest.mark.parametrize("use_pdl", (False, True))
@pytest.mark.parametrize("trigger", (False, True))
def test_launch_variants_agree(rows, cols, use_pdl, trigger) -> None:
    """Programmatic dependent launch and its early trigger change the schedule only."""
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(rows, cols, generator=g, device="cuda").to(torch.bfloat16)
    gate = (torch.randn(rows, cols, generator=g, device="cuda") * 4).to(torch.bfloat16)
    out = torch.empty_like(x)
    kernel.sigmoid_gate_mul_into(x, gate, out, use_pdl=use_pdl, trigger=trigger)
    assert torch.equal(_bits(out), _bits(op.reference(x, gate)))


def test_cuda_graph_replay() -> None:
    """Captured after a warm-up call, a chain of gates replays to the eager bits."""
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(72, 768, generator=g, device="cuda").to(torch.bfloat16)
    gate = (torch.randn(72, 768, generator=g, device="cuda") * 4).to(torch.bfloat16)
    expected = op.reference(op.reference(x, gate), gate)
    kernel.k31_sigmoid_gate_mul(x, gate)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = kernel.k31_sigmoid_gate_mul(kernel.k31_sigmoid_gate_mul(x, gate), gate)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(expected))


def test_empty() -> None:
    x = torch.empty(0, 768, device="cuda", dtype=torch.bfloat16)
    assert kernel.k31_sigmoid_gate_mul(x, x.clone()).shape == (0, 768)


@pytest.mark.parametrize("bad", ("fp16", "shape", "strided_x", "strided_gate_cols"))
def test_rejects(bad: str) -> None:
    x = torch.randn(4, 64, device="cuda").to(torch.bfloat16)
    gate = torch.randn(4, 64, device="cuda").to(torch.bfloat16)
    if bad == "fp16":
        x = x.half()
    elif bad == "shape":
        gate = gate[:, :32].contiguous()
    elif bad == "strided_x":
        x = torch.randn(4, 128, device="cuda").to(torch.bfloat16)[:, ::2]
    elif bad == "strided_gate_cols":
        gate = torch.randn(4, 128, device="cuda").to(torch.bfloat16)[:, ::2]
    with pytest.raises(ValueError):
        kernel.k31_sigmoid_gate_mul(x, gate)
