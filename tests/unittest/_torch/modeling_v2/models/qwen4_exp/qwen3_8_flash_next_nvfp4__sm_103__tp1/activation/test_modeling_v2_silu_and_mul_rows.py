# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the silu_and_mul_rows entry: the cells bitwise `flashinfer_silu_and_mul` on a
contiguous copy of the gate|up columns (and so inside its reference band) with the fused
output's extra columns present, graph capture, the width guard."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.activation.silu_and_mul_rows import (  # noqa: E501
    silu_and_mul_rows as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "silu_and_mul_rows requires a CUDA device"

INTER = 640


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, width = spec["rows"], 2 * INTER + 8
    x = (torch.randn(rows, width, generator=g, device="cuda") * 3.0).to(torch.bfloat16)
    if rows:
        x[0, :4] = torch.tensor([0.0, -0.0, 88.0, -100.0], device="cuda").to(torch.bfloat16)
    return x


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    x = _build(cell.spec, zlib.crc32(cell.why.encode()))
    out = op(x, INTER)
    assert out.shape == (x.shape[0], INTER) and out.is_contiguous()
    torch.testing.assert_close(out, op.reference(x, INTER))
    if x.shape[0]:
        assert torch.equal(out, flashinfer_silu_and_mul(x[:, : 2 * INTER].contiguous()))


def test_cuda_graph_capture_and_replay() -> None:
    x = _build(dict(rows=128), 8)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(x, INTER)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, INTER)
    x.copy_(_build(dict(rows=128), 9))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, op(x, INTER))


def test_guards() -> None:
    x = _build(dict(rows=4), 5)
    with validating(op), pytest.raises(AssertionError):
        op(x[:, : 2 * INTER - 2], INTER)
