# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the bf16_to_fp32 entry: the cells bit for bit `x.float()`, graph capture, the
contiguity guard."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.quantization.bf16_to_fp32 import (  # noqa: E501
    bf16_to_fp32 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "bf16_to_fp32 requires a CUDA device"


def _build(spec, seed):
    shape = spec["shape"]
    if spec.get("source") == "all_codes":
        return (
            torch.arange(-32768, 32768, dtype=torch.int32, device="cuda")
            .to(torch.int16)
            .view(torch.bfloat16)
            .reshape(shape)
        )
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, generator=g, device="cuda") * 8).to(torch.bfloat16)


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    x = _build(cell.spec, zlib.crc32(cell.why.encode()))
    out = op(x)
    ref = op.reference(x)
    assert out.dtype == torch.float32 and out.is_contiguous()
    assert _bitwise(out, ref), "widening is exact; a bit difference is a defect"


def test_cuda_graph_capture_and_replay() -> None:
    x = _build(dict(shape=(128, 4099)), 1)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(x)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x)
    x.copy_(_build(dict(shape=(128, 4099)), 2))
    graph.replay()
    torch.cuda.synchronize()
    assert _bitwise(out, x.float())


def test_guard_refuses_a_strided_input() -> None:
    x = _build(dict(shape=(8, 64)), 3)
    with validating(op), pytest.raises(AssertionError):
        op(x[:, ::2])
