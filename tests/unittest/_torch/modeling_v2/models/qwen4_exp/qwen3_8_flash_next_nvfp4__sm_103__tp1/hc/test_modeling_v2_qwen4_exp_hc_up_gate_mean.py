# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_up_gate_mean entry: cells in the band
`qwen4_exp_gated_residual`'s test derives for `mixed`, the guard, graph
capture."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_gated_residual import (  # noqa: E501
    gate_ratio,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_up_gate_mean import (  # noqa: E501
    qwen4_exp_hc_up_gate_mean as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_up_gate_mean requires a CUDA device"

DEV = "cuda"


def _compare(out, ref) -> None:
    """`rtol = eps`, `atol = eps * max|ref|` (`gate_ratio <= 1`), the band
    `qwen4_exp_gated_residual` derives for `mixed`."""
    assert out.dtype == ref.dtype and out.shape == ref.shape and out.is_contiguous()
    assert gate_ratio(out, ref) <= 1.0, f"{gate_ratio(out, ref):.4g}x the band"


def _build(spec, seed):
    hc, hidden, rank = spec.get("hc_count", 4), spec.get("hidden", 2560), spec.get("rank", 320)
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows, width = spec["rows"], hc * hidden
    # This checkpoint's statistics: low = silu(.) of the bottleneck (|x| < ~3), the up
    # weight ~N(0, 0.068) (gate logits within about +-8), the normed stream ~+-30.
    low = torch.nn.functional.silu(torch.randn(rows, rank, generator=g, device=DEV) * 1.5).to(
        torch.bfloat16
    )
    up_w = (torch.randn(width, rank, generator=g, device=DEV) * 0.068).to(torch.bfloat16)
    normed = (torch.randn(rows, width, generator=g, device=DEV) * 6).to(torch.bfloat16)
    return low, up_w.t(), normed, hc


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    mixed = op(*args)
    assert mixed.shape == (cell.spec["rows"], cell.spec.get("hidden", 2560))
    _compare(mixed, op.reference(*args))


def test_guard_refuses_a_contiguous_up() -> None:
    low, up, normed, c = _build(dict(rows=4), seed=4)
    with validating(op), pytest.raises(AssertionError):
        op(low, up.contiguous(), normed, c)


def test_cuda_graph_capture_and_replay() -> None:
    low, up, normed, c = _build(dict(rows=8), seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(low, up, normed, c)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_mixed = op(low, up, normed, c)
    fresh = _build(dict(rows=8), seed=6)
    low.copy_(fresh[0])
    normed.copy_(fresh[2])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(g_mixed, op(fresh[0], up, fresh[2], c))
