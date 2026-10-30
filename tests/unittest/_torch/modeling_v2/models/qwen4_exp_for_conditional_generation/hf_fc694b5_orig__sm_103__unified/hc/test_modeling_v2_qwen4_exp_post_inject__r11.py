# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_post_inject entry (this target's local variant r11).

The cells are gated bitwise (16-bit) by `_compare`. Beyond them: the
leading-dim guard and graph capture.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.hc.qwen4_exp_post_inject__r11 import (  # noqa: E501
    qwen4_exp_post_inject as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_post_inject requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise for 16-bit operands: the product of two bf16 values is exact in
    fp32, so op and reference round the same fp32 sum once -- measured
    bit-exact at every token count."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert out.is_contiguous()
    assert torch.equal(out, ref), (out.float() - ref.float()).abs().max().item()


DEV = "cuda"
HC_COUNT = 4
HIDDEN = 2560


def _operands(leading, hc_count=HC_COUNT, hidden=HIDDEN, dtype=torch.bfloat16, seed=0):
    """Operands in the ranges the model produces: `inject` is 2*sigmoid(.), so [0, 2]."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    hyper_input = torch.randn(*leading, hc_count * hidden, generator=g, device=DEV).to(dtype)
    block_out = torch.randn(*leading, hidden, generator=g, device=DEV).to(dtype)
    inject = (2.0 * torch.sigmoid(torch.randn(*leading, hc_count, generator=g, device=DEV))).to(
        dtype
    )
    return hyper_input, block_out, inject


def _build(spec, seed):
    hc_count, hidden = spec["hc_count"], spec["hidden"]
    dtype = spec.get("dtype", torch.bfloat16)
    hyper_input, block_out, inject = _operands(spec["leading"], hc_count, hidden, dtype, seed)
    layout = spec.get("layout", "contiguous")
    width = hc_count * hidden
    if layout == "col_slice":
        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        buf = torch.randn(hyper_input.shape[0], 2 * width, generator=g, device=DEV).to(dtype)
        buf[:, :width] = hyper_input
        hyper_input = buf[:, :width]
        # Strides rather than is_contiguous(): at one row torch ignores the
        # size-1 dim's stride and calls the slice contiguous.
        assert hyper_input.stride() == (2 * width, 1), hyper_input.stride()
        assert hyper_input.shape[0] == 1 or not hyper_input.is_contiguous()
    return hyper_input, block_out, inject


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    hyper_input, block_out, inject = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = hyper_input.clone()
    out = op(hyper_input, block_out, inject)
    assert out.shape == hyper_input.shape and out.dtype == hyper_input.dtype
    assert out.data_ptr() != hyper_input.data_ptr(), "the result is a fresh tensor"
    assert torch.equal(hyper_input, before), "the residual it was given is read-only"
    _compare(out, op.reference(hyper_input, block_out, inject))


def test_guard_refuses_unequal_leading_dims() -> None:
    """Unguarded, a 1-D `inject` broadcasts one row over every token."""
    hyper_input, block_out, inject = _operands((8,), seed=10)
    unguarded = op(hyper_input, block_out, inject[0])
    assert unguarded.shape == hyper_input.shape, "the hazard this guard covers is gone"
    assert not torch.equal(unguarded, op.reference(hyper_input, block_out, inject))

    with pytest.raises(AssertionError, match="leading dims must match"):
        with validating(op):
            op(hyper_input, block_out, inject[0])


def test_cuda_graph_capture_and_replay() -> None:
    hyper_input, block_out, inject = _operands((64,), seed=14)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(hyper_input, block_out, inject)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(hyper_input, block_out, inject)

    fresh, _, _ = _operands((64,), seed=15)
    hyper_input.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, op.reference(hyper_input, block_out, inject))
