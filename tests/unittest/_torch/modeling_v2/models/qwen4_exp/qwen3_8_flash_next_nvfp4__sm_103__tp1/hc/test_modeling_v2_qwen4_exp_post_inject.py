# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_post_inject entry.

The cells are gated bitwise (16-bit) by `_compare`. Beyond them:
the separation from the plausible wrong operators, the bitwise identity and
one-hot claims, the leading-dim guard, and graph capture.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_post_inject import (  # noqa: E501
    qwen4_exp_post_inject as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_post_inject requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise for 16-bit operands: the product of two bf16 (or fp16)
    values is exact in fp32, so op and reference round the same fp32 sum
    once -- measured bit-exact at every token count. fp32 operands round the
    product, and whether it is fused into the add is the kernel's choice:
    torch's default band there."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert out.is_contiguous()
    if ref.dtype in (torch.bfloat16, torch.float16):
        assert torch.equal(out, ref), (out.float() - ref.float()).abs().max().item()
    else:
        torch.testing.assert_close(out, ref)


DEV = "cuda"
HC_COUNT = 4
HIDDEN = 2560
WIDTH = HC_COUNT * HIDDEN


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
    if layout == "row_slice":
        buf = torch.randn(2 * hyper_input.shape[0], width, device=DEV).to(dtype)
        buf[::2] = hyper_input
        hyper_input = buf[::2]
    elif layout == "col_slice":
        buf = torch.randn(hyper_input.shape[0], 2 * width, device=DEV).to(dtype)
        buf[:, :width] = hyper_input
        hyper_input = buf[:, :width]
    if layout != "contiguous":
        assert not hyper_input.is_contiguous() and hyper_input.stride(-1) == 1
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


def test_rounds_once_in_fp32_not_twice_in_bf16() -> None:
    """The bitwise gate is not the bf16-multiply-then-bf16-add form (0.0625 apart)."""
    for num_tokens in (64, 2048):
        hyper_input, block_out, inject = _operands((num_tokens,), seed=100 + num_tokens)
        out = op(hyper_input, block_out, inject)
        double_round = (
            hyper_input.unflatten(-1, (HC_COUNT, HIDDEN))
            + block_out.unsqueeze(-2) * inject.unsqueeze(-1)
        ).flatten(-2)
        assert torch.equal(out, op.reference(hyper_input, block_out, inject))
        assert (out.float() - double_round.float()).abs().max().item() > 0.0, (
            "the two rounding orders did not separate"
        )


def test_wrong_operators_are_separated() -> None:
    """Stream-minor packing, no gate, gate on the residual: each > 1.0 away."""
    hyper_input, block_out, inject = _operands((64,), seed=3)
    out = op(hyper_input, block_out, inject).float()
    hi32, bo32, in32 = hyper_input.float(), block_out.float(), inject.float()
    wrong = {
        "stream_minor": (
            hi32.unflatten(-1, (HIDDEN, HC_COUNT)) + bo32.unsqueeze(-1) * in32.unsqueeze(-2)
        ).flatten(-2),
        "no_gate": (hi32.unflatten(-1, (HC_COUNT, HIDDEN)) + bo32.unsqueeze(-2)).flatten(-2),
        "gate_on_residual": (
            hi32.unflatten(-1, (HC_COUNT, HIDDEN)) * in32.unsqueeze(-1) + bo32.unsqueeze(-2)
        ).flatten(-2),
    }
    for name, variant in wrong.items():
        dev = (out - variant).abs().max().item()
        print(f"  {name:<18} max deviation {dev:.6g}")
        assert dev > 1.0, (name, dev)


def test_zero_gate_is_the_identity() -> None:
    hyper_input, block_out, _ = _operands((33,), seed=4)
    inject = torch.zeros(33, HC_COUNT, dtype=torch.bfloat16, device=DEV)
    assert torch.equal(op(hyper_input, block_out, inject), hyper_input)


def test_one_hot_gate_touches_exactly_one_stream() -> None:
    """The property a token-broadcast bug destroys; asserted bitwise."""
    hyper_input, block_out, _ = _operands((16,), seed=5)
    streams = hyper_input.unflatten(-1, (HC_COUNT, HIDDEN))
    for live in range(HC_COUNT):
        inject = torch.zeros(16, HC_COUNT, dtype=torch.bfloat16, device=DEV)
        inject[:, live] = 1.0
        out = op(hyper_input, block_out, inject).unflatten(-1, (HC_COUNT, HIDDEN))
        for s in range(HC_COUNT):
            if s == live:
                expected = (streams[:, s].float() + block_out.float()).to(torch.bfloat16)
                assert torch.equal(out[:, s], expected), (live, s)
            else:
                assert torch.equal(out[:, s], streams[:, s]), (live, s)


def test_guard_refuses_unequal_leading_dims() -> None:
    """Unguarded, a 1-D `inject` or `block_out` broadcasts one row over every token."""
    hyper_input, block_out, inject = _operands((8,), seed=10)
    unguarded = op(hyper_input, block_out, inject[0])
    assert unguarded.shape == hyper_input.shape, "the hazard this guard covers is gone"
    assert not torch.equal(unguarded, op.reference(hyper_input, block_out, inject))
    unguarded = op(hyper_input, block_out[0], inject)
    assert unguarded.shape == hyper_input.shape, "the hazard this guard covers is gone"

    for args in (
        (hyper_input, block_out, inject[0]),
        (hyper_input, block_out[0], inject),
        (hyper_input, block_out[:4], inject),
        (hyper_input, block_out, inject[:4]),
    ):
        with pytest.raises(AssertionError, match="leading dims must match"):
            with validating(op):
                op(*args)


def test_loud_refusals_are_not_guarded() -> None:
    """A width that does not factor, and a host operand, raise on their own."""
    _, block_out, inject = _operands((8,), seed=11)
    wrong = torch.randn(8, WIDTH + 16, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="don't multiply up to the size of dim"):
        op(wrong, block_out, inject)
    hyper_input, _, inject = _operands((8,), seed=12)
    with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
        op(hyper_input, torch.randn(8, HIDDEN, dtype=torch.bfloat16), inject)


def test_mixed_dtypes_promote_the_output() -> None:
    """Recorded, not guarded: an fp32 operand silently widens the residual stream."""
    hyper_input, block_out, _ = _operands((8,), seed=13)
    fp32_inject = torch.rand(8, HC_COUNT, dtype=torch.float32, device=DEV) * 2.0
    out = op(hyper_input, block_out, fp32_inject)
    assert out.dtype == torch.float32
    _compare(out, op.reference(hyper_input, block_out, fp32_inject))
    fp32_block_out = torch.randn(8, HIDDEN, dtype=torch.float32, device=DEV)
    _, _, inject = _operands((8,), seed=13)
    assert op(hyper_input, fp32_block_out, inject).dtype == torch.float32


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
