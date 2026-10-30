# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_lowrank_act entry: cells within one ulp of the
float64 reference, the fp32-logits guard, graph capture."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_lowrank_act import (  # noqa: E501
    qwen4_exp_hc_lowrank_act as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_lowrank_act requires a CUDA device"

DEV = "cuda"
RANK = 320


def _ulp_ratio(out, ref) -> float:
    eps = torch.finfo(ref.dtype).eps
    tiny = torch.finfo(ref.dtype).tiny
    return ((out.double() - ref.double()).abs() / (eps * ref.double().abs() + tiny)).max().item()


def _compare(out, ref) -> None:
    """One ulp: each output is one fp32 activation of an fp32 input, rounded
    once, so it can differ from the once-rounded float64 value only at a
    rounding midpoint."""
    for got, want in zip(out, ref):
        assert (got is None) == (want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
        assert _ulp_ratio(got, want) <= 1.0, f"{_ulp_ratio(got, want):.4g} ulp"


def _build(spec, seed):
    hc_count = spec.get("hc_count", 4)
    g = torch.Generator(device=DEV).manual_seed(seed)
    # The first projection's logits on this checkpoint reach |67| (down) / |48| (inject).
    logits = torch.randn(spec["rows"], spec.get("logits_width", 384), generator=g, device=DEV) * 20
    return logits, RANK, hc_count, spec.get("inject", True), spec.get("dtype", torch.bfloat16)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    low, inject = op(*args)
    assert low.shape == (cell.spec["rows"], RANK)
    _compare((low, inject), op.reference(*args))


def test_guard_refuses_bf16_logits() -> None:
    logits, rank, c, wi, dt = _build(dict(rows=4), seed=4)
    with validating(op), pytest.raises(AssertionError):
        op(logits.to(torch.bfloat16), rank, c, wi, dt)


def test_cuda_graph_capture_and_replay() -> None:
    logits, rank, c, wi, dt = _build(dict(rows=8), seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(logits, rank, c, wi, dt)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_low, g_inject = op(logits, rank, c, wi, dt)
    logits.copy_(_build(dict(rows=8), seed=6)[0])
    graph.replay()
    torch.cuda.synchronize()
    e_low, e_inject = op(logits, rank, c, wi, dt)
    assert torch.equal(g_low, e_low) and torch.equal(g_inject, e_inject)
