# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the default target's qwen4_exp_hc_lowrank_act variant
(models/qwen4_exp/ops/hc/qwen4_exp_hc_lowrank_act): cells within one ulp of the float64
reference. A cell with `capture` runs its call inside a captured CUDA graph and replays it
over two steps' logits."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_lowrank_act import (  # noqa: E501
    qwen4_exp_hc_lowrank_act as op,
)

__extra_import_path__ = ["../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_lowrank_act requires a CUDA device"

DEV = "cuda"
RANK = 320


def _ulp_ratio(out, ref) -> float:
    eps = torch.finfo(ref.dtype).eps
    tiny = torch.finfo(ref.dtype).tiny
    return ((out.double() - ref.double()).abs() / (eps * ref.double().abs() + tiny)).max().item()


def _compare(out, ref, label: str = "") -> None:
    """One ulp: each output is one fp32 activation of an fp32 input, rounded
    once, so it can differ from the once-rounded float64 value only at a
    rounding midpoint."""
    worst = []
    for got, want in zip(out, ref):
        assert (got is None) == (want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
        worst.append(_ulp_ratio(got, want))
        assert worst[-1] <= 1.0, f"{worst[-1]:.4g} ulp"
    if label:
        print(f"  {label}: worst {', '.join(f'{u:.3g}' for u in worst)} ulp (low, inject)")


def _build(spec, seed):
    hc_count = spec.get("hc_count", 4)
    g = torch.Generator(device=DEV).manual_seed(seed)
    # The first projection's logits on this checkpoint reach |67| (down) / |48| (inject).
    logits = torch.randn(spec["rows"], spec.get("logits_width", 384), generator=g, device=DEV) * 20
    return logits, RANK, hc_count, spec.get("inject", True), spec.get("dtype", torch.bfloat16)


def _capture(args):
    """Capture one call over a private copy of the step's logits (the one argument a step
    writes), as the engine's graph runner does after its eager warmup:
    `(graph, static logits, static outputs)`. The Triton kernel is compiled by the eager
    call the caller makes first; one more call on a side stream warms the stream, as
    `torch.cuda.graph` asks."""
    static = args[0].clone()
    rest = args[1:]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(static, *rest)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(static, *rest)
    return graph, static, out


def _equal(a, b) -> bool:
    """Both outputs bitwise equal (`inject` None on both sides or equal)."""
    return all((x is None and y is None) or torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % (2**31)
    args = _build(cell.spec, seed=seed)
    low, inject = op(*args)
    assert low.shape == (cell.spec["rows"], RANK)
    _compare((low, inject), op.reference(*args), cell.why[:44])
    if not cell.spec.get("capture"):
        return

    # Captured: the graph replays the step it captured bitwise as the eager call
    # computed it, then a second step's logits, written into the captured buffer,
    # bitwise its own eager call and inside the gate against its reference.
    graph, static, replayed = _capture(args)
    graph.replay()
    torch.cuda.synchronize()
    assert _equal(replayed, (low, inject)), "the replay differs from the eager call it captured"
    second = _build(cell.spec, seed=(seed + 1) % (2**31))
    assert not torch.equal(second[0], args[0]), "the second step brings other logits"
    static.copy_(second[0])
    graph.replay()
    torch.cuda.synchronize()
    step2 = tuple(None if t is None else t.clone() for t in replayed)
    assert _equal(step2, op(*second)), "the second replay differs from its eager call"
    assert not _equal(step2, (low, inject)), "the second replay repeated the first step"
    _compare(step2, op.reference(*second), f"{cell.why[:44]} (second replay)")
