# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_ctm_gemv catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.k3_ctm_gemv import (
    k3_ctm_gemv,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "k3_ctm_gemv requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

M_ALL = range(1, 9)
TOL = 8e-3

# (N, K, split, push): the MLA o_proj at splits 1 and 2, the drafter o_proj (tuned and synthetic drafter).
CELLS = {
    cell.spec["name"]: tuple(cell.spec[key] for key in ("n", "k", "split", "push"))
    for cell in k3_ctm_gemv.CELLS
}


def _randn(rows: int, cols: int, seed: int, scale: float) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, cols, generator=gen, device="cuda") * scale).bfloat16()


def _weight(n: int, k: int) -> torch.Tensor:
    return _randn(n, k, 1, 0.03)


def _rows(k: int) -> torch.Tensor:
    return _randn(8, k, 2, 1.0)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    """max |y - ref| / max |ref|, in fp64."""
    return (y.double() - ref).abs().max().item() / ref.abs().max().item()


@pytest.mark.parametrize("cell", k3_ctm_gemv.CELLS, ids=[c.why[:40] for c in k3_ctm_gemv.CELLS])
def test_call_site_cells(cell) -> None:
    spec = cell.spec
    n, k, split, push = (spec[key] for key in ("n", "k", "split", "push"))
    seed = zlib.crc32(cell.why.encode()) % 2**31
    w = _randn(n, k, seed, 0.03)
    x8 = _randn(8, k, seed + 1, 1.0)
    ref8 = k3_ctm_gemv.reference(x8, w, True, split, push)
    with validating(k3_ctm_gemv):
        y8 = k3_ctm_gemv(x8, w, True, split, push)
        for m in spec["ms"]:
            label = f"{cell.why} M={m}"
            x = x8[:m].contiguous()
            y = k3_ctm_gemv(x, w, True, split, push)
            assert y.shape == (m, n) and y.dtype == torch.bfloat16, label
            rel = _rel(y, ref8[:m])
            assert rel <= TOL, f"{label}: max |y - ref| / max |ref| = {rel:.2e} > {TOL}"
            again = k3_ctm_gemv(x, w, True, split, push)
            assert torch.equal(_bits(y), _bits(again)), f"{label}: a repeated call changed bits"
            assert torch.equal(_bits(y), _bits(y8[:m])), f"{label}: rows differ from the 8-row call"


def test_schedule_flags_move_no_bits() -> None:
    """At the MLA o_proj with split 2, push=True and trigger_early=False return the call site's bits."""
    n, k, split = 7168, 768, 2
    w = _weight(n, k)
    x8 = _rows(k)
    for m in M_ALL:
        x = x8[:m].contiguous()
        base = _bits(k3_ctm_gemv(x, w, True, split, False))
        for trigger_early, push in ((True, True), (False, False)):
            y = k3_ctm_gemv(x, w, trigger_early, split, push)
            assert torch.equal(_bits(y), base), (
                f"M={m} trigger_early={trigger_early} push={push}: bits differ from the call site's"
            )


def test_cuda_graph_replay_matches_eager() -> None:
    """Captured after an eager call per key and replayed with x rewritten in place: the eager bits."""
    calls = []
    for name in ("mla_o_proj_s1", "mla_o_proj_s2"):
        n, k, split, push = CELLS[name]
        w = _weight(n, k)
        for m in (1, 8):
            x = _rows(k)[:m].clone()
            k3_ctm_gemv(x, w, True, split, push)  # compiles the key outside capture
            calls.append((f"{name} M={m}", x, w, split, push))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [k3_ctm_gemv(x, w, True, split, push) for _, x, w, split, push in calls]
    for seed, (_, x, *_) in enumerate(calls, start=100):
        x.copy_(_randn(x.shape[0], x.shape[1], seed, 1.0))
    graph.replay()
    for (cell, x, w, split, push), y in zip(calls, outs):
        eager = k3_ctm_gemv(x, w, True, split, push)
        assert torch.equal(_bits(y), _bits(eager)), f"{cell}: the replay differs from an eager call"


def test_unsupported_calls_raise_value_error() -> None:
    """The op's own check refuses these before compiling or launching anything."""
    w = _weight(7168, 768)
    x8 = _rows(768)
    cases = {
        "0 rows": (x8[:0], w, 1),
        "9 rows": (torch.cat([x8, x8[:1]]), w, 1),
        "fp16 x": (x8.half(), w, 1),
        "row-strided x": (torch.cat([x8, x8], dim=1)[:, :768], w, 1),
        "N % 128 != 0": (x8, w[:7104], 1),
        "K % 128 != 0": (x8[:, :704].contiguous(), w[:, :704].contiguous(), 1),
        "split 3": (x8, w, 3),
        "7 k-tiles on one CTA": (_rows(896), _weight(7168, 896), 1),
    }
    for name, (x, weight, split) in cases.items():
        try:
            k3_ctm_gemv(x, weight, True, split, False)
        except ValueError:
            continue
        raise AssertionError(f"{name}: expected ValueError, the op accepted the call")


@pytest.mark.parametrize("operand", ("x", "weight"))
def test_guard_refuses_misaligned_inputs(operand) -> None:
    x = _rows(768)[:1]
    weight = _weight(7168, 768)
    source = x if operand == "x" else weight
    flat = torch.empty(source.numel() + 1, dtype=source.dtype, device=source.device)
    misaligned = flat[1:].view(source.shape)
    if operand == "x":
        x = misaligned
    else:
        weight = misaligned
    with pytest.raises(AssertionError, match="16-byte-aligned"):
        with validating(k3_ctm_gemv):
            k3_ctm_gemv(x, weight)


def test_guard_refuses_weight_on_another_device() -> None:
    x = _rows(768)[:1]
    weight = torch.empty(128, 768, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="weight must be on"):
        with validating(k3_ctm_gemv):
            k3_ctm_gemv(x, weight)
