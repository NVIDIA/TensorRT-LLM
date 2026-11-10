# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_ctm_gemv_swiglu catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.k3_ctm_gemv_swiglu import (
    k3_ctm_gemv_swiglu,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "k3_ctm_gemv_swiglu requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

M_ALL = range(1, 9)
TOL = 8e-3

# (N, K, split, push): the drafter down projection (tuned and synthetic drafter); K 896 is 7 k-tiles.
CELLS = {
    cell.spec["name"]: tuple(cell.spec[key] for key in ("n", "k", "split", "push"))
    for cell in k3_ctm_gemv_swiglu.CELLS
}


def _randn(rows: int, cols: int, seed: int, scale: float) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, cols, generator=gen, device="cuda") * scale).bfloat16()


def _weight(n: int, k: int) -> torch.Tensor:
    return _randn(n, k, 7, 0.02)


def _gate_up(k: int) -> torch.Tensor:
    return _randn(8, 2 * k, 8, 2.0)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    """max |y - ref| / max |ref|, in fp64."""
    return (y.double() - ref).abs().max().item() / ref.abs().max().item()


@pytest.mark.parametrize(
    "cell", k3_ctm_gemv_swiglu.CELLS, ids=[c.why[:40] for c in k3_ctm_gemv_swiglu.CELLS]
)
def test_call_site_cells(cell) -> None:
    spec = cell.spec
    n, k, split, push = (spec[key] for key in ("n", "k", "split", "push"))
    seed = zlib.crc32(cell.why.encode()) % 2**31
    w = _randn(n, k, seed, 0.02)
    gu8 = _randn(8, 2 * k, seed + 1, 2.0)
    ref8 = k3_ctm_gemv_swiglu.reference(gu8, w, True, split, push)
    with validating(k3_ctm_gemv_swiglu):
        y8 = k3_ctm_gemv_swiglu(gu8, w, True, split, push)
        for m in spec["ms"]:
            label = f"{cell.why} M={m}"
            gu = gu8[:m].contiguous()
            y = k3_ctm_gemv_swiglu(gu, w, True, split, push)
            assert y.shape == (m, n) and y.dtype == torch.bfloat16, label
            rel = _rel(y, ref8[:m])
            assert rel <= TOL, f"{label}: max |y - ref| / max |ref| = {rel:.2e} > {TOL}"
            again = k3_ctm_gemv_swiglu(gu, w, True, split, push)
            assert torch.equal(_bits(y), _bits(again)), f"{label}: a repeated call changed bits"
            assert torch.equal(_bits(y), _bits(y8[:m])), f"{label}: rows differ from the 8-row call"


def test_schedule_flags_move_no_bits() -> None:
    """At the drafter down, push=False and trigger_early=False return the call site's bits."""
    n, k, split, push = CELLS["drafter_down"]
    w = _weight(n, k)
    gu8 = _gate_up(k)
    for m in M_ALL:
        gu = gu8[:m].contiguous()
        base = _bits(k3_ctm_gemv_swiglu(gu, w, True, split, push))
        for trigger_early, push_flag in ((True, not push), (False, push)):
            y = k3_ctm_gemv_swiglu(gu, w, trigger_early, split, push_flag)
            assert torch.equal(_bits(y), base), (
                f"M={m} trigger_early={trigger_early} push={push_flag}: bits differ from the call site's"
            )


def test_cuda_graph_replay_matches_eager() -> None:
    """Captured after an eager call per key and replayed with gu rewritten in place: the eager bits."""
    n, k, split, push = CELLS["drafter_down"]
    w = _weight(n, k)
    calls = []
    for m in (1, 8):
        gu = _gate_up(k)[:m].clone()
        k3_ctm_gemv_swiglu(gu, w, True, split, push)  # compiles the key outside capture
        calls.append((m, gu))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [k3_ctm_gemv_swiglu(gu, w, True, split, push) for _, gu in calls]
    for seed, (m, gu) in enumerate(calls, start=100):
        gu.copy_(_randn(m, 2 * k, seed, 2.0))
    graph.replay()
    for (m, gu), y in zip(calls, outs):
        eager = k3_ctm_gemv_swiglu(gu, w, True, split, push)
        assert torch.equal(_bits(y), _bits(eager)), f"M={m}: the replay differs from an eager call"


def test_unsupported_calls_raise_value_error() -> None:
    """The op's own check refuses these before compiling or launching anything."""
    w = _weight(7168, 896)
    gu8 = _gate_up(896)
    cases = {
        "0 rows": (gu8[:0], w, 2),
        "9 rows": (torch.cat([gu8, gu8[:1]]), w, 2),
        "fp16 gu": (gu8.half(), w, 2),
        "row-strided gu": (torch.cat([gu8, gu8], dim=1)[:, : 2 * 896], w, 2),
        "gu width != 2 K": (_randn(8, 1664, 8, 2.0), w, 2),
        "N % 128 != 0": (gu8, w[:7104], 2),
        "split 3": (gu8, w, 3),
        "7 k-tiles on one CTA": (gu8, w, 1),
    }
    for name, (gu, weight, split) in cases.items():
        try:
            k3_ctm_gemv_swiglu(gu, weight, True, split, True)
        except ValueError:
            continue
        raise AssertionError(f"{name}: expected ValueError, the op accepted the call")


@pytest.mark.parametrize("operand", ("gu", "weight"))
def test_guard_refuses_misaligned_inputs(operand) -> None:
    gu = _gate_up(896)[:1]
    weight = _weight(7168, 896)
    source = gu if operand == "gu" else weight
    flat = torch.empty(source.numel() + 1, dtype=source.dtype, device=source.device)
    misaligned = flat[1:].view(source.shape)
    if operand == "gu":
        gu = misaligned
    else:
        weight = misaligned
    with pytest.raises(AssertionError, match="16-byte-aligned"):
        with validating(k3_ctm_gemv_swiglu):
            k3_ctm_gemv_swiglu(gu, weight)


def test_guard_refuses_weight_on_another_device() -> None:
    gu = _gate_up(896)[:1]
    weight = torch.empty(128, 896, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="weight must be on"):
        with validating(k3_ctm_gemv_swiglu):
            k3_ctm_gemv_swiglu(gu, weight)
