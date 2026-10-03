# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_ctm_gemv_swiglu catalog entry.

The Kimi K3 TP16 per-rank drafter down projection with its call site's split and push, at every M in 1..8: error
against an fp64 product of torch's bf16 silu_and_mul activation, identical bits on a repeated call, and each M's rows
bit-identical to the same rows of the 8-row call. The schedule-only flags (push, trigger_early) move no bits, a
CUDA-graph replay returns the eager bits, and calls outside the op's preconditions raise ValueError.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_ctm_gemv_swiglu import (
    k3_ctm_gemv_swiglu,
)

assert torch.cuda.is_available(), "k3_ctm_gemv_swiglu requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

M_ALL = range(1, 9)
TOL = 8e-3  # max |y - ref| / max |ref| against an fp64 product of the bf16 activation

# (N, K, split, push): the drafter down projection (tuned and synthetic drafter); K 896 is 7 k-tiles.
CELLS = {
    "drafter_down": (7168, 896, 2, True),
    "drafter_down_synthetic": (7168, 768, 2, True),
}


def _randn(rows: int, cols: int, seed: int, scale: float) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, cols, generator=gen, device="cuda") * scale).bfloat16()


def _weight(n: int, k: int) -> torch.Tensor:
    return _randn(n, k, 7, 0.02)


def _gate_up(k: int) -> torch.Tensor:
    return _randn(8, 2 * k, 8, 2.0)


def _silu_and_mul(gu: torch.Tensor) -> torch.Tensor:
    """silu_and_mul's fp32 arithmetic and one bf16 rounding (gate columns first)."""
    k = gu.shape[1] // 2
    return (F.silu(gu[:, :k].float()) * gu[:, k:].float()).bfloat16()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    """max |y - ref| / max |ref|, in fp64."""
    return (y.double() - ref).abs().max().item() / ref.abs().max().item()


def test_call_site_cells() -> None:
    for name, (n, k, split, push) in CELLS.items():
        w = _weight(n, k)
        gu8 = _gate_up(k)
        ref8 = _silu_and_mul(gu8).double() @ w.double().t()
        y8 = k3_ctm_gemv_swiglu(gu8, w, True, split, push)
        for m in M_ALL:
            cell = f"{name} M={m}"
            gu = gu8[:m].contiguous()
            y = k3_ctm_gemv_swiglu(gu, w, True, split, push)
            assert y.shape == (m, n) and y.dtype == torch.bfloat16, cell
            rel = _rel(y, ref8[:m])
            assert rel <= TOL, f"{cell}: max |y - ref| / max |ref| = {rel:.2e} > {TOL}"
            again = k3_ctm_gemv_swiglu(gu, w, True, split, push)
            assert torch.equal(_bits(y), _bits(again)), f"{cell}: a repeated call changed bits"
            assert torch.equal(_bits(y), _bits(y8[:m])), f"{cell}: rows differ from the 8-row call"


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
