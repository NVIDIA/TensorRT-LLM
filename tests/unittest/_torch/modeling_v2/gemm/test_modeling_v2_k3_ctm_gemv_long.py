# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_ctm_gemv_long catalog entry.

The Kimi K3 TP16 per-rank shapes with their call sites' sig_col0, split, ring and push, at every M in 1..8: error of
the plain (sig_col0=-1) output against an fp64 product of the same bf16 inputs, the sigmoid columns torch.sigmoid of
that output bit for bit, identical bits on a repeated call, and each M's rows bit-identical to the same rows of the
8-row call. The schedule-only flags (ring, push, trigger_early) move no bits, a CUDA-graph replay returns the eager
bits, and calls outside the op's preconditions raise ValueError.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_ctm_gemv_long import (
    k3_ctm_gemv_long,
)

assert torch.cuda.is_available(), "k3_ctm_gemv_long requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

M_ALL = range(1, 9)
TOL = 8e-3  # max |y - ref| / max |ref| against an fp64 product of the same bf16 inputs

# (N, K, sig_col0, split, ring, push), as each call site passes them. mla_qkv_a_gate is the fused
# [W_a; W_g] projection, its gate columns returned as bf16(sigmoid).
CELLS = {
    "mla_qkv_a_gate": (2880, 7168, 2112, 6, 6, True),
    "dense_gate_up": (4224, 7168, -1, 4, 5, False),
    "dense_down": (7168, 2112, -1, 2, 6, False),  # K 2112 ends in a half k-tile
    "drafter_qkv": (512, 7168, -1, 8, 6, True),
    "drafter_gate_up": (1792, 7168, -1, 8, 6, True),
    "drafter_gate_up_synthetic": (1536, 7168, -1, 8, 6, True),
    "kda_qkvg": (3208, 7168, -1, 5, 6, True),  # 25 whole 128-row blocks and an 8-row last one
}

# One flag changed from a call site's values; each must return the call site's bits.
FLAG_CHANGES = {
    "dense_down": ({"push": True}, {"ring": 3}),
    "dense_gate_up": ({"push": True}, {"trigger_early": False}),
}


def _randn(rows: int, cols: int, seed: int, scale: float) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, cols, generator=gen, device="cuda") * scale).bfloat16()


def _weight(n: int, k: int) -> torch.Tensor:
    return _randn(n, k, 9, 0.02)


def _rows(k: int) -> torch.Tensor:
    return _randn(8, k, 10, 1.0)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    """max |y - ref| / max |ref|, in fp64."""
    return (y.double() - ref).abs().max().item() / ref.abs().max().item()


def test_call_site_cells() -> None:
    for name, (n, k, sig, split, ring, push) in CELLS.items():
        w = _weight(n, k)
        x8 = _rows(k)
        ref8 = x8.double() @ w.double().t()
        y8 = k3_ctm_gemv_long(x8, w, sig, split, ring, True, push)
        for m in M_ALL:
            cell = f"{name} M={m}"
            x = x8[:m].contiguous()
            y = k3_ctm_gemv_long(x, w, sig, split, ring, True, push)
            assert y.shape == (m, n) and y.dtype == torch.bfloat16, cell
            again = k3_ctm_gemv_long(x, w, sig, split, ring, True, push)
            assert torch.equal(_bits(y), _bits(again)), f"{cell}: a repeated call changed bits"
            assert torch.equal(_bits(y), _bits(y8[:m])), f"{cell}: rows differ from the 8-row call"
            plain = k3_ctm_gemv_long(x, w, -1, split, ring, True, push) if sig >= 0 else y
            rel = _rel(plain, ref8[:m])
            assert rel <= TOL, f"{cell}: max |y - ref| / max |ref| = {rel:.2e} > {TOL}"
            if sig >= 0:
                assert torch.equal(_bits(y[:, :sig]), _bits(plain[:, :sig])), (
                    f"{cell}: columns before sig_col0 differ from the sig_col0=-1 call"
                )
                assert torch.equal(_bits(y[:, sig:]), _bits(plain[:, sig:].sigmoid())), (
                    f"{cell}: columns from sig_col0 on are not torch.sigmoid of the sig_col0=-1 call"
                )


def test_schedule_flags_move_no_bits() -> None:
    """ring, push and trigger_early only schedule the call: one changed, the call site's bits come back."""
    for name, changes in FLAG_CHANGES.items():
        n, k, sig, split, ring, push = CELLS[name]
        w = _weight(n, k)
        x8 = _rows(k)
        site = {"split": split, "ring": ring, "trigger_early": True, "push": push}
        for m in M_ALL:
            x = x8[:m].contiguous()
            base = _bits(k3_ctm_gemv_long(x, w, sig, **site))
            for change in changes:
                y = k3_ctm_gemv_long(x, w, sig, **{**site, **change})
                assert torch.equal(_bits(y), base), (
                    f"{name} M={m} {change}: bits differ from the call site's"
                )


def test_cuda_graph_replay_matches_eager() -> None:
    """Captured after an eager call per key and replayed with x rewritten in place: the eager bits."""
    calls = []
    for name in ("mla_qkv_a_gate", "dense_down"):
        n, k, sig, split, ring, push = CELLS[name]
        w = _weight(n, k)
        for m in (1, 8):
            x = _rows(k)[:m].clone()
            k3_ctm_gemv_long(x, w, sig, split, ring, True, push)  # compiles the key outside capture
            calls.append((f"{name} M={m}", x, w, (sig, split, ring, True, push)))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [k3_ctm_gemv_long(x, w, *flags) for _, x, w, flags in calls]
    for seed, (_, x, _, _) in enumerate(calls, start=100):
        x.copy_(_randn(x.shape[0], x.shape[1], seed, 1.0))
    graph.replay()
    for (cell, x, w, flags), y in zip(calls, outs):
        eager = k3_ctm_gemv_long(x, w, *flags)
        assert torch.equal(_bits(y), _bits(eager)), f"{cell}: the replay differs from an eager call"


def test_unsupported_calls_raise_value_error() -> None:
    """The op's own check refuses these before compiling or launching anything."""
    w = _weight(7168, 2112)
    x8 = _rows(2112)
    cases = {
        "0 rows": (x8[:0], w, 2, 6),
        "9 rows": (torch.cat([x8, x8[:1]]), w, 2, 6),
        "fp16 x": (x8.half(), w, 2, 6),
        "row-strided x": (torch.cat([x8, x8], dim=1)[:, :2112], w, 2, 6),
        "K % 64 != 0": (x8[:, :2080].contiguous(), w[:, :2080].contiguous(), 2, 6),
        "split 3": (x8, w, 3, 5),
        "split 9": (x8, w, 9, 1),
        "ring 0": (x8, w, 2, 0),
        "ring past the rank's k-tiles": (x8, w, 8, 3),
        "ring past shared memory": (x8, w, 2, 7),
    }
    for name, (x, weight, split, ring) in cases.items():
        try:
            k3_ctm_gemv_long(x, weight, -1, split, ring, True, False)
        except ValueError:
            continue
        raise AssertionError(f"{name}: expected ValueError, the op accepted the call")
