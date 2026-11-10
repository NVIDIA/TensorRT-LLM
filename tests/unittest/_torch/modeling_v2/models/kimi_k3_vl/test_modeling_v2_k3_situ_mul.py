# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_situ_mul catalog entry.

The Kimi K3 dense MLP's SiTU-and-mul at TP16 (gate_up [M, 4224] -> [M, 2112]) with the checkpoint's and the default
(beta, linear_beta), at every M in 1..8: error against SituAndMul's formula evaluated in fp32, identical bits on a
repeated call, and each M's rows bit-identical to the same rows of the 8-row call. A CUDA-graph replay returns the
eager bits, calls outside the op's preconditions raise ValueError, and the wrapper refuses linear_beta=0.0.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.activation.k3_situ_mul import (
    k3_situ_mul,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "k3_situ_mul requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

TOL = 8e-3
K = 2112  # the dense MLP's intermediate size per rank at TP16
SITU = {"k3": (4.0, 25.0), "defaults": (1.0, None)}  # (beta, linear_beta)


def _randn(rows: int, cols: int, seed: int, scale: float) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, cols, generator=gen, device="cuda") * scale).bfloat16()


def _gate_up() -> torch.Tensor:
    return _randn(8, 2 * K, 14, 2.0)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    """max |y - ref| / max |ref|, in fp64."""
    return (y.double() - ref).abs().max().item() / ref.abs().max().item()


@pytest.mark.parametrize("cell", k3_situ_mul.CELLS, ids=[c.why[:40] for c in k3_situ_mul.CELLS])
def test_call_site_cells(cell) -> None:
    spec = cell.spec
    gu8 = _randn(8, spec["width"], zlib.crc32(cell.why.encode()) % 2**31, 2.0)
    m, beta, linear_beta = spec["rows"], spec["beta"], spec["linear_beta"]
    gu = gu8[:m].contiguous()
    with validating(k3_situ_mul):
        y8 = k3_situ_mul(gu8, beta, linear_beta)
        y = k3_situ_mul(gu, beta, linear_beta)
        again = k3_situ_mul(gu, beta, linear_beta)
    ref = k3_situ_mul.reference(gu, beta, linear_beta)
    assert y.shape == (m, K) and y.dtype == torch.bfloat16, cell.why
    rel = _rel(y, ref)
    assert rel <= TOL, f"{cell.why}: max |y - ref| / max |ref| = {rel:.2e} > {TOL}"
    assert torch.equal(_bits(y), _bits(again)), f"{cell.why}: a repeated call changed bits"
    assert torch.equal(_bits(y), _bits(y8[:m])), f"{cell.why}: rows differ from the 8-row call"


def test_cuda_graph_replay_matches_eager() -> None:
    """Captured after an eager call per key and replayed with gu rewritten in place: the eager bits."""
    calls = []
    for name, (beta, linear_beta) in SITU.items():
        for m in (1, 8):
            gu = _gate_up()[:m].clone()
            k3_situ_mul(gu, beta, linear_beta)  # compiles the key outside capture
            calls.append((f"{name} M={m}", gu, beta, linear_beta))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [k3_situ_mul(gu, beta, linear_beta) for _, gu, beta, linear_beta in calls]
    for seed, (_, gu, *_) in enumerate(calls, start=100):
        gu.copy_(_randn(gu.shape[0], gu.shape[1], seed, 2.0))
    graph.replay()
    for (cell, gu, beta, linear_beta), y in zip(calls, outs):
        eager = k3_situ_mul(gu, beta, linear_beta)
        assert torch.equal(_bits(y), _bits(eager)), f"{cell}: the replay differs from an eager call"


def test_unsupported_calls_raise_value_error() -> None:
    """The op's own check refuses these before compiling or launching anything."""
    gu8 = _gate_up()
    flat = torch.zeros(8 * 2 * K + 8, dtype=torch.bfloat16, device="cuda")
    cases = {
        "0 rows": gu8[:0],
        "9 rows": torch.cat([gu8, gu8[:1]]),
        "width % 16 != 0": _randn(8, 2 * K - 8, 14, 2.0),
        "gu 2 bytes past a 16-byte boundary": flat[1 : 1 + 8 * 2 * K].view(8, 2 * K),
        "fp16 gu": gu8.half(),
        "row-strided gu": torch.cat([gu8, gu8], dim=1)[:, : 2 * K],
        "1-D gu": gu8[0],
    }
    for name, gu in cases.items():
        try:
            k3_situ_mul(gu, 4.0, 25.0)
        except ValueError:
            continue
        raise AssertionError(f"{name}: expected ValueError, the op accepted the call")


def test_zero_linear_beta_is_refused_before_dispatch() -> None:
    """The op would run linear_beta=0.0 as 1.0 with no error; the wrapper refuses it instead."""
    gu = _gate_up()[:1]
    for linear_beta in (0.0, -0.0):
        try:
            with validating(k3_situ_mul):
                k3_situ_mul(gu, 4.0, linear_beta)
        except AssertionError:
            continue
        raise AssertionError(f"linear_beta={linear_beta} should have been refused before dispatch")
