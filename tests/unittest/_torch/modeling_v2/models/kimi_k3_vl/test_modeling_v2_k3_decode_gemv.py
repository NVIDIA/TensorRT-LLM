# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_decode_gemv catalog entry."""

import functools
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.k3_decode_gemv import (
    k3_decode_gemv,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "k3_decode_gemv requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

TOL = 8e-3
# (N, K) per rank at TP16. The KDA o_proj: short K, one CTA per 128-row weight tile with the whole K resident.
SHORT_K = tuple(k3_decode_gemv.CELLS[0].spec[key] for key in ("n", "k"))
# The KDA input projection: split K over clusters of 4 CTAs; N is not a multiple of 128 (the last tile has 8 rows).
SPLIT_K = tuple(k3_decode_gemv.CELLS[1].spec[key] for key in ("n", "k"))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return (y.double() - ref).abs().max().item() / ref.abs().max().item()


@functools.lru_cache(maxsize=None)
def _inputs(n: int, k: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The weight [N, K] and 8 activation rows [8, K]."""
    gen = torch.Generator(device="cuda").manual_seed(n * 100003 + k if seed is None else seed)
    weight = (torch.randn(n, k, generator=gen, device="cuda") * 0.03).bfloat16()
    x8 = torch.randn(8, k, generator=gen, device="cuda").bfloat16()
    # The kernels read the weight before their PDL grid-dependency wait: it is written well before any call.
    torch.cuda.synchronize()
    return weight, x8


@pytest.mark.parametrize(
    "cell", k3_decode_gemv.CELLS, ids=[c.why[:40] for c in k3_decode_gemv.CELLS]
)
def test_certified_cells(cell) -> None:
    n, k = (cell.spec[key] for key in ("n", "k"))
    weight, x8 = _inputs(n, k, zlib.crc32(cell.why.encode()) % 2**31)
    first = {}
    with validating(k3_decode_gemv):
        for trigger_early in cell.spec["trigger_early"]:
            ref8 = k3_decode_gemv.reference(x8, weight, trigger_early)
            y8 = k3_decode_gemv(x8, weight, trigger_early)
            for m in cell.spec["ms"]:
                x = x8[:m]
                y = k3_decode_gemv(x, weight, trigger_early)
                again = k3_decode_gemv(x, weight, trigger_early)
                assert y.shape == (m, n) and y.dtype == torch.bfloat16 and y.device == x.device
                assert y.is_contiguous()
                err = _rel(y, ref8[:m])
                assert err <= TOL, f"{cell.why} M={m} trigger_early={trigger_early}: rel {err:.3e}"
                assert _same(y, again), f"{cell.why} M={m}: a rerun differs"
                assert _same(y, y8[:m]), f"{cell.why} M={m}: rows differ from the 8-row call"
                if trigger_early:
                    first[m] = y
                else:
                    assert _same(y, first[m]), f"{cell.why} M={m}: trigger_early changes the result"


def test_graph_replay() -> None:
    """M 1 and M 8 calls on each weight in one CUDA graph, captured after eager calls compiled them and replayed with
    the activation rewritten in place: every replay bit-identical to eager calls on the same rows."""
    gen = torch.Generator(device="cuda").manual_seed(5)
    weights, x_bufs = [], []
    for n, k in (SHORT_K, SPLIT_K):
        weight, x8 = _inputs(n, k)
        weights.append(weight)
        x_bufs.append(x8.clone())
        for m in (1, 8):
            k3_decode_gemv(x_bufs[-1][:m], weight)  # compiles outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = [{m: k3_decode_gemv(x[:m], w) for m in (1, 8)} for w, x in zip(weights, x_bufs)]
    for _ in range(3):
        for x in x_bufs:
            x.copy_(torch.randn(x.shape, generator=gen, device="cuda").bfloat16())
        graph.replay()
        for i, (w, x) in enumerate(zip(weights, x_bufs)):
            for m in (1, 8):
                assert _same(out[i][m], k3_decode_gemv(x[:m], w)), f"replay of weight {i}, M={m}"


def test_refused_calls() -> None:
    """M 0 and 9, and weights neither kernel takes, raise ValueError before launching anything."""
    weight, _ = _inputs(*SHORT_K)
    for m in (0, 9):
        x = torch.zeros(m, SHORT_K[1], dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="unsupported call"):
            k3_decode_gemv(x, weight)
    # [128, 1152]: 9 k-tiles, more than short K holds and not a multiple of 4 for split K.
    # [200, 768]: short K needs whole 128-row tiles, split K more than 6 k-tiles.
    for n, k in ((128, 1152), (200, 768)):
        x = torch.zeros(1, k, dtype=torch.bfloat16, device="cuda")
        w = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="unsupported call"):
            k3_decode_gemv(x, w)


@pytest.mark.parametrize("operand", ("x", "weight"))
def test_guard_refuses_misaligned_inputs(operand) -> None:
    weight, x8 = _inputs(*SHORT_K)
    x = x8[:1]
    source = x if operand == "x" else weight
    flat = torch.empty(source.numel() + 1, dtype=source.dtype, device=source.device)
    misaligned = flat[1:].view(source.shape)
    if operand == "x":
        x = misaligned
    else:
        weight = misaligned
    with pytest.raises(AssertionError, match="16-byte-aligned"):
        with validating(k3_decode_gemv):
            k3_decode_gemv(x, weight)


def test_guard_refuses_weight_on_another_device() -> None:
    _, x8 = _inputs(*SHORT_K)
    weight = torch.empty(128, SHORT_K[1], dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="weight must be on"):
        with validating(k3_decode_gemv):
            k3_decode_gemv(x8[:1], weight)
