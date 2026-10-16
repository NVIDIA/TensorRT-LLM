# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_decode_gemv catalog entry.

Both kernels at Kimi K3's TP16 per-rank shapes, at every M in 1..8, against an fp64 torch product (error at most 8e-3
of max |ref|), with run-to-run bits, each M's rows equal to the same rows of the 8-row call, trigger_early False equal
to True, a CUDA graph replayed with its input rewritten, and the refused calls.
"""

import functools

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_decode_gemv import k3_decode_gemv

assert torch.cuda.is_available(), "k3_decode_gemv requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

TOL = 8e-3  # max |y - ref| / max |ref|, ref an fp64 product
M_ALL = range(1, 9)
# (N, K) per rank at TP16. The KDA o_proj: short K, one CTA per 128-row weight tile with the whole K resident.
SHORT_K = (7168, 768)
# The KDA input projection: split K over clusters of 4 CTAs; N is not a multiple of 128 (the last tile has 8 rows).
SPLIT_K = (3208, 7168)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return (y.double() - ref).abs().max().item() / ref.abs().max().item()


@functools.lru_cache(maxsize=None)
def _inputs(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The weight [N, K] and 8 activation rows [8, K]."""
    gen = torch.Generator(device="cuda").manual_seed(n * 100003 + k)
    weight = (torch.randn(n, k, generator=gen, device="cuda") * 0.03).bfloat16()
    x8 = torch.randn(8, k, generator=gen, device="cuda").bfloat16()
    # The kernels read the weight before their PDL grid-dependency wait: it is written well before any call.
    torch.cuda.synchronize()
    return weight, x8


def _check_cells(shape: tuple[int, int]) -> None:
    n, k = shape
    weight, x8 = _inputs(n, k)
    first = {}
    for trigger_early in (True, False):
        y8 = k3_decode_gemv(x8, weight, trigger_early)
        for m in M_ALL:
            x = x8[:m]
            y = k3_decode_gemv(x, weight, trigger_early)
            again = k3_decode_gemv(x, weight, trigger_early)
            ref = x.double() @ weight.double().t()
            assert y.shape == (m, n) and y.dtype == torch.bfloat16 and y.device == x.device
            assert y.is_contiguous()
            err = _rel(y, ref)
            assert err <= TOL, f"{shape} M={m} trigger_early={trigger_early}: rel {err:.3e}"
            assert _same(y, again), f"{shape} M={m}: a rerun differs"
            assert _same(y, y8[:m]), f"{shape} M={m}: rows differ from the 8-row call"
            if trigger_early:
                first[m] = y
            else:
                assert _same(y, first[m]), f"{shape} M={m}: trigger_early changes the result"


def test_short_k_cells() -> None:
    """[7168, 768] (one CTA per 128-row tile, the whole K resident) at every M, trigger_early True and False."""
    _check_cells(SHORT_K)


def test_split_k_cells() -> None:
    """[3208, 7168] (clusters of 4 CTAs over K) at every M, trigger_early True and False."""
    _check_cells(SPLIT_K)


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
