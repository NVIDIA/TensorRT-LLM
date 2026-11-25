# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's qwen4_exp_hc_decode_tc_first variant
(ops/hc/qwen4_exp_hc_decode_tc_first__decode): cells in the band against the float64
reference at the decode rows the members run, the row-count guard, graph replay at the
note's 1, 4, 16 and 256 rows (every tile variant)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default.ops.hc.qwen4_exp_hc_decode_tc_first__decode import (  # noqa: E501
    qwen4_exp_hc_decode_tc_first as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_decode_tc_first requires a CUDA device"

DEV = "cuda"
RANK = 320


def _ratio(out, ref) -> float:
    """Worst element's distance from `ref` as a fraction of the band: one ulp of
    the element (`eps * |ref|`) plus `eps / 16` of the tensor's largest value.
    An fp32-accumulated 10240-term projection rounded once can sit one ulp off
    the float64 value at a rounding midpoint; the absolute term covers SiLU's
    cancellation near zero, where an fp32 logit error of ~1e-6 relative to the
    row's scale is many ulps of a tiny output."""
    eps = torch.finfo(ref.dtype).eps
    r = ref.double()
    return ((out.double() - r).abs() / (eps * r.abs() + eps / 16 * r.abs().max())).max().item()


def _compare(out, ref) -> None:
    for got, want in zip(out, ref):
        assert (got is None) == (want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
        assert _ratio(got, want) <= 1.0, f"{_ratio(got, want):.4g}x the band"


def _build(spec, seed):
    hc, hidden = 4, 2560
    rows, width, inject = spec["rows"], hc * hidden, spec.get("inject", True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    # This checkpoint's layer statistics: down ~N(0, 0.0296), block_inject ~N(0, 0.0107),
    # the normed stream unit-RMS per stream times (1 + gamma), gamma ~N(0, 0.477).
    down = torch.randn(RANK, width, generator=g, device=DEV) * 0.0296
    parts = [down] + ([torch.randn(hc, width, generator=g, device=DEV) * 0.0107] if inject else [])
    w = torch.cat(parts)
    w = torch.cat([w, torch.zeros(384 - w.shape[0], width, device=DEV)])
    gamma = torch.randn(width, generator=g, device=DEV) * 0.477
    x = torch.randn(rows, hc, hidden, generator=g, device=DEV)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = (x.flatten(-2) * (1.0 + gamma)).to(torch.bfloat16)
    return normed, w.to(torch.bfloat16).t(), RANK, hc, inject


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    kw = {k: cell.spec[k] for k in ("tm", "rows_on_n", "tm64x2") if k in cell.spec}
    low, inject = op(*args, **kw)
    assert low.shape == (cell.spec["rows"], RANK)
    _compare((low, inject), op.reference(*args, **kw))


def test_guard_refuses_more_than_two_tiles_of_rows() -> None:
    normed, first, rank, c, wi = _build(dict(rows=257), seed=4)
    with validating(op), pytest.raises(AssertionError, match="at most 256 rows"):
        op(normed, first, rank, c, wi)


@pytest.mark.parametrize("rows", [1, 4, 16, 256])
def test_cuda_graph_capture_and_replay(rows) -> None:
    normed, first, rank, c, wi = _build(dict(rows=rows), seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(normed, first, rank, c, wi)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_low, g_inject = op(normed, first, rank, c, wi)
    fresh = _build(dict(rows=rows), seed=6)[0]
    normed.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    low, inject = op(fresh, first, rank, c, wi)
    assert torch.equal(g_low, low) and torch.equal(g_inject, inject)


@pytest.mark.parametrize("inject", [True, False])
@pytest.mark.parametrize("rows", [5, 8, 17, 36, 63, 64])
def test_tile64_is_bitwise_the_default_plan(rows, inject) -> None:
    """`tm=64` sums the same K slices in the same order as the 128-row tile: the same bits,
    eagerly and replayed from a CUDA graph."""
    normed, first, rank, c, wi = _build(dict(rows=rows, inject=inject), seed=700 + rows)
    ref = op(normed, first, rank, c, wi)
    got = op(normed, first, rank, c, wi, tm=64)
    torch.cuda.synchronize()
    for a, b in zip(ref, got):
        assert (a is None) == (b is None) and (a is None or torch.equal(a, b))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = op(normed, first, rank, c, wi, tm=64)
    fresh = _build(dict(rows=rows, inject=inject), seed=800 + rows)[0]
    normed.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(g_out, op(fresh, first, rank, c, wi)):
        assert (a is None) == (b is None) and (a is None or torch.equal(a, b))


@pytest.mark.parametrize("rows", [5, 8, 12, 16, 65, 72, 80, 81, 88, 96])
def test_rows_on_n_is_bitwise_the_default_plan(rows) -> None:
    """The rows-on-N kernel (65..96 rows, MMA N rounded up to 16) gives the 128-row tile's
    bits for a mixer with the injection, eagerly and replayed."""
    normed, first, rank, c, wi = _build(dict(rows=rows), seed=900 + rows)
    ref = op(normed, first, rank, c, wi)
    got = op(normed, first, rank, c, wi, rows_on_n=True)
    torch.cuda.synchronize()
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_low, g_inject = op(normed, first, rank, c, wi, rows_on_n=True)
    fresh = _build(dict(rows=rows), seed=1000 + rows)[0]
    normed.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    low, inject = op(fresh, first, rank, c, wi)
    assert torch.equal(g_low, low) and torch.equal(g_inject, inject)


def test_guard_refuses_rows_on_n_outside_its_rows() -> None:
    normed, first, rank, c, wi = _build(dict(rows=97), seed=7)
    with validating(op), pytest.raises(AssertionError, match="rows on N: 1..96"):
        op(normed, first, rank, c, wi, rows_on_n=True)


@pytest.mark.parametrize("rows", [65, 68, 80, 96, 112, 128])
def test_tm64x2_is_one_rounding_from_the_default_plan(rows) -> None:
    """`tm64x2=True` (two 64-row tiles, an 8-way K split) stays in the float64 reference's band, as
    every cell does, and moves `low` and `inject` by at most one bf16 step of the default plan's
    wherever a lane is not cancelling (at least `eps / 16` of the tensor's largest value, the
    band's absolute term). A cancelling lane can move further: the two fp32 sums of 10240 terms
    differ by a few fp32 ulps of the row's scale, many bf16 steps of a value near 0 (full-stack3
    GPU gate: 1 lane at 65 rows, 2 at 80, |low| about 1e-4, both plans inside the band). Its
    graph replay is bitwise its eager call."""
    normed, first, rank, c, wi = _build(dict(rows=rows), seed=1100 + rows)
    ref = op(normed, first, rank, c, wi)
    got = op(normed, first, rank, c, wi, tm64x2=True)
    exact = op.reference(normed, first, rank, c, wi)
    torch.cuda.synchronize()
    eps = torch.finfo(torch.bfloat16).eps
    for a, b, r in zip(ref, got, exact):
        if a is None:
            continue
        assert _ratio(b, r) <= 1.0, f"{_ratio(b, r):.4g}x the band"
        a, b = a.float(), b.float()
        kept = a.abs() >= eps / 16 * a.abs().max()
        assert ((a - b).abs() <= eps * a.abs())[kept].all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_low, g_inject = op(normed, first, rank, c, wi, tm64x2=True)
    fresh = _build(dict(rows=rows), seed=1200 + rows)[0]
    normed.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    low, inject = op(fresh, first, rank, c, wi, tm64x2=True)
    assert torch.equal(g_low, low) and torch.equal(g_inject, inject)


def test_guard_refuses_tm64x2_outside_its_rows() -> None:
    normed, first, rank, c, wi = _build(dict(rows=64), seed=8)
    with validating(op), pytest.raises(AssertionError, match="two 64-row tiles"):
        op(normed, first, rank, c, wi, tm64x2=True)
