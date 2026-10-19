# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_decode_tc_first entry: cells in the band against
the float64 reference, wrong formulas outside, determinism, the guard, graph
capture, and the decode mixer built on it against the gated-residual float64
reference."""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_tc_first import (  # noqa: E501
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
        if got.numel():
            assert _ratio(got, want) <= 1.0, f"{_ratio(got, want):.4g}x the band"


def _build(spec, seed):
    hc, hidden = spec.get("hc_count", 4), spec.get("hidden", 2560)
    rows, width, inject = spec["rows"], hc * hidden, spec.get("inject", True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    # This checkpoint's layer statistics: down ~N(0, 0.0296), block_inject ~N(0, 0.0107),
    # the normed stream unit-RMS per stream times (1 + gamma), gamma ~N(0, 0.477).
    down = torch.randn(RANK, width, generator=g, device=DEV) * 0.0296
    parts = [down] + ([torch.randn(hc, width, generator=g, device=DEV) * 0.0107] if inject else [])
    w = torch.cat(parts)
    n_rows = spec.get("n_rows", 384)
    if n_rows > w.shape[0]:
        w = torch.cat([w, torch.zeros(n_rows - w.shape[0], width, device=DEV)])
    gamma = torch.randn(width, generator=g, device=DEV) * 0.477
    x = torch.randn(rows, hc, hidden, generator=g, device=DEV)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = (x.flatten(-2) * (1.0 + gamma)).to(torch.bfloat16)
    return normed, w.to(torch.bfloat16).t(), RANK, hc, inject


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    low, inject = op(*args)
    assert low.shape == (cell.spec["rows"], RANK)
    _compare((low, inject), op.reference(*args))


def test_deterministic() -> None:
    args = _build(dict(rows=128), seed=7)
    a, b = op(*args), op(*args)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_wrong_variants_land_outside_the_band() -> None:
    normed, first, rank, c, wi = _build(dict(rows=64), seed=3)
    want_low, want_inject = op.reference(normed, first, rank, c, wi)
    got = op(normed, first, rank, c, wi)
    lg = normed.double() @ first.double()
    for name, value, want in (
        ("silu without /hc_count", F.silu(lg[:, :rank]), want_low),
        ("sigmoid instead of silu", torch.sigmoid(lg[:, :rank] / c), want_low),
        ("columns shifted by one", F.silu(lg[:, 1 : rank + 1] / c), want_low),
        ("inject missing the 2", torch.sigmoid(lg[:, rank : rank + c] / c), want_inject),
        (
            "inject from the padding",
            2 * torch.sigmoid(lg[:, rank + c : rank + 2 * c] / c),
            want_inject,
        ),
    ):
        ratio = _ratio(value.to(want.dtype), want)
        print(f"\n  {name:<28} {ratio:10.4g}x the band (op: {_ratio(got[0], want_low):.3g}x)")
        assert ratio > 5.0, name


def test_guard_refuses_a_contiguous_weight() -> None:
    normed, first, rank, c, wi = _build(dict(rows=4), seed=4)
    with validating(op), pytest.raises(AssertionError):
        op(normed, first.contiguous(), rank, c, wi)


def test_guard_refuses_more_than_one_tile_of_rows() -> None:
    normed, first, rank, c, wi = _build(dict(rows=129), seed=4)
    with validating(op), pytest.raises(AssertionError):
        op(normed, first, rank, c, wi)


def test_cuda_graph_capture_and_replay() -> None:
    normed, first, rank, c, wi = _build(dict(rows=8), seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(normed, first, rank, c, wi)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_low, g_inject = op(normed, first, rank, c, wi)
    fresh = _build(dict(rows=8), seed=6)[0]
    normed.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    low, inject = op(fresh, first, rank, c, wi)
    assert torch.equal(g_low, low) and torch.equal(g_inject, inject)


@pytest.mark.parametrize("rows", [2, 33, 128])
@pytest.mark.parametrize("with_inject", [True, False])
def test_mixer_matches_the_gated_residual_reference(rows, with_inject) -> None:
    """The decode mixer as `DecodeTarget._hc_mixer` runs it from 2 rows -- this op,
    then `qwen4_exp_hc_up_gate_mean` -- against `qwen4_exp_gated_residual.reference`
    (float64 from `normed`) in the band that entry certifies its own op in, beside
    the cuBLAS + `qwen4_exp_hc_lowrank_act` mixer it replaces."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_gated_residual import (  # noqa: E501
        gate_ratio,
        qwen4_exp_gated_residual,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_lowrank_act import (  # noqa: E501
        qwen4_exp_hc_lowrank_act,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_up_gate_mean import (  # noqa: E501
        qwen4_exp_hc_up_gate_mean,
    )

    hc, hidden, eps = 4, 2560, 1e-6
    width = hc * hidden
    g = torch.Generator(device=DEV).manual_seed(rows)

    def normal(*shape, scale):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(torch.bfloat16)

    # This checkpoint's layer-0 hyper-connection element statistics.
    down, up = normal(RANK, width, scale=0.0296), normal(width, RANK, scale=0.0681)
    inject_w, gamma = normal(hc, width, scale=0.0107), normal(width, scale=0.477)
    gamma[0], gamma[width // 2] = 6.625, -6.625
    x = torch.randn(rows, hc, hidden, generator=g, device=DEV)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    normed = (x.flatten(-2) * (1.0 + gamma.float())).to(torch.bfloat16)
    first = torch.cat([down, inject_w], 0) if with_inject else down
    first = torch.cat(
        [first, torch.zeros(384 - first.shape[0], width, device=DEV, dtype=first.dtype)]
    )
    low, inject = op(normed, first.t(), RANK, hc, with_inject)
    mixed = qwen4_exp_hc_up_gate_mean(low, up.t(), normed, hc)
    want = qwen4_exp_gated_residual.reference(
        normed, down, up, inject_w if with_inject else None, hc
    )
    assert gate_ratio(mixed, want[0]) <= 1.0, f"mixed {gate_ratio(mixed, want[0]):.3f}x the band"
    if with_inject:
        assert gate_ratio(inject, want[1]) <= 1.0, f"inject {gate_ratio(inject, want[1]):.3f}x"
    old_low, _ = qwen4_exp_hc_lowrank_act(
        cublas_mm(normed, first.t(), None, torch.float32), RANK, hc, with_inject, torch.bfloat16
    )
    prev = qwen4_exp_hc_up_gate_mean(old_low, up.t(), normed, hc)
    print(
        f"\n  rows {rows}: tc first {gate_ratio(mixed, want[0]):.3f}x band, "
        f"cublas + lowrank_act {gate_ratio(prev, want[0]):.3f}x"
    )
