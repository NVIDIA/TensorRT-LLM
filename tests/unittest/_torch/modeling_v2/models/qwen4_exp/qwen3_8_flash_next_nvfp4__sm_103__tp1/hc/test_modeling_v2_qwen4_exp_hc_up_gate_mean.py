# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_up_gate_mean entry: cells in the band
`qwen4_exp_gated_residual`'s test derives for `mixed`, wrong formulas outside,
the guard, graph capture, and the mixer as the forward runs it against the
gated-residual float64 reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_gated_residual import (  # noqa: E501
    gate_ratio,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_up_gate_mean import (  # noqa: E501
    qwen4_exp_hc_up_gate_mean as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_up_gate_mean requires a CUDA device"

DEV = "cuda"


def _compare(out, ref) -> None:
    """`rtol = eps`, `atol = eps * max|ref|` (`gate_ratio <= 1`), the band
    `qwen4_exp_gated_residual` derives for `mixed`."""
    assert out.dtype == ref.dtype and out.shape == ref.shape and out.is_contiguous()
    if out.numel():
        assert gate_ratio(out, ref) <= 1.0, f"{gate_ratio(out, ref):.4g}x the band"


def _build(spec, seed):
    hc, hidden, rank = spec.get("hc_count", 4), spec.get("hidden", 2560), spec.get("rank", 320)
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows, width = spec["rows"], hc * hidden
    # This checkpoint's statistics: low = silu(.) of the bottleneck (|x| < ~3), the up
    # weight ~N(0, 0.068) (gate logits within about +-8), the normed stream ~+-30.
    low = torch.nn.functional.silu(torch.randn(rows, rank, generator=g, device=DEV) * 1.5).to(
        torch.bfloat16
    )
    up_w = (torch.randn(width, rank, generator=g, device=DEV) * 0.068).to(torch.bfloat16)
    normed = (torch.randn(rows, width, generator=g, device=DEV) * 6).to(torch.bfloat16)
    return low, up_w.t(), normed, hc


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    mixed = op(*args)
    assert mixed.shape == (cell.spec["rows"], cell.spec.get("hidden", 2560))
    _compare(mixed, op.reference(*args))


def test_wrong_variants_land_outside_the_band() -> None:
    low, up, normed, c = _build(dict(rows=128), seed=3)
    want = op.reference(low, up, normed, c)
    got = op(low, up, normed, c)
    h = normed.shape[1] // c
    logits = (low.double() @ up.double()).unflatten(-1, (c, h))
    x = normed.double().unflatten(-1, (c, h))
    g = torch.sigmoid(logits)
    for name, value in (
        ("sum instead of mean", (g * x).sum(-2)),
        ("no sigmoid", (logits * x).mean(-2)),
        ("stream-minor packing", (g.transpose(-1, -2).reshape(-1, c, h) * x).mean(-2)),
        ("ungated mean", x.mean(-2)),
    ):
        ratio = gate_ratio(value.to(want.dtype), want)
        print(f"\n  {name:<24} {ratio:10.4g}x the band (op: {gate_ratio(got, want):.3g}x)")
        assert ratio > 5.0, name


def test_guard_refuses_a_contiguous_up() -> None:
    low, up, normed, c = _build(dict(rows=4), seed=4)
    with validating(op), pytest.raises(AssertionError):
        op(low, up.contiguous(), normed, c)


def test_cuda_graph_capture_and_replay() -> None:
    low, up, normed, c = _build(dict(rows=8), seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(low, up, normed, c)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_mixed = op(low, up, normed, c)
    fresh = _build(dict(rows=8), seed=6)
    low.copy_(fresh[0])
    normed.copy_(fresh[2])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(g_mixed, op(fresh[0], up, fresh[2], c))


@pytest.mark.parametrize("rows", [1, 33, 128, 2048, 8192, 16384])
@pytest.mark.parametrize("with_inject", [True, False])
def test_mixer_matches_the_gated_residual_reference(rows, with_inject) -> None:
    """The mixer as the forward runs it -- `cublas_mm` over the zero-padded
    `[down; inject]` (fp32 out), `qwen4_exp_hc_lowrank_act`, this op -- against
    `qwen4_exp_gated_residual.reference` (float64 from `normed`), in the band
    that entry certifies its own op in."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_gated_residual import (  # noqa: E501
        qwen4_exp_gated_residual,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_lowrank_act import (  # noqa: E501
        qwen4_exp_hc_lowrank_act,
    )

    hc, hidden, rank, eps = 4, 2560, 320, 1e-6
    width = hc * hidden
    g = torch.Generator(device=DEV).manual_seed(rows)

    def normal(*shape, scale):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(torch.bfloat16)

    # This checkpoint's layer-0 hyper-connection element statistics.
    down, up = normal(rank, width, scale=0.0296), normal(width, rank, scale=0.0681)
    inject_w, gamma = normal(hc, width, scale=0.0107), normal(width, scale=0.477)
    gamma[0], gamma[width // 2] = 6.625, -6.625
    x = torch.randn(rows, hc, hidden, generator=g, device=DEV)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    normed = (x.flatten(-2) * (1.0 + gamma.float())).to(torch.bfloat16)
    first = torch.cat([down, inject_w], 0) if with_inject else down
    first = torch.cat(
        [first, torch.zeros(384 - first.shape[0], width, device=DEV, dtype=first.dtype)]
    )
    low, _ = qwen4_exp_hc_lowrank_act(
        cublas_mm(normed, first.t(), None, torch.float32), rank, hc, with_inject, torch.bfloat16
    )
    mixed = op(low, up.t(), normed, hc)
    want = qwen4_exp_gated_residual.reference(
        normed, down, up, inject_w if with_inject else None, hc
    )
    _compare(mixed, want[0])
    print(f"\n  rows {rows}: fused {gate_ratio(mixed, want[0]):.3f}x band")
