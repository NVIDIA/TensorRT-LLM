# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_decode_mix entry: cells inside
`qwen4_exp_gated_residual`'s band (`rtol = eps`, `atol = eps * max|ref|` of the
activation dtype) against the float64 reference for both `mixed` and
`inject_out`, wrong formulas outside, the stride guards, graph capture."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_mix import (  # noqa: E501
    qwen4_exp_hc_decode_mix as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_decode_mix requires a CUDA device"

DEV = "cuda"
HC_COUNT, HIDDEN, RANK, EPS = 4, 2560, 320, 1e-6
UP_RMS, GAMMA_RMS, GAMMA_ABSMAX = 0.0681, 0.477, 6.625


def _band(got, want) -> float:
    eps = torch.finfo(want.dtype).eps
    w = want.double()
    return ((got.double() - w).abs() / (eps * w.abs().max().item() + eps * w.abs())).max().item()


def _compare(out, ref) -> None:
    (mixed, inj), (ref_mixed, ref_inj) = out, ref
    assert (inj is None) == (ref_inj is None)
    for got, want in [(mixed, ref_mixed)] + ([] if inj is None else [(inj, ref_inj)]):
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
        if got.numel():
            assert _band(got, want) <= 1.0, f"{_band(got, want):.4g}x the band"


def _build(spec, seed):
    rows = spec["rows"]
    hc_count = spec.get("hc_count", HC_COUNT)
    hidden = spec.get("hidden", HIDDEN)
    rank = spec.get("rank", RANK)
    dtype = spec.get("dtype", torch.bfloat16)
    with_inject = spec.get("with_inject", True)
    width = hc_count * hidden
    g = torch.Generator(device=DEV).manual_seed(seed)

    def normal(*shape, scale):
        return torch.randn(*shape, generator=g, device=DEV) * scale

    hc_out = normal(rows, width, scale=3.0).to(dtype)
    if spec.get("layout") == "row_slice":
        buf = normal(2 * rows, width, scale=1.0).to(dtype)
        buf[::2] = hc_out
        hc_out = buf[::2]
    ss = hc_out.double().unflatten(-1, (hc_count, hidden)).pow(2).sum(-1).t().float().contiguous()
    gamma = normal(width, scale=GAMMA_RMS)
    gamma[0], gamma[width // 2] = GAMMA_ABSMAX, -GAMMA_ABSMAX
    gamma = gamma.to(dtype)
    # Per-stream partials sized so the combined logits reach this checkpoint's ~|67|.
    n_pad = -(-(rank + hc_count) // 4) * 4
    part = normal(hc_count, rows, n_pad + 4, scale=60.0)[:, :, :n_pad]
    up = normal(width, rank, scale=UP_RMS).to(dtype).t()
    return part, ss, up, hc_out, gamma, EPS, hc_count, with_inject


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    out = op(*args)
    assert out[0].shape == (cell.spec["rows"], args[3].shape[1] // args[6])
    _compare(out, op.reference(*args))


def test_measured_distance_from_float64() -> None:
    for rows in (1, 16):
        args = _build(dict(rows=rows), seed=rows)
        (m, i), (rm, ri) = op(*args), op.reference(*args)
        print(f"\n  rows {rows}: mixed {_band(m, rm):.4f}x, inject {_band(i, ri):.4f}x the band")


def test_programs_agree_and_repeat() -> None:
    args = _build(dict(rows=4), seed=3)
    a, b = op(*args), op(*args)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_wrong_variants_land_outside_the_band() -> None:
    part, ss, up, hc_out, gamma, eps, c, _ = args = _build(dict(rows=8), seed=7)
    want_mixed, want_inj = op.reference(*args)
    rstd = torch.rsqrt(ss.double().t() / HIDDEN + eps)
    x = hc_out.double().unflatten(-1, (c, HIDDEN))
    g = (1.0 + gamma.double()).unflatten(-1, (c, HIDDEN))
    logits = torch.einsum("ts,stn->tn", rstd, part.double())
    raw_logits = part.double().sum(0)

    def mix(lg, normed, reduce="mean"):
        gate = torch.sigmoid(torch.nn.functional.silu(lg[:, :RANK] / c) @ up.double())
        v = gate.unflatten(-1, (c, HIDDEN)) * normed
        return v.mean(-2) if reduce == "mean" else v.sum(-2)

    normed = x * rstd.unsqueeze(-1) * g
    variants = {
        "sum instead of mean": mix(logits, normed, "sum"),
        "gates the raw stream": mix(logits, x),
        "w instead of 1 + w": mix(logits, x * rstd.unsqueeze(-1) * (g - 1.0)),
        "no rstd on the logits": mix(raw_logits, normed),
        "no /hc_count before silu": mix(logits * c, normed),
    }
    for name, value in variants.items():
        ratio = _band(value.to(torch.bfloat16), want_mixed)
        print(f"\n  mixed {name:<28} {ratio:10.4g}x the band")
        assert ratio > 5.0, name
    zi = logits[:, RANK : RANK + c]
    for name, value in {
        "inject missing the factor of 2": torch.sigmoid(zi / c),
        "inject without rstd": 2.0 * torch.sigmoid(raw_logits[:, RANK : RANK + c] / c),
    }.items():
        ratio = _band(value.to(torch.bfloat16), want_inj)
        print(f"  {name:<34} {ratio:10.4g}x the band")
        assert ratio > 5.0, name


def test_guard_refuses_a_strided_ss() -> None:
    """`ss` laid out `[T, C]` and viewed as `[C, T]` is read row-major and
    answers plausibly wrong unguarded; the guard refuses it."""
    args = list(_build(dict(rows=8), seed=6))
    want = op.reference(*args)[0]
    args[1] = args[1].t().contiguous().t()
    assert _band(op(*args)[0], want) > 1.0, "a transposed ss should answer outside the band"
    with validating(op), pytest.raises(AssertionError):
        op(*args)


def test_cuda_graph_capture_and_replay() -> None:
    args = list(_build(dict(rows=1), seed=8))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = op(*args)
    fresh = _build(dict(rows=1), seed=18)
    for k in (0, 1, 3):
        args[k].copy_(fresh[k])
    graph.replay()
    torch.cuda.synchronize()
    eager = op(*args)
    assert torch.equal(graph_out[0], eager[0]) and torch.equal(graph_out[1], eager[1])
