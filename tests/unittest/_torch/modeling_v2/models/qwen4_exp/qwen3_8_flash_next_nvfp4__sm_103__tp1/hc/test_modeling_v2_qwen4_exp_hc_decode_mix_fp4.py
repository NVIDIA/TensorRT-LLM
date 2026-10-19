# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_decode_mix_fp4 entry: `mixed` and `inject_out`
inside `qwen4_exp_hc_decode_mix`'s band against the float64 reference, the
NVFP4 outputs exact on the returned `mixed` (outside the reciprocal tie window)
and bitwise against `qwen4_exp_hc_decode_mix` + `torch.ops.trtllm.fp4_quantize`,
guards, graph capture."""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_mix import (  # noqa: E501
    qwen4_exp_hc_decode_mix,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_mix_fp4 import (  # noqa: E501
    nvfp4_near_tie,
    nvfp4_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_mix_fp4 import (  # noqa: E501
    qwen4_exp_hc_decode_mix_fp4 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_decode_mix_fp4 requires a CUDA device"

DEV = "cuda"
HC_COUNT, HIDDEN, RANK, EPS = 4, 2560, 320, 1e-6
UP_RMS, GAMMA_RMS, GAMMA_ABSMAX = 0.0681, 0.477, 6.625


def _band(got, want) -> float:
    eps = torch.finfo(want.dtype).eps
    w = want.double()
    return ((got.double() - w).abs() / (eps * w.abs().max().item() + eps * w.abs())).max().item()


def _unpack(xq: torch.Tensor) -> torch.Tensor:
    return torch.stack([xq & 0xF, xq >> 4], -1).flatten(-2)


def _check_quant(mixed, xq, xsf, gs) -> None:
    want_q, want_sf = nvfp4_reference(mixed, gs)
    assert xq.dtype == torch.uint8 and xq.shape == want_q.shape and xq.is_contiguous()
    assert xsf.dtype == torch.uint8 and xsf.shape == want_sf.shape
    assert torch.equal(xsf, want_sf), f"{(xsf != want_sf).sum().item()} scale bytes differ"
    stray = (_unpack(xq) != _unpack(want_q)) & ~nvfp4_near_tie(mixed, gs)
    assert not stray.any(), f"{stray.sum().item()} e2m1 codes differ outside the tie window"


def _compare(out, ref, gs) -> None:
    """`mixed` / `inject_out` in `rtol = eps`, `atol = eps * max|ref|` (the band
    `qwen4_exp_hc_decode_mix` is certified in); NVFP4 exact on the returned
    `mixed`, and equal to the reference's on 16-blocks whose `mixed` matches."""
    (mixed, inj, xq, xsf), (ref_mixed, ref_inj, ref_q, ref_sf) = out, ref
    assert (inj is None) == (ref_inj is None)
    for got, want in [(mixed, ref_mixed)] + ([] if inj is None else [(inj, ref_inj)]):
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
        if got.numel():
            assert _band(got, want) <= 1.0, f"{_band(got, want):.4g}x the band"
    if mixed.numel() == 0:
        assert xq.numel() == 0 and xsf.numel() == 0
        return
    _check_quant(mixed, xq, xsf, gs)
    rows = mixed.shape[0]
    same = (mixed.view(rows, -1, 16) == ref_mixed.view(rows, -1, 16)).all(-1)
    assert torch.equal(xsf.view(rows, -1)[same], ref_sf.view(rows, -1)[same])
    q, rq = _unpack(xq).view(rows, -1, 16), _unpack(ref_q).view(rows, -1, 16)
    tie = nvfp4_near_tie(ref_mixed, gs).view(rows, -1, 16)
    assert not ((q != rq) & ~tie)[same].any()


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
    n_pad = -(-(rank + hc_count) // 4) * 4
    part = normal(hc_count, rows, n_pad + 4, scale=60.0)[:, :, :n_pad]
    up = normal(width, rank, scale=UP_RMS).to(dtype).t()
    # A calibrated activation global scale: 448 * 6 / amax of the mix (~|8|).
    gs = torch.tensor([448.0 * 6.0 / 8.0], device=DEV, dtype=torch.float32)
    return part, ss, up, hc_out, gamma, EPS, hc_count, with_inject, gs


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    out = op(*args)
    assert out[0].shape == (cell.spec["rows"], args[3].shape[1] // args[6])
    _compare(out, op.reference(*args), args[8])


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_bitwise_the_composition_it_replaces(cell) -> None:
    """`qwen4_exp_hc_decode_mix` then `torch.ops.trtllm.fp4_quantize` (linear
    scales). The NVFP4 outputs are bitwise `fp4_quantize` of the returned
    `mixed`, ties included; `inject_out` is bitwise `qwen4_exp_hc_decode_mix`'s.
    `mixed` is held to `qwen4_exp_hc_decode_mix`'s within the band: the
    16-column program tile reduces the up projection in another order than the
    8-column one, so a few elements in ten thousand round the other way
    (measured: 0..5 of 10240 elements at 1..4 rows)."""
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    mixed, inj, xq, xsf = op(*args)
    old_mixed, old_inj = qwen4_exp_hc_decode_mix(*args[:8])
    assert (inj is None and old_inj is None) or torch.equal(inj, old_inj)
    if mixed.numel():
        diff = (mixed != old_mixed).sum().item()
        print(
            f"\n  {mixed.dtype}: {diff} elements differ from decode_mix, "
            f"{_band(mixed, old_mixed):.4f}x the band"
        )
        assert _band(mixed, old_mixed) <= 1.0
    if mixed.shape[0] == 0:
        return
    old_q, old_sf = torch.ops.trtllm.fp4_quantize(mixed, args[8], 16, False, False)
    assert torch.equal(xq, old_q.view(torch.uint8))
    assert torch.equal(xsf, old_sf.view(torch.uint8).reshape(-1))


def test_wrong_recipes_fail_the_quant_check() -> None:
    """Each plausible wrong layout or recipe is refused by `_check_quant`."""
    args = _build(dict(rows=16), seed=3)
    mixed, _, xq, xsf = op(*args)
    gs = args[8]
    _check_quant(mixed, xq, xsf, gs)
    swapped = ((xq & 0xF) << 4) | (xq >> 4)
    shifted = torch.roll(xsf.view(16, -1), 1, dims=1).reshape(-1)
    half_gs = nvfp4_reference(mixed, gs * 0.5)
    for name, bad in (
        ("nibbles swapped", (mixed, swapped, xsf)),
        ("scales one block off", (mixed, xq, shifted)),
        ("global scale halved", (mixed,) + half_gs),
    ):
        with pytest.raises(AssertionError):
            _check_quant(*bad, gs)
        print(f"\n  {name:<24} refused")


def test_guards() -> None:
    args = list(_build(dict(rows=4), seed=4))
    with validating(op), pytest.raises(AssertionError):
        op(*args[:8], torch.cat([args[8], args[8]]))
    bad = list(_build(dict(rows=2, hidden=136, rank=32), seed=5))
    with validating(op), pytest.raises(AssertionError):
        op(*bad)


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
    args[8].mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    eager = op(*args)
    for got, want in zip(graph_out, eager):
        assert torch.equal(got, want)
