# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_decode_mix_fp4_cute entry: cells against the Triton
entry's float64 reference in its band (`mixed` and `inject_out` in `rtol = eps`,
`atol = eps * max|ref|`; the NVFP4 exact on the returned `mixed` outside the
reciprocal tie window), agreement with the Triton entry inside the same band, a
stale-read check under CUDA-graph replay with PDL, the guards."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_mix_fp4 import (  # noqa: E501
    nvfp4_near_tie,
    nvfp4_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_mix_fp4 import (  # noqa: E501
    qwen4_exp_hc_decode_mix_fp4 as triton_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_mix_fp4_cute import (  # noqa: E501
    qwen4_exp_hc_decode_mix_fp4_cute as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_decode_mix_fp4_cute requires a CUDA device"

DEV = "cuda"
HC_COUNT, HIDDEN, RANK, EPS = 4, 2560, 320, 1e-6
UP_RMS, GAMMA_RMS, GAMMA_ABSMAX = 0.0681, 0.477, 6.625


def _band(got, want) -> float:
    eps = torch.finfo(want.dtype).eps
    w = want.double()
    return ((got.double() - w).abs() / (eps * w.abs().max().item() + eps * w.abs())).max().item()


def _compare(out, ref) -> None:
    (mixed, inj), (ref_mixed, ref_inj) = out[:2], ref[:2]
    assert (inj is None) == (ref_inj is None)
    for got, want in [(mixed, ref_mixed)] + ([] if inj is None else [(inj, ref_inj)]):
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
        assert _band(got, want) <= 1.0, f"{_band(got, want):.4g}x the band"


def _unpack(xq):
    return torch.stack([xq & 0xF, xq >> 4], -1).flatten(-2)


def _check_quant(mixed, xq, xsf, gs) -> None:
    """The NVFP4 outputs exact on the returned `mixed`: scales bitwise, e2m1 codes
    bitwise outside the window where the approximate reciprocals may pick either side."""
    want_q, want_sf = nvfp4_reference(mixed, gs)
    assert xq.dtype == torch.uint8 and xq.shape == want_q.shape and xq.is_contiguous()
    assert xsf.dtype == torch.uint8 and xsf.shape == want_sf.shape
    assert torch.equal(xsf, want_sf), f"{(xsf != want_sf).sum().item()} scale bytes differ"
    stray = (_unpack(xq) != _unpack(want_q)) & ~nvfp4_near_tie(mixed, gs)
    assert not stray.any(), f"{stray.sum().item()} e2m1 codes differ outside the tie window"


def _build(spec, seed):
    rows, hc_count, hidden, rank, dtype = 1, HC_COUNT, HIDDEN, RANK, torch.bfloat16
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
    return [part, ss, up, hc_out, gamma, EPS, hc_count, with_inject, gs]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    out = op(*args)
    assert out[0].shape == (1, HIDDEN)
    _compare(out, op.reference(*args))
    _check_quant(out[0], out[2], out[3], args[8])


@pytest.mark.parametrize("with_inject", [True, False])
def test_matches_the_triton_entry(with_inject) -> None:
    """`mixed` / `inject_out` inside the band of the Triton entry's; the NVFP4 of each
    is exact on its own `mixed`, so where the two `mixed` agree the codes agree."""
    args = _build(dict(with_inject=with_inject), seed=77)
    mixed, inj, _, _ = op(*args)
    t_mixed, t_inj, _, _ = triton_op(*args)
    assert _band(mixed, t_mixed) <= 1.0
    if with_inject:
        assert _band(inj, t_inj) <= 1.0
    print(f"\n  mixed bitwise the Triton entry's: {(mixed == t_mixed).float().mean().item():.4f}")


def test_deterministic() -> None:
    args = _build(dict(), seed=3)
    for a, b in zip(op(*args), op(*args)):
        assert torch.equal(a, b)


def test_reads_after_the_wait_under_graph_replay() -> None:
    """Producer kernels rewrite `part`, `ss` and `hc_out` inside the graph right before
    the op; with PDL a read before `griddepcontrol.wait` would see the previous replay's."""
    args = _build(dict(), seed=8)
    targets = (args[0], args[1], args[3])
    sources = [t.clone() for t in targets]

    def step():
        for dst, src in zip(targets, sources):
            dst.copy_(src * 1.0)
        return op(*args)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    for trial in range(3):
        fresh = _build(dict(), seed=100 + trial)
        for src, new in zip(sources, (fresh[0], fresh[1], fresh[3])):
            src.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(graph_out, op(*args)):
            assert torch.equal(a, b), f"replay {trial}"


def test_guards() -> None:
    args = _build(dict(), seed=6)
    two = args[3].expand(2, -1).contiguous()
    with validating(op), pytest.raises(AssertionError):
        op(args[0].expand(-1, 2, -1), args[1].expand(-1, 2), args[2], two, *args[4:])
    contiguous_up = args[2].contiguous()
    with validating(op), pytest.raises(AssertionError):
        op(args[0], args[1], contiguous_up, *args[3:])
