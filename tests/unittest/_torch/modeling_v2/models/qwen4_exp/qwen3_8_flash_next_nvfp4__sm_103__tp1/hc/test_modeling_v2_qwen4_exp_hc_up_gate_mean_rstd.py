# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_up_gate_mean_rstd entries (bf16 and NVFP4):
`mixed` in the gate band of the float64 reference, the epilogue's fp32 operation
sequence bitwise (gate at zero logits), the NVFP4 outputs against the native-torch
quantization of the returned `mixed` outside the reciprocal tie window, the NVFP4
build's `mixed` bitwise the bf16 build's, the global-scale guard, graph capture.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    Fp4Quantize,
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_gated_residual import (  # noqa: E501
    gate_ratio,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_up_gate_mean_rstd import (  # noqa: E501
    qwen4_exp_hc_up_gate_mean_rstd as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_up_gate_mean_rstd import (  # noqa: E501
    qwen4_exp_hc_up_gate_mean_rstd_fp4 as op4,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_up_gate_mean_rstd requires a CUDA device"

DEV = "cuda"
HC_COUNT = 4
HIDDEN = 2560
RANK = 320
EPS = 1e-6
GAMMA_RMS = 0.477
GAMMA_ABSMAX = 6.625


def _unpack(xq: torch.Tensor) -> torch.Tensor:
    return torch.stack([xq & 0xF, xq >> 4], -1).flatten(-2)


def _check_quant(mixed, xq, xsf, gs) -> None:
    """NVFP4 outputs against the native-torch quantization of the `mixed` this call
    returned: scale bytes equal, e2m1 codes equal except inside the reciprocal tie
    window.
    """
    want_q, want_sf = fp4_quantize.reference(mixed, gs, 16, False, False)
    assert xq.dtype == torch.uint8 and xq.shape == want_q.shape and xq.is_contiguous()
    assert xsf.dtype == torch.uint8 and xsf.shape == want_sf.shape
    assert torch.equal(xsf, want_sf), f"{(xsf != want_sf).sum().item()} scale bytes differ"
    diff = _unpack(xq) != _unpack(want_q)
    stray = diff & ~Fp4Quantize.near_tie(fp4_quantize.scaled(mixed, gs, 16))
    assert not stray.any(), f"{stray.sum().item()} e2m1 codes differ outside the tie window"


def _check_mixed(mixed, ref_mixed) -> None:
    """gate_ratio <= 1, the band `qwen4_exp_hc_up_gate_mean` is certified in."""
    assert mixed.dtype == ref_mixed.dtype and mixed.shape == ref_mixed.shape
    assert mixed.is_contiguous()
    assert gate_ratio(mixed, ref_mixed) <= 1.0, f"{gate_ratio(mixed, ref_mixed):.4g}x the band"


def _build(spec, seed):
    """`(low, up, hc_out, rstd, norm_w1, hc_count, global_scale)`."""
    rows, width = spec["rows"], HC_COUNT * HIDDEN
    g = torch.Generator(device=DEV).manual_seed(seed)
    # This checkpoint's statistics: low = silu(.) of the bottleneck (|x| < ~3), the up
    # weight ~N(0, 0.068) (gate logits within about +-8).
    low = torch.nn.functional.silu(torch.randn(rows, RANK, generator=g, device=DEV) * 1.5).to(
        torch.bfloat16
    )
    up_w = (torch.randn(width, RANK, generator=g, device=DEV) * 0.068).to(torch.bfloat16)
    hc_out = (torch.randn(rows, width, generator=g, device=DEV) * 3.0).to(torch.bfloat16)
    x = hc_out.double().unflatten(-1, (HC_COUNT, HIDDEN))
    rstd = torch.rsqrt(x.pow(2).mean(-1) + EPS).float().contiguous()
    gamma = torch.randn(width, generator=g, device=DEV) * GAMMA_RMS
    gamma[0], gamma[width // 2] = GAMMA_ABSMAX, -GAMMA_ABSMAX
    norm_w1 = (1.0 + gamma.to(torch.bfloat16).float()).contiguous()
    # A calibrated activation global scale: 448 * 6 / amax over a typical batch.
    gs = torch.tensor([448.0 * 6.0 / 4.0], device=DEV, dtype=torch.float32)
    return low, up_w.t(), hc_out, rstd, norm_w1, HC_COUNT, gs


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % (2**31)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell))[:6]
    mixed = op(*args)
    assert mixed.shape == (cell.spec["rows"], HIDDEN)
    _check_mixed(mixed, op.reference(*args))


@pytest.mark.parametrize("cell", op4.CELLS, ids=[c.why[:44] for c in op4.CELLS])
def test_certified_cells_fp4(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op4)
    args = _build(cell.spec, seed=_seed(cell))
    mixed, xq, xsf = op4(*args)
    assert mixed.shape == (cell.spec["rows"], HIDDEN)
    ref_mixed = op4.reference(*args)[0]
    _check_mixed(mixed, ref_mixed)
    _check_quant(mixed, xq, xsf, args[6])


@pytest.mark.parametrize("rows", [5, 300, 4100])
def test_epilogue_operation_sequence_bitwise(rows) -> None:
    """With `low = 0` every logit is 0 and `tanh.approx(0) = 0`, so the gate is the
    folded `rstd / 2C` and `mixed` is the documented fp32 sequence alone:
    `acc = (rstd_0 / 8) * (x_0 * w1_0)`, then `acc = fma(rstd_s / 8, x_s * w1_s, acc)`
    in stream order, one bf16 rounding. Each step is emulated in float64 (products
    of the fp32 operands are exact there) and rounded to fp32.
    """
    low, up, hc_out, rstd, w1, c, _ = _build(dict(rows=rows), seed=200 + rows)
    low = torch.zeros_like(low)
    mixed = op(low, up, hc_out, rstd, w1, c)
    x = hc_out.double().unflatten(-1, (c, HIDDEN))
    xw = (x * w1.double().unflatten(-1, (c, HIDDEN))).float().double()
    hr = (rstd * (0.5 / c)).double()
    acc = (hr[:, :1] * xw[:, 0]).float().double()
    for s in range(1, c):
        acc = (hr[:, s : s + 1] * xw[:, s] + acc).float().double()
    want = acc.float().to(torch.bfloat16)
    assert torch.equal(mixed, want), f"{(mixed != want).sum().item()} elements differ"


@pytest.mark.parametrize("rows", [5, 2048, 4100])
def test_fp4_mixed_is_the_bf16_entry_bitwise(rows) -> None:
    args = _build(dict(rows=rows), seed=100 + rows)
    assert torch.equal(op4(*args)[0], op(*args[:6]))


def test_guards() -> None:
    """The kernel loads `global_scale[0]` only; a per-row scale is refused. rstd must be
    fp32.
    """
    low, up, hc_out, rstd, w1, c, gs = _build(dict(rows=4), seed=4)
    with validating(op4), pytest.raises(AssertionError):
        op4(low, up, hc_out, rstd, w1, c, torch.cat([gs, gs]))
    with validating(op), pytest.raises(AssertionError, match="rstd"):
        op(low, up, hc_out, rstd.to(torch.bfloat16), w1, c)


@pytest.mark.parametrize("fp4", [False, True])
def test_cuda_graph_capture_and_replay(fp4) -> None:
    args = list(_build(dict(rows=8), seed=5))
    if not fp4:
        args = args[:6]
    entry = op4 if fp4 else op
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        entry(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = entry(*args)
    fresh = _build(dict(rows=8), seed=6)
    for i in (0, 2, 3):
        args[i].copy_(fresh[i])
    if fp4:
        args[6].mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    want = entry(*args)
    g_out, want = (g_out, want) if fp4 else ((g_out,), (want,))
    for got, exp in zip(g_out, want):
        assert torch.equal(got, exp)
