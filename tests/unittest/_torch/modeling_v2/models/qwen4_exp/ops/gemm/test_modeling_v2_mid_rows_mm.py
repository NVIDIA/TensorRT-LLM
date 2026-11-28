# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `mid_rows_mm` (FP8 with the per-row activation quantization, and bf16) against an
fp32 reference over the operands it multiplies, per cell; `fp8_row_quant` bitwise against
torch's e4m3 cast; strided activation rows; CUDA-graph replay with new activations; the inject
mode (`mid_mm_inject`) against the plain GEMM followed by `qwen4_exp_hc_inject_norm`."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.mid_rows_mm import (  # noqa: E501
    AMAX_FLOOR,
    E4M3_MAX,
    INJECT_UNIT_COLS,
    MidWeight,
    Plan,
    fp8_row_quant,
    fp8_row_quant_op,
    mid_mm_inject,
    mid_mm_raw,
    mid_rows_mm_op,
    precompile,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_norm import (  # noqa: E501
    qwen4_exp_hc_inject_norm,
)

assert torch.cuda.is_available(), "mid_rows_mm requires a CUDA device"

DEV = "cuda"


def _weight(n, k, mode, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=DEV) * k**-0.5
    w[:, 7] *= 20.0  # outlier column
    w = w.to(torch.bfloat16)
    if mode == "bf16":
        return w, MidWeight(w.contiguous())
    if mode == "row":
        scale = (w.float().abs().amax(1) / E4M3_MAX).clamp_min(1e-12)
        return w, MidWeight((w.float() / scale[:, None]).to(torch.float8_e4m3fn), scale)
    scale = (w.float().abs().amax() / E4M3_MAX).clamp_min(1e-12).reshape(1)
    return w, MidWeight((w.float() / scale).to(torch.float8_e4m3fn), scale)


def _activation(m, k, seed, pad=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    wide = torch.randn(m, k + pad, generator=g, device=DEV) * 0.7
    wide[:, 3] *= 12.0  # outlier column
    if m > 2:
        wide[1] = 0.0  # an all-zero row
    return wide.to(torch.bfloat16)[:, :k]


def _dequantized(x, mw):
    """The fp32 operands the call multiplies."""
    if not mw.fp8:
        return x.float(), mw.w.float()
    q, s = fp8_row_quant(x)
    wd = mw.w.float() * (mw.scale[:, None] if mw.scale.numel() > 1 else mw.scale)
    return q.float() * s[:, None], wd


def _close(y, x, mw) -> None:
    """Within one bf16 rounding of the fp32 product of the operands (plus fp32 summation
    noise relative to sum |x w|)."""
    xd, wd = _dequantized(x, mw)
    with true_fp32_matmul():
        ref = xd @ wd.t()
        absref = xd.abs() @ wd.abs().t()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    excess = ((y.float() - ref).abs() - 0.5 * ulp).clamp_min(0) / absref.clamp_min(1e-30)
    assert torch.isfinite(y.float()).all()
    assert excess.max().item() < 1e-5, f"error beyond one rounding: {excess.max().item():.2e}"


@pytest.mark.parametrize(
    "cell", mid_rows_mm_op.CELLS, ids=[c.why[:48] for c in mid_rows_mm_op.CELLS]
)
def test_cells(cell) -> None:
    n, k, m, mode = cell.spec["n"], cell.spec["k"], cell.spec["m"], cell.spec["mode"]
    w, mw = _weight(n, k, mode, n + m)
    x = _activation(m, k, m, pad=64 if mode != "bf16" else 0)
    mid_rows_mm_op.is_valid(x, mw)
    y = mid_rows_mm_op(x, mw)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16 and y.is_contiguous()
    _close(y, x, mw)
    if mw.fp8:
        # The FP8 product against the bf16 projection: e4m3 keeps 3 mantissa bits on both
        # operands; per-row activation scales are at least as accurate as one per tensor.
        full = x.float() @ w.float().t()
        assert ((y.float() - full).norm() / full.norm()).item() < 0.06


@pytest.mark.parametrize(
    "cell", fp8_row_quant_op.CELLS, ids=[c.why[:48] for c in fp8_row_quant_op.CELLS]
)
def test_row_quant_is_torchs_cast(cell) -> None:
    m, k, pad = cell.spec["m"], cell.spec["k"], cell.spec.get("pad", 0)
    x = _activation(m, k, 5 + m, pad=pad) * 3
    if m > 4:
        x[4, 9] = 900.0  # a row whose maximum is far above the rest
    q, s = fp8_row_quant_op(x)
    # The kernel divides correctly rounded (torch divides by a scalar as a product with
    # its reciprocal): the quotients in float64, rounded once.
    amax = x.float().abs().amax(1).clamp_min(AMAX_FLOOR).double()
    assert torch.equal(s, (amax / E4M3_MAX).float())
    want = (x.float() * (E4M3_MAX / amax).float()[:, None]).to(torch.float8_e4m3fn)
    assert torch.equal(q.view(torch.uint8), want.view(torch.uint8))


def test_graph_replay_matches_eager() -> None:
    for mode, n, k, m in (("tensor", 16480, 2560, 64), ("bf16", 512, 2560, 200)):
        _, mw = _weight(n, k, mode, 3)
        precompile(mw)
        x = _activation(m, k, 1).contiguous()
        mid_rows_mm_op(x, mw)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = mid_rows_mm_op(x, mw)
        x.copy_(_activation(m, k, 2) * 4)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, mid_rows_mm_op(x, mw))


def _inject_operands(rows, seed):
    """An attention-DP output projection's operands at the checkpoint's shapes, and the site's."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n, k, c = 2560, 6144, 4
    w = (torch.randn(n, k, generator=g, device=DEV) * k**-0.5).to(torch.bfloat16)
    x = torch.randn(rows, k, generator=g, device=DEV).to(torch.bfloat16)
    hc = (torch.randn(rows, c * n, generator=g, device=DEV) * 4).to(torch.bfloat16)
    inject = (torch.rand(rows, c, generator=g, device=DEV) * 2).to(torch.bfloat16)
    norm = (torch.randn(c * n, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    return x, w, hc, inject, norm


#: `(rows, MMA N, plan)`: the output projections' table entries the fold runs, and the plan
#: `DP_SITEB_PLAN32` moves MMA N 32 to.
_INJECT_CASES = (
    (20, 32, Plan(12, 2, 8, 2)),
    (20, 32, Plan(6, 4, 10, 1)),
    (28, 32, Plan(6, 4, 10, 1)),
    (48, 48, Plan(6, 4, 10, 1)),
    (80, 80, Plan(6, 4, 4, 2)),
    (144, 144, Plan(6, 4, 6, 2)),
)


@pytest.mark.parametrize("rows, mpad, plan", _INJECT_CASES)
def test_inject_mode_is_the_inject_kernel_on_the_gemm(rows, mpad, plan) -> None:
    """`hc'` bitwise `qwen4_exp_hc_inject_norm`'s residual of the plain GEMM's output at the same
    plan, `y` bitwise `bf16(hc' * (1 + w))` in fp32, `ss` the fp32 sums of `hc'^2` a 32-column
    unit (to fp32 rounding of the fixed order)."""
    x, w, hc, inject, norm = _inject_operands(rows, rows)
    hc2, y, ss = mid_mm_inject(x, w, plan, hc, inject, norm, mpad=mpad)
    block = mid_mm_raw(x, w, plan, mpad=mpad)
    want, _ = qwen4_exp_hc_inject_norm(hc, block, inject, norm, 1e-6, 4)
    assert torch.equal(hc2, want)
    assert torch.equal(y, (hc2.float() * (1.0 + norm.float())).to(torch.bfloat16))
    ref = hc2.double().pow(2).view(rows, 4, -1, INJECT_UNIT_COLS).sum(-1)
    assert ss.shape == (rows, 4, 2560 // INJECT_UNIT_COLS)
    torch.testing.assert_close(ss.double(), ref, rtol=1e-5, atol=1e-5)


def test_inject_mode_graph_replay_matches_eager() -> None:
    rows, mpad, plan = _INJECT_CASES[1]
    x, w, hc, inject, norm = _inject_operands(rows, 1)
    mid_mm_inject(x, w, plan, hc, inject, norm, mpad=mpad)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mid_mm_inject(x, w, plan, hc, inject, norm, mpad=mpad)
    fresh = _inject_operands(rows, 2)
    for t, f in zip((x, hc, inject), (fresh[0], fresh[2], fresh[3])):
        t.copy_(f)
    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(out, mid_mm_inject(x, w, plan, hc, inject, norm, mpad=mpad)):
        assert torch.equal(got, want)


def test_an_mpad_off_the_epilogue_width_is_refused_before_compile() -> None:
    """`mpad` must be a multiple of the plan's epilogue width `min(epi, mpad)`: 48 at `epi` 32
    is refused by the entry, before an output is allocated or a kernel compiled (the default
    `mma_rows(17)` = 32 is a multiple)."""
    from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm import mid_rows_mm as mod

    compiled = []
    real = mod._compiled_mm
    mod._compiled_mm = lambda *a, **k: compiled.append(a) or real(*a, **k)
    try:
        x = torch.zeros(17, 512, dtype=torch.bfloat16, device=DEV)
        w = torch.zeros(64, 512, dtype=torch.bfloat16, device=DEV)
        with pytest.raises(AssertionError, match="not a multiple of epi 32"):
            mod.mid_mm_raw(x, w, mod.Plan(epi=32), mpad=48)
        assert not compiled
    finally:
        mod._compiled_mm = real
