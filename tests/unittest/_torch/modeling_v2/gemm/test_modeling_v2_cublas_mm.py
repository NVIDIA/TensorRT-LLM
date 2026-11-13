# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cublas_mm catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import (  # noqa: E501
    cublas_mm as op,  # noqa: E501
)

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — needs the path declared above
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "cublas_mm requires a CUDA device"


def _band(out_dtype: torch.dtype) -> tuple[float, float, float]:
    """(element_ulp, rms_ulp, ulp) for an output of `out_dtype`; derived in `_compare`."""
    if out_dtype is torch.float32:
        return 512.0, 256.0, ULP[torch.float32]
    return 4.0, 2.0, ULP[torch.bfloat16]


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference, in the output's own ulp.

    Kernel and reference consume bit-identical operands and differ only in
    the order cuBLAS and torch sum over K. Over random operands a few
    outputs land near zero, where that reordering shows up as a relative
    difference of 10 on an absolute difference of 1.5e-4 -- which is what
    torch's default band rejects. So both gates are against the row's scale
    (`assert_within_ulp`): element-wise against each row's largest magnitude,
    and the relative RMS.

    bf16 out: 4 ulp element-wise, 2 ulp of relative RMS. The two fp32 sums
    differ in their last bits, and where that straddles a bf16 rounding
    boundary they round to neighbouring values: one bf16 spacing, which is
    between 1 and 2 ulp of the row's scale as `ULP` counts it. Measured worst
    case over every m in 1..9216 at k 6144, n 576 / 512 / 160: 1.98 ulp
    element-wise and 0.053 RMS. A wrong product -- the weight's two K halves
    swapped -- lands at 310-640 ulp element-wise and 330-370 RMS. An error
    below one bf16 spacing is below what a bf16 result can show: dropping one
    of the 6144 K terms lands at 1.6-15 ulp element-wise, 1.5-6.8 RMS. Over
    every m in 1..32768 the worst case is 1.97 and 0.081; at a 32768-row call
    the swapped halves land at 596-688 and 362, one dropped K term at 12.6-13.8
    and 3.2-3.3.

    fp32 out (the routers' logits): 512 ulp element-wise, 256 RMS, in fp32's
    ulp. bf16 products are exact in fp32 and the sum is fp32, so the result
    is held to fp32's resolution: scored in bf16 ulp instead, the same logits
    rounded through bf16 and widened back (0.6-1.0 bf16 ulp) would pass for
    them. Measured worst case over every m in 1..9216 at k 6144, n 128: 122
    ulp element-wise and 57 RMS; rounded through bf16 they land at 2.2e4-3.3e4
    and 1.4e4-1.6e4, 42x and 54x outside the gate. Over every m in 1..32768:
    121 and 57; at a 32768-row call, 124 and 57, and rounded through bf16
    3.3e4 and 1.4e4. Over the gathered multiples of 4 in 4..131072, with zero
    padding rows: 132.9 and 57.2; at a 131072-row call on other operands,
    126.5 and 57.1, where rounded through bf16 lands at 3.25e4 and 1.39e4,
    one dropped K term at 5.45e5 and 1.05e5, and the weight's K halves
    swapped at 2.41e7 and 1.19e7.
    """
    element_ulp, rms_ulp, ulp = _band(out.dtype)
    assert_within_ulp(out, ref, element_ulp=element_ulp, rms_ulp=rms_ulp, ulp=ulp)


def _padding_rows(t: torch.Tensor, period: int) -> torch.Tensor:
    """The zero padding rows of a `zero_row_period` cell's operand or output: every
    `period`-th row (i % period == period - 1), a view.

    At period 3, every tile of every kernel the heuristic picks holds some,
    whatever its extent over the rows. Where they sit in a gathered step --
    the tail of each rank's T_max block -- does not matter to a GEMM, whose
    rows are independent; the entry's note records the whole-block skew too.
    """
    return t[period - 1 :: period]


def _padding_logits_not_zero(out: torch.Tensor, period: int) -> torch.Tensor:
    """On the device: whether any padding row's output is anything but +0.0.

    Read as integer bits, since -0.0 compares equal to +0.0 and only the
    bits tell them apart.
    """
    bits = torch.int32 if out.element_size() == 4 else torch.int16
    return _padding_rows(out, period).view(bits).any()


def _build(spec, seed, m=None):
    """Materialize one cell's operands, at `m` rows or the cell's largest.

    `mat_b` is built as [N, K] and transposed, because that is the one thing a
    spec cannot say: the op reads it column-major, and a stored weight's `.t()`
    already is that, which is why no transpose kernel runs in a target.
    """
    if m is None:
        m = spec["m_range"][1] if "m_range" in spec else spec["m"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    k, n, dt = spec["k"], spec["n"], spec["dtype"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(dt)
    if "zero_row_period" in spec:
        _padding_rows(mat_a, spec["zero_row_period"]).zero_()
    mat_b = torch.randn(n, k, generator=g, device="cuda").to(dt).t()
    out_dtype = spec["out_dtype"]
    bias = None
    if spec["bias"]:
        bias = torch.randn(n, generator=g, device="cuda").to(out_dtype or dt)
    return mat_a, mat_b, bias, out_dtype


def _sweep(spec, seed) -> None:
    """Every m of `m_range` (every `m_step`-th one when the cell names a step), each
    its own call, on the leading m rows of one operand set.

    The reference is taken once, at the top of the range, and sliced: row i of
    a matmul reads only row i of `mat_a`, so its leading m rows are the
    reference for an m-row call. The gates are `assert_within_ulp`'s two, per
    m, evaluated on the device so the sweep pays one synchronization rather
    than one per m.

    Per m, the difference is formed once, in fp32, and read twice by two
    reductions: each row's largest error and the error's sum of squares. That
    sum accumulates in fp32, which resolves it to about 1e-6 of itself, far
    inside either gate; the reference's own sum of squares is fp64.

    A `zero_row_period` cell also holds every m's padding rows to +0.0 bit
    for bit, which the band cannot see: their reference is +0.0, so -0.0
    sits at zero distance from it.
    """
    lo, hi = spec["m_range"]
    step = spec.get("m_step", 1)
    ms = range(-(-lo // step) * step, hi + 1, step)
    period = spec.get("zero_row_period")
    mat_a, mat_b, bias, out_dtype = _build(spec, seed)
    want_dtype = out_dtype or mat_a.dtype
    ref = op.reference(mat_a, mat_b, bias, out_dtype).float()
    row_scale = ref.abs().amax(dim=1).clamp_min(1e-9)
    ref_sq = ref.double().pow(2).sum(dim=1).cumsum(0)
    element = torch.empty(len(ms), device="cuda")
    diff_sq = torch.empty(len(ms), device="cuda")
    not_zero = torch.zeros(len(ms), dtype=torch.bool, device="cuda")
    for i, m in enumerate(ms):
        out = op(mat_a[:m], mat_b, bias, out_dtype)
        assert out.shape == (m, mat_b.shape[1]) and out.dtype is want_dtype
        if period:
            not_zero[i] = _padding_logits_not_zero(out, period)
        diff = torch.sub(out, ref[:m])
        element[i] = (torch.linalg.vector_norm(diff, float("inf"), dim=1) / row_scale[:m]).amax()
        diff_sq[i] = torch.linalg.vector_norm(diff).square()
    rms = (diff_sq / ref_sq[ms.start - 1 : hi : step].clamp_min(1e-18)).sqrt()
    element_ulp, rms_ulp, ulp = _band(want_dtype)
    worst = int(element.argmax())
    assert element[worst] <= element_ulp * ulp, (
        f"m={ms[worst]}: {element[worst].item() / ulp:.3f} ulp element-wise exceeds {element_ulp}"
    )
    worst = int(rms.argmax())
    assert rms[worst] <= rms_ulp * ulp, (
        f"m={ms[worst]}: relative RMS {rms[worst].item() / ulp:.3f} ulp exceeds {rms_ulp}"
    )
    bad = not_zero.nonzero()
    assert bad.numel() == 0, f"m={ms[int(bad[0])]}: a zero padding row's output is not +0.0"


def _replays(spec, seed) -> None:
    """Each captured batch in its own CUDA graph, replayed on new data.

    The replay has to equal an eager call on the same data bit for bit -- a
    graph that froze something of the capture would differ -- and is gated
    against the reference like any other output. Each capture is preceded by
    one call on the capture stream, as an engine's warm-up is: the thop
    caller creates its cuBLASLt workspace on a stream's first call. A
    `zero_row_period` cell's new data carries its padding rows too, and the
    replay's logits for them are +0.0 bit for bit.
    """
    stream = torch.cuda.Stream()
    g = torch.Generator(device="cuda").manual_seed(seed)
    period = spec.get("zero_row_period")
    for m in spec["graph_m"]:
        mat_a, mat_b, bias, out_dtype = _build(spec, seed + m, m=m)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(mat_a, mat_b, bias, out_dtype)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = op(mat_a, mat_b, bias, out_dtype)
        fresh = torch.randn(mat_a.shape, generator=g, device="cuda").to(mat_a.dtype)
        if period:
            _padding_rows(fresh, period).zero_()
        mat_a.copy_(fresh)
        graph.replay()
        eager = op(fresh, mat_b, bias, out_dtype)
        assert torch.equal(out, eager), f"m={m}: graph replay differs from eager"
        if period:
            assert not _padding_logits_not_zero(out, period), f"m={m}: padding row not +0.0"
        _compare(out, op.reference(fresh, mat_b, bias, out_dtype))


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS],
    ids=[c.why[:44] for c in op.CELLS],
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    if "m_range" in cell.spec:
        _sweep(cell.spec, seed)
        return
    if "graph_m" in cell.spec:
        _replays(cell.spec, seed)
        return
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed)
    out = op(mat_a, mat_b, bias, out_dtype)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype is (out_dtype or mat_a.dtype)
    _compare(out, op.reference(mat_a, mat_b, bias, out_dtype))


def test_the_fp32_gate_refuses_logits_rounded_through_bf16() -> None:
    """The routers ask for fp32 logits; a bf16 result widened back is not one."""
    spec = dict(m=16, k=6144, n=128, dtype=torch.bfloat16, bias=False, out_dtype=torch.float32)
    mat_a, mat_b, bias, out_dtype = _build(spec, seed=7)
    out = op(mat_a, mat_b, bias, out_dtype)
    ref = op.reference(mat_a, mat_b, bias, out_dtype)
    _compare(out, ref)
    with pytest.raises(AssertionError):
        _compare(out.to(torch.bfloat16).to(torch.float32), ref)


def test_the_padding_gate_refuses_a_negative_zero() -> None:
    """The padding rows' logits are held to +0.0 by their bits; -0.0 passes the band."""
    spec = dict(
        m=64,
        k=6144,
        n=128,
        dtype=torch.bfloat16,
        bias=False,
        out_dtype=torch.float32,
        zero_row_period=3,
    )
    mat_a, mat_b, bias, out_dtype = _build(spec, seed=11)
    out = op(mat_a, mat_b, bias, out_dtype)
    assert not _padding_logits_not_zero(out, 3)
    flipped = out.clone()
    _padding_rows(flipped, 3).neg_()
    _compare(flipped, op.reference(mat_a, mat_b, bias, out_dtype))
    assert _padding_logits_not_zero(flipped, 3)


def _ok(dtype=torch.bfloat16):
    a = torch.randn(8, 64, dtype=dtype, device="cuda")
    b = torch.randn(32, 64, dtype=dtype, device="cuda").t()
    return a, b


def test_guard_refuses_a_row_major_mat_b() -> None:
    """Read column-major regardless; a row-major b is a different product."""
    a, _ = _ok()
    b = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="column-major"):
        with validating(op):
            op(a, b)


def test_guard_refuses_a_strided_mat_a() -> None:
    a = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")[:, :64]
    _, b = _ok()
    with pytest.raises(AssertionError, match="row-major"):
        with validating(op):
            op(a, b)


def test_guard_refuses_a_bias_in_the_wrong_dtype() -> None:
    """Accepted by the op, and the result is silently wrong."""
    a, b = _ok()
    bias = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="bias dtype"):
        with validating(op):
            op(a, b, bias)


def test_guard_refuses_a_bias_on_fp32_inputs() -> None:
    """The op takes it and never adds it."""
    a, b = _ok(torch.float32)
    bias = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="silently ignored"):
        with validating(op):
            op(a, b, bias)
