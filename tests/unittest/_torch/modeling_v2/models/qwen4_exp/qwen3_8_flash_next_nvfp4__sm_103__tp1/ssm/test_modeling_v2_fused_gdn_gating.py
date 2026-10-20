# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_gdn_gating entry.

Geometry is a Qwen3.8-Flash-Next linear-attention layer: 48 value heads, `a`
bf16, `A_log` / `dt_bias` fp32 drawn across the checkpoint's own ranges.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_gdn_gating import (  # noqa: E501
    A_LOG_RANGE,
    APPROX_ULP,
    BA_WIDTH,
    DT_BIAS_RANGE,
    NUM_V_HEADS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_gdn_gating import (  # noqa: E501
    fused_gdn_gating as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_gdn_gating requires a CUDA device"


#: The smallest softplus `beta` a cell drives; the tail error scales as 1 / beta.
_MIN_CERTIFIED_BETA = 0.5


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Both fp32. `g` multiplies two approximate transcendentals, so the
    relative term is `4 * 2^-22`. The absolute floor is the unstabilised
    softplus's tail error -- about one approximate ulp of the log's
    argument -- scaled by `1 / beta` and by the `exp(A_log)` it multiplies;
    `_compare` cannot see either, so it takes the certified extremes
    (exp(A_log) <= 139.42, beta >= 0.5): atol = 4 * 2^-22 * 139.42 / 0.5 =
    2.66e-04.

    Measured against this gate on 256 checkpoint-range rows: worst error /
    allowance 0.215. Five wrong formulas (dt_bias dropped, sign dropped,
    A_log without exp, exp(-A_log), sigmoid for softplus) land 8.7e+05 to
    1.5e+07 allowances away.
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    atol = 4.0 * APPROX_ULP * math.exp(A_LOG_RANGE[1]) / _MIN_CERTIFIED_BETA
    torch.testing.assert_close(out, ref, rtol=4.0 * APPROX_ULP, atol=atol)


DEV = "cuda"
SOFTPLUS_THRESHOLD = 20.0
#: log(1 + exp(x)) in fp32 is exactly 0 once exp(x) <= 2**-24. Measured: the
#: largest x with g == 0 is -16.635532, the smallest with g != 0 -16.635530.
ZERO_BELOW = -24.0 * math.log(2.0)


def _params(g, num_heads=NUM_V_HEADS, dtype=torch.float32):
    lo, hi = A_LOG_RANGE
    A_log = torch.rand(num_heads, generator=g, device=DEV) * (hi - lo) + lo
    lo, hi = DT_BIAS_RANGE
    dt_bias = torch.rand(num_heads, generator=g, device=DEV) * (hi - lo) + lo
    return A_log.to(dtype), dt_bias.to(dtype)


def _build(spec, seed):
    """`__call__`'s positional arguments for one spec."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows, heads = spec["rows"], spec.get("num_heads", NUM_V_HEADS)
    a_dtype = spec.get("a_dtype", torch.bfloat16)
    layout = spec.get("layout", "contiguous")

    def randn(*shape):
        return (torch.randn(*shape, generator=g, device=DEV) * spec.get("a_scale", 1.0)).to(a_dtype)

    if "fill" in spec:
        a = torch.full((rows, heads), spec["fill"], dtype=a_dtype, device=DEV)
    elif layout == "target_rows":
        # `_gdn_block` splits the [T, 48] projection into prefill and decode rows.
        a = randn(rows + 23, heads)[23:]
        assert a.is_contiguous() and a.storage_offset() == 23 * heads
    elif layout == "ba_column_half":
        a = randn(rows, BA_WIDTH)[:, NUM_V_HEADS:]
        assert a.stride() == (BA_WIDTH, 1)
    elif layout == "ba_row_and_column":
        a = randn(rows + 48, BA_WIDTH)[48:, NUM_V_HEADS:]
        assert a.stride() == (BA_WIDTH, 1)
    else:
        a = randn(rows, heads)
    A_log, dt_bias = _params(g, heads, spec.get("param_dtype", torch.float32))
    if "overlong" in spec:
        pad = torch.full((spec["overlong"],), 9.0, dtype=A_log.dtype, device=DEV)
        A_log, dt_bias = torch.cat([A_log, pad]), torch.cat([dt_bias, pad])
    return A_log, a, dt_bias, spec.get("beta", 1.0), spec.get("threshold", SOFTPLUS_THRESHOLD)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    g = op(*args)
    _compare(g, op.reference(*args))
    assert g.is_contiguous() and torch.isfinite(g).all()
    A_log, a, dt_bias = args[:3]
    if not a.is_contiguous():
        # The row stride is consumed in place, not approximated.
        assert torch.equal(g, op(A_log, a.contiguous(), dt_bias, *args[3:]))
    if "overlong" in cell.spec:
        heads = a.shape[1]
        trimmed = op(A_log[:heads].clone(), a, dt_bias[:heads].clone(), *args[3:])
        assert torch.equal(g, trimmed), "elements past a.shape[1] must not be read"


def test_threshold_branch_is_live_for_this_checkpoint() -> None:
    """dt_bias reaching 19.25 puts part of an ordinary batch past the default
    threshold, so both softplus branches run in one launch."""
    args = _build(dict(rows=4096), seed=106)
    A_log, a, dt_bias = args[:3]
    over = ((a.double() + dt_bias.double()) > SOFTPLUS_THRESHOLD).double().mean().item()
    assert 0.0 < over < 1.0, f"expected a mixed batch, got {over} of elements over the threshold"
    _compare(op(*args), op.reference(*args))


def test_unstabilised_softplus_tail() -> None:
    """The kernel's `log(1 + exp(x))` is exactly 0 below -24 ln 2; the absolute
    error that costs stays inside `_compare`, the relative error reaches 1."""
    heads = 8
    A_log = torch.zeros(heads, device=DEV)  # exp(A_log) == 1, so g == -softplus
    dt_bias = torch.zeros(heads, device=DEV)
    x = torch.linspace(-40.0, 5.0, heads * 8192, device=DEV).view(-1, heads)
    g = op(A_log, x, dt_bias)
    ref = op.reference(A_log, x, dt_bias)
    _compare(g, ref)

    assert (g[x <= ZERO_BELOW - 1e-3] == 0).all()
    assert (g[x >= ZERO_BELOW + 1e-3] != 0).all()
    exact = -torch.log1p(torch.exp(x.double()))
    rel = (g.double() - exact).abs() / exact.abs().clamp_min(1e-300)
    assert float(rel[x <= ZERO_BELOW].max()) == 1.0
    assert float(rel[x >= -9.0].max()) < 1e-3


def test_determinism() -> None:
    args = _build(dict(rows=256), seed=105)
    assert torch.equal(op(*args), op(*args))


def test_gate_discriminates_wrong_variants() -> None:
    """Five plausible mis-implementations, each rejected by `_compare` -- and
    by a margin: each lands at least 1e3 times the allowance away."""
    args = _build(dict(rows=256), seed=108)
    A_log, a, dt_bias = args[:3]
    g = op(*args)
    ref = op.reference(*args)
    _compare(g, ref)

    e = torch.exp(A_log.double())
    x = a.double() + dt_bias.double()
    sp = torch.where(x <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x)), x)
    sp_no_bias = torch.where(
        a.double() <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(a.double())), a.double()
    )
    wrong = {
        "dt_bias dropped": -e * sp_no_bias,
        "leading minus dropped": e * sp,
        "A_log without exp": -A_log.double() * sp,
        "exp(-A_log)": -torch.exp(-A_log.double()) * sp,
        "sigmoid for softplus": -e * torch.sigmoid(x),
    }
    allowance = 4.0 * APPROX_ULP * (math.exp(A_LOG_RANGE[1]) / 0.5 + ref.double().abs())
    correct = float(((g.double() - ref.double()).abs() / allowance).max())
    assert correct < 1.0, correct
    for name, variant in wrong.items():
        with pytest.raises(AssertionError):
            _compare(g, variant.float())
        ratio = float(((variant - ref.double()).abs() / allowance).max())
        print(f"{name}: {ratio:.3e}x the allowance (correct {correct:.3e})")
        assert ratio > 1e3, f"{name} lands only {ratio}x outside the gate"


@pytest.mark.parametrize(
    "shape,message",
    [
        ((2, 4, NUM_V_HEADS), "too many values to unpack"),
        ((NUM_V_HEADS,), "not enough values to unpack"),
    ],
)
def test_op_rejects_a_non_2d_a(shape, message) -> None:
    """Loud, from the op's own `batch, num_heads = a.shape` -- which is why the
    guard reads `shape[-1]` / `stride(-1)` and does not preempt it."""
    A_log, dt_bias = _params(torch.Generator(device=DEV).manual_seed(109))
    a = torch.randn(*shape, device=DEV).to(torch.bfloat16)
    with pytest.raises(ValueError, match=message):
        with validating(op):
            op(A_log, a, dt_bias)


def test_op_rejects_cpu_tensors() -> None:
    A_log, dt_bias = _params(torch.Generator(device=DEV).manual_seed(110))
    a = torch.randn(4, NUM_V_HEADS, device=DEV).to(torch.bfloat16)
    with pytest.raises(ValueError, match="Pointer argument cannot be accessed from Triton"):
        op(A_log.cpu(), a.cpu(), dt_bias.cpu())


def _guard_case(case):
    """`(args, hazard)`: the violating call and what it silently computes instead."""
    g = torch.Generator(device=DEV).manual_seed(111)
    A_log, dt_bias = _params(g)
    a = torch.randn(32, NUM_V_HEADS, generator=g, device=DEV).to(torch.bfloat16)
    if case == "a_head_stride_2":
        wide = torch.randn(32, BA_WIDTH, generator=g, device=DEV).to(torch.bfloat16)
        bad = (A_log, wide[:, ::2], dt_bias)
        # Read as `row * stride(0) + head`: the first 48 columns of each row.
        return bad, (A_log, wide[:, :NUM_V_HEADS], dt_bias)
    wide = torch.randn(2 * NUM_V_HEADS, generator=g, device=DEV)
    if case == "A_log_stride_2":
        return (wide[::2], a, dt_bias), (wide[:NUM_V_HEADS], a, dt_bias)
    if case == "dt_bias_stride_2":
        return (A_log, a, wide[::2]), (A_log, a, wide[:NUM_V_HEADS])
    # Short by 8: a view into a longer buffer, so the over-read lands in known memory.
    if case == "A_log_short":
        return (wide[:40], a, dt_bias), (wide[:NUM_V_HEADS], a, dt_bias)
    return (A_log, a, wide[:40]), (A_log, a, wide[:NUM_V_HEADS])


@pytest.mark.parametrize(
    "case,match",
    [
        ("a_head_stride_2", "head axis must be unit-stride"),
        ("A_log_stride_2", "must be contiguous 1-D"),
        ("dt_bias_stride_2", "must be contiguous 1-D"),
        ("A_log_short", "at least a.shape"),
        ("dt_bias_short", "at least a.shape"),
    ],
)
def test_guard_refuses_what_the_kernel_reads_at_the_wrong_addresses(case, match) -> None:
    """Unguarded, each call returns a finite `g` -- exactly the one for the
    memory the kernel actually addressed, not the tensor it was handed; the
    guard refuses it."""
    bad, addressed = _guard_case(case)
    unguarded = op(*bad)
    assert torch.isfinite(unguarded).all()
    assert torch.equal(unguarded, op(*addressed)), "the hazard this guard covers is gone"
    if "short" not in case:
        with pytest.raises(AssertionError):
            _compare(unguarded, op.reference(*bad))
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(*bad)
