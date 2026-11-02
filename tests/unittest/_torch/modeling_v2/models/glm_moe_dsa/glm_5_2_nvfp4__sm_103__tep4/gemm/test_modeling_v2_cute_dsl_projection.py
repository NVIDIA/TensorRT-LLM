# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_projection entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_projection runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_projection import (  # noqa: E402, E501
    PREFILL_MAX_ROWS,
    SHARED_DOWN_PREFILL_MIN_ROWS,
    PrefillProjectionRunner,
    ProjectionRunner,
    SharedDownPrefillRunner,
    build,
    build_prefill,
    build_shared_down_prefill,
    shape_of,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_projection import (  # noqa: E402, E501
    cute_dsl_projection as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

#: One unit of a TF32 operand's 10-bit mantissa.
TF32_ULP = 2.0**-10
#: One unit of fp32's mantissa.
FP32_ULP = 2.0**-23


def _compare(kernel: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    """The band, by what each kernel rounds.

    bf16 output (eight kernels, the prefill GEMM's bf16 entries, and the
    shared expert's prefill down kernel): both sides accumulate in fp32 and
    round once to bf16, so they differ by at most one rounding of nearly
    equal fp32 sums: one ulp of an element, at most two of the row-scale
    units the gate counts in. 4 ulp element-wise and 2 ulp relative RMS, as
    `bmm_out` uses; a wrong product misses by orders.

    fp32 output from bf16 operands (router): products are exact, so the only
    difference is the order of K = 6144 fp32 additions against an fp64
    reference. A random-walk estimate is a few fp32 ulps of the row scale.
    The band is 512 ulp (2**-14) element-wise and 128 ulp (2**-16) RMS, wide
    enough for an accumulator that truncates instead of rounding; a wrong
    logit is O(1).

    ix_fused: fp32 operands rounded to TF32 inside the kernel, against the
    exact fp32 product. `x` holds bf16 values (as the target's `x.float()`
    does), so only the weight's rounding counts: at most half a TF32 ulp per
    product, which sums to about 0.4 TF32 ulp of relative RMS over random
    signs and 1-2 ulp at the worst element of a row. 4 ulp element-wise and
    2 ulp RMS, in TF32 units; that also holds if the hardware truncates
    instead of rounding.
    """
    if kernel == "ix_fused":
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=TF32_ULP)
    elif out.dtype == torch.float32:
        assert_within_ulp(out, ref, element_ulp=512.0, rms_ulp=128.0, ulp=FP32_ULP)
    else:
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _weight(kernel: str, g: torch.Generator) -> torch.Tensor:
    """A row-major `[N, K]` weight scaled so each output is O(1). ix_fused's
    is full-mantissa fp32, so the kernel's TF32 rounding is exercised."""
    s = shape_of(kernel)
    return (torch.randn(s.n, s.k, generator=g, device="cuda") * s.k**-0.5).to(s.in_dtype)


#: The shared expert's prefill down runner, built once per session.
_SHARED_DOWN_PREFILL: dict = {}


def _shared_down_prefill_runner() -> SharedDownPrefillRunner:
    if "runner" not in _SHARED_DOWN_PREFILL:
        g = torch.Generator(device="cuda").manual_seed(0)
        _SHARED_DOWN_PREFILL["runner"] = build_shared_down_prefill(_weight("shared_down", g))
    return _SHARED_DOWN_PREFILL["runner"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    kernel, rows = cell.spec["kernel"], cell.spec["rows"]
    s = shape_of(kernel)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    if cell.spec.get("prefill"):
        runner = build_prefill(kernel, _weight(kernel, g))
        assert (runner.n, runner.k, runner.out_dtype) == (s.n, s.k, s.out_dtype)
    elif cell.spec.get("shared_prefill"):
        runner = _shared_down_prefill_runner()
        assert (runner.n, runner.k, runner.out_dtype) == (s.n, s.k, s.out_dtype)
    else:
        runner = build(kernel, rows, _weight(kernel, g))
    # Another layer's weight: the target builds once against layer 0 and runs
    # every layer through that runner, so the runner must not have kept it.
    weight = _weight(kernel, g)
    # bf16 values even for ix_fused, whose caller passes `x.float()`.
    x = torch.randn(rows, s.k, generator=g, device="cuda").to(torch.bfloat16).to(s.in_dtype)
    # NaN, so an element the kernel never writes fails the comparison.
    out = torch.full((rows, s.n), float("nan"), dtype=s.out_dtype, device="cuda")
    with validating(op):
        assert op(x, weight, out, runner) is out, "the op writes `out` and returns it"
    _compare(kernel, out, op.reference(x, weight, out, runner))


def test_build_refuses_a_row_count_without_a_schedule() -> None:
    """The target sends such a row count to cuBLAS; `build` must not pretend."""
    with pytest.raises(ValueError, match="no schedule for 3 rows"):
        build("qkva", 3, torch.zeros(2624, 6144, dtype=torch.bfloat16, device="cuda"))


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def _runner(kernel: str, rows: int) -> ProjectionRunner:
    """A runner whose kernel is never reached: `is_valid` runs first."""
    s = shape_of(kernel)
    return ProjectionRunner(kernel, rows, s.n, s.k, s.in_dtype, s.out_dtype, _unreachable)


def _ok(kernel: str, rows: int):
    s = shape_of(kernel)
    x = torch.zeros(rows, s.k, dtype=s.in_dtype, device="cuda")
    w = torch.zeros(s.n, s.k, dtype=s.in_dtype, device="cuda")
    out = torch.empty(rows, s.n, dtype=s.out_dtype, device="cuda")
    return x, w, out, _runner(kernel, rows)


def test_guard_refuses_the_column_major_view_of_a_square_weight() -> None:
    """dense_gate_up is 6144 x 6144: the `[K, N]` view `cublas_mm` takes has
    the right shape, and the kernel checks layout on a runner's first call only."""
    x, w, out, runner = _ok("dense_gate_up", 24)
    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(op):
            op(x, w.t(), out, runner)


def test_guard_refuses_a_bf16_out_for_the_router() -> None:
    """The router kernel writes fp32 into whatever buffer it is given."""
    x, w, out, runner = _ok("router", 8)
    with pytest.raises(AssertionError, match="writes torch.float32"):
        with validating(op):
            op(x, w, out.to(torch.bfloat16), runner)


def test_guard_refuses_a_strided_input() -> None:
    """A column slice of a wider buffer. The bf16 kernels check layout on a
    runner's first call only, so nothing else stops it."""
    s = shape_of("qkva")
    wide = torch.zeros(8, s.k + 64, dtype=torch.bfloat16, device="cuda")
    _, w, out, runner = _ok("qkva", 8)
    with pytest.raises(AssertionError, match="x must be contiguous"):
        with validating(op):
            op(wide[:, : s.k], w, out, runner)


def _prefill_runner(kernel: str) -> PrefillProjectionRunner:
    """A prefill runner whose GEMM is never reached: `is_valid` runs first."""
    s = shape_of(kernel)
    return PrefillProjectionRunner(
        kernel, s.n, s.k, s.in_dtype, s.out_dtype, PREFILL_MAX_ROWS, _unreachable
    )


def test_guard_refuses_a_prefill_row_count_past_the_ceiling() -> None:
    """The target chunks above max_num_tokens; the GEMM's schedule table ends there."""
    s = shape_of("q_b")
    rows = PREFILL_MAX_ROWS + 1
    x = torch.zeros(rows, s.k, dtype=s.in_dtype, device="cuda")
    w = torch.zeros(s.n, s.k, dtype=s.in_dtype, device="cuda")
    out = torch.empty(rows, s.n, dtype=s.out_dtype, device="cuda")
    with pytest.raises(AssertionError, match="serves 1..8192 rows"):
        with validating(op):
            op(x, w, out, _prefill_runner("q_b"))


def test_guard_refuses_the_prefill_gemm_a_view_of_a_weight_it_has_met() -> None:
    """The prefill GEMM caches each weight's operand view by address and checks
    layout only on first sight: the column-major view of the square
    dense_gate_up weight shares that address and the shape."""
    s = shape_of("dense_gate_up")
    x = torch.zeros(80, s.k, dtype=s.in_dtype, device="cuda")
    w = torch.zeros(s.n, s.k, dtype=s.in_dtype, device="cuda")
    out = torch.empty(80, s.n, dtype=s.out_dtype, device="cuda")
    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(op):
            op(x, w.t(), out, _prefill_runner("dense_gate_up"))


def test_build_prefill_refuses_an_entry_the_kernel_lacks_or_mistypes() -> None:
    """Every refusal leaves the projection on cuBLAS at the call site, never on a
    launcher that writes the wrong dtype."""
    for kernel in ("qkva", "router", "mtp_eh"):
        s = (6144, 12288) if kernel == "mtp_eh" else (shape_of(kernel).n, shape_of(kernel).k)
        weight = torch.zeros(*s, dtype=torch.bfloat16, device="cuda")
        try:
            runner = build_prefill(kernel, weight)
        except ValueError as exc:
            assert "has no" in str(exc) or "writes" in str(exc), exc
            continue
        expected = torch.float32 if kernel == "router" else torch.bfloat16
        assert runner.out_dtype == expected, (kernel, runner.out_dtype)


def test_guard_refuses_a_misaligned_input() -> None:
    """The bf16 kernels' TMA needs 128-byte aligned operands."""
    s = shape_of("o_proj")
    flat = torch.zeros(8 * s.k + 8, dtype=torch.bfloat16, device="cuda")
    _, w, out, runner = _ok("o_proj", 8)
    with pytest.raises(AssertionError, match="x must be 128-byte aligned"):
        with validating(op):
            op(flat[8:].view(8, s.k), w, out, runner)


def test_the_shared_prefill_down_refuses_bad_operands_loudly() -> None:
    """The shared expert's prefill kernel checks rows, shapes, dtypes, alignment and
    overlap on every call, so `is_valid` adds nothing for it; this pins that the
    refusals are loud: 64 rows (the decode kernel's), and an `out` aligned for the
    decode kernel (32 bytes) but not for this one (128)."""
    runner = _shared_down_prefill_runner()
    s = shape_of("shared_down")
    x = torch.zeros(64, s.k, dtype=s.in_dtype, device="cuda")
    w = torch.zeros(s.n, s.k, dtype=s.in_dtype, device="cuda")
    with pytest.raises(ValueError, match="65..8192"):
        op(x, w, torch.empty(64, s.n, dtype=s.out_dtype, device="cuda"), runner)
    rows = SHARED_DOWN_PREFILL_MIN_ROWS
    x = torch.zeros(rows, s.k, dtype=s.in_dtype, device="cuda")
    # 16 bf16 elements past a 512-byte aligned allocation: 32-byte aligned, not 128.
    flat = torch.empty(rows * s.n + 16, dtype=s.out_dtype, device="cuda")
    out = flat[16:].view(rows, s.n)
    assert out.data_ptr() % 32 == 0 and out.data_ptr() % 128 != 0
    with pytest.raises(ValueError, match="alignment"):
        op(x, w, out, runner)
