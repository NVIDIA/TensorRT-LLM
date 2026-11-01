# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_router entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_moe_router runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_router import (  # noqa: E402, E501
    DECODE_MAX_ROWS,
    EXPERTS,
    HIDDEN,
    SIMT_MAX_ROWS,
    RouterRunner,
    build,
    build_prefill,
    output_shape,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_router import (  # noqa: E402, E501
    cute_dsl_moe_router as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

#: One unit of fp32's mantissa.
FP32_ULP = 2.0**-23

#: Compiled runners, one per decode row count plus one prefill runner, built once a session.
_RUNNERS: dict[object, RouterRunner] = {}


def _runner(rows: int) -> RouterRunner:
    key = "prefill" if rows > DECODE_MAX_ROWS else rows
    if key not in _RUNNERS:
        device = torch.cuda.current_device()
        _RUNNERS[key] = build_prefill(device) if key == "prefill" else build(rows, device)
    return _RUNNERS[key]


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The band, by what each side rounds.

    Every product is a bf16 x bf16 product, exact in fp32 (8 + 8 mantissa bits
    fit in 24). So the kernel differs from the reference only in how it sums:
    fp32 additions in the kernel's order over one K slice (6144 columns, or
    1536 for a split-K partial), against an fp64 sum rounded once to fp32. For
    K rounded fp32 additions a random-walk estimate is sqrt(K / 24), about 16
    ulps of the slice's typical magnitude at K = 6144, and the row's largest
    element is about three times that magnitude. A tensor-core accumulator
    that truncates instead of rounding loses at most one ulp per 16-column MMA
    step: 96 steps per partial, 384 over the full K, about half of that on
    average, and less while the running sum is still small. That stays inside
    512 element-wise and near 64 of the 128 RMS in row-scale units. The band is
    the one the projection test's router uses:
    512 ulp (2**-14) of the row's largest magnitude element-wise and 128 ulp
    (2**-16) relative RMS. Rows are compared as `[rows, planes * 256]`, so a
    partial is held to its own row's scale. A wrong logit, a partial written
    to the wrong plane, or an unwritten (NaN) element misses by O(1).
    """
    rows = out.shape[0]
    assert_within_ulp(
        out.reshape(rows, -1), ref.reshape(rows, -1), element_ulp=512.0, rms_ulp=128.0, ulp=FP32_ULP
    )


def _inputs(rows: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 `x` and a gate weight scaled by 6144**-0.5, so each logit is O(1)."""
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn(EXPERTS, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5).to(torch.bfloat16)
    return x, w


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    rows = cell.spec["rows"]
    runner = _runner(rows)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x, w = _inputs(rows, g)
    # NaN, so an element the kernel never writes fails the comparison.
    out = torch.full(output_shape(rows), float("nan"), dtype=torch.float32, device="cuda")
    with validating(op):
        assert op(x, w, out, runner) is out, "the op writes `out` and returns it"
    _compare(out, op.reference(x, w, out, runner))


_PARTIAL_CELLS = [c for c in op.CELLS if SIMT_MAX_ROWS < c.spec["rows"] <= DECODE_MAX_ROWS]


@pytest.mark.parametrize("cell", _PARTIAL_CELLS, ids=[c.why[:44] for c in _PARTIAL_CELLS])
def test_the_partials_sum_to_the_logits(cell) -> None:
    """The four planes are K-quarter partials of one logit, not four logits.

    The fused top-k adds them in increasing split order before the sigmoid, so
    that sum, done here the same way, has to be the full `[rows, 256]` product
    (the reference at a single plane, fp64 rounded once to fp32). The three
    extra fp32 additions add at most 1.5 ulp of the logit's scale, so the
    certified band above still holds. Planes in the wrong order of K, or a
    plane holding a full product, miss by O(1). The reference's own planes are
    held to the same sum, which pins what its slicing means.
    """
    rows = cell.spec["rows"]
    runner = _runner(rows)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x, w = _inputs(rows, g)
    out = torch.full(output_shape(rows), float("nan"), dtype=torch.float32, device="cuda")
    op(x, w, out, runner)
    logits = op.reference(x, w, torch.empty(rows, EXPERTS, device="cuda"), runner)
    ref = op.reference(x, w, out, runner)
    for planes in (out, ref):
        total = planes[:, 0]
        for s in range(1, planes.shape[1]):
            total = total + planes[:, s]
        _compare(total, logits)


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def _ok(rows: int):
    x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    w = torch.zeros(EXPERTS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(output_shape(rows), dtype=torch.float32, device="cuda")
    return x, w, out


def test_guard_refuses_a_row_count_the_runner_was_not_compiled_for() -> None:
    """A decode runner bakes its row count into the launch: 9 rows on an
    8-row runner leave the last row's logits unwritten."""
    x, w, out = _ok(9)
    with pytest.raises(AssertionError, match="compiled for 8 rows"):
        with validating(op):
            op(x, w, out, RouterRunner("tc_partials", 8, _unreachable))


def test_guard_refuses_a_single_plane_out_for_the_partials_runner() -> None:
    """`[rows, 256]` has a quarter of the partials' elements; the kernel would
    write all four planes through it."""
    x, w, _ = _ok(8)
    out = torch.empty(8, EXPERTS, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="out must be fp32"):
        with validating(op):
            op(x, w, out, RouterRunner("tc_partials", 8, _unreachable))


def test_guard_refuses_a_strided_input() -> None:
    """A column slice of a wider buffer, read as dense rows through its pointer."""
    wide = torch.zeros(8, HIDDEN + 64, dtype=torch.bfloat16, device="cuda")
    _, w, out = _ok(8)
    with pytest.raises(AssertionError, match="x must be contiguous"):
        with validating(op):
            op(wide[:, :HIDDEN], w, out, RouterRunner("tc_partials", 8, _unreachable))


def test_guard_refuses_a_decode_row_count_on_the_prefill_runner() -> None:
    """The prefill runner's schedule table starts at 65 rows."""
    x, w, _ = _ok(64)
    out = torch.empty(64, 1, EXPERTS, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="the prefill router serves 65..8192 rows"):
        with validating(op):
            op(x, w, out, RouterRunner("prefill", None, _unreachable))
