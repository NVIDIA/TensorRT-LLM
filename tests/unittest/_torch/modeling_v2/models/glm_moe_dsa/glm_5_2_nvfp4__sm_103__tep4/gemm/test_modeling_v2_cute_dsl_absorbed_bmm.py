# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_absorbed_bmm entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_absorbed_bmm runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_absorbed_bmm import (  # noqa: E402, E501
    LAYOUTS,
    PREFILL_MAX_ROWS,
    AbsorbedBmmRunner,
    build,
    build_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_absorbed_bmm import (  # noqa: E402, E501
    cute_dsl_absorbed_bmm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

#: What the storage around `out` holds before the call; it must hold it after.
SENTINEL = -12345.0
#: Rows of storage past the last row the op is given.
CANARY_ROWS = 8


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Both sides take bf16 operands, sum K = 192 or 512 exact products in
    fp32, and round once to bf16: they differ by at most one rounding of
    nearly equal sums, one ulp of an element. 4 ulp element-wise against the
    row scale and 2 ulp relative RMS, the band `bmm_out` uses for the same two
    GEMMs; a wrong product misses it by orders. The decode and prefill kernels
    differ only in tiling, so one band serves both."""
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _group(rows: int) -> str:
    return "m_le_8" if rows <= 8 else "m_ge_16"


def _build(kernel: str, rows: int, seed: int):
    """`(a, weight, out, out_parent)` as the target passes them.

    `a` and `out` are `transpose(0, 1)` views of token-major `[M, 16, row]`
    storage: `a` the first K columns of each row, `out` the first N. The
    parent of `out` starts as SENTINEL so a write past N, or past the rows,
    shows.
    """
    lay = LAYOUTS[kernel]
    g = torch.Generator(device="cuda").manual_seed(seed)
    a_parent = torch.randn(rows, lay.heads, lay.a_row, generator=g, device="cuda")
    a = a_parent.to(torch.bfloat16)[..., : lay.k].transpose(0, 1)
    weight = torch.randn(lay.heads, lay.n, lay.k, generator=g, device="cuda") * lay.k**-0.5
    weight = weight.to(torch.bfloat16)
    # CANARY_ROWS rows past the view: same strides, and room to see an
    # unpredicated store write past M.
    parent = torch.full(
        (rows + CANARY_ROWS, lay.heads, lay.out_row), SENTINEL, dtype=torch.bfloat16, device="cuda"
    )
    out = parent[:rows, :, : lay.n].transpose(0, 1)
    return a, weight, out, parent


_PREFILL: dict[str, AbsorbedBmmRunner] = {}


def _prefill(kernel: str) -> AbsorbedBmmRunner:
    """One prefill runner per kernel for the module, as the target builds it."""
    if kernel not in _PREFILL:
        _PREFILL[kernel] = build_prefill(kernel, torch.cuda.current_device())
    return _PREFILL[kernel]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    kernel, rows = cell.spec["kernel"], cell.spec["rows"]
    if cell.spec.get("prefill"):
        runner = _prefill(kernel)
    else:
        runner = build(kernel, _group(rows), torch.cuda.current_device())
    assert rows in runner.rows
    a, weight, out, parent = _build(kernel, rows, zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        assert op(a, weight, out, runner) is out, "the op writes `out` and returns it"
    _compare(out, op.reference(a, weight, out, runner))
    n = LAYOUTS[kernel].n
    # K-B writes the latent half of the fused query; its rope tail is not the
    # kernel's to touch.
    assert bool((parent[:rows, :, n:] == SENTINEL).all()), "the kernel wrote past N"
    assert bool((parent[rows:] == SENTINEL).all()), "the kernel wrote past the last row"


@pytest.mark.parametrize("kernel", ["k_b", "v_b"])
def test_a_runner_refuses_rows_outside_its_group(kernel) -> None:
    """M=1 is in neither group and the target sends it to `bmm_out`; a row
    count of the other group must raise rather than run the wrong schedule."""
    small = build(kernel, "m_le_8", torch.cuda.current_device())
    for rows in (1, 16):
        a, weight, out, _ = _build(kernel, rows, seed=rows)
        with pytest.raises(ValueError, match="rows"):
            op(a, weight, out, small)


@pytest.mark.parametrize("kernel", ["k_b", "v_b"])
def test_the_prefill_runner_refuses_rows_past_its_ceiling(kernel) -> None:
    """The target sends a row count above 8192 to `bmm_out`."""
    a, weight, out, _ = _build(kernel, PREFILL_MAX_ROWS + 1, seed=1)
    with pytest.raises(ValueError, match="T <="):
        op(a, weight, out, _prefill(kernel))


def test_guard_refuses_the_bmm_out_view_of_a_weight() -> None:
    """The prefill kernel caches its weight operand by address and checks it on
    first sight only; the `[16, K, N]` view `bmm_out` takes shares the address."""
    a, weight, out, _ = _build("v_b", 80, seed=2)
    unreachable = AbsorbedBmmRunner("v_b", "prefill", range(1, PREFILL_MAX_ROWS + 1), None)
    with pytest.raises(AssertionError, match="contiguous bf16"):
        with validating(op):
            op(a, weight.transpose(1, 2), out, unreachable)
