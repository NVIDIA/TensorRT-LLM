# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's bmm_out entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.bmm_out import (  # noqa: E501
    bmm_out as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "bmm_out requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference.

    Kernel and reference consume bit-identical operands and differ only in
    the order cuBLAS and torch sum over K. Over random operands a few
    outputs land near zero, where that reordering shows up as a relative
    difference of 10 on an absolute difference of 1.5e-4 -- which is what
    torch's default band rejects, and what no bf16 result could represent
    anyway: one ulp at these rows' scale is about 6e-1.

    So: 4 ulp element-wise against the row scale, 2 ulp of relative RMS.
    At the deepseek geometry the measured worst case was 1.5e-4 absolute,
    roughly 3e-4 ulp. GLM's K of 192 and 512 is in the same range, so the
    band is kept; a genuinely wrong product misses it by orders.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _build(spec, seed):
    """Materialize one cell.

    With `a_row` / `out_row` set, `a` and `out` are transposed head-major views
    of token-major `[M, H, row]` buffers, as the GLM target passes them: `a` is
    the first `k` columns of each row, `out` the first `n`. The parent of `out`
    comes back too, because "did it land there" is the thing worth asserting.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    b, m, k, n, dt = spec["batch"], spec["m"], spec["k"], spec["n"], spec["dtype"]
    w = torch.randn(b, k, n, generator=g, device="cuda").to(dt)
    if spec["a_row"] is None:
        a = torch.randn(b, m, k, generator=g, device="cuda").to(dt)
    else:
        a_parent = torch.randn(m, b, spec["a_row"], generator=g, device="cuda").to(dt)
        a = a_parent[..., :k].transpose(0, 1)
    if spec["out_row"] is None:
        parent = torch.zeros(b, m, n, dtype=dt, device="cuda")
        out = parent
    else:
        parent = torch.zeros(m, b, spec["out_row"], dtype=dt, device="cuda")
        out = parent[..., :n].transpose(0, 1)
        # Only a decode-shaped cell escapes this: a dim of extent 1 leaves a
        # view torch may still call contiguous.
        assert m == 1 or not out.is_contiguous()
    return a, w, out, parent


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    a, w, out, parent = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    parent_before = parent.clone()
    parent_ptr = parent.data_ptr()
    with validating(op):
        # Identity, not equality: the op must have written this very buffer and
        # returned it, not produced a copy that happens to match.
        assert op(a, w, out) is out, "the op writes in place and returns what it wrote"
    # The product has to be in the caller's buffer, not in a replacement the op
    # allocated: that is the whole reason this op exists over torch.bmm.
    assert parent.data_ptr() == parent_ptr
    _compare(out, op.reference(a, w, out))
    # A strided `out` must not spill into the parent's other columns.
    if cell.spec["out_row"] is not None:
        n = cell.spec["n"]
        assert torch.equal(parent[..., n:], parent_before[..., n:]), (
            "the op wrote outside the out view"
        )


def _ok(dtype=torch.bfloat16):
    a = torch.randn(4, 8, 16, dtype=dtype, device="cuda")
    w = torch.randn(4, 16, 32, dtype=dtype, device="cuda")
    out = torch.zeros(4, 8, 32, dtype=dtype, device="cuda")
    return a, w, out


def test_guard_refuses_a_wrong_shaped_out() -> None:
    """The op silently reallocates it, detaching the caller's buffer."""
    a, w, _ = _ok()
    with pytest.raises(AssertionError, match=r"out must be \[B, M, N\]"):
        with validating(op):
            op(a, w, torch.zeros(4, 8, 31, dtype=torch.bfloat16, device="cuda"))


def test_guard_refuses_a_2d_operand() -> None:
    a, w, out = _ok()
    with pytest.raises(AssertionError, match="must be 3D"):
        with validating(op):
            op(a[0], w, out)


def test_guard_refuses_mixed_dtypes() -> None:
    """Promoted rather than rejected, which changes the required out dtype."""
    a, w, out = _ok()
    with pytest.raises(AssertionError, match="share one dtype"):
        with validating(op):
            op(a, w.to(torch.float16), out)
