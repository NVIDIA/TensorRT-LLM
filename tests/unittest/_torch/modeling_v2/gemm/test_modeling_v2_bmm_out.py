# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the bmm_out catalog entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out as op

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "bmm_out requires a CUDA device"


def _build(spec, seed):
    """Materialize one cell.

    `transposed_out` is the part a spec cannot express as a plain `empty`: the
    targets hand the op a transposed view of a buffer the forward already
    holds, so the product lands in place. The parent buffer comes back too,
    because "did it land there" is the thing worth asserting.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    b, m, k, n, dt = spec["batch"], spec["m"], spec["k"], spec["n"], spec["dtype"]
    a = torch.randn(b, m, k, generator=g, device="cuda").to(dt)
    w = torch.randn(b, k, n, generator=g, device="cuda").to(dt)
    if spec["transposed_out"]:
        parent = torch.zeros(m, b, n, dtype=dt, device="cuda")
        out = torch.transpose(parent, 0, 1)
        # Only a decode-shaped cell escapes this: transposing a dim of extent 1
        # leaves a tensor torch still calls contiguous, so the assertion would
        # be about the shape rather than about the view.
        assert m == 1 or not out.is_contiguous()
    else:
        parent = torch.zeros(b, m, n, dtype=dt, device="cuda")
        out = parent
    return a, w, out, parent


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    a, w, out, parent = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    parent_ptr = parent.data_ptr()
    with validating(op):
        assert op(a, w, out) is None, "the op writes in place"
    # The product has to be in the caller's buffer, not in a replacement the op
    # allocated: that is the whole reason this op exists over torch.bmm.
    assert parent.data_ptr() == parent_ptr
    op.compare(out, op.reference(a, w, out))


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
