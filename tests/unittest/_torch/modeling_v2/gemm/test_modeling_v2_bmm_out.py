# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the bmm_out catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch.modeling_v2.catalog.gemm.bmm_out import bmm_out as op  # noqa: E501

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
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
    Measured worst case across this entry's cells is 1.5e-4 absolute, or
    roughly 3e-4 ulp -- four orders of margin, and a band that a genuinely
    wrong product would miss by orders rather than squeak past.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _build(spec, seed):
    """Materialize one cell.

    `broadcast_a`, `transposed_b` and `out_row` are the context KV expansion's
    operands: the latent shared by every head, a transposed weight view, and an
    out laid into the FMHA's packed rows. `transposed_out` is the part a spec cannot express as a plain `empty`: the
    targets hand the op a transposed view of a buffer the forward already
    holds, so the product lands in place. The parent buffer comes back too,
    because "did it land there" is the thing worth asserting.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    b, m, k, n, dt = spec["batch"], spec["m"], spec["k"], spec["n"], spec["dtype"]
    if spec.get("broadcast_a"):
        # One [m, k] operand shared by every batch index: batch stride 0.
        a = torch.randn(m, k, generator=g, device="cuda").to(dt).unsqueeze(0).expand(b, m, k)
    else:
        a = torch.randn(b, m, k, generator=g, device="cuda").to(dt)
    if spec.get("transposed_b"):
        w = torch.transpose(torch.randn(b, n, k, generator=g, device="cuda").to(dt), 1, 2)
    else:
        w = torch.randn(b, k, n, generator=g, device="cuda").to(dt)
    if "out_row" in spec:
        # A view into rows of `row` elements: batch index i's [m, n] block starts at
        # `offset + i * head_stride` of each row, as the context FMHA's K and V buffers do.
        row, head_stride, offset = spec["out_row"]
        parent = torch.zeros(m, row, dtype=dt, device="cuda")
        out = torch.as_strided(parent, (b, m, n), (head_stride, row, 1), offset)
    elif spec["transposed_out"]:
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


# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="deepseek default at attention TP8, decode absorb: 16 heads per rank",
            spec=dict(batch=16, m=1, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="deepseek default at attention TP4, decode expand: 32 heads per rank",
            spec=dict(batch=32, m=1, k=512, n=128, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="deepseek default at attention TP2, prefill-height absorb: 64 heads per rank",
            spec=dict(batch=64, m=2048, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="deepseek default at attention TP4, context V expansion: 32 heads per rank",
            spec=dict(
                batch=32,
                m=8192,
                k=512,
                n=128,
                dtype=torch.bfloat16,
                transposed_out=True,
                broadcast_a=True,
                transposed_b=True,
                out_row=(32 * 256, 128, 32 * 128),
            ),
        ),
        Cell(
            why="deepseek default at attention TP4, context K expansion: 32 heads per rank",
            spec=dict(
                batch=32,
                m=8192,
                k=512,
                n=128,
                dtype=torch.bfloat16,
                transposed_out=True,
                broadcast_a=True,
                transposed_b=True,
                out_row=(32 * 192, 192, 0),
            ),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    a, w, out, parent = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    parent_ptr = parent.data_ptr()
    with validating(op):
        # Identity, not equality: the op must have written this very buffer and
        # returned it, not produced a copy that happens to match.
        assert op(a, w, out) is out, "the op writes in place and returns what it wrote"
    # The product has to be in the caller's buffer, not in a replacement the op
    # allocated: that is the whole reason this op exists over torch.bmm.
    assert parent.data_ptr() == parent_ptr
    _compare(out, op.reference(a, w, out))
    if "out_row" in cell.spec:
        # Only the view's columns are written: the rest of each packed row (K's rope half,
        # the k_nope half of [k_nope | v]) belongs to other writers.
        rest = torch.ones_like(parent, dtype=torch.bool)
        torch.as_strided(rest, out.shape, out.stride(), out.storage_offset()).fill_(False)
        assert not parent[rest].any(), "the op wrote outside its view"


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
