# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's bmm_out variant (`__r16`).

Drives every cell the variant certifies -- the shared entry's deepseek-r1 MLA
cells, kept, and this checkpoint's latent-to-value projection at 32 sparse
heads, 64 SWA heads and the 8/16 a head set splits into per rank, at one
decode row, the 6-row verify width, a 512-row block and the 5120-row prefill
chunk -- through the entry's guard and against its own fp32 reference, plus
the band's negative control and the three metadata refusals.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.gemm.bmm_out__r16 import (  # noqa: E501
    bmm_out as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "bmm_out requires a CUDA device"

CELLS = op.CELLS


def _measure(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """The comparison's two margins, in ulp of the row's largest element."""
    flat_out = out.float().reshape(-1, out.shape[-1])
    flat_ref = ref.float().reshape(-1, ref.shape[-1])
    row_scale = flat_ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((flat_out - flat_ref).abs() / row_scale).max().item()
    rms = (
        (flat_out - flat_ref).pow(2).mean().sqrt() / flat_ref.pow(2).mean().sqrt().clamp_min(1e-9)
    ).item()
    return element / ULP[ref.dtype], rms / ULP[ref.dtype]


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference.

    Kernel and reference consume bit-identical operands and differ only in
    the order cuBLAS and torch sum over K. Over random operands a few
    outputs land near zero, where that reordering shows up as a large
    relative difference on an absolute difference no bf16 result could
    represent -- which is what torch's default per-element band rejects.

    So: 4 ulp element-wise against the row scale, 2 ulp of relative RMS.
    Measured across this variant's cells on sm_103: worst element 1.969 ulp,
    relative RMS 0.0268 ulp; a product over another weight measures 505 ulp,
    126 times the element band.
    """
    element, rms = _measure(out, ref)
    print(f"\nmeasured: element {element:.3f} ulp, rms {rms:.4f} ulp")
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


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


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    a, w, out, parent = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    parent_ptr = parent.data_ptr()
    # Identity, not equality: the op must have written this very buffer and
    # returned it, not produced a copy that happens to match.
    assert op(a, w, out) is out, "the op writes in place and returns what it wrote"
    # The product has to be in the caller's buffer, not in a replacement the op
    # allocated: that is the whole reason this op exists over torch.bmm.
    assert parent.data_ptr() == parent_ptr
    _compare(out, op.reference(a, w, out))


def _ok(dtype=torch.bfloat16):
    a = torch.randn(4, 8, 16, dtype=dtype, device="cuda")
    w = torch.randn(4, 16, 32, dtype=dtype, device="cuda")
    out = torch.zeros(4, 8, 32, dtype=dtype, device="cuda")
    return a, w, out


def test_band_rejects_a_different_weight() -> None:
    """The band's separation: a product over another weight differs by O(1) of
    the row scale, orders outside the 4-ulp element gate."""
    a, w, out = _ok()
    op(a, w, out)
    _, other, _ = _ok()
    with pytest.raises(AssertionError):
        _compare(out, op.reference(a, other, out))


def test_guard_refuses_a_wrong_shaped_out() -> None:
    """The op silently resizes it, detaching the caller's buffer."""
    a, w, _ = _ok()
    out = torch.zeros(4, 8, 31, dtype=torch.bfloat16, device="cuda")
    torch.ops.trtllm.bmm_out(a, w, out)
    assert out.shape == (4, 8, 32), "the hazard this guard covers is gone"
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
