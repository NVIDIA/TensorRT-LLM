# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's cublas_mm variant.

Drives every cell the variant certifies -- the checkpoint's q_a, kv_a, kv_b at
both its widths (14336 and the context engine's 7168 at attn_tp 2), router,
layer-0 MLP, latent/shared-expert down and lm_head projections at one decode
row, a 64-row block and a 5120-token prefill chunk -- through the entry's
guard and against its own fp32 reference, plus the band's negative control and
the four metadata refusals. Each cell prints the two band units it landed at,
so the band the entry states is in the log of the commit that claims it.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.gemm.cublas_mm__r11 import (  # noqa: E501
    cublas_mm as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import (  # noqa: E402,F401 — needs the path declared above
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "cublas_mm requires a CUDA device"

CELLS = op.CELLS


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference.

    Kernel and reference consume bit-identical operands and differ only in
    the order cuBLAS and torch sum over K. Over random operands a few
    outputs land near zero, where that reordering shows up as a large
    relative difference on an absolute difference no bf16 result could
    represent: one ulp at these rows' scale is about 6e-1.

    So: 4 ulp element-wise against the row scale, 2 ulp of relative RMS.
    Measured across this variant's cells: the worst element is 0.46 of the
    4-ulp band (about 1.9 ulp) and the worst relative RMS 2.8e-4, 3.6% of the
    2-ulp band. A product over another weight is 146x the element band, so
    the band separates a wrong result by two orders rather than squeaking
    past it.
    """
    # Scored in bf16 ulp even when the output is fp32: every cell's
    # operands are bf16, so the fp32 out_dtype the router asks for is a
    # wider container for a bf16-limited product, not a more precise one.
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _build(spec, seed):
    """Materialize one cell.

    `mat_b` is built as [N, K] and transposed, because that is the one thing a
    spec cannot say: the op reads it column-major, and a stored weight's `.t()`
    already is that, which is why no transpose kernel runs in a target. With
    `rows`, the weight is `[rows, K]` and `mat_b` the first N columns of its
    `.t()` (column stride still K); rows from `live` on are zero.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n, dt = spec["m"], spec["k"], spec["n"], spec["dtype"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(dt)
    w = torch.randn(spec.get("rows", n), k, generator=g, device="cuda").to(dt)
    if "live" in spec:
        w[spec["live"] :] = 0
    mat_b = w.t()[:, :n]
    out_dtype = spec["out_dtype"]
    bias = None
    if spec["bias"]:
        bias = torch.randn(n, generator=g, device="cuda").to(out_dtype or dt)
    return mat_a, mat_b, bias, out_dtype


def _band_units(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """One cell's measurements in the units `_compare` gates: the worst element
    as a multiple of one bf16 ulp of its row scale, and the relative RMS."""
    flat_out, flat_ref = (
        out.float().reshape(-1, out.shape[-1]),
        ref.float().reshape(-1, ref.shape[-1]),
    )
    row_scale = flat_ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    worst = ((flat_out - flat_ref).abs() / row_scale).amax().item()
    rel_rms = (
        (flat_out - flat_ref).pow(2).mean().sqrt() / flat_ref.pow(2).mean().sqrt().clamp_min(1e-9)
    ).item()
    return worst / ULP[torch.bfloat16], rel_rms


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 - pytest resolves the fixture by this name
    certifying(op)
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, bias, out_dtype)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype is (out_dtype or mat_a.dtype)
    ref = op.reference(mat_a, mat_b, bias, out_dtype)
    _compare(out, ref)
    element_ulp, rel_rms = _band_units(out, ref)
    print(
        f"{cell.why[:56]:56s} element {element_ulp / 4.0:.3f} of the 4-ulp band; "
        f"relative RMS {rel_rms:.3e} ({rel_rms / (2.0 * ULP[torch.bfloat16]):.2%} of the "
        f"2-ulp band)"
    )


def _ok(dtype=torch.bfloat16):
    a = torch.randn(8, 64, dtype=dtype, device="cuda")
    b = torch.randn(32, 64, dtype=dtype, device="cuda").t()
    return a, b


def test_band_rejects_a_different_weight() -> None:
    """The band's separation: a product over another weight differs by O(1) of
    the row scale, orders outside the 4-ulp element gate."""
    a, b = _ok()
    out = op(a, b)
    _, other = _ok()
    with pytest.raises(AssertionError):
        _compare(out, op.reference(a, other))


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
