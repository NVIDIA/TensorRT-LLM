# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cublas_mm catalog entry."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm as op

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402 — needs the path declared above
    ULP,
    assert_within_ulp,
    validating,
)

assert torch.cuda.is_available(), "cublas_mm requires a CUDA device"


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
    wrong product would miss by orders rather than squeak past. On sm_107 the
    nemotron_h cells, up to K 8192 and 8192 rows, reach 1.8 ulp element-wise
    and 0.06 of RMS.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _compare_fp32(out: torch.Tensor, ref: torch.Tensor, k: int) -> None:
    """An fp32 output, gated at fp32 accumulation error.

    A product of two bf16 operands is exact in fp32, so an fp32 out_dtype is a
    genuinely fp32 result and the bf16 band above would pass one rounded to
    bf16 (about 14,000 fp32 ulp RMS). Kernel and reference differ only in how
    they sum K exact products in fp32. A rounding error of up to half an ulp
    per add, accumulated as a random walk over partial sums that grow like the
    final one, gives about sqrt(K / 8) ulp of relative RMS: 32 at K 8192.

    Measured on sm_107 against the exact (fp64) product at K 8192: 5-12 ulp RMS
    up to 32 rows, and 76 at 8192 rows, where the tensor-core kernels add error
    inside each MMA; element-wise up to 148 against `reference`. The deepseek
    router cell (K 7168, 16 rows) sits at 8 / 6. The gate is 2 * sqrt(K) ulp of
    relative RMS and 4 * sqrt(K) element-wise against the row scale, 181 / 362
    at K 8192: 2.4x the measured worst on both, and still some 75x inside a
    bf16-rounded result.
    """
    root_k = math.sqrt(k)
    assert_within_ulp(
        out, ref, element_ulp=4.0 * root_k, rms_ulp=2.0 * root_k, ulp=ULP[torch.float32]
    )


def _build(spec, seed):
    """Materialize one cell.

    `mat_b` is built as [N, K] and transposed, because that is the one thing a
    spec cannot say: the op reads it column-major, and a stored weight's `.t()`
    already is that, which is why no transpose kernel runs in a target.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n, dt = spec["m"], spec["k"], spec["n"], spec["dtype"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(dt)
    mat_b = torch.randn(n, k, generator=g, device="cuda").to(dt).t()
    out_dtype = spec["out_dtype"]
    bias = None
    if spec["bias"]:
        bias = torch.randn(n, generator=g, device="cuda").to(out_dtype or dt)
    return mat_a, mat_b, bias, out_dtype


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(mat_a, mat_b, bias, out_dtype)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype is (out_dtype or mat_a.dtype)
    ref = op.reference(mat_a, mat_b, bias, out_dtype)
    if out.dtype is torch.float32 and mat_a.dtype is not torch.float32:
        _compare_fp32(out, ref, k=mat_a.shape[1])
    else:
        _compare(out, ref)


def _router(rows: int = 32, k: int = 8192, n: int = 512):
    """The nemotron_h router call: bf16 activations against a bf16-exact weight."""
    g = torch.Generator(device="cuda").manual_seed(8192)
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    return x, w


def test_fp32_logits_are_not_bf16_limited() -> None:
    """The control for `_compare_fp32`: the same logits rounded to bf16 are what
    the bf16 band would accept, and the fp32 gate must refuse them."""
    x, w = _router()
    out = op(x, w.t(), None, torch.float32)
    ref = op.reference(x, w.t(), None, torch.float32)
    _compare_fp32(out, ref, k=x.shape[1])
    _compare(out.to(torch.bfloat16).float(), ref)
    with pytest.raises(AssertionError):
        _compare_fp32(out.to(torch.bfloat16).float(), ref, k=x.shape[1])


def _tf32_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch's fp32 matmul with TF32 forced on, the setting `true_fp32_matmul` undoes."""
    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    torch.backends.cuda.matmul.allow_tf32 = True
    if prev_prec is not None:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    try:
        return a.float() @ b.float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        if prev_prec is not None:
            torch.backends.cuda.matmul.fp32_precision = prev_prec


def test_tf32_is_inert_on_bf16_exact_operands() -> None:
    """Why a router weight stored in fp32 but bf16-exact loses nothing either way.

    TF32 keeps a 10-bit mantissa, so it cannot truncate a bf16 value: a TF32
    product of bf16-exact operands stays inside `_compare_fp32` of the true fp32
    one. The control is the hazard itself, operands that are not bf16-exact,
    where TF32 truncation moves the product orders of magnitude past the gate.
    """
    x, w = _router()
    k = x.shape[1]
    ref = op.reference(x, w.t(), None, torch.float32)
    _compare_fp32(_tf32_matmul(x, w.t()), ref, k=k)

    g = torch.Generator(device="cuda").manual_seed(8193)
    x32 = x.float() + torch.randn(x.shape, generator=g, device="cuda") * 1e-3
    exact = op.reference(x32, w.t(), None, torch.float32)
    with pytest.raises(AssertionError):
        _compare_fp32(_tf32_matmul(x32, w.t()), exact, k=k)


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
