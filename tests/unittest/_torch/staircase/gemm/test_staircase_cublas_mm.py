# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cublas_mm catalog entry."""

import contextlib

import torch

from tensorrt_llm._torch.staircase.catalog.gemm.cublas_mm import cublas_mm

assert torch.cuda.is_available(), "cublas_mm requires a CUDA device"


@contextlib.contextmanager
def _true_fp32_matmul():
    """Make torch's fp32 matmul actually fp32 for the duration.

    torch 2.12 defaults ``matmul.fp32_precision`` to ``tf32`` and
    ``allow_tf32`` to True on this hardware, so a plain ``a.float() @
    b.float()`` is a *TF32* product -- ~1e-3 relative, which is 30x the error
    of the op being tested. Left alone the reference is the inaccurate side of
    the comparison and the entry fails against a correct kernel. Measured
    here: with TF32 off the op is bit-identical to torch and both sit 1.9e-5
    from a float64 product; with TF32 on the reference alone moves by 0.035.
    """
    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    torch.backends.cuda.matmul.allow_tf32 = False
    if prev_prec is not None:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        if prev_prec is not None:
            torch.backends.cuda.matmul.fp32_precision = prev_prec


def _ref_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
) -> torch.Tensor:
    """fp32-accumulated reference: mat_a @ mat_b (+ bias), cast to out dtype."""
    with _true_fp32_matmul():
        ref = mat_a.float() @ mat_b.float()
    if bias is not None:
        ref = ref + bias.float()
    return ref.to(out_dtype if out_dtype is not None else mat_a.dtype)


def _check(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> None:
    out = cublas_mm(mat_a, mat_b, bias, out_dtype)
    ref = _ref_mm(mat_a, mat_b, bias, out_dtype)
    expected_dtype = out_dtype if out_dtype is not None else mat_a.dtype
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype == expected_dtype
    # rtol: torch.testing defaults per output dtype. atol: kernel and reference
    # both accumulate in fp32 but in different summation orders; for K <= 4096
    # unit-variance inputs the order-dependent absolute noise is up to
    # ~K * 2^-24 ~= 2.4e-4, which dominates on near-zero outputs produced by
    # cancellation, so atol=1e-3 instead of the ~1e-5 defaults.
    rtol = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}
    torch.testing.assert_close(out, ref, rtol=rtol[expected_dtype], atol=1e-3)


def _make(m: int, k: int, n: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Build mat_a [M,K] row-major, mat_b [K,N] column-major, bias [N]."""
    mat_a = torch.randn(m, k, device="cuda").to(dtype)
    weight = torch.randn(n, k, device="cuda").to(dtype)  # linear weight [N, K]
    bias = torch.randn(n, device="cuda").to(dtype)
    return mat_a, weight.t(), bias


def test_bf16_no_bias() -> None:
    torch.manual_seed(0)
    # decode-like (few tokens) and prefill-like (many tokens) shapes
    for m, k, n in [(1, 4096, 4096), (8, 4096, 11008), (2048, 4096, 4096)]:
        mat_a, mat_b, _ = _make(m, k, n, torch.bfloat16)
        _check(mat_a, mat_b)


def test_bf16_bias() -> None:
    torch.manual_seed(1)
    for m, k, n in [(1, 4096, 4096), (512, 2048, 6144)]:
        mat_a, mat_b, bias = _make(m, k, n, torch.bfloat16)
        _check(mat_a, mat_b, bias)


def test_bf16_out_fp32() -> None:
    # bias must match the output dtype (fp32 here), not the input dtype
    torch.manual_seed(2)
    mat_a, mat_b, _ = _make(16, 1024, 2048, torch.bfloat16)
    bias_fp32 = torch.randn(2048, device="cuda", dtype=torch.float32)
    _check(mat_a, mat_b, out_dtype=torch.float32)
    _check(mat_a, mat_b, bias_fp32, out_dtype=torch.float32)


def test_bf16_unaligned_shapes() -> None:
    # dims not multiples of typical tile/vector widths
    torch.manual_seed(3)
    for m, k, n in [(5, 100, 60), (7, 333, 129)]:
        mat_a, mat_b, bias = _make(m, k, n, torch.bfloat16)
        _check(mat_a, mat_b)
        _check(mat_a, mat_b, bias)


def test_fp16() -> None:
    torch.manual_seed(4)
    for m, k, n in [(1, 4096, 4096), (1024, 2048, 2048)]:
        mat_a, mat_b, bias = _make(m, k, n, torch.float16)
        _check(mat_a, mat_b)
        _check(mat_a, mat_b, bias)


def test_fp32() -> None:
    # no bias: the op silently ignores bias when inputs are fp32
    # (contract precondition; guarded by an assert in the wrapper)
    torch.manual_seed(5)
    for m, k, n in [(2, 1024, 1024), (256, 2048, 1024)]:
        mat_a, mat_b, _ = _make(m, k, n, torch.float32)
        _check(mat_a, mat_b)


def test_fp8_e4m3_to_bf16() -> None:
    # fp8 inputs require an explicit out_dtype; no scales are applied (alpha=1)
    torch.manual_seed(6)
    for m, k, n in [(1, 1024, 1024), (128, 1024, 512)]:
        mat_a = (torch.randn(m, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        weight = (torch.randn(n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        _check(mat_a, weight.t(), out_dtype=torch.bfloat16)
        _check(mat_a, weight.t(), bias, out_dtype=torch.bfloat16)
