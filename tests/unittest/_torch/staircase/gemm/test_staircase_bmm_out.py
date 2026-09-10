# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the bmm_out catalog entry."""

import torch

from tensorrt_llm._torch.staircase.catalog.gemm.bmm_out import bmm_out

assert torch.cuda.is_available(), "bmm_out requires a CUDA device"


def _check(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    ref = torch.bmm(a.float(), b.float()).to(out.dtype)
    bmm_out(a, b, out)
    # rtol: torch.testing defaults per output dtype. atol: kernel and reference
    # both accumulate in fp32 but in different summation orders; for K <= 1024
    # unit-variance inputs the order-dependent absolute noise is up to
    # ~K * 2^-24 ~= 6e-5, which dominates on near-zero outputs produced by
    # cancellation, so atol=1e-3 instead of the ~1e-5 defaults.
    rtol = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}
    torch.testing.assert_close(out, ref, rtol=rtol[out.dtype], atol=1e-3)


def _make(
    batch: int, m: int, k: int, n: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randn(batch, m, k, device="cuda").to(dtype)
    b = torch.randn(batch, k, n, device="cuda").to(dtype)
    out = torch.empty(batch, m, n, device="cuda", dtype=dtype)
    return a, b, out


def test_bf16() -> None:
    torch.manual_seed(0)
    # decode-like (few tokens per batch entry) and prefill-like (many) shapes
    for batch, m, k, n in [(64, 1, 512, 128), (16, 8, 576, 512), (8, 2048, 512, 128)]:
        _check(*_make(batch, m, k, n, torch.bfloat16))


def test_bf16_noncontiguous_views() -> None:
    # The op exists so `out` may be a strided view (e.g. a transposed slice of
    # a [tokens, groups, rank] buffer, as MLA uses it); `a` and `b` may be
    # strided views too. Exercise all three.
    torch.manual_seed(1)
    batch, m, k, n = 16, 32, 256, 64
    a_wide = torch.randn(batch, m, 2 * k, device="cuda").to(torch.bfloat16)
    a = a_wide[:, :, :k]  # row-strided view
    b = (
        torch.randn(batch, n, k, device="cuda").to(torch.bfloat16).transpose(1, 2)
    )  # transposed view
    out_buf = torch.empty(m, batch, n, device="cuda", dtype=torch.bfloat16)
    out = out_buf.transpose(0, 1)  # non-contiguous out
    _check(a, b, out)
    # writes landed in the aliased buffer, not a reallocation
    ref = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
    torch.testing.assert_close(out_buf.transpose(0, 1), ref, rtol=1.6e-2, atol=1e-3)


def test_bf16_unaligned_shapes() -> None:
    # dims not multiples of typical tile/vector widths
    torch.manual_seed(2)
    for batch, m, k, n in [(3, 5, 100, 60), (7, 13, 333, 129)]:
        _check(*_make(batch, m, k, n, torch.bfloat16))


def test_fp16() -> None:
    torch.manual_seed(3)
    for batch, m, k, n in [(64, 1, 512, 128), (8, 1024, 512, 256)]:
        _check(*_make(batch, m, k, n, torch.float16))


def test_fp32() -> None:
    torch.manual_seed(4)
    for batch, m, k, n in [(32, 2, 256, 128), (4, 1024, 512, 256)]:
        _check(*_make(batch, m, k, n, torch.float32))
