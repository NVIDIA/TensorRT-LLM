# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The error band of a BF16 GEMM with FP32 accumulation, against its FP64 product."""

import torch


def assert_bf16_gemm_close(
    out: torch.Tensor, ref: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> None:
    """``out`` and ``ref`` are each the exact product rounded once to BF16 after a sum of K products:
    each is within half a BF16 ulp (2^-8 relative) of the FP32 sum, the FP32 sum within K FP32 unit
    roundoffs (2^-24) of the summed magnitudes of the exact one (any summation order, split-K
    included). So ``|out - ref| <= 2^-7 |ref| + K 2^-23 sum |x| |w|`` (plus BF16's smallest normal,
    for the zeros)."""
    k = x.shape[-1]
    mag = x.double().abs().reshape(-1, k) @ weight.double().abs().t()
    mag = mag.reshape(ref.shape)
    err = (out.double() - ref.double()).abs()
    band = 2.0**-7 * ref.double().abs() + k * 2.0**-23 * mag + 2.0**-126
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert bool(torch.isfinite(out).all()), "non-finite output"
    worst = (err / band).max().item()
    assert worst <= 1.0, f"max err / band {worst:.3f} (max err {err.max().item():.3e})"
