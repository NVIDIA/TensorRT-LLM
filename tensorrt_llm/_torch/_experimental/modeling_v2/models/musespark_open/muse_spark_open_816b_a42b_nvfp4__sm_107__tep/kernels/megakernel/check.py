# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check-mode references of the megakernel's stages (float64 between the stock path's rounding points) and the
error statistics the tests gate on."""

from __future__ import annotations

from typing import Dict

import torch


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """The BF16 unit in the last place at ``|x|`` (8 significant bits; the smallest normal's below it), float64."""
    a = x.double().abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(a)) - 7.0)


def stat(ref: torch.Tensor, got: torch.Tensor) -> Dict[str, float]:
    """Relative L2 error, max abs error, max error in BF16 ulps of the reference value, the reference's max |value|,
    finiteness, and the fraction of elements that differ from the reference rounded to ``got``'s dtype (the
    correctly rounded value)."""
    ref64 = ref.double()
    got64 = got.double()
    d = ref64 - got64
    rounded = ref64.float().to(got.dtype).double() if got.dtype != torch.float64 else ref64
    return dict(
        rel=float(d.norm() / ref64.norm().clamp_min(1e-300)),
        max=float(d.abs().max()),
        max_ulp=float((d.abs() / bf16_ulp(ref64)).max()),
        ref_max=float(ref64.abs().max()),
        finite=bool(torch.isfinite(got64).all()),
        mismatch=float((got64 != rounded).double().mean()),
    )


def pre_qkv_input(residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The stock attention input of a layer >= 1 (``unpool_gather_boundary._reference_boundary`` with
    ``norm_fp32_residual=False``): ``bf16(c * rsqrt(mean(c^2) + eps) * (w + 1))`` with ``c = bf16(residual)``, in
    float64 between the rounding points. ``residual`` FP32 ``[8, 8192]``, ``weight`` BF16 ``[8192]``."""
    c = residual.float().to(torch.bfloat16).double()
    x = c * torch.rsqrt(c.square().mean(-1, keepdim=True) + eps) * (weight.double() + 1.0)
    return x.to(torch.bfloat16)


def qkv_exact(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ w^T`` in float64 (mode A's exact value): ``x`` ``[8, K]``, ``w`` ``[N, K]``."""
    return x.double() @ w.double().t()


def qkv_exact_from_residual(
    residual: torch.Tensor, weight: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    """Mode B's exact value: the pre-QKV norm of ``c = bf16(residual)`` and the product, all in float64 (no BF16
    rounding of the normed input)."""
    c = residual.float().to(torch.bfloat16).double()
    x = c * torch.rsqrt(c.square().mean(-1, keepdim=True) + eps) * (weight.double() + 1.0)
    return x @ w.double().t()


def within_budget(
    kernel: Dict[str, float], stock: Dict[str, float], factor: float = 1.25, max_key: str = "max"
) -> bool:
    """The kernel's relative error and max error (``max_key``: ``max`` = absolute, ``max_ulp`` = in BF16 ulps of
    the exact value) are within ``factor`` x the stock op's, both against the same exact value."""
    return (
        kernel["finite"]
        and kernel["rel"] <= factor * stock["rel"]
        and kernel[max_key] <= factor * stock[max_key]
    )
