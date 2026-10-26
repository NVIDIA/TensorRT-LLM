# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared operands and gates of the router entry's tests (m3_router_tc)."""

import zlib

import torch

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp  # noqa: E402 — needs the path declared above

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.moe.m3_router_tc import (  # noqa: E501
    m3_router_tc as tc,
)

__all__ = ["_build", "_compare_logits", "_seed", "tc"]


def _build(m: int, e: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, k, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    w = torch.randn(e, k, generator=g, device="cuda") * 0.02
    return x, w


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _compare_logits(out: torch.Tensor, ref: torch.Tensor) -> None:
    """fp32 logits that differ from the reference only in summation order.

    Over K = 6144 products a reordered fp32 sum moves by a few ulps of the sum of magnitudes, which near
    cancellation is ~sqrt(K) times the row's largest logit's ulp at most. Gated in fp32 ulps of each row's largest
    logit: 64 element-wise, 8 in relative RMS. The tensor-core logits against a true fp32 product (the TF32 rounding
    the band must see) sit at 2200+ / 1700 ulp.
    """
    assert out.dtype == ref.dtype == torch.float32 and out.shape == ref.shape
    assert_within_ulp(out, ref, element_ulp=64.0, rms_ulp=8.0)
