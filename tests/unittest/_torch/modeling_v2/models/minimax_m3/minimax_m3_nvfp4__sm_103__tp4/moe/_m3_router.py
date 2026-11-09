# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared operands and gates of the router entries' tests (m3_router_gemv, m3_router_gemv_fp4, m3_router_tc: one
kernel source)."""

import zlib

import torch

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp  # noqa: E402 — needs the path declared above

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.moe.m3_router_gemv import (  # noqa: E501
    m3_router_gemv as gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.moe.m3_router_gemv import (  # noqa: E501
    near_tie,
    nvfp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.moe.m3_router_gemv_fp4 import (  # noqa: E501
    m3_router_gemv_fp4 as gemv_fp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.moe.m3_router_tc import (  # noqa: E501
    m3_router_tc as tc,
)

__all__ = [
    "SF_SCALE",
    "_build",
    "_compare_fp4",
    "_compare_logits",
    "_seed",
    "gemv",
    "gemv_fp4",
    "tc",
]

#: The experts' NVFP4 input scale the tc cells use: 448 * 6 / 8, block scales well inside E4M3.
SF_SCALE = 336.0


def _build(m: int, e: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, k, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    w = torch.randn(e, k, generator=g, device="cuda") * 0.02
    return x, w


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _compare_logits(out: torch.Tensor, ref: torch.Tensor) -> None:
    """fp32 logits that differ from a true fp32 product only in summation order.

    Over K = 6144 products a reordered fp32 sum moves by a few ulps of the sum of magnitudes, which near
    cancellation is ~sqrt(K) times the row's largest logit's ulp at most. Gated in fp32 ulps of each row's largest
    logit: 64 element-wise, 8 in relative RMS. Measured on GB300: GEMV 1.6 / 1.4 ulp, tensor-core router 3.8 / 2.0
    ulp at 2..48 tokens; the tensor-core logits against a true fp32 product (the TF32 rounding the band must see)
    sit at 2200+ / 1700 ulp.
    """
    assert out.dtype == ref.dtype == torch.float32 and out.shape == ref.shape
    assert_within_ulp(out, ref, element_ulp=64.0, rms_ulp=8.0)


def _codes(packed: torch.Tensor) -> torch.Tensor:
    """[M, K / 2] packed e2m1 bytes -> [M, K] 4-bit codes, low nibble first."""
    lo, hi = packed & 0xF, packed >> 4
    return torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)


def _compare_fp4(
    q: torch.Tensor, sf: torch.Tensor, x: torch.Tensor, sf_scale: torch.Tensor
) -> None:
    """Scales bit-exact; codes bit-exact except within ``TIE_WINDOW`` of an e2m1 midpoint, where the kernel's
    approximate reciprocals choose the side (and then only by one code)."""
    ref_q, ref_sf, scaled = nvfp4_quantize(x, sf_scale)
    assert q.dtype == torch.uint8 and q.shape == ref_q.shape
    assert torch.equal(sf, ref_sf), f"{(sf != ref_sf).sum().item()} block scales differ"
    got, want = _codes(q), _codes(ref_q)
    differ = got != want
    assert not bool((differ & ~near_tie(scaled)).any()), (
        f"{(differ & ~near_tie(scaled)).sum().item()} e2m1 codes differ away from a rounding tie"
    )
    magnitude = ((got & 7).int() - (want & 7).int()).abs()
    assert bool((magnitude[differ] <= 1).all()), "a tie resolved more than one code away"
