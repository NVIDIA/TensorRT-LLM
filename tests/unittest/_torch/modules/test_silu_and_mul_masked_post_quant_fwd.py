# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `silu_and_mul_masked_post_quant_fwd` and
`per_token_quant_and_transform` at multiple group sizes.

Regression guard for a triton kernel bug where
`_silu_and_mul_post_quant_kernel` and `_per_token_quant_and_transform_kernel`
both hard-coded the pack stride to `128`, making them silently produce wrong
outputs for any `quant_group_size != 128` (notably gs=32 used by
MXFP4/MXFP8 MoE).
"""

import pytest
import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.quantization.utils.fp8_utils import (
    per_token_quant_and_transform,
    silu_and_mul_masked_post_quant_fwd,
)


def _dequant(fp8_bytes: torch.Tensor, sf_int32_packed: torch.Tensor, gran_k: int) -> torch.Tensor:
    """Dequantize per-group UE8M0-scaled FP8 output to float32.

    ``sf_int32_packed`` has shape ``[m, K_packed]`` int32 where each int32 packs
    four UE8M0 exponents along the group-k dim. When ``K_packed == 1`` the
    tensor looks contiguous to PyTorch but ``.view(uint8)`` rejects it because
    ``stride(-1) != 1``; unpack bytes via shifts instead.
    """
    m = fp8_bytes.shape[0]
    x = sf_int32_packed.to(torch.int32)
    b0 = x & 0xFF
    b1 = (x >> 8) & 0xFF
    b2 = (x >> 16) & 0xFF
    b3 = (x >> 24) & 0xFF
    sf_u8 = torch.stack([b0, b1, b2, b3], dim=-1).reshape(m, -1)
    sf_f = (sf_u8.to(torch.int32) << 23).view(torch.float32)
    sf_exp = sf_f.repeat_interleave(gran_k, dim=-1)
    # scale_k is padded up to a multiple of 4 ints, so sf_exp may be longer
    # than the actual k dimension. Trim to match.
    sf_exp = sf_exp[:, : fp8_bytes.shape[-1]]
    return fp8_bytes.float() * sf_exp


def _run_kernel_and_dequant(h1: torch.Tensor, masked_m: torch.Tensor, quant_group_size: int):
    """Run the kernel and return (dequant_bf16, bf16_reference)."""
    E, M_pad, H2 = h1.shape
    assert H2 % 2 == 0
    inter_dim = H2 // 2

    scale_k = (inter_dim + quant_group_size - 1) // quant_group_size
    scale_k_padded = (scale_k + 3) // 4 * 4

    out_fp8 = torch.zeros(E, M_pad, inter_dim, dtype=torch.float8_e4m3fn, device="cuda")
    out_sf = torch.zeros(E, scale_k_padded // 4, M_pad, dtype=torch.int32, device="cuda")

    ret_sf = silu_and_mul_masked_post_quant_fwd(
        output=out_fp8,
        output_scale=out_sf,
        input=h1.contiguous(),
        quant_group_size=quant_group_size,
        masked_m=masked_m,
        scale_ue8m0=True,
    )

    # BF16 reference (kernel reads first_half = up, second_half = gate)
    up = h1[:, :, :inter_dim].float()
    gate = h1[:, :, inter_dim:].float()
    ref = up * (gate / (1 + torch.exp(-gate)))

    return out_fp8, ret_sf, ref


@pytest.mark.skipif(get_sm_version() < 100, reason="Triton E8M0 path only validated on SM100+")
@pytest.mark.parametrize("quant_group_size", [32, 64, 128])
@pytest.mark.parametrize("intermediate_size", [256, 1024])
def test_silu_and_mul_gs_regression(quant_group_size, intermediate_size):
    """Validates correctness across multiple group sizes.

    Before the PACK_STRIDE fix, quant_group_size=32 produced garbage output
    (cos_diff ~0.57 vs reference). After the fix, all group sizes give
    cos_diff < 0.001.
    """
    torch.manual_seed(0)
    E, M_pad = 4, 16
    h1 = torch.randn(E, M_pad, 2 * intermediate_size, dtype=torch.bfloat16, device="cuda") * 2.0
    # Use variable masked_m to exercise the masked-token path
    masked_m = torch.tensor([8, 12, 4, M_pad], dtype=torch.int32, device="cuda")

    out_fp8, out_sf, ref = _run_kernel_and_dequant(h1, masked_m, quant_group_size)

    for e in range(E):
        m = masked_m[e].item()
        dequant = _dequant(out_fp8[e, :m], out_sf[e, :m].contiguous(), gran_k=quant_group_size)
        ref_e = ref[e, :m]
        a_sq = (ref_e * ref_e).sum()
        b_sq = (dequant * dequant).sum()
        dot = (ref_e * dequant).sum()
        cos_diff = float(1 - 2 * dot / (a_sq + b_sq))
        # Regression threshold: pre-fix this was ~0.57 for gs=32; post-fix <0.001
        assert cos_diff < 0.005, (
            f"gs={quant_group_size} inter={intermediate_size} expert={e} m={m}: "
            f"cos_diff={cos_diff:.6f} — kernel may have regressed the "
            f"hard-coded `128` pack stride bug."
        )


@pytest.mark.skipif(get_sm_version() < 100, reason="Triton E8M0 path only validated on SM100+")
@pytest.mark.parametrize("quant_group_size", [32, 64, 128])
@pytest.mark.parametrize("hidden_size", [256, 1024])
def test_per_token_quant_and_transform_gs_regression(quant_group_size, hidden_size):
    """Sibling regression to `_silu_and_mul_post_quant_kernel`:
    `_per_token_quant_and_transform_kernel` had the same hard-coded `128`
    pack stride and silently mis-quantized for ``quant_group_size != 128``.
    """
    torch.manual_seed(0)
    B, M = 1, 32
    x = torch.randn(B, M, hidden_size, dtype=torch.bfloat16, device="cuda") * 2.0

    out_fp8, out_sf = per_token_quant_and_transform(
        x, quant_group_size=quant_group_size, scale_ue8m0=True
    )
    # After the wrapper's internal transpose, out_sf is [B, M, K_p_div4] int32.
    for b in range(B):
        dequant = _dequant(out_fp8[b], out_sf[b].contiguous(), gran_k=quant_group_size)
        ref = x[b].float()
        a_sq = (ref * ref).sum()
        b_sq = (dequant * dequant).sum()
        dot = (ref * dequant).sum()
        cos_diff = float(1 - 2 * dot / (a_sq + b_sq))
        assert cos_diff < 0.005, (
            f"per_token gs={quant_group_size} H={hidden_size} b={b}: "
            f"cos_diff={cos_diff:.6f} — _per_token_quant_and_transform_kernel "
            f"may have regressed the hard-coded `128` pack stride bug."
        )
