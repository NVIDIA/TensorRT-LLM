# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU check of the family's ``rubin_mxfp8_gemm`` entry: its reference is the dequantized product.

No GPU and no kernel: the entry imports the kernel on first call, so the reference runs here on a
small shape.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm import rubin_mxfp8_gemm

__extra_import_path__ = ["../.."]
from _k31_utils import cpu_or_sm

pytestmark = pytest.mark.cpu_only
cpu_or_sm107 = cpu_or_sm(107)

G = rubin_mxfp8_gemm


@cpu_or_sm107
@pytest.mark.parametrize("m, n, k", [(130, 24, 256)])
def test_gemm_reference_is_the_dequantized_product(m, n, k) -> None:
    gen = torch.Generator().manual_seed(m + n + k)
    a, a_s = G.quantize_mxfp8_reference(torch.randn(m, k, generator=gen).to(torch.bfloat16))
    b, b_s = G.quantize_mxfp8_reference(torch.randn(n, k, generator=gen).to(torch.bfloat16))
    out = G.rubin_mxfp8_gemm.reference(
        a, b, G.mxfp8_scales_to_r128c4(a_s), G.mxfp8_scales_to_r128c4(b_s)
    )
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    # Independently: float64 values times 2^(byte - 127), row-major scales, no R128c4.
    da = a.double().view(m, k // 32, 32) * torch.exp2(a_s.double() - 127.0).unsqueeze(-1)
    db = b.double().view(n, k // 32, 32) * torch.exp2(b_s.double() - 127.0).unsqueeze(-1)
    exact = da.view(m, k) @ db.view(n, k).t()
    # One BF16 rounding (at most 2^-9 of the value) on an FP32 sum: within 2^-8 of the row's
    # largest magnitude.
    row_max = exact.abs().amax(dim=1, keepdim=True)
    assert torch.all((out.double() - exact).abs() <= row_max * 2.0**-8)
