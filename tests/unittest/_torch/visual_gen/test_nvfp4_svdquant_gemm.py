# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Qwen-Image NVFP4 SVDQuant fused kernels (SM100 / Blackwell):
- nvfp4_svdquant_gemm   : residual NVFP4 GEMM + rank-r LoRA-up == fp4_gemm + D @ L1ᵀ
- nvfp4_quantize_smooth : NVFP4-quantize(x * pre_quant_scale) byte-identical to fp4_quantize(x*s)
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm torch ops)

_IS_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
skip_sm100 = pytest.mark.skipif(
    not _IS_SM100, reason="NVFP4 SVDQuant kernels require SM100 (Blackwell)"
)

_RANK = 32  # SVDQuant low-rank r (== the kernel's LoRA K)


def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    err = (ref - got).float()
    noise = (err**2).mean()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10((ref.float() ** 2).mean() / noise))


@skip_sm100
@pytest.mark.parametrize("m, n, k", [(256, 3072, 3072), (6912, 3072, 3072), (6912, 3072, 12288)])
def test_nvfp4_svdquant_gemm(m, n, k):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev) / (k**0.25)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=dev) / (k**0.25)
    gx = ((448.0 * 6.0) / x.float().abs().max()).reshape(1).contiguous().cuda()
    gw = ((448.0 * 6.0) / w.float().abs().max()).reshape(1).contiguous().cuda()
    xq, x_sf = torch.ops.trtllm.fp4_quantize(x, gx, 16, False, True)
    wq, w_sf = torch.ops.trtllm.fp4_quantize(w, gw, 16, False, True)
    alpha = (1.0 / (gx * gw)).reshape(1).contiguous()
    D = torch.randn(m, _RANK, dtype=torch.bfloat16, device=dev) / (_RANK**0.25)
    L1 = torch.randn(n, _RANK, dtype=torch.bfloat16, device=dev) / (_RANK**0.25)
    # 1/alpha is folded into L1 so the epilogue out = alpha*acc yields the unscaled D @ L1ᵀ.
    L1_scaled = (L1.float() / float(alpha[0])).to(torch.bfloat16).contiguous()

    ref = torch.ops.trtllm.fp4_gemm(xq, wq, x_sf, w_sf, alpha, 0, torch.bfloat16).float()
    ref = ref + D.float() @ L1.float().t()
    out = torch.ops.trtllm.nvfp4_svdquant_gemm(
        xq.view(torch.uint8),
        wq.view(torch.uint8),
        x_sf.view(torch.uint8),
        w_sf.view(torch.uint8),
        alpha,
        D,
        L1_scaled,
        torch.bfloat16,
    )
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert _sqnr_db(ref, out.float()) > 40.0


@skip_sm100
@pytest.mark.parametrize("m, k", [(256, 3072), (6912, 12288)])
def test_nvfp4_quantize_smooth(m, k):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev) / (k**0.25)
    pqs = (1.0 + 0.3 * torch.randn(k, dtype=torch.bfloat16, device=dev)).abs()
    gs = ((448.0 * 6.0) / (x.float() * pqs.float()).abs().max()).reshape(1).contiguous()
    # Reference: quantize the pre-smoothed activation with the stock fp4_quantize.
    xq_ref, sf_ref = torch.ops.trtllm.fp4_quantize(
        (x * pqs).to(torch.bfloat16), gs, 16, False, True
    )
    xq, sf = torch.ops.trtllm.nvfp4_quantize_smooth(x, pqs, gs)
    assert torch.equal(xq.view(torch.uint8), xq_ref.view(torch.uint8))
    assert torch.equal(sf.view(torch.uint8), sf_ref.view(torch.uint8))
