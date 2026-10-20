# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerics of the MiniMax-M3 CTX prefill's fused quantizers against the unfused ops they replace.

- ``trtllm::minimax_m3_ctx_residual_rmsnorm_mxfp8``: the residual sum it writes back is the BF16 add of the two inputs;
  its RMSNorm output is within a BF16 rounding of an fp64 reference (its row reduction order is the stock rows kernel's,
  not torch's); its MXFP8 output and swizzled scales are bitwise ``mxfp8_quantize(norm, True)`` of that RMSNorm output,
  and its NVFP4 output and linear scales bitwise ``fp4_quantize(norm, scale, 16, False, False)``.
- ``swiglu_mxfp8``: bitwise ``mxfp8_quantize(swiglu(x), True)``, with and without the SwiGLU-OAI clamp / alpha / beta.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.swiglu import swiglu, swiglu_mxfp8
from tensorrt_llm._utils import get_sm_version

HIDDEN = 6144
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100,
    reason="MXFP8 quantization needs an SM100-family GPU",
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the CTX residual RMSNorm + MXFP8 op is built for SM107",
)
@pytest.mark.parametrize("rows", [8192, 8197, 16384])
@pytest.mark.parametrize("fp4", [False, True], ids=["mxfp8", "mxfp8_nvfp4"])
def test_ctx_residual_rmsnorm_mxfp8_matches_the_unfused_ops(rows: int, fp4: bool) -> None:
    gen = torch.Generator(device="cuda").manual_seed(rows + fp4)
    inter = torch.randn((rows, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
    residual = torch.randn((rows, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn((HIDDEN,), generator=gen, device="cuda")).to(torch.bfloat16)
    expected_sum = (inter.float() + residual.float()).to(torch.bfloat16)
    scale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")

    args = (inter, residual, weight, EPS, True) + ((scale,) if fp4 else ())
    norm, q, sf, q4, sf4 = torch.ops.trtllm.minimax_m3_ctx_residual_rmsnorm_mxfp8(*args)
    torch.cuda.synchronize()

    assert torch.equal(_bits(inter), _bits(expected_sum)), "the residual sum written back"
    s = expected_sum.double()
    ref = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + EPS) * weight.double()
    torch.testing.assert_close(norm.double(), ref, rtol=2**-7, atol=1e-3)

    ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(norm, True)
    assert torch.equal(_bits(q), _bits(ref_q)), "MXFP8 values"
    assert torch.equal(sf.view(-1), ref_sf.view(torch.uint8).view(-1)), "MXFP8 swizzled scales"
    if fp4:
        ref_q4, ref_sf4 = torch.ops.trtllm.fp4_quantize(norm, scale, 16, False, False)
        assert torch.equal(q4.view(torch.uint8), ref_q4.view(torch.uint8)), "NVFP4 values"
        assert torch.equal(sf4.view(-1), ref_sf4.view(torch.uint8).view(-1)), "NVFP4 linear scales"
    else:
        assert q4.numel() == 0 and sf4.numel() == 0


@pytest.mark.parametrize("rows", [8192, 8197])
@pytest.mark.parametrize("d", [1536, 6144])
@pytest.mark.parametrize(
    "limit, alpha, beta", [(7.0, 1.702, 1.0), (None, None, None)], ids=["oai", "plain"]
)
def test_swiglu_mxfp8_is_bitwise_the_quantized_swiglu(
    rows: int, d: int, limit, alpha, beta
) -> None:
    gen = torch.Generator(device="cuda").manual_seed(rows * d)
    x = (4.0 * torch.randn((rows, 2 * d), generator=gen, device="cuda")).to(torch.bfloat16)
    q, sf = swiglu_mxfp8(x, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta)
    ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(
        swiglu(x, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta), True
    )
    assert torch.equal(_bits(q), _bits(ref_q)), "MXFP8 values"
    assert torch.equal(sf.view(-1), ref_sf.view(torch.uint8).view(-1)), "MXFP8 swizzled scales"
