# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm::cublas_scaled_mm GELU / bias epilogues and e4m3 output."""

import pytest
import torch
import torch.nn.functional as F
from utils.util import getSMVersion

import tensorrt_llm  # noqa: F401  -- registers trtllm ops

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or getSMVersion() < 89, reason="FP8 GEMMs need SM89+"
)


def _cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _operands(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", generator=g)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.05
    scale_a = (a.abs().amax() / E4M3_MAX).reshape(1)
    scale_w = (w.abs().amax() / E4M3_MAX).reshape(1)
    a8 = (a / scale_a).to(E4M3)
    w8 = (w / scale_w).to(E4M3)
    bias = (torch.randn(n, device="cuda", generator=g) * 0.1).bfloat16()
    ref = (a8.float() * scale_a) @ (w8.float() * scale_w).t()
    return a8, w8, scale_a, scale_w, bias, ref


@pytest.mark.parametrize("m,n,k", [(4097, 13824, 5120), (2048, 5120, 5120), (33, 3072, 3072)])
@pytest.mark.parametrize("gelu", [False, True])
@pytest.mark.parametrize("out_fp8", [False, True])
def test_epilogue_matches_reference(m, n, k, gelu, out_fp8):
    a8, w8, scale_a, scale_w, bias, ref = _operands(m, n, k, seed=m + n)
    ref = ref + bias.float()
    if gelu:
        ref = F.gelu(ref, approximate="tanh")
    d_scale = (E4M3_MAX / ref.abs().amax()).reshape(1) if out_fp8 else None
    out = torch.ops.trtllm.cublas_scaled_mm(
        a8,
        w8.t(),
        scale_a,
        scale_w,
        bias,
        E4M3 if out_fp8 else torch.bfloat16,
        epilogue=1 if gelu else 0,
        d_scale=d_scale,
    )
    assert out.shape == (m, n) and out.dtype == (E4M3 if out_fp8 else torch.bfloat16)
    result = out.float() / d_scale if out_fp8 else out.float()
    assert _cosine(result, ref) > (0.998 if out_fp8 else 0.9999)


def test_default_epilogue_is_unchanged():
    a8, w8, scale_a, scale_w, bias, _ = _operands(256, 1024, 512, seed=3)
    plain = torch.ops.trtllm.cublas_scaled_mm(a8, w8.t(), scale_a, scale_w, bias, torch.bfloat16)
    explicit = torch.ops.trtllm.cublas_scaled_mm(
        a8, w8.t(), scale_a, scale_w, bias, torch.bfloat16, epilogue=0
    )
    assert torch.equal(plain, explicit)


def test_rejects_bad_arguments():
    a8, w8, scale_a, scale_w, bias, _ = _operands(64, 256, 128, seed=4)
    with pytest.raises(RuntimeError, match="epilogue"):
        torch.ops.trtllm.cublas_scaled_mm(
            a8, w8.t(), scale_a, scale_w, bias, torch.bfloat16, epilogue=2
        )
    with pytest.raises(RuntimeError, match="d_scale"):
        torch.ops.trtllm.cublas_scaled_mm(a8, w8.t(), scale_a, scale_w, bias, E4M3, epilogue=1)
