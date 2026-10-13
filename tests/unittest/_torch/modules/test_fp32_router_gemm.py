# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tensorrt_llm._torch.modules.fp32_router_gemm import fp32_router_gemm


# The GEMV keeps the FP32 mantissa that TF32 cuBLAS would round off.
@pytest.mark.parametrize("num_tokens", [1, 3, 8, 16])
def test_fp32_router_gemm_matches_fp64(num_tokens):
    torch.manual_seed(0)
    # Rows strided like a slice of a wider activation.
    hidden_states = torch.randn(num_tokens, 8192, dtype=torch.bfloat16, device="cuda")[:, :6144]
    weight = torch.randn(128, 6144, dtype=torch.float32, device="cuda") * 0.02
    ref = hidden_states.double() @ weight.double().t()
    torch.testing.assert_close(
        fp32_router_gemm(hidden_states, weight), ref.float(), rtol=1e-5, atol=1e-4
    )
    torch.library.opcheck(fp32_router_gemm, (hidden_states, weight))
