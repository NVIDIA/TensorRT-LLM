# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


def fused_a_gemm_ref(input, weight, bias, dtype):
    logits_ref = torch.matmul(input, weight)
    return logits_ref


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("hd_out", [2112])
@pytest.mark.parametrize("hd_in", [7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_a_gemm_run(num_tokens, hd_out, hd_in, dtype):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    device = torch.device("cuda")
    input = torch.randn(num_tokens, hd_in, dtype=dtype, device=device)
    weight = torch.randn((hd_out, hd_in), dtype=dtype, device=device)
    bias = None
    logits = torch.ops.trtllm.dsv3_fused_a_gemm_op(input, weight.t(), bias,
                                                   dtype)
    logtis_ref = fused_a_gemm_ref(input, weight.t(), bias, dtype)
    assert torch.allclose(logits, logtis_ref, rtol=0.1)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Fused MXFP8 epilogue requires SM100 or newer",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 9, 16])
def test_fused_a_gemm_mxfp8_matches_separate_quantize(num_tokens):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    device = torch.device("cuda")
    input = torch.randn(num_tokens, 7168, dtype=torch.bfloat16, device=device)
    weight = torch.randn(3584, 7168, dtype=torch.bfloat16, device=device)

    bf16_output = torch.ops.trtllm.dsv3_fused_a_gemm_op(input, weight.t(), None,
                                                        torch.bfloat16)
    expected, expected_sf = torch.ops.trtllm.mxfp8_quantize(bf16_output,
                                                            False,
                                                            alignment=512)
    actual, actual_sf = torch.ops.trtllm.dsv3_fused_a_gemm_mxfp8_op(
        input, weight.t())

    torch.testing.assert_close(actual.view(torch.uint8),
                               expected.view(torch.uint8),
                               rtol=0,
                               atol=0)
    torch.testing.assert_close(actual_sf,
                               expected_sf.view(num_tokens, -1),
                               rtol=0,
                               atol=0)
