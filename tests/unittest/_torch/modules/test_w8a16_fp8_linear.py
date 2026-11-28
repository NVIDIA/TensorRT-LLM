# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modules.linear import W8A16FP8LinearMethod

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def flashinfer_w8a16():
    from flashinfer.gemm import mm_bf16_fp8, mm_fp16_fp8

    major, minor = torch.cuda.get_device_capability()
    if not mm_bf16_fp8.is_compute_capability_supported(major * 10 + minor):
        pytest.skip("requires a GPU supported by FlashInfer W8A16")
    assert IS_CUTLASS_DSL_AVAILABLE, "FlashInfer W8A16 requires CuTe DSL"
    return {torch.bfloat16: mm_bf16_fp8, torch.float16: mm_fp16_fp8}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compiled_flashinfer_and_strided_fallback(dtype, flashinfer_w8a16):
    generator = torch.Generator(device="cuda").manual_seed(42)
    module = SimpleNamespace(
        weight=torch.randn((128, 128), device="cuda", generator=generator).to(torch.float8_e4m3fn),
        weight_scale=torch.tensor(0.013, device="cuda"),
    )
    bias = torch.randn((128,), device="cuda", dtype=dtype, generator=generator) / 16
    strided = torch.randn((128, 8), device="cuda", dtype=dtype, generator=generator).T
    contiguous = strided.contiguous()
    method = W8A16FP8LinearMethod(backend="flashinfer")
    torch_method = W8A16FP8LinearMethod(backend="torch")

    with torch.inference_mode():
        run = torch.compile(lambda x: method.apply(module, x, bias), fullgraph=True)
        run_torch = torch.compile(lambda x: torch_method.apply(module, x, bias), fullgraph=True)
        expected_fused = flashinfer_w8a16[dtype](
            contiguous, module.weight, module.weight_scale, bias=bias
        )
        expected_torch = run_torch(strided)
        assert not torch.equal(expected_fused, expected_torch)
        torch.testing.assert_close(run(contiguous), expected_fused, atol=0, rtol=0)
        torch.testing.assert_close(run(strided), expected_torch, atol=0, rtol=0)


@pytest.mark.parametrize("n,k", [(127, 128), (128, 127), (0, 128), (128, 0)])
def test_flashinfer_falls_back_for_unsupported_dimensions(n, k, flashinfer_w8a16):
    module = SimpleNamespace(
        weight=torch.ones((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn),
        weight_scale=torch.tensor(0.5, device="cuda"),
    )
    x = torch.ones((2, k), device="cuda", dtype=torch.bfloat16)
    expected = W8A16FP8LinearMethod(backend="torch").apply(module, x, bias=None)
    actual = W8A16FP8LinearMethod(backend="flashinfer").apply(module, x, bias=None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
