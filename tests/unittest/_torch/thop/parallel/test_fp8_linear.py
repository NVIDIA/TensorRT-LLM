# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod, Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 correctness requires Hopper or newer",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [1, 8, 9, 64])
def test_fp8_linear(dtype, seq_len):
    SEQ_LEN = seq_len
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 256
    torch.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_fp8, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
    x_fp8 = x_fp8.view(torch.float8_e4m3fn)
    x_scale = x_scale.float().squeeze()
    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_fp8, w_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(w)
    w_fp8 = w_fp8.view(torch.float8_e4m3fn)
    w_scale = w_scale.float().squeeze()

    qc = QuantConfig(quant_algo=QuantAlgo.FP8)
    l0 = Linear(in_features=HIDDEN_SIZE,
                out_features=OUTPUT_SIZE,
                bias=False,
                dtype=dtype,
                quant_config=qc)
    assert l0.weight.dtype == torch.float8_e4m3fn
    l0.load_weights([{
        'weight': w_fp8,
        'weight_scale': w_scale,
        'input_scale': x_scale
    }])
    l0.cuda()
    torch.testing.assert_close(l0.weight, w_fp8)
    torch.testing.assert_close(l0.weight_scale, w_scale)
    torch.testing.assert_close(l0.input_scale, x_scale)

    with torch.inference_mode():
        output = l0.forward(x)

    # torch run
    def ref_quant(x_, x_scale_):
        x_ = x_.float()
        finfo = torch.finfo(torch.float8_e4m3fn)
        inv_scale = x_scale_.reciprocal()
        x_fp8_ = (x_ * inv_scale).clamp(min=finfo.min, max=finfo.max)
        return x_fp8_.to(torch.float8_e4m3fn)

    def ref_linear():
        ref_x_fp8 = ref_quant(x, x_scale)
        # Align cublaslt workspace size with trtllm's 32MB.
        # Details see in test_scaled_mm.py
        old_env = os.environ.get("CUBLASLT_WORKSPACE_SIZE", "")
        os.environ["CUBLASLT_WORKSPACE_SIZE"] = f"{32*1024}"
        ref_output = torch._scaled_mm(ref_x_fp8,
                                      w_fp8.t(),
                                      out_dtype=dtype,
                                      scale_a=x_scale,
                                      scale_b=w_scale,
                                      use_fast_accum=True,
                                      bias=l0.bias)
        os.environ["CUBLASLT_WORKSPACE_SIZE"] = old_env
        return ref_output

    with torch.inference_mode():
        ref_output = ref_linear()

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output)


@pytest.mark.parametrize("seq_len", [1, 8, 9, 64])
@pytest.mark.parametrize("enable_cuda_core", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_fp8_linear_dispatch(monkeypatch, seq_len, enable_cuda_core, batched):
    """Exercise the real apply method without launching CUDA kernels."""
    shape = (1, seq_len, 128) if batched else (seq_len, 128)
    inputs = torch.zeros(shape).to(torch.float8_e4m3fn)
    module = SimpleNamespace(
        weight=torch.zeros((256, 128)).to(torch.float8_e4m3fn),
        weight_scale=torch.tensor(0.5),
        input_scale=torch.tensor(0.25),
        force_dynamic_quantization=False,
        dtype=torch.bfloat16,
        enable_cuda_core=enable_cuda_core,
        all_reduce=None,
    )
    expected = torch.zeros((seq_len, 256), dtype=module.dtype)
    cuda_mm = Mock(return_value=expected)
    cublas_mm = Mock(return_value=expected)
    monkeypatch.setattr(torch.ops.trtllm, "cuda_scaled_mm", cuda_mm)
    monkeypatch.setattr(torch.ops.trtllm, "cublas_scaled_mm", cublas_mm)

    output = FP8QDQLinearMethod().apply(module, inputs, bias=None)

    selected, unused = ((cuda_mm,
                         cublas_mm) if enable_cuda_core and seq_len <= 8 else
                        (cublas_mm, cuda_mm))
    selected.assert_called_once()
    unused.assert_not_called()
    args, kwargs = selected.call_args
    assert args[0].shape == (seq_len, 128)
    assert args[0].data_ptr() == inputs.data_ptr()
    assert args[1].shape == (128, 256)
    assert args[1].data_ptr() == module.weight.data_ptr()
    assert args[1].stride() == module.weight.t().stride()
    assert kwargs["scale_a"] is module.input_scale
    assert kwargs["scale_b"] is module.weight_scale
    assert kwargs["out_dtype"] == module.dtype
    assert kwargs["bias"] is None
    assert kwargs["group"] is None
    assert output.shape == (*shape[:-1], 256)
    assert output.data_ptr() == expected.data_ptr()
