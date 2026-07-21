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

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.modules.linear as linear_module
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MXFP8LinearMethod,
    flashinfer_mxfp8_decode_graph_capture,
    get_quant_method,
)
from tensorrt_llm._torch.modules.mxfp8_utils import dequant_mxfp8_weight, quant_bf16_to_mxfp8
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def test_quant_dequant_roundtrip_is_close():
    torch.manual_seed(0)
    out_features, in_features = 64, 128  # in_features divisible by 32
    w = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    w_e4m3, scale_ue8m0 = quant_bf16_to_mxfp8(w, block_size=32)
    assert w_e4m3.dtype == torch.float8_e4m3fn
    assert scale_ue8m0.dtype == torch.uint8
    assert scale_ue8m0.shape == (out_features, in_features // 32)

    w_deq = dequant_mxfp8_weight(w_e4m3, scale_ue8m0, block_size=32)
    assert w_deq.shape == (out_features, in_features)
    # MXFP8 is coarse; check relative error of the reconstructed matmul output.
    x = torch.randn(8, in_features, dtype=torch.bfloat16)
    ref = x.float() @ w.float().t()
    got = x.float() @ w_deq.float().t()
    rel = (got - ref).norm() / ref.norm().clamp_min(1e-6)
    assert rel < 0.1, f"relative error too high: {rel}"


def test_mxfp8_dispatch_returns_mxfp8_method(monkeypatch):
    """get_quant_method must dispatch QuantAlgo.MXFP8 to MXFP8LinearMethod.

    This is a pure dispatch check; no CUDA required.
    """
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    qc = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)
    method = get_quant_method(qc)
    assert isinstance(method, MXFP8LinearMethod)
    assert method.backend == "trtllm"


def _mock_mxfp8_ops(monkeypatch):
    quantized = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
    activation_scale = torch.empty(512, dtype=torch.uint8)
    quantize = Mock(return_value=(quantized, activation_scale))
    native_output = torch.empty((2, 3), dtype=torch.bfloat16)
    native_gemm = Mock(return_value=native_output)
    fake_trtllm_ops = SimpleNamespace(mxfp8_quantize=quantize, mxfp8_mxfp8_gemm=native_gemm)
    monkeypatch.setattr(linear_module.torch, "ops", SimpleNamespace(trtllm=fake_trtllm_ops))
    return quantized, activation_scale, quantize, native_gemm, native_output


def test_mxfp8_flashinfer_call_contract(monkeypatch):
    """The forced backend reuses TRT tensors and a zero-copy weight transpose."""
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", "flashinfer")
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)

    expected = torch.empty((2, 3), dtype=torch.bfloat16)
    mm_mxfp8 = Mock(return_value=expected)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(mm_mxfp8=mm_mxfp8))
    quantized, activation_scale, quantize, _, _ = _mock_mxfp8_ops(monkeypatch)

    weight = torch.empty((3, 4), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(512, dtype=torch.uint8)
    module = SimpleNamespace(weight=weight, weight_scale=weight_scale, dtype=torch.bfloat16)
    activation = torch.randn((2, 4), dtype=torch.bfloat16)

    method = MXFP8LinearMethod()
    output = method.apply(module, activation, bias=None)

    assert output is expected
    quantize.assert_called_once_with(activation, True)
    args = mm_mxfp8.call_args.args
    kwargs = mm_mxfp8.call_args.kwargs
    assert args[0] is quantized
    assert args[1].shape == (4, 3)
    assert args[1].data_ptr() == weight.data_ptr()
    assert args[2] is activation_scale
    assert args[3] is weight_scale
    assert kwargs == {
        "out_dtype": torch.bfloat16,
        "use_8x4_sf_layout": False,
        "backend": "cutlass",
    }


def test_mxfp8_auto_keeps_eager_native_and_captures_flashinfer(monkeypatch):
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)

    flashinfer_output = torch.empty((2, 3), dtype=torch.bfloat16)
    mm_mxfp8 = Mock(return_value=flashinfer_output)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(mm_mxfp8=mm_mxfp8))
    _, _, _, native_gemm, native_output = _mock_mxfp8_ops(monkeypatch)

    module = SimpleNamespace(
        weight=torch.empty((3, 4), dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(512, dtype=torch.uint8),
        dtype=torch.bfloat16,
    )
    activation = torch.randn((2, 4), dtype=torch.bfloat16)
    method = MXFP8LinearMethod()
    assert method.enable_flashinfer_auto()

    assert method.apply(module, activation, bias=None) is native_output
    native_gemm.assert_called_once()
    mm_mxfp8.assert_not_called()

    method.mark_flashinfer_autotuned()
    with flashinfer_mxfp8_decode_graph_capture():
        assert method.apply(module, activation, bias=None) is flashinfer_output
    mm_mxfp8.assert_called_once()

    # Leaving the decode-capture scope restores the eager/native path.
    assert method.apply(module, activation, bias=None) is native_output
    assert native_gemm.call_count == 2


def test_mxfp8_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", "unknown")
    with pytest.raises(ValueError, match="TRTLLM_MXFP8_GEMM_BACKEND"):
        MXFP8LinearMethod()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP8 Linear load path requires CUDA")
def test_mxfp8_linear_reference_matches_dequant():
    """End-to-end Linear MXFP8 forward (whichever path is active) vs the
    out-of-module dequant reference. Uses norm-relative tolerance because the
    CUTLASS path's per-element error is larger than 2% but the aggregate
    output is still aligned (this is fundamental to MXFP8's coarse 32-element
    block scaling, not a kernel bug).
    """
    torch.manual_seed(0)
    out_f, in_f = 128, 256
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16)
    w_e4m3, scale = quant_bf16_to_mxfp8(w, 32)

    qc = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)
    lin = Linear(
        in_features=in_f, out_features=out_f, bias=False, dtype=torch.bfloat16, quant_config=qc
    ).cuda()
    # Mirror the checkpoint key naming (`weight_scale_inv`).
    lin.load_weights([{"weight": w_e4m3, "weight_scale_inv": scale}])

    x = torch.randn(4, in_f, dtype=torch.bfloat16, device="cuda")
    got = lin(x)
    w_deq = dequant_mxfp8_weight(w_e4m3, scale, 32).cuda()
    ref = (x.float() @ w_deq.t()).to(torch.bfloat16)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert rel < 0.1, f"rel err {rel} (got={got.dtype}, ref={ref.dtype})"


def _mxfp8_cutlass_op_available():
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 10:
        return False
    return hasattr(torch.ops.trtllm, "mxfp8_mxfp8_gemm")


@pytest.mark.skipif(
    not _mxfp8_cutlass_op_available(), reason="MXFP8xMXFP8 GEMM op not compiled or sm < 100"
)
def test_mxfp8_linear_cutlass_matches_reference():
    """End-to-end CUTLASS path: must agree with the dequant reference."""
    torch.manual_seed(0)
    out_f, in_f = 256, 512
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16)
    w_e4m3, scale = quant_bf16_to_mxfp8(w, 32)
    x = torch.randn(16, in_f, dtype=torch.bfloat16, device="cuda")

    qc = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)
    lin = Linear(
        in_features=in_f, out_features=out_f, bias=False, dtype=torch.bfloat16, quant_config=qc
    ).cuda()
    lin.load_weights([{"weight": w_e4m3, "weight_scale_inv": scale}])

    got = lin(x)
    w_deq = dequant_mxfp8_weight(w_e4m3, scale, 32).cuda()
    ref = (x.float() @ w_deq.t()).to(torch.bfloat16)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert rel < 0.05, f"CUTLASS vs reference rel err {rel}"
