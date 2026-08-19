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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.modules.linear as linear_module
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
    MXFP8GemmRunner,
    _get_mxfp8_large_m_tuning_buckets,
    _map_to_mxfp8_large_m_bucket,
)
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MXFP8LinearMethod,
    WeightMode,
    WeightsLoadingConfig,
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
    assert method.use_native_autotuner


def test_mxfp8_fused_qkv_creates_nvfp4_kv_scales(monkeypatch):
    """MXFP8 QKV weights retain the scales required by an NVFP4 KV cache."""
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: False)
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.MXFP8,
        kv_cache_quant_algo=QuantAlgo.NVFP4,
        group_size=32,
    )
    linear = Linear(
        in_features=128,
        out_features=384,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=quant_config,
        weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_QKV_LINEAR),
    )

    torch.testing.assert_close(linear.kv_scales, torch.ones(3))
    torch.testing.assert_close(linear.inv_kv_scales, torch.ones(3))


def _mock_mxfp8_ops(monkeypatch):
    quantized = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
    activation_scale = torch.empty(512, dtype=torch.uint8)
    quantize = Mock(return_value=(quantized, activation_scale))
    native_output = torch.empty((2, 3), dtype=torch.bfloat16)
    native_gemm = Mock(return_value=native_output)
    autotuned_output = torch.empty((2, 3), dtype=torch.bfloat16)
    autotuned_gemm = Mock(return_value=autotuned_output)
    fake_trtllm_ops = SimpleNamespace(
        mxfp8_quantize=quantize,
        mxfp8_mxfp8_gemm=native_gemm,
        mxfp8_mxfp8_gemm_autotuned=autotuned_gemm,
    )
    monkeypatch.setattr(linear_module.torch, "ops", SimpleNamespace(trtllm=fake_trtllm_ops))
    return (
        quantized,
        activation_scale,
        quantize,
        native_gemm,
        native_output,
        autotuned_gemm,
        autotuned_output,
    )


def _mock_flashinfer_mxfp8_op(monkeypatch, output):
    """Stand in for the trtllm::flashinfer_mm_mxfp8 wrapper op."""
    op = Mock(return_value=output)
    monkeypatch.setattr(linear_module, "_flashinfer_mxfp8_op", lambda: op)
    return op


def test_mxfp8_flashinfer_call_contract(monkeypatch):
    """The forced backend hands the wrapper op the TRT-LLM quantized tensors."""
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", "flashinfer")
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)

    expected = torch.empty((2, 3), dtype=torch.bfloat16)
    flashinfer_gemm = _mock_flashinfer_mxfp8_op(monkeypatch, expected)
    quantized, activation_scale, quantize, _, _, _, _ = _mock_mxfp8_ops(monkeypatch)

    weight = torch.empty((3, 4), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(512, dtype=torch.uint8)
    module = SimpleNamespace(weight=weight, weight_scale=weight_scale, dtype=torch.bfloat16)
    activation = torch.randn((2, 4), dtype=torch.bfloat16)

    method = MXFP8LinearMethod()
    output = method.apply(module, activation, bias=None)

    assert output is expected
    quantize.assert_called_once_with(activation, True)
    # The [K, N] transpose and the mm_mxfp8 kwargs live inside the wrapper op,
    # so the weight is passed through in its stored [N, K] layout.
    flashinfer_gemm.assert_called_once_with(
        quantized, activation_scale, weight, weight_scale, torch.bfloat16
    )


def test_mxfp8_explicit_flashinfer_survives_torch_compile(monkeypatch):
    """A pinned backend is a compile-time constant dispatched through an op."""
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", "flashinfer")
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)
    monkeypatch.setattr(linear_module, "is_torch_compiling", lambda: True)

    expected = torch.empty((2, 3), dtype=torch.bfloat16)
    flashinfer_gemm = _mock_flashinfer_mxfp8_op(monkeypatch, expected)
    _, _, _, native_gemm, _, _, _ = _mock_mxfp8_ops(monkeypatch)

    module = SimpleNamespace(
        weight=torch.empty((3, 4), dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(512, dtype=torch.uint8),
        dtype=torch.bfloat16,
    )
    activation = torch.randn((2, 4), dtype=torch.bfloat16)
    method = MXFP8LinearMethod()

    assert method.apply(module, activation, bias=None) is expected
    flashinfer_gemm.assert_called_once()
    native_gemm.assert_not_called()


def test_mxfp8_auto_keeps_eager_native_and_captures_flashinfer(monkeypatch):
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)

    flashinfer_output = torch.empty((2, 3), dtype=torch.bfloat16)
    flashinfer_gemm = _mock_flashinfer_mxfp8_op(monkeypatch, flashinfer_output)
    _, _, _, native_gemm, native_output, _, _ = _mock_mxfp8_ops(monkeypatch)

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
    flashinfer_gemm.assert_not_called()

    method.mark_flashinfer_autotuned()
    with flashinfer_mxfp8_decode_graph_capture():
        assert method.apply(module, activation, bias=None) is flashinfer_output
    flashinfer_gemm.assert_called_once()

    # Leaving the decode-capture scope restores the eager/native path.
    assert method.apply(module, activation, bias=None) is native_output
    assert native_gemm.call_count == 2


def test_mxfp8_graph_backend_selection_is_an_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)
    monkeypatch.setattr(linear_module, "is_flashinfer_mxfp8_cute_dsl_available", lambda: True)

    graph_output = torch.empty((2, 3), dtype=torch.bfloat16)
    flashinfer_gemm = _mock_flashinfer_mxfp8_op(monkeypatch, graph_output)
    quantized, activation_scale, _, native_gemm, native_output, _, _ = _mock_mxfp8_ops(monkeypatch)
    graph_quantize = Mock(return_value=(quantized, activation_scale))
    graph_gemm = Mock(return_value=graph_output)
    monkeypatch.setattr(linear_module, "mxfp8_quantize_autotuned", graph_quantize)
    monkeypatch.setattr(linear_module, "flashinfer_mxfp8_gemm_autotuned", graph_gemm)

    module = SimpleNamespace(
        weight=torch.empty((3, 4), dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(512, dtype=torch.uint8),
        dtype=torch.bfloat16,
    )
    activation = torch.randn((2, 4), dtype=torch.bfloat16)
    method = MXFP8LinearMethod()
    method.configure_default_graph_dispatch(enable_backend_tuning=True)
    assert method.uses_graph_backend_selection
    assert not method.needs_flashinfer_autotune

    assert method.apply(module, activation, bias=None) is native_output
    native_gemm.assert_called_once()
    flashinfer_gemm.assert_not_called()

    with flashinfer_mxfp8_decode_graph_capture(tune_backends=True):
        assert method.apply(module, activation, bias=None) is graph_output
    graph_quantize.assert_called_once_with(activation, tune=True)
    graph_gemm.assert_called_once_with(
        quantized,
        activation_scale,
        module.weight,
        module.weight_scale,
        module.dtype,
        tune=True,
    )
    flashinfer_gemm.assert_not_called()

    # Leaving the decode-capture scope restores the eager/native path.
    assert method.apply(module, activation, bias=None) is native_output
    assert native_gemm.call_count == 2


def test_mxfp8_auto_stays_native_under_torch_compile(monkeypatch):
    """Dynamo cannot trace the context lookups that gate FlashInfer dispatch."""
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)
    monkeypatch.setattr(linear_module, "is_torch_compiling", lambda: True)

    flashinfer_gemm = _mock_flashinfer_mxfp8_op(
        monkeypatch, torch.empty((2, 3), dtype=torch.bfloat16)
    )
    _, _, _, native_gemm, native_output, _, _ = _mock_mxfp8_ops(monkeypatch)

    module = SimpleNamespace(
        weight=torch.empty((3, 4), dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(512, dtype=torch.uint8),
        dtype=torch.bfloat16,
    )
    activation = torch.randn((2, 4), dtype=torch.bfloat16)
    method = MXFP8LinearMethod()
    assert method.enable_flashinfer_auto()
    method.mark_flashinfer_autotuned()

    with flashinfer_mxfp8_decode_graph_capture():
        assert method.apply(module, activation, bias=None) is native_output
    native_gemm.assert_called_once()
    flashinfer_gemm.assert_not_called()


@pytest.mark.parametrize(
    "num_tokens,expected",
    [
        (1, 1),
        (6552, 6552),
        (6553, 8192),
        (8192, 8192),
        (8193, 8193),
        (13105, 13105),
        (13106, 16384),
        (16384, 16384),
        (16385, 16385),
        (19658, 19658),
        (19659, 32768),
        (32768, 32768),
        (32769, 32769),
    ],
)
def test_mxfp8_large_m_bucket_mapping(num_tokens, expected):
    assert _map_to_mxfp8_large_m_bucket(num_tokens) == expected


@pytest.mark.parametrize(
    "max_num_tokens,expected",
    [
        (4096, ()),
        (6599, (8192,)),
        (14906, (8192, 16384)),
        (29765, (8192, 16384, 32768)),
    ],
)
def test_mxfp8_large_m_tuning_buckets(max_num_tokens, expected):
    assert _get_mxfp8_large_m_tuning_buckets(max_num_tokens) == expected


def test_mxfp8_large_m_cache_profile_maps_act_and_constrains_scale():
    AutoTuner._find_nearest_profile.cache_clear()
    input_shapes = (
        torch.Size((6599, 6144)),
        torch.Size((1277952,)),
        torch.Size((9216, 6144)),
        torch.Size((1769472,)),
        torch.Size((1,)),
    )
    profile = AutoTuner._find_nearest_profile(
        input_shapes,
        MXFP8GemmRunner.tuning_config.dynamic_tensor_specs,
        MXFP8GemmRunner.tuning_config.constraint_specs,
    )
    assert profile == (
        (8192, 6144),
        (-1,),
        (9216, 6144),
        (1769472,),
        (1,),
    )


def test_mxfp8_native_autotuner_dispatch(monkeypatch):
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)
    _, _, _, native_gemm, native_output, autotuned_gemm, autotuned_output = _mock_mxfp8_ops(
        monkeypatch
    )

    module = SimpleNamespace(
        weight=torch.empty((3, 4), dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(512, dtype=torch.uint8),
        dtype=torch.bfloat16,
    )
    activation = torch.randn((2, 4), dtype=torch.bfloat16)

    method = MXFP8LinearMethod()
    assert method.use_native_autotuner
    assert method.needs_native_autotune
    assert method.apply(module, activation, bias=None) is autotuned_output
    autotuned_gemm.assert_called_once()
    native_gemm.assert_not_called()

    method.mark_native_autotuned()
    assert not method.needs_native_autotune
    assert method.apply(module, activation, bias=None) is native_output
    autotuned_gemm.assert_called_once()
    native_gemm.assert_called_once()


def test_mxfp8_native_autotuner_syncs_profiles():
    runner = object.__new__(MXFP8GemmRunner)
    runner.output_dtype = torch.bfloat16
    runner.sm_version = 100
    runner.mxfp8_gemm_runner = Mock()
    profile = (
        (8192, 6144),
        (-1,),
        (9216, 6144),
        (1769472,),
        (1,),
    )
    cache_key = (
        "trtllm::mxfp8_mxfp8_gemm_autotuned::gemm",
        "MXFP8GemmRunner",
        str(runner.unique_id()),
        profile,
    )
    profiling_cache = Mock()
    profiling_cache.get_specific_custom_op.return_value = {cache_key: (0, 17, 0.25)}

    runner.sync_tactic_cache(SimpleNamespace(profiling_cache=profiling_cache))

    runner.mxfp8_gemm_runner.register_tactic.assert_called_once_with(8192, 9216, 6144, 17)


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
