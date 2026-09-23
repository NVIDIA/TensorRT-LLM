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

import tensorrt_llm._torch.custom_ops.flashinfer_custom_ops as flashinfer_ops_module
import tensorrt_llm._torch.custom_ops.torch_custom_ops as custom_ops_module
import tensorrt_llm._torch.modules.linear as linear_module
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
    IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE,
    FlashInferMXFP8GemmRunner,
    MXFP8GemmRunner,
    MXFP8QuantizeRunner,
    _get_mxfp8_large_m_tuning_buckets,
    _map_to_mxfp8_large_m_bucket,
)
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MXFP8LinearMethod,
    WeightMode,
    WeightsLoadingConfig,
    _load_kv_cache_scales,
    flashinfer_mxfp8_autotune,
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
    assert not method.use_native_autotuner


@pytest.mark.cpu_only
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
    pointers = (linear.kv_scales.data_ptr(), linear.inv_kv_scales.data_ptr())

    linear.quant_method.load_kv_cache_scales(
        linear, [{"k_scale": torch.tensor(0.5)}, {"v_scale": torch.tensor(0.25)}]
    )
    torch.testing.assert_close(linear.kv_scales, torch.tensor([1.0, 0.5, 0.25]))
    torch.testing.assert_close(linear.inv_kv_scales, torch.tensor([1.0, 2.0, 4.0]))
    assert (linear.kv_scales.data_ptr(), linear.inv_kv_scales.data_ptr()) == pointers


@pytest.mark.cpu_only
@pytest.mark.parametrize("loader", ["shared", "mxfp8"])
def test_kv_scale_loading_defaults_validation_and_reload(monkeypatch, loader):
    """Both loader inputs obey the same calibration and storage contract."""
    module = torch.nn.Module()
    module.kv_scales = torch.nn.Parameter(torch.ones(3), requires_grad=False)
    module.inv_kv_scales = torch.nn.Parameter(torch.ones(3), requires_grad=False)
    pointers = (module.kv_scales.data_ptr(), module.inv_kv_scales.data_ptr())

    def load(k, v):
        if loader == "shared":
            _load_kv_cache_scales(module, k, v)
        else:
            MXFP8LinearMethod.load_kv_cache_scales(
                None, module, [{"k_scale": s} for s in k] + [{"v_scale": s} for s in v]
            )

    monkeypatch.setenv("TRTLLM_LOAD_KV_SCALES", "1")
    load([], [])
    torch.testing.assert_close(module.kv_scales, torch.ones(3))
    for k, v in [([torch.tensor(0.5)], []), ([], [torch.tensor(0.25)])]:
        with pytest.raises(AssertionError, match="must be loaded together"):
            load(k, v)
    load([torch.tensor(0.25), torch.tensor(0.5)], [torch.tensor(0.25)])
    torch.testing.assert_close(module.kv_scales, torch.tensor([1.0, 0.5, 0.25]))
    torch.testing.assert_close(module.inv_kv_scales, torch.tensor([1.0, 2.0, 4.0]))
    monkeypatch.setenv("TRTLLM_LOAD_KV_SCALES", "0")
    load([torch.tensor(0.125)], [])
    load([torch.ones(2)], [])
    torch.testing.assert_close(module.kv_scales, torch.tensor([1.0, 0.5, 0.25]))
    monkeypatch.setenv("TRTLLM_LOAD_KV_SCALES", "1")
    load([torch.tensor(0.125)], [torch.tensor(0.5)])
    torch.testing.assert_close(module.kv_scales, torch.tensor([1.0, 0.125, 0.5]))
    torch.testing.assert_close(module.inv_kv_scales, torch.tensor([1.0, 8.0, 2.0]))
    assert (module.kv_scales.data_ptr(), module.inv_kv_scales.data_ptr()) == pointers


def _mock_mxfp8_ops(
    monkeypatch: pytest.MonkeyPatch,
    flashinfer_gemm: Mock | None = None,
    *,
    flashinfer_op_available: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Mock, Mock, torch.Tensor, Mock, torch.Tensor]:
    """Replace MXFP8 kernels with CPU doubles and optional FlashInfer registration."""
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
    if flashinfer_op_available:
        fake_trtllm_ops.flashinfer_mm_mxfp8 = flashinfer_gemm or Mock()
    fake_torch = SimpleNamespace(
        ops=SimpleNamespace(trtllm=fake_trtllm_ops),
        ones=torch.ones,
        float32=torch.float32,
    )
    monkeypatch.setattr(linear_module, "torch", fake_torch)
    return (
        quantized,
        activation_scale,
        quantize,
        native_gemm,
        native_output,
        autotuned_gemm,
        autotuned_output,
    )


@pytest.mark.cpu_only
def test_registered_flashinfer_mxfp8_wrapper_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real registered wrapper, mocking only the FlashInfer kernel."""
    expected = torch.full((2, 3), 7, dtype=torch.bfloat16)
    kernel = Mock(return_value=expected)
    monkeypatch.setattr(flashinfer_ops_module, "mm_mxfp8", kernel)
    act = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.float8_e4m3fn)
    act_scale = torch.ones(512, dtype=torch.uint8)
    weight_scale = torch.ones(512, dtype=torch.uint8)

    output = torch.ops.trtllm.flashinfer_mm_mxfp8(
        act, act_scale, weight, weight_scale, torch.bfloat16
    )

    torch.testing.assert_close(output, expected)
    kernel.assert_called_once()
    args = kernel.call_args.args
    assert args[0] is act
    assert args[1].shape == (4, 3)
    assert args[1].stride() == weight.t().stride()
    assert args[1].data_ptr() == weight.data_ptr()
    torch.testing.assert_close(args[1].float(), weight.t().float())
    assert args[2] is act_scale and args[3] is weight_scale
    assert kernel.call_args.kwargs == {
        "out_dtype": torch.bfloat16,
        "use_8x4_sf_layout": False,
        "backend": "cutlass",
    }


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["auto", "flashinfer"])
def test_mxfp8_missing_registered_flashinfer_op(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Allow automatic fallback but reject an unavailable explicitly chosen op."""
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", backend)
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(autotune=Mock()))
    _mock_mxfp8_ops(monkeypatch, flashinfer_op_available=False)
    if backend == "flashinfer":
        with pytest.raises(
            RuntimeError, match="requires the pinned flashinfer-python package"
        ) as exc:
            MXFP8LinearMethod()
        assert str(exc.value.__cause__) == "trtllm::flashinfer_mm_mxfp8 is unavailable"
    else:
        method = MXFP8LinearMethod()
        assert method.backend == "trtllm"
        assert not method.uses_flashinfer
        assert method._flashinfer_mxfp8 is None


def test_mxfp8_flashinfer_call_contract(monkeypatch):
    """The forced backend passes native-layout tensors to the opaque op."""
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", "flashinfer")
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)

    expected = torch.empty((2, 3), dtype=torch.bfloat16)
    mm_mxfp8 = Mock(return_value=expected)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(mm_mxfp8=mm_mxfp8, autotune=Mock()),
    )
    quantized, activation_scale, quantize, _, _, _, _ = _mock_mxfp8_ops(monkeypatch, mm_mxfp8)

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
    assert args[1] is activation_scale
    assert args[2] is weight
    assert args[3] is weight_scale
    assert args[4] == torch.bfloat16
    assert kwargs == {}


def test_mxfp8_auto_keeps_eager_native_and_captures_flashinfer(monkeypatch):
    """Keep native eager GEMM while routing captured work to the opaque wrapper."""
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)

    flashinfer_output = torch.empty((2, 3), dtype=torch.bfloat16)
    mm_mxfp8 = Mock(return_value=flashinfer_output)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(mm_mxfp8=mm_mxfp8, autotune=Mock()),
    )
    _, _, _, native_gemm, native_output, autotuned_gemm, autotuned_output = _mock_mxfp8_ops(
        monkeypatch, mm_mxfp8
    )
    monkeypatch.setattr(linear_module, "is_torch_compiling", lambda: False)

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
    autotuned_gemm.assert_not_called()
    mm_mxfp8.assert_not_called()

    method.enable_native_autotune()
    assert method.apply(module, activation, bias=None) is autotuned_output
    autotuned_gemm.assert_called_once()
    method.mark_native_autotuned()
    method.mark_flashinfer_autotuned()
    with flashinfer_mxfp8_decode_graph_capture():
        assert method.apply(module, activation, bias=None) is flashinfer_output
    mm_mxfp8.assert_called_once()

    # Leaving the decode-capture scope restores the eager/native path.
    assert method.apply(module, activation, bias=None) is native_output
    assert native_gemm.call_count == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["auto", "flashinfer"])
def test_mxfp8_compile_skips_context_dispatch(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Compilation must not read Python ContextVars, even after decode tuning."""
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", backend)
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)
    monkeypatch.setattr(linear_module, "is_torch_compiling", lambda: True)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(autotune=Mock()))
    flashinfer_output = torch.empty((2, 3), dtype=torch.bfloat16)
    flashinfer_gemm = Mock(return_value=flashinfer_output)
    _, _, _, native_gemm, native_output, _, _ = _mock_mxfp8_ops(monkeypatch, flashinfer_gemm)
    for name in (
        "_FLASHINFER_MXFP8_AUTOTUNE_ACTIVE",
        "_FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE",
    ):
        monkeypatch.setattr(
            linear_module, name, SimpleNamespace(get=Mock(side_effect=AssertionError))
        )
    method = MXFP8LinearMethod()
    method.tune_decode_graph_backends = True
    method.mark_flashinfer_autotuned()
    module = SimpleNamespace(
        weight=torch.empty((3, 4), dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(512, dtype=torch.uint8),
        dtype=torch.bfloat16,
    )
    result = method.apply(module, torch.empty((2, 4), dtype=torch.bfloat16), bias=None)
    assert result is (native_output if backend == "auto" else flashinfer_output)
    assert native_gemm.call_count == (backend == "auto")
    assert flashinfer_gemm.call_count == (backend == "flashinfer")


def test_mxfp8_auto_fallback_does_not_rearm_native_autotuning(monkeypatch):
    """Falling back after native warmup keeps serving on the plain native op."""
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(mm_mxfp8=Mock(), autotune=Mock()),
    )
    _, _, _, native_gemm, native_output, autotuned_gemm, _ = _mock_mxfp8_ops(monkeypatch)

    module = SimpleNamespace(
        weight=torch.empty((3, 4), dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(512, dtype=torch.uint8),
        dtype=torch.bfloat16,
    )
    activation = torch.randn((2, 4), dtype=torch.bfloat16)
    method = MXFP8LinearMethod()
    method.mark_native_autotuned()
    assert method.enable_flashinfer_auto()

    method.disable_flashinfer_auto()

    assert method.backend == "trtllm"
    assert not method.needs_native_autotune
    assert method.apply(module, activation, bias=None) is native_output
    native_gemm.assert_called_once()
    autotuned_gemm.assert_not_called()


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
    assert not method.use_native_autotuner
    assert not method.needs_native_autotune
    assert method.apply(module, activation, bias=None) is native_output
    native_gemm.assert_called_once()
    autotuned_gemm.assert_not_called()

    method.enable_native_autotune()
    assert method.use_native_autotuner
    assert method.needs_native_autotune
    assert method.apply(module, activation, bias=None) is autotuned_output
    autotuned_gemm.assert_called_once()

    method.mark_native_autotuned()
    assert not method.needs_native_autotune
    assert method.apply(module, activation, bias=None) is native_output
    autotuned_gemm.assert_called_once()
    assert native_gemm.call_count == 2


def test_mxfp8_native_autotuner_syncs_all_profiles(monkeypatch):
    bf16_runner = Mock()
    fp16_runner = Mock()
    monkeypatch.setattr(
        MXFP8GemmRunner,
        "runner_dict",
        {
            (torch.bfloat16, 100): bf16_runner,
            (torch.float16, 100): fp16_runner,
        },
    )
    bf16_profile = (
        (8192, 6144),
        (-1,),
        (9216, 6144),
        (1769472,),
        (1,),
    )
    fp16_profile = (
        (16384, 6144),
        (-1,),
        (9216, 6144),
        (3538944,),
        (1,),
    )
    cache = {
        (
            "trtllm::mxfp8_mxfp8_gemm_autotuned::gemm",
            "MXFP8GemmRunner",
            str((torch.bfloat16, 100)),
            bf16_profile,
        ): (0, 17, 0.25),
        (
            "trtllm::mxfp8_mxfp8_gemm_autotuned::gemm",
            "MXFP8GemmRunner",
            str((torch.float16, 100)),
            fp16_profile,
        ): (0, 23, 0.20),
        (
            "trtllm::mxfp8_mxfp8_gemm_autotuned::gemm",
            "OtherRunner",
            str((torch.bfloat16, 100)),
            bf16_profile,
        ): (0, 29, 0.15),
        (
            "trtllm::mxfp8_mxfp8_gemm_autotuned::gemm",
            "MXFP8GemmRunner",
            str((torch.float32, 100)),
            bf16_profile,
        ): (0, 31, 0.10),
    }
    profiling_cache = Mock()
    profiling_cache.get_specific_custom_op.return_value = cache
    tuner = Mock(profiling_cache=profiling_cache)

    MXFP8GemmRunner.sync_all_tactic_caches(tuner)

    bf16_runner.register_tactic.assert_called_once_with(8192, 9216, 6144, 17)
    fp16_runner.register_tactic.assert_called_once_with(16384, 9216, 6144, 23)


def test_mxfp8_native_autotuner_rejects_mismatched_k(monkeypatch):
    native_runner = Mock()
    monkeypatch.setattr(
        MXFP8GemmRunner,
        "runner_dict",
        {(torch.bfloat16, 100): native_runner},
    )
    cache_key = (
        "trtllm::mxfp8_mxfp8_gemm_autotuned::gemm",
        "MXFP8GemmRunner",
        str((torch.bfloat16, 100)),
        (
            (8192, 6144),
            (-1,),
            (9216, 4096),
            (1769472,),
            (1,),
        ),
    )
    profiling_cache = Mock()
    profiling_cache.get_specific_custom_op.return_value = {cache_key: (0, 17, 0.25)}
    tuner = Mock(profiling_cache=profiling_cache)
    with pytest.raises(ValueError, match="mismatched K dimensions"):
        MXFP8GemmRunner.sync_all_tactic_caches(tuner)


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
        in_features=in_f,
        out_features=out_f,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=qc,
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
    not _mxfp8_cutlass_op_available(),
    reason="MXFP8xMXFP8 GEMM op not compiled or sm < 100",
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
        in_features=in_f,
        out_features=out_f,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=qc,
    ).cuda()
    lin.load_weights([{"weight": w_e4m3, "weight_scale_inv": scale}])

    got = lin(x)
    w_deq = dequant_mxfp8_weight(w_e4m3, scale, 32).cuda()
    ref = (x.float() @ w_deq.t()).to(torch.bfloat16)
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert rel < 0.05, f"CUTLASS vs reference rel err {rel}"


@pytest.mark.skipif(
    not _mxfp8_cutlass_op_available(),
    reason="MXFP8xMXFP8 GEMM op not compiled or sm < 100",
)
@pytest.mark.parametrize("batch_size", (1, 8, 16, 32))
def test_mxfp8_flashinfer_decode_graph_matches_native(monkeypatch, batch_size):
    """FlashInfer must consume TRT-LLM's swizzled scales like the native op.

    Tune a large-M warmup shape, then replay several decode graph shapes. This
    protects the decode-only path from a silent scale-layout or tactic-cache
    miss during graph capture.
    """
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        pytest.skip("FlashInfer is not installed")

    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    torch.manual_seed(0)
    out_f, in_f = 256, 512
    weight = torch.randn(out_f, in_f, dtype=torch.bfloat16)
    weight_e4m3, weight_scale = quant_bf16_to_mxfp8(weight, 32)
    warmup_x = torch.randn(128, in_f, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(batch_size, in_f, dtype=torch.bfloat16, device="cuda")
    quant_config = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)

    native = Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=quant_config,
    ).cuda()
    flashinfer_linear = Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=quant_config,
    ).cuda()
    weights = [{"weight": weight_e4m3, "weight_scale_inv": weight_scale}]
    native.load_weights(weights)
    flashinfer_linear.load_weights(weights)
    native_output = native(x)
    method = flashinfer_linear.quant_method
    assert isinstance(method, MXFP8LinearMethod)
    assert method.enable_flashinfer_auto()
    with flashinfer_mxfp8_autotune():
        warmup_output = flashinfer_linear(warmup_x)
    method.mark_flashinfer_autotuned()
    torch.testing.assert_close(warmup_output, native(warmup_x), rtol=2e-2, atol=2e-2)

    flashinfer_gemm = Mock(wraps=method._flashinfer_mxfp8)
    method._flashinfer_mxfp8 = flashinfer_gemm
    static_x = x.clone()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        with flashinfer_mxfp8_decode_graph_capture():
            graph_output = flashinfer_linear(static_x)
    assert flashinfer_gemm.call_count == 1
    graph.replay()
    torch.testing.assert_close(graph_output, native_output, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not _mxfp8_cutlass_op_available(),
    reason="MXFP8xMXFP8 GEMM op not compiled or sm < 100",
)
@pytest.mark.parametrize("batch_size", (1, 8, 16, 32))
def test_mxfp8_decode_graph_backend_tuning_matches_native(monkeypatch, batch_size):
    """Per-bucket tuned decode graphs must match the native op.

    Profile the quantizer and GEMM backends for a decode bucket during the
    warmup-only pass, then capture the same shape and replay it. This covers
    the CuTeDSL scale layouts and the in-process winner cache used by capture.
    """
    if not IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE:
        pytest.skip("FlashInfer CuTeDSL MXFP8 kernels are not available")

    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    torch.manual_seed(0)
    out_f, in_f = 256, 512
    weight = torch.randn(out_f, in_f, dtype=torch.bfloat16)
    weight_e4m3, weight_scale = quant_bf16_to_mxfp8(weight, 32)
    x = torch.randn(batch_size, in_f, dtype=torch.bfloat16, device="cuda")
    quant_config = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)

    native = Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=quant_config,
    ).cuda()
    tuned = Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=quant_config,
    ).cuda()
    weights = [{"weight": weight_e4m3, "weight_scale_inv": weight_scale}]
    native.load_weights(weights)
    tuned.load_weights(weights)
    native_output = native(x)

    method = tuned.quant_method
    assert isinstance(method, MXFP8LinearMethod)
    assert method.enable_flashinfer_auto()
    method.tune_decode_graph_backends = True

    # Record the (quantize, GEMM) tactics each pass selects.
    chosen_tactics = []
    choose_tactic = custom_ops_module._choose_mxfp8_tactic

    def record_tactic(*args, **kwargs):
        tactic = choose_tactic(*args, **kwargs)
        chosen_tactics.append(tactic)
        return tactic

    monkeypatch.setattr(custom_ops_module, "_choose_mxfp8_tactic", record_tactic)

    # Both CuTeDSL candidates must run and match, independent of which
    # backend wins the profiling below.
    act, act_scale = MXFP8QuantizeRunner(x.dtype)([x], tactic=MXFP8QuantizeRunner.CUTE_DSL)
    cute_dsl_output = FlashInferMXFP8GemmRunner(tuned.dtype)(
        [act, act_scale, tuned.weight, tuned.weight_scale],
        tactic=FlashInferMXFP8GemmRunner.CUTE_DSL,
    )
    torch.testing.assert_close(cute_dsl_output, native_output, rtol=2e-2, atol=2e-2)

    # Warmup-only pass: profile both backends of each stage for this bucket.
    with flashinfer_mxfp8_autotune(), flashinfer_mxfp8_decode_graph_capture():
        warmup_output = tuned(x)
    torch.testing.assert_close(warmup_output, native_output, rtol=2e-2, atol=2e-2)
    assert len(chosen_tactics) == 2
    warmup_tactics = tuple(chosen_tactics)
    chosen_tactics.clear()

    # Capture pass: the in-process winners are reused without profiling.
    static_x = x.clone()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        with flashinfer_mxfp8_decode_graph_capture():
            graph_output = tuned(static_x)
    assert tuple(chosen_tactics) == warmup_tactics

    # Replay on a fresh input so the check cannot pass on the captured result.
    replay_x = torch.randn_like(x)
    static_x.copy_(replay_x)
    graph.replay()
    torch.testing.assert_close(graph_output, native(replay_x), rtol=2e-2, atol=2e-2)
