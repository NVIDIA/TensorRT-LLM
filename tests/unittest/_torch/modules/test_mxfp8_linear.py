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

import contextlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.custom_ops.torch_custom_ops as torch_custom_ops
import tensorrt_llm._torch.modules.linear as linear_module
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
    FlashInferMXFP8GemmRunner,
    MXFP8GemmRunner,
    MXFP8QuantizeRunner,
    _get_mxfp8_large_m_tuning_buckets,
    _map_to_mxfp8_large_m_bucket,
    is_flashinfer_mxfp8_cute_dsl_available,
)
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MXFP8LinearMethod,
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


def test_mxfp8_flashinfer_call_contract(monkeypatch):
    """The forced backend reuses TRT tensors and a zero-copy weight transpose."""
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", "flashinfer")
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)

    expected = torch.empty((2, 3), dtype=torch.bfloat16)
    mm_mxfp8 = Mock(return_value=expected)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(mm_mxfp8=mm_mxfp8, autotune=Mock()),
    )
    quantized, activation_scale, quantize, _, _, _, _ = _mock_mxfp8_ops(monkeypatch)

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
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(mm_mxfp8=mm_mxfp8, autotune=Mock()),
    )
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


def test_mxfp8_graph_backend_tuning_routes_only_decode_capture(monkeypatch):
    """Per-bucket backend tuning replaces the FlashInfer graph path only inside decode capture."""
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)
    mm_mxfp8 = Mock()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(mm_mxfp8=mm_mxfp8, autotune=Mock(return_value=contextlib.nullcontext())),
    )
    _, _, quantize, native_gemm, native_output, _, _ = _mock_mxfp8_ops(monkeypatch)
    graph_output = torch.empty((2, 3), dtype=torch.bfloat16)
    graph_linear = Mock(return_value=graph_output)
    monkeypatch.setattr(linear_module, "mxfp8_graph_tuned_linear", graph_linear)

    module = SimpleNamespace(
        weight=torch.empty((3, 4), dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(512, dtype=torch.uint8),
        dtype=torch.bfloat16,
    )
    activation = torch.randn((2, 4), dtype=torch.bfloat16)
    method = MXFP8LinearMethod()
    assert method.enable_flashinfer_auto()
    method.tune_graph_backends = True

    # Eager execution stays on the native op.
    assert method.apply(module, activation, bias=None) is native_output
    native_gemm.assert_called_once()
    quantize.assert_called_once_with(activation, True)
    graph_linear.assert_not_called()

    # The eager FlashInfer tuning pass is not graph capture.
    with flashinfer_mxfp8_autotune():
        method.apply(module, activation, bias=None)
    mm_mxfp8.assert_called_once()
    graph_linear.assert_not_called()

    # The warmup-only graph pass tunes both stages for this bucket.
    with flashinfer_mxfp8_autotune(), flashinfer_mxfp8_decode_graph_capture():
        assert method.apply(module, activation, bias=None) is graph_output
    graph_linear.assert_called_once_with(
        activation, module.weight, module.weight_scale, module.dtype, tune=True
    )

    # The capture pass reuses the winners without tuning.
    with flashinfer_mxfp8_decode_graph_capture():
        assert method.apply(module, activation, bias=None) is graph_output
    assert graph_linear.call_args.kwargs == {"tune": False}
    mm_mxfp8.assert_called_once()

    # Leaving the decode-capture scope restores the eager/native path.
    assert method.apply(module, activation, bias=None) is native_output
    assert native_gemm.call_count == 2

    # Falling back to the native backend disarms graph tuning as well.
    method.disable_flashinfer_auto()
    assert method.backend == "trtllm"
    assert not method.tune_graph_backends
    with flashinfer_mxfp8_decode_graph_capture():
        assert method.apply(module, activation, bias=None) is native_output
    assert graph_linear.call_count == 2


def test_mxfp8_quantize_runner_dispatches_backends(monkeypatch):
    expected = (object(), object())
    cute_dsl_quantize = Mock(return_value=expected)
    monkeypatch.setattr(
        torch_custom_ops, "_flashinfer_mxfp8_quantize", cute_dsl_quantize, raising=False
    )
    native_quantize = Mock(return_value=(object(), object()))
    monkeypatch.setattr(
        torch_custom_ops,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(trtllm=SimpleNamespace(mxfp8_quantize=native_quantize))
        ),
    )
    activation = torch.randn((2, 4), dtype=torch.bfloat16)

    runner = MXFP8QuantizeRunner(activation.dtype)
    assert runner.unique_id() == (torch.bfloat16,)
    assert runner.get_valid_tactics([activation], None) == [runner.TRTLLM, runner.CUTE_DSL]

    assert runner([activation], tactic=runner.TRTLLM) is native_quantize.return_value
    native_quantize.assert_called_once_with(activation, True)
    cute_dsl_quantize.assert_not_called()

    assert runner([activation], tactic=runner.CUTE_DSL) is expected
    cute_dsl_quantize.assert_called_once_with(
        activation,
        is_sf_swizzled_layout=True,
        alignment=32,
        enable_pdl=None,
        backend="cute-dsl",
    )


def test_flashinfer_mxfp8_gemm_runner_dispatches_backends(monkeypatch):
    """Both tactics reuse the eager call contract; only CuTeDSL skips FlashInfer's own tuning."""
    output = object()
    flashinfer_gemm = Mock(return_value=output)
    flashinfer_autotune = Mock(return_value=contextlib.nullcontext())
    monkeypatch.setattr(torch_custom_ops, "_flashinfer_mm_mxfp8", flashinfer_gemm, raising=False)
    monkeypatch.setattr(
        torch_custom_ops, "_flashinfer_autotune", flashinfer_autotune, raising=False
    )
    act = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
    act_scale = torch.empty(512, dtype=torch.uint8)
    weight = torch.empty((3, 4), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(512, dtype=torch.uint8)
    inputs = [act, act_scale, weight, weight_scale]

    runner = FlashInferMXFP8GemmRunner(torch.bfloat16)
    assert runner.unique_id() == (torch.bfloat16,)
    assert runner.get_valid_tactics(inputs, None) == [runner.CUTLASS, runner.CUTE_DSL]

    assert runner(inputs, tactic=runner.CUTLASS) is output
    args = flashinfer_gemm.call_args.args
    kwargs = flashinfer_gemm.call_args.kwargs
    assert args[0] is act
    assert args[1].shape == (4, 3)
    assert args[1].data_ptr() == weight.data_ptr()
    assert args[2] is act_scale
    assert args[3] is weight_scale
    assert kwargs == {
        "out_dtype": torch.bfloat16,
        "use_8x4_sf_layout": False,
        "backend": "cutlass",
    }
    flashinfer_autotune.assert_not_called()

    assert runner(inputs, tactic=runner.CUTE_DSL) is output
    assert flashinfer_gemm.call_args.kwargs["backend"] == "cute-dsl"
    flashinfer_autotune.assert_called_once_with(tune_mode=False, skip_ops="mxfp8_gemm")


def test_mxfp8_graph_tuned_linear_tunes_both_stages_only_when_asked(monkeypatch):
    """Each stage selects its backend through the AutoTuner; ``tune`` gates tuning mode."""
    tuning_modes = []

    @contextlib.contextmanager
    def trtllm_autotune(*, tune_mode, skip_dynamic_tuning_buckets):
        assert skip_dynamic_tuning_buckets
        tuning_modes.append(tune_mode)
        yield

    quantized, scale, output = object(), object(), object()
    monkeypatch.setattr(torch_custom_ops, "autotune", trtllm_autotune)
    monkeypatch.setattr(
        torch_custom_ops,
        "_flashinfer_mxfp8_quantize",
        Mock(return_value=(quantized, scale)),
        raising=False,
    )
    flashinfer_gemm = Mock(return_value=output)
    monkeypatch.setattr(torch_custom_ops, "_flashinfer_mm_mxfp8", flashinfer_gemm, raising=False)
    monkeypatch.setattr(
        torch_custom_ops,
        "_flashinfer_autotune",
        Mock(return_value=contextlib.nullcontext()),
        raising=False,
    )
    choose_one = Mock(
        side_effect=lambda op, runners, config, inputs: (runners[0], runners[0].CUTE_DSL)
    )
    monkeypatch.setattr(
        torch_custom_ops.AutoTuner, "get", lambda: SimpleNamespace(choose_one=choose_one)
    )
    activation = torch.randn((2, 4), dtype=torch.bfloat16)
    weight = torch.empty((3, 4), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(512, dtype=torch.uint8)

    tuned = torch_custom_ops.mxfp8_graph_tuned_linear
    assert tuned(activation, weight, weight_scale, torch.bfloat16, tune=True) is output
    assert tuned(activation, weight, weight_scale, torch.bfloat16) is output

    assert tuning_modes == [True, True, False, False]
    quantize_call, gemm_call = choose_one.call_args_list[:2]
    assert [call.args[0] for call in choose_one.call_args_list] == [
        "trtllm::mxfp8_quantize_autotuned::quantize",
        "trtllm::flashinfer_mxfp8_gemm_autotuned::gemm",
    ] * 2
    assert isinstance(quantize_call.args[1][0], MXFP8QuantizeRunner)
    assert quantize_call.args[2] is MXFP8QuantizeRunner.tuning_config
    assert quantize_call.args[3] == [activation]
    assert isinstance(gemm_call.args[1][0], FlashInferMXFP8GemmRunner)
    assert gemm_call.args[2] is FlashInferMXFP8GemmRunner.tuning_config
    assert gemm_call.args[3] == [quantized, scale, weight, weight_scale]
    assert flashinfer_gemm.call_args.kwargs["backend"] == "cute-dsl"
    # Winners are ordinary AutoTuner entries: repeated buckets hit the cache
    # instead of re-profiling, and nothing is excluded from persistence.
    assert not MXFP8QuantizeRunner.tuning_config.exclude_from_cache
    assert not FlashInferMXFP8GemmRunner.tuning_config.exclude_from_cache


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
def test_mxfp8_graph_backend_tuning_matches_native(monkeypatch, batch_size):
    """Per-bucket tuned decode graphs must match the native op.

    Profile the quantizer and GEMM backends for a decode bucket during the
    warmup-only pass, then capture the same shape and replay it. This covers
    the CuTeDSL scale layouts and the in-process winner cache used by capture.
    """
    if not is_flashinfer_mxfp8_cute_dsl_available():
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
    method.tune_graph_backends = True

    # Warmup-only pass: profile both backends of each stage for this bucket.
    with flashinfer_mxfp8_autotune(), flashinfer_mxfp8_decode_graph_capture():
        warmup_output = tuned(x)
    torch.testing.assert_close(warmup_output, native_output, rtol=2e-2, atol=2e-2)

    # Capture pass: the in-process winners are reused without profiling.
    static_x = x.clone()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        with flashinfer_mxfp8_decode_graph_capture():
            graph_output = tuned(static_x)
    graph.replay()
    torch.testing.assert_close(graph_output, native_output, rtol=2e-2, atol=2e-2)
