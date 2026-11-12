# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output slices preserve loaded weights and FP8 scale-block boundaries."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.modules.linear as linear_module
from tensorrt_llm._torch.modules.linear import (
    Fp8BlockScaleLowMGemmRange,
    FP8BlockScalesLinearMethod,
    FP8QDQLinearMethod,
    FP8RowwiseLinearMethod,
    Linear,
    NVFP4LinearMethod,
    TensorParallelMode,
    UnquantizedLinearMethod,
    W8A16FP8LinearMethod,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
from tensorrt_llm.quantization.utils.fp8_utils import Fp8BlockScaleWeightSliceCache


def _make_linear(backend, bias):
    torch.manual_seed(42)
    in_features, out_features = 512, 1152
    quantized = backend != "bf16"
    with torch.device("cuda"):
        linear = Linear(
            in_features,
            out_features,
            bias=bias,
            dtype=torch.bfloat16,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
            if quantized
            else QuantConfig(),
            use_cute_dsl_blockscaling_mm=backend == "fp8_cute",
        )
    weight = torch.randn(out_features, in_features, device="cuda") * 0.02
    checkpoint = {}
    if quantized:
        blocks = weight.view(out_features // 128, 128, in_features // 128, 128)
        scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
        expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        checkpoint["weight"] = (weight / expanded).to(torch.float8_e4m3fn)
        checkpoint["weight_scale"] = scale
    else:
        checkpoint["weight"] = weight.bfloat16()
    if bias:
        checkpoint["bias"] = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")
    linear.load_weights([checkpoint])
    linear.post_load_weights()
    return linear, checkpoint


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100, reason="requires Blackwell or later"
)
@pytest.mark.parametrize("backend", ["bf16", "fp8_cute", "fp8_deepgemm"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("start,end", [(0, 576), (576, 1152), (513, 1107)])
@torch.inference_mode()
def test_linear_output_slice(backend, bias, start, end):
    if get_sm_version() == 107 and backend == "fp8_deepgemm":
        pytest.skip("SM107 native FP8 slices are covered by fp8_cute")
    linear, _ = _make_linear(backend, bias)
    quantized = backend != "bf16"
    original_weight = linear.weight.clone()
    original_scale = linear.weight_scale.clone() if quantized else None
    inputs = torch.randn(31, linear.in_features, dtype=torch.bfloat16, device="cuda")
    expected = linear(inputs)[:, start:end]
    linear.prepare_output_slice(start, end)
    cached_scale = (
        linear._output_slice_cache.get(linear.weight, start, end)[1] if quantized else None
    )
    actual = linear.forward_slice(inputs, start, end)
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.03)
    torch.testing.assert_close(
        linear.weight.view(torch.uint8), original_weight.view(torch.uint8), atol=0, rtol=0
    )
    if quantized:
        torch.testing.assert_close(linear.weight_scale, original_scale, atol=0, rtol=0)
        assert linear._output_slice_cache.get(linear.weight, start, end)[1] is cached_scale


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100, reason="requires Blackwell or later"
)
@pytest.mark.parametrize("start,end", [(0, 576), (576, 1152), (513, 1107)])
@torch.inference_mode()
def test_kimi_fp8_output_slice(start, end):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    backend = "fp8_cute" if get_sm_version() == 107 else "fp8_deepgemm"
    linear, _ = _make_linear(backend, False)
    projection = _Fp8BlockScaleWeightReadLinear(
        linear.weight.detach(), linear.weight_scale.detach(), linear.out_features
    )
    inputs = torch.randn(2, 17, linear.in_features, dtype=torch.bfloat16, device="cuda")
    expected = projection(inputs)[..., start:end]
    projection.prepare_output_slice(start, end)
    cached_scale = projection._output_slice_cache.get(projection.weight, start, end)[1]
    actual = projection.forward_slice(inputs, start, end)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.03)
    assert actual.is_contiguous()
    assert projection._output_slice_cache.get(projection.weight, start, end)[1] is cached_scale


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100, reason="requires Blackwell or later"
)
@pytest.mark.parametrize("backend", ["fp8_cute", "fp8_deepgemm"])
@torch.inference_mode()
def test_linear_output_slice_reload(backend):
    if get_sm_version() == 107 and backend == "fp8_deepgemm":
        pytest.skip("SM107 native FP8 slices are covered by fp8_cute")
    linear, checkpoint = _make_linear(backend, False)
    inputs = torch.randn(7, linear.in_features, dtype=torch.bfloat16, device="cuda")
    linear.prepare_output_slice(513, 1107)
    old_cache = linear._output_slice_cache
    linear.pre_reload_weights()
    assert linear._output_slice_cache is None
    reloaded = {name: value.clone() for name, value in checkpoint.items()}
    reloaded["weight_scale"].mul_(2)
    linear.load_weights([reloaded])
    linear.post_load_weights()
    actual = linear.forward_slice(inputs, 513, 1107)
    torch.testing.assert_close(actual, linear(inputs)[:, 513:1107], atol=0.005, rtol=0.03)
    assert linear._output_slice_cache is not old_cache


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "quant_method",
    [NVFP4LinearMethod, FP8QDQLinearMethod, FP8RowwiseLinearMethod, W8A16FP8LinearMethod],
)
def test_prepare_output_slice_rejects_unsupported_projection_before_forward(quant_method):
    linear = Linear(
        8,
        16,
        bias=False,
        dtype=torch.bfloat16,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        quant_config=QuantConfig(),
        reduce_output=False,
    )

    linear.lora = object()
    with pytest.raises(NotImplementedError, match="LoRA"):
        linear.prepare_output_slice(0, 8)

    linear.lora = None
    linear.quant_method = quant_method()
    with pytest.raises(NotImplementedError, match="output slicing"):
        linear.prepare_output_slice(0, 8)


@pytest.mark.cpu_only
def test_prepare_output_slice_dispatches_to_linear_method():
    prepared_ranges = []

    class PreparedLinearMethod(UnquantizedLinearMethod):
        def prepare_output_slice(self, module: Linear, start: int, end: int) -> None:
            prepared_ranges.append((start, end))

    linear = Linear(
        8,
        16,
        bias=False,
        dtype=torch.float32,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        quant_config=QuantConfig(),
        reduce_output=False,
    )
    linear.quant_method = PreparedLinearMethod()
    with torch.no_grad():
        linear.weight.copy_(torch.arange(128, dtype=torch.float32).view(16, 8))
    inputs = torch.arange(16, dtype=torch.float32).view(2, 8)

    linear.prepare_output_slice(4, 12)
    actual = linear.forward_slice(inputs, 4, 12)
    torch.testing.assert_close(actual, torch.nn.functional.linear(inputs, linear.weight[4:12]))
    assert prepared_ranges == [(4, 12), (4, 12)]


@pytest.mark.cpu_only
def test_fp8_block_scale_low_m_dispatch_uses_full_weight_only(monkeypatch):
    """A prepared row slice must not reuse the full projection's low-M range."""
    monkeypatch.setattr(linear_module, "is_sm_100f", lambda: True)
    monkeypatch.setattr(linear_module, "get_sm_version", lambda: 103)

    weight = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
    scale = torch.empty((256, 1), dtype=torch.int32)
    cache = Fp8BlockScaleWeightSliceCache()
    cache.prepare(weight, scale, 129, 190)
    module = SimpleNamespace(
        weight=weight,
        weight_scale=scale,
        _output_slice_cache=cache,
        fp8_low_m_gemm=Fp8BlockScaleLowMGemmRange(max_m=32),
        use_cute_dsl_blockscaling_mm=False,
        disable_deep_gemm=False,
        bias=None,
    )
    calls = []

    def low_m_gemm(inputs, selected_weight, selected_scale):
        calls.append(("low_m", selected_weight is module.weight, selected_weight.shape[0]))
        assert selected_scale.shape[0] == selected_weight.shape[0]
        return torch.ones((inputs.shape[0], selected_weight.shape[0]), dtype=torch.bfloat16)

    def regular_gemm(inputs, selected_weight, selected_scale, **kwargs):
        calls.append(("regular", selected_weight is module.weight, selected_weight.shape[0]))
        assert selected_scale.shape[0] == selected_weight.shape[0]
        return torch.zeros((inputs.shape[0], selected_weight.shape[0]), dtype=torch.bfloat16)

    monkeypatch.setattr(torch.ops.trtllm, "fp8_block_scale_low_m_gemm", low_m_gemm, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "fp8_swap_ab_gemm", regular_gemm, raising=False)
    method = FP8BlockScalesLinearMethod()
    inputs = torch.empty((2, 128), dtype=torch.bfloat16)

    assert torch.all(method.apply(module, inputs, None) == 1)
    sliced = method.apply_slice(module, inputs, 129, 190)
    assert sliced.shape == (2, 61)
    assert torch.all(sliced == 0)
    assert calls == [("low_m", True, 256), ("regular", False, 128)]


@pytest.mark.cpu_only
@pytest.mark.parametrize("rubin", [False, True])
def test_kimi_fp8_low_m_dispatch_uses_full_weight_only(monkeypatch, rubin):
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi_module

    monkeypatch.setattr(kimi_module, "_rubin_mxfp8_is_available", lambda: rubin)
    weight = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
    scale = (
        torch.empty((256 * 4,), dtype=torch.uint8)
        if rubin
        else torch.empty((256, 1), dtype=torch.int32)
    )
    projection = kimi_module._Fp8BlockScaleWeightReadLinear(weight, scale, 256)
    projection._low_m = Fp8BlockScaleLowMGemmRange(max_m=32)
    projection._low_m_weight_scale = torch.empty((256, 1), dtype=torch.int32)
    calls = []

    if rubin:

        def low_m_gemm(inputs, selected_weight, selected_scale):
            assert selected_weight is projection.weight
            assert selected_scale is projection._low_m_weight_scale
            calls.append(("low_m", selected_weight.shape[0]))
            return torch.ones((inputs.shape[0], 256), dtype=torch.bfloat16)

        def regular_gemm(activation, selected_weight, activation_scale, selected_scale, **kwargs):
            assert selected_weight.data_ptr() == projection.weight[128:].data_ptr()
            assert selected_scale.data_ptr() == projection.weight_scale[128 * 4 :].data_ptr()
            calls.append(("regular", selected_weight.shape[0]))
            return torch.zeros(
                (activation.shape[0], selected_weight.shape[0]), dtype=torch.bfloat16
            )

        monkeypatch.setattr(
            torch.ops.trtllm, "fp8_block_scale_low_m_gemm", low_m_gemm, raising=False
        )
        monkeypatch.setattr(
            torch.ops.trtllm,
            "fp8_quantize_1x128_cutedsl_ue8m0",
            lambda inputs: (inputs, torch.empty(0)),
            raising=False,
        )
        monkeypatch.setattr(
            torch.ops.trtllm, "cute_dsl_mxfp8_gemm_rubin", regular_gemm, raising=False
        )
    else:

        def low_m_gemm(inputs, selected_weight, selected_scale):
            assert selected_weight is projection.weight
            calls.append(("low_m", selected_weight.shape[0]))
            assert selected_scale.shape[0] == selected_weight.shape[0]
            return torch.ones((inputs.shape[0], selected_weight.shape[0]), dtype=torch.bfloat16)

        def regular_gemm(inputs, selected_weight, selected_scale, **kwargs):
            assert selected_weight.data_ptr() == projection.weight[128:].data_ptr()
            calls.append(("regular", selected_weight.shape[0]))
            assert selected_scale.shape[0] == selected_weight.shape[0]
            return torch.zeros((inputs.shape[0], selected_weight.shape[0]), dtype=torch.bfloat16)

        monkeypatch.setattr(
            torch.ops.trtllm, "fp8_block_scale_low_m_gemm", low_m_gemm, raising=False
        )
        monkeypatch.setattr(torch.ops.trtllm, "fp8_swap_ab_gemm", regular_gemm, raising=False)

    inputs = torch.empty((2, 128), dtype=torch.bfloat16)
    assert torch.all(projection(inputs) == 1)
    sliced = projection.forward_slice(inputs, 129, 190)
    assert sliced.shape == (2, 61)
    assert torch.all(sliced == 0)
    assert calls == [("low_m", 256), ("regular", 128)]


@pytest.mark.cpu_only
@pytest.mark.parametrize("layout", ["fp32", "packed", "r128c4"])
@pytest.mark.parametrize("start,end", [(0, 576), (576, 1152), (513, 1107)])
@pytest.mark.parametrize("rows", [1152, 1107])
def test_fp8_weight_slice_cache(layout, start, end, rows):
    cols = 512
    end = min(end, rows)
    weight = torch.empty(rows, cols, dtype=torch.float8_e4m3fn)
    if layout == "fp32":
        scales = torch.arange((rows + 127) // 128 * (cols // 128), dtype=torch.float32).view(
            -1, cols // 128
        )
    elif layout == "packed":
        scales = torch.arange(rows * 2, dtype=torch.int32).view(2, rows).t()
    else:
        scales = torch.randint(
            0,
            256,
            (rows, cols // 32),
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(42),
        )
        scales = torch.ops.trtllm.block_scale_interleave(scales)
    cache = Fp8BlockScaleWeightSliceCache()
    cache.prepare(weight, scales, start, end)
    sliced, prepared_scale, crop = cache.get(weight, start, end)
    first, last = start // 128 * 128, min((end + 127) // 128 * 128, rows)
    assert sliced.data_ptr() == weight[first:last].data_ptr()
    assert crop == start - first
    if layout == "fp32":
        expected = scales[first // 128 : (last + 127) // 128]
    elif layout == "packed":
        expected = scales[first:last]
    else:
        padded_rows = (rows + 127) // 128 * 128
        unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(scales.view(padded_rows, -1))
        expected = torch.ops.trtllm.block_scale_interleave(unswizzled[first:last].contiguous())
    torch.testing.assert_close(prepared_scale, expected, atol=0, rtol=0)
    cache.prepare(weight, scales, start, end)
    assert cache.get(weight, start, end)[1] is prepared_scale
    assert not cache.state_dict()
    cache.to("meta")
    assert cache.get(weight, start, end)[1].is_meta
