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
from unittest.mock import patch

import pytest
import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MarlinNVFP4LinearMethod,
    W4A16NVFP4LinearMethod,
    get_quant_method,
    get_sm_version,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _run_w4a16_marlin_reference_case(m: int, n: int, k: int) -> None:
    act, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(m, n, k, torch.bfloat16)
    expected = torch.ops.trtllm.w4a16_nvfp4_gemm(
        act,
        weight,
        weight_scale,
        weight_scale_2,
        torch.bfloat16,
        bias=None,
    )

    linear = Linear(
        k,
        n,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4),
        reduce_output=False,
    ).cuda()
    assert isinstance(linear.quant_method, MarlinNVFP4LinearMethod)
    linear.weight.data.copy_(weight)
    linear.weight_scale.data.copy_(weight_scale)
    linear.weight_scale_2.data.copy_(weight_scale_2)
    linear.transform_weights()
    actual = linear(act)
    torch.testing.assert_close(actual, expected, atol=0.75, rtol=0.02)


def _run_w4a16_triton_reference_case(m: int, n: int, k: int) -> None:
    act, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(m, n, k, torch.float16)
    expected = torch.ops.trtllm.w4a16_nvfp4_gemm(
        act,
        weight,
        weight_scale,
        weight_scale_2,
        torch.float16,
        bias=None,
    )

    linear = Linear(
        k,
        n,
        bias=False,
        dtype=torch.float16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4),
        reduce_output=False,
    ).cuda()
    assert type(linear.quant_method) is W4A16NVFP4LinearMethod
    linear.weight.data.copy_(weight)
    linear.weight_scale.data.copy_(weight_scale)
    linear.weight_scale_2.data.copy_(weight_scale_2)
    linear.transform_weights()
    actual = linear(act)
    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)


def _make_w4a16_nvfp4_case(m: int, n: int, k: int, dtype: torch.dtype):
    torch.manual_seed(m + n + k)
    act = torch.randn((m, k), device="cuda", dtype=dtype)
    weight = torch.empty((n, k // 2), device="cuda", dtype=fp4_utils.float4_e2m1x2)
    weight_u8 = torch.randint(
        0,
        256,
        (n, k // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    weight.copy_(weight_u8.view(fp4_utils.float4_e2m1x2))

    scale_cols = fp4_utils.pad_up(k // 16, 4)
    scale_rows = fp4_utils.pad_up(n, 128)
    # E4M3 bit patterns in [0x30, 0x40] represent scales from 0.5 to 2.0.
    weight_scale_linear = torch.randint(
        0x30,
        0x41,
        (scale_rows, scale_cols),
        device="cuda",
        dtype=torch.uint8,
    )
    weight_scale = torch.ops.trtllm.block_scale_interleave(weight_scale_linear).view(
        fp4_utils.float4_sf_dtype
    )
    weight_scale_2 = torch.ones((1,), device="cuda", dtype=torch.float32)
    return act, weight, weight_scale, weight_scale_2


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 256, 256),
        (4, 160, 288),
        (4, 256, 256),
        (32, 256, 256),
        (128, 512, 1024),
    ],
)
def test_w4a16_nvfp4_marlin_bf16_matches_cuda_core(shape):
    m, n, k = shape
    _run_w4a16_marlin_reference_case(m, n, k)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
def test_w4a16_nvfp4_triton_fallback_matches_cuda_core():
    _run_w4a16_triton_reference_case(32, 64, 64)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="requires CUDA SM120/121",
)
def test_w4a16_nvfp4_gemm_large_m_chunks_cuda_core():
    m, n, k = 32, 64, 64
    act, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(m, n, k, torch.bfloat16)
    expected = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for start in range(0, m, 16):
        stop = min(start + 16, m)
        expected[start:stop, :] = torch.ops.trtllm.w4a16_nvfp4_gemm(
            act[start:stop, :],
            weight,
            weight_scale,
            weight_scale_2,
            torch.bfloat16,
            bias=None,
        )

    actual = torch.ops.trtllm.w4a16_nvfp4_gemm(
        act,
        weight,
        weight_scale,
        weight_scale_2,
        torch.bfloat16,
        bias=None,
    )

    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)


def test_get_quant_method_returns_w4a16_nvfp4_linear_method():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)

    method = get_quant_method(quant_config)

    assert type(method) is W4A16NVFP4LinearMethod


def test_w4a16_nvfp4_linear_uses_high_precision_activation_without_fp4_quantize():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 32), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((4, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.25], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=4,
        pre_quant_scale=None,
    )
    captured = {}

    def fake_w4a16_nvfp4_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        captured["input"] = input_arg
        captured["weight"] = weight
        captured["weight_scale"] = weight_scale
        captured["weight_scale_2"] = weight_scale_2
        captured["out_dtype"] = out_dtype
        captured["bias"] = bias
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    def fail_fp4_quantize(*args, **kwargs):
        raise AssertionError("W4A16 NVFP4 must not quantize activations")

    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120),
        patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True),
        patch("torch.ops.trtllm.fp4_quantize", side_effect=fail_fp4_quantize, create=True),
    ):
        output = method.apply(module, input_tensor, bias)

    assert captured["input"] is input_tensor
    assert captured["weight"] is module.weight
    assert captured["weight_scale"] is module.weight_scale
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["out_dtype"] is torch.bfloat16
    assert captured["bias"] is None
    expected = torch.tensor([[2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(output, expected)


def test_w4a16_nvfp4_linear_restores_high_rank_input_shape():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 3, 32), dtype=torch.float16)
    module = SimpleNamespace(
        weight=torch.empty((8, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.float16,
        out_features=8,
        pre_quant_scale=None,
    )

    def fake_w4a16_nvfp4_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        assert input_arg.shape == (6, 32)
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120),
        patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True),
    ):
        output = method.apply(module, input_tensor, bias=None)

    assert output.shape == (2, 3, 8)


def test_w4a16_nvfp4_linear_uses_triton_dequant_for_large_m():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((17, 32), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((4, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        _w4a16_weight_scale_linear=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=4,
        scaling_vector_size=16,
        pre_quant_scale=None,
        use_custom_cublas_mm=False,
    )
    captured = {}

    def fake_dequant(weight, weight_scale, weight_scale_2, **kwargs):
        captured["weight"] = weight
        captured["weight_scale"] = weight_scale
        captured["weight_scale_2"] = weight_scale_2
        captured.update(kwargs)
        return torch.ones((4, 32), dtype=torch.bfloat16)

    def fail_w4a16_gemm(*args, **kwargs):
        raise AssertionError("large-M W4A16 must use Triton dequantization")

    with (
        patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fail_w4a16_gemm, create=True),
        patch(
            "tensorrt_llm._torch.modules.fused_moe.triton_dequant_nvfp4.dequant_nvfp4_2d_triton",
            side_effect=fake_dequant,
        ),
    ):
        output = method.apply(module, input_tensor, bias=bias)

    assert captured["weight"].data_ptr() == module.weight.data_ptr()
    assert captured["weight_scale"] is module._w4a16_weight_scale_linear
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["target_dtype"] is torch.bfloat16
    assert captured["sf_vec_size"] == 16
    expected = torch.tensor([33.0, 34.0, 35.0, 36.0], dtype=torch.bfloat16).expand(17, 4)
    torch.testing.assert_close(output, expected)


def test_w4a16_nvfp4_linear_uses_marlin_op_after_weight_transform():
    method = MarlinNVFP4LinearMethod()
    input_tensor = torch.ones((1, 32), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((8, 16), dtype=torch.int32),
        weight_scale=torch.empty((2, 128), dtype=torch.float8_e4m3fn),
        weight_global_scale=torch.tensor([0.5], dtype=torch.bfloat16),
        dtype=torch.bfloat16,
        in_features=32,
        out_features=3,
        pre_quant_scale=None,
        _marlin_size_k=32,
        _marlin_size_n=32,
    )
    captured = {}

    def fake_marlin_nvfp4_gemm(input_arg, weight, **kwargs):
        captured["input"] = input_arg
        captured["weight"] = weight
        captured.update(kwargs)
        return torch.ones((input_arg.shape[0], kwargs["size_n"]), dtype=kwargs["out_dtype"])

    def fail_w4a16_gemm(*args, **kwargs):
        raise AssertionError("Marlin W4A16 must not call the default W4A16 op")

    with patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fail_w4a16_gemm, create=True):
        with patch(
            "torch.ops.trtllm.marlin_nvfp4_gemm",
            side_effect=fake_marlin_nvfp4_gemm,
            create=True,
        ):
            output = method.apply(module, input_tensor, bias=None)

    assert captured["input"] is input_tensor
    assert captured["weight"] is module.weight
    assert captured["out_dtype"] is torch.bfloat16
    assert captured["bias"] is None
    assert captured["scale_b"] is module.weight_scale
    assert captured["weight_global_scale"] is module.weight_global_scale
    assert output.shape == (1, 3)


def test_w4a16_nvfp4_linear_marlin_restores_high_rank_input_shape():
    method = MarlinNVFP4LinearMethod()
    input_tensor = torch.ones((2, 9, 32), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((8, 16), dtype=torch.int32),
        weight_scale=torch.empty((2, 128), dtype=torch.float8_e4m3fn),
        weight_global_scale=torch.tensor([0.5], dtype=torch.bfloat16),
        dtype=torch.bfloat16,
        in_features=32,
        out_features=3,
        pre_quant_scale=None,
        _marlin_size_k=32,
        _marlin_size_n=32,
    )
    captured = {}

    def fake_marlin_nvfp4_gemm(input_arg, weight, **kwargs):
        captured["input_shape"] = input_arg.shape
        return torch.ones((input_arg.shape[0], kwargs["size_n"]), dtype=kwargs["out_dtype"])

    with patch(
        "torch.ops.trtllm.marlin_nvfp4_gemm",
        side_effect=fake_marlin_nvfp4_gemm,
        create=True,
    ):
        output = method.apply(module, input_tensor, bias=bias)

    assert captured["input_shape"] == (18, 32)
    assert output.shape == (2, 9, 3)
    expected = torch.tensor([2.0, 3.0, 4.0], dtype=torch.bfloat16).expand(2, 9, 3)
    torch.testing.assert_close(output, expected)


def test_w4a16_nvfp4_linear_marlin_applies_pre_quant_scale_once():
    method = MarlinNVFP4LinearMethod()
    input_tensor = torch.ones((1, 32), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((8, 16), dtype=torch.int32),
        weight_scale=torch.empty((2, 128), dtype=torch.float8_e4m3fn),
        weight_global_scale=torch.tensor([0.5], dtype=torch.bfloat16),
        dtype=torch.bfloat16,
        in_features=32,
        out_features=3,
        pre_quant_scale=torch.full((32,), 2.0, dtype=torch.bfloat16),
        _marlin_size_k=32,
        _marlin_size_n=32,
    )
    captured = {}

    def fake_marlin_nvfp4_gemm(input_arg, weight, **kwargs):
        captured["input"] = input_arg
        return torch.ones((input_arg.shape[0], kwargs["size_n"]), dtype=kwargs["out_dtype"])

    with patch(
        "torch.ops.trtllm.marlin_nvfp4_gemm",
        side_effect=fake_marlin_nvfp4_gemm,
        create=True,
    ):
        method.apply(module, input_tensor, bias=None)

    torch.testing.assert_close(captured["input"], input_tensor * module.pre_quant_scale)


@pytest.mark.parametrize(
    ("dtype", "use_fused_gemm_allreduce"),
    [
        (torch.float16, False),
        (torch.bfloat16, True),
    ],
)
def test_w4a16_nvfp4_marlin_selection_requires_supported_module(dtype, use_fused_gemm_allreduce):
    module = SimpleNamespace(dtype=dtype, use_fused_gemm_allreduce=use_fused_gemm_allreduce)

    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
        patch("torch.ops.trtllm.gptq_marlin_repack", create=True),
    ):
        assert not MarlinNVFP4LinearMethod.is_supported(module)


def test_w4a16_nvfp4_linear_selects_marlin_for_supported_module():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)

    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
        patch("torch.ops.trtllm.gptq_marlin_repack", create=True),
    ):
        linear = Linear(
            32,
            32,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=quant_config,
            reduce_output=False,
        )

    assert isinstance(linear.quant_method, MarlinNVFP4LinearMethod)
    assert isinstance(linear.quant_method, W4A16NVFP4LinearMethod)


def test_w4a16_nvfp4_linear_keeps_default_method_for_fp16():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)

    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
        patch("torch.ops.trtllm.gptq_marlin_repack", create=True),
    ):
        linear = Linear(
            32,
            32,
            bias=False,
            dtype=torch.float16,
            quant_config=quant_config,
            reduce_output=False,
        )

    assert type(linear.quant_method) is W4A16NVFP4LinearMethod


def test_w4a16_nvfp4_post_load_preserves_checkpoint_weight_global_scale():
    method = W4A16NVFP4LinearMethod()
    module = SimpleNamespace(
        input_scale=None,
        inv_input_scale=None,
        alpha=None,
        weight_scale_2=torch.empty([1], dtype=torch.float32),
        tmp_nvfp4_input_scales_list=[torch.tensor(1.0, dtype=torch.float32)],
        tmp_nvfp4_weight_scale_2_list=[torch.tensor(0.25, dtype=torch.float32)],
    )

    method.process_weights_after_loading_vanilla(module)

    assert module.input_scale is None
    assert module.inv_input_scale is None
    assert module.alpha is None
    torch.testing.assert_close(module.weight_scale_2, torch.tensor([0.25], dtype=torch.float32))
    assert not hasattr(module, "tmp_nvfp4_input_scales_list")
    assert not hasattr(module, "tmp_nvfp4_weight_scale_2_list")


def test_lm_head_uses_w4a16_nvfp4_quant_method_for_packed_lm_head():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)

    lm_head = LMHead(
        num_embeddings=3, embedding_dim=16, dtype=torch.float16, quant_config=quant_config
    )

    assert isinstance(lm_head.quant_method, W4A16NVFP4LinearMethod)
    assert lm_head.weight.dtype == torch.uint8
    assert lm_head.weight.shape == (3, 8)
    assert lm_head.weight_scale.shape == (128 * 4,)
    assert lm_head.weight_scale_2.shape == (1,)


def test_lm_head_w4a16_nvfp4_forward_dispatches_to_w4a16_op():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    lm_head = LMHead(
        num_embeddings=4,
        embedding_dim=32,
        dtype=torch.float16,
        quant_config=quant_config,
    )
    input_tensor = torch.ones((2, 32), dtype=torch.float16)
    captured = {}

    def fake_w4a16_nvfp4_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        captured["input"] = input_arg
        captured["weight"] = weight
        captured["weight_scale"] = weight_scale
        captured["weight_scale_2"] = weight_scale_2
        captured["out_dtype"] = out_dtype
        captured["bias"] = bias
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120),
        patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True),
    ):
        output = lm_head(input_tensor)

    assert captured["input"] is input_tensor
    assert captured["weight"] is lm_head.weight
    assert captured["weight_scale"] is lm_head.weight_scale
    assert captured["weight_scale_2"] is lm_head.weight_scale_2
    assert captured["out_dtype"] is torch.float16
    assert captured["bias"] is None
    assert output.shape == (2, 4)
