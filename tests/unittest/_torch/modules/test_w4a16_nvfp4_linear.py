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

import torch

from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import W4A16NVFP4LinearMethod, get_quant_method
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def test_get_quant_method_returns_w4a16_nvfp4_linear_method():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)

    method = get_quant_method(quant_config)

    assert isinstance(method, W4A16NVFP4LinearMethod)


def test_w4a16_nvfp4_linear_uses_high_precision_activation_without_fp4_quantize():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 4), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((5, 2), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.25], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=3,
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

    with patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True):
        with patch("torch.ops.trtllm.fp4_quantize", side_effect=fail_fp4_quantize, create=True):
            output = method.apply(module, input_tensor, bias)

    assert captured["input"] is input_tensor
    assert captured["weight"] is module.weight
    assert captured["weight_scale"] is module.weight_scale
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["out_dtype"] is torch.bfloat16
    assert captured["bias"] is None
    expected = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(output, expected)


def test_w4a16_nvfp4_linear_restores_high_rank_input_shape():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 3, 4), dtype=torch.float16)
    module = SimpleNamespace(
        weight=torch.empty((7, 2), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.float16,
        out_features=5,
        pre_quant_scale=None,
    )

    def fake_w4a16_nvfp4_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        assert input_arg.shape == (6, 4)
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    with patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True):
        output = method.apply(module, input_tensor, bias=None)

    assert output.shape == (2, 3, 5)


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


def test_lm_head_w4a16_nvfp4_forward_dispatches_to_dense_op():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    lm_head = LMHead(
        num_embeddings=3, embedding_dim=16, dtype=torch.float16, quant_config=quant_config
    )
    input_tensor = torch.ones((2, 16), dtype=torch.float16)
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

    with patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True):
        output = lm_head(input_tensor)

    assert captured["input"] is input_tensor
    assert captured["weight"] is lm_head.weight
    assert captured["weight_scale"] is lm_head.weight_scale
    assert captured["weight_scale_2"] is lm_head.weight_scale_2
    assert captured["out_dtype"] is torch.float16
    assert captured["bias"] is None
    assert output.shape == (2, 3)
