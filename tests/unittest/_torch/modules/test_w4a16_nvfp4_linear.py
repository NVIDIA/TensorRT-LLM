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

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import (
    W4A16NVFP4LinearMethod,
    get_quant_method,
    get_sm_version,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _run_w4a16_cutlass3_reference_case(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    assert k % 32 == 0
    assert n % 32 == 0
    act, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(m, n, k, dtype)

    expected = torch.empty((m, n), device="cuda", dtype=dtype)
    for start in range(0, m, 16):
        stop = min(start + 16, m)
        expected[start:stop, :] = torch.ops.trtllm.w4a16_nvfp4_gemm(
            act[start:stop, :],
            weight,
            weight_scale,
            weight_scale_2,
            dtype,
            bias=None,
        )

    actual = torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm(
        act,
        weight,
        weight_scale,
        weight_scale_2,
        dtype,
        bias=None,
    )
    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)


def _make_w4a16_nvfp4_case(m: int, n: int, k: int, dtype: torch.dtype):
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
    weight_scale_linear = torch.randint(
        1,
        120,
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
    [(32, 256, 256), (128, 512, 1024), (256, 1024, 2048)],
)
def test_w4a16_nvfp4_cutlass3_bf16_matches_cuda_core(shape):
    m, n, k = shape
    _run_w4a16_cutlass3_reference_case(m, n, k, torch.bfloat16)


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


def test_w4a16_nvfp4_linear_uses_chunked_w4a16_op_for_large_m():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((17, 16), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((5, 8), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
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
        raise AssertionError("large-M W4A16 path must not quantize activations")

    with patch("torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True):
        with patch("torch.ops.trtllm.fp4_quantize", side_effect=fail_fp4_quantize, create=True):
            output = method.apply(module, input_tensor, bias=None)

    assert captured["input"] is input_tensor
    assert captured["weight"] is module.weight
    assert captured["weight_scale"] is module.weight_scale
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["out_dtype"] is torch.bfloat16
    assert output.shape == (17, 3)


def test_w4a16_nvfp4_linear_uses_cutlass3_op_for_large_bf16_m_when_enabled():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((17, 32), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((32, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=3,
        pre_quant_scale=None,
    )
    captured = {}

    def fake_w4a16_nvfp4_cutlass_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        captured["input"] = input_arg
        captured["weight"] = weight
        captured["weight_scale"] = weight_scale
        captured["weight_scale_2"] = weight_scale_2
        captured["out_dtype"] = out_dtype
        captured["bias"] = bias
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    def fail_w4a16_gemm(*args, **kwargs):
        raise AssertionError("CUTLASS3 W4A16 prefill must not call the default W4A16 op")

    def fail_fp4_quantize(*args, **kwargs):
        raise AssertionError("CUTLASS3 W4A16 prefill must not quantize activations")

    with patch.dict(os.environ, {"TRTLLM_W4A16_NVFP4_CUTLASS3": "1"}):
        with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120):
            with patch(
                "torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fail_w4a16_gemm, create=True
            ):
                with patch(
                    "torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm",
                    side_effect=fake_w4a16_nvfp4_cutlass_gemm,
                    create=True,
                ):
                    with patch(
                        "torch.ops.trtllm.fp4_quantize", side_effect=fail_fp4_quantize, create=True
                    ):
                        output = method.apply(module, input_tensor, bias=None)

    assert captured["input"] is input_tensor
    assert captured["weight"] is module.weight
    assert captured["weight_scale"] is module.weight_scale
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["out_dtype"] is torch.bfloat16
    assert captured["bias"] is None
    assert output.shape == (17, 3)


def test_w4a16_nvfp4_linear_cutlass3_restores_high_rank_input_shape():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 9, 32), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((32, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=3,
        pre_quant_scale=None,
    )
    captured = {}

    def fake_w4a16_nvfp4_cutlass_gemm(
        input_arg, weight, weight_scale, weight_scale_2, out_dtype, bias=None
    ):
        captured["input_shape"] = input_arg.shape
        return torch.ones((input_arg.shape[0], weight.shape[0]), dtype=out_dtype)

    def fail_w4a16_gemm(*args, **kwargs):
        raise AssertionError("large-M CUTLASS3 path must not call the default W4A16 op")

    with patch.dict(os.environ, {"TRTLLM_W4A16_NVFP4_CUTLASS3": "1"}):
        with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120):
            with patch(
                "torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fail_w4a16_gemm, create=True
            ):
                with patch(
                    "torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm",
                    side_effect=fake_w4a16_nvfp4_cutlass_gemm,
                    create=True,
                ):
                    output = method.apply(module, input_tensor, bias=bias)

    assert captured["input_shape"] == (18, 32)
    assert output.shape == (2, 9, 3)
    expected = torch.tensor([2.0, 3.0, 4.0], dtype=torch.bfloat16).expand(2, 9, 3)
    torch.testing.assert_close(output, expected)


def test_w4a16_nvfp4_cutlass3_prefill_requires_supported_shape():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((17, 32), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((6, 16), dtype=torch.uint8),
        dtype=torch.bfloat16,
    )

    with patch.dict(os.environ, {"TRTLLM_W4A16_NVFP4_CUTLASS3": "1"}):
        with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120):
            assert not method._can_use_cutlass3_w4a16_prefill(module, input_tensor, m=17)


def test_w4a16_nvfp4_linear_cutlass3_unsupported_shape_uses_default_w4a16_op():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((17, 32), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((6, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
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

    def fail_cutlass_gemm(*args, **kwargs):
        raise AssertionError("unsupported CUTLASS3 shape must use the default W4A16 op")

    def fail_fp4_quantize(*args, **kwargs):
        raise AssertionError("unsupported CUTLASS3 W4A16 path must not quantize activations")

    with patch.dict(os.environ, {"TRTLLM_W4A16_NVFP4_CUTLASS3": "1"}):
        with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120):
            with patch(
                "torch.ops.trtllm.w4a16_nvfp4_gemm", side_effect=fake_w4a16_nvfp4_gemm, create=True
            ):
                with patch(
                    "torch.ops.trtllm.w4a16_nvfp4_cutlass_gemm",
                    side_effect=fail_cutlass_gemm,
                    create=True,
                ):
                    with patch(
                        "torch.ops.trtllm.fp4_quantize", side_effect=fail_fp4_quantize, create=True
                    ):
                        output = method.apply(module, input_tensor, bias=None)

    assert captured["input"] is input_tensor
    assert captured["bias"] is None
    assert captured["weight"] is module.weight
    assert captured["weight_scale"] is module.weight_scale
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["out_dtype"] is torch.bfloat16
    assert output.shape == (17, 3)


def test_w4a16_nvfp4_post_load_ignores_checkpoint_activation_scale():
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
    torch.testing.assert_close(module.weight_scale_2, torch.tensor([4.0], dtype=torch.float32))
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
