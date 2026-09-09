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

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MarlinNVFP4LinearMethod,
    NVFP4LinearMethod,
    W4A16NVFP4LinearMethod,
    quant_config_has_nvfp4_activation_quantization,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def test_nvfp4_activation_quantization_excludes_w4a16():
    assert quant_config_has_nvfp4_activation_quantization(QuantConfig(quant_algo=QuantAlgo.NVFP4))
    assert not quant_config_has_nvfp4_activation_quantization(
        QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    )


def test_w4a16_attention_does_not_quantize_output_to_fp4():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    o_proj = Linear(
        32,
        32,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=quant_config,
        reduce_output=False,
    )
    attention = SimpleNamespace(
        attn=SimpleNamespace(has_nvfp4=False),
        o_proj=o_proj,
        quant_config=quant_config,
        has_quant_scale=True,
        attn_output_gate=False,
        is_marlin_enabled=False,
    )

    assert not Attention._use_quantize_output(attention)


def test_static_nvfp4_attention_can_quantize_output_to_fp4_on_blackwell():
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=100):
        o_proj = Linear(
            32,
            32,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=quant_config,
            reduce_output=False,
        )
    attention = SimpleNamespace(
        attn=SimpleNamespace(has_nvfp4=False),
        o_proj=o_proj,
        quant_config=quant_config,
        has_quant_scale=True,
        attn_output_gate=False,
        is_marlin_enabled=False,
    )

    assert Attention._use_quantize_output(attention)


def test_nvfp4_attention_keeps_high_precision_output_for_hopper_marlin():
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=90),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
        patch("torch.ops.trtllm.gptq_marlin_repack", create=True),
    ):
        o_proj = Linear(
            32,
            32,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=quant_config,
            reduce_output=False,
            nvfp4_allowed_backends=["marlin"],
        )
        attention = SimpleNamespace(
            attn=SimpleNamespace(has_nvfp4=False),
            o_proj=o_proj,
            quant_config=quant_config,
            has_quant_scale=o_proj.has_nvfp4_activation_quantization,
            attn_output_gate=False,
            is_marlin_enabled=o_proj.uses_marlin_nvfp4,
        )

        assert o_proj.uses_marlin_nvfp4
        assert isinstance(o_proj.quant_method, MarlinNVFP4LinearMethod)
        assert not o_proj.has_nvfp4_activation_quantization
        assert not Attention._use_quantize_output(attention)


@pytest.mark.parametrize(
    ("allowed_backends", "expected_backends"),
    [
        (["cutlass", "cublaslt", "cuda_core"], "cutlass,cublaslt,cuda_core"),
        (["marlin"], "marlin"),
    ],
)
def test_nvfp4_linear_forwards_allowed_backends_to_gemm(allowed_backends, expected_backends):
    """The module's backend list reaches the unified GEMM op verbatim. Which
    backends are eligible in the first place is covered by
    ``test_nvfp4_linear_keeps_activation_quant_method``."""
    method = NVFP4LinearMethod()
    input_tensor = torch.ones((2, 32), dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((4, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        out_features=4,
        dtype=torch.bfloat16,
        nvfp4_allowed_backends=allowed_backends,
        all_reduce=None,
        mapping=None,
    )
    act_fp4 = torch.empty((2, 16), dtype=torch.uint8)
    act_sf = torch.empty((128 * 4,), dtype=torch.uint8)
    alpha = torch.ones((1,), dtype=torch.float32)
    captured = {}

    def fake_nvfp4_gemm(*args, **kwargs):
        captured["allowed_backends"] = kwargs["allowed_backends"]
        return torch.ones((2, 4), dtype=torch.bfloat16)

    with (
        patch.object(method, "_input_prepare", return_value=(act_fp4, act_sf, alpha)),
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=90),
        patch("torch.ops.trtllm.nvfp4_gemm", side_effect=fake_nvfp4_gemm, create=True),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
        patch("torch.ops.trtllm.gptq_marlin_repack", create=True),
    ):
        output = method.apply(module, input_tensor, bias=None)

    assert captured["allowed_backends"] == expected_backends
    assert output.shape == (2, 4)


@pytest.mark.parametrize(
    ("sm_version", "allowed_backends", "expect_marlin"),
    [
        # Honoured only on Ada/Hopper, and only when opted in.
        (89, ["marlin"], True),
        (90, ["marlin"], True),
        (89, None, False),
        (90, None, False),
        (120, None, False),
        (121, None, False),
        # Opt-in ignored off Ada/Hopper: Marlin is not the right NVFP4 backend
        # on SM120/121, where the W4A4 kernels are faster.
        (120, ["marlin"], False),
        (121, ["marlin"], False),
    ],
)
def test_nvfp4_linear_marlin_opt_in_switches_to_weight_only_method(
    sm_version, allowed_backends, expect_marlin
):
    """The Marlin kernel is W4A16, so opting a W4A4 checkpoint into it converts
    the layer to the weight-only method, which pads N/K for the kernel. Without
    the opt-in the layer keeps NVFP4LinearMethod and its activation quantize."""
    kwargs = {} if allowed_backends is None else {"nvfp4_allowed_backends": allowed_backends}

    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=sm_version),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
        patch("torch.ops.trtllm.gptq_marlin_repack", create=True),
    ):
        linear = Linear(
            32,
            32,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
            reduce_output=False,
            **kwargs,
        )

        # Asserted inside the patch: uses_marlin_nvfp4 re-reads the SM on every
        # access rather than caching what create_weights decided.
        assert isinstance(linear.quant_method, MarlinNVFP4LinearMethod) is expect_marlin
        assert linear.uses_marlin_nvfp4 is expect_marlin
        # Marlin consumes BF16 activations; the plain NVFP4 path quantizes them.
        assert linear.has_nvfp4_activation_quantization is not expect_marlin
        if not expect_marlin:
            assert type(linear.quant_method) is NVFP4LinearMethod


def test_nvfp4_linear_hopper_marlin_applies_bias_as_post_op():
    method = NVFP4LinearMethod()
    input_tensor = torch.ones((2, 32), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((4, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        out_features=4,
        dtype=torch.bfloat16,
        nvfp4_allowed_backends=["marlin"],
        all_reduce=None,
        mapping=None,
    )
    act_fp4 = torch.empty((2, 16), dtype=torch.uint8)
    act_sf = torch.empty((128 * 4,), dtype=torch.uint8)
    alpha = torch.ones((1,), dtype=torch.float32)
    captured = {}

    def fake_nvfp4_gemm(*args, **kwargs):
        captured.update(kwargs)
        return torch.ones((2, 4), dtype=torch.bfloat16)

    with (
        patch.object(method, "_input_prepare", return_value=(act_fp4, act_sf, alpha)),
        patch(
            "tensorrt_llm._torch.modules.linear.get_sm_version",
            return_value=90,
        ),
        patch(
            "torch.ops.trtllm.nvfp4_gemm",
            side_effect=fake_nvfp4_gemm,
            create=True,
        ),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
        patch("torch.ops.trtllm.gptq_marlin_repack", create=True),
    ):
        output = method.apply(module, input_tensor, bias=bias)

    assert captured["allowed_backends"] == "marlin"
    assert captured["bias"] is None
    torch.testing.assert_close(output, torch.ones((2, 4), dtype=torch.bfloat16) + bias)


def test_w4a16_nvfp4_linear_uses_high_precision_activation_without_fp4_quantize():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 32), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    module = SimpleNamespace(
        weight=torch.empty((4, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        _w4a16_weight_scale_linear=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.25], dtype=torch.float32),
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

    def fail_fp4_quantize(*args, **kwargs):
        raise AssertionError("W4A16 NVFP4 must not quantize activations")

    with (
        patch(
            "tensorrt_llm._torch.moe.fused_moe.triton_dequant_nvfp4.dequant_nvfp4_2d_triton",
            side_effect=fake_dequant,
        ),
        patch("torch.ops.trtllm.fp4_quantize", side_effect=fail_fp4_quantize, create=True),
    ):
        output = method.apply(module, input_tensor, bias)

    assert captured["weight"].data_ptr() == module.weight.data_ptr()
    assert captured["weight_scale"] is module._w4a16_weight_scale_linear
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["target_dtype"] is torch.bfloat16
    assert captured["sf_vec_size"] == 16
    expected = torch.tensor(
        [[33.0, 34.0, 35.0, 36.0], [33.0, 34.0, 35.0, 36.0]],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "bad_input",
    [
        pytest.param((torch.empty(1), torch.empty(1)), id="prequantized_fp4_tuple"),
        pytest.param(torch.ones((2, 32), dtype=torch.float8_e4m3fn), id="fp8_activation"),
    ],
)
def test_w4a16_nvfp4_linear_rejects_quantized_input(bad_input):
    """W4A16 has no activation scale, so an upstream FP4/FP8 fusion must fail
    loudly here rather than silently produce garbage."""
    method = W4A16NVFP4LinearMethod()
    module = SimpleNamespace(
        weight=torch.empty((4, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        _w4a16_weight_scale_linear=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.25], dtype=torch.float32),
        dtype=torch.bfloat16,
        out_features=4,
        scaling_vector_size=16,
        inv_input_scale=None,
        pre_quant_scale=None,
        use_custom_cublas_mm=False,
    )

    with pytest.raises(RuntimeError, match="high-precision input"):
        method.apply(module, bad_input, bias=None)


def test_w4a16_nvfp4_linear_scale_cache_is_nonpersistent_buffer():
    with patch.object(MarlinNVFP4LinearMethod, "is_supported", return_value=False):
        linear = Linear(
            32,
            4,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4),
            reduce_output=False,
        )

    scale_linear = torch.arange(128 * 4, dtype=torch.int32).to(torch.uint8)
    with patch(
        "torch.ops.trtllm.block_scale_interleave_reverse",
        return_value=scale_linear,
        create=True,
    ):
        linear.quant_method.cache_derived_state(linear)

    assert linear._buffers["_w4a16_weight_scale_linear"].data_ptr() == scale_linear.data_ptr()
    assert "_w4a16_weight_scale_linear" not in linear.state_dict()


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


@pytest.mark.parametrize(
    ("sm_version", "dtype", "allowed_backends", "expect_marlin"),
    [
        # Marlin is the default on SM120/121, opt-in on SM89-99, never on SM100.
        (120, torch.bfloat16, None, True),
        (121, torch.bfloat16, None, True),
        (89, torch.bfloat16, ["marlin"], True),
        (90, torch.bfloat16, ["marlin"], True),
        (89, torch.bfloat16, None, False),
        (90, torch.bfloat16, None, False),
        (100, torch.bfloat16, None, False),
        (120, torch.float16, None, False),  # Marlin is bf16-only
    ],
)
def test_w4a16_nvfp4_linear_method_selection(sm_version, dtype, allowed_backends, expect_marlin):
    kwargs = {} if allowed_backends is None else {"nvfp4_allowed_backends": allowed_backends}

    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=sm_version),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
        patch("torch.ops.trtllm.gptq_marlin_repack", create=True),
    ):
        linear = Linear(
            32,
            32,
            bias=False,
            dtype=dtype,
            quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4),
            reduce_output=False,
            **kwargs,
        )

    assert isinstance(linear.quant_method, W4A16NVFP4LinearMethod)
    assert isinstance(linear.quant_method, MarlinNVFP4LinearMethod) is expect_marlin


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


@pytest.mark.parametrize(
    ("checkpoint_has_scale", "exclude_modules", "expected_width"),
    [
        (True, None, 1344),
        (True, ["lm_head"], 2688),
        (False, None, 2688),
    ],
)
def test_causal_lm_head_uses_global_w4a16_nvfp4_config(
    checkpoint_has_scale,
    exclude_modules,
    expected_width,
):
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.W4A16_NVFP4,
        exclude_modules=exclude_modules,
    )
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(
            torch_dtype=torch.float16,
            tie_word_embeddings=False,
        ),
        quant_config=quant_config,
    )

    with patch.object(
        DecoderModelForCausalLM,
        "_checkpoint_has_lm_head_scale",
        return_value=checkpoint_has_scale,
    ):
        causal_lm = DecoderModelForCausalLM(
            torch.nn.Module(),
            config=model_config,
            hidden_size=2688,
            vocab_size=32,
        )

    assert causal_lm.lm_head.weight.shape == (32, expected_width)
    if checkpoint_has_scale and exclude_modules is None:
        assert causal_lm.lm_head.quant_config is quant_config
        assert isinstance(causal_lm.lm_head.quant_method, W4A16NVFP4LinearMethod)
    else:
        assert not causal_lm.lm_head.has_any_quant
