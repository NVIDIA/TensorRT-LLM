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
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MarlinNVFP4LinearMethod,
    NVFP4LinearMethod,
    TensorParallelMode,
    W4A16NVFP4LinearMethod,
    get_quant_method,
    get_sm_version,
    quant_config_has_nvfp4_activation_quantization,
)
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import gelu_tanh, is_nvfp4_marlin_enabled, model_extra_attrs, relu2
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _run_w4a16_marlin_reference_case(m: int, n: int, k: int) -> None:
    act, weight, weight_scale, weight_scale_2 = _make_w4a16_nvfp4_case(m, n, k, torch.bfloat16)

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

    with patch.object(MarlinNVFP4LinearMethod, "is_supported", return_value=False):
        reference = Linear(
            k,
            n,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4),
            reduce_output=False,
        ).cuda()
    assert type(reference.quant_method) is W4A16NVFP4LinearMethod
    reference.weight.data.copy_(weight)
    reference.weight_scale.data.copy_(weight_scale)
    reference.weight_scale_2.data.copy_(weight_scale_2)
    reference.transform_weights()

    expected = reference(act)
    actual = linear(act)
    torch.testing.assert_close(actual, expected, atol=0.75, rtol=0.02)


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
    not torch.cuda.is_available() or get_sm_version() not in (90, 120, 121),
    reason="requires CUDA SM90 or SM120/121",
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
def test_w4a16_nvfp4_marlin_bf16_matches_triton(shape):
    m, n, k = shape
    _run_w4a16_marlin_reference_case(m, n, k)


def test_get_quant_method_returns_w4a16_nvfp4_linear_method():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)

    method = get_quant_method(quant_config)

    assert type(method) is W4A16NVFP4LinearMethod


def test_nvfp4_activation_quantization_excludes_w4a16():
    assert quant_config_has_nvfp4_activation_quantization(QuantConfig(quant_algo=QuantAlgo.NVFP4))
    assert not quant_config_has_nvfp4_activation_quantization(
        QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    )


def test_nvfp4_marlin_utility_requires_explicit_opt_in():
    with (
        patch("tensorrt_llm._torch.utils.get_sm_version", return_value=90),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
    ):
        assert not is_nvfp4_marlin_enabled()
        with model_extra_attrs({"nvfp4_gemm_allowed_backends": ["cutlass", "marlin"]}):
            assert is_nvfp4_marlin_enabled()


def test_nvfp4_rmsnorm_keeps_high_precision_output_for_hopper_marlin():
    with (
        patch("tensorrt_llm._torch.modules.rms_norm.get_sm_version", return_value=90),
        patch("tensorrt_llm._torch.modules.rms_norm.is_nvfp4_marlin_enabled", return_value=True),
    ):
        norm = RMSNorm(
            hidden_size=32,
            eps=1e-5,
            dtype=torch.bfloat16,
            quantize_type="nvfp4",
            return_hp_output=True,
        )

    assert not norm.is_nvfp4
    assert not norm.return_hp_output
    assert norm.nvfp4_scale is None


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
        assert type(o_proj.quant_method) is NVFP4LinearMethod
        assert o_proj.has_nvfp4_activation_quantization
        assert not Attention._use_quantize_output(attention)


def test_w4a16_disables_fused_gemm_allreduce(monkeypatch):
    monkeypatch.setenv("TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED", "1")
    mapping = Mapping(world_size=2, rank=0, tp_size=2)

    with (
        patch("tensorrt_llm._torch.modules.linear.mpi_disabled", return_value=False),
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=120),
        patch("tensorrt_llm._torch.modules.linear.ipc_nvls_supported", return_value=True),
        patch("tensorrt_llm._torch.distributed.AllReduce"),
    ):
        linear = Linear(
            256,
            64,
            bias=False,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
            reduce_output=True,
            skip_create_weights_in_init=True,
        )
        assert linear.use_fused_gemm_allreduce

        # Simulate apply_layerwise_quant_config rebinding a mixed-precision
        # layer after Linear.__init__ but before deferred weight creation.
        linear.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
        linear.create_weights()

    assert not linear.use_fused_gemm_allreduce


@pytest.mark.parametrize(
    ("sm_version", "dtype", "expected_backends"),
    [
        (90, torch.bfloat16, "marlin"),
        (90, torch.float16, "cutlass,cublaslt,cuda_core"),
        (120, torch.bfloat16, "cutlass,cublaslt,cuda_core"),
        (121, torch.bfloat16, "cutlass,cublaslt,cuda_core"),
    ],
)
def test_nvfp4_linear_uses_architecture_default_backend(sm_version, dtype, expected_backends):
    method = NVFP4LinearMethod()
    input_tensor = torch.ones((2, 32), dtype=dtype)
    module = SimpleNamespace(
        weight=torch.empty((4, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        out_features=4,
        dtype=dtype,
        nvfp4_allowed_backends=["cutlass", "cublaslt", "cuda_core"],
        all_reduce=None,
        mapping=None,
    )
    act_fp4 = torch.empty((2, 16), dtype=torch.uint8)
    act_sf = torch.empty((128 * 4,), dtype=torch.uint8)
    alpha = torch.ones((1,), dtype=torch.float32)
    captured = {}

    def fake_nvfp4_gemm(*args, **kwargs):
        captured["allowed_backends"] = kwargs["allowed_backends"]
        return torch.ones((2, 4), dtype=dtype)

    with (
        patch.object(method, "_input_prepare", return_value=(act_fp4, act_sf, alpha)),
        patch(
            "tensorrt_llm._torch.modules.linear.get_sm_version",
            return_value=sm_version,
        ),
        patch(
            "torch.ops.trtllm.nvfp4_gemm",
            side_effect=fake_nvfp4_gemm,
            create=True,
        ),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
        patch("torch.ops.trtllm.gptq_marlin_repack", create=True),
    ):
        output = method.apply(module, input_tensor, bias=None)

    assert captured["allowed_backends"] == expected_backends
    assert output.shape == (2, 4)


@pytest.mark.parametrize("sm_version", [90, 120, 121])
def test_nvfp4_linear_preserves_activation_quant_method(sm_version):
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    with (
        patch(
            "tensorrt_llm._torch.modules.linear.get_sm_version",
            return_value=sm_version,
        ),
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
        assert type(linear.quant_method) is NVFP4LinearMethod
        assert linear.has_nvfp4_activation_quantization
        assert linear.uses_marlin_nvfp4 is (sm_version == 90)


def test_nvfp4_linear_hopper_fp16_keeps_normal_method():
    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=90),
        patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True),
        patch("torch.ops.trtllm.gptq_marlin_repack", create=True),
    ):
        linear = Linear(
            32,
            32,
            bias=False,
            dtype=torch.float16,
            quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
            reduce_output=False,
        )

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
        nvfp4_allowed_backends=["cutlass", "cublaslt", "cuda_core"],
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


def test_w4a16_nvfp4_mlp_disables_relu2_fp4_fusion_without_input_scale():
    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4))

    with (
        patch("tensorrt_llm._torch.modules.mlp.get_sm_version", return_value=121),
        patch("torch.ops.trtllm.fused_relu2_quantize", create=True),
    ):
        mlp = MLP(
            hidden_size=32,
            intermediate_size=64,
            bias=False,
            activation=relu2,
            dtype=torch.bfloat16,
            config=model_config,
            reduce_output=False,
        )
        mlp.create_weights()

    assert mlp.down_proj.has_nvfp4
    assert mlp.down_proj.input_scale is None
    assert not mlp._use_fused_relu2_quant


def test_nvfp4_mlp_enables_relu2_fp4_fusion_with_static_input_scale():
    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4))

    with (
        patch("tensorrt_llm._torch.modules.mlp.get_sm_version", return_value=121),
        patch("torch.ops.trtllm.fused_relu2_quantize", create=True),
    ):
        mlp = MLP(
            hidden_size=32,
            intermediate_size=64,
            bias=False,
            activation=relu2,
            dtype=torch.bfloat16,
            config=model_config,
            reduce_output=False,
        )
        mlp.create_weights()

    assert mlp.down_proj.input_scale is not None
    assert mlp._use_fused_relu2_quant


def test_w4a16_nvfp4_mlp_rechecks_relu2_fp4_fusion_before_forward():
    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4))

    with (
        patch("tensorrt_llm._torch.modules.mlp.get_sm_version", return_value=121),
        patch("torch.ops.trtllm.fused_relu2_quantize", create=True),
    ):
        mlp = MLP(
            hidden_size=32,
            intermediate_size=64,
            bias=False,
            activation=relu2,
            dtype=torch.bfloat16,
            config=model_config,
            reduce_output=False,
        )
        mlp.create_weights()

    # Simulate eligibility cached before weight loading replaced the linear
    # method with W4A16, whose high-precision activation has no input_scale.
    mlp._use_fused_relu2_quant = True
    x_up = torch.tensor([[-2.0, 3.0]], dtype=torch.bfloat16)
    with (
        patch.object(mlp.up_proj, "forward", return_value=x_up),
        patch.object(mlp.down_proj, "forward", side_effect=lambda x: x),
        patch.object(
            MLP,
            "_fused_relu2_quant",
            side_effect=AssertionError("missing input_scale must use unfused ReLU2"),
        ),
    ):
        output = mlp(torch.empty((1, 32), dtype=torch.bfloat16))

    torch.testing.assert_close(output, relu2(x_up))


def test_w4a16_nvfp4_mlp_disables_cutedsl_gelu_fusion():
    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4))

    with (
        patch("tensorrt_llm._torch.modules.mlp.get_sm_version", return_value=100),
        patch("torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_gelu_blackwell", create=True),
        patch(
            "torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell",
            create=True,
        ),
    ):
        mlp = MLP(
            hidden_size=32,
            intermediate_size=64,
            bias=False,
            activation=gelu_tanh,
            dtype=torch.bfloat16,
            config=model_config,
            reduce_output=False,
        )
        mlp.create_weights()

    assert not mlp._use_fused_gelu
    assert not mlp._use_fused_gelu_fp4out


def test_dynamic_nvfp4_mlp_keeps_bf16_cutedsl_gelu_fusion():
    model_config = ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        force_dynamic_quantization=True,
    )

    with (
        patch("tensorrt_llm._torch.modules.mlp.get_sm_version", return_value=100),
        patch("torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_gelu_blackwell", create=True),
        patch(
            "torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell",
            create=True,
        ),
    ):
        mlp = MLP(
            hidden_size=32,
            intermediate_size=64,
            bias=False,
            activation=gelu_tanh,
            dtype=torch.bfloat16,
            config=model_config,
            reduce_output=False,
        )
        mlp.create_weights()

    assert mlp._use_fused_gelu
    assert not mlp._use_fused_gelu_fp4out


def test_w4a16_nvfp4_gated_mlp_disables_cutedsl_swiglu_fusion():
    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4))
    mlp = GatedMLP(
        hidden_size=32,
        intermediate_size=64,
        bias=False,
        dtype=torch.bfloat16,
        config=model_config,
        reduce_output=False,
        use_cute_dsl_blockscaling_mm=True,
    )

    assert not mlp._can_fuse_gate_up_swiglu()
    assert not mlp._can_fuse_gate_up_swiglu_fp4out()


def test_dynamic_nvfp4_gated_mlp_keeps_bf16_cutedsl_swiglu_fusion():
    model_config = ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        force_dynamic_quantization=True,
    )
    mlp = GatedMLP(
        hidden_size=32,
        intermediate_size=64,
        bias=False,
        dtype=torch.bfloat16,
        config=model_config,
        reduce_output=False,
        use_cute_dsl_blockscaling_mm=True,
    )

    assert mlp._can_fuse_gate_up_swiglu()
    assert not mlp._can_fuse_gate_up_swiglu_fp4out()


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
            "tensorrt_llm._torch.modules.fused_moe.triton_dequant_nvfp4.dequant_nvfp4_2d_triton",
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


def test_w4a16_nvfp4_linear_restores_high_rank_input_shape():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 3, 32), dtype=torch.float16)
    module = SimpleNamespace(
        weight=torch.empty((8, 16), dtype=torch.uint8),
        weight_scale=torch.empty((128 * 4,), dtype=torch.uint8),
        _w4a16_weight_scale_linear=torch.empty((128 * 4,), dtype=torch.uint8),
        weight_scale_2=torch.tensor([0.5], dtype=torch.float32),
        dtype=torch.float16,
        out_features=8,
        scaling_vector_size=16,
        pre_quant_scale=None,
        use_custom_cublas_mm=False,
    )

    def fake_dequant(*args, **kwargs):
        return torch.ones((8, 32), dtype=torch.float16)

    with patch(
        "tensorrt_llm._torch.modules.fused_moe.triton_dequant_nvfp4.dequant_nvfp4_2d_triton",
        side_effect=fake_dequant,
    ):
        output = method.apply(module, input_tensor, bias=None)

    assert output.shape == (2, 3, 8)


def test_w4a16_nvfp4_linear_uses_triton_dequant():
    method = W4A16NVFP4LinearMethod()
    input_tensor = torch.ones((2, 32), dtype=torch.bfloat16)
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

    with patch(
        "tensorrt_llm._torch.modules.fused_moe.triton_dequant_nvfp4.dequant_nvfp4_2d_triton",
        side_effect=fake_dequant,
    ):
        output = method.apply(module, input_tensor, bias=bias)

    assert captured["weight"].data_ptr() == module.weight.data_ptr()
    assert captured["weight_scale"] is module._w4a16_weight_scale_linear
    assert captured["weight_scale_2"] is module.weight_scale_2
    assert captured["target_dtype"] is torch.bfloat16
    assert captured["sf_vec_size"] == 16
    expected = torch.tensor([33.0, 34.0, 35.0, 36.0], dtype=torch.bfloat16).expand(2, 4)
    torch.testing.assert_close(output, expected)


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


@pytest.mark.parametrize("sm_version", [90, 120, 121])
def test_w4a16_nvfp4_linear_selects_marlin_by_default(sm_version):
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)

    with (
        patch(
            "tensorrt_llm._torch.modules.linear.get_sm_version",
            return_value=sm_version,
        ),
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


def test_w4a16_nvfp4_linear_uses_default_method_on_sm100():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)

    with (
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=100),
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

    assert type(linear.quant_method) is W4A16NVFP4LinearMethod


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


def test_lm_head_w4a16_nvfp4_forward_uses_triton_dequant():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    lm_head = LMHead(
        num_embeddings=4,
        embedding_dim=32,
        dtype=torch.float16,
        quant_config=quant_config,
    )
    input_tensor = torch.ones((2, 32), dtype=torch.float16)
    lm_head._w4a16_weight_scale_linear = torch.empty((128 * 4,), dtype=torch.uint8)
    captured = {}

    def fake_dequant(weight, weight_scale, weight_scale_2, **kwargs):
        captured["weight"] = weight
        captured["weight_scale"] = weight_scale
        captured["weight_scale_2"] = weight_scale_2
        captured.update(kwargs)
        return torch.ones((4, 32), dtype=torch.float16)

    with patch(
        "tensorrt_llm._torch.modules.fused_moe.triton_dequant_nvfp4.dequant_nvfp4_2d_triton",
        side_effect=fake_dequant,
    ):
        output = lm_head(input_tensor)

    assert captured["weight"].data_ptr() == lm_head.weight.data_ptr()
    assert captured["weight_scale"] is lm_head._w4a16_weight_scale_linear
    assert captured["weight_scale_2"] is lm_head.weight_scale_2
    assert captured["target_dtype"] is torch.float16
    assert captured["sf_vec_size"] == 16
    assert output.shape == (2, 4)
