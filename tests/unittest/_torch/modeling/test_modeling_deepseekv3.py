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
"""Per-module quant config resolution for DeepSeek MIXED_PRECISION checkpoints.

DeepSeek-R1-W4AFP8 ships an hf_quant_config with the global
quant_algo=MIXED_PRECISION, which does not map to a single QuantMode. The MoE
experts must resolve their own per-module config (W4A8_AWQ) instead of using
the global one. These tests exercise that resolution on CPU without weights.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.locality_domain import policy
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_deepseekv3, modeling_utils
from tensorrt_llm._torch.models.modeling_deepseekv3 import Deepseekv3MoE
from tensorrt_llm._torch.modules import linear
from tensorrt_llm._torch.moe.fused_moe import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

EXPERTS_KEY = "model.layers.{}.mlp.experts"


@pytest.mark.parametrize("enable_locality_domains", [False, True])
def test_linear_reassigned_quant_config_guard(enable_locality_domains: bool) -> None:
    module = linear.Linear(
        16,
        16,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(),
        locality_domain_policy=policy.LocalityDomainPolicy(enabled=enable_locality_domains),
    )
    weight = module.weight
    module.quant_config = QuantConfig()
    if enable_locality_domains:
        with pytest.raises(RuntimeError, match="quant_config changed"):
            module.create_weights()
    else:
        module.create_weights()
        assert module.weight is weight


def test_layerwise_quant_config_preserves_unquantized_router() -> None:
    gate = modeling_deepseekv3.DeepseekV3Gate(
        hidden_size=16,
        num_experts=4,
        top_k=2,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
        dtype=torch.bfloat16,
    )
    projection = linear.Linear(16, 16, bias=False, skip_create_weights_in_init=True)
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    model = torch.nn.Module()
    model.add_module("gate", gate)
    model.add_module("projection", projection)
    model.model_config = SimpleNamespace(
        quant_config_dict={"gate": quant_config, "projection": quant_config}
    )

    modeling_utils.DecoderModelForCausalLM.apply_layerwise_quant_config(model)
    gate.create_weights()

    assert gate.quant_config is None
    assert gate.weight.dtype == torch.bfloat16
    assert projection.quant_config is quant_config


@pytest.mark.parametrize("use_cute_dsl_bf16_gemm", [False, True])
@pytest.mark.parametrize("attach_lora", [False, True])
@pytest.mark.parametrize("localized_weights", [False, True])
def test_deepseek_small_m_selects_kernel_for_localized_weights(
    use_cute_dsl_bf16_gemm: bool,
    attach_lora: bool,
    localized_weights: bool,
) -> None:
    module = modeling_deepseekv3.DeepseekV3Linear(
        16,
        16,
        bias=False,
        dtype=torch.bfloat16,
        use_cute_dsl_bf16_gemm=use_cute_dsl_bf16_gemm,
        lora=LoraLayer([LoraModuleType.ATTENTION_Q], [16]) if attach_lora else None,
    )
    if localized_weights:
        module._locality_domain_weight_shards = tuple(module.weight.detach().chunk(2))
        module.weight = torch.nn.Parameter(
            torch.empty(0, dtype=torch.bfloat16), requires_grad=False
        )
    inputs = torch.ones(1, 16, dtype=torch.bfloat16)
    expected = torch.ones_like(inputs)
    with (
        patch.object(modeling_deepseekv3, "get_sm_version", return_value=107),
        patch.object(modeling_deepseekv3, "is_sm_100f", return_value=True),
        patch.object(torch.ops.trtllm, "dsv3_fused_a_gemm_op", return_value=expected) as fused,
        patch.object(linear.Linear, "apply_linear", return_value=expected) as fallback,
    ):
        assert module.apply_linear(inputs, None, layer_idx=3) is expected
    if localized_weights:
        fallback.assert_called_once_with(inputs, None, None, 3)
        fused.assert_not_called()
    else:
        fused.assert_called_once()
        fallback.assert_not_called()


@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("use_cute_dsl_bf16_gemm", [False, True])
def test_deepseek_small_m_preserves_lora(num_tokens: int, use_cute_dsl_bf16_gemm: bool) -> None:
    lora_layer = LoraLayer([LoraModuleType.ATTENTION_Q], [16])
    module = modeling_deepseekv3.DeepseekV3Linear(
        16,
        16,
        dtype=torch.bfloat16,
        lora=lora_layer,
        use_cute_dsl_bf16_gemm=use_cute_dsl_bf16_gemm,
    )
    module.weight.data.copy_(2 * torch.eye(16, dtype=torch.bfloat16))
    module.bias.data.fill_(1)
    inputs = torch.ones(num_tokens, 16, dtype=torch.bfloat16)
    adapter_output = torch.full_like(inputs, 3)
    ranks = torch.tensor([1], dtype=torch.int32)
    weight_pointers = torch.zeros((1, 2), dtype=torch.int64)
    lora_params = {
        "num_seqs": 1,
        "host_request_types": torch.zeros(1, dtype=torch.int32),
        "prompt_lens_cpu": torch.tensor([num_tokens], dtype=torch.int32),
        3: {
            int(LoraModuleType.ATTENTION_Q): {
                "adapter_size": ranks,
                "weight_pointers": weight_pointers,
            }
        },
    }

    # Keep Linear.apply_linear and the LoRA merge real. Use CPU implementations
    # for the fused base GEMM and the native adapter GEMM.
    with (
        patch.object(modeling_deepseekv3, "get_sm_version", return_value=107),
        patch.object(linear, "is_sm_100f", return_value=False),
        patch.object(
            torch.ops.trtllm,
            "dsv3_fused_a_gemm_op",
            side_effect=lambda x, weight, bias, _: x @ weight + bias,
        ) as fused,
        patch.object(
            torch.ops.trtllm, "lora_grouped_gemm", return_value=adapter_output
        ) as adapter_gemm,
    ):
        output = module.apply_linear(inputs, module.bias, lora_params, layer_idx=3)

    torch.testing.assert_close(output, torch.full_like(inputs, 6))
    fused.assert_not_called()
    adapter_gemm.assert_called_once()
    assert adapter_gemm.call_args.args[2][0] is ranks
    assert adapter_gemm.call_args.args[3][0] is weight_pointers


@pytest.mark.parametrize("quant_algo", [None, QuantAlgo.NVFP4], ids=["bf16", "nvfp4"])
@pytest.mark.parametrize("per_module_quant", [False, True])
@pytest.mark.parametrize("frozen_config", [False, True])
def test_shared_experts_do_not_partition_locality_domains(
    quant_algo, per_module_quant: bool, frozen_config: bool
) -> None:
    shared_quant_config = QuantConfig(quant_algo=quant_algo, group_size=16)
    global_quant_config = (
        QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION)
        if per_module_quant
        else shared_quant_config
    )
    model_policy = policy.LocalityDomainPolicy(enabled=True)
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(n_group=1, topk_group=1, routed_scaling_factor=1.0),
        quant_config=global_quant_config,
        quant_config_dict=(
            {"model.layers.0.mlp.shared_experts": shared_quant_config} if per_module_quant else None
        ),
        skip_create_weights_in_init=True,
        use_cute_dsl_bf16_gemm=True,
        locality_domain_policy=model_policy,
    )
    model_config._frozen = frozen_config

    # Exercise the real GatedMLP/Linear construction and planner on CPU, while
    # presenting hardware on which both BF16 and NVFP4 can be partitioned.
    with (
        patch(
            "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled",
            return_value=True,
        ),
        patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True),
        patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_RUBIN_AVAILABLE", True),
        patch.object(
            modeling_deepseekv3, "create_moe", return_value=torch.nn.Module()
        ) as create_moe,
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.cuda, "Event"),
    ):
        module = Deepseekv3MoE(
            num_experts=4,
            top_k=2,
            hidden_size=256,
            intermediate_size=256,
            shared_expert_intermediate_size=256,
            aux_stream_dict={modeling_deepseekv3.AuxStreamType.MoeShared: None},
            dtype=torch.bfloat16,
            model_config=model_config,
            layer_idx=0,
        )
        for projection in (module.shared_experts.gate_up_proj, module.shared_experts.down_proj):
            projection.create_weights()
            assert projection.quant_config is shared_quant_config
            assert not projection.partition_plan.enabled
            assert projection._locality_domain_runtime is None

    assert module.shared_experts_use_fp4 == (quant_algo == QuantAlgo.NVFP4)
    assert create_moe.call_args.kwargs["model_config"] is model_config
    assert model_config.quant_config is global_quant_config
    assert model_config.locality_domain_policy is model_policy
    assert model_config.extra_attrs["locality_domain_policy"] is model_policy
    assert model_config.locality_domain_policy.enabled
    assert model_config._frozen == frozen_config


@pytest.fixture
def mixed_precision_config():
    return QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION)


@pytest.fixture
def w4a8_awq_config():
    return QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ, group_size=128)


def test_w4a8_awq_config_is_int4_weight_only_per_group(w4a8_awq_config):
    # This predicate is what selects MoEWeightLoadingMode.W4A8_CUSTOM for the
    # experts in Deepseekv3MoE, so pin it down explicitly.
    assert w4a8_awq_config.layer_quant_mode.is_int4_weight_only_per_group()


def test_experts_quant_config_resolved_from_per_module_dict(
    mixed_precision_config, w4a8_awq_config
):
    model_config = SimpleNamespace(
        quant_config=mixed_precision_config,
        quant_config_dict={EXPERTS_KEY.format(0): w4a8_awq_config},
    )

    resolved = Deepseekv3MoE._get_experts_quant_config(model_config, 0)

    assert resolved is w4a8_awq_config
    assert resolved.quant_algo == QuantAlgo.W4A8_AWQ
    assert resolved.layer_quant_mode.is_int4_weight_only_per_group()
    assert not mixed_precision_config.layer_quant_mode.is_int4_weight_only_per_group()


def test_experts_quant_config_falls_back_to_global_for_unlisted_layer(
    mixed_precision_config, w4a8_awq_config
):
    model_config = SimpleNamespace(
        quant_config=mixed_precision_config,
        quant_config_dict={EXPERTS_KEY.format(0): w4a8_awq_config},
    )

    assert Deepseekv3MoE._get_experts_quant_config(model_config, 1) is mixed_precision_config


def test_experts_quant_config_falls_back_to_global_without_dict(mixed_precision_config):
    model_config = SimpleNamespace(quant_config=mixed_precision_config, quant_config_dict=None)

    assert Deepseekv3MoE._get_experts_quant_config(model_config, 0) is mixed_precision_config


def test_expert_weight_loading_mode_w4a8_custom_for_w4a8_awq(w4a8_awq_config):
    # The resolved W4A8_AWQ expert config selects the custom loading mode; this
    # is the assignment the MoE construction makes from the resolved config.
    assert (
        Deepseekv3MoE._expert_weight_loading_mode(w4a8_awq_config)
        is MoEWeightLoadingMode.W4A8_CUSTOM
    )


def test_expert_weight_loading_mode_vanilla_for_non_int4(mixed_precision_config):
    # Neither the ambiguous MIXED_PRECISION global nor a plain FP8 config is
    # int4-weight-per-group, so both fall to VANILLA.
    assert (
        Deepseekv3MoE._expert_weight_loading_mode(mixed_precision_config)
        is MoEWeightLoadingMode.VANILLA
    )
    assert (
        Deepseekv3MoE._expert_weight_loading_mode(QuantConfig(quant_algo=QuantAlgo.FP8))
        is MoEWeightLoadingMode.VANILLA
    )


def test_expert_weight_loading_mode_none_is_vanilla():
    # override_quant_config is Optional, so the resolved expert config is None on
    # an unquantized layer; the mode selection must not dereference it.
    assert Deepseekv3MoE._expert_weight_loading_mode(None) is MoEWeightLoadingMode.VANILLA


def _bare_configurable_moe(override_quant_config):
    moe = object.__new__(ConfigurableMoE)
    moe._override_quant_config = override_quant_config
    return moe


def test_quant_config_dict_prefers_override_over_mixed_precision_global(
    mixed_precision_config, w4a8_awq_config
):
    moe = _bare_configurable_moe(w4a8_awq_config)
    model_config = SimpleNamespace(quant_config=mixed_precision_config)

    result = ConfigurableMoE._get_quant_config_dict(moe, model_config)

    assert result == {
        "has_fp8_qdq": False,
        "has_nvfp4": False,
        "has_w4afp8": True,
        "has_fp8_block_scales": False,
    }
    # The global MIXED_PRECISION mode would not have flagged w4afp8.
    assert not mixed_precision_config.layer_quant_mode.is_int4_weight_only_per_group()


def test_quant_config_dict_falls_back_to_global_without_override():
    fp8_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    moe = _bare_configurable_moe(None)
    model_config = SimpleNamespace(quant_config=fp8_config)

    result = ConfigurableMoE._get_quant_config_dict(moe, model_config)

    assert result == {
        "has_fp8_qdq": True,
        "has_nvfp4": False,
        "has_w4afp8": False,
        "has_fp8_block_scales": False,
    }


def test_quant_config_dict_is_none_when_unquantized():
    moe = _bare_configurable_moe(None)
    model_config = SimpleNamespace(quant_config=None)

    assert ConfigurableMoE._get_quant_config_dict(moe, model_config) is None
