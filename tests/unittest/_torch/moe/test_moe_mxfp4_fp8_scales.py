# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accumulator-unit contracts for MXFP4 weights with per-tensor FP8 inputs."""

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.impl_blocks import MoEWeightOwnerMixin
from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    FusedMoEMethodBase,
    MXFP4WeightTRTLLMGenFusedMoEMethod,
    W4A8MXFP4FP8TRTLLMGenFusedMoEMethod,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("gptoss", [False, True])
@pytest.mark.parametrize(
    "loading_mode",
    [MoEWeightLoadingMode.VANILLA, MoEWeightLoadingMode.FUSED_GATE_UP_PROJ],
)
def test_mxfp4_fp8_trtllm_gen_accumulator_params(monkeypatch, bias, gptoss, loading_mode):
    method = W4A8MXFP4FP8TRTLLMGenFusedMoEMethod()

    class WeightOwner(MoEWeightOwnerMixin, torch.nn.Module):
        def _get_quant_method(self):
            return method

    def create_module() -> WeightOwner:
        owner = WeightOwner()
        owner.num_experts = 4
        owner.expert_size_per_partition = 2
        owner.initial_local_expert_ids = [1, 3]
        owner.weight_loading_mode = loading_mode
        owner.bias = bias
        owner.register_buffer(
            "act_alpha", torch.tensor([1.702, 1.5]) if gptoss else None, persistent=False
        )
        owner.register_buffer(
            "act_beta", torch.tensor([1.0, 2.0]) if gptoss else None, persistent=False
        )
        owner.register_buffer(
            "act_clamp", torch.tensor([7.0, 9.0]) if gptoss else None, persistent=False
        )
        return owner

    module = create_module()

    def create_weight_storage(self, module):
        module.register_parameter("w3_w1_weight", torch.nn.Parameter(torch.zeros(2, 4, 2)))
        module.register_parameter("w2_weight", torch.nn.Parameter(torch.zeros(2, 2, 2)))
        module.register_buffer("w3_w1_bias", torch.zeros(2, 4) if bias else None)
        module.register_buffer("w2_bias", torch.zeros(2, 2) if bias else None)

    def load_experts(module, weights, mode, expert_ids, w31, w2, b31, b2):
        if bias:
            b31.copy_(weights["fc1_bias"])
            b2.copy_(weights["fc2_bias"])

    # Weight packing is orthogonal to the scale contract and requires CUDA.
    monkeypatch.setattr(MXFP4WeightTRTLLMGenFusedMoEMethod, "create_weights", create_weight_storage)
    monkeypatch.setattr(MXFP4WeightTRTLLMGenFusedMoEMethod, "load_quant_scales", lambda *a: None)
    monkeypatch.setattr(
        MXFP4WeightTRTLLMGenFusedMoEMethod,
        "cache_derived_state",
        FusedMoEMethodBase.cache_derived_state,
    )
    monkeypatch.setattr(method, "load_expert_weights_to_dst", load_experts)
    module.create_weights()

    raw_beta = torch.tensor([1.0, 2.0])
    raw_clamp = torch.tensor([7.0, 9.0])
    fc1_bias = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    fc2_bias = torch.arange(1, 5, dtype=torch.float32).reshape(2, 2)
    for fc1_scale, fc2_scale in ((0.5, 0.25), (0.25, 0.5)):
        weights = {"fc1_bias": fc1_bias, "fc2_bias": fc2_bias}
        if loading_mode == MoEWeightLoadingMode.VANILLA:
            for expert in range(module.num_experts):
                weights[f"{expert}.w1.input_scale"] = torch.tensor(fc1_scale)
                weights[f"{expert}.w3.input_scale"] = torch.tensor(fc1_scale)
                weights[f"{expert}.w2.input_scale"] = torch.tensor(fc2_scale)
        else:
            weights["gate_up_proj_input_scale"] = torch.tensor(fc1_scale)
            weights["down_proj_input_scale"] = torch.tensor(fc2_scale)

        module.load_weights([weights])
        module.post_load_weights()
        module.post_load_weights()
        module.process_weights_after_loading()

        torch.testing.assert_close(
            module.fc31_input_dequant, torch.full((2,), fc1_scale / fc2_scale)
        )
        if gptoss:
            torch.testing.assert_close(module.act_alpha, torch.tensor([1.702, 1.5]))
            torch.testing.assert_close(module.act_beta, raw_beta / fc1_scale)
            torch.testing.assert_close(module.act_clamp, raw_clamp / fc1_scale)
        else:
            assert module.act_beta is None
            assert module.act_clamp is None
        if bias:
            torch.testing.assert_close(module.w3_w1_bias, fc1_bias / fc1_scale)
            torch.testing.assert_close(module.w2_bias, fc2_bias / fc2_scale)
        assert "_act_beta_unscaled" not in module.state_dict()
        assert "_act_clamp_unscaled" not in module.state_dict()

        # Runtime weights omit the non-persistent activation configuration.
        restored = create_module()
        restored.create_weights()
        restored.load_state_dict(module.state_dict())
        restored.post_load_weights()
        restored.post_load_weights()
        if gptoss:
            torch.testing.assert_close(restored.act_beta, raw_beta / fc1_scale)
            torch.testing.assert_close(restored.act_clamp, raw_clamp / fc1_scale)
        else:
            assert restored.act_beta is None
            assert restored.act_clamp is None
        if bias:
            torch.testing.assert_close(restored.w3_w1_bias, module.w3_w1_bias)
            torch.testing.assert_close(restored.w2_bias, module.w2_bias)
