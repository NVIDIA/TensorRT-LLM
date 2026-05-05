# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A patch for MiniMax-M2 MoE to make it compatible with torch.export.

MiniMax-M2 is loaded from HuggingFace hub (trust_remote_code), so we cannot
import MiniMaxM2SparseMoeBlock directly. Instead, we use the same pattern as
DeepSeek: patching AutoModelForCausalLM.from_config to iterate over modules
and patch by class name.
"""

import types
from typing import Dict

import torch
from transformers import AutoModelForCausalLM


def minimax_m2_moe(self, hidden_states: torch.Tensor):
    """MiniMaxM2SparseMoeBlock forward function rewritten to enable torch.export.

    Key differences from Mixtral:
    - Uses sigmoid activation for routing (not softmax)
    - Has e_score_correction_bias added for expert selection only
    - Gathers original sigmoid weights after topk selection
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(
            1.0 - self.jitter_noise, 1.0 + self.jitter_noise
        )
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    can_use_fused_router = hidden_states.is_cuda
    if can_use_fused_router and self.gate.weight.dtype != torch.float32:
        router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
            hidden_states, self.gate.weight.t(), bias=None, out_dtype=torch.float32
        )
    else:
        router_logits = self.gate(hidden_states)
    if can_use_fused_router:
        top_k_weights, selected_experts = torch.ops.trtllm.noaux_tc_op(
            router_logits,
            self.e_score_correction_bias,
            1,
            1,
            self.top_k,
            1.0,
        )
    else:
        routing_weights = torch.sigmoid(router_logits)
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, selected_experts)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        top_k_weights,
        w1_weight=[expert.w1.weight for expert in self.experts],  # gate projection
        w2_weight=[expert.w2.weight for expert in self.experts],  # down projection
        w3_weight=[expert.w3.weight for expert in self.experts],  # up projection
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


_from_config_previous = AutoModelForCausalLM.from_config

CUSTOM_MODULE_PATCHES: Dict[str, callable] = {"MiniMaxM2SparseMoeBlock": minimax_m2_moe}


def get_model_from_config_patched(config, **kwargs):
    model = _from_config_previous(config, **kwargs)
    # Patch modules by class name
    for _, module in model.named_modules():
        if type(module).__name__ in CUSTOM_MODULE_PATCHES:
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[type(module).__name__], module)

    return model


# Patch AutoModelForCausalLM.from_config
AutoModelForCausalLM.from_config = get_model_from_config_patched
