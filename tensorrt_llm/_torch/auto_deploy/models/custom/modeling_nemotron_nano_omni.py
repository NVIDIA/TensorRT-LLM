# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prefill-only NemotronH Nano Omni model for AutoDeploy.

Source checkpoint:
  - nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning

This is a multimodal model (vision + audio + text) whose LLM backbone is NemotronH, a hybrid
Mamba/Attention/MoE architecture. Only the text path is exported by AutoDeploy; vision and
audio towers are kept as stubs whose weights are dropped at load time.

The NemotronH backbone is translated fresh from the HuggingFace source into a lean prefill-only
implementation using AutoDeploy canonical ops for SSM, causal conv1d, attention, and MoE.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType

# =============================================================================
# NemotronH Text Backbone — self-contained prefill-only implementation
# =============================================================================


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size, group_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, hidden_states, gate=None):
        return torch.ops.auto_deploy.torch_rmsnorm_gated(
            x=hidden_states,
            weight=self.weight,
            gate=gate,
            eps=self.variance_epsilon,
            group_size=self.group_size,
            norm_before_gate=False,
        )


class NemotronHRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class NemotronHMamba2Mixer(nn.Module):
    """Mamba-2 SSM mixer using AD canonical ops for causal conv1d and SSM."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.num_heads = config.mamba_num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.mamba_hidden_act
        self.act = ACT2FN[config.mamba_hidden_act]
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.n_groups = config.n_groups
        self.head_dim = config.mamba_head_dim
        self.chunk_size = config.chunk_size
        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)

        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=self.layer_norm_epsilon,
            group_size=self.intermediate_size // self.n_groups,
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def forward(self, input_states):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        projected_states = self.in_proj(input_states)
        gate, hidden_states_B_C, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        # Causal conv1d via canonical op
        hidden_states_B_C = self.act(
            torch.ops.auto_deploy.torch_causal_conv1d(
                hidden_states_B_C,
                self.conv1d.weight,
                self.conv1d.bias,
                self.conv1d.stride[0],
                self.conv1d.padding[0],
                self.conv1d.dilation[0],
                self.conv1d.groups,
                self.conv1d.padding_mode,
            )
        )

        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )

        # SSM via canonical op
        A = -torch.exp(self.A_log.float())
        y = torch.ops.auto_deploy.torch_ssm(
            hidden_states=hidden_states.view(batch_size, seq_len, -1, self.head_dim),
            A=A,
            B=B.view(batch_size, seq_len, -1, self.ssm_state_size),
            C=C.view(batch_size, seq_len, -1, self.ssm_state_size),
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            time_step_limit=list(self.time_step_limit),
            chunk_size=self.chunk_size,
        )
        y = y.reshape(batch_size, seq_len, -1)

        scan_output = self.norm(y, gate)
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states


class NemotronHMLP(nn.Module):
    def __init__(self, config, layer_idx: int, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class NemotronHTopkRouter(nn.Module):
    """DeepSeek-V3 style noaux_tc router in vanilla PyTorch."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )

    def _get_topk_indices(self, scores):
        scores_for_choice = scores.view(
            -1, self.n_routed_experts
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        topk_indices = self._get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class NemotronHMOE(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                NemotronHMLP(
                    config,
                    layer_idx=layer_idx,
                    intermediate_size=config.moe_intermediate_size,
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = NemotronHTopkRouter(config)
        self.shared_experts = NemotronHMLP(
            config=config,
            intermediate_size=config.moe_shared_expert_intermediate_size,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        x_flat = hidden_states.view(-1, hidden_states.shape[-1])

        # Shared expert first (dispatch order matters for AD stream forking)
        shared_out = self.shared_experts(residuals)

        # Routed experts via canonical MoE op
        out_flat = torch.ops.auto_deploy.torch_moe(
            x_flat,
            topk_indices,
            topk_weights,
            w1_weight=[e.up_proj.weight for e in self.experts],
            w2_weight=[e.down_proj.weight for e in self.experts],
            w3_weight=[],
            act_fn=ActivationType.Relu2,
            is_gated_mlp=False,
        )

        routed_out = out_flat.view(*orig_shape)
        return shared_out + routed_out


class NemotronHAttention(nn.Module):
    """Multi-headed attention using AD canonical attention op (handles GQA natively)."""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_heads, self.hidden_size, bias=config.attention_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # AD canonical attention op handles GQA natively — no repeat_kv needed
        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            layout="bsnd",
        )
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class NemotronHBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = NemotronHAttention(config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config, layer_idx=layer_idx)
        elif self.block_type == "moe":
            self.mixer = NemotronHMOE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid block type '{self.block_type}' at layer {layer_idx}")

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states)
        return residual + hidden_states


@dataclass
class NemotronHOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class NemotronHCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class NemotronHPreTrainedModel(PreTrainedModel):
    # config_class intentionally omitted — config loaded via trust_remote_code
    base_model_prefix = "backbone"
    _no_split_modules = ["NemotronHBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        if isinstance(module, NemotronHMamba2Mixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True
            dt = torch.exp(
                torch.rand(self.config.mamba_num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


class NemotronHModel(NemotronHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NemotronHBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm_f = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in list(state_dict.keys()):
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHOutput]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds
        for mixer_block in self.layers:
            hidden_states = mixer_block(hidden_states)

        hidden_states = self.norm_f(hidden_states)
        return NemotronHOutput(last_hidden_state=hidden_states)


class NemotronHForCausalLM(NemotronHPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = NemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHCausalLMOutput]:
        nemotron_h_outputs = self.backbone(input_ids, inputs_embeds=inputs_embeds)
        hidden_states = nemotron_h_outputs[0]
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return NemotronHCausalLMOutput(logits=logits)


# =============================================================================
# Multimodal wrapper — text-only export with weight-dropping stubs
# =============================================================================


class NemotronNanoOmniPreTrainedModel(PreTrainedModel):
    """Base class for the multimodal wrapper. No config_class — loaded via trust_remote_code."""

    _no_split_modules = ["NemotronHBlock"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        pass  # Weights loaded from checkpoint


class NemotronNanoOmniForConditionalGeneration(NemotronNanoOmniPreTrainedModel, GenerationMixin):
    """Multimodal wrapper that holds the NemotronH LLM backbone.

    Vision (RADIO) and audio (Parakeet) towers are represented as empty stubs
    (following the same pattern as Gemma4, Gemma3n, etc.). Their checkpoint weights
    are dropped at load time via a pre-hook. Only the text path
    (``self.language_model``) participates in the exported graph.

    Checkpoint weight key structure (verified against model.safetensors.index.json):
        language_model.backbone.embeddings.weight   → self.language_model.backbone.embeddings
        language_model.backbone.layers.N.mixer.*     → self.language_model.backbone.layers.N.mixer
        language_model.backbone.norm_f.weight        → self.language_model.backbone.norm_f
        language_model.lm_head.weight                → self.language_model.lm_head
        vision_model.radio_model.*                   → DROPPED (stub)
        mlp1.*                                       → DROPPED (stub)
        sound_encoder.encoder.*                      → DROPPED (stub)
        sound_projection.*                           → DROPPED (stub)
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        # LLM backbone — receives the llm sub-config
        self.language_model = NemotronHForCausalLM(config.llm_config)

        # Empty stubs for non-text components (accept weight keys, dropped below)
        self.vision_model = nn.Module()
        self.mlp1 = nn.Module()
        self.sound_encoder = nn.Module()
        self.sound_projection = nn.Module()

        # Drop non-LLM weights at load time
        self._register_load_state_dict_pre_hook(self._drop_multimodal_weights)

        # Expose text_config for TextModelExportInfo.from_autoinferred()
        if not hasattr(config, "text_config"):
            config.text_config = config.llm_config

    @staticmethod
    def _drop_multimodal_weights(state_dict, prefix, *args, **kwargs):
        _MULTIMODAL_PREFIXES = ("vision_model.", "mlp1.", "sound_encoder.", "sound_projection.")
        for key in list(state_dict.keys()):
            if any(key.startswith(prefix + p) for p in _MULTIMODAL_PREFIXES):
                state_dict.pop(key)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.language_model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHCausalLMOutput]:
        assert position_ids is not None
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


# =============================================================================
# Registration
# =============================================================================

# Registration key matches the config class name from the checkpoint's configuration.py:
# class NemotronH_Nano_Omni_Reasoning_V3_Config(PretrainedConfig):
#     model_type = 'NemotronH_Nano_Omni_Reasoning_V3'
# AutoConfig.from_pretrained(model_id, trust_remote_code=True) yields this class.
AutoModelForCausalLMFactory.register_custom_model_cls(
    "NemotronH_Nano_Omni_Reasoning_V3_Config",
    NemotronNanoOmniForConditionalGeneration,
)
