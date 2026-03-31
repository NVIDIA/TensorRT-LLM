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

"""Slimmed down Gemma 3n text implementation for AutoDeploy export.

This implementation follows the Hugging Face Gemma 3n text stack closely while
keeping only the prefill path needed by AutoDeploy. The outer
``Gemma3nForConditionalGeneration`` wrapper preserves the HF text checkpoint
layout (``model.language_model.*`` + ``lm_head``) and drops unsupported
vision/audio tower weights at load time. The forward path intentionally
supports only text-only export.
"""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gemma3n.configuration_gemma3n import (
    Gemma3nAudioConfig,
    Gemma3nConfig,
    Gemma3nTextConfig,
    Gemma3nVisionConfig,
)
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


def _build_rope_cache(
    config: Gemma3nTextConfig,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
    else:
        rope_type = "default"

    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, device=None)
    positions = torch.arange(config.max_position_embeddings, dtype=inv_freq.dtype)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin(), attention_scaling


class Gemma3nRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


class Gemma3nTextScaledWordEmbedding(nn.Embedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(dtype=self.weight.dtype)


class Gemma3nTextLaurelBlock(nn.Module):
    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        self.linear_left = nn.Linear(config.hidden_size, config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(config.laurel_rank, config.hidden_size, bias=False)
        self.post_laurel_norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
        return hidden_states + laurel_hidden_states


class Gemma3nTextMLP(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size[layer_idx]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]
        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        if self.activation_sparsity > 0.0:
            normal_dist = torch.distributions.normal.Normal(0, 1)
            std_multiplier = normal_dist.icdf(
                torch.tensor(self.activation_sparsity, dtype=torch.float32)
            )
            self.register_buffer(
                "activation_sparsity_std_multiplier", std_multiplier, persistent=False
            )
        else:
            self.register_buffer(
                "activation_sparsity_std_multiplier",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=False,
            )

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        std_multiplier = self.activation_sparsity_std_multiplier.to(
            device=inputs.device, dtype=inputs.dtype
        )
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff = inputs_mean + inputs_std * std_multiplier
        return torch.relu(inputs - cutoff)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_proj = self.gate_proj(hidden_states)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(hidden_states)
        return self.down_proj(activations * up_proj)


class Gemma3nTextAltUp(nn.Module):
    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        self.config = config
        self.correct_output_scale = nn.Parameter(torch.zeros(config.hidden_size))
        self.correction_coefs = nn.Linear(
            config.altup_num_inputs, config.altup_num_inputs, bias=False
        )
        self.prediction_coefs = nn.Linear(
            config.altup_num_inputs, config.altup_num_inputs**2, bias=False
        )
        self.modality_router = nn.Linear(config.hidden_size, config.altup_num_inputs, bias=False)
        self.router_norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer(
            "router_input_scale", torch.tensor(config.hidden_size**-1.0), persistent=False
        )

    def compute_router_modalities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(hidden_states) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(hidden_states)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        modalities = self.compute_router_modalities(hidden_states[self.config.altup_active_idx])
        all_coefs = self.prediction_coefs(modalities).reshape(
            *modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs
        )
        all_coefs = all_coefs.permute(0, 1, 3, 2)
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)
        return (predictions + hidden_states).contiguous().type_as(hidden_states)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.config.altup_active_idx]
        innovation = innovation.repeat(self.config.altup_num_inputs, 1, 1, 1)
        all_coefs = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)
        corrected = torch.mul(innovation, all_coefs)
        return (corrected + predictions).contiguous().type_as(activated)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(
            corrected
        )


class Gemma3nTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        cos, sin, attention_scaling = _build_rope_cache(config)
        self.register_buffer("_ad_cos_cached", cos * attention_scaling, persistent=False)
        self.register_buffer("_ad_sin_cached", sin * attention_scaling, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del position_ids
        cos = self._ad_cos_cached.to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached.to(dtype=x.dtype, device=x.device)
        return cos, sin


def _slice_rope_cache(
    position_embeddings: Tuple[torch.Tensor, torch.Tensor], position_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    return cos[position_ids], sin[position_ids]


class Gemma3nMultimodalEmbedder(nn.Module):
    def __init__(
        self,
        multimodal_config: Gemma3nAudioConfig | Gemma3nVisionConfig,
        text_config: Gemma3nTextConfig,
    ):
        super().__init__()
        self.multimodal_hidden_size = multimodal_config.hidden_size
        self.eps = multimodal_config.rms_norm_eps
        self.vocab_offset = multimodal_config.vocab_offset
        self.vocab_size = multimodal_config.vocab_size
        self.text_hidden_size = text_config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.multimodal_hidden_size)
        self.hard_embedding_norm = Gemma3nRMSNorm(self.multimodal_hidden_size, eps=self.eps)
        self.soft_embedding_norm = Gemma3nRMSNorm(self.multimodal_hidden_size, eps=self.eps)
        self.embedding_projection = nn.Linear(
            self.multimodal_hidden_size, self.text_hidden_size, bias=False
        )
        self.embedding_post_projection_norm = Gemma3nRMSNorm(
            self.text_hidden_size, eps=self.eps, with_scale=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is not None:
            embeddings = self.soft_embedding_norm(inputs_embeds)
        else:
            embeddings = self.embedding(input_ids - self.vocab_offset)
            embeddings = self.hard_embedding_norm(embeddings)
        embeddings = self.embedding_projection(embeddings)
        return self.embedding_post_projection_norm(embeddings)


class Gemma3nTextAttention(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            self.kv_shared_layer_index = (
                len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
            )
        else:
            self.kv_shared_layer_index = None

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Gemma3nRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3nRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma3nRMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        cos, sin = position_embeddings
        query_states, key_states = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            query_states,
            key_states,
            cos,
            sin,
            2,
        )

        if self.is_kv_shared_layer:
            attn_output = torch.ops.auto_deploy.torch_attention_shared_kv(
                query_states,
                key_states,
                value_states,
                None,
                0.0,
                True,
                1.0,
                None,
                self.sliding_window,
                None,
                "bsnd",
                self.layer_idx,
                self.kv_shared_layer_index,
            )
        else:
            attn_output = torch.ops.auto_deploy.torch_attention(
                query_states,
                key_states,
                value_states,
                None,
                0.0,
                True,
                1.0,
                None,
                self.sliding_window,
                None,
                "bsnd",
                self.layer_idx,
            )
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class Gemma3nTextDecoderLayer(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3nTextAttention(config, layer_idx)
        self.mlp = Gemma3nTextMLP(config, layer_idx=layer_idx)
        self.input_layernorm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3nRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.act_fn = ACT2FN[config.hidden_activation]

        self.altup = Gemma3nTextAltUp(config)
        self.laurel = Gemma3nTextLaurelBlock(config)
        self.per_layer_input_gate = nn.Linear(
            config.hidden_size, config.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = nn.Linear(
            config.hidden_size_per_layer_input, config.hidden_size, bias=False
        )
        self.post_per_layer_input_norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
        position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
        per_layer_input: torch.Tensor,
    ) -> torch.Tensor:
        predictions = self.altup.predict(hidden_states)
        active_idx = getattr(self.altup, "active_idx", self.altup.config.altup_active_idx)
        active_prediction = predictions[active_idx]
        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        attn = self.self_attn(active_prediction_normed, position_embeddings)
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2.0)

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        corrected_predictions = self.altup.correct(predictions, attn_laurel + attn_ffw_norm)

        first_prediction = corrected_predictions[active_idx].clone()
        if self.altup.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = self.act_fn(first_prediction)
        first_prediction = torch.multiply(first_prediction, per_layer_input)
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)
        for idx in range(corrected_predictions.shape[0]):
            if idx != active_idx:
                corrected_predictions[idx] += first_prediction
        return corrected_predictions


@dataclass
class Gemma3nTextOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Gemma3nCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class Gemma3nConditionalOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class Gemma3nTextPreTrainedModel(PreTrainedModel):
    config_class = Gemma3nTextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Gemma3nTextDecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Gemma3nTextAltUp):
            module.correct_output_scale.data.zero_()


class Gemma3nPreTrainedModel(PreTrainedModel):
    config_class = Gemma3nConfig
    base_model_prefix = "model"
    _no_split_modules = ["Gemma3nTextDecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Gemma3nTextAltUp):
            module.correct_output_scale.data.zero_()


class Gemma3nTextModel(Gemma3nTextPreTrainedModel):
    def __init__(self, config: Gemma3nTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = Gemma3nTextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                Gemma3nTextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3nTextRotaryEmbedding(config)

        local_config = copy.deepcopy(config)
        local_config.rope_theta = local_config.rope_local_base_freq
        local_config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3nTextRotaryEmbedding(local_config)

        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            self.padding_idx,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = Gemma3nRMSNorm(
            config.hidden_size_per_layer_input, eps=config.rms_norm_eps
        )
        self.altup_projections = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                for _ in range(1, config.altup_num_inputs)
            ]
        )
        self.altup_unembed_projections = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                for _ in range(1, config.altup_num_inputs)
            ]
        )
        self.register_buffer(
            "per_layer_projection_scale", torch.tensor(config.hidden_size**-0.5), persistent=False
        )
        self.register_buffer(
            "per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False
        )
        self.register_buffer("_ad_eps", torch.tensor(1e-5), persistent=False)
        self._register_load_state_dict_pre_hook(self._slice_reduced_layer_weights)
        self.post_init()

    def _slice_reduced_layer_weights(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        keys_to_params = {
            prefix + "embed_tokens_per_layer.weight": self.embed_tokens_per_layer.weight,
            prefix + "per_layer_model_projection.weight": self.per_layer_model_projection.weight,
        }
        for state_key, target_param in keys_to_params.items():
            if state_key not in state_dict:
                continue
            checkpoint_weight = state_dict[state_key]
            if checkpoint_weight.ndim != 2:
                continue
            if (
                checkpoint_weight.shape[0] == target_param.shape[0]
                and checkpoint_weight.shape[1] > target_param.shape[1]
            ):
                state_dict[state_key] = checkpoint_weight[:, : target_param.shape[1]]
            elif (
                checkpoint_weight.shape[0] > target_param.shape[0]
                and checkpoint_weight.shape[1] == target_param.shape[1]
            ):
                state_dict[state_key] = checkpoint_weight[: target_param.shape[0]]

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale.to(
            dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(
            dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma3nTextOutput:
        del kwargs
        assert position_ids is not None, "position_ids must be provided"
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            per_layer_inputs = self.get_per_layer_inputs(input_ids)

        assert inputs_embeds is not None
        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        position_embeddings_global = _slice_rope_cache(
            self.rotary_emb(inputs_embeds, position_ids), position_ids
        )
        position_embeddings_local = _slice_rope_cache(
            self.rotary_emb_local(inputs_embeds, position_ids), position_ids
        )

        target_magnitude = torch.mean(inputs_embeds**2, dim=-1, keepdim=True) ** 0.5
        hidden_states = [inputs_embeds]
        for projection in self.altup_projections:
            current_hidden_state = projection(inputs_embeds).to(dtype=inputs_embeds.dtype)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(
                torch.maximum(
                    new_magnitude,
                    self._ad_eps.to(device=inputs_embeds.device, dtype=new_magnitude.dtype),
                )
            )
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            hidden_states.append(current_hidden_state)
        hidden_states = torch.stack(hidden_states, dim=0)

        for decoder_layer in self.layers:
            layer_per_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings_global,
                position_embeddings_local,
                layer_per_input,
            )

        target_magnitude = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        reduced_hidden_states = [hidden_states[0]]
        for i, projection in enumerate(self.altup_unembed_projections, start=1):
            current_hidden_state = projection(hidden_states[i]).to(dtype=inputs_embeds.dtype)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(
                torch.maximum(
                    new_magnitude,
                    self._ad_eps.to(device=inputs_embeds.device, dtype=new_magnitude.dtype),
                )
            )
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            reduced_hidden_states.append(current_hidden_state)

        hidden_states = torch.mean(torch.stack(reduced_hidden_states), dim=0)
        hidden_states = self.norm(hidden_states)
        return Gemma3nTextOutput(last_hidden_state=hidden_states)


class Gemma3nForCausalLM(Gemma3nTextPreTrainedModel, GenerationMixin):
    config_class = Gemma3nTextConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3nTextConfig, **kwargs):
        del kwargs
        super().__init__(config)
        self.model = Gemma3nTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma3nCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return Gemma3nCausalLMOutput(logits=logits)


class Gemma3nModel(Gemma3nPreTrainedModel):
    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input
        self.vision_tower = nn.Module()
        self.language_model = Gemma3nTextModel(config.text_config)
        self.audio_tower = nn.Module()
        self.embed_vision = Gemma3nMultimodalEmbedder(config.vision_config, config.text_config)
        self.embed_audio = Gemma3nMultimodalEmbedder(config.audio_config, config.text_config)
        self._register_load_state_dict_pre_hook(self._drop_unsupported_multimodal_tower_weights)
        self.post_init()

    @staticmethod
    def _drop_unsupported_multimodal_tower_weights(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        unsupported_prefixes = (
            prefix + "vision_tower.",
            prefix + "audio_tower.",
        )
        for key in list(state_dict):
            if key.startswith(unsupported_prefixes):
                state_dict.pop(key)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma3nTextOutput:
        del kwargs
        del input_features_mask
        assert position_ids is not None, "position_ids must be provided"
        if pixel_values is not None or input_features is not None:
            raise NotImplementedError(
                "Gemma3n multimodal inputs are not supported by the current AutoDeploy export path. "
                "Use text-only prompts for this onboarding."
            )
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        per_layer_inputs = None
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            per_layer_inputs_mask = torch.logical_and(
                input_ids >= 0, input_ids < self.vocab_size_per_layer_input
            )
            per_layer_inputs_tokens = torch.where(
                per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
            )
            per_layer_inputs = self.language_model.get_per_layer_inputs(per_layer_inputs_tokens)

        return self.language_model(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
        )


class Gemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    config_class = Gemma3nConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3nConfig, **kwargs):
        del kwargs
        super().__init__(config)
        self.model = Gemma3nModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model.get_decoder()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma3nConditionalOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            input_features=input_features,
            input_features_mask=input_features_mask,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping
        return Gemma3nConditionalOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("Gemma3nTextConfig", Gemma3nForCausalLM)
AutoModelForCausalLMFactory.register_custom_model_cls(
    "Gemma3nConfig", Gemma3nForConditionalGeneration
)
