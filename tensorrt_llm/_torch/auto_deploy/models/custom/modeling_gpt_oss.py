# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Slimmed down PyTorch GPT-OSS model implementation for auto_deploy export.

Source:
https://huggingface.co/openai/gpt-oss-20b

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Uses torch_attention with native sink support instead of HF eager attention
* Uses torch_moe_router for MoE routing
* Expert computation uses vectorized BMM for export-friendliness
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)

The GPT-OSS model is a Mixture-of-Experts model with:
- GQA attention (64 Q heads, 8 KV heads) with learnable attention sinks
- YaRN RoPE with non-interleaved (half/half) rotation
- MoE with custom clamped SwiGLU activation and biased expert projections
- Alternating sliding-window and full-attention layers
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class GptOssRMSNorm(nn.Module):
    """RMS Normalization using AutoDeploy torch_rmsnorm reference op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class GptOssYarnRotaryEmbedding(nn.Module):
    """YaRN-extended Rotary Position Embedding for GPT-OSS.

    Precomputes and caches cos/sin values with YaRN frequency interpolation.
    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    Forward returns pre-sliced (cos, sin) indexed by position_ids.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        dim = config.head_dim
        base = config.rope_theta
        max_pos = config.max_position_embeddings
        scaling = config.rope_scaling

        factor = scaling["factor"]
        original_max_pos = scaling["original_max_position_embeddings"]
        beta_fast = scaling["beta_fast"]
        beta_slow = scaling["beta_slow"]

        # Compute YaRN-adjusted inv_freq
        freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = freq_extra / factor

        low, high = self._find_correction_range(beta_fast, beta_slow, dim, base, original_max_pos)
        mask = 1.0 - self._linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - mask) + freq_extra * mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Attention scaling (standard YaRN mscale)
        if factor > 1:
            attention_scaling = 0.1 * math.log(factor) + 1.0
        else:
            attention_scaling = 1.0

        # Build cos/sin cache in cat(freqs, freqs) format for torch_rope compatibility
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, head_dim]
        self.register_buffer("_ad_cos_cached", emb.cos() * attention_scaling, persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin() * attention_scaling, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached[position_ids].to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached[position_ids].to(dtype=x.dtype, device=x.device)
        return cos, sin

    @staticmethod
    def _find_correction_dim(
        num_rotations: float, dim: int, base: float, max_position_embeddings: int
    ) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    @classmethod
    def _find_correction_range(
        cls, beta_fast: float, beta_slow: float, dim: int, base: float, max_pos: int
    ) -> Tuple[int, int]:
        low = math.floor(cls._find_correction_dim(beta_fast, dim, base, max_pos))
        high = math.ceil(cls._find_correction_dim(beta_slow, dim, base, max_pos))
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _linear_ramp_mask(low: float, high: float, dim: int) -> torch.Tensor:
        if low == high:
            high += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(linear_func, 0, 1)


class GptOssExperts(nn.Module):
    """Stacked expert parameters with vectorized BMM for export-friendly inference.

    NOTE on checkpoint compatibility: Uses stacked nn.Parameter tensors (not nn.ModuleList)
    to directly match the HF checkpoint layout which stores expert weights as fused tensors:
    - gate_up_proj: [num_experts, hidden_size, 2 * expert_dim]
    - gate_up_proj_bias: [num_experts, 2 * expert_dim]
    - down_proj: [num_experts, expert_dim, hidden_size]
    - down_proj_bias: [num_experts, hidden_size]
    This avoids the need for weight conversion hooks at load time.

    NOTE on canonical ops: torch_moe / torch_moe_fused cannot be used here because:
    1. GPT-OSS experts have per-expert biases on both gate_up_proj and down_proj,
       which torch_moe does not support (it only accepts weight lists, no bias).
    2. The activation is a custom clamped SwiGLU variant (gate*sigmoid(gate*alpha)
       with clamping and (up+1) scaling) that is not in the ActivationType enum
       supported by torch_moe (only Silu, Swiglu, Relu2 are supported).
    3. The gate/up projections use an interleaved layout (gate_up[::2], gate_up[1::2])
       rather than the split gate/up format expected by torch_moe.
    Plain vectorized BMM is used instead for export-friendly computation.

    The activation is a clamped SwiGLU variant:
        gate = clamp(gate, max=limit)
        up = clamp(up, -limit, limit)
        output = (up + 1) * gate * sigmoid(gate * alpha)
    where gate and up are interleaved in gate_up_proj output.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.expert_dim = config.intermediate_size
        self.alpha = 1.702
        self.limit = config.swiglu_limit

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size)
        )
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

    def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        """Vectorized expert computation over all experts.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            routing_weights: [batch_size * seq_len, num_experts] sparse routing scores

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size = hidden_states.shape[0]
        flat = hidden_states.reshape(-1, self.hidden_size)  # [T, H]

        # Expand for all experts: [E, T, H]
        hs = flat.unsqueeze(0).expand(self.num_experts, -1, -1)

        # gate_up: [E, T, 2*D]
        gate_up = torch.bmm(hs, self.gate_up_proj) + self.gate_up_proj_bias.unsqueeze(1)

        # Interleaved split (gate at even indices, up at odd indices)
        gate = gate_up[..., ::2]  # [E, T, D]
        up = gate_up[..., 1::2]  # [E, T, D]

        # Clamped SwiGLU activation
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu

        # down_proj: [E, T, H]
        out = torch.bmm(gated_output, self.down_proj) + self.down_proj_bias.unsqueeze(1)

        # Weight by routing scores and sum over experts
        # routing_weights: [T, E] -> [E, T, 1]
        weights = routing_weights.T.unsqueeze(-1)
        result = (out * weights).sum(dim=0)  # [T, H]

        return result.view(batch_size, -1, self.hidden_size)


class GptOssTopKRouter(nn.Module):
    """MoE top-k router using torch_moe_router canonical op."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns sparse routing scores [B*S, num_experts]."""
        return torch.ops.auto_deploy.torch_moe_router(
            hidden_states, self.weight, self.bias, self.top_k
        )


class GptOssMLP(nn.Module):
    """MoE MLP: router + expert computation."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_scores = self.router(hidden_states)  # [B*S, E]
        return self.experts(hidden_states, routing_weights=router_scores)


class GptOssAttention(nn.Module):
    """GQA attention with learnable sinks using torch_attention canonical op.

    The sinks are learnable per-head bias terms added to the softmax denominator,
    creating a virtual "sink" position that absorbs excess attention probability.
    """

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # Learnable attention sinks (per head)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))

        # Sliding window for sliding_attention layers, None for full_attention
        self.sliding_window = (
            config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q/K/V and reshape to [B, S, N, head_dim] (BSND layout)
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Get pre-sliced cos/sin from position_embeddings
        cos, sin = position_embeddings  # [B, S, head_dim]

        # Apply RoPE (BSND layout, unsqueeze_dim=2)
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 2)

        # Attention with sinks and optional sliding window (BSND layout)
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,
            self.sinks,
            self.sliding_window,
            None,  # logit_cap
            "bsnd",
        )

        # Reshape and project output
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class GptOssDecoderLayer(nn.Module):
    """Transformer decoder layer for GPT-OSS."""

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config, layer_idx=layer_idx)
        self.mlp = GptOssMLP(config)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        # MLP (MoE)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class GptOssOutput(ModelOutput):
    """Output for GptOssModel."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class GptOssCausalLMOutput(ModelOutput):
    """Output for GptOssForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class GptOssPreTrainedModel(PreTrainedModel):
    """Base class for GPT-OSS models."""

    config_class = GptOssConfig
    base_model_prefix = "model"
    _no_split_modules = ["GptOssDecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GptOssRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, GptOssExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.gate_up_proj_bias.data.zero_()
            module.down_proj.data.normal_(mean=0.0, std=std)
            module.down_proj_bias.data.zero_()
        elif isinstance(module, GptOssAttention):
            module.sinks.data.normal_(mean=0.0, std=std)
        elif isinstance(module, GptOssTopKRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.bias.data.normal_(mean=0.0, std=std)


class GptOssModel(GptOssPreTrainedModel):
    """GPT-OSS transformer decoder model."""

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        self.rotary_emb = GptOssYarnRotaryEmbedding(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> GptOssOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Cast to compute dtype for FP8/quantized models
        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        # Compute position embeddings once (pre-sliced by position_ids)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return GptOssOutput(last_hidden_state=hidden_states)


def _mxfp4_dequant_load_hook(state_dict, prefix, *args, **kwargs):
    """Dequantize MXFP4 expert weights (blocks + scales) to bfloat16 during loading.

    GPT-OSS checkpoints store expert MLP weights in MXFP4 quantized format:
    - *_blocks (uint8): packed FP4 E2M1 values (2 per byte)
    - *_scales (uint8): per-block E8M0 shared exponents

    This hook converts them to standard bfloat16 tensors before load_state_dict.
    """
    try:
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    except ImportError:
        return

    keys_to_process = []
    for key in list(state_dict.keys()):
        if key.endswith("_blocks"):
            base = key[: -len("_blocks")]
            scales_key = base + "_scales"
            if scales_key in state_dict:
                keys_to_process.append((base, key, scales_key))

    for base, blocks_key, scales_key in keys_to_process:
        blocks = state_dict.pop(blocks_key).cpu()
        scales = state_dict.pop(scales_key).cpu()
        dequantized = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)
        state_dict[base] = dequantized


class GptOssForCausalLM(GptOssPreTrainedModel, GenerationMixin):
    """GPT-OSS model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Dequantize MXFP4 expert weights during checkpoint loading.
        # The GPT-OSS checkpoint stores expert weights in MXFP4 quantized format
        # (*_blocks uint8 + *_scales uint8). This hook converts them to bfloat16.
        self._register_load_state_dict_pre_hook(_mxfp4_dequant_load_hook)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> GptOssCausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return GptOssCausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("GptOssConfig", GptOssForCausalLM)
