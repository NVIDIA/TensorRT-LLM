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

"""Slimmed down PyTorch Llama 4 model implementation for auto_deploy export.

Source:
https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)
* No repeat_kv — AD attention ops handle GQA natively
* Complex RoPE frequencies precomputed once at model level

The Llama 4 family features:
* GQA with complex-frequency RoPE (with llama3-style scaling)
* NoPE layers (interleaved layers that skip RoPE)
* L2 QK normalization on RoPE layers
* Attention temperature tuning on NoPE layers
* MoE layers with sigmoid router + shared expert (interleaved with dense MLP)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama4.configuration_llama4 import Llama4Config, Llama4TextConfig
from transformers.utils import ModelOutput

from ..hf import AutoModelForCausalLMFactory

# =========================================================================
# Normalization
# =========================================================================


class Llama4RMSNorm(nn.Module):
    """RMS Normalization using AutoDeploy torch_rmsnorm reference op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class Llama4L2Norm(nn.Module):
    """L2 Normalization for QK norm (mean-based, matching HF implementation).

    Note: This uses mean-based L2 norm (x * rsqrt(mean(x^2) + eps)), which differs
    from the AD torch_l2norm op (sum-based). Implemented in plain PyTorch.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)


# =========================================================================
# Rotary Position Embedding (Complex Frequency)
# =========================================================================


class Llama4RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Llama 4 using complex frequencies.

    Supports llama3-style rope scaling via transformers ROPE_INIT_FUNCTIONS.
    Precomputes and caches complex freqs_cis values. Slices by position_ids
    once and returns pre-sliced freqs to all layers.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        # transformers >=5.x auto-populates ``rope_parameters`` to
        # ``{"rope_type": "default", ...}`` even when the user passes
        # ``rope_scaling=None``, so always read the type from the dict.
        rope_parameters = config.rope_parameters
        rope_type = rope_parameters["rope_type"]

        if rope_type == "default":
            # ROPE_INIT_FUNCTIONS no longer carries a "default" entry in
            # transformers 5.x; replicate upstream's
            # ``Llama4TextRotaryEmbedding.compute_default_rope_parameters``.
            base = rope_parameters["rope_theta"]
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
            )
            self.attention_scaling = 1.0
        else:
            # Use HF's ROPE_INIT_FUNCTIONS to compute inv_freq with proper scaling
            inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, device=None)

        # Precompute complex frequencies for all positions
        max_pos = config.max_position_embeddings
        t = torch.arange(max_pos, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim//2]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex [max_pos, D//2]
        freqs_cis = freqs_cis * self.attention_scaling
        self.register_buffer("_ad_freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the full precomputed complex freqs_cis table.

        Args:
            x: Input tensor (used only for device).

        Returns:
            freqs_cis: [max_pos, head_dim//2] complex tensor (full table).
        """
        return self._ad_freqs_cis.to(device=x.device)


# =========================================================================
# MLP
# =========================================================================


class Llama4MLP(nn.Module):
    """MLP layer for Llama 4 (SwiGLU activation)."""

    def __init__(self, config: Llama4TextConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =========================================================================
# MoE (Mixture of Experts with sigmoid router + shared expert)
# =========================================================================


class Llama4Experts(nn.Module):
    """Stacked expert weights for Llama 4 MoE.

    Uses nn.Parameter with bmm for parallel expert computation, matching the HF
    checkpoint format directly. The AD MatchBmmMoePattern transform will detect
    this bmm pattern and replace it with torch_moe at deployment time.
    """

    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size)
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(self.gate_up_proj.shape[0], -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        next_states = torch.bmm(up * self.act_fn(gate), self.down_proj)
        next_states = next_states.view(-1, self.hidden_size)
        return next_states


class Llama4Router(nn.Linear):
    """Sigmoid top-k router for Llama 4 MoE.

    Unlike standard softmax routers, Llama 4 uses sigmoid activation on
    scattered top-k logits, producing per-expert routing scores.
    """

    def __init__(self, config: Llama4TextConfig):
        super().__init__(config.hidden_size, config.num_local_experts, bias=False)
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_logits = super().forward(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = torch.full_like(router_logits, float("-inf")).scatter_(
            1, router_indices, router_top_value
        )
        router_scores = torch.nn.functional.sigmoid(router_scores.float()).to(router_scores.dtype)
        return router_scores, router_logits


class Llama4MoE(nn.Module):
    """Mixture of Experts layer for Llama 4.

    Uses stacked expert weights with bmm (matching HF checkpoint format directly).
    The AD MatchBmmMoePattern transform will detect the bmm pattern at deployment
    time and replace it with optimized torch_moe ops.

    Features:
    - Sigmoid-based top-k routing (not softmax)
    - Shared expert (dense MLP added to routed output)
    - Routing weight scales input before expert MLP
    """

    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.experts = Llama4Experts(config)
        self.router = Llama4Router(config)
        self.shared_expert = Llama4MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_scores, _ = self.router(hidden_states)
        routed_in = hidden_states.repeat(router_scores.shape[1], 1)
        routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)
        out = out + routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim=0)
        return out.view(*orig_shape)


# =========================================================================
# Attention
# =========================================================================


class Llama4Attention(nn.Module):
    """Grouped Query Attention for Llama 4.

    Uses AD canonical ops for attention and complex-frequency RoPE.
    GQA is handled natively by torch_attention — no repeat_kv needed.

    Features:
    - NoPE layers: skip RoPE application based on config
    - L2 QK norm on RoPE layers (when use_qk_norm=True)
    - Attention temperature tuning on NoPE layers
    """

    def __init__(self, config: Llama4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** (-0.5)

        # Whether this layer uses RoPE (1=yes, 0=NoPE)
        self.use_rope = bool(config.no_rope_layers[layer_idx])

        # Temperature tuning for NoPE layers
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # L2 QK norm (only on RoPE layers when use_qk_norm is set)
        if config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4L2Norm(config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q/K/V — use BSND layout throughout
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE (only on RoPE layers)
        if self.use_rope:
            # Slice the full freqs_cis table by position_ids
            sliced_freqs = freqs_cis[position_ids]  # [B, S, head_dim//2]
            q, k = torch.ops.auto_deploy.torch_rope_with_complex_freqs(
                q,
                k,
                sliced_freqs,
                2,  # unsqueeze_dim=2 for BSND layout
            )

        # Apply L2 QK norm (only on RoPE layers with qk_norm)
        if hasattr(self, "qk_norm"):
            q = self.qk_norm(q)
            k = self.qk_norm(k)

        # Apply temperature tuning (only on NoPE layers)
        if self.attn_temperature_tuning and not self.use_rope:
            # position_ids: [B, S]
            attn_scales = (
                torch.log1p(torch.floor((position_ids.float() + 1.0) / self.floor_scale))
                * self.attn_scale
                + 1.0
            )
            # attn_scales: [B, S] → [B, S, 1, 1] for broadcasting with [B, S, N, D]
            attn_scales = attn_scales.unsqueeze(-1).unsqueeze(-1)
            q = (q * attn_scales).to(q.dtype)

        # Attention using custom op with GQA support (BSND layout)
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,  # [B, S, N, head_dim]
            k,  # [B, S, N_kv, head_dim]
            v,  # [B, S, N_kv, head_dim]
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,  # scale
            None,  # sinks
            None,  # sliding_window
            None,  # logit_cap
            "bsnd",  # layout
        )

        # Reshape [B, S, N, head_dim] -> [B, S, N * head_dim] and project
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


# =========================================================================
# Decoder Layer
# =========================================================================


class Llama4DecoderLayer(nn.Module):
    """Transformer decoder layer for Llama 4.

    Heterogeneous layers: some use MoE, others use dense MLP.
    """

    def __init__(self, config: Llama4TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = Llama4Attention(config, layer_idx=layer_idx)

        # Feed-forward: MoE or dense MLP
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:
            self.feed_forward = Llama4MoE(config)
        else:
            self.feed_forward = Llama4MLP(config, intermediate_size=config.intermediate_size_mlp)

        # Layer norms
        self.input_layernorm = Llama4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Llama4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, freqs_cis)
        hidden_states = residual + hidden_states

        # Feed-forward (MoE or dense MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =========================================================================
# Output dataclasses
# =========================================================================


@dataclass
class Llama4TextOutput(ModelOutput):
    """Output for Llama4TextModel."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Llama4CausalLMOutput(ModelOutput):
    """Output for Llama4ForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


@dataclass
class Llama4ConditionalOutput(ModelOutput):
    """Output for Llama4ForConditionalGeneration."""

    logits: Optional[torch.FloatTensor] = None


# =========================================================================
# Base model
# =========================================================================


class Llama4PreTrainedModel(PreTrainedModel):
    """Base class for Llama 4 models."""

    config_class = Llama4Config
    base_model_prefix = "model"
    _no_split_modules = ["Llama4DecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Llama4Experts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)


# =========================================================================
# Text Model
# =========================================================================


class Llama4TextModel(Llama4PreTrainedModel):
    """Llama 4 text transformer decoder model."""

    config_class = Llama4TextConfig
    base_model_prefix = "model"

    def __init__(self, config: Llama4TextConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Llama4DecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = Llama4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        self.rotary_emb = Llama4RotaryEmbedding(config)

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
    ) -> Llama4TextOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute complex RoPE frequencies once (full table, sliced in attention)
        freqs_cis = self.rotary_emb(inputs_embeds)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids, freqs_cis)

        hidden_states = self.norm(hidden_states)

        return Llama4TextOutput(last_hidden_state=hidden_states)


# =========================================================================
# Causal LM (text-only)
# =========================================================================


class Llama4ForCausalLM(Llama4PreTrainedModel, GenerationMixin):
    """Llama 4 text model with language modeling head."""

    config_class = Llama4TextConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Llama4TextConfig, **kwargs):
        super().__init__(config)
        self.model = Llama4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
    ) -> Llama4CausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return Llama4CausalLMOutput(logits=logits)


# =========================================================================
# Multimodal Wrapper (ForConditionalGeneration)
# =========================================================================


class Llama4MultiModalProjector(nn.Module):
    """Projects vision features to text embedding space."""

    def __init__(self, config: Llama4Config):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=False,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear_1(image_features)


class Llama4ForConditionalGeneration(Llama4PreTrainedModel, GenerationMixin):
    """Multimodal conditional generation model wrapping text + vision.

    The vision tower stays in eager PyTorch — only the text path is exported.
    The weight layout matches the HF checkpoint structure:
      vision_model.*, multi_modal_projector.*, language_model.*
    """

    config_class = Llama4Config
    _tied_weights_keys = ["language_model.lm_head.weight"]
    base_model_prefix = ""

    def __init__(self, config: Llama4Config, **kwargs):
        super().__init__(config)
        # Import HF's vision model for weight loading compatibility
        from transformers.models.llama4.modeling_llama4 import (
            Llama4VisionModel as HFLlama4VisionModel,
        )

        self.vision_model = HFLlama4VisionModel(config.vision_config)
        self.multi_modal_projector = Llama4MultiModalProjector(config)
        self.language_model = Llama4ForCausalLM(config.text_config)
        self.vocab_size = config.text_config.vocab_size

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

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
    ) -> Llama4ConditionalOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return Llama4ConditionalOutput(logits=outputs.logits)


# =========================================================================
# Registration
# =========================================================================

AutoModelForCausalLMFactory.register_custom_model_cls("Llama4TextConfig", Llama4ForCausalLM)
AutoModelForCausalLMFactory.register_custom_model_cls(
    "Llama4Config", Llama4ForConditionalGeneration
)
