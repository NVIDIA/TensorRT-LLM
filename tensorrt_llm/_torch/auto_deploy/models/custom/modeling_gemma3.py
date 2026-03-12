# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill-only Gemma 3 model for AutoDeploy export.

Source:
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gemma3/modeling_gemma3.py

Differences from the Hugging Face implementation:
* KV cache, attention-mask plumbing, and training-only code paths are removed.
* The vision tower and projector are kept only for checkpoint compatibility; the forward/export path
  remains text-only.
* RoPE tables are precomputed into `_ad_` buffers and sliced by `position_ids` downstream.
* Attention uses the AutoDeploy `torch_attention` reference op with native sliding-window support.

The outer multimodal wrapper mirrors the checkpoint hierarchy required by
`google/gemma-3-27b-it`:
  language_model.model.*
  language_model.lm_head.*
"""

import copy
import inspect
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoModel, Gemma3Config, Gemma3TextConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import (
    AutoModelForCausalLMFactory,
    AutoModelForImageTextToTextFactory,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half for RoPE."""
    half_dim = x.shape[-1] // 2
    return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key states."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """Embedding layer with Gemma's sqrt(hidden_size) output scaling."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3MLP(nn.Module):
    """Gemma 3 feed-forward block."""

    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Gemma3RMSNorm(nn.Module):
    """Gemma 3 RMSNorm variant with `(1 + weight)` scaling."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + self.eps)
        normalized = normalized * (1.0 + self.weight.float())
        return normalized.to(hidden_states.dtype)


class Gemma3RotaryEmbedding(nn.Module):
    """RoPE table builder that returns full cached tables for downstream slicing."""

    def __init__(
        self,
        config: Gemma3TextConfig,
        rope_theta: Optional[float] = None,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()

        rope_config = copy.deepcopy(config)
        if rope_theta is not None:
            rope_config.rope_theta = rope_theta
        if rope_scaling is not None:
            rope_config.rope_scaling = rope_scaling

        if isinstance(rope_config.rope_scaling, dict):
            rope_type = rope_config.rope_scaling.get(
                "rope_type", rope_config.rope_scaling.get("type", "default")
            )
        else:
            rope_type = "default"

        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = rope_init_fn(rope_config, device=None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer(
            "_ad_cos_cached",
            self._build_cache(
                inv_freq=inv_freq,
                max_position_embeddings=rope_config.max_position_embeddings,
                attention_scaling=attention_scaling,
                fn=torch.cos,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_ad_sin_cached",
            self._build_cache(
                inv_freq=inv_freq,
                max_position_embeddings=rope_config.max_position_embeddings,
                attention_scaling=attention_scaling,
                fn=torch.sin,
            ),
            persistent=False,
        )

    @staticmethod
    def _build_cache(
        inv_freq: torch.Tensor,
        max_position_embeddings: int,
        attention_scaling: float,
        fn,
    ) -> torch.Tensor:
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        return fn(emb) * attention_scaling

    def forward(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._ad_cos_cached.to(dtype=dtype), self._ad_sin_cached.to(dtype=dtype)


def _gather_position_embeddings(
    cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather per-token RoPE slices from full cached tables."""
    flat_position_ids = position_ids.reshape(-1)
    cos = cos.index_select(0, flat_position_ids).view(*position_ids.shape, -1)
    sin = sin.index_select(0, flat_position_ids).view(*position_ids.shape, -1)
    return cos, sin


class Gemma3Attention(nn.Module):
    """Grouped-query attention with optional sliding window."""

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.is_causal = not config.use_bidirectional_attention

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = _gather_position_embeddings(*position_embeddings, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos=cos, sin=sin, unsqueeze_dim=2
        )

        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=self.is_causal,
            scale=self.scaling,
            sliding_window=self.sliding_window,
            logit_cap=self.attn_logit_softcapping,
            layout="bsnd",
        )
        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        return self.o_proj(attn_output)


class Gemma3DecoderLayer(nn.Module):
    """Gemma 3 decoder block."""

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
        position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = (
            position_embeddings_local
            if self.attention_type == "sliding_attention"
            else position_embeddings_global
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


@dataclass
class Gemma3ModelOutput(ModelOutput):
    """Minimal text-model output for prefill export."""

    last_hidden_state: torch.Tensor


@dataclass
class Gemma3CausalLMOutput(ModelOutput):
    """Minimal CausalLM output for prefill export."""

    logits: torch.Tensor


class Gemma3MultiModalProjector(nn.Module):
    """Gemma 3 image projector copied for checkpoint-compatible parameter layout."""

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )
        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_length = vision_outputs.shape
        reshaped = vision_outputs.transpose(1, 2).reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        pooled = self.avg_pool(reshaped.contiguous()).flatten(2).transpose(1, 2)
        normed = self.mm_soft_emb_norm(pooled)
        projected = torch.matmul(normed, self.mm_input_projection_weight)
        return projected.type_as(vision_outputs)


class Gemma3PreTrainedModel(PreTrainedModel):
    """Base class for the custom Gemma 3 implementations."""

    config_class = Gemma3TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Gemma3DecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Gemma3RMSNorm):
            module.weight.data.zero_()


class Gemma3TextModel(Gemma3PreTrainedModel):
    """Text-only Gemma 3 decoder backbone."""

    config: Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=math.sqrt(config.hidden_size),
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.rotary_emb_local = Gemma3RotaryEmbedding(
            config=config,
            rope_theta=config.rope_local_base_freq,
            rope_scaling={"rope_type": "default"},
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Gemma3ModelOutput:
        del kwargs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=inputs_embeds.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        position_embeddings_global = self.rotary_emb(dtype=inputs_embeds.dtype)
        position_embeddings_local = self.rotary_emb_local(dtype=inputs_embeds.dtype)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
            )

        hidden_states = self.norm(hidden_states)
        return Gemma3ModelOutput(last_hidden_state=hidden_states)


class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    """Text-only Gemma 3 language model."""

    config: Gemma3TextConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3TextConfig, **kwargs):
        super().__init__(config)
        del kwargs
        self.model = Gemma3TextModel(config)
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
    ) -> Gemma3CausalLMOutput:
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
        return Gemma3CausalLMOutput(logits=logits.float())


class Gemma3ForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Gemma 3 multimodal wrapper exporting the text path only."""

    config_class = Gemma3Config
    base_model_prefix = ""
    _tied_weights_keys = ["language_model.lm_head.weight"]
    _no_split_modules = ["Gemma3DecoderLayer"]
    supports_gradient_checkpointing = False

    def __init__(self, config: Gemma3Config, **kwargs):
        super().__init__(config)
        del kwargs
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.language_model = Gemma3ForCausalLM(config.text_config)
        self.post_init()

    def _init_weights(self, module):
        std = getattr(self.config.text_config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Gemma3RMSNorm):
            module.weight.data.zero_()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Gemma3CausalLMOutput:
        language_model_kwargs = {}
        if input_ids is not None:
            language_model_kwargs["input_ids"] = input_ids
        if position_ids is not None:
            language_model_kwargs["position_ids"] = position_ids
        if inputs_embeds is not None:
            language_model_kwargs["inputs_embeds"] = inputs_embeds

        language_model_signature = inspect.signature(self.language_model.forward)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in language_model_signature.parameters.values()
        )
        if not accepts_var_kwargs:
            allowed_extra_kwargs = set(language_model_signature.parameters) - set(
                language_model_kwargs
            )
            language_model_kwargs.update(
                {key: value for key, value in kwargs.items() if key in allowed_extra_kwargs}
            )

        return self.language_model(**language_model_kwargs)


AutoModelForCausalLMFactory.register_custom_model_cls("Gemma3TextConfig", Gemma3ForCausalLM)
AutoModelForCausalLMFactory.register_custom_model_cls(
    "Gemma3Config", Gemma3ForConditionalGeneration
)
AutoModelForImageTextToTextFactory.register_custom_model_cls(
    "Gemma3Config", Gemma3ForConditionalGeneration
)
