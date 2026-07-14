# Copyright 2018 The HuggingFace Team
# Licensed under the Apache License, Version 2.0.
# Original source: https://github.com/huggingface/transformers
#
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

"""Prefill-only OpenELM model for auto_deploy export (sharding IR).

Source: https://huggingface.co/apple/OpenELM-270M-Instruct

OpenELM is a heterogeneous transformer with per-layer varying:
- Number of query/KV heads (GQA)
- FFN intermediate size (via ffn_multipliers)
- Fused QKV projection
- Q/K normalization before RoPE
- Shared input/output embeddings (no separate lm_head)

The config is bundled locally as ``OpenELMConfig`` (registered with ``AutoConfig``)
rather than loaded from Apple's hub remote code. Apple's ``configuration_openelm.py``
was written for transformers 4.x and its private ``__post_init__(self)`` collides
with the transformers 5.x strict-dataclass ``PreTrainedConfig`` (which forwards
unrecognized kwargs such as ``use_cache`` to ``self.__post_init__(**kwargs)``),
raising ``TypeError`` before any model code runs. Vendoring the config locally
mirrors the pattern used by the other AutoDeploy custom models (e.g. EXAONE) and
removes Apple's frozen remote code from the execution path entirely.

Uses AutoDeploy sharding-IR hint ops (``torch_linear_simple`` / ``view`` /
``split_with_sizes`` / ``all_reduce``) so the exported FX graph fully specifies
tensor-parallel sharding for ``apply_sharding_hints`` (no legacy
``detect_sharding`` heuristics needed).
"""

from dataclasses import dataclass
from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ..hf import AutoModelForCausalLMFactory

# =============================================================================
# Helpers
# =============================================================================


def _make_divisible(v, divisor=8, min_value=None):
    """Ensure value is divisible by divisor (from HF OpenELM config)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _compute_heads(model_dim: int, head_dim: int) -> int:
    """Number of heads given model/head dim (from HF OpenELM config)."""
    if model_dim % head_dim != 0:
        raise ValueError(
            f"Model dimension should be divisible by head dimension. "
            f"Got: {model_dim} and {head_dim}."
        )
    return model_dim // head_dim


# =============================================================================
# Config (vendored locally; see module docstring)
# =============================================================================


class OpenELMConfig(PretrainedConfig):
    """OpenELM configuration, vendored from Apple's ``configuration_openelm.py``.

    Bundled with the custom model implementation so AutoDeploy never executes
    Apple's hub remote code (which is incompatible with transformers 5.x). The
    per-layer derivation that Apple performed in ``__post_init__`` is inlined
    into ``__init__`` here to avoid the transformers 5.x dataclass collision on
    the reserved ``__post_init__`` name.

    Derivation logic is adapted from Apple's OpenELM config
    (Copyright (C) 2024 Apple Inc.; https://huggingface.co/apple/OpenELM-270M-Instruct).
    """

    model_type = "openelm"

    def __init__(
        self,
        vocab_size: int = 32000,
        max_context_length: int = 2048,
        num_transformer_layers: int = 12,
        model_dim: int = 2048,
        head_dim: int = 128,
        qkv_multipliers: Union[Number, List[Number]] = 1.0,
        num_query_heads: Union[int, None] = None,
        num_gqa_groups: int = 1,
        ffn_multipliers: Union[Number, List[Number]] = 4.0,
        ffn_with_glu: bool = True,
        ffn_dim_divisor: int = 256,
        activation_fn_name: str = "swish",
        normalization_layer_name: str = "rms_norm",
        normalize_qk_projections: bool = False,
        share_input_output_layers: bool = False,
        rope_freq_constant: int = 10000,
        rope_max_length: int = 4096,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.num_transformer_layers = num_transformer_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.qkv_multipliers = qkv_multipliers
        self.num_gqa_groups = num_gqa_groups
        self.ffn_multipliers = ffn_multipliers
        self.ffn_with_glu = ffn_with_glu
        self.ffn_dim_divisor = ffn_dim_divisor
        self.activation_fn_name = activation_fn_name
        self.normalization_layer_name = normalization_layer_name
        self.normalize_qk_projections = normalize_qk_projections
        self.share_input_output_layers = share_input_output_layers
        self.rope_freq_constant = rope_freq_constant
        self.rope_max_length = rope_max_length
        self.initializer_range = initializer_range
        # NOTE: the `num_query_heads` parameter stays accepted for config.json schema
        # fidelity (published checkpoints carry the precomputed list), but the
        # per-layer derivation below is the source of truth, same as Apple's original.

        # --- per-layer derivation (inlined from Apple's __post_init__) ---
        head_multiple_of = self.num_gqa_groups if self.num_gqa_groups is not None else 2

        if isinstance(self.qkv_multipliers, Number):
            qkv_dim = _make_divisible(
                self.model_dim * self.qkv_multipliers,
                divisor=self.head_dim * head_multiple_of,
            )
            query_dims = [int(qkv_dim)] * self.num_transformer_layers
        elif isinstance(self.qkv_multipliers, (tuple, list)) and len(self.qkv_multipliers) == 2:
            qkv_multipliers = [
                round(v, 2)
                for v in np.linspace(
                    self.qkv_multipliers[0],
                    self.qkv_multipliers[1],
                    num=self.num_transformer_layers,
                    dtype=float,
                )
            ]
            query_dims = [
                int(_make_divisible(self.model_dim * m, divisor=self.head_dim * head_multiple_of))
                for m in qkv_multipliers
            ]
        else:
            raise NotImplementedError(
                f"QKV multipliers should be a single number or a list of exactly two numbers. "
                f"Got: {self.qkv_multipliers}."
            )

        self.num_query_heads = [int(_compute_heads(q_dim, self.head_dim)) for q_dim in query_dims]
        self.num_kv_heads = [q // self.num_gqa_groups for q in self.num_query_heads]

        if isinstance(self.ffn_multipliers, Number):
            self.ffn_multipliers = [self.ffn_multipliers] * self.num_transformer_layers
        elif isinstance(self.ffn_multipliers, (tuple, list)):
            if len(self.ffn_multipliers) == 2:
                self.ffn_multipliers = [
                    round(v, 2)
                    for v in np.linspace(
                        self.ffn_multipliers[0],
                        self.ffn_multipliers[1],
                        num=self.num_transformer_layers,
                        dtype=float,
                    )
                ]
            else:
                assert len(self.ffn_multipliers) == self.num_transformer_layers, (
                    f"{len(self.ffn_multipliers)=}!={self.num_transformer_layers=}"
                )
        else:
            raise NotImplementedError(
                f"FFN multipliers should be a single number or a list of exactly two numbers. "
                f"Got: {self.ffn_multipliers}."
            )

        for layer_idx in range(len(query_dims)):
            assert self.num_query_heads[layer_idx] % self.num_kv_heads[layer_idx] == 0

        super().__init__(
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


# =============================================================================
# RMSNorm (canonical AD op)
# =============================================================================


class OpenELMRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


# =============================================================================
# Rotary Embedding
# =============================================================================


class OpenELMRotaryEmbedding(nn.Module):
    """Shared rotary embedding with _ad_ prefixed buffers.

    Returns cos/sin indexed by position_ids: [B, S, head_dim].
    """

    def __init__(self, head_dim: int, max_seq_length: int, freq_constant: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            freq_constant ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_length)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin indexed by position_ids: [B, S, head_dim]."""
        cos = self._ad_cos_cached.to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached.to(dtype=x.dtype, device=x.device)
        return cos[position_ids], sin[position_ids]


# =============================================================================
# FFN
# =============================================================================


class OpenELMFeedForwardNetwork(nn.Module):
    """GLU-style FFN with fused gate+up projection (proj_1) and down projection (proj_2).

    Sharding strategy:
      proj_1 (fused gate|up) -> colwise (output_sizes=[inter, inter] for the GLU
                                variant so each half shards proportionally)
      split (gate|up)        -> enable_sharding (split sizes scale with TP)
      proj_2 (down)          -> rowwise + all_reduce
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            _make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor)
        )
        self.intermediate_dim = intermediate_dim

        if config.ffn_with_glu:
            self.proj_1 = nn.Linear(config.model_dim, 2 * intermediate_dim, bias=False)
            self.proj_2 = nn.Linear(intermediate_dim, config.model_dim, bias=False)
            self.ffn_with_glu = True
        else:
            self.proj_1 = nn.Linear(config.model_dim, intermediate_dim, bias=False)
            self.proj_2 = nn.Linear(intermediate_dim, config.model_dim, bias=False)
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_with_glu:
            y_12 = torch.ops.auto_deploy.torch_linear_simple(
                x,
                self.proj_1.weight,
                self.proj_1.bias,
                tp_mode="colwise",
                output_sizes=[self.intermediate_dim, self.intermediate_dim],
                layer_type="mlp",
            )
            y_1, y_2 = torch.ops.auto_deploy.split_with_sizes(
                y_12,
                [self.intermediate_dim, self.intermediate_dim],
                dim=-1,
                enable_sharding=True,
                layer_type="mlp",
            )
            out = torch.ops.auto_deploy.torch_linear_simple(
                self.act(y_1) * y_2,
                self.proj_2.weight,
                self.proj_2.bias,
                tp_mode="rowwise",
                layer_type="mlp",
            )
            return torch.ops.auto_deploy.all_reduce(out, layer_type="mlp")
        else:
            y = torch.ops.auto_deploy.torch_linear_simple(
                x,
                self.proj_1.weight,
                self.proj_1.bias,
                tp_mode="colwise",
                layer_type="mlp",
            )
            out = torch.ops.auto_deploy.torch_linear_simple(
                self.act(y),
                self.proj_2.weight,
                self.proj_2.bias,
                tp_mode="rowwise",
                layer_type="mlp",
            )
            return torch.ops.auto_deploy.all_reduce(out, layer_type="mlp")


# =============================================================================
# Attention
# =============================================================================


class OpenELMAttention(nn.Module):
    """GQA attention with fused QKV proj, Q/K norms, canonical AD ops.

    Sharding strategy:
      qkv_proj (fused Q|K|V) -> colwise (output_sizes=[q_dim, k_dim, v_dim],
                                tp_min_local_shape=head_dim for GQA)
      view (head-count dim)  -> tp_scaled_dim=2
      split (Q|K|V)          -> enable_sharding (split sizes scale with TP)
      out_proj               -> rowwise + all_reduce
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_query_heads[layer_idx]
        self.num_k_heads = config.num_kv_heads[layer_idx]
        self.num_v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Linear(
            config.model_dim,
            (self.num_q_heads + self.num_k_heads + self.num_v_heads) * self.head_dim,
            bias=False,
        )

        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(config.head_dim)
            self.k_norm = OpenELMRMSNorm(config.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Linear(self.num_q_heads * self.head_dim, config.model_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        # Fused QKV projection → view → split. ``output_sizes`` shards each of the
        # Q/K/V blocks proportionally under colwise TP; the view's head-count dim and
        # the split sizes both scale with TP so the per-rank shapes stay consistent.
        qkv = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.qkv_proj.weight,
            self.qkv_proj.bias,
            tp_mode="colwise",
            output_sizes=[
                self.num_q_heads * self.head_dim,
                self.num_k_heads * self.head_dim,
                self.num_v_heads * self.head_dim,
            ],
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        qkv = torch.ops.auto_deploy.view(
            qkv,
            [bsz, seq_len, self.num_q_heads + self.num_k_heads + self.num_v_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        queries, keys, values = torch.ops.auto_deploy.split_with_sizes(
            qkv,
            [self.num_q_heads, self.num_k_heads, self.num_v_heads],
            dim=2,
            enable_sharding=True,
            layer_type="mha",
        )

        # Q/K normalization (per-head, operates on last dim = head_dim)
        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)

        # RoPE via canonical AD op (cos/sin already indexed by position_ids)
        cos, sin = position_embeddings  # [B, S, head_dim]
        queries, keys = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            queries,
            keys,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for bsnd layout
        )

        # Attention via canonical AD op (bsnd layout, handles GQA natively)
        attn_output = torch.ops.auto_deploy.torch_attention(
            queries, keys, values, is_causal=True, dropout_p=0.0, layout="bsnd"
        )

        attn_output = torch.ops.auto_deploy.view(
            attn_output,
            [bsz, seq_len, self.num_q_heads * self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        attn_output = torch.ops.auto_deploy.torch_linear_simple(
            attn_output,
            self.out_proj.weight,
            self.out_proj.bias,
            tp_mode="rowwise",
            layer_type="mha",
        )
        return torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mha")


# =============================================================================
# Decoder Layer
# =============================================================================


class OpenELMDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.attn = OpenELMAttention(config, layer_idx)
        self.ffn = OpenELMFeedForwardNetwork(config, layer_idx)
        self.attn_norm = OpenELMRMSNorm(config.model_dim)
        self.ffn_norm = OpenELMRMSNorm(config.model_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Model Outputs
# =============================================================================


@dataclass
class OpenELMModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class OpenELMCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# =============================================================================
# Full Model
# =============================================================================


class OpenELMPreTrainedModel(PreTrainedModel):
    base_model_prefix = "transformer"
    _no_split_modules = ["OpenELMDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class OpenELMModel(OpenELMPreTrainedModel):
    """OpenELM transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = nn.ModuleList(
            [
                OpenELMDecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_transformer_layers)
            ]
        )
        self.norm = OpenELMRMSNorm(config.model_dim)

        # Shared rotary embedding
        self.rotary_emb = OpenELMRotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_length=config.rope_max_length,
            freq_constant=config.rope_freq_constant,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, value):
        self.token_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> OpenELMModelOutput:
        assert position_ids is not None, "position_ids is required"

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return OpenELMModelOutput(last_hidden_state=hidden_states)


class OpenELMForCausalLM(OpenELMPreTrainedModel, GenerationMixin):
    """OpenELM model with language modeling head (shared embeddings)."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = OpenELMModel(config)
        self.vocab_size = config.vocab_size

        # OpenELM shares input/output embeddings; no separate lm_head
        # But we still need the attribute for weight tying
        if not config.share_input_output_layers:
            self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.token_embeddings

    def set_input_embeddings(self, value):
        self.transformer.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.transformer

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> OpenELMCausalLMOutput:
        assert position_ids is not None, "position_ids is required"

        outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.transformer.token_embeddings.weight)
        logits = logits.float()

        return OpenELMCausalLMOutput(logits=logits)


# Register the vendored config with AutoConfig so AutoConfig.from_pretrained resolves
# `model_type: openelm` to this local class instead of executing Apple's hub remote
# code. Because this class's module is not under `transformers.`, transformers treats
# it as `explicit_local_code` and uses it even when trust_remote_code=True and the
# checkpoint's config.json declares an auto_map.
AutoConfig.register("openelm", OpenELMConfig, exist_ok=True)

# Register with AutoModelForCausalLMFactory (keyed by config class name "OpenELMConfig")
AutoModelForCausalLMFactory.register_custom_model_cls("OpenELMConfig", OpenELMForCausalLM)
