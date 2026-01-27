# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Eagle3 model implementation for AutoDeploy.

Eagle3 is a speculative decoding draft model that predicts next tokens based on
hidden states from a target model (e.g., Llama-3.1-8B-Instruct).

This implementation:
- Defines Eagle3LlamaConfig with model_type="Eagle3LlamaForCausalLM"
- Wraps Eagle3Model in Eagle3DrafterForCausalLM (HuggingFace-compatible interface)
- Registers with AutoDeploy's custom model mechanism via AutoModelForCausalLMFactory

Note: Eagle3 uses the same tokenizer as its target model (e.g., Llama), so when
using this model, you must explicitly specify the tokenizer path pointing to the
target model.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, dim, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = getattr(config, "rope_theta", 10000.0)
        self.config = config

        self.factor = 2

        max_position_embeddings = self.config.max_position_embeddings

        if (
            not hasattr(config, "rope_type")
            or config.rope_type is None
            or config.rope_type == "default"
        ):
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )
            self.max_seq_len_cached = max_position_embeddings

        elif config.rope_type == "ntk":
            assert self.config.orig_max_position_embeddings is not None
            orig_max_position_embeddings = self.config.orig_max_position_embeddings

            self.base = self.base * (
                (self.factor * max_position_embeddings / orig_max_position_embeddings)
                - (self.factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )

            self.max_seq_len_cached = orig_max_position_embeddings
        else:
            raise ValueError(f"Not support rope_type: {config.rope_type}")

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    else:
        q_embed = None

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


class EagleRMSNorm(nn.Module):
    """RMSNorm implementation that uses the torch_rmsnorm custom op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )
        return result


class EagleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Eagle3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.is_causal = True

        # Note: Eagle3Attention expects 2 * hidden_size input, which is the concatenation of the hidden states
        # and the input embeddings.

        self.q_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        cos, sin = position_embeddings

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [Batch, Seq, Heads, Dim]
        query_states = query_states.view(bsz, q_len, -1, self.head_dim)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=self.is_causal,
            layout="bsnd",
        )

        attn_output = attn_output.view(bsz, q_len, self.num_attention_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Eagle3DecoderLayer(nn.Module):
    """Eagle decoder layer with modified attention and hidden state normalization."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.self_attn = Eagle3Attention(config, layer_idx=layer_idx)
        self.hidden_norm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = EagleMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embeds: torch.Tensor,
        position_embeds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)

        embeds = self.input_layernorm(embeds)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeds,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Eagle3Model(nn.Module):
    """Core Eagle model architecture.

    The model outputs logits over the draft vocabulary.
    """

    def __init__(self, config):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Store original vocab size before patching
        self._original_vocab_size = config.vocab_size

        if config.draft_vocab_size is not None and config.draft_vocab_size != config.vocab_size:
            # Vocab mappings for draft <-> target token conversion
            # Needed to convert draft outputs to target inputs for Eagle3.
            # Since we reuse the target model's embedding in the drafter, we need
            # to do this conversion after every draft iteration.
            self.d2t = nn.Parameter(
                torch.empty((config.draft_vocab_size,), dtype=torch.int32),
                requires_grad=False,
            )

        # Input feature fusion: 3 * hidden_size -> hidden_size for Eagle3.
        # TODO: Can make this configurable based on number of capture layers.
        self.fc = nn.Linear(
            config.hidden_size * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
            dtype=config.torch_dtype,
        )

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.rotary_emb = LlamaRotaryEmbedding(
            config=config, dim=self.head_dim, device=torch.device("cuda")
        )

        if config.num_hidden_layers > 1:
            self.midlayer = nn.ModuleList(
                [Eagle3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
            )
        else:
            self.midlayer = Eagle3DecoderLayer(config, layer_idx=0)

        self.num_hidden_layers = config.num_hidden_layers

    # Assumption: The hidden states are already fused if necessary
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_embeds = self.embed_tokens(input_ids)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeds = (cos, sin)

        if self.num_hidden_layers > 1:
            for layer in self.midlayer:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    embeds=input_embeds,
                    position_embeds=position_embeds,
                )
        else:
            hidden_states = self.midlayer(
                hidden_states=hidden_states,
                embeds=input_embeds,
                position_embeds=position_embeds,
            )

        return hidden_states

    def apply_eagle3_fc(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc(target_hidden_states)


@dataclass
class Eagle3DraftOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class Eagle3DrafterForCausalLM(PreTrainedModel):
    """HuggingFace-compatible wrapper for EagleModel.

    This wrapper makes EagleModel compatible with AutoDeploy's model loading
    and inference pipeline.
    """

    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Eagle3DecoderLayer"]

    # Checkpoint conversion mapping: Eagle checkpoints have keys like "fc.weight"
    # but the wrapper model expects "model.fc.weight" (due to self.model = Eagle3Model).
    # This mapping tells the factory to add "model." prefix when loading weights.
    # Used by AutoModelForCausalLMFactory._remap_param_names_load_hook()

    _checkpoint_conversion_mapping = {
        "^(?!lm_head|norm)": "model.",  # Prepend "model." to all keys EXCEPT lm_head and norm
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = Eagle3Model(config)
        self.norm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        eagle_config = getattr(config, "eagle_config", {})
        self._return_hidden_post_norm = eagle_config.get("return_hidden_post_norm", False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Eagle3DraftOutput:
        """Forward pass compatible with HuggingFace/AutoDeploy interface.

        Args:
            hidden_states: Hidden states from the target model. Required.
                In production speculative decoding, these come from the target model.

        Raises:
            ValueError: If hidden_states is not provided in kwargs.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise ValueError("hidden_states must be provided.")

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
        )

        norm_hidden_states = self.norm(hidden_states)
        logits = self.lm_head(norm_hidden_states)

        last_hidden_state = norm_hidden_states if self._return_hidden_post_norm else hidden_states

        return Eagle3DraftOutput(
            logits=logits,
            last_hidden_state=last_hidden_state,
        )
