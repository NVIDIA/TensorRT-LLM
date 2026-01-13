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

Eagle is a speculative decoding draft model that predicts next tokens based on
hidden states from a target model (e.g., Llama-3.1-8B-Instruct).

This implementation:
- Defines EagleConfig extending LlamaConfig with model_type="eagle3"
- Wraps EagleModel in a HuggingFace-compatible interface
- Registers with AutoDeploy's custom model mechanism

Note: Eagle uses the same tokenizer as its target model (Llama), so when using
this model, you must explicitly specify the tokenizer path pointing to the
target model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, LlamaConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class EagleConfig(LlamaConfig):
    """Eagle-specific config that extends LlamaConfig.

    This config is used to identify Eagle models and load them with the
    correct architecture. Eagle models have a single transformer layer
    and may have a different draft vocabulary size.

    Attributes:
        model_type: Set to "eagle3" to identify this as an Eagle model.
        draft_vocab_size: The vocabulary size for draft predictions (may differ
            from the target model's vocab_size).
    """

    model_type = "eagle3"

    def __init__(self, draft_vocab_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.draft_vocab_size = draft_vocab_size

        # Eagle has a single transformer layer
        if self.num_hidden_layers > 1:
            self.num_hidden_layers = 1


# Register EagleConfig with HuggingFace's AutoConfig
AutoConfig.register("eagle3", EagleConfig)


class Eagle3Attention(LlamaAttention):
    """Eagle attention layer with doubled input projection dimension.

    Eagle's attention takes concatenated hidden states [embeds, hidden_states]
    of size 2 * hidden_size as input.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__(config, layer_idx)
        # Projection layers accept 2 * hidden_size (concatenated input)
        self.q_proj = nn.Linear(
            2 * config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            2 * config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            2 * config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        # o_proj remains [hidden_size, hidden_size]


class Eagle3DecoderLayer(LlamaDecoderLayer):
    """Eagle decoder layer with modified attention and hidden state normalization."""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__(config, layer_idx)
        self.self_attn = Eagle3Attention(config, layer_idx=layer_idx)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_embeds: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)

        embeds = self.input_layernorm(embeds)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeds,
        )[0]

        hidden_states = residual + hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class EagleModel(nn.Module):
    """Core Eagle model architecture.

    Eagle is a lightweight draft model for speculative decoding. It takes:
    - input_ids: Token IDs from the target model
    - position_ids: Position indices
    - hidden_states: Concatenated hidden states from target model layers
      (typically 3 layers, resulting in hidden_size * 3 dimensions)

    The model outputs logits over the draft vocabulary.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        # Vocab mappings for draft <-> target token conversion
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.long))
        self.register_buffer(
            "d2t",
            torch.zeros(
                config.draft_vocab_size if config.draft_vocab_size else config.vocab_size,
                dtype=torch.long,
            ),
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Store original vocab size before patching
        self._original_vocab_size = config.vocab_size

        if config.draft_vocab_size is not None and config.draft_vocab_size != config.vocab_size:
            config.vocab_size = config.draft_vocab_size

        # Input feature fusion: 3 * hidden_size -> hidden_size
        self.fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Single transformer layer
        self.midlayer = Eagle3DecoderLayer(config, layer_idx=0)

        # Output head
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Fuse hidden states from target model layers
        hidden_states = self.fc(hidden_states)

        input_embeds = self.embed_tokens(input_ids)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeds = (cos, sin)

        out = self.midlayer(
            hidden_states=hidden_states,
            embeds=input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_embeds=position_embeds,
        )[0]

        logits = self.lm_head(self.norm(out))
        return logits


class EagleModelForCausalLM(PreTrainedModel):
    """HuggingFace-compatible wrapper for EagleModel.

    This wrapper makes EagleModel compatible with AutoDeploy's model loading
    and inference pipeline.

    For standalone testing, mock hidden states are used. In production with
    speculative decoding, real hidden states come from the target model.
    """

    config_class = EagleConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Eagle3DecoderLayer"]

    # Settings for mock hidden states (testing only)
    _mock_max_batch_size: int = 8
    _mock_max_seq_len: int = 512

    def __init__(self, config: EagleConfig):
        super().__init__(config)
        self.model = EagleModel(config)

        # Pre-allocate mock hidden states buffer for testing
        # Use register_buffer so it moves with the model (important for AutoDeploy tracing)
        hidden_dim = config.hidden_size
        generator = torch.Generator().manual_seed(42)
        self.register_buffer(
            "_mock_hidden_states",
            torch.randn(
                (self._mock_max_batch_size, self._mock_max_seq_len, hidden_dim * 3),
                generator=generator,
            ),
            persistent=False,  # Don't save to state_dict
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass compatible with HuggingFace/AutoDeploy interface.

        For testing, uses pre-allocated mock hidden states.
        In production, hidden_states would come from the target model.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.model.fc.weight.dtype

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        # Use mock hidden states for testing
        # The buffer is registered so it moves with the model automatically
        # Just need to handle dtype conversion if needed
        mock_states = self._mock_hidden_states
        if mock_states.dtype != dtype:
            mock_states = mock_states.to(dtype=dtype)

        hidden_states = mock_states[:batch_size, :seq_len, :]

        logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @classmethod
    def _from_config(cls, config, **kwargs):
        """Required by AutoDeploy's factory for custom model loading."""
        return cls(config)


# Register EagleModelForCausalLM with AutoDeploy's factory
# This enables AutoDeploy to use our custom implementation when it encounters EagleConfig
AutoModelForCausalLMFactory.register_custom_model_cls("EagleConfig", EagleModelForCausalLM)
