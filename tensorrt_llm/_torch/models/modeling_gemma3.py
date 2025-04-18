import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import Gemma3TextConfig
from transformers.activations import ACT2FN

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams, PredefinedAttentionMask
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..pipeline_interface import PipelineInterface
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class Gemma3Attention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
        layer_idx: Optional[int] = None,
        is_sliding: bool = False,
    ):
        self.is_sliding = is_sliding
        config = model_config.pretrained_config
        rope_params = RopeParams.from_config(config)
        self.attention_window_size = None
        if is_sliding:
            rope_params.theta = 10000
            self.attention_window_size = config.sliding_window
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )
        q_scaling = math.sqrt(config.query_pre_attn_scalar) / math.sqrt(
            config.head_dim)
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            qk_layernorm=True,
            q_scaling=q_scaling,
        )

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:

        attention_window_size = self.attention_window_size or attn_metadata.max_seq_len
        return super().forward(position_ids=position_ids,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata,
                               attention_mask=attention_mask,
                               mrope_config=mrope_config,
                               all_reduce_params=all_reduce_params,
                               lora_params=lora_params,
                               attention_window_size=attention_window_size,
                               **kwargs)


class Gemma3MLP(nn.Module):

    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dtype = config.torch_dtype
        self.gate_proj = Linear(self.hidden_size,
                                self.intermediate_size,
                                bias=False,
                                dtype=self.dtype)
        self.up_proj = Linear(self.hidden_size,
                              self.intermediate_size,
                              bias=False,
                              dtype=self.dtype)
        self.down_proj = Linear(self.intermediate_size,
                                self.hidden_size,
                                bias=False,
                                dtype=self.dtype)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        is_sliding = bool((layer_idx + 1) % config.sliding_window_pattern)
        self.self_attn = Gemma3Attention(
            model_config,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
        )

        self.mlp = Gemma3MLP(config)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                 eps=config.rms_norm_eps,
                                                 dtype=config.torch_dtype)
        self.post_feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3TextModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Gemma3TextConfig]):
        super().__init__(model_config)
        config = self.model_config
        self.hidden_size = config.pretrained_config.hidden_size
        self.padding_idx = config.pretrained_config.pad_token_id

        self.embed_tokens = Embedding(
            config.pretrained_config.vocab_size,
            config.pretrained_config.hidden_size,
            dtype=config.pretrained_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.pretrained_config.hidden_size,
                            eps=config.pretrained_config.rms_norm_eps,
                            dtype=config.pretrained_config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds * math.sqrt(self.hidden_size)

        hidden_states = inputs_embeds.to(self.dtype)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(position_ids=position_ids,
                                          hidden_states=hidden_states,
                                          attn_metadata=attn_metadata)

        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("Gemma3ForCausalLM")
class Gemma3ForCausalLM(DecoderModelForCausalLM[Gemma3TextModel,
                                                Gemma3TextConfig]):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
    ):
        super().__init__(Gemma3TextModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pipeline_interface: Optional[PipelineInterface] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:

        if self._supports_pp and self.pp_size > 1:
            output = self.model(
                input_ids=input_ids,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pipeline_interface=pipeline_interface,
            )

            # No need to compute logits for non-last PP ranks
            if self.pp_rank < self.pp_size - 1:
                return output
        else:
            output = self.model(
                input_ids=input_ids,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )
