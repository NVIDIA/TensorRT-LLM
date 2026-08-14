from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Size, Tensor, nn
from transformers import NemotronConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.mlp import MLP
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class NemotronAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[NemotronConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


class NemotronLayerNormPlus1(nn.LayerNorm):
    ##### No FlashInfer support, unlike built-in RMSNorm #####
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(normalized_shape, eps, elementwise_affine, bias,
                         device, dtype)

    def reset_parameters(self) -> None:
        # Skip the initialization operations that conflict with MetaInitMode
        pass

    def forward(self, input: Tensor) -> Tensor:
        args = (input, self.normalized_shape, self.weight + 1, self.bias,
                self.eps)
        return F.layer_norm(*args)


class NemotronDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[NemotronConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        self.self_attn = NemotronAttention(
            model_config,
            layer_idx=layer_idx,
        )

        relu_squared = lambda x: torch.square(torch.nn.functional.relu(x))
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            activation=relu_squared,
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = NemotronLayerNormPlus1(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
            dtype=config.torch_dtype)
        self.post_attention_layernorm = NemotronLayerNormPlus1(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
            dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class NemotronModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[NemotronConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList([
            NemotronDecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = NemotronLayerNormPlus1(normalized_shape=config.hidden_size,
                                           eps=config.norm_eps,
                                           dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(position_ids=position_ids,
                                          hidden_states=hidden_states,
                                          attn_metadata=attn_metadata)

        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("NemotronForCausalLM")
class NemotronForCausalLM(DecoderModelForCausalLM[NemotronModel,
                                                  NemotronConfig]):

    def __init__(
        self,
        model_config: ModelConfig[NemotronConfig],
    ):
        super().__init__(NemotronModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
