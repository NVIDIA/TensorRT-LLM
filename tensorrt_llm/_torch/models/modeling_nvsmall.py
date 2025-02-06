from typing import Any, Dict

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.rms_norm import RMSNorm
from ..modules.rotary_embedding import RotaryEmbedding
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class NVSmallRotaryEmbedding(RotaryEmbedding):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        if config.rope_scaling is not None:
            rope_type = config.rope_scaling.get("rope_type",
                                                config.rope_scaling.get("type"))
        else:
            rope_type = "default"
        super().__init__(config,
                         head_dim=config.hidden_size //
                         config.num_attention_heads,
                         num_attention_heads=config.num_attention_heads,
                         max_position_embeddings=config.max_position_embeddings,
                         rope_type=rope_type)


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
    intermediate_size = int(2 * ffn_mult * n_embd / 3)
    return _find_multiple(intermediate_size, 256)


def _find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class LinearMLP(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.linear_mlp = nn.Linear(config.hidden_size,
                                    config.hidden_size,
                                    bias=False,
                                    dtype=config.torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_mlp(x)


class NVSmallAttention(Attention):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int):
        config = model_config.pretrained_config
        if model_config.fuse_pos_embd:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )
        else:
            pos_embd_params = None
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads[layer_idx],
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            rotary_emb=NVSmallRotaryEmbedding(config, layer_idx=layer_idx),
            layer_idx=layer_idx,
            config=model_config)


class LinearAttention(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.linear_attn = nn.Linear(config.hidden_size,
                                     config.hidden_size,
                                     bias=False,
                                     dtype=config.torch_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.linear_attn(hidden_states)


class NVSmallDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 block_config: Dict[str, Any], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.block_config = block_config
        if not self.block_config.attention.no_op:
            self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                           eps=config.rms_norm_eps,
                                           dtype=config.torch_dtype)
            if self.block_config.attention.replace_with_linear:
                self.self_attn = LinearAttention(config)
            else:
                self.self_attn = NVSmallAttention(model_config=model_config,
                                                  layer_idx=layer_idx)
        if not self.block_config.ffn.no_op:
            self.post_attention_layernorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype)
            if self.block_config.ffn.replace_with_linear:
                self.mlp = LinearMLP(config)
            else:
                self.mlp = GatedMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=_ffn_mult_to_intermediate_size(
                        self.block_config.ffn.ffn_mult, config.hidden_size),
                    bias=False,
                    dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        if not self.block_config.attention.no_op:
            # Self Attention
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            hidden_states = self.self_attn(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                **kwargs,
            )
            hidden_states = residual + hidden_states

        if not self.block_config.ffn.no_op:
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states


class NVSmallModel(DecoderModel):

    def __init__(self, model_config):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        config.num_key_value_heads = [
            config.num_attention_heads //
            block.attention.n_heads_in_group if block.attention.n_heads_in_group
            else block.attention.n_heads_in_group
            for block in config.block_configs
        ]

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList([
            NVSmallDecoderLayer(model_config, block_config, layer_idx)
            for layer_idx, block_config in enumerate(config.block_configs)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)


@register_auto_model("DeciLMForCausalLM")
class NVSmallForCausalLM(DecoderModelForCausalLM[NVSmallModel,
                                                 PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(NVSmallModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
