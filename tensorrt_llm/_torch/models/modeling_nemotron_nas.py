from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.lora_manager import HfLoraLoader
from tensorrt_llm.models.convert_utils import split_matrix_tp

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
    intermediate_size = int(2 * ffn_mult * n_embd / 3)
    return _find_multiple(intermediate_size, 256)


def _find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def _create_linear_from_configs(model_config: ModelConfig[PretrainedConfig],
                                config: PretrainedConfig):
    return Linear(
        config.hidden_size,
        config.hidden_size,
        bias=False,
        dtype=config.torch_dtype,
        mapping=model_config.mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        quant_config=model_config.get_quant_config(),
        skip_create_weights_in_init=model_config.skip_create_weights_in_init,
    )


class NemotronNASAttention(Attention):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads[layer_idx],
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config)


class LinearAttention(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 config: PretrainedConfig):
        super().__init__()
        self.linear_attn = _create_linear_from_configs(model_config, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lora_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if lora_params is not None:
            raise NotImplementedError(
                "LinearAttention with LoRA is not supported yet")
        return self.linear_attn(hidden_states)


class LinearMLP(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 config: PretrainedConfig):
        super().__init__()
        self.linear_mlp = _create_linear_from_configs(model_config, config)

    def forward(self,
                hidden_states: torch.Tensor,
                lora_params: Optional[dict] = None) -> torch.Tensor:
        if lora_params is not None:
            raise NotImplementedError(
                "LinearMLP with LoRA is not supported yet")
        return self.linear_mlp(hidden_states)


class NemotronNASDecoderLayer(DecoderLayer):

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
                self.self_attn = LinearAttention(model_config, config)
            else:
                self.self_attn = NemotronNASAttention(model_config=model_config,
                                                      layer_idx=layer_idx)
        if not self.block_config.ffn.no_op:
            self.post_attention_layernorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype)
            if self.block_config.ffn.replace_with_linear:
                self.mlp = LinearMLP(model_config, config)
            else:
                self.mlp = GatedMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=_ffn_mult_to_intermediate_size(
                        self.block_config.ffn.ffn_mult, config.hidden_size),
                    bias=False,
                    dtype=config.torch_dtype,
                    config=model_config,
                    layer_idx=layer_idx)

    def forward(
        self,
        position_ids: torch.IntTensor,
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
            hidden_states = self.mlp(hidden_states, **kwargs)
            hidden_states = residual + hidden_states

        return hidden_states


class NemotronNASModel(DecoderModel):

    def __init__(self, model_config):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        config.num_key_value_heads = [
            config.num_attention_heads //
            block.attention.n_heads_in_group if block.attention.n_heads_in_group
            else 0  # Set to 0 when block is configured to be a NO-OP
            for block in config.block_configs
        ]

        vocab_size = config.vocab_size
        self.has_custom_embed_tokens = False
        if hasattr(
                model_config,
                'lora_config') and model_config.lora_config is not None and len(
                    model_config.lora_config.lora_dir) == 1:
            lora_loader = HfLoraLoader(model_config.lora_config.lora_dir)
            if lora_loader.vocab_size != 0 and lora_loader.embed_tokens is not None:
                vocab_size = lora_loader.vocab_size
                weight = lora_loader.embed_tokens
                self.has_custom_embed_tokens = True

        self.embed_tokens = Embedding(
            vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        if self.has_custom_embed_tokens:
            with torch.no_grad():
                if model_config.mapping.tp_size > 1:
                    weight = split_matrix_tp(
                        weight,
                        model_config.mapping.tp_size,
                        model_config.mapping.tp_rank,
                        dim=0)  # split by vocabulary dimension
                x = weight.to(self.embed_tokens.dtype)
                self.embed_tokens.weight.data.copy_(x)

        self.layers = nn.ModuleList([
            NemotronNASDecoderLayer(model_config, block_config, layer_idx)
            for layer_idx, block_config in enumerate(config.block_configs)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)


@register_auto_model("DeciLMForCausalLM")
class NemotronNASForCausalLM(DecoderModelForCausalLM[NemotronNASModel,
                                                     PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(NemotronNASModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
