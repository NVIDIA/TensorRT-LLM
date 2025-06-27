from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import Phi3Config

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import (DecoderModel,
                                                       DecoderModelForCausalLM,
                                                       register_auto_model)
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import (TensorParallelMode, WeightMode,
                                                WeightsLoadingConfig)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType


def create_sinusoidal_positions_long_rope(num_pos: int,
                                          dim: int,
                                          theta: float,
                                          original_max_pos: int,
                                          short_factor: List[float],
                                          long_factor: List[float],
                                          dtype=np.float32):

    short_factor = torch.tensor(short_factor, dtype=torch.float32)
    long_factor = torch.tensor(long_factor, dtype=torch.float32)

    inv_freq = 1.0 / (theta
                      **(torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Short part
    inv_freq_short = inv_freq / short_factor
    t_short = torch.arange(min(num_pos, original_max_pos), dtype=torch.float32)
    freqs_short = torch.einsum("i,j->ij", t_short, inv_freq_short)

    # Long part
    inv_freq_long = inv_freq / long_factor
    t_long = torch.arange(max(0, num_pos - original_max_pos),
                          dtype=torch.float32) + original_max_pos
    freqs_long = torch.einsum("i,j->ij", t_long, inv_freq_long)

    freqs = torch.cat([freqs_short, freqs_long], dim=0)

    sinusoid_inp = freqs.float().unsqueeze(-1).numpy()

    # fuse cos/sin into float2 (cos, sin).
    concat = np.concatenate((np.cos(sinusoid_inp), np.sin(sinusoid_inp)),
                            axis=-1)

    return None, concat.reshape(1, -1).astype(dtype)


_old_create_rope_const_params = RopeParams.create_rope_const_params


def _new_create_rope_const_params(self, interleave: bool = True):
    # self is a RopeParams object
    if hasattr(self,
               'scale_type') and self.scale_type == RotaryScalingType.longrope:
        rope_inv_freq = None
        _, rope_cos_sin = create_sinusoidal_positions_long_rope(
            num_pos=self.max_positions,
            dim=self.dim,
            theta=self.theta,
            original_max_pos=self.original_max_positions,
            short_factor=self.short_factor,
            long_factor=self.long_factor,
        )
        if not interleave:
            rope_cos_sin = rope_cos_sin.reshape(
                self.max_positions, -1,
                2)[:, :self.dim // 2, :].transpose(0, 2, 1).reshape(1, -1)

        rope_cos_sin = torch.tensor(
            rope_cos_sin,
            dtype=torch.float32,
            device='cuda',
        )
        return rope_inv_freq, rope_cos_sin
    else:
        return _old_create_rope_const_params(self, interleave)


RopeParams.create_rope_const_params = _new_create_rope_const_params


class Phi3Attention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[Phi3Config],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config

        rope_params = RopeParams.from_config(config)
        if hasattr(
                config, "rope_scaling"
        ) and config.rope_scaling is not None and config.rope_scaling[
                'type'] == 'longrope':
            rope_params.scale_type = RotaryScalingType.longrope
            rope_params.short_factor = config.rope_scaling['short_factor']
            rope_params.long_factor = config.rope_scaling['long_factor']
            rope_params.original_max_positions = config.original_max_position_embeddings

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=rope_params,
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )
        # Override the weights_loading_config for qkv_proj.
        self.qkv_proj.weights_loading_config = WeightsLoadingConfig(
            weight_mode=WeightMode.VANILLA, )


class Phi3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Phi3Config],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        self.self_attn = Phi3Attention(model_config, layer_idx=layer_idx)

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
            config=model_config,
        )
        # Override the weights_loading_config for gate_up_proj.
        self.mlp.gate_up_proj.weights_loading_config = WeightsLoadingConfig(
            weight_mode=WeightMode.VANILLA, )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        lora_params=None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=None,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            lora_params=lora_params,
            **kwargs,
        )

        # Fully connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states, **kwargs)
        return hidden_states, residual


class Phi3Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[Phi3Config]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.padding_idx = config.pad_token_id

        config.vocab_size
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            Phi3DecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        lora_params=None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                residual=residual,
                attn_metadata=attn_metadata,
                lora_params=lora_params,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("Phi3ForCausalLM")
class Phi3ForCausalLM(DecoderModelForCausalLM[Phi3Model, Phi3Config]):

    def __init__(
        self,
        model_config: ModelConfig[Phi3Config],
    ):
        super().__init__(Phi3Model(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: dict):
        # TODO (williamj): maybe need to update it for tp_size>1.

        def filter_weights(prefix: str, weights: dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                # skip load weights if tie word embeddings is enabled and layer is lm_head
                if self.config.tie_word_embeddings and name.startswith(
                        'lm_head'):
                    continue

                module_weights = filter_weights(name, weights)
                if hasattr(module, 'load_weights'):
                    module.load_weights(weights=[module_weights])
                else:
                    for n, p in module._parameters.items():
                        if p is not None:
                            p.data.copy_(module_weights[n][:])
