from typing import Optional, Tuple

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
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.functional import PositionEmbeddingType


class Phi3Attention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[Phi3Config],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        # Pass max_seq_len to config for LongRoPE.
        config.max_seq_len = model_config.max_seq_len
        rope_params = RopeParams.from_config(config)
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
            layer_idx=layer_idx,
        )

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
        hidden_states = self.mlp(
            hidden_states,
            lora_params=lora_params,
            **kwargs,
        )
        return hidden_states, residual


class Phi3Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[Phi3Config]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.padding_idx = config.pad_token_id

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
        self.model_config.mapping.tp_size
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = hidden_size // num_heads

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
                    if "self_attn.qkv_proj" in name:
                        # The weights need to be split correctly before sharding to support tp_size >1.
                        qkv_weight = module_weights['weight'][:]
                        qk_split_index = hidden_size
                        kv_split_index = hidden_size + num_kv_heads * head_dim

                        q_dict = {'weight': qkv_weight[:qk_split_index, :]}
                        k_dict = {
                            'weight':
                            qkv_weight[qk_split_index:kv_split_index, :]
                        }
                        v_dict = {'weight': qkv_weight[kv_split_index:, :]}

                        # Get the scale factor for the fused QKV projection
                        qkv_scale = module_weights.get('weight_scale', None)

                        if qkv_scale is not None:
                            if qkv_scale.shape and qkv_scale.shape[
                                    0] == qkv_weight.shape[0]:
                                q_dict[
                                    'weight_scale'] = qkv_scale[:
                                                                qk_split_index, :]
                                k_dict['weight_scale'] = qkv_scale[
                                    qk_split_index:kv_split_index, :]
                                v_dict['weight_scale'] = qkv_scale[
                                    kv_split_index:, :]
                            else:  # use same scale
                                q_dict['weight_scale'] = qkv_scale
                                k_dict['weight_scale'] = qkv_scale
                                v_dict['weight_scale'] = qkv_scale

                        input_scale = module_weights.get('input_scale', None)
                        if input_scale is not None:
                            q_dict['input_scale'] = input_scale
                            k_dict['input_scale'] = input_scale
                            v_dict['input_scale'] = input_scale

                        weight_scale_2 = module_weights.get(
                            'weight_scale_2', None)
                        if weight_scale_2 is not None:
                            q_dict['weight_scale_2'] = weight_scale_2
                            k_dict['weight_scale_2'] = weight_scale_2
                            v_dict['weight_scale_2'] = weight_scale_2

                        module.load_weights(weights=[q_dict, k_dict, v_dict])
                    elif "mlp.gate_up_proj" in name:
                        # The weights need to be split correctly before sharding to support tp_size >1.
                        intermediate_size = self.config.intermediate_size
                        gate_up_weight = module_weights['weight'][:]
                        gate_weight = gate_up_weight[:intermediate_size, :]
                        up_weight = gate_up_weight[intermediate_size:, :]

                        gate_dict = {'weight': gate_weight}
                        up_dict = {'weight': up_weight}

                        # Get the scale factors if they exist
                        gate_up_scale = module_weights.get('weight_scale', None)
                        if gate_up_scale is not None:
                            if gate_up_scale.shape and gate_up_scale.shape[
                                    0] == gate_up_weight.shape[0]:
                                gate_dict[
                                    'weight_scale'] = gate_up_scale[:
                                                                    intermediate_size, :]
                                up_dict['weight_scale'] = gate_up_scale[
                                    intermediate_size:, :]
                            else:  # use same scale
                                gate_dict['weight_scale'] = gate_up_scale
                                up_dict['weight_scale'] = gate_up_scale

                        input_scale = module_weights.get('input_scale', None)
                        if input_scale is not None:
                            gate_dict['input_scale'] = input_scale
                            up_dict['input_scale'] = input_scale

                        weight_scale_2 = module_weights.get(
                            'weight_scale_2', None)
                        if weight_scale_2 is not None:
                            gate_dict['weight_scale_2'] = weight_scale_2
                            up_dict['weight_scale_2'] = weight_scale_2

                        module.load_weights(weights=[gate_dict, up_dict])
                    else:
                        module.load_weights(weights=[module_weights])
                else:
                    for n, p in module._parameters.items():
                        if p is not None:
                            p.data.copy_(module_weights[n][:])
