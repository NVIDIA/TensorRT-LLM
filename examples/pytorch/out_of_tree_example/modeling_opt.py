from typing import Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers import OPTConfig
from transformers.activations import ACT2FN

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import (DecoderModel,
                                                       DecoderModelForCausalLM,
                                                       duplicate_kv_weight,
                                                       register_auto_model)
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode


class LayerNorm(nn.LayerNorm):

    def reset_parameters(self) -> None:
        # Skip the initialization operations that conflict with MetaInitMode
        pass


class OPTAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[OPTConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.enable_bias,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


class OPTDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[OPTConfig],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config

        self.self_attn = OPTAttention(model_config, layer_idx=layer_idx)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = LayerNorm(
            config.hidden_size,
            elementwise_affine=config.layer_norm_elementwise_affine,
            dtype=config.torch_dtype)
        self.fc1 = Linear(config.hidden_size,
                          config.ffn_dim,
                          bias=config.enable_bias,
                          dtype=config.torch_dtype,
                          mapping=model_config.mapping,
                          tensor_parallel_mode=TensorParallelMode.COLUMN,
                          quant_config=model_config.get_quant_config(),
                          allreduce_strategy=model_config.allreduce_strategy)
        self.fc2 = Linear(config.ffn_dim,
                          config.hidden_size,
                          bias=config.enable_bias,
                          dtype=config.torch_dtype,
                          mapping=model_config.mapping,
                          tensor_parallel_mode=TensorParallelMode.ROW,
                          quant_config=model_config.get_quant_config(),
                          allreduce_strategy=model_config.allreduce_strategy)
        self.final_layer_norm = LayerNorm(
            config.hidden_size,
            elementwise_affine=config.layer_norm_elementwise_affine,
            dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(
            position_ids=None,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class OPTModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[OPTConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config

        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.word_embed_proj_dim,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        self.embed_positions = Embedding(
            config.max_position_embeddings + 2,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size,
                                         config.word_embed_proj_dim,
                                         bias=False,
                                         dtype=config.torch_dtype)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim,
                                        config.hidden_size,
                                        bias=False,
                                        dtype=config.torch_dtype)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to
        # keep backward compatibility with checkpoints that have been fine-tuned
        # before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine,
                dtype=config.torch_dtype)
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([
            OPTDecoderLayer(model_config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

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

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        pos_embeds = self.embed_positions(position_ids.squeeze(0) + 2)
        hidden_states = inputs_embeds + pos_embeds

        # residual = None
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
            )

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


@register_auto_model("OPTForCausalLM")
class OPTForCausalLM(DecoderModelForCausalLM[OPTModel, OPTConfig]):

    def __init__(
        self,
        model_config: ModelConfig[OPTConfig],
    ):
        super().__init__(OPTModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: dict):
        tp_size = self.model_config.mapping.tp_size
        num_kv_heads = self.model_config.pretrained_config.num_attention_heads

        def filter_weights(prefix: str, weights: dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        params_map = {
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
            'o_proj': ['out_proj']
        }

        weight_prefix = 'model.decoder'
        if any(name.startswith('decoder') for name, _ in weights.items()):
            weight_prefix = 'decoder'

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                # skip load weights if tie word embeddings is enabled and layer is lm_head
                if self.config.tie_word_embeddings and name.startswith(
                        'lm_head'):
                    continue

                if name.startswith('model'):
                    name = name.replace('model', weight_prefix, 1)

                names = name.split('.')
                if names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        fw = filter_weights('.'.join(names[:-1] + [new_name]),
                                            weights)
                        if new_name in ['k_proj', 'v_proj']:
                            fw = {
                                k:
                                duplicate_kv_weight(
                                    weight=v[:],
                                    num_kv_heads=num_kv_heads,
                                    tensor_parallel_size=tp_size)
                                if k in ['weight', 'bias'] else v
                                for k, v in fw.items()
                            }
                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                else:
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                p.data.copy_(module_weights[n][:])
