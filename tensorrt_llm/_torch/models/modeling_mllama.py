# Adapted from https://github.com/huggingface/transformers/blob/a7f5479b45a8040392af80bf1107a2bdd796931c/src/transformers/models/mllama/modeling_mllama.py
#
# Copyright 2024 the NVIDIA Inc. team. All rights reserved.
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import transformers.models.mllama.configuration_mllama as config_mllama
from torch import nn
from tqdm import tqdm

from ...logger import logger
from ..attention_backend.interface import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding, LMHead
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.logits_processor import LogitsProcessor
from ..modules.rms_norm import RMSNorm
from .modeling_llama import LlamaAttention
from .modeling_utils import (duplicate_kv_weight, filter_weights,
                             register_auto_model)


class MllamaDecoderLayer(DecoderLayer):

    def __init__(
        self,
        config: ModelConfig[config_mllama.MllamaConfig],
        layer_idx: int,
    ) -> None:
        super().__init__()
        pretrained_config = config.pretrained_config
        # llama_config for reusing the LlamaAttention
        llama_config = ModelConfig(
            pretrained_config=pretrained_config.text_config,
            mapping=config.mapping,
            quant_config=config.quant_config,
            quant_config_dict=config.quant_config_dict,
            attn_backend=config.attn_backend)
        llama_config.pretrained_config.attention_bias = False
        llama_config.pretrained_config.mlp_bias = False

        self.layer_idx = layer_idx

        self.self_attn = LlamaAttention(
            llama_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=llama_config.pretrained_config.hidden_size,
            intermediate_size=llama_config.pretrained_config.intermediate_size,
            bias=llama_config.pretrained_config.mlp_bias,
            dtype=llama_config.pretrained_config.torch_dtype,
            config=llama_config,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=llama_config.pretrained_config.hidden_size,
            eps=llama_config.pretrained_config.rms_norm_eps,
            dtype=llama_config.pretrained_config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(
            hidden_size=llama_config.pretrained_config.hidden_size,
            eps=llama_config.pretrained_config.rms_norm_eps,
            dtype=llama_config.pretrained_config.torch_dtype)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
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
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MllamaTextModel(nn.Module):

    def __init__(
        self,
        config: ModelConfig[config_mllama.MllamaConfig],
        cache_config=None,
    ):
        super().__init__()
        pretrained_config = config.pretrained_config
        text_config = pretrained_config.text_config
        self.padding_id = text_config.pad_token_id
        self.vocab_size = text_config.vocab_size
        self.embed_tokens = Embedding(
            text_config.vocab_size + 8,
            text_config.hidden_size,
            dtype=text_config.torch_dtype,
        )

        self.cross_attention_layers = text_config.cross_attention_layers

        layers = []
        for layer_id in range(text_config.num_hidden_layers):
            if layer_id in self.cross_attention_layers:
                # TODO: Cross-attention decoder layer impl.
                layers.append(None)
            else:
                layers.append(MllamaDecoderLayer(config, layer_idx=layer_id))

        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(hidden_size=text_config.hidden_size,
                            eps=text_config.rms_norm_eps,
                            dtype=text_config.torch_dtype)

    def forward(
        self,
        input_ids: torch.IntTensor,
        positions: Optional[torch.IntTensor],
        cross_attention_states: Optional[torch.LongTensor],
        cross_attention_mask: Optional[torch.LongTensor],
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor,
                                                      torch.Tensor]],
        attn_metadata: AttentionMetadata,
        skip_cross_attention: bool,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        residual = None
        for _, decoder_layer in enumerate(self.layers):
            if decoder_layer is None:
                assert skip_cross_attention == True, 'Cross-attention is not supported yet'
            elif isinstance(decoder_layer, MllamaDecoderLayer):
                hidden_states, residual = decoder_layer(
                    position_ids=positions,
                    hidden_states=hidden_states,
                    attn_metadata=attn_metadata,
                    residual=residual)
            else:
                raise ValueError(
                    f"Unknown decoder layer type {type(decoder_layer)}")

        hidden_states, residual = self.norm(hidden_states, residual)
        return hidden_states


class MllamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: ModelConfig[config_mllama.MllamaConfig],
        cache_config=None,
    ):
        super().__init__()
        self.config = config
        pretrain_config = config.pretrained_config
        text_config = pretrain_config.text_config

        self.vocab_size = text_config.vocab_size
        self.model = MllamaTextModel(config, cache_config)

        # TODO(zhenhuanc): Currently lm_head Linear will not accept QuantConfig
        # will considering per layer QuantConfig in the future.
        self.lm_head = LMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            dtype=text_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        # use embedding weights in lm_head if tie word embedding is enabled
        if text_config.tie_word_embeddings:
            assert self.lm_head.tp_size == self.model.embed_tokens.tp_size, (
                "lm_head and vocab embedding should use the same TP size")
            assert self.lm_head.tp_mode == self.model.embed_tokens.tp_mode, (
                "lm_head and vocab embedding should use the same TP mode")
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.IntTensor,
        positions: Optional[torch.IntTensor],
        cross_attention_states: Optional[torch.LongTensor],
        cross_attention_mask: Optional[torch.LongTensor],
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor,
                                                      torch.Tensor]],
        attn_metadata: AttentionMetadata,
        skip_cross_attention: bool,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            attn_metadata=attn_metadata,
            skip_cross_attention=skip_cross_attention,
        )
        return hidden_states

    def infer_max_seq_len(self) -> int:
        # NOTE: Copied from DecoderModelForCausalLM.infer_max_seq_len
        # Modified from tensorrt_llm/builder.py _init_max_seq_len
        rope_scaling = getattr(self.config, 'rope_scaling', None)
        rope_factor = 1
        if rope_scaling is not None:
            rope_type = rope_scaling.get('type', rope_scaling.get('rope_type'))
            if rope_type not in ("su", "longrope", "llama3", "yarn"):
                rope_factor = rope_scaling.get('factor', 1.0)

        # Step 1: Find the upper bound of max_seq_len
        inferred_max_seq_len = 2048
        if getattr(self.config, 'max_position_embeddings', None) is not None:
            inferred_max_seq_len = self.config.max_position_embeddings

        # Step 2: Scale max_seq_len with rotary scaling
        if rope_factor != 1:
            inferred_max_seq_len = int(
                math.ceil(inferred_max_seq_len * rope_factor))
            logger.warning(
                f'max_seq_len is scaled to {inferred_max_seq_len} by rope scaling {rope_factor}'
            )

        # Step 3: Return the new max_seq_len
        return inferred_max_seq_len


@register_auto_model("MllamaForConditionalGeneration")
class MllamaForConditionalGeneration(nn.Module):

    def __init__(
        self,
        config: ModelConfig[config_mllama.MllamaConfig],
        cache_config=None,
    ):
        super().__init__()

        self.config = self.model_config = config
        pretrained_config = config.pretrained_config
        self.vocab_size = pretrained_config.text_config.vocab_size
        self.hidden_size = pretrained_config.text_config.hidden_size
        self.max_num_tiles = pretrained_config.vision_config.max_num_tiles
        self.vision_output_dim = pretrained_config.vision_config.vision_output_dim
        self.pad_token_id = (pretrained_config.pad_token_id if
                             pretrained_config.pad_token_id is not None else -1)
        self.image_size = pretrained_config.vision_config.image_size

        # hack config
        pretrained_config_backup = copy.deepcopy(pretrained_config.__dict__)
        self.config.pretrained_config.__dict__.update(
            pretrained_config.text_config.__dict__)
        self.config.pretrained_config.__dict__.update(pretrained_config_backup)
        self.config.__dict__.update(pretrained_config.text_config.__dict__)

        self.language_model = MllamaForCausalLM(
            config,
            cache_config=cache_config,
        )
        self.multi_modal_projector = nn.Linear(
            pretrained_config.vision_config.vision_output_dim,
            pretrained_config.text_config.hidden_size,
            bias=True,
            dtype=config.pretrained_config.vision_config.torch_dtype)
        self.logits_processor = LogitsProcessor()

    def infer_max_seq_len(self) -> int:
        return self.language_model.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        # TODO:  figure out the image-related inputs in kwargs
        **kwargs: object,
    ):
        cross_attention_states = None
        cross_attention_mask = None

        # TODO:  Enable image_inputs
        full_text_row_masked_out_mask = None
        skip_cross_attention = True

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            attn_metadata=attn_metadata,
            skip_cross_attention=skip_cross_attention,
        )
        return self.logits_processor(hidden_states, self.language_model.lm_head,
                                     attn_metadata, return_context_logits)

    def load_weights(self, weights: Dict):
        tp_size = self.config.mapping.tp_size
        vision_config = self.config.pretrained_config.vision_config
        text_config = self.config.pretrained_config.text_config
        text_config.hidden_size // text_config.num_attention_heads
        vision_config.hidden_size // vision_config.attention_heads

        params_map = {
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
            'gate_up_proj': ['gate_proj', 'up_proj']
        }

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                # skip load weights if tie word embeddings is enabled and layer is lm_head
                if text_config.tie_word_embeddings and "lm_head" in name:
                    continue
                num_kv_heads = vision_config.num_key_value_heads if "vision_model" in name else text_config.num_key_value_heads

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
                                if k in ["weight", "bias"] else v
                                for k, v in fw.items()
                            }
                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                else:
                    if 'patch_embedding._linear' in name:
                        name = name.replace('patch_embedding._linear',
                                            'patch_embedding')
                        module_weights = filter_weights(name, weights)
                        for k in module_weights.keys():
                            v = module_weights[k][:]
                            module_weights[k] = v.view(v.shape[0], -1)
                    else:
                        module_weights = filter_weights(name, weights)

                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                weight = module_weights[n][:]
                                p.data.copy_(weight)
