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
from typing import Dict, Optional, Tuple

import torch
import transformers.models.mllama.configuration_mllama as config_mllama
from torch import nn
from tqdm import tqdm

from ..attention_backend.interface import AttentionMetadata
from ..distributed import ParallelConfig, TensorParallelMode
from ..model_config import ModelConfig
from ..modules.embedding import Embedding, LMHead
from ..modules.logits_procesor import LogitsProcessor
from .modeling_llama import LlamaDecoderLayer
from .modeling_utils import duplicate_kv_weight, register_auto_model


# TODO: For performance consideration, should use from ..modules.rms_norm import RMSNorm
# to utilize fused add rms norm after flashinfer rmsnorm accuracy issue is resolved.
class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual: Optional[torch.Tensor] = None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if residual is not None:
            hidden_states = hidden_states + residual.to(torch.float32)
            residual = hidden_states.to(input_dtype)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)

        hidden_states = self.weight * hidden_states.to(input_dtype)

        if residual is not None:
            return hidden_states, residual
        return hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MllamaTextModel(nn.Module):

    def __init__(
        self,
        config: ModelConfig[config_mllama.MllamaConfig],
        cache_config=None,
    ):
        super().__init__()
        pretrained_config = config.pretrained_config
        text_config = pretrained_config.text_config
        # llama_config for reusing the LlamaDecoderLayer
        llama_config = ModelConfig(
            pretrained_config=pretrained_config.text_config,
            mapping=config.mapping,
            quant_config=config.quant_config,
            quant_config_dict=config.quant_config_dict,
            attn_backend=config.attn_backend)
        llama_config.pretrained_config.attention_bias = False
        llama_config.pretrained_config.mlp_bias = False

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
                layers.append(
                    LlamaDecoderLayer(llama_config, layer_idx=layer_id))

        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(hidden_size=text_config.hidden_size,
                            eps=text_config.rms_norm_eps,
                            dtype=text_config.torch_dtype)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
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
            if decoder_layer == None:
                assert skip_cross_attention == True, 'Cross-attention is not supported yet'
            elif isinstance(decoder_layer, LlamaDecoderLayer):
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
            parallel_config=ParallelConfig(
                tensor_parallel_rank=config.mapping.tp_rank,
                tensor_parallel_size=config.mapping.tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                gpus_per_node=config.mapping.gpus_per_node,
            ),
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
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
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
        self.create_weights()

    def create_weights(self):
        for _, module in self.named_modules():
            if callable(getattr(module, "_create_weights", None)):
                module._create_weights()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
        text_head_dim = text_config.hidden_size // text_config.num_attention_heads
        vision_head_dim = vision_config.hidden_size // vision_config.attention_heads

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

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
                head_dim = vision_head_dim if "vision_model" in name else text_head_dim

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
                                    head_dim=head_dim,
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
