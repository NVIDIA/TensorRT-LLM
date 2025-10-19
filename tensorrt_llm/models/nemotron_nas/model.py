# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Union

from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.functional import (AllReduceFusionOp, AllReduceParams,
                                     AttentionMaskType, PositionEmbeddingType,
                                     Tensor, gather_last_token_logits, recv,
                                     send)
from tensorrt_llm.layers.attention import (Attention, AttentionParams,
                                           KeyValueCacheParams,
                                           SpecDecodingParams)
from tensorrt_llm.layers.embedding import Embedding
from tensorrt_llm.layers.linear import ColumnLinear
from tensorrt_llm.layers.lora import LoraParams
from tensorrt_llm.layers.mlp import GatedMLP
from tensorrt_llm.layers.normalization import RmsNorm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import has_safetensors
from tensorrt_llm.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm.models.nemotron_nas.config import DeciConfig
from tensorrt_llm.models.nemotron_nas.convert import (
    load_weights_from_hf_model, load_weights_from_hf_safetensors,
    update_weights_following_modelopt_optimization)
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.plugin.plugin import init_all_reduce_helper

from ..._common import default_net
from ..._utils import pad_vocab_size
from ..modeling_utils import PretrainedConfig, QuantConfig, preprocess_weights


@dataclass
class DeciLMLayerOutput:
    hidden_states: Tensor
    present_kv: Optional[Tensor] = None


@dataclass
class DeciLMLayerListOutput:
    hidden_states: Tensor
    present_kvs: List[Tensor]


class NoOp(Module):

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> int:
        return 0


class NoOpAttention(NoOp):

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache: bool = False,
                *args,
                **kwargs) -> Union[int, Tuple[int, None]]:
        out = super().forward(hidden_states=hidden_states,
                              attention_mask=attention_mask,
                              use_cache=use_cache,
                              *args,
                              **kwargs)
        if use_cache:
            return out, None
        return out


class LinearAttention(ColumnLinear):

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache: bool = False,
                *args,
                **kwargs) -> Union[Tensor, Tuple[Tensor, None]]:
        out = super().forward(x=hidden_states,
                              lora_runtime_params=None,
                              lora_hidden_state=None)

        if use_cache:
            return out, None
        return out


class LinearFFN(ColumnLinear):

    def forward(self,
                hidden_states,
                lora_layer_params=None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        return super().forward(x=hidden_states,
                               lora_runtime_params=None,
                               lora_hidden_state=None)


NoOpFFN = NoOp
NoOpLayerNorm = NoOp


class DeciLMDecoderLayer(Module):

    def __init__(self, config: DeciConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        self.local_layer_idx = layer_idx - layers_range[0]

        self.layer_config = self.config.get_layer_config(self.layer_idx)

        self._init_attention()
        self._init_ffn()

    @property
    def input_layernorm_was_fused(self) -> bool:
        """
        The previous layer ran our input_layernorm for us if:
        1. The reduce_fusion plugin is enabled and
        2. We are not the first local model layer and
        3. The previous layer is an MLP layer
        """
        return default_net(
        ).plugin_config.reduce_fusion and self.local_layer_idx > 0 and self.config.get_layer_config(
            self.layer_idx -
            1).is_mlp_layer and self.needs_input_layernorm_fusion

    @property
    def needs_input_layernorm_fusion(self) -> bool:
        """
        This layer needs the previous layer to perform input_layernorm fusion if:
        1. The reduce_fusion plugin is enabled and
        2. This is not a NOOP attention layer (otherwise it has no input_layernorm)
        """
        return default_net(
        ).plugin_config.reduce_fusion and not self.layer_config.is_noop_attention_layer

    @property
    def can_fuse_post_layernorm(self) -> bool:
        """
        This layer can fuse attention and post_layernorm if:
        1. The reduce_fusion plugin is enabled and
        2. It is an attention layer and
        3. It is not a NOOP FFN layer (othrewise it has no post_layernorm)
        """
        return default_net(
        ).plugin_config.reduce_fusion and self.layer_config.is_attention_layer and not self.layer_config.is_noop_ffn_layer

    @property
    def can_fuse_input_layernorm(self) -> bool:
        """
        This layer can run the next layer's input_layernorm if:
        1. The reduce_fusion plugin is enable and
        2. It is an MLP layer
        """
        return default_net(
        ).plugin_config.reduce_fusion and self.layer_config.is_mlp_layer

    def _init_attention(self) -> None:
        """
        Initialize some attention alternative
        """
        # normal attention
        if self.layer_config.is_attention_layer:
            # according to recurrentgemma, len(layer_types) can be less than num_hidden_layers
            # in this case, the list should wrap-around
            # for example, if layer_types = ["attention", "recurrent", "recurrent"], and we have 5 layers, we get:
            # layer 0 ==> attention
            # layer 1 ==> recurrent
            # layer 2 ==> recurrent
            # layer 3 ==> attention
            # layer 4 ==> recurrent
            # we check which layers are local to our rank
            layers_range = self.config.mapping.pp_layers(
                self.config.num_hidden_layers)
            # then take the size of layer_types in the config
            layer_type_len = len(self.config.layer_types)
            # collect the layer types of all the local layers
            local_layer_types = [
                self.config.layer_types[layer_id % layer_type_len]
                for layer_id in layers_range
            ]
            # and see how many of them are attention layers to determine our local attention layer idx
            local_attn_layer_idx = local_layer_types[:self.
                                                     local_layer_idx].count(
                                                         "attention")

            # Iterate over all local layer configs, getting num_kv_heads of the attention ones
            num_kv_heads_per_local_layer = [
                layer_config.attention.num_key_value_heads for layer_config in
                [self.config.layer_configs[idx] for idx in layers_range]
                if layer_config.is_attention_layer
            ]

            # adjust num heads according to tp size
            num_kv_heads_per_local_layer = [
                (nheads + self.config.mapping.tp_size - 1) //
                self.config.mapping.tp_size
                for nheads in num_kv_heads_per_local_layer
            ]
            nheads_tp = (self.layer_config.attention.num_key_value_heads +
                         self.config.mapping.tp_size -
                         1) // self.config.mapping.tp_size

            self.input_layernorm = RmsNorm(
                normalized_shape=self.config.hidden_size,
                eps=self.config.norm_epsilon,
                dtype=self.config.dtype,
            )

            self.attention = Attention(
                local_layer_idx=local_attn_layer_idx,
                hidden_size=self.config.hidden_size,
                attention_head_size=self.config.head_size,
                num_attention_heads=self.config.num_attention_heads,
                num_kv_heads=self.layer_config.attention.num_key_value_heads,
                max_position_embeddings=self.config.max_position_embeddings,
                dtype=self.config.dtype,
                attention_mask_type=AttentionMaskType.causal,
                bias=False,
                position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                rotary_embedding_base=self.config.rotary_base,
                rotary_embedding_scaling=self.config.rotary_scaling,
                tp_group=self.config.mapping.tp_group,
                tp_size=self.config.mapping.tp_size,
                tp_rank=self.config.mapping.tp_rank,
                quant_mode=self.config.quant_mode)

        elif self.layer_config.is_noop_attention_layer:
            self.input_layernorm = NoOpLayerNorm()
            self.attention = NoOpAttention()

        elif self.layer_config.is_linear_attention_layer:
            self.input_layernorm = RmsNorm(
                normalized_shape=self.config.hidden_size,
                eps=self.config.norm_epsilon,
                dtype=self.config.dtype,
            )

            self.attention = LinearAttention(
                in_features=self.config.hidden_size,
                out_features=self.config.hidden_size,
                bias=False,
                dtype=self.config.dtype,
                tp_group=self.config.mapping.tp_group,
                tp_size=self.config.mapping.tp_size,
                gather_output=True)

        else:
            raise NotImplementedError(
                f"Attention of type {str(self.layer_config.attention.impl)} is not implemented"
            )

    def _init_ffn(self) -> None:
        """
        Initialize some ffn alternative
        """

        if self.layer_config.is_mlp_layer:
            intermediate_size = self.layer_config.ffn.intermediate_size or self.config.intermediate_size
            mlp_hidden_size = intermediate_size or self.config.hidden_size * 4

            self.post_layernorm = RmsNorm(
                normalized_shape=self.config.hidden_size,
                eps=self.config.norm_epsilon,
                dtype=self.config.dtype,
            )

            self.ffn = GatedMLP(
                hidden_size=self.config.hidden_size,
                ffn_hidden_size=mlp_hidden_size,
                hidden_act=self.config.hidden_act,
                bias=False,
                dtype=self.config.dtype,
                tp_group=self.config.mapping.tp_group,
                tp_size=self.config.mapping.tp_size,
                quant_mode=self.config.quant_mode,
            )

        elif self.layer_config.is_noop_ffn_layer:
            self.post_layernorm = NoOpLayerNorm()
            self.ffn = NoOpFFN()

        elif self.layer_config.is_linear_ffn_layer:
            self.post_layernorm = RmsNorm(
                normalized_shape=self.config.hidden_size,
                eps=self.config.norm_epsilon,
                dtype=self.config.dtype,
            )

            self.ffn = LinearFFN(in_features=self.config.hidden_size,
                                 out_features=self.config.hidden_size,
                                 bias=False,
                                 dtype=self.config.dtype,
                                 tp_group=self.config.mapping.tp_group,
                                 tp_size=self.config.mapping.tp_size,
                                 gather_output=True)

        else:
            raise NotImplementedError(
                f"FFN of type {str(self.layer_config.ffn.impl)} is not implemented"
            )

    def forward(self,
                hidden_states: Tensor | Tuple[Tensor, Tensor],
                attention_mask: Optional[Tensor] = None,
                use_cache: bool = False,
                spec_decoding_params=None,
                kv_cache_params: Optional[KeyValueCacheParams] = None,
                attention_params: Optional[AttentionParams] = None,
                lora_layer_params: Optional[LoraParams] = None,
                next_layer_input_layernorm_args: Optional[Tuple[Tensor,
                                                                float]] = None):
        if self.input_layernorm_was_fused:
            # previous layer already performed our layer norm
            assert isinstance(hidden_states, tuple)
            hidden_states, residual = hidden_states
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        if self.can_fuse_post_layernorm:
            all_reduce_params = AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=self.post_layernorm.weight.value,
                eps=self.post_layernorm.eps)
        else:
            all_reduce_params = None

        attention_output = self._run_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
            all_reduce_params=all_reduce_params)

        if use_cache:
            attention_output, present_kv = attention_output
        else:
            present_kv = None

        if self.can_fuse_post_layernorm:
            hidden_states, residual = attention_output
        else:
            hidden_states = residual + attention_output
            residual = hidden_states
            hidden_states = self.post_layernorm(hidden_states)

        if next_layer_input_layernorm_args is not None:
            assert self.can_fuse_input_layernorm
            norm_weight, eps = next_layer_input_layernorm_args
            all_reduce_params = AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=norm_weight,
                eps=eps)
            hidden_states = self._run_ffn(hidden_states,
                                          lora_layer_params=lora_layer_params,
                                          all_reduce_params=all_reduce_params)

        else:
            hidden_states = self._run_ffn(hidden_states,
                                          lora_layer_params=lora_layer_params)
            hidden_states = residual + hidden_states

        return DeciLMLayerOutput(hidden_states=hidden_states,
                                 present_kv=present_kv)

    def _run_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        spec_decoding_params=None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        lora_layer_params: Optional[LoraParams] = None,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> Union[Tensor, Tuple[Tensor, None]]:
        """
        Ideally, this functionality would be encapsulated in a LinearAttention class, but during
        FP8 and lower quantization, our linear classes get overrun by ModelOpt, thus we must
        control the attention inputs at the DecoderLayer level.
        """
        if self.layer_config.is_linear_attention_layer:
            out = self.attention(hidden_states)
            return out, None if use_cache else out
        else:
            if not self.layer_config.is_attention_layer:
                assert all_reduce_params is None, f"Layer with attention of type {self.layer_config.attention.impl} can't do reduce_fusion"

            return self.attention(hidden_states=hidden_states,
                                  attention_mask=attention_mask,
                                  use_cache=use_cache,
                                  spec_decoding_params=spec_decoding_params,
                                  kv_cache_params=kv_cache_params,
                                  attention_params=attention_params,
                                  lora_layer_params=lora_layer_params,
                                  all_reduce_params=all_reduce_params)

    def _run_ffn(self,
                 hidden_states,
                 lora_layer_params=None,
                 all_reduce_params: Optional[AllReduceParams] = None):
        """
        Ideally, this functionality would be encapsulated in a LinearMLP class, but during
        FP8 and lower quantization, our linear classes get overrun by ModelOpt, thus we must
        control the MLP inputs at the DecoderLayer level.
        """
        if all_reduce_params is not None:
            assert self.layer_config.is_mlp_layer, f"Layer with FFN of type {self.layer_config.ffn.impl} can't do reduce_fusion"

        if self.layer_config.is_linear_ffn_layer:
            return self.ffn(hidden_states)
        else:
            return self.ffn(hidden_states,
                            lora_layer_params=lora_layer_params,
                            all_reduce_params=all_reduce_params)


class DeciLMDecoderLayerList(ModuleList):

    def __init__(self, cls: Type[DeciLMDecoderLayer], config: DeciConfig):
        self.num_hidden_layers = config.num_hidden_layers
        # global indices of local layers
        self.layer_list = config.mapping.pp_layers(config.num_hidden_layers)
        super().__init__([cls(config, idx) for idx in self.layer_list])
        # global indices of local attention layers
        self.attention_layer_list = [
            self.layer_list[i] for i, layer in enumerate(self)
            if layer.layer_config.is_attention_layer
        ]

    def forward(
        self,
        hidden_states: Tensor,
        use_cache: bool,
        attention_mask: Optional[Tensor],
        kv_cache_params: KeyValueCacheParams,
        attention_params: Optional[AttentionParams] = None,
        position_ids: Optional[Tensor] = None,
        lora_params: Optional[LoraParams] = None,
        spec_decoding_params: Optional[SpecDecodingParams] = None,
    ) -> DeciLMLayerListOutput:
        kv_cache_params.fill_none_tensor_list(len(self.layer_list))

        presents = []

        # put None where we don't have attention layers
        pkv_iter = iter(kv_cache_params.past_key_value)

        past_key_values = [x for x in pkv_iter]

        for layer_idx, (layer, past) in enumerate(zip(self, past_key_values)):
            next_layer_input_layernorm_args = None
            if default_net().plugin_config.reduce_fusion:
                if layer_idx < self.layer_list[-1]:
                    # this is not the last layer
                    next_layer = self[layer_idx + 1]
                    if layer.can_fuse_input_layernorm and next_layer.needs_input_layernorm_fusion:
                        # this layer can fuse the next layer's input_layernorm
                        next_layer_input_layernorm_args = (
                            next_layer.input_layernorm.weight.value,
                            next_layer.input_layernorm.eps)

            layer_out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attention_params=attention_params,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=kv_cache_params.
                    host_max_attention_window_sizes,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    kv_cache_block_offsets=kv_cache_params.
                    kv_cache_block_offsets,
                    host_kv_cache_block_offsets=kv_cache_params.
                    host_kv_cache_block_offsets,
                    host_kv_cache_pool_pointers=kv_cache_params.
                    host_kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=kv_cache_params.
                    host_kv_cache_pool_mapping,
                    cache_indirection=kv_cache_params.cache_indirection,
                ),
                spec_decoding_params=spec_decoding_params,
                use_cache=use_cache,
                lora_layer_params=lora_params.get_layer_config(layer_idx)
                if lora_params is not None
                and lora_params.lora_ranks is not None else None,
                next_layer_input_layernorm_args=next_layer_input_layernorm_args)

            hidden_states = layer_out.hidden_states
            if use_cache and layer_out.present_kv is not None:
                presents.append(layer_out.present_kv)

        return DeciLMLayerListOutput(hidden_states=hidden_states,
                                     present_kvs=presents)


class DeciLMModel(Module):

    def __init__(self, config: DeciConfig) -> None:
        super().__init__()
        init_all_reduce_helper()

        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            # first rank in pipeline-parallel handles token embedding
            assert config.vocab_size is not None
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.position_embedding_type = config.position_embedding_type
        self.layers = DeciLMDecoderLayerList(DeciLMDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            # last rank in pipeline-parallel handles final norm
            self.ln_f = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype,
            )

    def _vocab_embedding(self,
                         input_ids: Tensor,
                         prompt_embedding_table: Optional[Tensor] = None,
                         prompt_tasks: Optional[Tensor] = None,
                         prompt_vocab_size: Optional[Tensor] = None) -> Tensor:
        # prompt tuning
        ptuning_args = ([
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else [])

        hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        return hidden_states

    def forward(
        self,
        input_ids,
        position_ids=None,
        use_cache: bool = False,
        attention_mask: Optional[Tensor] = None,
        spec_decoding_params=None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        hidden_states: Optional[Tensor] = None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        lora_params: Optional[LoraParams] = None,
    ) -> DeciLMLayerListOutput:

        if self.mapping.is_first_pp_rank():
            # first pipeline rank ==> do prompt embedding
            hidden_states = self._vocab_embedding(
                input_ids=input_ids,
                prompt_embedding_table=prompt_embedding_table,
                prompt_tasks=prompt_tasks,
                prompt_vocab_size=prompt_vocab_size)
        else:
            # receive hidden states from prior rank in the pipeline
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        layers_out = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_params=lora_params,
            spec_decoding_params=spec_decoding_params,
        )

        if self.mapping.is_last_pp_rank():
            # last pipeline rank ==> do final norm
            hidden_states = self.ln_f(layers_out.hidden_states)
        else:
            # send hidden states to next rank in the pipeline
            hidden_states = send(layers_out.hidden_states,
                                 self.mapping.next_pp_rank())

        return DeciLMLayerListOutput(hidden_states=hidden_states,
                                     present_kvs=layers_out.present_kvs)


class DeciLMForCausalLM(DecoderModelForCausalLM):
    config_class = DeciConfig

    def __init__(self, config: DeciConfig):

        transformer = DeciLMModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        if config.mapping.is_last_pp_rank():
            # last pipeline rank needs to do calculate logits
            lm_head = ColumnLinear(
                config.hidden_size,
                vocab_size_padded,
                bias=False,
                dtype=config.dtype,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                gather_output=True,
            )
        else:
            lm_head = None
        super().__init__(config, transformer, lm_head)

        # Create constant attention parameters to be reused by all layers.
        Attention.create_attention_const_params(self, config)
        self.position_embedding_type = config.position_embedding_type

    @classmethod
    def from_hugging_face(cls,
                          hf_model_or_dir: Union[
                              str, 'transformers.PreTrainedModel'],
                          dtype: str = 'auto',
                          mapping: Optional[Mapping] = None,
                          quant_config: Optional[QuantConfig] = None,
                          load_by_shard: bool = False,
                          load_model_on_cpu: bool = False,
                          trust_remote_code: bool = True,
                          **kwargs) -> "DeciLMForCausalLM":
        import transformers

        # TODO(oargov): add support for these
        assert not load_by_shard, "load_by_shard is not implemented yet"

        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_config_or_dir = hf_model_or_dir.config
        else:
            hf_config_or_dir = hf_model_or_dir

        config = DeciConfig.from_hugging_face(
            hf_config_or_dir,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            trust_remote_code=trust_remote_code,
            **kwargs)

        if use_preloading:
            assert not load_by_shard
            weights = load_weights_from_hf_model(hf_model_or_dir, config)
        elif has_safetensors(
                hf_model_or_dir) and not config.quant_mode.has_any_quant():
            weights = load_weights_from_hf_safetensors(hf_model_or_dir, config)
        else:
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                hf_model_or_dir,
                device_map='auto' if not load_model_on_cpu else 'cpu',
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            weights = load_weights_from_hf_model(hf_model, config)
        preprocess_weights(weights, config)

        model = DeciLMForCausalLM(config)
        model.load(weights)
        return model

    @classmethod
    def from_checkpoint(cls,
                        ckpt_dir: str,
                        rank: Optional[int] = None,
                        config: Optional["PretrainedConfig"] = None):
        return super().from_checkpoint(
            ckpt_dir,
            rank,
            config,
            preprocess_weights_hook=
            update_weights_following_modelopt_optimization,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        use_cache: bool = False,
        last_token_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        hidden_states: Optional[Tensor] = None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        lora_params: Optional[LoraParams] = None,
        spec_decoding_params=None,
    ):
        # fill attention params.
        attention_params = Attention.fill_attention_params(
            self, attention_params)

        model_out = self.transformer.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_params=lora_params,
            hidden_states=hidden_states,
            prompt_embedding_table=prompt_embedding_table,
            prompt_tasks=prompt_tasks,
            prompt_vocab_size=prompt_vocab_size,
            spec_decoding_params=spec_decoding_params)
        hidden_states = model_out.hidden_states

        if self.config.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states,
                last_token_ids,
                default_net().plugin_config.remove_input_padding,
            )

            lm_logits = self.lm_head(hidden_states)
            lm_logits.mark_output("logits", self.config.logits_dtype)
        else:
            hidden_states.mark_output("hidden_states_output", self.config.dtype)

        if use_cache and not default_net().plugin_config.paged_kv_cache:
            presents = model_out.present_kvs
            for i, present in zip(self.transformer.layers.attention_layer_list,
                                  presents):
                present.mark_output(f"present_key_value_{i}",
                                    self.config.kv_dtype)
            if self.config.mapping.is_last_pp_rank():
                return (lm_logits, presents, hidden_states)
            return (hidden_states, presents)
        else:
            if self.config.mapping.is_last_pp_rank():
                return lm_logits, hidden_states
            return hidden_states

    def prepare_attention_inputs(
            self,
            *,
            max_batch_size: int,
            max_beam_width: int,
            max_input_len: int,
            max_seq_len: int,
            num_kv_heads: int,
            head_size: int,
            num_layers: int,
            kv_dtype: str,
            kv_cache_type: KVCacheType,
            num_profiles: int = 1,
            enable_ctx_gen_opt_profiles: bool = False,
            remove_input_padding: bool = False,
            use_gpt_attention_plugin: bool = False,
            paged_kv_cache: bool = False,
            tokens_per_block: int = 32,
            mapping: Mapping = Mapping(),
            use_cache: bool = True,
            streamingllm: bool = False,
            attn_layer_idx: Optional[List[int]] = None,
            opt_batch_size: Optional[int] = None,
            num_kv_heads_per_layer: Optional[List[int]] = None):

        if attn_layer_idx is None:
            attn_layer_idx, num_kv_heads_per_layer = [], []
            for layer_idx in range(self.config.num_hidden_layers):
                layer_config = self.config.get_layer_config(layer_idx)
                if layer_config.is_attention_layer:
                    attn_layer_idx.append(layer_idx)
                    num_kv_heads_per_layer.append(
                        layer_config.attention.num_key_value_heads)

        attention_inputs = super().prepare_attention_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            num_layers=num_layers,
            kv_dtype=kv_dtype,
            num_profiles=num_profiles,
            kv_cache_type=kv_cache_type,
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            tokens_per_block=tokens_per_block,
            mapping=mapping,
            streamingllm=streamingllm,
            attn_layer_idx=attn_layer_idx,
            opt_batch_size=opt_batch_size,
            num_kv_heads_per_layer=num_kv_heads_per_layer)

        return attention_inputs
