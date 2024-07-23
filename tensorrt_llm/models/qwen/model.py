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

from typing import Optional, Union

from tensorrt_llm.lora_manager import LoraConfig, use_lora

from ..._utils import pad_vocab_size
from ...functional import Tensor, recv, send, sigmoid
from ...layers import (MLP, MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, GatedMLP, RmsNorm, RowLinear)
from ...mapping import Mapping
from ...module import Module
from ...quantization import W8A8_SQ_PLUGIN_LIST, QuantAlgo
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig, check_share_embedding)
from .config import QWenConfig
from .convert import (load_hf_qwen, load_weights_from_hf_gptq_model,
                      load_weights_from_hf_model)


class QWenDecoderLayer(Module):

    def __init__(self, config: QWenConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        dtype = config.dtype
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode,
            dense_bias=False)

        ClsMLP = GatedMLP
        mlp_kwargs = {}
        if config.moe.has_moe():
            ClsMLP = MOE
            mlp_kwargs = {
                "moe_config": config.moe,
                "mapping": config.mapping,
            }

        if config.qwen_type == 'qwen2_moe':
            self.shared_expert = MLP(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.moe_shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                dtype=dtype,
                bias=False,
                tp_group=tp_group,
                tp_size=tp_size,
                quant_mode=config.quant_mode)
            self.shared_expert_gate = RowLinear(config.hidden_size,
                                                1,
                                                bias=False,
                                                dtype=dtype,
                                                tp_group=None,
                                                tp_size=1)

        # Qwen's real inter_size depends on qwen_type
        if self.config.qwen_type == 'qwen':
            intermediate_size = config.intermediate_size // 2
        elif self.config.qwen_type == 'qwen2_moe':
            intermediate_size = config.moe_intermediate_size
        else:
            intermediate_size = config.intermediate_size

        self.mlp = ClsMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=intermediate_size,
                          hidden_act=config.hidden_act,
                          dtype=dtype,
                          bias=config.mlp_bias,
                          tp_group=tp_group,
                          tp_size=tp_size,
                          quant_mode=config.quant_mode,
                          **mlp_kwargs)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
        lora_layer_params=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
        )
        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states

        hidden_states = self.post_layernorm(hidden_states)

        shared_output = None
        if self.config.qwen_type == 'qwen2_moe':
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = sigmoid(
                    self.shared_expert_gate(hidden_states)) * shared_output

        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)

        if shared_output is not None:
            hidden_states = hidden_states + shared_output

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class QWenModel(Module):

    def __init__(self, config: QWenConfig) -> None:
        super().__init__()
        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(QWenDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None):

        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers.forward(hidden_states,
                                            use_cache=use_cache,
                                            attention_mask=attention_mask,
                                            kv_cache_params=kv_cache_params,
                                            attention_params=attention_params,
                                            lora_params=lora_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class QWenForCausalLM(DecoderModelForCausalLM):
    config_class = QWenConfig

    def __init__(self, config: QWenConfig):
        transformer = QWenModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        if config.qwen_type == 'qwen':
            self.trtllm_modules_to_hf_modules = {
                "attn_qkv": "c_attn",
                "attn_dense": "attn.c_proj",
                "mlp_h_to_4h": "w2",
                "mlp_4h_to_h": "mlp.c_proj",
                "mlp_gate": "w1",
            }
        else:
            self.trtllm_modules_to_hf_modules = None
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            use_hf_gptq_checkpoint=False,
            **kwargs):
        ''' Create a QWenForCausalLM object from give parameters
        '''
        import transformers

        load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = QWenConfig.from_hugging_face(hf_config_or_dir,
                                              dtype=dtype,
                                              mapping=mapping,
                                              quant_config=quant_config,
                                              **kwargs)

        if not use_preloading:
            hf_model = load_hf_qwen(hf_model_dir, load_model_on_cpu)
        if use_hf_gptq_checkpoint:
            weights = load_weights_from_hf_gptq_model(hf_model, config)
        else:
            weights = load_weights_from_hf_model(hf_model, config)

        check_share_embedding(weights, config)
        model = QWenForCausalLM(config)
        model.load(weights)
        return model

    def default_plugin_config(self, **kwargs):
        plugin_config = super().default_plugin_config(**kwargs)
        if self.quant_mode.is_int4_weight_only_per_group():
            plugin_config.weight_only_groupwise_quant_matmul_plugin = 'auto'
        return plugin_config

    @classmethod
    def quantize(
        cls,
        hf_model_dir: str,
        output_dir: str,
        dtype: str = 'auto',
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        *,
        calib_dataset='cnn_dailymail',
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=512,
        random_seed=1234,
        tokenizer_max_seq_length=2048,
        **kwargs,
    ):
        DEFAULT_MODELOPT_FLOW = [
            QuantAlgo.W4A16_AWQ, QuantAlgo.FP8, QuantAlgo.W8A8_SQ_PER_CHANNEL,
            QuantAlgo.W4A8_AWQ
        ]
        config = QWenConfig.from_hugging_face(hf_model_dir,
                                              dtype=dtype,
                                              mapping=mapping,
                                              quant_config=quant_config,
                                              **kwargs)

        if quant_config.quant_algo in DEFAULT_MODELOPT_FLOW:
            super().quantize(hf_model_dir,
                             output_dir,
                             dtype=config.dtype,
                             mapping=config.mapping,
                             quant_config=config.quantization,
                             calib_dataset=calib_dataset,
                             calib_batches=calib_batches,
                             calib_batch_size=calib_batch_size,
                             calib_max_seq_length=calib_max_seq_length,
                             random_seed=random_seed,
                             tokenizer_max_seq_length=tokenizer_max_seq_length)
        else:
            # non-modelopt, the legacy TRT-LLM native quantization algorithm:
            # sq, int4/int8 weights only, int8 kv cache
            NATIVE_QUANT_FLOW = [QuantAlgo.W4A16, QuantAlgo.W8A16, None
                                 ] + W8A8_SQ_PLUGIN_LIST
            is_valid_native_quant = (quant_config.quant_algo in NATIVE_QUANT_FLOW) and \
                (quant_config.kv_cache_quant_algo in [QuantAlgo.INT8, None])
            assert quant_config.quant_algo is not None or quant_config.kv_cache_quant_algo is not None, \
                "There is no point to call the quantize function if both quant_algo and kv_cache_quant_algo is None"
            assert is_valid_native_quant, f"Internal error: shall call Modelopt for this quantization {quant_config}"

            from . import convert
            convert.quantize(hf_model_dir,
                             output_dir,
                             config=config,
                             calib_dataset=calib_dataset)

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)
