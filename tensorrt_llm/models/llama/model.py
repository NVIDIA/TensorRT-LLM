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
from typing import List, Optional

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (RotaryScalingType, Tensor, gather_last_token_logits,
                           recv, send)
from ...layers import (MOE, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, FusedGatedMLP, GatedMLP,
                       KeyValueCacheParams, LoraParams, MoeConfig,
                       PositionEmbeddingType, PromptTuningEmbedding, RmsNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode
from ...quantization.layers import FP8Linear, FP8RowLinear
from ...top_model_mixin import TopModelMixin
from ..generation_mixin import GenerationMixin
from ..modeling_utils import PretrainedConfig


class LLaMADecoderLayer(Module):

    def __init__(self,
                 layer_id,
                 hidden_size,
                 num_attention_heads,
                 num_kv_heads=None,
                 max_position_embeddings=2048,
                 dtype=None,
                 attention_mask_type=AttentionMaskType.causal,
                 hidden_act='silu',
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 rotary_base=10000.0,
                 rotary_scaling=None,
                 mlp_hidden_size=None,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 use_auto_parallel=False,
                 quant_mode=QuantMode(0),
                 rms_norm_eps=1e-06,
                 attn_bias=False,
                 mlp_bias=False,
                 use_fused_mlp=False,
                 enable_pos_shift=False,
                 dense_context_fmha=False,
                 moe_config: MoeConfig = MoeConfig()):
        super().__init__()
        self._layer_id = layer_id  # useful for debugging
        # used for quantizing model
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.mlp_hidden_size = mlp_hidden_size
        self.attention_mask_type = attention_mask_type
        self.position_embedding_type = position_embedding_type
        self.input_layernorm = RmsNorm(normalized_shape=hidden_size,
                                       eps=rms_norm_eps,
                                       dtype=dtype)
        self.mlp_bias = mlp_bias
        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=attn_bias,
            position_embedding_type=position_embedding_type,
            rotary_embedding_base=rotary_base,
            rotary_embedding_scaling=rotary_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            use_auto_parallel=use_auto_parallel,
            quant_mode=quant_mode,
            instance_id=2 * layer_id,
            enable_pos_shift=enable_pos_shift,
            dense_context_fmha=dense_context_fmha,
        )
        if not mlp_hidden_size:
            self.mlp_hidden_size = hidden_size * 4

        ClsMLP = GatedMLP
        mlp_kwargs = {}
        if moe_config.has_moe():
            ClsMLP = MOE
            mlp_kwargs = {
                "moe_config": moe_config,
                "tp_rank": tp_rank,
            }
        elif use_fused_mlp:
            ClsMLP = FusedGatedMLP
        self.mlp = ClsMLP(hidden_size=hidden_size,
                          ffn_hidden_size=self.mlp_hidden_size,
                          hidden_act=hidden_act,
                          dtype=dtype,
                          bias=mlp_bias,
                          tp_group=tp_group,
                          tp_size=tp_size,
                          quant_mode=quant_mode,
                          instance_id=2 * layer_id + 1,
                          **mlp_kwargs)
        self.post_layernorm = RmsNorm(normalized_shape=hidden_size,
                                      eps=rms_norm_eps,
                                      dtype=dtype)

    def forward(self,
                hidden_states,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                all_reduce_workspace=None,
                lora_layer_params=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self._layer_id == 0:
            self.register_network_output(f"norm0", hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params,
                                          workspace=all_reduce_workspace,
                                          lora_layer_params=lora_layer_params)

        if use_cache:
            attention_output, presents = attention_output
        if self._layer_id == 0:
            self.register_network_output(f"attn", attention_output)

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        if self._layer_id == 0:
            self.register_network_output(f"norm1", hidden_states)

        hidden_states = self.mlp(hidden_states,
                                 all_reduce_workspace,
                                 lora_layer_params=lora_layer_params)
        if self._layer_id == 0:
            self.register_network_output(f"mlp", hidden_states)

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class LLaMAModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 num_kv_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 mlp_hidden_size=None,
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 rotary_base=10000.0,
                 rotary_scaling=None,
                 mapping=Mapping(),
                 use_auto_parallel=False,
                 quant_mode=QuantMode(0),
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 rms_norm_eps=1e-06,
                 use_fused_mlp=False,
                 attn_bias=False,
                 mlp_bias=False,
                 moe_config: MoeConfig = MoeConfig(),
                 use_prompt_tuning: bool = False,
                 enable_pos_shift=False,
                 dense_context_fmha=False):
        super().__init__()
        self.mapping = mapping
        self.use_prompt_tuning = use_prompt_tuning

        EmbeddingCls = PromptTuningEmbedding if use_prompt_tuning else Embedding
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = EmbeddingCls(
                num_embeddings=vocab_size,
                embedding_dim=hidden_size,
                dtype=dtype,
                tp_size=mapping.tp_size if use_parallel_embedding else 1,
                tp_group=mapping.tp_group if use_parallel_embedding else None,
                sharding_dim=embedding_sharding_dim,
                tp_rank=mapping.tp_rank,
                instance_id=2 *
                num_layers,  # ids in [0, 2 * (num_layers - 1) + 1] already used
            )

        self.layers = ModuleList([
            LLaMADecoderLayer(
                layer_id=i,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_kv_heads=num_kv_heads,
                max_position_embeddings=max_position_embeddings,
                dtype=dtype,
                hidden_act=hidden_act,
                mlp_hidden_size=mlp_hidden_size,
                position_embedding_type=position_embedding_type,
                rotary_base=rotary_base,
                rotary_scaling=rotary_scaling,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank,
                use_auto_parallel=use_auto_parallel,
                quant_mode=quant_mode,
                rms_norm_eps=rms_norm_eps,
                attn_bias=attn_bias,
                mlp_bias=mlp_bias,
                use_fused_mlp=use_fused_mlp,
                enable_pos_shift=enable_pos_shift,
                dense_context_fmha=dense_context_fmha,
                moe_config=moe_config,
            ) for i in self.mapping.pp_layers(num_layers)
        ])

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=hidden_size,
                                eps=rms_norm_eps,
                                dtype=dtype)

    def forward(self,
                input_ids,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                all_reduce_workspace=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None):

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        ptuning_args = []
        if self.use_prompt_tuning:
            ptuning_args = [
                prompt_embedding_table, prompt_tasks, prompt_vocab_size
            ]
        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args,
                                                 all_reduce_workspace)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())
        self.register_network_output(f"embd", hidden_states)

        for layer_idx, (
                layer, past, pointer, host_pointer,
                max_attention_window_size) in enumerate(
                    zip(self.layers, kv_cache_params.past_key_value,
                        kv_cache_params.kv_cache_block_pointers,
                        kv_cache_params.host_kv_cache_block_pointers,
                        kv_cache_params.host_max_attention_window_sizes)):
            lora_layer_params = None
            if lora_params.lora_ranks is not None:
                lora_layer_params = lora_params.get_layer_params(layer_idx)

            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=max_attention_window_size,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    kv_cache_block_pointers=[pointer],
                    host_kv_cache_block_pointers=[host_pointer],
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params,
                all_reduce_workspace=all_reduce_workspace,
                lora_layer_params=lora_layer_params)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class LLaMAForCausalLM(LLaMAModel, GenerationMixin, TopModelMixin):

    def __init__(self,
                 num_layers,
                 num_heads,
                 num_kv_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 logits_dtype="float32",
                 mlp_hidden_size=None,
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 rotary_base=10000.0,
                 rotary_scaling=None,
                 mapping=Mapping(),
                 use_auto_parallel=False,
                 quant_mode=QuantMode(0),
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 rms_norm_eps=1e-06,
                 use_fused_mlp=False,
                 attn_bias=False,
                 mlp_bias=False,
                 moe_config=MoeConfig(),
                 use_prompt_tuning: bool = False,
                 enable_pos_shift=False,
                 dense_context_fmha=False):
        config = PretrainedConfig(
            architecture="LLaMAForCausalLM",
            dtype=dtype,
            logits_dtype=logits_dtype,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_act=hidden_act,
            intermediate_size=mlp_hidden_size,
            norm_epsilon=rms_norm_eps,
            position_embedding_type=str(position_embedding_type),
            world_size=mapping.world_size,
            tp_size=mapping.tp_size,
            pp_size=mapping.pp_size,
            quant_mode=quant_mode,
            quant_kwargs={},
            use_prompt_tuning=use_prompt_tuning)
        self.config = config
        # TODO: there is an issue of PretrainedConfig that it does not hold the info of "current rank"
        # it internally constructs a mapping object from the world_size/tp_size/pp_size
        # thus override the config.mapping here to the user provided one, which shall include current rank
        self.config.mapping = mapping

        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.dtype = dtype

        if isinstance(logits_dtype, str):
            self.logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self.logits_dtype = logits_dtype

        self.num_layers = num_layers
        self.num_heads = num_heads
        if num_kv_heads is None or num_kv_heads <= 0:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = mapping.tp_size

        self.kv_dtype = self.dtype
        if quant_mode.has_int8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('fp8')

        self.quant_mode = quant_mode
        self.use_parallel_embedding = use_parallel_embedding
        self.embedding_sharding_dim = embedding_sharding_dim
        self.moe_config = moe_config
        self.use_fused_mlp = use_fused_mlp

        super().__init__(num_layers, num_heads, num_kv_heads, hidden_size,
                         vocab_size, hidden_act, max_position_embeddings, dtype,
                         mlp_hidden_size, position_embedding_type, rotary_base,
                         rotary_scaling, mapping, use_auto_parallel, quant_mode,
                         use_parallel_embedding, embedding_sharding_dim,
                         rms_norm_eps, use_fused_mlp, attn_bias, mlp_bias,
                         moe_config, use_prompt_tuning, enable_pos_shift,
                         dense_context_fmha)

        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        if self.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(hidden_size,
                                        vocab_size_padded,
                                        bias=False,
                                        dtype=dtype,
                                        tp_group=mapping.tp_group,
                                        tp_size=mapping.tp_size,
                                        gather_output=True)

    def forward(self,
                input_ids,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                all_reduce_workspace=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None):
        hidden_states = super().forward(input_ids, position_ids, use_cache,
                                        attention_mask, kv_cache_params,
                                        attention_params, hidden_states,
                                        all_reduce_workspace,
                                        prompt_embedding_table, prompt_tasks,
                                        prompt_vocab_size, lora_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)

            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            lm_logits.mark_output('logits', self.logits_dtype)
        else:
            hidden_states.mark_output('hidden_states_output', self.dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in zip(self.mapping.pp_layers(self.num_layers),
                                  presents):
                present.mark_output(f'present_key_value_{i}', self.kv_dtype)
            if self.mapping.is_last_pp_rank():
                return (lm_logits, presents)
            return (hidden_states, presents)
        else:
            if self.mapping.is_last_pp_rank():
                return lm_logits
            return hidden_states

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_new_tokens,
                       use_cache,
                       max_beam_width,
                       max_num_tokens: int = None,
                       prompt_embedding_table_size: int = 0,
                       gather_all_token_logits: bool = False,
                       lora_target_modules: List[str] = None):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self.hidden_size // self.num_heads
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_custom_all_reduce = default_net(
        ).plugin_config.use_custom_all_reduce
        use_lora_plugin = default_net().plugin_config.lora_plugin

        model_inputs = self.prepare_basic_inputs(
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_new_tokens,
            self.num_kv_heads,
            head_size,
            self.num_layers,
            self.kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            use_custom_all_reduce=use_custom_all_reduce,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            dtype=self.dtype,
            num_heads=self.num_heads,
            mapping=self.mapping,
            max_num_tokens=max_num_tokens,
            prompt_embedding_table_size=prompt_embedding_table_size,
            gather_all_token_logits=gather_all_token_logits,
            use_lora_plugin=use_lora_plugin,
            lora_target_modules=lora_target_modules)

        return (
            model_inputs['input_ids'],
            model_inputs['position_ids'],
            True,
            model_inputs['last_token_ids'],
            model_inputs['attention_mask'],
            KeyValueCacheParams(
                past_key_value=model_inputs['past_key_value'],
                host_past_key_value_lengths=model_inputs[
                    'host_past_key_value_lengths'],
                host_max_attention_window_sizes=model_inputs[
                    'host_max_attention_window_sizes'],
                host_sink_token_length=model_inputs['host_sink_token_length'],
                kv_cache_block_pointers=model_inputs[
                    'kv_cache_block_pointers_list'],
                host_kv_cache_block_pointers=model_inputs[
                    'host_kv_cache_block_pointers_list'],
                cache_indirection=model_inputs['cache_indirection'],
            ),
            AttentionParams(
                sequence_length=model_inputs['sequence_length'],
                context_lengths=model_inputs['context_lengths'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types']),
            model_inputs['hidden_states_input'],
            model_inputs['all_reduce_workspace'],
            model_inputs['prompt_embedding_table'],
            model_inputs['tasks'],
            model_inputs['prompt_vocab_size'],
            LoraParams(
                model_inputs['lora_ranks'],
                model_inputs['lora_weights_pointers'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types']),
        )

    @classmethod
    def from_hugging_face(cls,
                          hf_model_dir,
                          dtype='float16',
                          mapping: Optional[Mapping] = None,
                          **kwargs):
        # TODO: discuss this can be our own config object if needed, instead of from transformers
        # but why reinvent wheels. And we already direct depends on transformers lib
        import transformers
        from transformers import LlamaConfig

        from ... import profiler
        from ...runtime.lora_manager import LoraConfig
        from .weight import load_from_hf_llama
        cfg = LlamaConfig.from_pretrained(hf_model_dir)

        num_kv_heads = cfg.num_key_value_heads if hasattr(cfg, "num_key_value_heads") \
            else cfg.num_attention_heads
        if mapping is None:
            mapping = Mapping()
        tllm_llama = LLaMAForCausalLM(
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
            hidden_act=cfg.hidden_act,
            max_position_embeddings=cfg.max_position_embeddings,
            dtype=dtype,
            mlp_hidden_size=cfg.intermediate_size,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            mapping=mapping,
            rotary_base=getattr(cfg, 'rotary_base', 10000.0),
            rotary_scaling=getattr(cfg, 'rotary_scaling', None),
            rms_norm_eps=cfg.rms_norm_eps,
            # current load_from_hf_llama can read these, so need to set these here,
            # ideally these attributes shall be set after the from_hugging_face returned an object to user
            # since these attributes are not related to the hugging face model, they only affect the TRT-LLM module
            # the weights transformation or any model optimization shall be done outside from_hugging_face
            quant_mode=kwargs.get("quant_mode", QuantMode(0)),
            use_parallel_embedding=kwargs.get("use_parallel_embedding", False),
            embedding_sharding_dim=kwargs.get("embedding_sharding_dim", 0),
            use_fused_mlp=kwargs.get("use_fused_mlp", False),
            moe_config=kwargs.get("moe_config",
                                  MoeConfig()),  # load weights use this
            use_prompt_tuning=kwargs.get("use_prompt_tuning", False))

        # For debug purpose, skip weights loading to be faster
        if kwargs.get("skip_loading_weights", False):
            return tllm_llama

        hf_model = transformers.LlamaForCausalLM
        #TODO: support mixtral
        profiler.start("Loading weights from HF")
        hf_llama = hf_model.from_pretrained(
            hf_model_dir,
            device_map={
                "model": "cpu",
                "lm_head": "cpu",
                "embed_tokens": "cpu",
                "layers": "cpu",
                "norm": "cpu",
            },  # Load to CPU memory
            torch_dtype='auto',
        )
        load_from_hf_llama(
            tllm_llama,
            hf_llama,
            mapping=mapping,
            dtype=dtype,
            #TODO: these shall be outside from_hugging_face too.
            use_gemm_woq_plugin=kwargs.get("use_gemm_woq_plugin", False),
            lora_config=kwargs.get("lora_config", LoraConfig()),
        )
        profiler.stop("Loading weights from HF")
        del hf_llama
        return tllm_llama

    def quantize(self, quant_mode: QuantMode):
        '''Quantize the model, raise Exception when the quantization failed
        '''
        #TODO: support all quantized scheme
        if quant_mode.has_int8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('fp8')
        for layer_idx in range(len(self.layers)):
            decoder = self.layers[layer_idx]
            assert isinstance(
                decoder, LLaMADecoderLayer
            ), f"Unexpected layer type, expecting LLaMADecoderLayer, got {type(decoder)}"
            attention = decoder.attention
            assert isinstance(
                attention, Attention
            ), f"Unexpected layer type, expecting Attention, got {type(attention)}"

            # Code from the tensorrt_llm/layers/attention.py Attention.__init__()
            # TODO: can remove the duplicate later
            attention.quant_mode = quant_mode
            attention.use_int8_kv_cache = self.quant_mode.has_int8_kv_cache()
            if self.quant_mode.has_kv_cache_quant():
                attention.kv_orig_quant_scale = Parameter(shape=(1, ),
                                                          dtype='float32')
                attention.kv_quant_orig_scale = Parameter(shape=(1, ),
                                                          dtype='float32')
            if self.quant_mode.has_fp8_qdq():
                attention.qkv = FP8Linear(
                    self.hidden_size,
                    self.mapping.tp_size * self.num_attention_heads *
                    self.attention_head_size +
                    (2 * self.mapping.tp_size * self.num_attention_kv_heads *
                     self.attention_head_size),
                    bias=attention.
                    bias,  #WARNING: quantize does not support changing the bias/dtype after module ctor
                    dtype=attention.dtype,
                    tp_group=self.mapping.tp_group,
                    tp_size=self.mapping.tp_size,
                    gather_output=False)
                attention.dense = FP8RowLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=attention.bias,
                    dtype=attention.dtype,
                    tp_group=self.mapping.tp_group,
                    tp_size=self.mapping.tp_size,
                    instance_id=attention.instance_id)
            mlp = decoder.mlp
            assert isinstance(
                mlp, (GatedMLP, MOE,
                      FusedGatedMLP)), f"Unexpected type, got {type(mlp)}"
            mlp_kwargs = {}
            if self.moe_config:
                ClsMLP = MOE
                mlp_kwargs = {
                    "moe_config": self.moe_config,
                    "tp_rank": self.mapping.tp_rank,
                }
            elif self.use_fused_mlp:
                ClsMLP = FusedGatedMLP
            self.mlp = ClsMLP(hidden_size=self.hidden_size,
                              ffn_hidden_size=decoder.mlp_hidden_size,
                              hidden_act=decoder.hidden_act,
                              dtype=self.dtype,
                              bias=decoder.mlp_bias,
                              tp_group=self.mapping.tp_group,
                              tp_size=self.mapping.tp_size,
                              quant_mode=quant_mode,
                              instance_id=2 * layer_idx + 1,
                              **mlp_kwargs)
        self.quant_mode = quant_mode  # to_trt may use this
        return self

    # llama specific setters, user shall has the chance to change the module attributes after
    # from_hugging_face factory method created the model when these attributes is not included in the huggingface checkpoint

    def rotary_base(self, val):
        for decoder in self.layers:
            decoder.attention.rotary_embedding_base = val
        return self

    def rotary_scaling(self, scaling_type, factor):
        #TODO: what if there are some other behaviors triggered by the these changes?
        # should implement these assignment as setters of the Attention Module
        assert scaling_type in ("linear", "dynamic"), f"Got {scaling_type}"
        assert factor > 1.0, f"Got {factor}"
        for decoder in self.layers:
            decoder.attention.rotary_embedding_scale_type = RotaryScalingType.linear if scaling_type == "linear" else RotaryScalingType.dynamic
            decoder.attention.rotary_embedding_scale = factor
        return self
