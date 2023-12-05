# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tensorrt as trt
import numpy as np
from collections import OrderedDict

from tensorrt_llm.quantization import QuantMode

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt, trt_dtype_to_np
from ...functional import (gather_last_token_logits, recv, send, select, pow, repeat_interleave, 
                           layer_norm, group_norm, cast, matmul, slice, arange, concat,
                           silu, sigmoid, squared_relu, constant, padding_nd, Tensor)
from ...layers import (Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, GatedMLP, KeyValueCacheParams,
                       PositionEmbeddingType, RmsNorm, PromptTuningEmbedding)
from ...parameter import Parameter
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin

class RwkvSelfAttentionLayer(Module):
    def __init__(self, 
                  dim_att, 
                  num_head, 
                  hidden_size, 
                  dtype = None
                  ):
        super().__init__()
        self.dim_att = dim_att
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.rkv_dtype = self.kv_dtype = str_dtype_to_trt('float32')
        
        self.rxw = ColumnLinear(dim_att, hidden_size, dtype=dtype)
        self.kxw = ColumnLinear(dim_att, hidden_size, dtype=dtype)
        self.vxw = ColumnLinear(dim_att, hidden_size, dtype=dtype)
        self.gxw = ColumnLinear(dim_att, hidden_size, dtype=dtype)
        self.oxw = ColumnLinear(hidden_size, dim_att, dtype=dtype)
        
        self.ln_w = Parameter(shape=[hidden_size], dtype=dtype)
        self.ln_b = Parameter(shape=[hidden_size], dtype=dtype)
        self.lx_w = Parameter(shape=[hidden_size], dtype=str_dtype_to_trt('float32'))  # special case
        self.lx_b = Parameter(shape=[hidden_size], dtype=str_dtype_to_trt('float32'))  # special case
        
        self.k_mix = Parameter(shape=[hidden_size], dtype=dtype)
        self.v_mix = Parameter(shape=[hidden_size], dtype=dtype)
        self.r_mix = Parameter(shape=[hidden_size], dtype=dtype)
        self.g_mix = Parameter(shape=[hidden_size], dtype=dtype)
        
        self.t_decay = Parameter(shape=(num_head, dim_att // num_head), dtype=str_dtype_to_trt('float32')) # special case
        self.t_first = Parameter(shape=(num_head, dim_att // num_head), dtype=str_dtype_to_trt('float32')) # special case
        
    def forward(self, x, sx, s):
        xx = layer_norm(x, [x.shape[2]], weight=self.ln_w.value, bias=self.ln_b.value)
        constant_one = constant(np.ones(1, dtype=trt_dtype_to_np(self.dtype)))
        kx = xx * self.k_mix.value + sx * (constant_one - self.k_mix.value)
        vx = xx * self.v_mix.value + sx * (constant_one - self.v_mix.value)
        rx = xx * self.r_mix.value + sx * (constant_one - self.r_mix.value)
        gx = xx * self.g_mix.value + sx * (constant_one - self.g_mix.value)
        
        T = x.shape[-2] # query length
        H = self.num_head
        S = self.hidden_size // self.num_head
        
        r = cast(self.rxw(rx), self.rkv_dtype).view((T, H, S)).transpose(0, 1)
        k = cast(self.kxw(kx), self.rkv_dtype).view((T, H, S)).transpose(0, 1).transpose(1, 2)
        v = cast(self.vxw(vx), self.rkv_dtype).view((T, H, S)).transpose(0, 1)
        g = silu(self.gxw(gx))
        
        print("xx: ", xx.shape)
        print("&&&&&&&&&&&&&&  shape: ", x.shape)
        print(T, H, S)
        print(self.k_mix._value)
        print(self.k_mix._value.__class__)
        print(self.t_decay._value)
        print(self.t_decay._value.__class__)
        
        kv = matmul(k.transpose(-2, -1).view((H, T, S, 1)), v.view((H, T, 1, S)))
        print("kv shape: ", kv.shape)
        w = self.t_decay.value.view((H, S, 1))
        print("w shape: ", w.shape)
        ws = pow(w, T)
        print("ws shape: ", ws.shape)
        ind = arange(T-1, -1, self.dtype).view((1, 1, T))
        print("ind shape: ", ind.shape)
        w = pow(repeat_interleave(w, T, 2), ind)
        print("w shape: ", w.shape)
        wk = w.view((H, S, T))
        print("wk shape: ", wk.shape)
        u = self.t_first.value.view((H, S, 1))
        print("u shape: ", u.shape)
        w = concat([slice(w, [0, 0, 1], [w.size(0), w.size(1), w.size(2) - 1]), u], dim=2)
        print("w shape: ", w.shape)
        w = padding_nd(w, [0], [T])
        print("w shape: ", w.shape)
        w = repeat_interleave(w, T, dim=2)
        print("w shape: ", w.shape)
        w = slice(w, [0, 0, 0], [w.size(0), w.size(1), 2 * T - 1]).view((H, S, T, 2 * T - 1))
        print("w shape: ", w.shape)
        w = slice(w, [0, 0, 0, T - 1], [w.size(0), w.size(1), w.size(2), T]).view(H, S, 1, T, T)
        print("w shape: ", w.shape)
        w = repeat_interleave(w, S, dim=2).view((H * S * S, T, T)).transpose(1, 2)
        print("w shape: ", w.shape) 
        
        kv = kv.transpose(1, 2).transpose(2, 3).view((H * S, 1, T))
        out = matmul(kv, w).view((H * S * S, T)).transpose(0, 1).view((T, H, S, S))
        out = matmul(r.transpose(0, 1).view(T, H, 1, S), out).view((T, H, S))
        s = ws * s + matmul(k * wk, v)
        
        # # TODO(Rinne): this loop will impact on the performance, needs to be replaced by cuda kernel.
        # out = (r + constant_one).view((T, H, S))
        # for t in range(T):
        #     rt = r[:,t:t+1,:]
        #     kt = k[:,:,t:t+1]
        #     vt = v[:,t:t+1,:]
        #     at = matmul(kt, vt)
        #     out[t] = matmul(rt, (cast(self.t_first.value, self.dtype) * at + s)).view((H, S))
        #     s = at + self.t_decay.value * s
            
        out = out.view((T, H * S))
        out = group_norm(out, num_groups=H, weight=self.lx_w.value, bias=self.lx_b.value)
        out = cast(out, x.dtype) * g
        out = self.oxw(out)
        
        return x + out, select(xx, 1, -1).view((-1, xx.size(2))), s
        

class RwkvFeedForwardLayer(Module):
    def __init__(self, 
                  dim_ffn,  
                  num_head, 
                  hidden_size, 
                  dtype = None
                  ):
        super().__init__()
        self.dim_ffn = dim_ffn
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.dtype = dtype
        
        self.rxw = ColumnLinear(hidden_size, hidden_size, dtype=dtype)
        self.kxw = ColumnLinear(hidden_size, dim_ffn, dtype=dtype)
        self.vxw = ColumnLinear(dim_ffn, hidden_size, dtype=dtype)
        
        self.ln_w = Parameter(shape=[hidden_size], dtype=dtype)
        self.ln_b = Parameter(shape=[hidden_size], dtype=dtype)
        
        self.k_mix = Parameter(shape=[hidden_size], dtype=dtype)
        self.r_mix = Parameter(shape=[hidden_size], dtype=dtype)
        
    def forward(self, x, sx):
        xx = layer_norm(x, [x.shape[2]], weight=self.ln_w.value, bias=self.ln_b.value)
        constant_one = constant(np.ones(1, dtype=trt_dtype_to_np(self.dtype)))
        kx = xx * self.k_mix.value + sx * (constant_one - self.k_mix.value)
        rx = xx * self.r_mix.value + sx * (constant_one - self.r_mix.value)
        
        r = sigmoid(self.rxw(rx))
        vx = squared_relu(self.kxw(kx))
        out = r * self.vxw(vx)
        
        print("xx: ", xx.shape)
        return x + out, select(xx, 1, -1).view((-1, xx.size(2),))

class RwkvDecoderLayer(Module):
    def __init__(self, 
                  layer_id,
                  dim_att, 
                  dim_ffn, 
                  num_head, 
                  hidden_size, 
                  rescale_layer, 
                  dtype = None
                  ):
        super().__init__()
        self._layer_id = layer_id
        self.dim_att = dim_att
        self.dim_ffn = dim_ffn
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.rescale_layer = rescale_layer
        self.dtype = dtype
        
        self.attention = RwkvSelfAttentionLayer(self.dim_att, 
                                                self.num_head, 
                                                self.hidden_size, 
                                                self.dtype)
        self.feed_forward = RwkvFeedForwardLayer(self.dim_ffn, 
                                                self.num_head, 
                                                self.hidden_size, 
                                                self.dtype)
        
    def forward(self, 
                x, 
                input_states, 
                rnn_states, 
                output_states):
        x = cast(x, self.dtype)
        x, input_states, rnn_states = self.attention(x, input_states, rnn_states)
        print("<<<<<<<<<<<<<<<<<<<< input_states: ", input_states.shape)
        x, output_states = self.feed_forward(x, output_states)
        constant_two = constant(np.ones(1, dtype=trt_dtype_to_np(self.dtype)) + 1)
        if self.rescale_layer > 0:
            x /= constant_two
        
        print("<<<<<<<<<<<<<<<<<<<<**** ", x.shape)
        return x, input_states, rnn_states, output_states
            

class RwkvModel(Module):
    def __init__(self, 
                 num_layers, 
                 dim_att, 
                 dim_ffn, 
                 hidden_size, 
                 num_head,
                 rescale_layer,
                 vocab_size, 
                 dtype, 
                 mapping=Mapping(),
                 quant_mode=QuantMode(0),
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0, 
                 use_prompt_tuning: bool = False):
        super().__init__()
        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.dtype = dtype
            
        self.num_layers = num_layers
        self.dim_att = dim_att
        self.dim_ffn = dim_ffn
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.rescale_layer = rescale_layer
        self.vocab_size = vocab_size
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
        
        layer_modules = []
        index = 0
        for i in self.get_transformer_layers(self.mapping, num_layers):
            layer_modules.append(RwkvDecoderLayer(layer_id=i, 
                                                    dim_att=dim_att, 
                                                    dim_ffn=dim_ffn, 
                                                    num_head=num_head, 
                                                    hidden_size=hidden_size, 
                                                    rescale_layer=rescale_layer if index % rescale_layer == 0 else 0, 
                                                    dtype=dtype))
            index += 1
        
        self.layers = ModuleList(layer_modules)
        # self.embedding_weight = Parameter(shape=(hidden_size, vocab_size), dtype=dtype)
        self.ln_out_w = Parameter(shape=[hidden_size], dtype=dtype)
        self.ln_out_b = Parameter(shape=[hidden_size], dtype=dtype)
        self.head_layer = ColumnLinear(hidden_size, vocab_size, dtype=dtype)
        
        if self.mapping.is_last_pp_rank():
            # TODO(Rinne): add process for this case
            pass
        
    # TODO(Rinne): deal with batch decode
    def forward(
        self,
        input_ids,
        input_states=None,
        rnn_states=None,
        output_states=None,
        hidden_states=None, 
        all_reduce_workspace=None,
    ):
        # TODO(Rinne): cache, kv_cache, attention_mask, tuning, all_reduce_workspace
        
        ptuning_args = []
        if self.use_prompt_tuning:
            raise NotImplementedError
     
        hidden_states = self.vocab_embedding(input_ids, *ptuning_args,
                                                 all_reduce_workspace)
        # if self.mapping.is_first_pp_rank():
        #     hidden_states = self.vocab_embedding(input_ids, *ptuning_args,
        #                                          all_reduce_workspace)
        #     print('=====================================')
        #     print(input_ids.shape)
        #     print(hidden_states.shape)
        #     print('=====================================')
        # else:
        #     hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())
     
        # self.register_network_output(f"hidden_states", hidden_states)
        # self.register_network_output(f"input_states", input_states)
        # self.register_network_output(f"rnn_states", rnn_states)
        # self.register_network_output(f"output_states", output_states)
        
        print(">>>>>>>>>>>>> ", input_states.__class__)
        
        for i, layer in enumerate(self.layers):
            hidden_states, input_states[i], rnn_states[i], output_states[i] = \
                layer(hidden_states, input_states[i], rnn_states[i], output_states[i])
            print("::::::::::::::::::::::::: hidden states: ", hidden_states.shape)
        
        # TODO(Rinne): deal with strategy
        # hidden_states = select(hidden_states, 1, -1)
        
        print(":::::::::::::@@@:::::::::::: hidden states: ", hidden_states.shape)
        hidden_states = layer_norm(hidden_states, (self.hidden_size,), self.ln_out_w.value, self.ln_out_b.value)
        
        print(":::::::::$$$:::::::::::::::: hidden states: ", hidden_states.shape)
        # TODO(Rinne): deal with quantization here
        hidden_states = self.head_layer(hidden_states)
        
        print("::::::::******:::::::::::: hidden states: ", hidden_states.shape)

        # if self.mapping.is_last_pp_rank():
        #     raise NotImplementedError
        # else:
        #     hidden_states = send(hidden_states, self.mapping.next_pp_rank())
            
        return hidden_states, input_states, rnn_states, output_states
    
    def init_states(self):
        input_states = []
        rnn_states = []
        output_states = []
        for i in range(self.num_layers):
            input_states.append(Tensor(
                name=f'input_state{i}', 
                dtype=self.dtype, 
                shape=[self.hidden_size]
            ))
            rnn_states.append(Tensor(
                name=f'rnn_state{i}', 
                dtype=self.dtype, 
                shape=(self.num_head, self.dim_att // self.num_head, self.dim_att // self.num_head)
            ))
            output_states.append(Tensor(
                name=f'output_state{i}', 
                dtype=self.dtype, 
                shape=[self.hidden_size]
            ))
        return {
            'input_states': input_states,
            'rnn_states': rnn_states, 
            'output_states': output_states
        }
        
    def _prepare_states(self, 
                        max_batch_size,
                        max_beam_width,
                        remove_input_padding=False,
                        paged_kv_cache=False,
                        use_gpt_attention_plugin=False, 
                        use_gemm_plugin=False):
        enable_two_optimization_profiles = False
        if use_gpt_attention_plugin == False or use_gemm_plugin == False:
            use_in_flight_batching = use_gpt_attention_plugin and remove_input_padding and paged_kv_cache
            enable_two_optimization_profiles = not use_in_flight_batching
        
        bb_range_cxt = [1, (max_batch_size + 1) // 2, max_batch_size]
        bb_range_gen = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        
        if enable_two_optimization_profiles:
            bb_range = [bb_range_cxt, bb_range_gen]
        else:
            bb_range = [bb_range_gen]
        
        input_states = []
        rnn_states = []
        output_states = []
        for i in range(self.num_layers):
            input_states.append(Tensor(
                name=f'input_state{i}', 
                dtype=self.dtype, 
                shape=[-1, self.hidden_size], 
                dim_range=OrderedDict([
                    ('batch_size', bb_range),
                    ('embedding_length', [self.hidden_size, self.hidden_size]
                        if enable_two_optimization_profiles else [self.hidden_size])
                ])
            ))
            head_length = self.dim_att // self.num_head
            rnn_states.append(Tensor(
                name=f'rnn_state{i}', 
                dtype=self.dtype, 
                shape=(-1, self.num_head, head_length, head_length), 
                dim_range=OrderedDict([
                    ('batch_size', bb_range),
                    ('head_count', [self.num_head, self.num_head]
                        if enable_two_optimization_profiles else [self.num_head]), 
                    ('head_length0', [head_length, head_length]
                        if enable_two_optimization_profiles else [head_length]), 
                    ('head_length1', [head_length, head_length]
                        if enable_two_optimization_profiles else [head_length])
                ])
            ))
            output_states.append(Tensor(
                name=f'output_state{i}', 
                dtype=self.dtype, 
                shape=[-1, self.hidden_size], 
                dim_range=OrderedDict([
                    ('batch_size', bb_range),
                    ('embedding_length', [self.hidden_size, self.hidden_size]
                        if enable_two_optimization_profiles else [self.hidden_size])
                ])
            ))
        return {
            'input_states': input_states,
            'rnn_states': rnn_states, 
            'output_states': output_states
        }
    
    
class RwkvForCausalLM(RwkvModel, GenerationMixin):
    def __init__(self, 
                 num_layers, 
                 dim_att, 
                 dim_ffn, 
                 hidden_size, 
                 num_head, 
                 rescale_layer, 
                 vocab_size, 
                 dtype, 
                 logits_dtype="float32",
                 mapping=Mapping(), 
                 quant_mode=QuantMode(0), 
                 use_parallel_embedding=False, 
                 embedding_sharding_dim=0, 
                 use_prompt_tuning: bool = False):
        self.kv_dtype = dtype
        if quant_mode.has_int8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('fp8')

        if isinstance(logits_dtype, str):
            self.logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self.logits_dtype = logits_dtype
            
        self.quant_mode = quant_mode
        self.use_parallel_embedding = use_parallel_embedding
        self.embedding_sharding_dim = embedding_sharding_dim
        
        super().__init__(num_layers, 
                         dim_att, 
                         dim_ffn,
                         hidden_size, 
                         num_head, 
                         rescale_layer, 
                         vocab_size,
                         dtype, 
                         mapping, 
                         quant_mode, 
                         use_parallel_embedding, 
                         embedding_sharding_dim, 
                         use_prompt_tuning)
        
        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        if self.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(hidden_size,
                                        vocab_size_padded,
                                        bias=False,
                                        dtype=dtype,
                                        tp_group=mapping.tp_group,
                                        tp_size=mapping.tp_size,
                                        gather_output=True)
            
    def forward(
        self, 
        input_ids,
        last_token_ids=None, 
        input_states=None,
        rnn_states=None,
        output_states=None,
        hidden_states=None, 
        all_reduce_workspace=None
    ):
        hidden_states, input_states, rnn_states, output_states = \
            super().forward(input_ids, input_states, rnn_states, output_states, 
                                     hidden_states, all_reduce_workspace)
        
        
        print("hidden states: ", hidden_states.shape)
        print("input_ids: ", input_ids.shape)
        print("last_token_ids: ", last_token_ids.shape)
        # TODO(Rinne): deal with cache
        if self.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)

            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            lm_logits.mark_output('logits', self.logits_dtype)
        else:
            hidden_states.mark_output('embeddings_output', self.dtype)
            
        if self.mapping.is_last_pp_rank():
            return lm_logits
        return hidden_states, input_states, rnn_states, output_states

     
    def prepare_inputs(
        self, 
        max_batch_size,
        max_input_len,
        max_new_tokens,
        max_beam_width,
        max_num_tokens: int = None,
        prompt_embedding_table_size: int = 0,
    ):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''
        
        head_size = self.hidden_size // self.num_head
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_custom_all_reduce = default_net(
        ).plugin_config.use_custom_all_reduce

        
        model_inputs = self.prepare_basic_inputs(
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_new_tokens,
            self.num_head, ## ??? need to be checked
            head_size, ## ??? need to be checked
            self.num_layers,
            self.kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            use_custom_all_reduce=use_custom_all_reduce,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            dtype=self.dtype,
            num_heads=self.num_head,
            mapping=self.mapping,
            max_num_tokens=max_num_tokens,
            prompt_embedding_table_size=prompt_embedding_table_size,
        )
        
        model_states = self._prepare_states(
            max_batch_size, 
            max_beam_width, 
            remove_input_padding, 
            paged_kv_cache, 
            use_gpt_attention_plugin, 
            use_gemm_plugin
        )
        
        return (
            model_inputs['input_ids'],
            # model_inputs['position_ids'],
            # True,
            model_inputs['last_token_ids'],
            model_states['input_states'], 
            model_states['rnn_states'], 
            model_states['output_states'], 
            # model_inputs['attention_mask'],
            # KeyValueCacheParams(
            #     past_key_value=model_inputs['past_key_value'],
            #     host_past_key_value_lengths=model_inputs[
            #         'host_past_key_value_lengths'],
            #     host_max_kv_cache_lengths=model_inputs[
            #         'host_max_kv_cache_lengths'],
            #     kv_cache_block_pointers=model_inputs[
            #         'kv_cache_block_pointers_list'],
            #     cache_indirection=model_inputs['cache_indirection'],
            # ), 
            # AttentionParams(
            #     sequence_length=model_inputs['sequence_length'],
            #     context_lengths=model_inputs['context_lengths'],
            #     host_context_lengths=model_inputs['host_context_lengths'],
            #     max_context_length=max_input_len,
            #     host_request_types=model_inputs['host_request_types']),
            model_inputs['hidden_states_input'],
            model_inputs['all_reduce_workspace'],
            # model_inputs['prompt_embedding_table'],
            # model_inputs['tasks'],
            # model_inputs['prompt_vocab_size'],
        )
