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

from tensorrt_llm.quantization import QuantMode

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import gather_last_token_logits, recv, send, layer_norm, group_norm, view, matmul, silu, sigmoid, squared_relu, Tensor
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
                  num_att, 
                  num_head, 
                  hidden_size, 
                  dtype = None
                  ):
        super().__init__()
        self.num_att = num_att
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.dtype = dtype
        
        self.rxw = ColumnLinear(hidden_size, hidden_size)
        self.kxw = ColumnLinear(hidden_size, hidden_size)
        self.vxw = ColumnLinear(hidden_size, hidden_size)
        self.gxw = ColumnLinear(hidden_size, hidden_size)
        self.oxw = ColumnLinear(hidden_size, hidden_size)
        
        self.ln_w = Parameter(shape=(hidden_size), dtype=dtype)
        self.ln_b = Parameter(shape=(hidden_size), dtype=dtype)
        self.lx_w = Parameter(shape=(hidden_size), dtype='float32')  # special case
        self.lx_b = Parameter(shape=(hidden_size), dtype='float32')  # special case
        
        self.k_mix = Parameter(shape=(hidden_size), dtype=dtype)
        self.v_mix = Parameter(shape=(hidden_size), dtype=dtype)
        self.r_mix = Parameter(shape=(hidden_size), dtype=dtype)
        self.g_mix = Parameter(shape=(hidden_size), dtype=dtype)
        
        self.t_decay = Parameter(shape=(num_head, num_att // num_head, 1), dtype='float32') # special case
        self.t_first = Parameter(shape=(num_head, num_att // num_head, 1), dtype='float32') # special case
        
    def forward(self, x, sx, s):
        xx = layer_norm(x, (x.shape[-1],), weight=self.ln_w, bias=self.ln_b)
        kx = xx * self.k_mix + sx * (1 - self.k_mix)
        vx = xx * self.v_mix + sx * (1 - self.v_mix)
        rx = xx * self.r_mix + sx * (1 - self.r_mix)
        gx = xx * self.g_mix + sx * (1 - self.g_mix)
        
        T = x.shape[0] # query length
        H = self.num_head
        S = self.hidden_size // self.num_head
        
        r = self.rxw(rx).to('float32').view(T, H, S).transpose(0, 1)
        k = self.kxw(kx).to('float32').view(T, H, S).transpose(0, 1).transpose(-2, -1)
        v = self.vxw(vx).to('float32').view(T, H, S).transpose(0, 1)
        g = silu(self.gxw(gx))
        
        # TODO(Rinne): this loop will impact on the performance, needs to be replaced by cuda kernel.
        out = Tensor(name="rnn_output", dtype=self.dtype, shape=(T, H, S))
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = matmul(kt, vt)
            out[t] = matmul(rt, (self.t_first * at + s)).view(H, S)
            s = at + self.t_decay * s
            
        out = out.reshape(T, H * S)
        out = group_norm(out, num_groups=H, weight=self.lx_w, bias=self.lx_b)
        out = out.to(dtype=x.dtype) * g
        out = self.oxw(out)
        
        return x + out, xx[-1, :], s
        

class RwkvFeedForwardLayer(Module):
    def __init__(self, 
                  num_att, 
                  num_head, 
                  hidden_size, 
                  dtype = None
                  ):
        super().__init__()
        self.num_att = num_att
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.dtype = dtype
        
        self.rxw = ColumnLinear(hidden_size, hidden_size)
        self.kxw = ColumnLinear(hidden_size, hidden_size)
        self.vxw = ColumnLinear(hidden_size, hidden_size)
        
        self.ln_w = Parameter(shape=(hidden_size), dtype=dtype)
        self.ln_b = Parameter(shape=(hidden_size), dtype=dtype)
        
        self.k_mix = Parameter(shape=(hidden_size), dtype=dtype)
        self.r_mix = Parameter(shape=(hidden_size), dtype=dtype)
        
    def forward(self, x, sx):
        xx = layer_norm(x, (x.shape[-1],), weight=self.ln_w, bias=self.ln_b)
        kx = xx * self.k_mix + sx * (1 - self.k_mix)
        rx = xx * self.r_mix + sx * (1 - self.r_mix)
        
        r = sigmoid(self.rxw(rx))
        vx = squared_relu(self.kxw(kx))
        out = r * self.vxw(vx)
        return x + out, xx[-1, :]

class RwkvDecoderLayer(Module):
    def __init__(self, 
                  layer_id,
                  num_att, 
                  num_head, 
                  hidden_size, 
                  rescale_layer, 
                  dtype = None
                  ):
        super().__init__()
        self._layer_id = layer_id
        self.num_att = num_att
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.rescale_layer = rescale_layer
        self.dtype = dtype
        
        self.attention = RwkvSelfAttentionLayer(self.num_att, 
                                                self.num_head, 
                                                self.hidden_size, 
                                                self.dtype)
        self.feed_forward = RwkvFeedForwardLayer(self.num_att, 
                                                self.num_head, 
                                                self.hidden_size, 
                                                self.dtype)
        
    def forward(self, 
                x, 
                input_states, 
                hidden_states, 
                output_states):
        x = x.to(self.dtype)
        x, input_states, hidden_states = self.attention(x, input_states, hidden_states)
        x, output_states = self.feed_forward(x, output_states)
        if self.rescale_layer > 0:
            x /= 2
            
        return x, input_states, hidden_states, output_states
            

class RwkvModel(Module):
    def __init__(self, 
                 num_layers, 
                 num_att, 
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
                                                    num_att=num_att, 
                                                    num_head=num_head, 
                                                    hidden_size=hidden_size, 
                                                    rescale_layer=rescale_layer if index % rescale_layer == 0 else 0, 
                                                    dtype=dtype))
            index += 1
        
        self.layers = ModuleList(layer_modules)
        self.embedding_weight = Parameter(shape=(vocab_size, hidden_size))
        self.ln_out_w = Parameter(shape=(hidden_size))
        self.ln_out_b = Parameter(shape=(hidden_size))
        self.head_layer = ColumnLinear(vocab_size, hidden_size)
        
        if self.mapping.is_last_pp_rank():
            # TODO(Rinne): add process for this case
            raise NotImplementedError
        
    # TODO(Rinne): deal with batch decode
    def forward(self,
                input_ids,
                input_states=None,
                hidden_states=None,
                output_states=None,
                embeddings=None, 
                all_reduce_workspace=None,
                ):
        # TODO(Rinne): cache, kv_cache, attention_mask, tuning, all_reduce_workspace
        
        ptuning_args = []
        if self.use_prompt_tuning:
            raise NotImplementedError
     
        if self.mapping.is_first_pp_rank():
            embeddings = self.vocab_embedding(input_ids, *ptuning_args,
                                                 all_reduce_workspace)
        else:
            embeddings = recv(embeddings, self.mapping.prev_pp_rank())
     
        self.register_network_output(f"embd", embeddings)
        self.register_network_output(f"input_states", input_states)
        self.register_network_output(f"hidden_states", hidden_states)
        self.register_network_output(f"output_states", output_states)
        
        for layer in self.layers:
            embeddings, input_states, hidden_states, output_states = \
                layer(embeddings, input_states, hidden_states, output_states)
        
        # TODO(Rinne): deal with strategy
        embeddings = embeddings[-1, :]
        
        embeddings = layer_norm(embeddings, (hidden_states, ), self.ln_out_w, self.ln_out_b)
        
        # TODO(Rinne): deal with quantization here
        embeddings = self.head_layer(embeddings)

        if self.mapping.is_last_pp_rank():
            raise NotImplementedError
        else:
            embeddings = send(embeddings, self.mapping.next_pp_rank())
            
        return embeddings, input_states, hidden_states, output_states
    
    
class RwkvForCausalLM(RwkvModel, GenerationMixin):
    def __init__(self, 
                 num_layers, 
                 num_att, 
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
        
        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.dtype = dtype
            
        self.num_layers = num_layers
        self.num_att = num_att
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.rescale_layer = rescale_layer
        self.vocab_size = vocab_size
        
        self.kv_dtype = self.dtype
        if quant_mode.has_int8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('fp8')

        self.quant_mode = quant_mode
        self.use_parallel_embedding = use_parallel_embedding
        self.embedding_sharding_dim = embedding_sharding_dim
        
        super().__init__(num_layers, 
                         num_att, 
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
            
    def forward(self):
         raise NotImplementedError
     
    def prepare_inputs(self):
         raise NotImplementedError