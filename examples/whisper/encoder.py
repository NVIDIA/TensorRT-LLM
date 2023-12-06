import math
from collections import OrderedDict
from typing import Optional
from typing import List, Optional, Sequence, Tuple, Union


import numpy as np
import torch
import tensorrt as trt

from tensorrt_llm._common import default_net
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import (LayerNormPositionType, LayerNormType,
                                     PositionEmbeddingType, Tensor, assertion,
                                     concat, constant, expand, expand_mask,
                                     gather_last_token_logits, shape, slice,
                                     ACT2FN, add, permute, view, elementwise_binary)
from tensorrt_llm.layers import (MLP, Attention, AttentionMaskType,
                                 AttentionParams, BertAttention, ColumnLinear,
                                 Embedding, GroupNorm, KeyValueCacheParams,
                                 LayerNorm, RmsNorm, Conv2d)
from tensorrt_llm.module import Module, ModuleList

layernorm_map = {
    LayerNormType.LayerNorm: LayerNorm,
    LayerNormType.RmsNorm: RmsNorm,
    LayerNormType.GroupNorm: GroupNorm,
}


# from tensorrt_llm._common import default_net, precision
# from tensorrt_llm._utils import numpy_fp32_to_bf16, trt_dtype_to_np
# from tensorrt_llm.functional import (AttentionMaskType, PositionEmbeddingType,
#                           RotaryScalingType, Tensor, bert_attention, cast, clip,
#                           concat, constant, embedding, expand_dims, expand_mask,
#                           generate_alibi_biases, generate_alibi_slopes,
#                           gpt_attention, matmul, repeat_interleave, round,
#                           shape, slice, softmax, split, view, where)
# from tensorrt_llm.module import Module
# from tensorrt_llm.parameter import Parameter
# from tensorrt_llm.quantization import QuantMode
# from tensorrt_llm.quantization.functional import dequantize, quantize
# from tensorrt_llm.quantization.layers import FP8Linear, FP8RowLinear
# from tensorrt_llm.layers import RowLinear
# from tensorrt_llm.layers.lora import Lora, LoraRuntimeParam



# class Linear(Module):

#     def __init__(self,
#                  in_features,
#                  out_features,
#                  bias=True,
#                  dtype=None,
#                  tp_group=None,
#                  tp_size=1,
#                  gather_output=True,
#                  share_weight=None):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features // tp_size
#         self.dtype = dtype

#         if not share_weight:
#             self.weight = Parameter(shape=(self.out_features, self.in_features),
#                                     dtype=dtype)
#         else:
#             self.weight = share_weight

#         self.tp_size = tp_size
#         self.tp_group = tp_group
#         self.gather_output = gather_output

#         if bias:
#             self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
#         else:
#             self.register_parameter('bias', None)

#         self.lora = Lora(
#             in_hidden_size=self.in_features,
#             out_hidden_size=self.out_features,
#             max_low_rank=min(
#                 self.in_features, self.out_features
#             ),  # Assume low rank is smaller than in/out features
#         )

#     def multiply_gather(self,
#                         x,
#                         weight,
#                         gemm_plugin,
#                         use_fp8=False,
#                         lora_runtime_param: LoraRuntimeParam = None):
#         hidden_state = x

#         if gemm_plugin:
#             x = _gemm_plugin(x, weight, transb=True, use_fp8=use_fp8)
#         else:
#             x = matmul(x, weight, transb=True)

#         if default_net(
#         ).plugin_config.lora_plugin and lora_runtime_param is not None:
#             x = x + self.lora(hidden_state,
#                               lora_runtime_param=lora_runtime_param)
#         print(x.dtype, self.bias.value.dtype)
#         if self.bias is not None:
#             if x.dtype != self.bias.value.dtype:
#                 x = cast(x, self.bias.value.dtype)
#             x = x + self.bias.value

#         if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
#             # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
#             x = allgather(x, self.tp_group, gather_dim=1)
#         print(x.dtype)
#         xx
#         return x

#     def forward(self, x, lora_runtime_param: LoraRuntimeParam = None):
#         return self.multiply_gather(x,
#                                     self.weight.value,
#                                     default_net().plugin_config.gemm_plugin,
#                                     lora_runtime_param=lora_runtime_param)
        

# class BertAttention(Module):

#     def __init__(self,
#                  hidden_size,
#                  num_attention_heads,
#                  max_position_embeddings=1024,
#                  num_layers=1,
#                  attention_head_size=None,
#                  num_kv_heads=None,
#                  q_scaling=1.0,
#                  apply_query_key_layer_scaling=False,
#                  bias=True,
#                  dtype=None,
#                  tp_group=None,
#                  tp_size=1,
#                  tp_rank=0,
#                  relative_attention=False,
#                  max_distance=0,
#                  num_buckets=0):
#         super().__init__()

#         self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
#         self.num_attention_heads = num_attention_heads // tp_size
#         self.num_attention_kv_heads = (
#             num_kv_heads + tp_size - 1
#         ) // tp_size if num_kv_heads is not None else self.num_attention_heads
#         self.hidden_size = hidden_size // tp_size
#         self.max_position_embeddings = max_position_embeddings
#         self.norm_factor = math.sqrt(self.attention_head_size)
#         self.tp_size = tp_size
#         self.tp_rank = tp_rank

#         self.num_layers = num_layers
#         self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
#         self.norm_factor = math.sqrt(self.attention_head_size)
#         self.q_scaling = q_scaling
#         if self.apply_query_key_layer_scaling:
#             self.norm_factor *= self.num_layers
#             self.q_scaling *= self.num_layers

#         self.dtype = dtype

#         self.relative_attention = relative_attention
#         self.max_distance = max_distance

#         # out dim is not necessarily hidden_size + kv specific size (in MQA/GQA), but num_heads * heads_size
#         # example: d_model != num_heads * head_size in Flan-T5
#         self.qkv = Linear(
#             hidden_size,
#             tp_size * self.num_attention_heads * self.attention_head_size +
#             (2 * tp_size * self.num_attention_kv_heads *
#              self.attention_head_size),
#             bias=bias,
#             dtype=dtype,
#             tp_group=tp_group,
#             tp_size=tp_size,
#             gather_output=False)
#         self.dense = RowLinear(tp_size * self.num_attention_heads *
#                                self.attention_head_size,
#                                hidden_size,
#                                bias=bias,
#                                dtype=dtype,
#                                tp_group=tp_group,
#                                tp_size=tp_size)

#         # per-layer relative attention table
#         if relative_attention:
#             self.rel_attn_table = Parameter(shape=(num_attention_heads //
#                                                    tp_size, num_buckets),
#                                             dtype=dtype)

#     def forward(self,
#                 hidden_states: Tensor,
#                 attention_mask=None,
#                 input_lengths=None,
#                 workspace=None,
#                 max_input_length=None):
#         assert isinstance(hidden_states, Tensor)

#         print(hidden_states.dtype, self.qkv.weight.value.dtype)
#         qkv = self.qkv(hidden_states)
#         print(qkv.dtype)
#         xx
#         if default_net().plugin_config.bert_attention_plugin:
#             # TRT plugin mode
#             assert input_lengths is not None
#             context = bert_attention(
#                 qkv,
#                 input_lengths,
#                 self.num_attention_heads,
#                 self.attention_head_size,
#                 q_scaling=self.q_scaling,
#                 relative_attention=self.relative_attention,
#                 max_distance=self.max_distance,
#                 relative_attention_bias=self.rel_attn_table.value
#                 if self.relative_attention else None,
#                 max_input_length=max_input_length)
#         else:
#             # plain TRT mode
#             def transpose_for_scores(x):
#                 new_x_shape = concat([
#                     shape(x, 0),
#                     shape(x, 1), self.num_attention_heads,
#                     self.attention_head_size
#                 ])
#                 return x.view(new_x_shape).permute([0, 2, 1, 3])
            
#             query, key, value = split(qkv, self.hidden_size, dim=2)
#             print(query.dtype, key.dtype, value.dtype)
#             query = transpose_for_scores(query)
#             key = transpose_for_scores(key)
#             value = transpose_for_scores(value)

#             key = key.permute([0, 1, 3, 2])
#             attention_scores = matmul(query, key)
#             attention_scores = attention_scores / self.norm_factor

#             if attention_mask is not None:
#                 attention_scores = attention_scores + attention_mask

#             attention_probs = softmax(attention_scores, dim=-1)

#             context = matmul(attention_probs, value).permute([0, 2, 1, 3])
#             context = context.view(
#                 concat([shape(context, 0),
#                         shape(context, 1), self.hidden_size]))

#         context = self.dense(context, workspace)

#         return context

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def gelu(x: Tensor) -> Tensor:
    '''
    Add a GELU operation.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    if not default_net().strongly_typed:
        return 0.5 * x * (
            tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * pow(x, 3.0))) + 1.0)

    array_fn = {
        trt.float32: fp32_array,
        trt.float16: fp16_array,
        trt.bfloat16: bf16_array,
    }[x.dtype]

    v1 = constant(array_fn([0.5]))
    v2 = constant(array_fn([math.sqrt(2.0 / math.pi)]))
    v3 = constant(array_fn([0.044715]))
    v4 = constant(array_fn([3.0]))
    v5 = constant(array_fn([1.0]))
    return v1 * x * (tanh(v2 * (x + v3 * pow(x, v4))) + v5)


class EncoderLayer(Module):

    def __init__(self,
                 n_state,
                 num_attention_heads,
                 num_kv_heads=None,
                 q_scaling=1.0,
                 layernorm_eps=1e-5,
                 hidden_act="gelu",
                 tp_group=None,
                 tp_size=1,
                 dtype=None,
                ):
        super().__init__()
        ln_type = layernorm_map[LayerNormType.LayerNorm]
        print(q_scaling, dtype)
        self.attention = BertAttention(
            n_state,
            num_attention_heads,
            num_kv_heads=num_kv_heads,
            q_scaling=q_scaling,
            bias=True,
            tp_group=tp_group,
            tp_size=tp_size,
            dtype=dtype,
        )

        self.attention_layernorm = ln_type(normalized_shape=n_state,
                                           eps=layernorm_eps,
                                           dtype=dtype)
        n_mlp = n_state * 4 
        self.mlp = MLP(
            hidden_size=n_state,
            ffn_hidden_size=n_mlp,
            hidden_act=hidden_act,
            bias=True,
            tp_group=tp_group,
            tp_size=tp_size,
            dtype=dtype,
        )
        self.mlp_layernorm = ln_type(normalized_shape=n_state,
                                     eps=layernorm_eps,
                                     dtype=dtype)
        self._dtype = dtype

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                layer_idx=None):
        residual = hidden_states
        hidden_states = self.attention_layernorm(hidden_states)
        
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = residual + attention_output


        # mlp
        residual = hidden_states
        hidden_states = self.mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states
        

class AudioEncoder(Module):
    def __init__(self,
                 n_state,
                 num_layers,
                 n_ctx,
                 num_heads,
                 num_kv_heads=None,
                 q_scaling=1/(math.sqrt(6)*(((384 // 6) ** -0.25))),
                 layernorm_eps=1e-5,
                 hidden_act="gelu",
                 tp_group=None,
                 tp_size=1,
                 dtype=None,
                 n_mels=80):
        super().__init__()
        self.n_state = n_state
        self.n_ctx = n_ctx
        # add convolutions
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        else:
            self.hidden_act = hidden_act
        if isinstance(dtype, str):
            self._dtype = str_dtype_to_trt(dtype)
        else:
            print(dtype)
            assert isinstance(dtype, trt.DataType)
            self._dtype = dtype

        self.num_layers = num_layers
        self.conv1 = Conv2d(n_mels, n_state, (3,1), padding=(1, 0), dtype=self._dtype)
        self.conv2 = Conv2d(n_state, n_state, (3, 1), padding=(1, 0), stride=(2, 1), dtype=self._dtype)

        # add as a constant and then use elementwise op to add to inputs
        # self.positional_embedding = constant(sinusoids(n_state, n_ctx))
        self.blocks = ModuleList([
            EncoderLayer(
                n_state,
                num_heads,
                num_kv_heads if num_kv_heads else num_heads,
                q_scaling=q_scaling,
                dtype=self._dtype,
            ) for _ in range(self.num_layers)
        ])
        ln_type = layernorm_map[LayerNormType.LayerNorm]
        self.ln_post = ln_type(normalized_shape=n_state,
                                     eps=layernorm_eps,
                                     dtype=self._dtype)
        self.position_emb = np.expand_dims(
            sinusoids(self.n_ctx, self.n_state).numpy(), axis=0)
        if self._dtype==str_dtype_to_trt("float32"):
            self.position_emb = self.position_emb.astype(np.float32)
        else:
            print(self._dtype)
            self.position_emb = self.position_emb.astype(np.float16)
        
        print(self.position_emb.shape, type(self.position_emb))
        
    def forward(self, x):
        # expand dims for 2d conv
        x = view(x, [*x.shape, 1])
        x = self.conv1(x)
        x = ACT2FN[self.hidden_act](x)
        
        x = self.conv2(x)
        x = ACT2FN[self.hidden_act](x)
        x = view(x, [1, self.n_state, self.n_ctx])

        x = permute(x, [0, 2, 1])

        # add positional embeddings
        position_embedding = constant(self.position_emb)
        
        hidden_states = add(x, position_embedding)

        # attention blocks
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states=hidden_states, layer_idx=layer_idx)
            
        hidden_states = self.ln_post(hidden_states)
        hidden_states.mark_output('encoder_output', self._dtype)
        return hidden_states

    def prepare_inputs(self, max_batch_size=4):
        """
        @brief: Prepare inputs Tensors for the model

        @return: a list contains values which can be fed into the self.forward()
        """
        # bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        input_features = Tensor(
            name="input_features",
            dtype=self._dtype,
            shape=[1,80,3000],
            dim_range=OrderedDict([
                ('batch_size', [1]), ('n_mels', [80]), ('input_length', [3000])
            ])
        )
        return [input_features]