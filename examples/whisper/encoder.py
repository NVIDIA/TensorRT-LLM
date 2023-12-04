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



def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


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
        self.conv1 = Conv2d(n_mels, n_state, (3,1), padding=(1, 0))
        self.conv2 = Conv2d(n_state, n_state, (3, 1), padding=(1, 0), stride=(2, 1))

        # add as a constant and then use elementwise op to add to inputs
        # self.positional_embedding = constant(sinusoids(n_state, n_ctx))
        self.blocks = ModuleList([
            EncoderLayer(
                n_state,
                num_heads,
                num_kv_heads if num_kv_heads else num_heads,
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
            dtype=trt.float32,
            shape=[1,80,3000, 1],
            dim_range=OrderedDict([
                ('batch_size', [1]), ('n_mels', [80]), ('input_length', [3000]), ('conv_unsqueeze', [1])
            ])
        )
        return [input_features]