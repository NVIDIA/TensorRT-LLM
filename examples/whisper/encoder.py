import math
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union


import numpy as np
import torch
import tensorrt as trt

from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import (LayerNormPositionType, LayerNormType,
                                     MLPType, Tensor, assertion,
                                     concat, constant, expand, expand_mask,
                                     gather_last_token_logits, shape, slice,
                                     ACT2FN, add, permute, view, elementwise_binary)
from tensorrt_llm.layers import (MLP, Attention, AttentionMaskType,
                                 AttentionParams, BertAttention, ColumnLinear,
                                 Embedding, GroupNorm, KeyValueCacheParams,
                                 LayerNorm, RmsNorm, Conv2d, FusedGatedMLP, GatedMLP)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.module import Module, ModuleList

layernorm_map = {
    LayerNormType.LayerNorm: LayerNorm,
    LayerNormType.RmsNorm: RmsNorm,
    LayerNormType.GroupNorm: GroupNorm,
}

mlp_map = {
    MLPType.MLP: MLP,
    MLPType.GatedMLP: GatedMLP,
    MLPType.FusedGatedMLP: FusedGatedMLP,
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
                 hidden_size,
                 num_attention_heads,
                 num_kv_heads,
                 head_size,
                 max_position_embeddings=None,
                 q_scaling=1.0,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 layernorm_type=LayerNormType.LayerNorm,
                 layernorm_eps=1e-5,
                 hidden_act="gelu",
                 mlp_type=MLPType.MLP,
                 mapping=Mapping(),
                 dtype=None,
                 residual_scaling=1.0,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0
                ):
        super().__init__()

        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        self.layernorm_position = layernorm_position

        self.attention = BertAttention(
            hidden_size,
            num_attention_heads,
            attention_head_size=head_size,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            q_scaling=q_scaling,
            bias=has_attention_qkvo_bias,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            dtype=dtype,
            relative_attention=relative_attention,
            max_distance=max_distance,
            num_buckets=num_buckets
        )

        self.attention_layernorm = ln_type(normalized_shape=hidden_size,
                                           eps=layernorm_eps,
                                           dtype=dtype)
        n_mlp = hidden_size * 4 
        self.mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=n_mlp,
            hidden_act=hidden_act,
            bias=has_mlp_bias,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            dtype=dtype,
        )
        self.mlp_layernorm = ln_type(normalized_shape=hidden_size,
                                     eps=layernorm_eps,
                                     dtype=dtype)
        self.residual_scaling = residual_scaling

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                input_lengths=None,
                all_reduce_workspace=None,
                max_input_length=None):
        assert isinstance(hidden_states, Tensor)

        # self attention
        residual = hidden_states * self.residual_scaling
        
        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.attention_layernorm(hidden_states)
        
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            workspace=all_reduce_workspace,
            max_input_length=max_input_length
        )

        hidden_states = residual + attention_output

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.attention_layernorm(hidden_states)

        # mlp
        residual = hidden_states * self.residual_scaling
        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states, workspace=all_reduce_workspace)

        hidden_states = residual + hidden_states

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)
        return hidden_states
        

class AudioEncoder(Module, GenerationMixin):
    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 dtype,
                 head_size=None,
                 num_kv_heads=None,
                 max_position_embeddings=None,
                 relative_attention=False,
                 max_distance=None,
                 num_buckets=None,
                 has_embedding_layernorm=False,
                 has_embedding_scale=False,
                 q_scaling=1.0,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 has_model_final_layernorm=False,
                 layernorm_eps=1e-5,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 layernorm_type=LayerNormType.LayerNorm,
                 hidden_act="gelu",
                 mlp_type=MLPType.MLP,
                 residual_scaling=1.0,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 mapping=Mapping(),
                 n_mels=80):
        super().__init__()

        self.mapping = mapping

        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        self.has_attention_qkvo_bias = has_attention_qkvo_bias
        self.has_mlp_bias = has_mlp_bias

        self.has_model_final_layernorm = has_model_final_layernorm
        self.n_mels = n_mels

        if isinstance(dtype, str):
            self._dtype = str_dtype_to_trt(dtype)
        else:
            print(dtype)
            assert isinstance(dtype, trt.DataType)
            self._dtype = dtype
        
        self.total_num_layers = num_layers
        self.num_layers = num_layers // self.mapping.pp_size

        self.hidden_size = hidden_size
        self.n_ctx = max_position_embeddings

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None or num_kv_heads <= 0:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = self.hidden_size // self.num_heads if head_size is None else head_size
        
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        else:
            self.hidden_act = hidden_act
        self.num_layers = num_layers
        self.conv1 = Conv2d(n_mels, hidden_size, (3,1), padding=(1, 0), dtype=self._dtype)
        self.conv2 = Conv2d(hidden_size, hidden_size, (3, 1), padding=(1, 0), stride=(2, 1), dtype=self._dtype)

        # add as a constant and then use elementwise op to add to inputs
        # self.positional_embedding = constant(sinusoids(hidden_size, n_ctx))
        self.blocks = ModuleList([
            EncoderLayer(hidden_size=hidden_size,
                         num_attention_heads=num_heads,
                         num_kv_heads=num_kv_heads,
                         head_size=self.head_size,
                         max_position_embeddings=max_position_embeddings,
                         q_scaling=q_scaling,
                         has_attention_qkvo_bias=has_attention_qkvo_bias,
                         has_mlp_bias=has_mlp_bias,
                         layernorm_position=layernorm_position,
                         layernorm_eps=layernorm_eps,
                         layernorm_type=layernorm_type,
                         hidden_act=hidden_act,
                         mlp_type=mlp_type,
                         mapping=self.mapping,
                         dtype=dtype,
                         residual_scaling=residual_scaling,
                         relative_attention=relative_attention,
                         max_distance=max_distance,
                         num_buckets=num_buckets) for _ in
            self.get_transformer_layers(self.mapping, self.total_num_layers)
        ])
        if self.mapping.is_last_pp_rank():
            if self.has_model_final_layernorm:
                self.final_layernorm = ln_type(normalized_shape=hidden_size,
                                               eps=layernorm_eps,
                                               dtype=dtype)
        self.position_emb = np.expand_dims(
            sinusoids(self.n_ctx, self.hidden_size).numpy(), axis=0)
        if self._dtype==str_dtype_to_trt("float32"):
            self.position_emb = self.position_emb.astype(np.float32)
        else:
            print(self._dtype)
            self.position_emb = self.position_emb.astype(np.float16)
        
    def forward(self, x):
        # expand dims for 2d conv
        x = view(x, [*x.shape, 1])
        x = self.conv1(x)
        x = ACT2FN[self.hidden_act](x)
        
        x = self.conv2(x)
        x = ACT2FN[self.hidden_act](x)
        x = view(x, [x.shape[0], self.hidden_size, self.n_ctx])

        x = permute(x, [0, 2, 1])

        # add positional embeddings
        position_embedding = constant(self.position_emb)
        
        hidden_states = add(x, position_embedding)

        # attention blocks
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states=hidden_states)
            
        if self.final_layernorm:
            hidden_states = self.final_layernorm(hidden_states)
        hidden_states.mark_output('encoder_output', self._dtype)
        return hidden_states

    def prepare_inputs(self, max_batch_size=4):
        """
        @brief: Prepare inputs Tensors for the model

        @return: a list contains values which can be fed into the self.forward()
        """
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        input_features = Tensor(
            name="input_features",
            dtype=self._dtype,
            shape=[-1, self.n_mels, 3000],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]), ('n_mels', [self.n_mels]), ('input_length', [3000])
            ])
        )

        return [input_features]