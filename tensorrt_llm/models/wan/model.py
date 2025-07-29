from typing import Optional

import tensorrt as trt

import tensorrt_llm as tllm
from tensorrt_llm._common import default_net
from tensorrt_llm.functional import Tensor, concat, expand, pad, shape, slice
from tensorrt_llm.layers import RmsNorm
from tensorrt_llm.layers.attention import DiffusersAttention, bert_attention
from tensorrt_llm.mapping import Mapping

ADD_DEBUG_TENSOR = True


class CrossAttention(DiffusersAttention):

    def __init__(
            self,
            query_dim=5120,
            cross_attention_dim=5120,
            heads=40,
            kv_heads=40,
            dim_head=128,
            dropout=0.0,
            bias=True,
            upcast_attention=False,
            upcast_softmax=False,
            cross_attention_norm=None,
            cross_attention_norm_num_groups=0,
            qk_norm='rms_norm',  # This is not used since we need rms_norm_cross_attention, which is not released yet
            added_kv_proj_dim=5120,
            added_proj_bias=True,
            norm_num_groups=None,
            spatial_norm_dim=None,
            out_bias=True,
            scale_qk=True,
            only_cross_attention=False,
            eps=1e-5,
            rescale_output_factor=1.0,
            residual_connection=False,
            out_dim=5120,
            out_context_dim=None,
            context_pre_only=None,
            pre_only=False,
            elementwise_affine=False,
            is_causal=False,
            attn_forward_funcname='wan_cross_attn',
            mapping=Mapping(),
            dtype=trt.bfloat16,
    ):
        super().__init__(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            kv_heads=kv_heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cross_attention_norm=cross_attention_norm,
            cross_attention_norm_num_groups=cross_attention_norm_num_groups,
            qk_norm=qk_norm,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=added_proj_bias,
            norm_num_groups=norm_num_groups,
            spatial_norm_dim=spatial_norm_dim,
            out_bias=out_bias,
            scale_qk=scale_qk,
            only_cross_attention=only_cross_attention,
            eps=eps,
            rescale_output_factor=rescale_output_factor,
            residual_connection=residual_connection,
            out_dim=out_dim,
            out_context_dim=out_context_dim,
            context_pre_only=context_pre_only,
            pre_only=pre_only,
            elementwise_affine=elementwise_affine,
            is_causal=is_causal,
            attn_forward_funcname=attn_forward_funcname,
            mapping=mapping,
            dtype=dtype,
        )
        self.norm_added_k = RmsNorm(dim_head * kv_heads, eps=eps, dtype=dtype)
        self.norm_q = RmsNorm(dim_head * heads, eps=eps, dtype=dtype)
        self.norm_k = RmsNorm(dim_head * kv_heads, eps=eps, dtype=dtype)

    # i2v
    def wan_cross_attn(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        max_input_length: int = 81920,
        *args,
        **kwargs,
    ):
        batch_size = shape(hidden_states, 0)
        encoder_context_len = shape(hidden_states, 1)
        image_context_len = encoder_context_len - 512

        starts = concat([0, 0, 0])
        sizes = concat(
            [batch_size, image_context_len, self.cross_attention_dim])
        encoder_hidden_states_img = slice(encoder_hidden_states,
                                          starts=starts,
                                          sizes=sizes)

        starts = concat([0, image_context_len, 0])
        sizes = concat([batch_size, 512, self.cross_attention_dim])
        encoder_hidden_states = slice(encoder_hidden_states,
                                      starts=starts,
                                      sizes=sizes)

        query = self.to_q(hidden_states)
        q_dtype = query.dtype
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        head_dim = self.dim_head
        inner_dim = head_dim * self.heads

        query = self.norm_q(query)
        key = self.norm_k(key)

        # query = query.view(concat([batch_size, -1, self.heads,
        #                            head_dim])).permute([0, 2, 1, 3])
        # key = key.view(concat([batch_size, -1, self.heads,
        #                        head_dim])).permute([0, 2, 1, 3])
        # value = value.view(concat([batch_size, -1, self.heads,
        #                            head_dim])).permute([0, 2, 1, 3])

        # `context` projections.
        key_img = self.add_k_proj(encoder_hidden_states_img)

        key_img = self.norm_added_k(key_img)

        value_img = self.add_v_proj(encoder_hidden_states_img)

        # key_img = key_img.view(
        #     concat([batch_size, -1, self.heads,
        #             head_dim])).permute([0, 2, 1, 3])
        # value_img = value_img.view(
        #     concat([batch_size, -1, self.heads,
        #             head_dim])).permute([0, 2, 1, 3])

        # Transpose from [batch_size, num_heads, seq_len, head_dim] back to
        #   [batch_size, seq_len, num_heads * head_dim] for attention plugin.
        # print(query.shape, key_img.shape, value_img.shape)

        # query = query.permute([0, 2, 1,
        #                        3]).view(concat([batch_size, -1, inner_dim]))

        assert default_net(
        ).plugin_config.bert_attention_plugin is not None, 'Enable bert attention to run this!'

        key_img = pad(key_img, (0, 0, 0, 512), mode='constant', value=0.0)
        value_img = pad(value_img, (0, 0, 0, 512), mode='constant', value=0.0)

        # print(query.shape, key_img.shape, value_img.shape)

        qkv_img = concat([query, key_img, value_img], dim=-1)

        print(qkv_img.shape)

        input_lengths = expand(
            shape(qkv_img, 1).unsqueeze(0),
            shape(qkv_img, 0).unsqueeze(0)).cast("int32")

        hidden_states_img = bert_attention(qkv_img,
                                           input_lengths,
                                           self.heads,
                                           head_dim,
                                           q_scaling=self.q_scaling,
                                           relative_attention=False,
                                           max_distance=self.max_distance,
                                           max_input_length=max_input_length)
        print('Hidden states shape: ', hidden_states.shape)

        hidden_states_img = hidden_states_img.permute([0, 2, 1,
                                                       3]).flatten(2, 3)
        hidden_states_img = hidden_states_img.cast(q_dtype)

        key = key.permute([0, 2, 1, 3]).view(concat([batch_size, -1,
                                                     inner_dim]))
        value = value.permute([0, 2, 1,
                               3]).view(concat([batch_size, -1, inner_dim]))

        qkv = concat([query, key, value], dim=-1)
        input_lengths = expand(
            shape(qkv, 1).unsqueeze(0),
            shape(qkv, 0).unsqueeze(0)).cast("int32")

        hidden_states = bert_attention(qkv,
                                       input_lengths,
                                       self.heads,
                                       head_dim,
                                       q_scaling=self.q_scaling,
                                       relative_attention=False,
                                       max_distance=self.max_distance,
                                       max_input_length=max_input_length)

        hidden_states = hidden_states.permute(0, 2, 1, 3).flatten(2, 3)
        hidden_states = hidden_states.cast(q_dtype)
        hidden_states = hidden_states + hidden_states_img
        hidden_states = self.to_out[0](hidden_states)
        return hidden_states


attn = CrossAttention()

logger = trt.Logger(trt.Logger.VERBOSE)

builder = trt.Builder(logger)
trt_network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.BF16)

network = tllm.Network()
network._init(trt_network)

tllm._common.set_network(network)

hidden_states = tllm.Tensor(
    name='hidden_states',
    dtype=trt.bfloat16,
    shape=(1, 49392, 5120),
    is_network_input=True,
    network=network,
    location=trt.TensorLocation.DEVICE,
)
encoder_hidden_states = tllm.Tensor(
    name='encoder_hidden_states',
    dtype=trt.bfloat16,
    shape=(1, 769, 5120),
    is_network_input=True,
    network=network,
    location=trt.TensorLocation.DEVICE,
)

attn(hidden_states, encoder_hidden_states)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 25)  # 1 MiB
serialized_engine = builder.build_serialized_network(network, config)

with open("sample.engine.trt", "wb") as f:
    f.write(serialized_engine)
