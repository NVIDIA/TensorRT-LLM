# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/qwen3_next.py
# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import \
    fused_sigmoid_gating_delta_rule_update
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           MoEAllReduce, MoEAllReduceParams, allgather)
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (BaseMoeRoutingMethod,
                                 RenormalizeMoeRoutingMethod,
                                 RenormalizeNaiveMoeRoutingMethod,
                                 RoutingMethodType, TRTLLMGenFusedMoE,
                                 create_moe)
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from ..modules.mamba.layernorm_gated import RMSNorm as RMSNormGated
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType
from .modeling_qwen3 import Qwen3Attention
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, EagerFusionConfig, register_auto_model


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class Qwen3NextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3NextModel`]. It is used to instantiate a
    Qwen3-Next model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of
    Qwen3-Next-80B-A3B-Instruct [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids`.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
            Percentage of the query and keys which will have rotary embedding.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_dim (`int`, *optional*, defaults to 256):
            Projection weights dimension in multi-head attention.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size of the convolution used in linear attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each key head in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each value head in linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads used in linear attention layers.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads used in linear attention layers.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer.
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the routed expert.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the shared expert.
        num_experts_per_tok (`int`, *optional*, defaults to 10):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 512):
            Number of routed experts.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
            Indicate which layers use Qwen3NextMLP rather than Qwen3NextSparseMoeBlock
            The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
            If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.
        layer_types (`list[str]`, *optional*):
            Types of each layer (attention or linear).

    ```python
    >>> from transformers import Qwen3NextModel, Qwen3NextConfig

    >>> # Initializing a Qwen3Next style configuration
    >>> configuration =  Qwen3NextConfig()

    >>> # Initializing a model from the Qwen3-Next-80B-A3B style configuration
    >>> model = Qwen3NextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "qwen3_next"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=0.25,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        decoder_sparse_step=1,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=10,
        num_experts=512,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=[],
        layer_types=None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.get("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool(
                    (i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        # layer_type_validation(self.layer_types, self.num_hidden_layers)

        # linear attention part
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers


AutoConfig.register("qwen3_next", Qwen3NextConfig)


class Qwen3NextGate(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dtype: Optional[torch.dtype] = None,
        apply_routing: bool = False,
        routing_method_type: RoutingMethodType = RoutingMethodType.Renormalize,
        moe_backend: str = "CUTLASS",
    ):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size),
                                               dtype=dtype),
                                   requires_grad=False)
        self.routing_method_type = routing_method_type
        # FIXME: out_dtype=float32 does not work
        # self.out_dtype = torch.float32 if moe_backend == "TRTLLM" else dtype
        self.out_dtype = dtype

        assert not apply_routing, "Qwen3NextGate routing is called inside MoE"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = torch.ops.trtllm.cublas_mm(
            hidden_states, self.weight.t(), bias=None, out_dtype=self.out_dtype)
        return logits

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1

        self.weight.copy_(weights[0]["weight"][:])

    @property
    def routing_method(self) -> BaseMoeRoutingMethod:
        if self.routing_method_type == RoutingMethodType.RenormalizeNaive:
            return RenormalizeNaiveMoeRoutingMethod(top_k=self.top_k)
        elif self.routing_method_type == RoutingMethodType.Renormalize:
            return RenormalizeMoeRoutingMethod(top_k=self.top_k)
        else:
            raise ValueError(
                f"Unsupported routing method: {self.routing_method_type}")


class Qwen3NextSparseMoeBlock(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.mapping = model_config.mapping
        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy)

        self.gate = Qwen3NextGate(
            hidden_size=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=config.torch_dtype,
            apply_routing=False,
            routing_method_type=RoutingMethodType.Renormalize,
            moe_backend=model_config.moe_backend,
        )

        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=self.hidden_dim,
            intermediate_size=self.moe_intermediate_size,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            dtype=config.torch_dtype,
            reduce_results=False,
            model_config=model_config,
            layer_idx=layer_idx,
        )

        self.shared_expert = GatedMLP(
            hidden_size=self.hidden_dim,
            intermediate_size=config.shared_expert_intermediate_size,
            bias=config.mlp_bias if hasattr(config, 'mlp_bias') else False,
            dtype=config.torch_dtype,
            config=model_config,
            reduce_output=False,
        )

        self.shared_expert_gate = Linear(self.hidden_dim,
                                         1,
                                         bias=False,
                                         dtype=config.torch_dtype,
                                         quant_config=None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        do_finalize: Optional[bool] = True,
    ) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_dim
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        use_dp_padding = False
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        if not do_finalize:
            # TODO: support do_finalize == False
            raise NotImplementedError(
                "do_finalize == False is not supported yet")

        if self.enable_attention_dp and self.mapping.tp_size > 1:
            if isinstance(self.experts, TRTLLMGenFusedMoE):
                hidden_states = allgather(hidden_states,
                                          self.mapping,
                                          dim=0,
                                          sizes=all_rank_num_tokens)

        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
            do_finalize=do_finalize,
        )

        if not do_finalize:
            return final_hidden_states

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(
            self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            final_hidden_states = self.allreduce(
                final_hidden_states, all_reduce_params=all_reduce_params)

        return final_hidden_states.view(orig_shape)


@triton.jit
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    q_end: tl.constexpr = HEAD_QK
    blk_q_ptr = (mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T +
                 i_qk * QKVZ_DIM_T + tl.arange(0, q_end))
    k_end: tl.constexpr = q_end + HEAD_QK
    blk_k_ptr = (mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T +
                 i_qk * QKVZ_DIM_T + tl.arange(q_end, k_end))
    v_end: tl.constexpr = k_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_v_ptr = (mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T +
                 i_qk * QKVZ_DIM_T + tl.arange(k_end, v_end))
    z_end: tl.constexpr = v_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_z_ptr = (mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T +
                 i_qk * QKVZ_DIM_T + tl.arange(v_end, z_end))
    blk_q_st_ptr = (mixed_qkv + i_bs * NUM_HEADS_QK * QKV_DIM_T +
                    i_qk * HEAD_QK + tl.arange(0, HEAD_QK))
    blk_k_st_ptr = (mixed_qkv + i_bs * NUM_HEADS_QK * QKV_DIM_T +
                    NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK +
                    tl.arange(0, HEAD_QK))
    blk_v_st_ptr = (mixed_qkv + i_bs * NUM_HEADS_QK * QKV_DIM_T +
                    NUM_HEADS_QK * HEAD_QK * 2 +
                    i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK +
                    tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK))
    blk_z_st_ptr = (z + i_bs * NUM_HEADS_V * HEAD_V +
                    i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK +
                    tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK))
    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
    b_end: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V // NUM_HEADS_QK
    for i in tl.static_range(b_end):
        blk_b_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))
    for i in tl.static_range(b_end, a_end):
        blk_a_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_a_st_ptr = (a + i_bs * NUM_HEADS_V +
                        i_qk * NUM_HEADS_V // NUM_HEADS_QK + (i - b_end))
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    batch, seq_len = mixed_qkvz.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)
    grid = (batch * seq_len, num_heads_qk)
    fused_qkvzba_split_reshape_cat_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold,
                          (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty_like(a, dtype=torch.float32)
    fused_gdn_gating_kernel[grid](g,
                                  A_log,
                                  a,
                                  dt_bias,
                                  seq_len,
                                  num_heads,
                                  beta,
                                  threshold,
                                  8,
                                  num_warps=1)
    return g


class Qwen3NextGatedDeltaNet(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.pretrained_config = config

        # tensor parallel
        tp_size = model_config.mapping.tp_size
        pp_size = model_config.mapping.pp_size
        if model_config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=model_config.mapping.rank,
            gpus_per_node=model_config.mapping.gpus_per_node,
            enable_attention_dp=model_config.mapping.enable_attention_dp,
        )
        self.mapping = mapping

        self.attn_tp_rank = mapping.tp_rank
        self.attn_tp_size = mapping.tp_size
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = Linear(
            self.conv_kernel_size,
            self.conv_dim,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False)

        self.in_proj_qkvz = Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim * 2,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False)
        self.in_proj_ba = Linear(
            self.hidden_size,
            self.num_v_heads * 2,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False)

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(
                (self.num_v_heads // self.attn_tp_size),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        A = torch.empty(divide(self.num_v_heads, self.attn_tp_size),
                        dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(
            torch.log(A),
            requires_grad=False,
        )
        self.A_log._no_weight_decay = True

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.cuda.current_device(),
            dtype=config.torch_dtype,
        )

        # gemmaNorm is not supported in fused_all_reduce kernel.
        # So, we need to do allReduce in Linear and do gemmaNorm in separate kernel.
        self.out_proj = Linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=model_config.get_quant_config(),
            reduce_output=True,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False)

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.attn_tp_size,
            (self.head_k_dim + self.head_k_dim +
             (self.head_v_dim + self.head_v_dim) * self.num_v_heads //
             self.num_k_heads),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.attn_tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz,
                                             split_arg_list_qkvz,
                                             dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.attn_tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.attn_tp_size)

        return query, key, value, z, b, a

    def forward_decode(
        self,
        conv_states,
        ssm_states,
        num_decodes,
        cu_seqlens,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        a = kwargs["a"]
        b = kwargs["b"]
        cache_indices = kwargs["cache_indices"]

        query_start_loc = torch.arange(0,
                                       num_decodes + 1,
                                       device=cu_seqlens.device).to(torch.long)

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            self.conv1d.weight,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.attn_tp_size,
                self.key_dim // self.attn_tp_size,
                self.value_dim // self.attn_tp_size,
            ],
            dim=-1,
        )
        # Reshape from [l, h*d] to [1, l, h, d]
        seq_len = query.shape[0]
        num_heads = query.shape[1] // self.head_k_dim
        query = query.view(1, seq_len, num_heads, self.head_k_dim)
        key = key.view(1, seq_len, num_heads, self.head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // self.head_v_dim,
                           self.head_v_dim)

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

        return core_attn_out

    def forward_extend(
        self,
        conv_states,
        ssm_states,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        a = kwargs["a"]
        b = kwargs["b"]
        batch_size = kwargs["batch_size"]
        has_initial_states = kwargs["has_initial_states"][:batch_size]
        cache_indices = kwargs["cache_indices"]
        query_start_loc = kwargs["query_start_loc"][:batch_size + 1]
        num_prefill_tokens = kwargs["num_prefill_tokens"]
        num_decode_tokens = kwargs["num_decode_tokens"]
        state_indices_p = kwargs["state_indices_p"]
        state_indices_d = kwargs["state_indices_d"]
        num_prefill = kwargs["num_prefill"]
        num_decode = kwargs["num_decode"]

        conv_states_to_use = conv_states

        seqlen_split_size = [num_prefill_tokens, num_decode_tokens]
        if num_decode_tokens > 0:
            mixed_qkv_p, mixed_qkv_d = torch.split(mixed_qkv,
                                                   seqlen_split_size,
                                                   dim=0)
            query_start_loc_p = query_start_loc[:num_prefill + 1]
            has_initial_states_p = has_initial_states[:num_prefill]

            mixed_qkv_p = causal_conv1d_fn(
                mixed_qkv_p.transpose(0, 1),
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_states_to_use,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_p,
                query_start_loc=query_start_loc_p,
            ).transpose(0, 1)

            mixed_qkv_d = causal_conv1d_update(
                mixed_qkv_d,
                conv_states_to_use,
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_state_indices=state_indices_d,
            )
            mixed_qkv = torch.cat((mixed_qkv_p, mixed_qkv_d), dim=0)
        else:
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_states_to_use,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            ).transpose(0, 1)

        key_split_dim = self.key_dim // self.attn_tp_size
        value_split_dim = self.value_dim // self.attn_tp_size

        query, key, value = torch.split(
            mixed_qkv,
            [key_split_dim, key_split_dim, value_split_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        num_heads = query.shape[1] // self.head_k_dim
        num_value_heads = value.shape[1] // self.head_v_dim

        query = query.view(1, actual_seq_len, num_heads, self.head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, self.head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = fused_gdn_gating(self.A_log, a, self.dt_bias)

        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)

        recurrent_state = ssm_states[cache_indices]

        if num_decode > 0:
            # TODO set it in mambaCacheManager
            decode_query_start_loc = torch.arange(
                1, num_decode + 1,
                device=query_start_loc.device)  # num_decode 个
            decode_query_start_loc = decode_query_start_loc + query_start_loc[
                num_prefill]
            new_query_start_loc = torch.cat(
                [query_start_loc[:num_prefill + 1], decode_query_start_loc])
        else:
            new_query_start_loc = query_start_loc

        new_query_start_loc = new_query_start_loc.to(torch.long)
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            cu_seqlens=new_query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        last_recurrent_state = last_recurrent_state.to(ssm_states.dtype,
                                                       copy=False)
        ssm_states[cache_indices] = last_recurrent_state

        return core_attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        ### sglang linear attn
        # has_initial_states = None
        # if forward_batch.extend_prefix_lens is not None:
        #     has_initial_states = forward_batch.extend_prefix_lens > 0

        # # Set up dimensions for reshapes later
        seq_len, _ = hidden_states.shape
        conv_state, recurrent_state = None, None

        ### mamba2_mixer layer
        # calculate split size
        num_prefills = attn_metadata.num_contexts
        num_decodes = attn_metadata.seq_lens.shape[0] - num_prefills
        num_prefill_tokens = attn_metadata.num_ctx_tokens
        num_decode_tokens = attn_metadata.num_tokens - num_prefill_tokens
        batch_split_size = [num_prefills, num_decodes]
        has_initial_states = mamba_metadata.has_initial_states

        state_indices = attn_metadata.kv_cache_manager.get_state_indices(
        )[:num_prefills + num_decodes]

        state_indices_p, state_indices_d = torch.split(state_indices,
                                                       batch_split_size)
        conv_states = attn_metadata.kv_cache_manager.get_conv_states(
            self.layer_idx)
        ssm_states = attn_metadata.kv_cache_manager.get_ssm_states(
            self.layer_idx)
        if num_prefills > 0:
            ssm_states[state_indices_p] = 0
            # conv_states[state_indices_p] = 0 # not necessary

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba)

        if self.num_v_heads // self.num_k_heads in [1, 2,
                                                    4]:  # and is_cuda_graph:
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                projected_states_qkvz,
                projected_states_ba,
                triton.cdiv(self.num_k_heads, self.attn_tp_size),
                triton.cdiv(self.num_v_heads, self.attn_tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba)
            query, key, value = map(lambda x: x.reshape(x.shape[0], -1),
                                    (query, key, value))
            mixed_qkv = torch.cat((query, key, value), dim=-1)

        kwargs = {
            "mixed_qkv": mixed_qkv,
            "a": a,
            "b": b,
            "z": z,
            "has_initial_states": has_initial_states,
            "cache_indices": state_indices,
            "query_start_loc": mamba_metadata.cu_seqlens,
            "batch_size": attn_metadata.seq_lens.shape[0],
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "state_indices_p": state_indices_p,
            "state_indices_d": state_indices_d,
            "num_prefill": num_prefills,
            "num_decode": num_decodes,
        }

        new_implementation = True
        if new_implementation:
            if num_prefills > 0:
                attn_out = self.forward_extend(conv_states, ssm_states,
                                               **kwargs)
            else:
                attn_out = self.forward_decode(conv_states, ssm_states,
                                               num_decodes,
                                               mamba_metadata.cu_seqlens,
                                               **kwargs)

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        attn_out = attn_out.reshape(-1, attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        attn_out = self.norm(attn_out, z)
        attn_out = attn_out.reshape(z_shape_og)
        attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)

        output = self.out_proj(attn_out, all_reduce_params=all_reduce_params)
        return output


class Qwen3NextLinearDecoderLayer(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config
        self.linear_attn = Qwen3NextGatedDeltaNet(model_config, layer_idx)

        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.mlp = Qwen3NextSparseMoeBlock(model_config,
                                           aux_stream,
                                           layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype,
                                       use_gemma=True)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype,
                                                use_gemma=True)
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy)
        self.next_layer_layernorm: RMSNorm = None

        self.fusion_config = EagerFusionConfig()
        ### TODO: enable eager_fusion by default
        self.enable_fusion = os.environ.get(
            "TRTLLM_QWEN3_EAGER_FUSION_DISABLED", "1") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        self.mapping.has_tp()
        has_pp = self.mapping.has_pp()

        # self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
        self.fusion_config.PRE_MOE_FUSION = False  # the fusion kernel does not support gemmaNorm yet
        self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION and not has_pp
        self.disable_attn_allreduce = (self.fusion_config.PRE_MOE_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)
        self.moe_allreduce = MoEAllReduce(mapping=model_config.mapping)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False
        # Linear Attention
        ### FIXME: 1. forward_batch; 2. allreduce
        if hidden_states.shape[0] != 0:
            hidden_states = self.linear_attn(
                hidden_states,
                attn_metadata,
                all_reduce_params=AllReduceParams(
                    enable_allreduce=not (self.fusion_config.PRE_MOE_FUSION
                                          or self.mapping.tp_size == 1)),
                **kwargs)
        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    enable_allreduce=not (self.fusion_config.PRE_MOE_FUSION
                                          or self.mapping.tp_size == 1),
                ))
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # Note: this fusion pattern is only supported for TRTLLM-nvfp4 backend now
        do_finalize = not (hidden_states.shape[0]
                           <= self.moe_allreduce.max_token
                           and self.fusion_config.POST_MOE_FUSION
                           and self.model_config.moe_backend == 'TRTLLM'
                           and self.mlp.experts.has_nvfp4)

        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                      or self.mapping.tp_size == 1)),
            do_finalize=do_finalize,
        )
        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
            else:
                assert len(
                    hidden_states
                ) == 3, f"hidden_states must have 3 elements, but got {len(hidden_states)}"

                fc2_output = hidden_states[0]
                expert_scale_factor = hidden_states[1]
                expanded_idx_to_permuted_idx = hidden_states[2]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=None,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params)

        else:
            if spec_metadata and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(
                    self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(
                    hidden_states, residual)
        return hidden_states, residual


class Qwen3NextAttention(Qwen3Attention):

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig],
                 layer_idx: int, fuse_qk_norm_rope: bool):
        super().__init__(model_config,
                         layer_idx,
                         fuse_qk_norm_rope=fuse_qk_norm_rope,
                         attn_output_gate=True,
                         use_gemma_rms_norm=True)


class Qwen3NextFullAttentionDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig],
                 layer_idx: int, aux_stream: torch.cuda.Stream):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config

        self.self_attn = Qwen3NextAttention(
            model_config,
            layer_idx=layer_idx,
            fuse_qk_norm_rope=False,
        )
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.mlp = Qwen3NextSparseMoeBlock(model_config,
                                           aux_stream,
                                           layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype,
                                       use_gemma=True)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype,
                                                use_gemma=True)
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy)
        self.next_layer_layernorm: RMSNorm = None

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = os.environ.get(
            "TRTLLM_QWEN3_EAGER_FUSION_DISABLED", "0") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        self.mapping.has_tp()
        has_pp = self.mapping.has_pp()

        # self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
        self.fusion_config.PRE_MOE_FUSION = False
        self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION and not has_pp
        self.disable_attn_allreduce = (self.fusion_config.PRE_MOE_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)
        self.moe_allreduce = MoEAllReduce(mapping=model_config.mapping)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_attn_allreduce),
            **kwargs,
        )

        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # Note: this fusion pattern is only supported for TRTLLM-nvfp4 backend now
        do_finalize = not (hidden_states.shape[0]
                           <= self.moe_allreduce.max_token
                           and self.fusion_config.POST_MOE_FUSION
                           and self.model_config.moe_backend == 'TRTLLM'
                           and self.mlp.experts.has_nvfp4)

        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                      or self.mapping.tp_size == 1)),
            do_finalize=do_finalize,
        )

        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
            else:
                assert len(
                    hidden_states
                ) == 3, f"hidden_states must have 3 elements, but got {len(hidden_states)}"

                fc2_output = hidden_states[0]
                expert_scale_factor = hidden_states[1]
                expanded_idx_to_permuted_idx = hidden_states[2]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=None,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params)

        else:
            if spec_metadata and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(
                    self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(
                    hidden_states, residual)

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "full_attention": Qwen3NextFullAttentionDecoderLayer,
    "linear_attention": Qwen3NextLinearDecoderLayer,
}


class Qwen3NextModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig]):
        super().__init__(model_config)
        config = self.model_config
        pretrained_config = self.model_config.pretrained_config
        self.aux_stream = torch.cuda.Stream()
        self.preload_weight_modules = []
        if config.moe_backend == "TRTLLM":
            self.preload_weight_modules = [
                "experts",
                "routing_method",
                "all_reduce",
            ]

        if model_config.mapping.enable_attention_dp:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            self.embed_tokens = Embedding(pretrained_config.vocab_size,
                                          pretrained_config.hidden_size,
                                          dtype=pretrained_config.torch_dtype)
        else:
            self.embed_tokens = Embedding(
                pretrained_config.vocab_size,
                pretrained_config.hidden_size,
                dtype=pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )

        self.layers = nn.ModuleList([
            ALL_DECODER_LAYER_TYPES[pretrained_config.layer_types[layer_idx]](
                model_config,
                layer_idx,
                self.aux_stream,
            ) for layer_idx in range(pretrained_config.num_hidden_layers)
        ])

        self.norm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype,
            use_gemma=True,
        )

        self.mamba_metadata: Optional[Mamba2Metadata] = None

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.mamba_metadata is None or self.mamba_metadata.max_batch_size != attn_metadata.max_num_requests:
            self.mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests,
                # chunk_size=self.model_config.pretrained_config.mamba2_chunk_size)
                # TODO check how to get the correct chunk_size
                chunk_size=128)
        self.mamba_metadata.prepare(attn_metadata)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                mamba_metadata=self.mamba_metadata)
        return hidden_states


@register_auto_model("Qwen3NextForCausalLM")
class Qwen3NextForCausalLM(SpecDecOneEngineForCausalLM[Qwen3NextModel,
                                                       Qwen3NextConfig]):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
    ):
        super().__init__(
            Qwen3NextModel(model_config),
            model_config,
        )
        self.preload_weight_modules = self.model.preload_weight_modules

    def load_weights(self, weights: dict, weight_mapper: BaseWeightMapper):
        new_weights = weight_mapper.preprocess_weights(weights)
        super().load_weights(new_weights, weight_mapper)

        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm
