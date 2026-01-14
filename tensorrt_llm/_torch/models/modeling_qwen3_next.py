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
from transformers import Qwen3NextConfig

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
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, EventType
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
        self.aux_stream = aux_stream

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

        self.event_dict = {
            key: torch.cuda.Event()
            for key in [EventType.Main, EventType.MoeShared]
        }

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

        def _compute_routed_output():
            router_logits = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                do_finalize=do_finalize,
            )
            return final_hidden_states

        def _compute_shared_output():
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = F.sigmoid(
                self.shared_expert_gate(hidden_states)) * shared_expert_output
            return shared_expert_output

        final_hidden_states, shared_expert_output = maybe_execute_in_parallel(
            _compute_routed_output,
            _compute_shared_output,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
        )
        if not do_finalize:
            return final_hidden_states

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
    batch_seq = batch * seq_len

    # Directly allocate output tensors in their final shapes (no intermediate buffers)
    mixed_qkv = torch.empty((batch_seq, qkv_dim_t),
                            dtype=mixed_qkvz.dtype,
                            device=mixed_qkvz.device)
    z = torch.empty((batch_seq, num_heads_v, head_v),
                    dtype=mixed_qkvz.dtype,
                    device=mixed_qkvz.device)
    b = torch.empty((batch_seq, num_heads_v),
                    dtype=mixed_ba.dtype,
                    device=mixed_ba.device)
    a = torch.empty((batch_seq, num_heads_v),
                    dtype=mixed_ba.dtype,
                    device=mixed_ba.device)
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

    def __init__(self,
                 model_config: ModelConfig[Qwen3NextConfig],
                 aux_stream: torch.cuda.Stream,
                 layer_idx: Optional[int] = None):
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

        self.event_dict = {
            key: torch.cuda.Event()
            for key in [EventType.Main, EventType.Attention]
        }
        self.aux_stream = aux_stream

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        batch_size = mixed_qkvz.size(0)
        num_k_heads_local = self.num_k_heads // self.attn_tp_size
        num_v_heads_local = self.num_v_heads // self.attn_tp_size
        heads_ratio = self.num_v_heads // self.num_k_heads

        # Reshape qkvz: [b, d] -> [b, ng, (2*hk + 2*np/ng*hv)]
        qkvz_dim_per_head = (self.head_k_dim * 2 +
                             self.head_v_dim * heads_ratio * 2)
        mixed_qkvz = mixed_qkvz.view(batch_size, num_k_heads_local,
                                     qkvz_dim_per_head)

        # Reshape ba: [b, d] -> [b, ng, 2*np/ng]
        mixed_ba = mixed_ba.view(batch_size, num_k_heads_local, heads_ratio * 2)

        # Direct slicing instead of torch.split for better performance
        # Compute split boundaries once
        q_end = self.head_k_dim
        k_end = q_end + self.head_k_dim
        v_end = k_end + heads_ratio * self.head_v_dim
        z_end = v_end + heads_ratio * self.head_v_dim

        # Slice qkvz components: [b, ng, dim] -> individual components
        query = mixed_qkvz[..., :q_end]
        key = mixed_qkvz[..., q_end:k_end]

        # Optimize: Use view (zero-copy) instead of reshape for contiguous slices
        # Layout: [v_concat | z_concat], need to reshape each separately
        value = mixed_qkvz[..., k_end:v_end].view(batch_size, num_v_heads_local,
                                                  self.head_v_dim)
        z = mixed_qkvz[..., v_end:z_end].view(batch_size, num_v_heads_local,
                                              self.head_v_dim)

        # Slice ba components: [b, ng, 2*np/ng] -> [b, np] each
        # Optimize: Use view instead of reshape (zero-copy for contiguous data)
        b = mixed_ba[..., :heads_ratio].view(batch_size, num_v_heads_local)
        a = mixed_ba[..., heads_ratio:].view(batch_size, num_v_heads_local)

        return query, key, value, z, b, a

    def forward_decode(
        self,
        conv_states,
        ssm_states,
        query_start_loc_long,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        a = kwargs["a"]
        b = kwargs["b"]
        cache_indices = kwargs["cache_indices"]

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            self.conv1d.weight,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=cache_indices,
        )

        # Direct slicing instead of torch.split for better performance
        key_size = self.key_dim // self.attn_tp_size
        query = mixed_qkv[..., :key_size]
        key = mixed_qkv[..., key_size:key_size * 2]
        value = mixed_qkv[..., key_size * 2:]
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
            cu_seqlens=query_start_loc_long,
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
        query_start_loc = kwargs["query_start_loc"]
        query_start_loc_long = kwargs["query_start_loc_long"]
        num_prefill_tokens = kwargs["num_prefill_tokens"]
        num_decode_tokens = kwargs["num_decode_tokens"]
        state_indices_p = kwargs["state_indices_p"]
        state_indices_d = kwargs["state_indices_d"]
        num_prefill = kwargs["num_prefill"]

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
                query_start_loc=query_start_loc).transpose(0, 1)

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

        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            cu_seqlens=query_start_loc_long,
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
            ssm_states[state_indices_p] = torch.zeros((),
                                                      dtype=ssm_states.dtype,
                                                      device=ssm_states.device)

        def _compute_projected_states_qkvz():
            return self.in_proj_qkvz(hidden_states)

        def _compute_projected_states_ba():
            return self.in_proj_ba(hidden_states)

        projected_states_qkvz, projected_states_ba = maybe_execute_in_parallel(
            _compute_projected_states_qkvz,
            _compute_projected_states_ba,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.Attention],
            self.aux_stream,
        )

        # Use fused kernel when possible to avoid elementwise ops
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
            "query_start_loc": mamba_metadata.query_start_loc,
            "query_start_loc_long": mamba_metadata.query_start_loc_long,
            "batch_size": attn_metadata.seq_lens.shape[0],
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "state_indices_p": state_indices_p,
            "state_indices_d": state_indices_d,
            "num_prefill": num_prefills,
        }
        if num_prefills > 0:
            attn_out = self.forward_extend(conv_states, ssm_states, **kwargs)
        else:
            attn_out = self.forward_decode(conv_states, ssm_states, **kwargs)

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        attn_out = attn_out.reshape(-1, attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        attn_out = self.norm(attn_out, z)
        attn_out = attn_out.reshape(z_shape_og)
        attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)

        output = self.out_proj(attn_out, all_reduce_params=all_reduce_params)
        return output


class Qwen3NextLinearDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config
        self.linear_attn = Qwen3NextGatedDeltaNet(model_config, aux_stream,
                                                  layer_idx)

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

        # has_tp = self.mapping.has_tp()
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

        # has_tp = self.mapping.has_tp()
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

    def post_load_weights(self):
        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm
