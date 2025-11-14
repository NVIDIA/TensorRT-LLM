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

import re
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, PretrainedConfig
from transformers.activations import ACT2FN

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.mamba.mamba2_mixer import Mamba2Mixer
from ..modules.mlp import MLP
from ..modules.linear import Linear
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)

SHOW_LAYER_IDX = 1


def split(x: torch.Tensor,
          tp_size: int,
          idx: int,
          dim: int = 0) -> torch.Tensor:
    assert x.shape[dim] % tp_size == 0
    split_size = x.shape[dim] // tp_size
    if tp_size == 1:
        return x
    return torch.split(x, split_size, dim=dim)[idx]


def relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))


class NemotronHConfig(PretrainedConfig):
    model_type = "nemotron_h"


class MLPLayer(MLP):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[layer_idx]
        else:
            intermediate_size = config.intermediate_size

        super().__init__(hidden_size=config.hidden_size,
                         intermediate_size=intermediate_size,
                         bias=False,
                         activation=relu2,
                         dtype=config.torch_dtype,
                         config=model_config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(hidden_states)


class TransformerLayer(Attention):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=None,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(position_ids=None,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata)


class NemotronHNativeMOE(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.experts = nn.ModuleList(
            [
                NemotronHMLP(config, intermediate_size=config.moe_intermediate_size, layer_idx=layer_idx)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = NemotronHTopkRouter(config, layer_idx=layer_idx)
        self.shared_experts = NemotronHMLP(
            config=config, intermediate_size=config.moe_shared_expert_intermediate_size, layer_idx=layer_idx
        )
        self.dtype = config.torch_dtype

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor, show: bool = False):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        if show:
            print("="*100)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
                if show:
                    print("="*100)
                    print(f"NemotronHMOE expert_idx: {expert_idx}")
                    print(f"NemotronHMOE token_indices: {token_indices.shape=} \n{token_indices.dtype=} \n{token_indices.device=} \n{token_indices=!r}")
                    print(f"NemotronHMOE weight_indices: {weight_indices.shape=} \n{weight_indices.dtype=} \n{weight_indices.device=} \n{weight_indices=!r}")
                    print(f"NemotronHMOE expert_weights: {expert_weights.shape=} \n{expert_weights.dtype=} \n{expert_weights.device=} \n{expert_weights=!r}")
                    print(f"NemotronHMOE expert_input: {expert_input.shape=} \n{expert_input.dtype=} \n{expert_input.device=} \n{expert_input=!r}")
                    print(f"NemotronHMOE expert_output: {expert_output.shape=} \n{expert_output.dtype=} \n{expert_output.device=} \n{expert_output=!r}")
                    print(f"NemotronHMOE weighted_output: {weighted_output.shape=} \n{weighted_output.dtype=} \n{weighted_output.device=} \n{weighted_output=!r}")
                    print("="*100)

        if show:
            print("="*100)
            print(f"NemotronHMOE final_hidden_states: {final_hidden_states.shape=} \n{final_hidden_states.dtype=} \n{final_hidden_states.device=} \n{final_hidden_states=!r}")
            print("="*100)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def _forward(self, hidden_states, attn_metadata, **kwargs):
        residuals = hidden_states

        shared_expert_output = self.shared_experts(residuals)

        show = hidden_states.shape[1] == 6 and self.layer_idx == SHOW_LAYER_IDX
        show = False
        if show:
            print("="*100)
            print(f"shared_expert_output: {shared_expert_output.shape=} \n{shared_expert_output.dtype=} \n{shared_expert_output.device=} \n{shared_expert_output=!r}")
            print("="*100)


        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        hidden_states = self.moe(hidden_states, topk_indices, topk_weights, show).view(*orig_shape)

        hidden_states = hidden_states + shared_expert_output
        return hidden_states

    def forward(self, hidden_states, attn_metadata, **kwargs):
        hidden_states = hidden_states.unsqueeze(0)
        hidden_states = self._forward(hidden_states, attn_metadata, **kwargs)
        hidden_states = hidden_states.squeeze(0)
        return hidden_states


from ..modules.linear import Linear
from ..modules.fused_moe import (CutlassFusedMoE, RenormalizeMoeRoutingMethod,
                                 VanillaMoE, create_moe)
from ..modules.mlp import MLP
from ..modules.fused_moe.interface import MoEWeightLoadingMode
from ..modules.fused_moe import (DeepSeekV3MoeRoutingMethod,
                                 MoEWeightLoadingMode, create_moe)
from tensorrt_llm.llmapi.utils import enable_llm_debug
import warnings
from typing import Dict, List, Optional, Tuple


class Deepseekv3RoutingImpl():

    def __init__(
        self,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        is_fused: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        self.topk_group = topk_group
        self.n_group = n_group
        self.routed_scaling_factor = routed_scaling_factor
        self.is_fused = is_fused

    @staticmethod
    @torch.compile(options={"max-autotune": True})
    def get_scores(logits, e_score_correction_bias):
        scores = F.sigmoid(logits)
        scores_with_bias = scores + e_score_correction_bias
        return scores, scores_with_bias

    def noaux_tc(self, logits, e_score_correction_bias, show: bool = False):
        n_group = self.n_group

        if enable_llm_debug():
            has_nan = torch.isnan(scores_with_bias).any()
            if has_nan:
                warnings.warn(
                    "Detected NAN in the tensor scores_with_bias. Please check if it matches the expectation."
                )

        _, num_experts = logits.shape
        if self.n_group > 1:
            if self.top_k > 8 or (num_experts / n_group) > 32 or (
                    num_experts / n_group) * self.topk_group > 128:
                if (self.is_fused):
                    warnings.warn(
                        "The configuration is not supported by the fused routing kernel. We have to use the original pytorch implementation."
                    )
                self.is_fused = False
        else:
            if num_experts > 384 or self.top_k > 8:
                if (self.is_fused):
                    warnings.warn(
                        "The configuration is not supported by the fused routing kernel. We have to use the original pytorch implementation."
                    )
                self.is_fused = False

        if show:
            print("="*100)
            print(f"DSV3 is_fused: {self.is_fused}")
            print(f"DSV3 Router logits: {logits.shape=} \n{logits.dtype=} \n{logits.device=} \n{logits=!r}")
            print(f"DSV3 Router e_score_correction_bias: {e_score_correction_bias.shape=} \n{e_score_correction_bias.dtype=} \n{e_score_correction_bias.device=} \n{e_score_correction_bias=!r}")
            print("="*100)

        if not self.is_fused:
            scores, scores_with_bias = Deepseekv3RoutingImpl.get_scores(
                logits, e_score_correction_bias)

            if show:
                print("="*100)
                print(f"DSV3 scores: {scores.shape=} \n{scores.dtype=} \n{scores.device=} \n{scores=!r}")
                print(f"DSV3 scores_with_bias: {scores_with_bias.shape=} \n{scores_with_bias.dtype=} \n{scores_with_bias.device=} \n{scores_with_bias=!r}")
                print("="*100)

            scores_shape = list(scores_with_bias.shape)
            group_scores = torch.sum(torch.topk(
                scores_with_bias.view(scores_shape[:-1] +
                                      [n_group, scores_shape[-1] // n_group]),
                k=2,
                dim=-1,
                largest=True,
                sorted=True)[0],
                                     dim=-1)
            if show:
                print("="*100)
                print(f"DSV3 group_scores: {group_scores.shape=} \n{group_scores.dtype=} \n{group_scores.device=} \n{group_scores=!r}")
                print("="*100)
            _, group_idx = torch.topk(group_scores,
                                      k=self.topk_group,
                                      dim=-1,
                                      largest=True,
                                      sorted=True)
            if show:
                print("="*100)
                print(f"DSV3 group_idx: {group_idx.shape=} \n{group_idx.dtype=} \n{group_idx.device=} \n{group_idx=!r}")
                print("="*100)
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(-1, group_idx, 1)
            if show:
                print("="*100)
                print(f"DSV3 group_mask: {group_mask.shape=} \n{group_mask.dtype=} \n{group_mask.device=} \n{group_mask=!r}")
                print("="*100)
            score_mask = group_mask.unsqueeze(-1).expand(
                scores_shape[:-1] +
                [n_group, scores_shape[-1] // n_group]).reshape(scores_shape)
            if show:
                print("="*100)
                print(f"DSV3 score_mask: {score_mask.shape=} \n{score_mask.dtype=} \n{score_mask.device=} \n{score_mask=!r}")
                print("="*100)
            scores_with_bias = scores_with_bias * score_mask
            if show:
                print("="*100)
                print(f"DSV3 scores_with_bias after score_mask: {scores_with_bias.shape=} \n{scores_with_bias.dtype=} \n{scores_with_bias.device=} \n{scores_with_bias=!r}")
                print("="*100)
            _, topk_idx = torch.topk(scores_with_bias,
                                     k=self.top_k,
                                     dim=-1,
                                     largest=True,
                                     sorted=True)
            if show:
                print("="*100)
                print(f"DSV3 topk_idx: {topk_idx.shape=} \n{topk_idx.dtype=} \n{topk_idx.device=} \n{topk_idx=!r}")
                print("="*100)

            ###### End get_topk_indices ######

            new_mask = torch.zeros_like(scores)
            new_mask.scatter_(-1, topk_idx, 1)
            scores = scores * new_mask

            if show:
                print("="*100)
                print(f"DSV3 scores after new_mask: {scores.shape=} \n{scores.dtype=} \n{scores.device=} \n{scores=!r}")
                print("="*100)

            score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
            scores = scores / score_sum * \
                self.routed_scaling_factor

            if show:
                print("="*100)
                print(f"DSV3 scores after routed_scaling_factor: {scores.shape=} \n{scores.dtype=} \n{scores.device=} \n{scores=!r}")
                print("="*100)

            topk_values, topk_indices = torch.topk(scores,
                                                   k=self.top_k,
                                                   dim=-1,
                                                   largest=True)
            if show:
                print("="*100)
                print(f"DSV3 topk_values: {topk_values.shape=} \n{topk_values.dtype=} \n{topk_values.device=} \n{topk_values=!r}")
                print(f"DSV3 topk_indices: {topk_indices.shape=} \n{topk_indices.dtype=} \n{topk_indices.device=} \n{topk_indices=!r}")
                print("="*100)
            return topk_values, topk_indices
        else:
            # TODO (williamj): Change the codes in this block.
            topk_values, topk_indices = torch.ops.trtllm.noaux_tc_op(
                logits, e_score_correction_bias, n_group, self.topk_group,
                self.top_k, self.routed_scaling_factor)

            if show:
                print("="*100)
                print(f"DSV3 topk_values: {topk_values.shape=} \n{topk_values.dtype=} \n{topk_values.device=} \n{topk_values=!r}")
                print(f"DSV3 topk_indices: {topk_indices.shape=} \n{topk_indices.dtype=} \n{topk_indices.device=} \n{topk_indices=!r}")
                print("="*100)

            return topk_values, topk_indices

    def apply(
        self, logits: torch.Tensor, e_score_correction_bias: torch.Tensor, show: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_values, topk_indices = self.noaux_tc(logits,
                                                  e_score_correction_bias, show)
        return topk_indices.to(torch.int32), topk_values.to(torch.float32)


class DeepseekV3Gate(DeepSeekV3MoeRoutingMethod):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        dtype: Optional[torch.dtype] = None,
        fuse_routing_kernel: bool = True,
        apply_routing: bool = False,
        moe_backend: str = 'CUTLASS',
    ):
        super().__init__(top_k=top_k)
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size),
                                               dtype=dtype),
                                   requires_grad=False)
        self.moe_backend = moe_backend
        if moe_backend == 'TRTLLM':
            bias_dtype = torch.bfloat16
        else:
            bias_dtype = torch.float32

        self.e_score_correction_bias = nn.Parameter(torch.empty((num_experts), dtype=bias_dtype), requires_grad=False)

        assert not apply_routing, "DeepseekV3Gate routing is called inside MoE"

        # TODO: e_score_correction_bias belongs in this gate class but is required by the routing impl.
        # To avoid weight-loading issues, we treat this gate as the BaseMoeRoutingMethod and dispatch to the routing impl.
        # This is a temporary hack that should be refactored later.
        self.routing_impl = Deepseekv3RoutingImpl(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            is_fused=fuse_routing_kernel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.ops.trtllm.dsv3_router_gemm_op(hidden_states,
                                                      self.weight.t(),
                                                      bias=None,
                                                      out_dtype=torch.float32)
        return logits

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1

        self.weight.copy_(weights[0]["weight"][:])

        self.e_score_correction_bias.data.copy_(
            weights[0]["e_score_correction_bias"][:].to(
                self.e_score_correction_bias.dtype))

    def apply(self, logits: torch.Tensor, show: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        # topk routing
        return self.routing_impl.apply(logits, self.e_score_correction_bias, show)

    @property
    def routing_method(self) -> DeepSeekV3MoeRoutingMethod:
        return self

    def get_experts_per_token(self):
        return self.routing_impl.top_k




class NemotronHMOE(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        aux_stream: torch.cuda.Stream=None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.layer_idx = layer_idx
        self.moe_intermediate_size = config.moe_intermediate_size[0] \
            if isinstance(config.moe_intermediate_size, list) else config.moe_intermediate_size

        self.use_latent_moe: bool = getattr(config, "moe_latent_size", None) is not None
        self.moe_hidden_size: int = config.moe_latent_size if self.use_latent_moe else config.hidden_size
        self.mlp_bias = config.mlp_bias if hasattr(config, 'mlp_bias') else False

        self.moe_n_group = config.n_group
        self.num_experts = config.n_routed_experts

        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.n_shared_experts

        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp

        reduce_results = True  # Need to be True for VanillaMoE.

        self.routed_scaling_factor = config.routed_scaling_factor

        # Setup shared expert MLP.
        if config.n_shared_experts is None or config.n_shared_experts == 0:
            self.shared_experts = None
        else:
            shared_expert_intermediate_size = (
                config.moe_shared_expert_intermediate_size * config.n_shared_experts
            )
            self.shared_experts = MLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_expert_intermediate_size,
                bias=self.mlp_bias,
                activation=relu2,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=self.layer_idx,
            )

        self.gate = DeepseekV3Gate(
            self.hidden_size,
            self.num_experts,
            top_k=self.top_k,
            n_group=self.moe_n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            dtype=config.torch_dtype,
            fuse_routing_kernel=True,
            apply_routing=False,
            moe_backend=model_config.moe_backend)

        self.experts = create_moe(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=self.moe_intermediate_size,
            # aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            dtype=config.torch_dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            layer_idx=self.layer_idx,
            # Default values
            override_quant_config=None,
            aux_stream_dict=None,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA if model_config.moe_backend == 'VANILLA' else MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
            bias=self.mlp_bias,
            apply_router_weight_on_input=False,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
        )

        if self.use_latent_moe:
            self.fc1_latent_proj = Linear(
                in_features=self.hidden_size,
                out_features=self.moe_hidden_size,
                bias=self.mlp_bias,
                dtype=config.torch_dtype,
            )
            self.fc2_latent_proj = Linear(
                in_features=self.moe_hidden_size,
                out_features=self.hidden_size,
                bias=self.mlp_bias,
                dtype=config.torch_dtype,
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:

        show = hidden_states.shape[0] == 6 and self.layer_idx == SHOW_LAYER_IDX
        show = False
        if show:
            print("="*100)
            print(f"MOE input hidden_states: {hidden_states.shape=} \n{hidden_states.dtype=} \n{hidden_states.device=} \n{hidden_states=!r}")
            print("="*100)

        assert hidden_states.shape[-1] == self.hidden_dim
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts(hidden_states)
        else:
            shared_expert_output = 0

        if show:
            print("="*100)
            print(f"MOE shared_expert_output: {shared_expert_output.shape=} \n{shared_expert_output.dtype=} \n{shared_expert_output.device=} \n{shared_expert_output=!r}")
            print("="*100)

        router_logits = self.gate(hidden_states)

        if show:
            print("="*100)
            print(f"MOE router_logits: {router_logits.shape=} \n{router_logits.dtype=} \n{router_logits.device=} \n{router_logits=!r}")
            print(f"attn_metadata.all_rank_num_tokens: {attn_metadata.all_rank_num_tokens=}")
            print("="*100)

        if self.use_latent_moe:
            hidden_states = self.fc1_latent_proj(hidden_states)

        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        final_hidden_states = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=False)

        if self.use_latent_moe:
            final_hidden_states = self.fc2_latent_proj(final_hidden_states)

        final_hidden_states = shared_expert_output + final_hidden_states

        return final_hidden_states.view(orig_shape)


class NemotronHMLP(nn.Module):
    def __init__(self, config, intermediate_size=None, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.layer_idx = layer_idx
        # if layer_idx is None:
        #     logger.warning_once(
        #         f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
        #         "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
        #         "when creating this class."
        #     )
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, dtype=self.dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias, dtype=self.dtype)
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class NemotronHTopkRouter(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.dtype = config.torch_dtype
        self.layer_idx = layer_idx

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.n_routed_experts, dtype=torch.float32))

    @torch.no_grad()
    def get_topk_indices(self, scores, show: bool = False):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        if show:
            print("="*100)
            print(f"Router scores_for_choice: {scores_for_choice.shape=} \n{scores_for_choice.dtype=} \n{scores_for_choice.device=} \n{scores_for_choice=!r}")
            print("="*100)

        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )

        if show:
            print("="*100)
            print(f"Router group_scores: {group_scores.shape=} \n{group_scores.dtype=} \n{group_scores.device=} \n{group_scores=!r}")
            print("="*100)

        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]

        if show:
            print("="*100)
            print(f"Router group_idx: {group_idx.shape=} \n{group_idx.dtype=} \n{group_idx.device=} \n{group_idx=!r}")
            print("="*100)

        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        if show:
            print("="*100)
            print(f"Router group_mask: {group_mask.shape=} \n{group_mask.dtype=} \n{group_mask.device=} \n{group_mask=!r}")
            print("="*100)

        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )

        if show:
            print("="*100)
            print(f"Router score_mask: {score_mask.shape=} \n{score_mask.dtype=} \n{score_mask.device=} \n{score_mask=!r}")
            print("="*100)

        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        if show:
            print("="*100)
            print(f"Router scores_for_choice after masked_fill: {scores_for_choice.shape=} \n{scores_for_choice.dtype=} \n{scores_for_choice.device=} \n{scores_for_choice=!r}")
            print("="*100)

        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]


        if show:
            print("="*100)
            print(f"Router topk_indices: {topk_indices.shape=} \n{topk_indices.dtype=} \n{topk_indices.device=} \n{topk_indices=!r}")
            print("="*100)

        return topk_indices

    def forward(self, hidden_states):

        show = hidden_states.shape[1] == 6 and self.layer_idx == SHOW_LAYER_IDX
        show = False
        if show:
            print("="*100)
            print(f"Router input hidden_states: {hidden_states.shape=} \n{hidden_states.dtype=} \n{hidden_states.device=} \n{hidden_states=!r}")
            print("="*100)

        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        if show:
            print("="*100)
            print(f"Router router_logits: {router_logits.shape=} \n{router_logits.dtype=} \n{router_logits.device=} \n{router_logits=!r}")
            print("="*100)


        scores = router_logits.sigmoid()

        if show:
            print("="*100)
            print(f"Router scores: {scores.shape=} \n{scores.dtype=} \n{scores.device=} \n{scores=!r}")
            print("="*100)

        topk_indices = self.get_topk_indices(scores, show)
        if show:
            print("="*100)
            print(f"Router topk_indices: {topk_indices.shape=} \n{topk_indices.dtype=} \n{topk_indices.device=} \n{topk_indices=!r}")
            print("="*100)

        topk_weights = scores.gather(1, topk_indices)
        if show:
            print("="*100)
            print(f"Router topk_weights: {topk_weights.shape=} \n{topk_weights.dtype=} \n{topk_weights.device=} \n{topk_weights=!r}")
            print("="*100)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator


        if show:
            print("="*100)
            print(f"Router topk_weights after norm_topk_prob: {topk_weights.shape=} \n{topk_weights.dtype=} \n{topk_weights.device=} \n{topk_weights=!r}")
            print("="*100)

        topk_weights = topk_weights * self.routed_scaling_factor

        if show:
            print("="*100)
            print(f"Router topk_weights after routed_scaling_factor: {topk_weights.shape=} \n{topk_weights.dtype=} \n{topk_weights.device=} \n{topk_weights=!r}")
            print("="*100)

        topk_weights = topk_weights.to(self.dtype)
        return topk_indices, topk_weights


class NemotronHLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
        # M -> MambaLayer
        # - -> MLPLayer
        # * -> TransformerLayer
        layer_type: str,
    ):
        super().__init__()

        config = model_config.pretrained_config

        self.layer_idx = layer_idx
        self.layer_type = layer_type

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        if layer_type == "M":
            self.mixer = Mamba2Mixer(d_model=config.hidden_size,
                                     d_state=config.ssm_state_size,
                                     d_conv=config.conv_kernel,
                                     nheads=config.mamba_num_heads,
                                     n_groups=config.n_groups,
                                     head_dim=config.mamba_head_dim,
                                     chunk_size=config.chunk_size,
                                     layer_idx=layer_idx,
                                     rms_norm_eps=config.rms_norm_eps,
                                     dtype=config.torch_dtype,
                                     config=model_config)
        elif layer_type == "-":
            self.mixer = MLPLayer(model_config, layer_idx)
        elif layer_type == "*":
            self.mixer = TransformerLayer(model_config, layer_idx)
        elif layer_type == "E":
            use_native_moe = False
            if use_native_moe:
                self.mixer = NemotronHNativeMOE(config, layer_idx=layer_idx)
            else:
                self.mixer = NemotronHMOE(model_config, layer_idx=layer_idx)
        else:
            raise ValueError(f"{layer_type} is not supported")

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        show: bool = False,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        # hidden_states_shape = hidden_states.shape

        if show:
            print("="*100)
            print(f"NH layer 2 input hidden_states: {hidden_states.shape=} \n{hidden_states.dtype=} \n{hidden_states.device=} \n{hidden_states=!r}")
            print("="*100)

        hidden_states = self.norm(hidden_states, show=show)

        if show:
            print("="*100)
            print(f"NH layer 3 after norm hidden_states: {hidden_states.shape=} \n{hidden_states.dtype=} \n{hidden_states.device=} \n{hidden_states=!r}")
            print("="*100)

        # if self.layer_idx == 1 and hidden_states_shape[0] == 6:
        #     save_path = "/code/wj-models/Nemotron-Nano-3-30B-A3.5B-dev-1024/hidden_states.pt"
        #     # Detach and move to cpu for saving
        #     # hidden_states_to_save = hidden_states.detach().cpu()
        #     # torch.save(hidden_states_to_save, save_path)
        #     # Now reload from disk, and move back to the original device and dtype
        #     loaded_hidden_states = torch.load(save_path, map_location=hidden_states.device)
        #     loaded_hidden_states = loaded_hidden_states.to(hidden_states.dtype).to(hidden_states.device)
        #     # Overwrite hidden_states
        #     hidden_states = loaded_hidden_states.squeeze(0)
        #     # print("="*100)
        #     # print(f"after norm hidden_states: {hidden_states.shape=} \n{hidden_states.dtype=} \n{hidden_states.device=} \n{hidden_states=!r}")
        #     # print("="*100)


        hidden_states = self.mixer(hidden_states, attn_metadata, **kwargs)

        # if self.layer_idx == 1 and hidden_states_shape[0] == 6:
        #     print("="*100)
        #     print(f"after mixer hidden_states: {hidden_states.shape=} \n{hidden_states.dtype=} \n{hidden_states.device=} \n{hidden_states=!r}")
        #     print("="*100)

        hidden_states = torch.add(hidden_states, residual)

        # if self.layer_idx == 1 and hidden_states_shape[0] == 6:
        #     print("="*100)
        #     print(f"final output hidden_states: {hidden_states.shape=} \n{hidden_states.dtype=} \n{hidden_states.device=} \n{hidden_states=!r}")
        #     print("="*100)

        return hidden_states


class NemotronHModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[NemotronHConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config

        # calculate embeddings
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        # create layers
        layers = []
        for layer_idx, layer_type in enumerate(config.hybrid_override_pattern):
            layers.append(NemotronHLayer(model_config, layer_idx, layer_type))
        self.layers = nn.ModuleList(layers)

        # final norm
        self.norm_f = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        self.mamba_metadata: Optional[Mamba2Metadata] = None

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.mamba_metadata is None or self.mamba_metadata.max_batch_size != attn_metadata.max_num_requests:
            self.mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests,
                chunk_size=self.model_config.pretrained_config.chunk_size)
        self.mamba_metadata.prepare(attn_metadata)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds


        target_layer_idx = SHOW_LAYER_IDX
        for layer_idx, layer in enumerate(self.layers):
            show = target_layer_idx == layer_idx and hidden_states.shape[0] == 6
            show = False
            if show:
                print("="*100)
                print(f"1 input {layer_idx=} {layer.layer_type=} \n hidden_states: {hidden_states.shape=} \n{hidden_states.dtype=} \n{hidden_states.device=} \n{hidden_states=!r}")

            hidden_states = layer(position_ids,
                                  hidden_states,
                                  attn_metadata,
                                  mamba_metadata=self.mamba_metadata, show=show)
            if show:
                print()
                print(f"2 output hidden_states: {hidden_states.shape=} \n{hidden_states.dtype=} \n{hidden_states.device=} \n{hidden_states=!r}")
                print("="*100)

        hidden_states = self.norm_f(hidden_states)

        return hidden_states


@register_auto_model("NemotronHForCausalLM")
class NemotronHForCausalLM(DecoderModelForCausalLM[NemotronHModel,
                                                   NemotronHConfig]):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
    ):
        # rms_norm_eps is with a different name in the config.
        if hasattr(model_config.pretrained_config, "layer_norm_epsilon"):
            rms_epsilon = model_config.pretrained_config.layer_norm_epsilon
        elif hasattr(model_config.pretrained_config, "rms_norm_eps"):
            rms_epsilon = model_config.pretrained_config.rms_norm_eps
        else:
            raise ValueError("layer_norm_epsilon or rms_norm_eps is not set")
        model_config.pretrained_config.rms_norm_eps = rms_epsilon

        if not model_config.mapping.tp_size in [1, 2, 4, 8]:
            raise ValueError("TP has to be either 1, 2, 4 or 8")

        if model_config.quant_config.exclude_modules is not None:
            model_config.quant_config.exclude_modules = [
                re.sub(r'(model\.layers\.)?backbone', 'model', k)
                for k in model_config.quant_config.exclude_modules
            ]

        super().__init__(
            NemotronHModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: dict, weight_mapper: BaseWeightMapper):
        new_weights = weight_mapper.preprocess_weights(weights)


        '''
====================================================================================================
model.layers.15.mixer.gate.e_score_correction_bias: torch.Size([128])
model.layers.15.mixer.experts.0.down_proj.weight: torch.Size([2688, 1856])
model.layers.15.mixer.experts.0.up_proj.weight: torch.Size([1856, 2688])
...
model.layers.15.mixer.gate.weight: torch.Size([128, 2688])
model.layers.15.mixer.shared_experts.down_proj.weight: torch.Size([2688, 3712])
model.layers.15.mixer.shared_experts.up_proj.weight: torch.Size([3712, 2688])
====================================================================================================
        '''
        '''
    (15): NemotronHLayer(
      (norm): RMSNorm()
      (mixer): NemotronHMOE(
        (shared_experts): MLP(
          (up_lora): LoraLayer()
          (up_proj): Linear(
            (all_reduce): AllReduce()
            (lora): LoraLayer()
          )
          (down_lora): LoraLayer()
          (down_proj): Linear(
            (all_reduce): AllReduce()
            (lora): LoraLayer()
          )
        )
        (gate): Linear(
          (all_reduce): AllReduce()
        )
        (experts): VanillaMoE(
          (0): RenormalizeMoeRoutingMethod()
          (1): AllReduce()
          (2-129): 128 x GatedMLP(
            (gate_up_proj): Linear()
            (down_lora): LoraLayer()
            (down_proj): Linear(
              (lora): LoraLayer()
            )
            (splitted_gate_up_lora): LoraLayer()
            (fused_gate_up_lora): LoraLayer()
          )
        )
      )
    )
        '''


        # # Debug
        # print("="*100)
        # for key in new_weights.keys():
        #     if "model.layers.15.mixer" in key:
        #         if "_scale" in key:
        #             print(f"{key}: {new_weights[key].shape} {new_weights[key]=!r}")
        #         else:
        #             print(f"{key}: {new_weights[key].shape}")
        # print("="*100)


        # print("="*100)
        # print(f"model: {self.model}")
        # print("="*100)


        super().load_weights(new_weights, weight_mapper)


AutoConfig.register(NemotronHConfig.model_type, NemotronHConfig)
