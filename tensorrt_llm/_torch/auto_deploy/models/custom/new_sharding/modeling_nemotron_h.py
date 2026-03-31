# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Nemotron-H model with explicit sharding hint ops.

This is a rewrite of the original modeling_nemotron_h.py where all shardable
operations use AutoDeploy custom ops with sharding hint kwargs.  The graph
produced by this model is a complete, self-contained specification of "how this
model should be sharded."  The ``apply_sharding_hints`` transform reads the
hints together with a runtime ``Mapping`` to apply deterministic, node-local
sharding transformations.

Shardable custom ops used (each carries a ``layer_type`` hint as the last kwarg):
  - torch.ops.auto_deploy.torch_linear_simple  (tp_mode, output_sizes, layer_type)
  - torch.ops.auto_deploy.view                 (tp_scaled_dim, layer_type)
  - torch.ops.auto_deploy.split_with_sizes     (shardable, layer_type)
  - torch.ops.auto_deploy.all_reduce           (layer_type)
  - torch.ops.auto_deploy.torch_causal_conv1d  (shardable, layer_type)
  - torch.ops.auto_deploy.torch_ssm            (shardable, layer_type)
  - torch.ops.auto_deploy.torch_rmsnorm_gated  (tp_mode, layer_type)
  - torch.ops.auto_deploy.torch_attention      (sharding-invariant, no hints needed)
  - torch.ops.auto_deploy.torch_moe            (layer_type; sharded by apply_sharding_hints)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register all ops
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType


class MambaRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, group_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, hidden_states, gate=None):
        return torch.ops.auto_deploy.torch_rmsnorm_gated(
            x=hidden_states,
            weight=self.weight,
            gate=gate,
            eps=self.variance_epsilon,
            group_size=self.group_size,
            norm_before_gate=False,
            tp_mode="colwise",
            layer_type="ssm",
        )


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.num_heads = config.mamba_num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.mamba_hidden_act
        self.act = ACT2FN[config.mamba_hidden_act]

        self.layer_norm_epsilon = config.layer_norm_epsilon

        self.n_groups = config.n_groups
        self.head_dim = config.mamba_head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=self.layer_norm_epsilon,
            group_size=self.intermediate_size // self.n_groups,
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        # Precompute the 5-part fused output sizes for in_proj sharding.
        # in_proj output = [gate | conv_input (= hidden + B + C) | dt]
        # conv_input after conv1d+silu splits into [hidden | B | C]
        # True fused dims: [gate, hidden, B, C, dt]
        self._in_proj_output_sizes = [
            self.intermediate_size,
            self.intermediate_size,
            self.n_groups * self.ssm_state_size,
            self.n_groups * self.ssm_state_size,
            self.num_heads,
        ]
        # Conv1d fused output sizes: channels = [hidden | B | C]
        self._conv1d_output_sizes = [
            self.intermediate_size,
            self.n_groups * self.ssm_state_size,
            self.n_groups * self.ssm_state_size,
        ]

    def torch_forward(self, input_states):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection (colwise-sharded, 5-part fused)
        projected_states = torch.ops.auto_deploy.torch_linear_simple(
            input_states,
            self.in_proj.weight,
            self.in_proj.bias,
            tp_mode="colwise",
            output_sizes=self._in_proj_output_sizes,
            layer_type="ssm",
        )
        gate, hidden_states_B_C, dt = torch.ops.auto_deploy.split_with_sizes(
            projected_states,
            [self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
            shardable=True,
            layer_type="ssm",
        )

        # 2. Convolution sequence transformation
        hidden_states_B_C = self.act(
            torch.ops.auto_deploy.torch_causal_conv1d(
                hidden_states_B_C,
                self.conv1d.weight,
                self.conv1d.bias,
                self.conv1d.stride[0],
                self.conv1d.padding[0],
                self.conv1d.dilation[0],
                self.conv1d.groups,
                self.conv1d.padding_mode,
                shardable=True,
                output_sizes=self._conv1d_output_sizes,
                layer_type="ssm",
            )
        )

        hidden_states, B, C = torch.ops.auto_deploy.split_with_sizes(
            hidden_states_B_C,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
            shardable=True,
            layer_type="ssm",
        )

        # 3. SSM transformation
        A = -torch.exp(self.A_log.float())
        y = torch.ops.auto_deploy.torch_ssm(
            hidden_states=torch.ops.auto_deploy.view(
                hidden_states,
                [batch_size, seq_len, -1, self.head_dim],
                tp_scaled_dim=2,
                layer_type="ssm",
            ),
            A=A,
            B=torch.ops.auto_deploy.view(
                B,
                [batch_size, seq_len, -1, self.ssm_state_size],
                tp_scaled_dim=2,
                layer_type="ssm",
            ),
            C=torch.ops.auto_deploy.view(
                C,
                [batch_size, seq_len, -1, self.ssm_state_size],
                tp_scaled_dim=2,
                layer_type="ssm",
            ),
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            time_step_limit=list(self.time_step_limit),
            chunk_size=self.chunk_size,
            shardable=True,
            layer_type="ssm",
        )
        y = y.reshape(batch_size, seq_len, -1)

        scan_output = self.norm(y, gate)

        # 4. Final linear projection (rowwise) + all_reduce
        contextualized_states = torch.ops.auto_deploy.torch_linear_simple(
            scan_output.to(dtype),
            self.out_proj.weight,
            self.out_proj.bias,
            tp_mode="rowwise",
            layer_type="ssm",
        )
        contextualized_states = torch.ops.auto_deploy.all_reduce(
            contextualized_states, layer_type="ssm"
        )
        return contextualized_states

    def forward(self, hidden_states):
        return self.torch_forward(hidden_states)


class NemotronHRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


class NemotronHBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = NemotronHAttention(config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config, layer_idx=layer_idx)
        elif self.block_type == "moe":
            self.mixer = NemotronHMOE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid block type {self.block_type!r} at layer {layer_idx}")

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class NemotronHMLP(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        intermediate_size: Optional[int] = None,
        is_expert: bool = False,
        tp_sharded: bool = True,
        add_all_reduce: bool = True,
        sharding_layer_type: str = "mlp",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        use_latent_size = (getattr(self.config, "moe_latent_size", None) is not None) and is_expert
        input_size = self.config.moe_latent_size if use_latent_size else self.hidden_size
        self.up_proj = nn.Linear(input_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, input_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]
        self.tp_sharded = tp_sharded
        self.sharding_layer_type = sharding_layer_type
        self.add_all_reduce = add_all_reduce and self.tp_sharded

    def forward(self, x):
        up = torch.ops.auto_deploy.torch_linear_simple(
            x,
            self.up_proj.weight,
            self.up_proj.bias,
            tp_mode="colwise" if self.tp_sharded else "none",
            layer_type=self.sharding_layer_type,
        )
        down = torch.ops.auto_deploy.torch_linear_simple(
            self.act_fn(up),
            self.down_proj.weight,
            self.down_proj.bias,
            tp_mode="rowwise" if self.tp_sharded else "none",
            layer_type=self.sharding_layer_type,
        )
        if self.add_all_reduce:
            down = torch.ops.auto_deploy.all_reduce(down, layer_type=self.sharding_layer_type)
        return down


class NemotronHMOE(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                NemotronHMLP(
                    config,
                    layer_idx=layer_idx,
                    intermediate_size=config.moe_intermediate_size,
                    is_expert=True,
                    tp_sharded=False,
                    sharding_layer_type="moe",
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = NemotronHTopkRouter(config)
        self.shared_experts = NemotronHMLP(
            config=config,
            intermediate_size=config.moe_shared_expert_intermediate_size,
            layer_idx=layer_idx,
            is_expert=False,
            tp_sharded=True,
            add_all_reduce=False,
            sharding_layer_type="moe",
        )
        # Latent projections are REPLICATED (not sharded)
        if getattr(config, "moe_latent_size", None) is not None:
            self.fc1_latent_proj = nn.Linear(
                config.hidden_size, config.moe_latent_size, bias=config.mlp_bias
            )
            self.fc2_latent_proj = nn.Linear(
                config.moe_latent_size, config.hidden_size, bias=config.mlp_bias
            )
        else:
            self.fc1_latent_proj = nn.Identity()
            self.fc2_latent_proj = nn.Identity()

    def forward(self, hidden_states: torch.Tensor):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        x_flat = hidden_states.view(-1, hidden_states.shape[-1])

        shared_out = self.shared_experts(residuals)

        has_latent_proj = hasattr(self, "fc1_latent_proj") and hasattr(self, "fc2_latent_proj")

        if has_latent_proj:
            x_flat = self.fc1_latent_proj(x_flat)

        out_flat = torch.ops.auto_deploy.torch_moe(
            x_flat,
            topk_indices,
            topk_weights,
            w1_weight=[e.up_proj.weight for e in self.experts],
            w2_weight=[e.down_proj.weight for e in self.experts],
            w3_weight=[],
            act_fn=ActivationType.Relu2,
            is_gated_mlp=False,
            layer_type="moe",
        )

        if has_latent_proj:
            out_flat = self.fc2_latent_proj(out_flat)

        routed_out = out_flat.view(*orig_shape)
        out = shared_out + routed_out
        out = torch.ops.auto_deploy.all_reduce(out, layer_type="moe")
        return out


class NemotronHTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        if self.weight.dtype == torch.float32:
            router_logits = torch.nn.functional.linear(
                hidden_states.type(torch.float32), self.weight
            )
        else:
            router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_states, self.weight.t(), bias=None, out_dtype=torch.float32
            )

        topk_weights, topk_indices = torch.ops.trtllm.noaux_tc_op(
            router_logits,
            self.e_score_correction_bias,
            self.n_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
        )

        return topk_indices, topk_weights


class NemotronHAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            raise ValueError("Please make sure to provide a `layer_idx` when creating this class.")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        if hasattr(config, "head_dim"):
            head_dim = config.head_dim
        elif hasattr(config, "attention_head_dim"):
            head_dim = config.attention_head_dim
        else:
            raise AttributeError(
                "Expected either `head_dim` or `attention_head_dim` to be present in the config "
                "class, found neither."
            )

        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_heads, self.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.q_proj.weight,
            self.q_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        key_states = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.k_proj.weight,
            self.k_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        value_states = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.v_proj.weight,
            self.v_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )

        query_states = torch.ops.auto_deploy.view(
            query_states,
            [bsz, q_len, self.num_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        key_states = torch.ops.auto_deploy.view(
            key_states,
            [bsz, q_len, self.num_key_value_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        value_states = torch.ops.auto_deploy.view(
            value_states,
            [bsz, q_len, self.num_key_value_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )

        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            layout="bsnd",
        )

        attn_output = torch.ops.auto_deploy.view(
            attn_output,
            [bsz, q_len, self.num_heads * self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )

        attn_output = torch.ops.auto_deploy.torch_linear_simple(
            attn_output,
            self.o_proj.weight,
            self.o_proj.bias,
            tp_mode="rowwise",
            layer_type="mha",
        )
        attn_output = torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mha")

        return attn_output


class NemotronHPreTrainedModel(PreTrainedModel):
    base_model_prefix = "backbone"
    _no_split_modules = ["NemotronHBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        if isinstance(module, NemotronHMamba2Mixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.config.mamba_num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)

            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


@dataclass
class NemotronHOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class NemotronHCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class NemotronHModel(NemotronHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NemotronHBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )

        self.norm_f = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHOutput]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        for mixer_block in self.layers:
            hidden_states = mixer_block(hidden_states)

        hidden_states = self.norm_f(hidden_states)

        return NemotronHOutput(last_hidden_state=hidden_states)


class NemotronHForCausalLM(NemotronHPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = NemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHCausalLMOutput]:
        nemotron_h_outputs = self.backbone(input_ids, inputs_embeds=inputs_embeds)
        hidden_states = nemotron_h_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        return NemotronHCausalLMOutput(logits)


AutoModelForCausalLMFactory.register_custom_model_cls("NemotronHConfig", NemotronHForCausalLM)
