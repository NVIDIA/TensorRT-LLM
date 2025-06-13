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

from typing import Optional

import torch
from einops import rearrange, repeat
from torch import nn

from ...attention_backend import AttentionMetadata
from ...model_config import ModelConfig
from ..linear import Linear, TensorParallelMode
from .causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from .layernorm_gated import RMSNorm as RMSNormGated
from .selective_state_update import selective_state_update
from .ssd_combined import mamba_chunk_scan_combined


class Mamba2Mixer(nn.Module):

    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_groups: int,
        head_dim: int,
        chunk_size: int,
        layer_idx: int,
        bias: bool = False,
        conv_bias: bool = True,
        delta_rank: int = 0,
        delta_softplus: bool = True,
        remove_padding: bool = True,
        apply_silu: bool = True,
        rms_norm_eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()

        config = config or ModelConfig()
        self.mapping = config.mapping
        tp_rank = config.mapping.tp_rank
        tp_size = config.mapping.tp_size

        d_inner = d_model * expand
        nheads = d_inner // head_dim
        d_in_proj = 2 * d_inner + 2 * n_groups * d_state + nheads
        conv_dim = d_inner + 2 * n_groups * d_state

        # TP
        self.tp_conv_dim = conv_dim // tp_size
        self.tp_d_inner = d_inner // tp_size
        self.tp_nheads = nheads // tp_size
        self.tp_ngroups = n_groups // tp_size

        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.d_state = d_state
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.delta_rank = delta_rank
        self.delta_softplus = delta_softplus
        self.remove_padding = remove_padding
        self.apply_silu = apply_silu

        # tp
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # paged state parameters
        self.slot_mapping = None
        self.is_paged_state = False

        # in_proj
        self.in_proj = Linear(d_model,
                              d_in_proj,
                              bias=bias,
                              dtype=dtype,
                              mapping=self.mapping,
                              tensor_parallel_mode=TensorParallelMode.COLUMN,
                              quant_config=config.get_quant_config(),
                              allreduce_strategy=config.allreduce_strategy)

        # conv1d, reuse Linear to store weights since it has support for TP > 1 already
        self.conv1d = Linear(
            d_conv,
            conv_dim,
            bias=conv_bias,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy)

        # A
        self.A = nn.Parameter(
            torch.empty(self.tp_nheads,
                        dtype=torch.float32,
                        requires_grad=False))

        # D
        self.D = nn.Parameter(
            torch.empty(self.tp_nheads,
                        dtype=torch.float32,
                        requires_grad=False))

        # dt_bias
        self.dt_bias = nn.Parameter(
            torch.empty(self.tp_nheads,
                        dtype=torch.float32,
                        requires_grad=False))

        # norm
        self.norm = RMSNormGated(
            self.tp_d_inner,
            eps=rms_norm_eps,
            norm_before_gate=False,
            group_size=self.tp_d_inner // self.tp_ngroups,
            dtype=dtype,
        )

        # out_proj
        self.out_proj = Linear(d_inner,
                               d_model,
                               bias=bias,
                               dtype=dtype,
                               mapping=self.mapping,
                               tensor_parallel_mode=TensorParallelMode.ROW,
                               quant_config=config.get_quant_config(),
                               allreduce_strategy=config.allreduce_strategy)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        # calculate split size
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.seq_lens.shape[0] - num_contexts
        sum_seq = torch.cumsum(attn_metadata.seq_lens, dim=0)
        split_ctx = sum_seq[num_contexts - 1] if num_contexts > 0 else 0
        split_gen = sum_seq[-1] - split_ctx
        split_size = [split_ctx, split_gen]

        state_indices = attn_metadata.kv_cache_manager.get_state_indices()

        # warm up does not prepare resources, so no relevant state indices
        is_warmup = state_indices.numel() == 0
        if is_warmup:
            # in this case, assume batch takes first indices in mamba cache
            state_indices = torch.arange(num_contexts + num_generations,
                                         device=state_indices.device,
                                         dtype=state_indices.dtype)

        split_indices = torch.split(state_indices,
                                    [num_contexts, num_generations])

        split_seq_lens = torch.split(attn_metadata.seq_lens,
                                     [num_contexts, num_generations])

        # in_proj
        zxbcdt = self.in_proj(hidden_states)
        split_zxbcdt = torch.split(zxbcdt, split_size, dim=0)

        # a batch can have either:
        # * only context requests
        # * only generation requests
        # * both context and generation requests
        # req_type = 0 -> context
        # req_type = 1 -> generation
        batch = None
        # both context and generation requests
        if num_contexts > 0 and num_generations > 0:
            batch = [0, 1]
        # only context requests
        elif num_contexts > 0:
            batch = [0]
        # only generation requests
        elif num_generations > 0:
            batch = [1]

        out = []
        for req_type in batch:

            indices = split_indices[req_type].to(torch.device("cuda"))
            conv_states = attn_metadata.kv_cache_manager.get_conv_states(
                self.layer_idx)
            ssm_states = attn_metadata.kv_cache_manager.get_ssm_states(
                self.layer_idx)

            z, xbc, dt = torch.split(
                split_zxbcdt[req_type],
                [self.tp_d_inner, self.tp_conv_dim, self.tp_nheads],
                dim=-1,
            )

            # prefill
            if req_type == 0:

                cu_seqlens = (torch.cat(
                    [
                        torch.zeros(1),
                        torch.cumsum(split_seq_lens[req_type], dim=0)
                    ],
                    dim=0,
                ).to(torch.int32).to(torch.device("cuda")))

                seq_idx = torch.repeat_interleave(
                    torch.arange(len(split_seq_lens[req_type]),
                                 dtype=torch.int32,
                                 device=cu_seqlens.device),
                    cu_seqlens.diff(),
                    output_size=cu_seqlens[-1]).unsqueeze(0)

                xbc = causal_conv1d_fn(xbc.transpose(0, 1),
                                       self.conv1d.weight,
                                       self.conv1d.bias,
                                       activation="silu",
                                       conv_states=conv_states,
                                       query_start_loc=cu_seqlens,
                                       cache_indices=indices).transpose(0, 1)

                x, B, C = torch.split(xbc.unsqueeze(0), [
                    self.tp_d_inner,
                    self.tp_ngroups * self.d_state,
                    self.tp_ngroups * self.d_state,
                ],
                                      dim=-1)

                x = rearrange(x, "b l (h p) -> b l h p", h=self.tp_nheads)
                dt = dt.unsqueeze(0)
                B = rearrange(B, "b l (g n) -> b l g n", g=self.tp_ngroups)
                C = rearrange(C, "b l (g n) -> b l g n", g=self.tp_ngroups)
                z = rearrange(z.unsqueeze(0),
                              "b l (h p) -> b l h p",
                              h=self.tp_nheads)

                y, current_ssm_states = mamba_chunk_scan_combined(
                    x,
                    dt,
                    self.A,
                    B,
                    C,
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=z,
                    dt_bias=self.dt_bias,
                    initial_states=None,
                    dt_softplus=self.delta_softplus,
                    cu_seqlens=cu_seqlens,
                    seq_idx=seq_idx,
                    return_varlen_states=True,
                    return_final_states=False,
                )
                y = rearrange(y, "b l h p -> (b l) (h p)")

                # copy new ssm state
                ssm_states[indices] = current_ssm_states

            # decode
            else:
                xbc = causal_conv1d_update(xbc,
                                           conv_states,
                                           self.conv1d.weight,
                                           self.conv1d.bias,
                                           activation="silu",
                                           conv_state_indices=indices)

                x, B, C = torch.split(
                    xbc,
                    [
                        self.tp_d_inner,
                        self.tp_ngroups * self.d_state,
                        self.tp_ngroups * self.d_state,
                    ],
                    dim=-1,
                )

                A = repeat(self.A,
                           "h -> h p n",
                           p=self.head_dim,
                           n=self.d_state).to(dtype=torch.float32)
                dt = repeat(dt, "b h -> b h p", p=self.head_dim)
                dt_bias = repeat(self.dt_bias, "h -> h p", p=self.head_dim)
                D = repeat(self.D, "h -> h p", p=self.head_dim)
                B = rearrange(B, "b (g n) -> b g n", g=self.tp_ngroups)
                C = rearrange(C, "b (g n) -> b g n", g=self.tp_ngroups)
                x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.head_dim)
                z = rearrange(z, "b (h p) -> b h p", p=self.head_dim)

                y = selective_state_update(
                    ssm_states,
                    x_reshaped,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    z=z,
                    dt_bias=dt_bias,
                    dt_softplus=self.delta_softplus,
                    state_batch_indices=indices,
                )

                y = rearrange(y, "b h p -> b (h p)")

            # gated norm
            y = self.norm(y)

            # append output
            out.append(y)

        out = torch.cat(out, dim=0)

        # out_proj
        out = self.out_proj(out)

        return out
