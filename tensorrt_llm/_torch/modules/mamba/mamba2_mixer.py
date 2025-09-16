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

from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

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
        nheads: int,
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

        d_inner = head_dim * nheads
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
        self.in_proj = Linear(
            d_model,
            d_in_proj,
            bias=bias,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
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
        self.out_proj = Linear(
            d_inner,
            d_model,
            bias=bias,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy)

        self._mamba_ssm_cache_dtype = config.quant_config.mamba_ssm_cache_dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
    ) -> torch.Tensor:

        # calculate split size
        num_prefills = attn_metadata.num_contexts
        num_decodes = attn_metadata.seq_lens.shape[0] - num_prefills
        num_prefill_tokens = attn_metadata.num_ctx_tokens
        num_decode_tokens = attn_metadata.num_tokens - num_prefill_tokens
        seqlen_split_size = [num_prefill_tokens, num_decode_tokens]
        batch_split_size = [num_prefills, num_decodes]

        state_indices = attn_metadata.kv_cache_manager.get_state_indices(
        )[:num_prefills + num_decodes]

        state_indices_p, state_indices_d = torch.split(state_indices,
                                                       batch_split_size)
        conv_states = attn_metadata.kv_cache_manager.get_conv_states(
            self.layer_idx)
        ssm_states = attn_metadata.kv_cache_manager.get_ssm_states(
            self.layer_idx)

        # in_proj
        zxbcdt = self.in_proj(hidden_states)
        z, xbc, dt = torch.split(
            zxbcdt,
            [self.tp_d_inner, self.tp_conv_dim, self.tp_nheads],
            dim=-1,
        )
        z_p, z_d = torch.split(z, seqlen_split_size, dim=0)
        xbc_p, xbc_d = torch.split(xbc, seqlen_split_size, dim=0)
        dt_p, dt_d = torch.split(dt, seqlen_split_size, dim=0)

        out = []

        if num_prefills > 0:

            cu_seqlens = mamba_metadata.cu_seqlens[:num_prefills + 1]
            seq_idx = mamba_metadata.seq_idx
            has_initial_states = mamba_metadata.has_initial_states[:
                                                                   num_prefills]

            xbc_p = causal_conv1d_fn(xbc_p.transpose(0, 1),
                                     self.conv1d.weight,
                                     self.conv1d.bias,
                                     activation="silu",
                                     conv_states=conv_states,
                                     has_initial_state=has_initial_states,
                                     query_start_loc=cu_seqlens,
                                     cache_indices=state_indices_p).transpose(
                                         0, 1)

            x_p, B_p, C_p = torch.split(xbc_p.unsqueeze(0), [
                self.tp_d_inner,
                self.tp_ngroups * self.d_state,
                self.tp_ngroups * self.d_state,
            ],
                                        dim=-1)

            x_p = rearrange(x_p, "b l (h p) -> b l h p", h=self.tp_nheads)
            dt_p = dt_p.unsqueeze(0)
            B_p = rearrange(B_p, "b l (g n) -> b l g n", g=self.tp_ngroups)
            C_p = rearrange(C_p, "b l (g n) -> b l g n", g=self.tp_ngroups)
            z_p = rearrange(z_p.unsqueeze(0),
                            "b l (h p) -> b l h p",
                            h=self.tp_nheads)

            initial_states = None
            if mamba_metadata.use_initial_states:
                initial_states = torch.where(
                    has_initial_states[:, None, None, None],
                    ssm_states[state_indices_p], 0)

            y, current_ssm_states = mamba_chunk_scan_combined(
                x_p,
                dt_p,
                self.A,
                B_p,
                C_p,
                chunk_size=self.chunk_size,
                D=self.D,
                z=z_p,
                dt_bias=self.dt_bias,
                initial_states=initial_states,
                chunk_indices=mamba_metadata.chunk_indices,
                chunk_offsets=mamba_metadata.chunk_offsets,
                dt_softplus=self.delta_softplus,
                cu_seqlens=cu_seqlens,
                seq_idx=seq_idx,
                return_varlen_states=True,
                return_final_states=False,
                mamba_ssm_cache_dtype=self._mamba_ssm_cache_dtype,
            )
            out.append(rearrange(y, "b l h p -> (b l) (h p)"))

            # copy new ssm state
            ssm_states[state_indices_p] = current_ssm_states

        if num_decodes > 0:
            xbc_d = causal_conv1d_update(xbc_d,
                                         conv_states,
                                         self.conv1d.weight,
                                         self.conv1d.bias,
                                         activation="silu",
                                         conv_state_indices=state_indices_d)

            x_d, B_d, C_d = torch.split(
                xbc_d,
                [
                    self.tp_d_inner,
                    self.tp_ngroups * self.d_state,
                    self.tp_ngroups * self.d_state,
                ],
                dim=-1,
            )

            x_d = rearrange(x_d, "b (h p) -> b h p", p=self.head_dim)
            dt_d = repeat(dt_d, "b h -> b h p", p=self.head_dim)
            B_d = rearrange(B_d, "b (g n) -> b g n", g=self.tp_ngroups)
            C_d = rearrange(C_d, "b (g n) -> b g n", g=self.tp_ngroups)
            z_d = rearrange(z_d, "b (h p) -> b h p", p=self.head_dim)

            A = repeat(self.A, "h -> h p n", p=self.head_dim,
                       n=self.d_state).to(dtype=torch.float32)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.head_dim)
            D = repeat(self.D, "h -> h p", p=self.head_dim)

            y = selective_state_update(
                ssm_states,
                x_d,
                dt_d,
                A,
                B_d,
                C_d,
                D,
                z=z_d,
                dt_bias=dt_bias,
                dt_softplus=self.delta_softplus,
                state_batch_indices=state_indices_d,
            )

            out.append(rearrange(y, "b h p -> b (h p)"))

        out = torch.cat(out, dim=0)

        # norm
        out = self.norm(out)

        # out_proj
        out = self.out_proj(out)

        return out
