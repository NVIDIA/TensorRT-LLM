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
from torch import nn

from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...attention_backend import AttentionMetadata
from ...model_config import ModelConfig
from ...speculative import SpecMetadata
from ..linear import Linear, TensorParallelMode
from .causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from .causal_conv1d_triton import \
    causal_conv1d_update as causal_conv1d_update_triton
from .fuse_elementwise_ops import (extract_transpose_xbc_prefill,
                                   fused_split_rearrange_after_conv1d)
from .layernorm_gated import RMSNorm as RMSNormGated
from .selective_state_update import \
    selective_state_update as selective_state_update_native
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

        if config.mapping.enable_attention_dp:
            self.mapping = Mapping(
                world_size=config.mapping.pp_size,
                tp_size=1,
                pp_size=config.mapping.pp_size,
                rank=config.mapping.rank,
                gpus_per_node=config.mapping.gpus_per_node,
                enable_attention_dp=True,
            )
            tp_size = 1
        else:
            self.mapping = config.mapping
            tp_size = config.mapping.tp_size

        d_inner = head_dim * nheads
        d_in_proj = 2 * d_inner + 2 * n_groups * d_state + nheads
        conv_dim = d_inner + 2 * n_groups * d_state

        # TP
        self.tp_conv_dim = conv_dim // tp_size
        self.tp_d_inner = d_inner // tp_size
        self.tp_nheads = nheads // tp_size
        self.tp_ngroups = n_groups // tp_size
        self.num_heads = nheads
        self.tp_size = tp_size

        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.d_state = d_state
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.delta_rank = delta_rank
        self.delta_softplus = delta_softplus
        self.remove_padding = remove_padding
        self.apply_silu = apply_silu

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
            allreduce_strategy=config.allreduce_strategy,
        )

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
            allreduce_strategy=config.allreduce_strategy,
        )

        # A
        self.A = nn.Parameter(
            torch.empty(self.tp_nheads,
                        dtype=torch.float32,
                        requires_grad=False))

        # Use native (Triton) kernel for decode SSM update.
        # With expand-based A/dt_bias (stride=0), the Triton kernel enables
        # TIE_HDIM=True which loads A as a per-head scalar instead of a full
        # (nheads, headdim, dstate) tensor, saving ~2 MB HBM reads per call.
        self._mamba_ssm_cache_dtype = config.quant_config.mamba_ssm_cache_dtype
        logger.info_once(
            "Using native (Triton TIE_HDIM) for selective state update for no MTP",
            key="selective_state_update_no_mtp",
        )
        self.selective_state_update_func_no_mtp = selective_state_update_native
        # TODO: support MTP selective state update in flashinfer.
        logger.info_once("Using native for selective state update for MTP",
                         key="selective_state_update_mtp")
        self.selective_state_update_func_mtp = selective_state_update_native

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

        # Determine if NVFP4 quantization is enabled
        self.is_nvfp4 = (config.quant_config is not None
                         and config.quant_config.quant_mode is not None
                         and config.quant_config.quant_mode.has_nvfp4())

        # norm
        self.norm = RMSNormGated(
            self.tp_d_inner,
            eps=rms_norm_eps,
            norm_before_gate=False,
            group_size=self.tp_d_inner // self.tp_ngroups,
            dtype=dtype,
            # Enable fused NVFP4 quantization if possible.
            # It might be overridden in `_try_attach_nvfp4_scale` function.
            is_nvfp4=self.is_nvfp4,
        )

        # Auxiliary CUDA stream for overlapping BMM with chunk_state + state_passing
        # during prefill. BMM (C^T @ B) is independent of steps 1-3 in the SSD
        # pipeline, so it can run in parallel on a separate stream.
        self._ssd_aux_stream = torch.cuda.Stream()

        # Pre-expanded decode parameters (lazily initialized after weight loading).
        # Avoids repeat() + dtype cast every decode step.
        self._A_expanded = None
        self._dt_bias_expanded = None
        self._D_expanded = None

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
            allreduce_strategy=config.allreduce_strategy,
        )

    def _try_attach_nvfp4_scale(self):
        """Attach input_scale from out_proj to norm for fused RMSNorm+Quant.

        Called lazily on first forward (weights don't exist during __init__).
        """
        if getattr(self.out_proj, "input_scale", None) is not None:
            self.norm.nvfp4_scale = self.out_proj.input_scale
        else:
            self.norm.is_nvfp4 = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Lazily attach nvfp4_scale on first forward
        if self.norm.is_nvfp4 and self.norm.nvfp4_scale is None:
            self._try_attach_nvfp4_scale()

        # calculate split size
        num_prefills = attn_metadata.num_contexts
        num_decodes = attn_metadata.seq_lens.shape[0] - num_prefills
        num_prefill_tokens = attn_metadata.num_ctx_tokens
        num_decode_tokens = attn_metadata.num_tokens - num_prefill_tokens
        num_actual_tokens = attn_metadata.num_tokens
        seqlen_split_size = [num_prefill_tokens, num_decode_tokens]
        batch_split_size = [num_prefills, num_decodes]

        state_indices = attn_metadata.kv_cache_manager.get_state_indices(
        )[:num_prefills + num_decodes]

        state_indices_p, state_indices_d = torch.split(state_indices,
                                                       batch_split_size)
        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(
            self.layer_idx)
        conv_states = layer_cache.conv
        ssm_states = layer_cache.temporal

        # in_proj
        zxbcdt = self.in_proj(hidden_states)

        # Split z and dt with views.
        z = zxbcdt[:, :self.tp_d_inner]
        dt = zxbcdt[:, self.tp_d_inner + self.tp_conv_dim:]
        z_p, z_d = torch.split(z, seqlen_split_size, dim=0)
        dt_p, dt_d = torch.split(dt, seqlen_split_size, dim=0)

        # Decode path uses regular view since no transpose is needed.
        xbc_d = zxbcdt[
            num_prefill_tokens:num_actual_tokens,
            self.tp_d_inner:self.tp_d_inner + self.tp_conv_dim,
        ]

        # Preallocate output tensor to avoid memcpy cost for merging prefill
        # and decode outputs
        preallocated_ssm_out = torch.empty(
            [
                zxbcdt.shape[0],
                (self.num_heads * self.head_dim) // self.tp_size,
            ],
            dtype=zxbcdt.dtype,
            device=zxbcdt.device,
        )
        preallocated_ssm_out_p, preallocated_ssm_out_d = torch.split(
            preallocated_ssm_out,
            [num_prefill_tokens, num_decode_tokens],
            dim=0,
        )

        if num_prefills > 0:
            cu_seqlens = mamba_metadata.cu_seqlens[:num_prefills + 1]
            seq_idx = mamba_metadata.seq_idx
            has_initial_states = mamba_metadata.has_initial_states[:
                                                                   num_prefills]

            # Fused kernel to avoid expensive .contiguous() call in causal_conv1d_fn.
            xbc_p_t = extract_transpose_xbc_prefill(zxbcdt, num_prefill_tokens,
                                                    self.tp_d_inner,
                                                    self.tp_conv_dim)
            xbc_p = causal_conv1d_fn(
                xbc_p_t,
                self.conv1d.weight,
                self.conv1d.bias,
                activation="silu",
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                query_start_loc=cu_seqlens,
                cache_indices=state_indices_p,
            )

            # Fused kernel to avoid expensive .contiguous() calls after split/rearrange.
            x_p, B_p, C_p = fused_split_rearrange_after_conv1d(
                xbc_p,
                self.tp_d_inner,
                self.tp_ngroups,
                self.d_state,
                self.tp_nheads,
                self.head_dim,
            )
            dt_p = dt_p.unsqueeze(0)

            initial_states = None
            if mamba_metadata.use_initial_states:
                initial_states = torch.where(
                    has_initial_states[:, None, None, None],
                    ssm_states[state_indices_p], 0)

            current_ssm_states = mamba_chunk_scan_combined(
                x_p,
                dt_p,
                self.A,
                B_p,
                C_p,
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                initial_states=initial_states,
                chunk_indices=mamba_metadata.chunk_indices,
                chunk_offsets=mamba_metadata.chunk_offsets,
                dt_softplus=self.delta_softplus,
                dt_limit=(0.0, float("inf")),
                cu_seqlens=cu_seqlens,
                seq_idx=seq_idx,
                return_varlen_states=True,
                return_final_states=False,
                out=preallocated_ssm_out_p.view(1, num_prefill_tokens, -1,
                                                self.head_dim),
                state_dtype=self._mamba_ssm_cache_dtype,
                aux_stream=self._ssd_aux_stream,
            )

            # copy new ssm state
            ssm_states[state_indices_p] = current_ssm_states

        if num_decodes > 0:
            is_target_verify = (attn_metadata.kv_cache_manager.is_speculative()
                                and spec_metadata is not None)
            if is_target_verify:
                # TODO: support dynamic speculation, will add current_draft_len later [TRTLLM-10319]
                draft_token_num = spec_metadata.max_draft_len + 1
                intermediate_conv_states = layer_cache.intermediate_conv_window

                self.intermediate_state_indices = torch.arange(
                    num_decodes,
                    dtype=torch.int32,
                    device=state_indices_d.device)

                # Reshape for batch processing
                xbc_d_reshaped = xbc_d.view(num_decodes, draft_token_num,
                                            -1).transpose(1, 2)
                # TODO:support tree structure [TRTLLM-10320]
                xbc_d_processed = causal_conv1d_update_triton(
                    xbc_d_reshaped,
                    conv_states,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    activation="silu",
                    conv_state_indices=state_indices_d[:num_decodes],
                    intermediate_conv_window=intermediate_conv_states,
                    intermediate_state_indices=self.intermediate_state_indices,
                )

                xbc_d = xbc_d_processed.transpose(1, 2).view(
                    num_decode_tokens, -1)

            else:
                xbc_d = causal_conv1d_update(
                    xbc_d,
                    conv_states,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    activation="silu",
                    conv_state_indices=state_indices_d,
                )

            # Split + reshape using direct view ops instead of einops to
            # minimize Python dispatch overhead on the decode hot path.
            x_d = xbc_d[:, :self.tp_d_inner].view(-1, self.tp_nheads,
                                                  self.head_dim)
            bc_offset = self.tp_d_inner
            bc_size = self.tp_ngroups * self.d_state
            B_d = xbc_d[:, bc_offset:bc_offset + bc_size].view(
                -1, self.tp_ngroups, self.d_state)
            C_d = xbc_d[:, bc_offset + bc_size:].view(-1, self.tp_ngroups,
                                                      self.d_state)
            # Need to keep the same dtype as self.dt_bias and self.D to avoid garbage outputs.
            dt_d = dt_d.float().unsqueeze(-1).expand(-1, -1, self.head_dim)
            z_d = z_d.view(-1, self.tp_nheads, self.head_dim)

            if self._A_expanded is None:
                # Use expand (stride=0) instead of repeat (copies) so the
                # Triton kernel detects TIE_HDIM=True and loads A/dt_bias as
                # per-head scalars, avoiding a 2 MB read of the expanded A
                # tensor on every decode step.
                self._A_expanded = (self.A.float().reshape(-1, 1, 1).expand(
                    -1, self.head_dim, self.d_state))
                self._dt_bias_expanded = (self.dt_bias.float().reshape(
                    -1, 1).expand(-1, self.head_dim))
                self._D_expanded = self.D.float().reshape(-1, 1).expand(
                    -1, self.head_dim)
            A = self._A_expanded
            dt_bias = self._dt_bias_expanded
            D = self._D_expanded
            if is_target_verify:
                intermediate_ssm_states = layer_cache.intermediate_ssm
                self.selective_state_update_func_mtp(
                    ssm_states,
                    x_d.view(
                        num_decodes,
                        draft_token_num,
                        self.num_heads // self.tp_size,
                        self.head_dim,
                    ),
                    dt_d.view(
                        num_decodes,
                        draft_token_num,
                        self.num_heads // self.tp_size,
                        self.head_dim,
                    ),
                    A,
                    B_d.view(num_decodes, draft_token_num, self.tp_ngroups, -1),
                    C_d.view(num_decodes, draft_token_num, self.tp_ngroups, -1),
                    D,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    state_batch_indices=state_indices_d[:num_decodes],
                    out=preallocated_ssm_out_d.view(
                        num_decodes,
                        draft_token_num,
                        self.num_heads // self.tp_size,
                        self.head_dim,
                    ),
                    disable_state_update=True,
                    intermediate_states_buffer=intermediate_ssm_states,
                    cache_steps=draft_token_num,
                    intermediate_state_indices=self.intermediate_state_indices,
                )
            else:
                self.selective_state_update_func_no_mtp(
                    ssm_states,
                    x_d,
                    dt_d,
                    A,
                    B_d,
                    C_d,
                    D,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=self.delta_softplus,
                    state_batch_indices=state_indices_d,
                    out=preallocated_ssm_out_d.view(num_decodes, -1,
                                                    self.head_dim),
                )

        # norm
        hidden_states = self.norm(preallocated_ssm_out, z[:num_actual_tokens])

        # out_proj
        out = self.out_proj(hidden_states)

        return out[:num_actual_tokens]
