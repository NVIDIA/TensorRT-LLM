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

import functools
import os

import torch
from einops import rearrange, repeat
from flashinfer.mamba import selective_state_update as selective_state_update_fi
from torch import nn

from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.modules.multi_stream_utils import \
    maybe_execute_in_parallel
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import \
    use_cpp_mamba_cache_manager
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...attention_backend import AttentionMetadata
from ...model_config import ModelConfig
from ...peft.lora.layer import LoraLayer, LoraModuleType
from ...speculative import SpecMetadata
from ..linear import Linear, TensorParallelMode
from .causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from .causal_conv1d_triton import \
    causal_conv1d_update as causal_conv1d_update_triton
from .fuse_elementwise_ops import (extract_transpose_xbc_prefill,
                                   fused_split_rearrange_after_conv1d)
from .layernorm_gated import RMSNorm as RMSNormGated
from .layernorm_gated import fused_gated_rmsnorm_quant_shape_ok
from .replay_selective_state_update import replay_selective_state_update
from .selective_state_update import \
    selective_state_update as selective_state_update_native
from .selective_state_update import selective_state_update_mtp_ssm_cache_trtllm
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
        dtype: torch.dtype | None = None,
        config: ModelConfig | None = None,
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

        self.in_proj_lora = None
        if config.lora_config is not None:
            self.in_proj_lora = LoraLayer([LoraModuleType.MAMBA_IN_PROJ],
                                          [d_in_proj // tp_size])

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
            lora=self.in_proj_lora)

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

        # Choose between flashinfer and native implementation. (default to flashinfer)
        self._mamba_ssm_cache_dtype = config.quant_config.mamba_ssm_cache_dtype
        # TODO: Update head_dims and head_group_ratios once flashinfer is updated.
        supported_head_dims = [64, 128]
        supported_head_group_ratios = [1, 8, 16]
        head_group_ratio = (self.tp_nheads //
                            self.tp_ngroups if self.tp_ngroups > 0 else 0)
        self._use_flashinfer = (head_dim in supported_head_dims and
                                head_group_ratio in supported_head_group_ratios)
        self._stochastic_rounding_requested = (
            config.quant_config.mamba_ssm_stochastic_rounding)
        self._philox_rounds = config.quant_config.mamba_ssm_philox_rounds
        # SR needs fp16 cache.  Replay and flashinfer each supply a Philox impl;
        # custom_op does not.  Only use_replay is resolved per-forward (from the
        # cache manager), so precompute both gate values here.
        sr_base = (self._stochastic_rounding_requested
                   and self._mamba_ssm_cache_dtype == torch.float16)
        self._stochastic_rounding_for_replay = sr_base
        self._stochastic_rounding_for_flashinfer = sr_base and self._use_flashinfer

        self._use_mtp_custom_op = os.environ.get(
            "TRTLLM_MAMBA2_MTP_USE_CUSTOM_OP", "0") == "1"

        if self._use_flashinfer:
            logger.info_once("Using flashinfer for selective state update",
                             key="selective_state_update")
            self.selective_state_update_func = selective_state_update_fi
        else:
            logger.info_once("Using native for selective state update",
                             key="selective_state_update")
            self.selective_state_update_func = selective_state_update_native

        # Warn if stochastic rounding was requested but no path can supply it.
        if self._stochastic_rounding_requested:
            if self._mamba_ssm_cache_dtype != torch.float16:
                logger.warning_once(
                    f"Stochastic rounding needs fp16 SSM cache, "
                    f"have {self._mamba_ssm_cache_dtype}. Disabled.",
                    key="stochastic_rounding_disabled")
            elif not self._use_flashinfer:
                logger.warning_once(
                    "Stochastic rounding needs flashinfer or replay; "
                    "neither available with current configuration.",
                    key="stochastic_rounding_disabled")

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

        # LoRA layers require regular bf16 tensors, not Fp4QuantizedTensor.
        # Disable fused RMSNorm+NVFP4 when LoRA is configured.
        self.is_nvfp4 = (config.lora_config is None
                         and config.quant_config is not None
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

        self.out_proj_lora = None
        if config.lora_config is not None:
            self.out_proj_lora = LoraLayer([LoraModuleType.MAMBA_OUT_PROJ],
                                           [d_model])

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
            lora=self.out_proj_lora)

        self.aux_steram = torch.cuda.Stream()
        self.events = [torch.cuda.Event(), torch.cuda.Event()]

    def post_load_weights(self):
        """Post-process after loading weights."""
        if (self.norm.is_nvfp4 and fused_gated_rmsnorm_quant_shape_ok(
                self.norm.hidden_size, self.norm.group_size)
                and self.norm.nvfp4_scale is None):
            self._try_attach_nvfp4_scale()

        # Pre-expand A, D, dt_bias for the decode path.
        self._A_expanded = repeat(self.A,
                                  "h -> h p n",
                                  p=self.head_dim,
                                  n=self.d_state).to(dtype=torch.float32)
        self._dt_bias_expanded = repeat(self.dt_bias,
                                        "h -> h p",
                                        p=self.head_dim)
        self._D_expanded = repeat(self.D, "h -> h p", p=self.head_dim)

    def _try_attach_nvfp4_scale(self):
        """Attach input_scale from out_proj to norm for fused RMSNorm+Quant."""

        if getattr(self.out_proj, 'input_scale', None) is not None:
            self.norm.nvfp4_scale = self.out_proj.input_scale
        else:
            self.norm.is_nvfp4 = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        spec_metadata: SpecMetadata | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:

        # calculate split size
        num_prefills = attn_metadata.num_contexts
        num_decodes = attn_metadata.seq_lens.shape[0] - num_prefills
        num_prefill_tokens = attn_metadata.num_ctx_tokens
        num_decode_tokens = attn_metadata.num_tokens - num_prefill_tokens
        num_actual_tokens = attn_metadata.num_tokens
        seqlen_split_size = [num_prefill_tokens, num_decode_tokens]
        batch_split_size = [num_prefills, num_decodes]

        state_indices = mamba_metadata.state_indices[:num_prefills +
                                                     num_decodes]
        if use_cpp_mamba_cache_manager():
            conv_states = attn_metadata.kv_cache_manager.get_conv_states(
                self.layer_idx)
            ssm_states = attn_metadata.kv_cache_manager.get_ssm_states(
                self.layer_idx)
            layer_cache = None  # Not used in C++ path
        else:
            layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(
                self.layer_idx)
            conv_states = layer_cache.conv
            ssm_states = layer_cache.temporal

        state_indices_p, state_indices_d = torch.split(state_indices,
                                                       batch_split_size)

        # in_proj (LoRA is applied internally by Linear layer)
        zxbcdt = self.in_proj(hidden_states,
                              lora_params=lora_params,
                              layer_idx=self.layer_idx)

        # Split z and dt with views.
        z = zxbcdt[:, :self.tp_d_inner]
        dt = zxbcdt[:, self.tp_d_inner + self.tp_conv_dim:]
        dt_p, dt_d = torch.split(dt, seqlen_split_size, dim=0)

        # Decode path uses regular view since no transpose is needed.
        xbc_d = zxbcdt[num_prefill_tokens:num_actual_tokens,
                       self.tp_d_inner:self.tp_d_inner + self.tp_conv_dim]

        # Preallocate output tensor to avoid memcpy cost for merging prefill
        # and decode outputs
        preallocated_ssm_out = torch.empty(
            [zxbcdt.shape[0], (self.tp_nheads * self.head_dim)],
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
            xbc_p = causal_conv1d_fn(xbc_p_t,
                                     self.conv1d.weight,
                                     self.conv1d.bias,
                                     activation="silu",
                                     conv_states=conv_states,
                                     has_initial_state=has_initial_states,
                                     query_start_loc=cu_seqlens,
                                     cache_indices=state_indices_p)

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
            )

            # copy new ssm state
            ssm_states[state_indices_p] = current_ssm_states

        if num_decodes > 0:
            is_target_verify = attn_metadata.kv_cache_manager.is_speculative(
            ) and spec_metadata is not None
            if is_target_verify:
                # Speculative decoding only supported with Python path
                assert layer_cache is not None, \
                    "Speculative decoding requires Python MambaCacheManager"
                # TODO: support dynamic speculation, will add current_draft_len later [TRTLLM-10319]
                draft_token_num = spec_metadata.max_draft_len + 1
                intermediate_conv_states = layer_cache.intermediate_conv_window
                use_replay = getattr(attn_metadata.kv_cache_manager,
                                     'use_replay_state_update', False)

                intermediate_state_indices = _cached_arange(
                    attn_metadata.kv_cache_manager.get_max_resource_count(),
                    state_indices_d.device)[:num_decodes]

                # Reshape for batch processing
                xbc_d_reshaped = xbc_d.view(num_decodes, draft_token_num,
                                            -1).transpose(1, 2)

                def conv1d():
                    # TODO:support tree structure [TRTLLM-10320]
                    xbc_d_processed = causal_conv1d_update_triton(
                        xbc_d_reshaped,
                        conv_states,
                        self.conv1d.weight,
                        self.conv1d.bias,
                        activation="silu",
                        conv_state_indices=state_indices_d[:num_decodes],
                        intermediate_conv_window=intermediate_conv_states,
                        intermediate_state_indices=intermediate_state_indices,
                        # PDL chain: conv1d → precompute → main (replay only)
                        launch_dependent_kernels=use_replay,
                    )

                    return xbc_d_processed.transpose(1, 2).view(
                        num_decode_tokens, -1)

            else:

                def conv1d():
                    return causal_conv1d_update(
                        xbc_d,
                        conv_states,
                        self.conv1d.weight,
                        self.conv1d.bias,
                        activation="silu",
                        conv_state_indices=state_indices_d)

            if is_target_verify and use_replay:
                # Replay path: kernel handles bf16 dt natively (applies bias +
                # softplus internally).  No dt conversion needed.
                xbc_d = conv1d()
            else:
                # Non-replay paths: flashinfer/native needs fp32 dt.
                def convert_dt():
                    return dt_d.to(dtype=torch.float32)

                xbc_d, dt_d = maybe_execute_in_parallel(conv1d,
                                                        convert_dt,
                                                        self.events[0],
                                                        self.events[1],
                                                        self.aux_steram,
                                                        disable_on_compile=True)

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

            A = self._A_expanded
            dt_bias = self._dt_bias_expanded
            D = self._D_expanded
            if is_target_verify:
                # 4D views for multi-token processing (shared by all paths).
                intermediate_ssm_states = layer_cache.intermediate_ssm
                x_d_4d = x_d.view(num_decodes, draft_token_num, self.tp_nheads,
                                  self.head_dim)
                dt_d_4d = dt_d.view(num_decodes, draft_token_num,
                                    self.tp_nheads, self.head_dim)
                B_d_4d = B_d.view(num_decodes, draft_token_num, self.tp_ngroups,
                                  -1)
                C_d_4d = C_d.view(num_decodes, draft_token_num, self.tp_ngroups,
                                  -1)
                out_4d = preallocated_ssm_out_d.view(num_decodes,
                                                     draft_token_num,
                                                     self.tp_nheads,
                                                     self.head_dim)
                state_batch_indices = state_indices_d[:num_decodes]

                use_stochastic_rounding = (
                    self._stochastic_rounding_for_replay
                    if use_replay else self._stochastic_rounding_for_flashinfer)

                philox_kwargs = {}
                if use_stochastic_rounding:
                    philox_kwargs['rand_seed'] = torch.randint(
                        0, 2**62, (1, ), device=x_d.device, dtype=torch.int64)
                    philox_kwargs['philox_rounds'] = self._philox_rounds

                if use_replay:
                    replay_selective_state_update(
                        ssm_states,
                        layer_cache.old_x,
                        layer_cache.old_B,
                        layer_cache.old_dt,
                        layer_cache.old_dA_cumsum,
                        layer_cache.cache_buf_idx,
                        layer_cache.prev_num_accepted_tokens,
                        x_d_4d,
                        dt_d_4d,
                        A,
                        B_d_4d,
                        C_d_4d,
                        D=D,
                        dt_bias=dt_bias,
                        dt_softplus=self.delta_softplus,
                        state_batch_indices=state_batch_indices,
                        out=out_4d,
                        launch_with_pdl=True,
                        **philox_kwargs,
                    )
                elif self._use_mtp_custom_op and not use_stochastic_rounding:
                    # Upstream TRT-LLM CUDA custom op for MTP SSM cache update.
                    # Does not support stochastic rounding.
                    selective_state_update_mtp_ssm_cache_trtllm(
                        ssm_states,
                        x_d_4d,
                        dt_d_4d,
                        A,
                        B_d_4d,
                        C_d_4d,
                        out_4d,
                        intermediate_ssm_states,
                        draft_token_num,
                        D=D,
                        z=None,
                        dt_bias=dt_bias,
                        dt_softplus=True,
                        state_batch_indices=state_batch_indices,
                        disable_state_update=True,
                        intermediate_state_indices=intermediate_state_indices,
                    )
                else:
                    # Legacy flashinfer path: contiguous copies for alignment.
                    x_d_4d = x_d_4d.contiguous()
                    B_d_4d = B_d_4d.contiguous()
                    C_d_4d = C_d_4d.contiguous()
                    self.selective_state_update_func(
                        ssm_states,
                        x_d_4d,
                        dt_d_4d,
                        A,
                        B_d_4d,
                        C_d_4d,
                        D,
                        z=None,
                        dt_bias=dt_bias,
                        dt_softplus=True,
                        state_batch_indices=state_batch_indices,
                        out=out_4d,
                        disable_state_update=True,
                        intermediate_states_buffer=intermediate_ssm_states,
                        cache_steps=draft_token_num,
                        intermediate_state_indices=intermediate_state_indices,
                        **philox_kwargs,
                    )
            else:
                # Non-MTP single-token decode
                # flashinfer needs contiguous x/B/C with 128-byte alignment.
                x_d = x_d.contiguous()
                B_d = B_d.contiguous()
                C_d = C_d.contiguous()
                ssu_kwargs = dict(
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=self.delta_softplus,
                    state_batch_indices=state_indices_d,
                    out=preallocated_ssm_out_d.view(num_decodes, -1,
                                                    self.head_dim),
                )

                # Non-MTP decode only runs through flashinfer, no replay path.
                use_stochastic_rounding = self._stochastic_rounding_for_flashinfer
                if use_stochastic_rounding:
                    ssu_kwargs['rand_seed'] = torch.randint(0,
                                                            2**62, (1, ),
                                                            device=x_d.device,
                                                            dtype=torch.int64)
                    ssu_kwargs['philox_rounds'] = self._philox_rounds

                self.selective_state_update_func(
                    ssm_states,
                    x_d,
                    dt_d,
                    A,
                    B_d,
                    C_d,
                    D,
                    **ssu_kwargs,
                )

        # norm
        hidden_states = self.norm(preallocated_ssm_out, z[:num_actual_tokens])

        # out_proj
        out = self.out_proj(hidden_states,
                            lora_params=lora_params,
                            layer_idx=self.layer_idx)

        return out[:num_actual_tokens]


# We want to cache the largest indexing vector we'd ever need and mask it, vs
# recreating it.  But we don't know the size at __init__, and it could even
# change later if the mamba cache manager changes.
@functools.cache
def _cached_arange(n: int, device: torch.device) -> torch.Tensor:
    return torch.arange(n, dtype=torch.int32, device=device)
