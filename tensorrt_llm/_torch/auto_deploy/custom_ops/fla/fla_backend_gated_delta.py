# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
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

"""Cached attention op for the gated delta rule using fla kernels.

Gated Delta Rule is based on this paper: https://arxiv.org/abs/2412.06464

Kernels are based on this repo: https://github.com/fla-org/flash-linear-attention

This op accepts raw (un-normalized, un-expanded) q/k and raw gating projections
(a, b) together with per-head parameters (A_log, dt_bias). L2 normalization,
GQA repeat-interleave, and gating computation are performed internally:
  - Decode: fully fused in fused_sigmoid_gating_delta_rule_update (L2 norm, GQA, gating)
  - Prefill: explicit repeat-interleave + chunk_gated_delta_rule(use_qk_l2norm_in_kernel=True)
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch._ops import OpOverloadPacket
from torch.fx import Node

from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)

from ..._compat import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BatchInfo,
    Constant,
    IntermediateSSMStateHandler,
    MHACallable,
    ResourceHandlerDict,
    SSMResourceHandler,
)


@torch.library.custom_op(
    "auto_deploy::fla_cached_gated_delta_rule",
    mutates_args=("delta_cache", "intermediate_delta_cache"),
)
def fla_cached_gated_delta_rule(
    # INPUTS (raw, un-normalized, un-expanded)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    any_prefill_use_initial_states_host: torch.Tensor,
    # CACHES
    delta_cache: torch.Tensor,  # [max_batch_size, HV, K, V]
    intermediate_delta_cache: Optional[
        torch.Tensor
    ],  # [max_batch_size, max_draft_len + 1, HV, K, V]
    # CONSTANTS
    scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz, s, H_k, K = q.shape
    HV = v.shape[2]
    interleave = HV // H_k

    # Flatten batch and sequence dims
    q_flat = q.view(bsz * s, H_k, K)
    k_flat = k.view(bsz * s, H_k, K)
    v_flat = v.view(bsz * s, HV, -1)
    a_flat = a.view(bsz * s, HV)
    b_flat = b.view(bsz * s, HV)

    y = torch.empty_like(v, memory_format=torch.contiguous_format)
    y_flat = y.view(bsz * s, HV, -1)

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_extend, num_decode = batch_info.get_num_sequences()
    num_prefill_tokens, num_extend_tokens, num_decode_tokens = batch_info.get_num_tokens()
    num_seq = num_prefill + num_extend + num_decode
    num_total_tokens = num_prefill_tokens + num_extend_tokens + num_decode_tokens

    cu_seqlen_prefill = cu_seqlen[: num_prefill + 1]
    slot_idx = slot_idx[:num_seq].to(torch.long)
    use_initial_states = use_initial_states[:num_seq]

    if num_prefill > 0:
        initial_states = None
        # Use precomputed host flag to avoid GPU->CPU sync from torch.any()
        if any_prefill_use_initial_states_host.item():
            initial_states = torch.where(
                use_initial_states[:num_prefill, None, None, None],
                delta_cache[slot_idx[:num_prefill]],
                0,
            )

        q_pf = q_flat[None, :num_prefill_tokens]
        k_pf = k_flat[None, :num_prefill_tokens]
        v_pf = v_flat[None, :num_prefill_tokens]
        a_pf = a_flat[None, :num_prefill_tokens]
        b_pf = b_flat[None, :num_prefill_tokens]

        # GQA expand for chunk kernel (it does not handle H != HV natively)
        if interleave > 1:
            q_pf = q_pf.repeat_interleave(interleave, dim=2)
            k_pf = k_pf.repeat_interleave(interleave, dim=2)

        # Compute g and beta from raw parameters
        g_pf = -A_log.float().exp() * F.softplus(a_pf.float() + dt_bias)
        beta_pf = b_pf.float().sigmoid()

        y_prefill, final_state = chunk_gated_delta_rule(
            q=q_pf,
            k=k_pf,
            v=v_pf,
            g=g_pf,
            beta=beta_pf,
            scale=scale,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlen_prefill,
            use_qk_l2norm_in_kernel=True,
        )

        y_flat[None, :num_prefill_tokens] = y_prefill.to(y_flat.dtype)
        delta_cache.index_copy_(0, slot_idx[:num_prefill], final_state.to(delta_cache.dtype))
        del y_prefill, initial_states, final_state

    if num_extend > 0:
        tokens_per_extend = num_extend_tokens // num_extend
        extend_start = num_prefill_tokens
        extend_end = extend_start + num_extend_tokens
        slot_idx_extend = slot_idx[num_prefill : num_prefill + num_extend]

        q_ext = q_flat[extend_start:extend_end].view(num_extend, tokens_per_extend, H_k, K)
        k_ext = k_flat[extend_start:extend_end].view(num_extend, tokens_per_extend, H_k, K)
        v_ext = v_flat[extend_start:extend_end].view(num_extend, tokens_per_extend, HV, -1)
        a_ext = a_flat[extend_start:extend_end].view(num_extend, tokens_per_extend, HV)
        b_ext = b_flat[extend_start:extend_end].view(num_extend, tokens_per_extend, HV)

        if intermediate_delta_cache is None:
            raise RuntimeError(
                "fla_cached_gated_delta_rule requires an intermediate_delta_cache "
                "for extend requests"
            )
        if intermediate_delta_cache.size(1) < tokens_per_extend:
            raise RuntimeError(
                "fla_cached_gated_delta_rule received an intermediate_delta_cache "
                "that is too small for the extend branch"
            )

        recurrent_state_source = delta_cache[slot_idx_extend]
        recurrent_state_indices = torch.arange(
            num_extend, dtype=torch.int32, device=slot_idx_extend.device
        )
        g_ext = -A_log.float().exp() * F.softplus(a_ext.float() + dt_bias)
        beta_ext = b_ext.float().sigmoid()

        y_extend = fused_recurrent_gated_delta_rule_update(
            q=q_ext,
            k=k_ext,
            v=v_ext,
            g=g_ext,
            beta=beta_ext,
            initial_state_source=recurrent_state_source,
            initial_state_indices=recurrent_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_delta_cache,
            cache_steps=tokens_per_extend,
        )

        y_flat[extend_start:extend_end] = y_extend.view(num_extend_tokens, HV, -1).to(y_flat.dtype)
        del y_extend

    if num_decode > 0:
        cu_seqlen_decode = torch.arange(0, num_decode + 1, device=q.device, dtype=torch.long)
        decode_start = num_prefill_tokens + num_extend_tokens

        q_dec = q_flat[None, decode_start:num_total_tokens].contiguous()
        k_dec = k_flat[None, decode_start:num_total_tokens].contiguous()
        v_dec = v_flat[None, decode_start:num_total_tokens].contiguous()
        a_dec = a_flat[None, decode_start:num_total_tokens].contiguous()
        b_dec = b_flat[None, decode_start:num_total_tokens].contiguous()

        y_decode = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a_dec,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q_dec,
            k=k_dec,
            v=v_dec,
            b=b_dec,
            initial_state_source=delta_cache,
            initial_state_indices=slot_idx[num_prefill + num_extend : num_seq].contiguous(),
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlen_decode,
        )

        y_flat[None, decode_start:num_total_tokens] = y_decode.to(y_flat.dtype)
        del y_decode

    if out is not None:
        out_flat = out.view(bsz * s, HV, -1)
        out_flat[:num_total_tokens].copy_(y_flat[:num_total_tokens])
        if num_total_tokens < bsz * s:
            out_flat[num_total_tokens:].zero_()
        return out.new_empty(0)

    return y


@fla_cached_gated_delta_rule.register_fake
def fla_cached_gated_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    any_prefill_use_initial_states_host: torch.Tensor,
    delta_cache: torch.Tensor,
    intermediate_delta_cache: Optional[torch.Tensor],
    scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        return out.new_empty(0)
    return torch.empty_like(v)


@AttentionRegistry.register("fla_gated_delta")
class FlaGatedDeltaBackend(AttentionDescriptor):
    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # q, k, v, a, b, A_log, dt_bias
        return 7

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_gated_delta_rule

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.fla_cached_gated_delta_rule.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return [
            "batch_info_host",
            "cu_seqlen",
            "slot_idx",
            "use_initial_states",
            "any_prefill_use_initial_states_host",
        ]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        key_node = source_attn_node.args[1]
        value_node = source_attn_node.args[2]
        # Cache shape is [max_batch_size, HV, K, V] where HV = num_v_heads (state per value-head).
        # With GVA, q/k may have fewer heads (H_k) than v (HV), so read num_heads from value_node.
        num_heads = value_node.meta["val"].shape[-2]
        key_dim = key_node.meta["val"].shape[-1]
        value_dim = value_node.meta["val"].shape[-1]
        delta_cache = SSMResourceHandler(
            num_heads,
            key_dim,
            value_dim,
            # GDN state is a running recurrence (unlike KV caches which store
            # independent per-token values). Bfloat16 quantization errors
            # compound at every decode step through the recurrence update, so
            # we always use float32 to preserve accuracy over long sequences.
            dtype=torch.float32,
        )

        return {
            "delta_cache": delta_cache,
            "intermediate_delta_cache": IntermediateSSMStateHandler.from_base(delta_cache),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        scale = extract_op_args(source_attn_node, "scale")[0]
        if scale is None:
            key_node = source_attn_node.args[1]
            key_dim = key_node.meta["val"].shape[-1]
            scale = key_dim**-0.5
        return [scale]
