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

from typing import List

import torch
import torch.nn.functional as F
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ....modules.fla.chunk import chunk_gated_delta_rule
from ....modules.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    StateResourceHandler,
)


@torch.library.custom_op("auto_deploy::fla_cached_gated_delta_rule", mutates_args=("delta_cache",))
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
    # CACHES
    delta_cache: torch.Tensor,  # [max_batch_size, HV, K, V]
    # CONSTANTS
    scale: float,
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

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    cu_seqlen_prefill = cu_seqlen[: num_prefill + 1]
    slot_idx = slot_idx[:num_seq].to(torch.long)
    use_initial_states = use_initial_states[:num_seq]

    if num_prefill > 0:
        initial_states = None
        if torch.any(use_initial_states[:num_prefill]):
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

    if num_decode > 0:
        cu_seqlen_decode = torch.arange(0, num_decode + 1, device=q.device, dtype=torch.long)

        q_dec = q_flat[None, num_prefill_tokens:].contiguous()
        k_dec = k_flat[None, num_prefill_tokens:].contiguous()
        v_dec = v_flat[None, num_prefill_tokens:].contiguous()
        a_dec = a_flat[None, num_prefill_tokens:].contiguous()
        b_dec = b_flat[None, num_prefill_tokens:].contiguous()

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
            initial_state_indices=slot_idx[num_prefill:].contiguous(),
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlen_decode,
        )

        y_flat[None, num_prefill_tokens:] = y_decode.to(y_flat.dtype)
        del y_decode

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
    delta_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
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
        return ["batch_info_host", "cu_seqlen", "slot_idx", "use_initial_states"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        key_node = source_attn_node.args[1]
        value_node = source_attn_node.args[2]
        # State is per value-head: [HV, K, V]
        num_heads = value_node.meta["val"].shape[-2]
        key_dim = key_node.meta["val"].shape[-1]
        value_dim = value_node.meta["val"].shape[-1]

        return {
            "delta_cache": StateResourceHandler(
                num_heads,
                key_dim,
                value_dim,
                # GDN state is a running recurrence (unlike KV caches which store
                # independent per-token values). Bfloat16 quantization errors
                # compound at every decode step through the recurrence update, so
                # we always use float32 to preserve accuracy over long sequences.
                dtype=torch.float32,
            )
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        scale = extract_op_args(source_attn_node, "scale")[0]
        if scale is None:
            key_node = source_attn_node.args[1]
            key_dim = key_node.meta["val"].shape[-1]
            scale = key_dim**-0.5
        return [scale]
