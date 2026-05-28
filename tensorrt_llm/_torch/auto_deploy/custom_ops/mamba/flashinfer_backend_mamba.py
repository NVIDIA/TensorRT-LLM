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

from typing import List, Optional

import torch
from flashinfer.mamba import selective_state_update as _flashinfer_ssm_update
from torch.fx import Node

from ..._compat import KvCacheConfig
from ...utils.node_utils import DynamicOpPolicy, piecewise_dynamic_op
from ..attention_interface import (
    AttentionRegistry,
    BatchInfo,
    MHACallable,
    ResourceHandlerDict,
    SpecSSMResourceHandler,
)
from .mamba_backend_common import (
    BaseBackendSSM,
    _flatten_ssm_inputs,
    _prepare_ssm_decode_inputs,
    _prepare_ssm_grouped_state_update_inputs,
    _run_ssm_prefill,
)


def _fi_align(t: torch.Tensor) -> torch.Tensor:
    """Ensure 128-byte alignment required by FlashInfer kernels.

    - Contiguous + aligned: contiguous() is a no-op, returns t unchanged.
    - Non-contiguous: contiguous() allocates fresh aligned storage, returns it.
    - Contiguous + misaligned: contiguous() is a no-op, clone() forces a new aligned allocation.
    """
    t = t.contiguous()
    return t if t.data_ptr() % 128 == 0 else t.clone()


@piecewise_dynamic_op(DynamicOpPolicy.OUT_BUFFER)
@torch.library.custom_op(
    "auto_deploy::flashinfer_cached_ssm",
    mutates_args=("ssm_state_cache", "intermediate_ssm_state_cache"),
)
def _flashinfer_cached_ssm(
    # INPUTS (dense but may be flattened across sequences)
    hidden_states: torch.Tensor,  # [b, s, num_heads, head_dim]
    A: torch.Tensor,  # [num_heads]
    B: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    C: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    D: torch.Tensor,  # [num_heads]
    dt: torch.Tensor,  # [b, s, num_heads]
    dt_bias: torch.Tensor,  # [num_heads]
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    any_prefill_use_initial_states_host: torch.Tensor,
    # EXTRA METADATA
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    intermediate_ssm_state_cache: Optional[
        torch.Tensor
    ],  # [spec_state_size, max_draft_len+1, num_heads, head_dim, d_state]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    b, s, num_heads, head_dim, bs, hs_flat, B_flat, C_flat, dt_flat = _flatten_ssm_inputs(
        hidden_states, B, C, dt
    )
    ssm_state_size = B.shape[3]
    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_extend, num_decode = batch_info.get_num_sequences()
    num_prefill_tokens, num_extend_tokens, num_decode_tokens = batch_info.get_num_tokens()
    num_total_tokens = num_prefill_tokens + num_extend_tokens + num_decode_tokens

    if out is not None:
        preallocated_ssm_out = out.view(bs, num_heads, head_dim)
    else:
        preallocated_ssm_out = torch.zeros(
            [bs, num_heads, head_dim],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    # PREFILL
    _run_ssm_prefill(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        any_prefill_use_initial_states_host,
        chunk_indices,
        chunk_offsets,
        seq_idx_prefill,
        ssm_state_cache,
        time_step_limit,
        chunk_size,
        preallocated_ssm_out[:num_prefill_tokens].unsqueeze(0),
    )

    # EXTEND: multi-token MTP verification path, writes intermediate SSM states
    extend_inputs = _prepare_ssm_grouped_state_update_inputs(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        slot_idx,
        seq_start=num_prefill,
        token_start=num_prefill_tokens,
        num_seq=num_extend,
        num_tokens=num_extend_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        ssm_state_size=ssm_state_size,
    )

    if extend_inputs is not None:
        tokens_per_extend = num_extend_tokens // num_extend
        if intermediate_ssm_state_cache.size(1) < tokens_per_extend:
            raise RuntimeError(
                "flashinfer_cached_ssm: intermediate_ssm_state_cache is too small "
                f"for extend branch (size1={intermediate_ssm_state_cache.size(1)}, "
                f"tokens_per_extend={tokens_per_extend})"
            )
        (
            slot_idx_extend,
            x_extend,
            B_extend,
            C_extend,
            dt_extend,
            A_full,
            D_full,
            dt_bias_hp,
        ) = extend_inputs

        preallocated_ssm_out_e = preallocated_ssm_out[
            num_prefill_tokens : num_prefill_tokens + num_extend_tokens
        ].view(num_extend, tokens_per_extend, num_heads, head_dim)

        intermediate_state_indices = torch.arange(
            num_extend, dtype=torch.int32, device=slot_idx_extend.device
        )
        _flashinfer_ssm_update(
            ssm_state_cache,
            x_extend,
            dt_extend,
            A_full,
            B_extend,
            C_extend,
            D_full,
            z=None,
            dt_bias=dt_bias_hp,
            dt_softplus=True,
            state_batch_indices=slot_idx_extend.to(torch.int32),
            out=preallocated_ssm_out_e,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_ssm_state_cache,
            cache_steps=tokens_per_extend,
            intermediate_state_indices=intermediate_state_indices,
        )

    # DECODE: single-token autoregressive path
    decode_inputs = _prepare_ssm_decode_inputs(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        slot_idx,
        num_prefill + num_extend,
        num_prefill_tokens + num_extend_tokens,
        num_decode,
        num_decode_tokens,
        num_heads,
        head_dim,
        ssm_state_size,
    )

    if decode_inputs is not None:
        (
            slot_idx_decode,
            x_decode,
            B_decode,
            C_decode,
            dt_hp,
            dt_bias_hp,
            A_full,
            D_full,
        ) = decode_inputs

        x_decode = _fi_align(x_decode)
        B_decode = _fi_align(B_decode)
        C_decode = _fi_align(C_decode)

        slot_idx_decode_i32 = slot_idx_decode.to(torch.int32)
        y_decode = _flashinfer_ssm_update(
            ssm_state_cache,
            x_decode,
            dt_hp,
            A_full,
            B_decode,
            C_decode,
            D_full,
            z=None,
            dt_bias=dt_bias_hp,
            dt_softplus=True,
            state_batch_indices=slot_idx_decode_i32,
        )
        preallocated_ssm_out[num_prefill_tokens + num_extend_tokens : num_total_tokens].copy_(
            y_decode
        )

    if out is not None:
        # out is reused across CUDA graph replays with varying num_total_tokens,
        # so stale data from prior replays can linger in the padding region.
        if num_total_tokens < bs:
            preallocated_ssm_out[num_total_tokens:].zero_()
        return out.new_empty(0)

    return preallocated_ssm_out.view(b, s, num_heads, head_dim)


@_flashinfer_cached_ssm.register_fake
def _flashinfer_cached_ssm_fake(
    # INPUTS (dense but may be flattened across sequences)
    hidden_states: torch.Tensor,  # [b, s, num_heads, head_dim]
    A: torch.Tensor,  # [num_heads]
    B: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    C: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    D: torch.Tensor,  # [num_heads]
    dt: torch.Tensor,  # [b, s, num_heads]
    dt_bias: torch.Tensor,  # [num_heads]
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    any_prefill_use_initial_states_host: torch.Tensor,
    # EXTRA METADATA
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    intermediate_ssm_state_cache: Optional[
        torch.Tensor
    ],  # [spec_state_size, max_draft_len+1, num_heads, head_dim, d_state]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
    out: Optional[torch.Tensor] = None,
):
    if out is not None:
        return out.new_empty(0)
    # Return a correctly-shaped tensor for tracing with fake tensors
    return torch.empty_like(
        hidden_states,
        memory_format=torch.contiguous_format,
        dtype=hidden_states.dtype,
    )


# Flashinfer's selective_state_update kernel only supports these head dimensions
FLASHINFER_SUPPORTED_HEAD_DIMS = [64, 128]


@AttentionRegistry.register("flashinfer_ssm")
class FlashinferBackendSSM(BaseBackendSSM):
    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.flashinfer_cached_ssm.default

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        ret = super().get_cache_initializers(source_attn_node, cache_config)

        # check head_dim is supported by flashinfer
        if ret["ssm_state_cache"].head_dim not in FLASHINFER_SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"flashinfer_ssm only supports head_dim in {FLASHINFER_SUPPORTED_HEAD_DIMS}. "
                f"Got head_dim={ret['ssm_state_cache'].head_dim}. "
                "Consider using 'triton_ssm' backend instead."
            )

        ret["intermediate_ssm_state_cache"] = SpecSSMResourceHandler.from_base(
            ret["ssm_state_cache"]
        )
        return ret
