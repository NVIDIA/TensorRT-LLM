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

from typing import List

import torch
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ..attention_interface import (
    AttentionRegistry,
    MHACallable,
    ResourceHandler,
    ResourceHandlerDict,
    SequenceInfo,
)
from .mamba_backend_common import (
    BaseBackendSSM,
    _flatten_ssm_inputs,
    _prepare_ssm_decode_inputs,
    _run_ssm_prefill,
)


@torch.library.custom_op("auto_deploy::flashinfer_cached_ssm", mutates_args={})
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
    # EXTRA METADATA
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
) -> torch.Tensor:
    b, s, num_heads, head_dim, bs, hs_flat, B_flat, C_flat, dt_flat = _flatten_ssm_inputs(
        hidden_states, B, C, dt
    )
    ssm_state_size = B.shape[3]
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode
    # Preallocate output tensor to avoid memcpy cost for merging prefill
    # and decode outputs
    preallocated_ssm_out = torch.empty(
        [bs, num_heads, head_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    preallocated_ssm_out_p = preallocated_ssm_out[:num_prefill_tokens]

    num_prefill, num_prefill_tokens, num_total_tokens, num_seq = _run_ssm_prefill(
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
        chunk_indices,
        chunk_offsets,
        seq_idx_prefill,
        ssm_state_cache,
        time_step_limit,
        chunk_size,
        preallocated_ssm_out_p.unsqueeze(0),
    )

    num_decode = num_total_tokens - num_prefill_tokens
    decode_inputs = _prepare_ssm_decode_inputs(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        slot_idx,
        num_prefill,
        num_prefill_tokens,
        num_seq,
        num_total_tokens,
        num_heads,
        head_dim,
        ssm_state_size,
    )

    y_decode = None
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

        import flashinfer

        slot_idx_decode_i32 = slot_idx_decode.to(torch.int32)
        y_decode = flashinfer.mamba.selective_state_update(
            ssm_state_cache,
            x_decode,
            dt_hp,
            A_full,
            B_decode,
            C_decode,
            D=D_full,
            z=None,
            dt_bias=dt_bias_hp,
            dt_softplus=True,
            state_batch_indices=slot_idx_decode_i32,
        )
        preallocated_ssm_out[num_prefill_tokens:num_total_tokens].copy_(y_decode)
    if num_total_tokens > 0:
        return (
            preallocated_ssm_out[:num_total_tokens]
            .view(b, s, num_heads, head_dim)
            .to(hidden_states.dtype)
        )
    else:
        return torch.empty_like(hidden_states)


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
    # EXTRA METADATA
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
):
    # Return a correctly-shaped tensor for tracing with fake tensors
    return torch.empty_like(
        hidden_states,
        memory_format=torch.contiguous_format,
        dtype=hidden_states.dtype,
    )


# Flashinfer's selective_state_update kernel only supports these head dimensions
FLASHINFER_SUPPORTED_HEAD_DIMS = [64, 128]


class FlashInferStateResourceHandler(ResourceHandler):
    """Handler for flashinfer SSM state resources.

    Unlike the default StateResourceHandler which uses byte-level pooling (resulting
    in non-contiguous strided views), this handler allocates a separate contiguous
    buffer. This is required because flashinfer's selective_state_update kernel
    requires the entire state tensor to be contiguous.
    """

    def __init__(self, *state_shape: int, dtype: torch.dtype) -> None:
        self.state_shape = state_shape
        self.dtype = dtype

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        """Allocate a contiguous state buffer for flashinfer."""
        return torch.empty(
            sequence_info.max_num_state_slots,
            *self.state_shape,
            device=sequence_info.device,
            dtype=self.dtype,
        )


@AttentionRegistry.register("flashinfer_ssm")
class FlashinferBackendSSM(BaseBackendSSM):
    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.flashinfer_cached_ssm.default

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Get cache initializers using FlashInferStateResourceHandler.

        We use a custom handler that allocates contiguous buffers directly,
        instead of the default StateResourceHandler which creates non-contiguous
        views from a shared byte buffer. This is required because flashinfer's
        selective_state_update kernel requires contiguous state tensors.
        """
        # Shapes from fake tensors
        hs_fake: torch.Tensor = source_attn_node.args[0].meta["val"]
        B_fake: torch.Tensor = source_attn_node.args[2].meta["val"]

        num_heads = hs_fake.shape[-2]
        head_dim = hs_fake.shape[-1]

        # Validate head_dim is supported by flashinfer
        if head_dim not in FLASHINFER_SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"Flashinfer SSM backend only supports head_dim in {FLASHINFER_SUPPORTED_HEAD_DIMS}, "
                f"but got head_dim={head_dim}. Consider using 'triton_ssm' backend instead."
            )

        if B_fake.ndim >= 4:
            ssm_state_size = B_fake.shape[-1]
        else:
            ssm_state_size = max(1, B_fake.shape[-1])

        # Extract ssm_state_dtype from cache_config or hs_fake
        ssm_state_dtype = cls.resolve_cache_dtype(cache_config.mamba_ssm_cache_dtype, hs_fake.dtype)

        return {
            "ssm_state_cache": FlashInferStateResourceHandler(
                num_heads, head_dim, ssm_state_size, dtype=ssm_state_dtype
            )
        }
