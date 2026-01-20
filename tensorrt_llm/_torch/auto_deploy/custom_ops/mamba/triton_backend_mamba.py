# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

# Triton kernels
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import cu_seqlens_to_chunk_indices_offsets
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined

from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    CacheConfig,
    CacheInitializerDict,
    Constant,
    MHACallable,
    PrepareMetadataCallable,
    SequenceInfo,
)


@torch.library.custom_op("auto_deploy::triton_ssm_prepare_metadata", mutates_args=())
def _triton_ssm_prepare_metadata(
    # INPUTS
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # EXTRA METADATA PROVIDED BY THE DESCRIPTOR
    chunk_size: int,
) -> List[torch.Tensor]:
    """Prepare metadata for cached SSM transform.

    Returns a tuple of (seq_len_sanitized, seq_start, slot_idx_sanitized).
    """
    device = cu_seqlen.device
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()

    if num_prefill > 0:
        chunk_indices, chunk_offsets = cu_seqlens_to_chunk_indices_offsets(
            cu_seqlen[: num_prefill + 1], chunk_size
        )
        seq_idx_prefill = torch.repeat_interleave(
            torch.arange(num_prefill, device=device, dtype=torch.int32), seq_len[:num_prefill]
        ).view(1, -1)
    else:
        chunk_indices = torch.empty(0, dtype=torch.int32, device=device)
        chunk_offsets = torch.empty(0, dtype=torch.int32, device=device)
        seq_idx_prefill = torch.empty(1, 0, dtype=torch.int32, device=device)

    return (chunk_indices, chunk_offsets, seq_idx_prefill)


@_triton_ssm_prepare_metadata.register_fake
def _triton_ssm_prepare_metadata_fake(
    # INPUTS
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # EXTRA METADATA PROVIDED BY THE DESCRIPTOR
    chunk_size: int,
):
    b, s = position_ids.shape[:2]
    num_tokens = b * s
    device = cu_seqlen.device
    dtype = torch.int32
    if s > 1:
        # NOTE: this is only an upper bound for the shape in this case...
        return (
            torch.empty(num_tokens, dtype=dtype, device=device),  # chunk_indices
            torch.empty(num_tokens, dtype=dtype, device=device),  # chunk_offsets
            torch.empty(1, num_tokens, dtype=dtype, device=device),  # seq_idx_prefill
        )
    else:
        return (
            torch.empty(0, dtype=dtype, device=device),  # chunk_indices
            torch.empty(0, dtype=dtype, device=device),  # chunk_offsets
            torch.empty(1, 0, dtype=dtype, device=device),  # seq_idx_prefill
        )


@torch.library.custom_op("auto_deploy::triton_cached_ssm", mutates_args={})
def _triton_cached_ssm(
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
    """Flattened cached SSM transform op that respects slot-indexed state caches.

    Split mixed batches into prefill (seq_len>1) and decode (seq_len==1):
    - Prefill: run one varlen combined scan over concatenated prefill tokens and update final states per slot.
    - Decode: batch single-token updates with selective_state_update and update states per slot.
    """
    b, s, num_heads, head_dim = hidden_states.shape
    # Flatten tokens for indexing/scatter
    bs = b * s
    hs_flat = hidden_states.reshape(bs, *hidden_states.shape[2:])  # [bs, H, D]
    B_flat = B.reshape(bs, *B.shape[2:])  # [bs, G, N]
    C_flat = C.reshape(bs, *C.shape[2:])  # [bs, G, N]
    dt_flat = dt.reshape(bs, dt.shape[2])  # [bs, H]

    ssm_state_size = B.shape[3]

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode

    y_prefill = None
    y_decode = None

    # Prefill: concatenate tokens at the front and run combined scan
    if num_prefill > 0:
        hs_prefill = hs_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, H, D]
        B_prefill = B_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, G, N]
        C_prefill = C_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, G, N]
        dt_prefill = dt_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, H]

        initial_states = None
        if torch.any(use_initial_states[:num_prefill]):
            initial_states = torch.where(
                use_initial_states[:num_prefill, None, None, None],
                ssm_state_cache[slot_idx[:num_prefill]],
                0,
            )
        else:
            chunk_indices = None
            chunk_offsets = None

        y_prefill, varlen_states = mamba_chunk_scan_combined(
            hs_prefill,
            dt_prefill,
            A,
            B_prefill,
            C_prefill,
            chunk_size=chunk_size,
            D=D,
            z=None,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx_prefill,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlen[: num_prefill + 1],
            dt_softplus=True,
            dt_limit=(time_step_limit[0], time_step_limit[1]),
            return_final_states=False,
            return_varlen_states=True,
            mamba_ssm_cache_dtype=ssm_state_cache.dtype,
        )

        ssm_state_cache.index_copy_(
            0, slot_idx[:num_prefill], varlen_states.to(ssm_state_cache.dtype)
        )

    # Decode: batch single-token updates via selective_state_update
    if num_decode > 0:
        slot_idx_decode = slot_idx[num_prefill:num_seq]

        x_decode = hs_flat[num_prefill_tokens:num_total_tokens]  # [nd, H, D]
        B_decode = B_flat[num_prefill_tokens:num_total_tokens]  # [nd, G, N]
        C_decode = C_flat[num_prefill_tokens:num_total_tokens]  # [nd, G, N]
        dt_decode = dt_flat[num_prefill_tokens:num_total_tokens]  # [nd, H]

        dt_hp = dt_decode[:, :, None].expand(-1, num_heads, head_dim)
        dt_bias_hp = dt_bias[..., None].expand(num_heads, head_dim)
        A_full = A[..., None, None].expand(num_heads, head_dim, ssm_state_size)
        D_full = D[..., None].expand(num_heads, head_dim)

        y_decode = selective_state_update(
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
            state_batch_indices=slot_idx_decode,
        )  # [nd, H, D]

    # Dispatch return logic
    if num_prefill > 0 and num_decode > 0:
        y = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
        y_flat = y.view(bs, *y.shape[2:])
        y_flat[:num_prefill_tokens].copy_(y_prefill[0])
        y_flat[num_prefill_tokens:num_total_tokens].copy_(y_decode)
        return y
    elif num_prefill > 0:
        return y_prefill[0].view(b, s, num_heads, head_dim).to(hidden_states.dtype)
    elif num_decode > 0:
        return y_decode.view(b, s, num_heads, head_dim).to(hidden_states.dtype)
    else:
        return torch.empty_like(hidden_states)


@_triton_cached_ssm.register_fake
def _triton_cached_ssm_fake(
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


@AttentionRegistry.register("triton_ssm")
class TritonBackendSSM(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        return True

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        # Hidden states follow [b, s, n, d]
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # torch_ssm_transform signature has 7 node/state arguments
        return 7

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        # Keep source op unchanged (used for uncached pre-export)
        return torch.ops.auto_deploy.torch_ssm

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_cached_ssm.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "cu_seqlen", "slot_idx", "use_initial_states"]

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[PrepareMetadataCallable, int, List[Constant]]:
        return (
            torch.ops.auto_deploy.triton_ssm_prepare_metadata.default,
            3,  # chunk_indices, chunk_offsets, seq_idx_prefill
            extract_op_args(any_source_attn_node, "chunk_size"),
        )

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        # Shapes from fake tensors
        hs_fake: torch.Tensor = source_attn_node.args[0].meta["val"]
        B_fake: torch.Tensor = source_attn_node.args[2].meta["val"]

        num_heads = hs_fake.shape[-2]
        head_dim = hs_fake.shape[-1]

        if B_fake.ndim >= 4:
            ssm_state_size = B_fake.shape[-1]
        else:
            ssm_state_size = max(1, B_fake.shape[-1])

        # extract ssm_state_dtype from cache_config or hs_fake
        ssm_state_dtype = cache_config.mamba_dtype or hs_fake.dtype

        def _get_ssm_cache(si: SequenceInfo):
            return torch.empty(
                si.max_state_slots,
                num_heads,
                head_dim,
                ssm_state_size,
                device=si.device,
                dtype=ssm_state_dtype,
            )

        return {"ssm_state_cache": _get_ssm_cache}

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        time_step_limit, chunk_size = extract_op_args(
            source_attn_node, "time_step_limit", "chunk_size"
        )
        return [time_step_limit, chunk_size]
