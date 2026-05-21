# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Optional, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from tensorrt_llm._torch.modules.mamba.mamba2_metadata import cu_seqlens_to_chunk_indices_offsets
from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined

from ..._compat import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    BatchInfo,
    Constant,
    PrepareMetadataCallable,
    ResourceHandlerDict,
    SSMResourceHandler,
)


@torch.library.custom_op("auto_deploy::mamba_ssm_prepare_metadata", mutates_args=())
def _mamba_ssm_prepare_metadata(
    # INPUTS
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # EXTRA METADATA PROVIDED BY THE DESCRIPTOR
    chunk_size: int,
) -> List[torch.Tensor]:
    """Prepare metadata for cached SSM transform.

    Returns a tuple of (chunk_indices, chunk_offsets, seq_idx_prefill).
    """
    device = cu_seqlen.device
    batch_info = BatchInfo(batch_info_host)

    num_prefill, _, _ = batch_info.get_num_sequences()

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


@_mamba_ssm_prepare_metadata.register_fake
def _mamba_ssm_prepare_metadata_fake(
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


def _flatten_ssm_inputs(
    hidden_states: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[int, int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, num_heads, head_dim = hidden_states.shape
    bs = b * s
    hs_flat = hidden_states.reshape(bs, *hidden_states.shape[2:])  # [bs, H, D]
    B_flat = B.reshape(bs, *B.shape[2:])  # [bs, G, N]
    C_flat = C.reshape(bs, *C.shape[2:])  # [bs, G, N]
    dt_flat = dt.reshape(bs, dt.shape[2])  # [bs, H]
    return b, s, num_heads, head_dim, bs, hs_flat, B_flat, C_flat, dt_flat


def _run_ssm_prefill(
    hs_flat: torch.Tensor,
    B_flat: torch.Tensor,
    C_flat: torch.Tensor,
    dt_flat: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    any_prefill_use_initial_states_host: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    seq_idx_prefill: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    time_step_limit: List[float],
    chunk_size: int,
    out: Optional[torch.Tensor] = None,
):
    batch_info = BatchInfo(batch_info_host)
    num_prefill, _, _ = batch_info.get_num_sequences()
    num_prefill_tokens, _, _ = batch_info.get_num_tokens()

    if num_prefill <= 0:
        return

    hs_prefill = hs_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, H, D]
    B_prefill = B_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, G, N]
    C_prefill = C_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, G, N]
    dt_prefill = dt_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, H]

    seq_idx_prefill = seq_idx_prefill[:, :num_prefill_tokens]

    initial_states = None
    # Use precomputed host flag to avoid GPU->CPU sync from torch.any()
    if any_prefill_use_initial_states_host.item():
        initial_states = torch.where(
            use_initial_states[:num_prefill, None, None, None],
            ssm_state_cache[slot_idx[:num_prefill]],
            0,
        )
    else:
        chunk_indices = None
        chunk_offsets = None

    varlen_states = mamba_chunk_scan_combined(
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
        state_dtype=ssm_state_cache.dtype,
        out=out,
    )

    ssm_state_cache.index_copy_(
        0, slot_idx[:num_prefill].long(), varlen_states.to(ssm_state_cache.dtype)
    )


def _prepare_ssm_decode_inputs(
    hs_flat: torch.Tensor,
    B_flat: torch.Tensor,
    C_flat: torch.Tensor,
    dt_flat: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    slot_idx: torch.Tensor,
    decode_seq_start: int,
    decode_token_start: int,
    num_decode: int,
    num_decode_tokens: int,
    num_heads: int,
    head_dim: int,
    ssm_state_size: int,
) -> Optional[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
]:
    grouped_inputs = _prepare_ssm_grouped_state_update_inputs(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        slot_idx,
        seq_start=decode_seq_start,
        token_start=decode_token_start,
        num_seq=num_decode,
        num_tokens=num_decode_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        ssm_state_size=ssm_state_size,
    )
    if grouped_inputs is None:
        return None
    (
        slot_idx_decode,
        x_decode_g,
        B_decode_g,
        C_decode_g,
        dt_hp_g,
        A_full,
        D_full,
        dt_bias_hp,
    ) = grouped_inputs

    # Reshape from [num_decode, 1, ...] to [num_decode, ...]
    x_decode = x_decode_g.reshape(num_decode, num_heads, head_dim)
    B_decode = B_decode_g.reshape(num_decode, B_flat.shape[1], ssm_state_size)
    C_decode = C_decode_g.reshape(num_decode, C_flat.shape[1], ssm_state_size)
    dt_hp = dt_hp_g.reshape(num_decode, num_heads, head_dim)

    return slot_idx_decode, x_decode, B_decode, C_decode, dt_hp, dt_bias_hp, A_full, D_full


def _prepare_ssm_grouped_state_update_inputs(
    hs_flat: torch.Tensor,
    B_flat: torch.Tensor,
    C_flat: torch.Tensor,
    dt_flat: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    slot_idx: torch.Tensor,
    seq_start: int,
    token_start: int,
    num_seq: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    ssm_state_size: int,
) -> Optional[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
]:
    if num_seq <= 0 or num_tokens <= 0:
        return None
    seq_len = num_tokens // num_seq

    seq_end = seq_start + num_seq
    token_end = token_start + num_tokens
    slot_idx_slice = slot_idx[seq_start:seq_end]
    x_slice = hs_flat[token_start:token_end].view(num_seq, seq_len, num_heads, head_dim)
    B_slice = B_flat[token_start:token_end].view(num_seq, seq_len, B_flat.shape[1], ssm_state_size)
    C_slice = C_flat[token_start:token_end].view(num_seq, seq_len, C_flat.shape[1], ssm_state_size)
    dt_slice = dt_flat[token_start:token_end].view(num_seq, seq_len, num_heads)
    dt_hp_slice = dt_slice[..., None].expand(num_seq, seq_len, num_heads, head_dim)
    A_full = A[..., None, None].expand(num_heads, head_dim, ssm_state_size)
    D_full = D[..., None].expand(num_heads, head_dim)
    dt_bias_hp = dt_bias[..., None].expand(num_heads, head_dim)
    return slot_idx_slice, x_slice, B_slice, C_slice, dt_hp_slice, A_full, D_full, dt_bias_hp


class BaseBackendSSM(AttentionDescriptor):
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
    def get_standard_metadata_args(cls) -> List[str]:
        return [
            "batch_info_host",
            "cu_seqlen",
            "slot_idx",
            "use_initial_states",
            "any_prefill_use_initial_states_host",
        ]

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[PrepareMetadataCallable, int, List[Constant]]:
        return (
            torch.ops.auto_deploy.mamba_ssm_prepare_metadata.default,
            3,  # chunk_indices, chunk_offsets, seq_idx_prefill
            extract_op_args(any_source_attn_node, "chunk_size"),
        )

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
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
        ssm_state_dtype = cls.resolve_cache_dtype(cache_config.mamba_ssm_cache_dtype, hs_fake.dtype)

        return {
            "ssm_state_cache": SSMResourceHandler(
                num_heads=num_heads,
                head_dim=head_dim,
                d_state=ssm_state_size,
                dtype=ssm_state_dtype,
            )
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        time_step_limit, chunk_size = extract_op_args(
            source_attn_node, "time_step_limit", "chunk_size"
        )
        return [time_step_limit, chunk_size]
