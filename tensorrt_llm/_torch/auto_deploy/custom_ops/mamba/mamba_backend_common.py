# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .....llmapi.llm_args import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
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
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    seq_idx_prefill: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    time_step_limit: List[float],
    chunk_size: int,
    out: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], int, int, int, int]:
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode

    if num_prefill <= 0:
        return num_prefill, num_prefill_tokens, num_total_tokens, num_seq

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
    return num_prefill, num_prefill_tokens, num_total_tokens, num_seq


def _prepare_ssm_decode_inputs(
    hs_flat: torch.Tensor,
    B_flat: torch.Tensor,
    C_flat: torch.Tensor,
    dt_flat: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    slot_idx: torch.Tensor,
    num_prefill: int,
    num_prefill_tokens: int,
    num_seq: int,
    num_total_tokens: int,
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
    num_decode = num_total_tokens - num_prefill_tokens
    if num_decode <= 0:
        return None

    slot_idx_decode = slot_idx[num_prefill:num_seq]
    x_decode = hs_flat[num_prefill_tokens:num_total_tokens]  # [nd, H, D]
    B_decode = B_flat[num_prefill_tokens:num_total_tokens]  # [nd, G, N]
    C_decode = C_flat[num_prefill_tokens:num_total_tokens]  # [nd, G, N]
    dt_decode = dt_flat[num_prefill_tokens:num_total_tokens]  # [nd, H]

    dt_hp = dt_decode[:, :, None].expand(-1, num_heads, head_dim)
    dt_bias_hp = dt_bias[..., None].expand(num_heads, head_dim)
    A_full = A[..., None, None].expand(num_heads, head_dim, ssm_state_size)
    D_full = D[..., None].expand(num_heads, head_dim)

    return slot_idx_decode, x_decode, B_decode, C_decode, dt_hp, dt_bias_hp, A_full, D_full


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
        return ["batch_info_host", "cu_seqlen", "slot_idx", "use_initial_states"]

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
