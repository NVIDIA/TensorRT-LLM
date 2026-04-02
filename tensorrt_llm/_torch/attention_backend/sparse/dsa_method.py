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
"""DSA (Dynamic Sparse Attention) method for MLA.

Implements the SparseAttentionMethod protocol with DSA-specific dispatch:
short-seq MHA optimization, absorption path (SM100+), and sparse FlashMLA
kernels (SM90).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import get_sm_version, nvtx_range, nvtx_range_debug
from tensorrt_llm.logger import logger

from ..interface import AttentionMetadata
from .dsa import DSAtrtllmAttentionMetadata, transform_local_topk_and_prepare_pool_view

# Import FlashMLA sparse attention kernel
try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None

if TYPE_CHECKING:
    from ...modules.attention import MLA


def _should_use_short_mha(
    mla: MLA,
    attn_metadata: AttentionMetadata,
    position_ids: Optional[torch.Tensor],
) -> bool:
    """Check if the short-seq MHA optimization should be used for context.

    Uses max_ctx_kv_len (max total KV length per context sequence,
    including cached tokens) when available, to correctly account for
    chunked context where the full attention span exceeds the threshold
    even if the new token count is small.  Falls back to num_ctx_tokens
    (total new context tokens) when max_ctx_kv_len is not set.
    """
    if not (
        mla.short_seq_mha_threshold > 0
        and not mla.apply_rotary_emb
        and mla.mapping.cp_size == 1
        and position_ids is not None
    ):
        return False
    effective_len = getattr(attn_metadata, "max_ctx_kv_len", attn_metadata.num_ctx_tokens)
    return effective_len <= mla.short_seq_mha_threshold


@nvtx_range("forward_sparse_mla_kvcache_bf16")
def _forward_sparse_mla_kvcache_bf16(
    mla: MLA,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    attn_metadata: DSAtrtllmAttentionMetadata,
    output: torch.Tensor,
    topk_indices: torch.Tensor,
    is_generation: bool = False,
) -> torch.Tensor:
    """Forward sparse MLA (DSA) for BF16 KV cache using FlashMLA kernels.

    To form the input for FlashMLA kernel and adapt our KV cache manager:
    1. Append current tokens to paged cache and apply rope to q/k via
       mla_rope_append_paged_kv_assign_q
    2. Load full kv cache from paged memory (with k rope applied)
    3. Call FlashMLA sparse attention kernel for sparse prefill/decode
    """
    from ...modules.attention import fp8_block_scaling_bmm_out

    assert isinstance(attn_metadata, DSAtrtllmAttentionMetadata), (
        "DSA requires DSAtrtllmAttentionMetadata"
    )
    trtllm_attention = mla.mqa
    with nvtx_range_debug(f"mla_rope_append_paged_kv_assign_q_is_generation={is_generation}"):
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata, is_generation=is_generation
        )

    num_tokens = q.shape[0]
    q_nope, q_rope = q.view(-1, mla.num_heads_tp, mla.qk_head_dim).split(
        [mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1
    )
    q_nope_out = torch.empty(
        [num_tokens, mla.num_heads_tp, mla.kv_lora_rank],
        dtype=q.dtype,
        device=q.device,
    )

    if mla.k_b_proj_trans.dtype == torch.bfloat16:
        q_nope_t = q_nope.transpose(0, 1)
        q_nope_out = q_nope_out.transpose(0, 1)
        torch.ops.trtllm.bmm_out(q_nope_t, mla.k_b_proj_trans.transpose(1, 2), q_nope_out)
    elif mla.k_b_proj_trans.dtype == torch.float8_e4m3fn:
        q_nope_out = q_nope_out.transpose(0, 1)
        fp8_block_scaling_bmm_out(
            q_nope,
            mla.k_b_proj_trans,
            mla.k_b_proj_trans_scale,
            q_nope_out,
            mla.k_b_proj_trans_dequant,
            mla.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {mla.k_b_proj_trans.dtype}.")

    q_nope_out = q_nope_out.transpose(0, 1)
    q_concat = torch.cat([q_nope_out, q_rope], dim=-1)

    sm_version = get_sm_version()
    if sm_version >= 100:
        padding = 128
        assert mla.num_heads_tp <= padding, (
            f"SM100 FlashMLA sparse kernel requires exactly {padding} heads, "
            f"got {mla.num_heads_tp}. Padding from values > {padding} is not supported."
        )
    else:  # SM90
        padding = ((mla.num_heads_tp + 63) // 64) * 64

    if mla.num_heads_tp != padding:
        logger.warning_once(
            f"Padding num_heads from {mla.num_heads_tp} to {padding} "
            f"due to FlashMLA sparse attention kernel requirement",
            key="sparse_mla_padding_warning",
        )
        q_padded = q_concat.new_empty((num_tokens, padding, q_concat.shape[2]))
        q_padded[:, : mla.num_heads_tp, :] = q_concat
        q_concat = q_padded

    topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
        topk_indices,
        attn_metadata,
        layer_idx=mla.layer_idx,
        is_generation=is_generation,
    )
    topk_indices_pool = topk_indices_pool.view(num_tokens, 1, -1)
    if flash_mla_sparse_fwd is not None:
        attn_out_latent = flash_mla_sparse_fwd(
            q_concat, kv_cache_pool, topk_indices_pool, mla.softmax_scale
        )[0]
    else:
        raise RuntimeError(
            "flash_mla_sparse_fwd not available. Please ensure FlashMLA module is built."
        )

    attn_out_latent = attn_out_latent[:, : mla.num_heads_tp, :]
    attn_out_latent = attn_out_latent.view([-1, mla.num_heads_tp, mla.kv_lora_rank])
    if mla.num_heads_tp != padding:
        attn_out_latent = attn_out_latent.contiguous()

    assert attn_out_latent.shape[0] == q.shape[0] and attn_out_latent.shape[1] == mla.num_heads_tp

    attn_output = output.view([num_tokens, mla.num_heads_tp, mla.v_head_dim])

    if mla.v_b_proj.dtype == torch.bfloat16:
        torch.ops.trtllm.bmm_out(
            attn_out_latent.transpose(0, 1),
            mla.v_b_proj.transpose(1, 2),
            attn_output.transpose(0, 1),
        )
    elif mla.v_b_proj.dtype == torch.float8_e4m3fn:
        fp8_block_scaling_bmm_out(
            attn_out_latent,
            mla.v_b_proj,
            mla.v_b_proj_scale,
            attn_output.transpose(0, 1),
            mla.v_b_proj_dequant,
            mla.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {mla.v_b_proj.dtype}.")
    return output


class DSASparseMethod:
    """DSA (Dynamic Sparse Attention) implementation of SparseAttentionMethod.

    Handles context/generation dispatch for DSA: short-seq MHA optimization,
    absorption path (SM100+), and sparse FlashMLA kernels (SM90).
    """

    def __init__(self, short_seq_mha_threshold: int = 0):
        self.short_seq_mha_threshold = short_seq_mha_threshold

    def dispatch_context(
        self,
        mla: MLA,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor],
        topk_indices: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> None:
        """Dispatch context-phase attention for DSA."""
        if _should_use_short_mha(mla, attn_metadata, position_ids):
            mla.forward_context(
                q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
            )
        elif get_sm_version() >= 100:
            mla.forward_absorption_context(
                q,
                compressed_kv,
                k_pe,
                attn_metadata,
                output,
                latent_cache=latent_cache,
                topk_indices=topk_indices,
            )
        else:
            _forward_sparse_mla_kvcache_bf16(
                mla, q, latent_cache, attn_metadata, output, topk_indices, is_generation=False
            )

    def dispatch_generation(
        self,
        mla: MLA,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor],
        topk_indices: Optional[torch.Tensor],
    ) -> None:
        """Dispatch generation-phase attention for DSA."""
        if get_sm_version() >= 100:
            mla.forward_absorption_generation(
                q,
                compressed_kv,
                k_pe,
                attn_metadata,
                output,
                latent_cache=latent_cache,
                topk_indices=topk_indices,
            )
        else:
            _forward_sparse_mla_kvcache_bf16(
                mla, q, latent_cache, attn_metadata, output, topk_indices, is_generation=True
            )
