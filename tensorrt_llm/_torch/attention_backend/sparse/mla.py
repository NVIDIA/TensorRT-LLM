# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse backend phase dispatch for the shared MLA module."""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._utils import get_sm_version

from .dsa.mla_backend import forward_sparse_mla_kvcache_bf16, should_use_short_mha

if TYPE_CHECKING:
    pass


def forward_context_sparse_mla(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    enable_dsv4_epilogue_fusion: bool = False,
    dsv4_epilogue_output: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run context-phase attention for DSA models.

    Dispatches to the short-seq MHA path (forward_context) when the max
    per-sequence KV length (including cached tokens) is within the
    threshold, or falls through to the absorption/sparse MLA path
    otherwise.  forward_context() further dispatches to the appropriate
    handler (forward_context_default, forward_context_with_cached_kv, or
    forward_context_with_chunked_prefill) based on cached-KV state.

    Args:
        q: Query tensor, shape [num_ctx_tokens, num_heads * qk_head_dim].
        compressed_kv: Latent KV, shape [num_ctx_tokens, kv_lora_rank].
        k_pe: RoPE key portion, shape [num_ctx_tokens, qk_rope_head_dim].
        attn_metadata: Attention metadata for the current batch.
        output: Pre-allocated output tensor, written in-place.
        latent_cache: Concatenated [compressed_kv, k_pe] for KV cache.
        topk_indices: Sparse routing indices from the indexer (None when
            the short-seq MHA path is used).
        position_ids: Token position IDs (required for short-seq MHA).
    """
    # Short-sequence MHA: bypass absorption path for short prefills,
    # using kv_b_proj expansion + standard attention instead.
    # See __init__ comment for rationale. topk_indices is not used
    # because dense attention is faster than sparse routing at this scale.
    # forward_context() handles cached tokens by dispatching to
    # forward_context_with_cached_kv or forward_context_with_chunked_prefill.
    if not enable_dsv4_epilogue_fusion and should_use_short_mha(
        self, attn_metadata, position_ids
    ):
        return self.forward_context(
            q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
        )

    if get_sm_version() >= 100:
        return self.forward_absorption_context(
            q,
            compressed_kv,
            k_pe,
            attn_metadata,
            output,
            position_ids=position_ids,
            latent_cache=latent_cache,
            topk_indices=topk_indices,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
            dsv4_epilogue_output=dsv4_epilogue_output,
        )
    else:
        assert not self.is_deepseek_v4, "DeepSeek-V4 is not supported on pre-blackwell GPUs."
        return forward_sparse_mla_kvcache_bf16(
            self, q, latent_cache, attn_metadata, output, topk_indices, is_generation=False
        )


def forward_generation_sparse_mla(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    latent_cache: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    enable_dsv4_epilogue_fusion: bool = False,
    dsv4_epilogue_output: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if get_sm_version() >= 100:
        return self.forward_absorption_generation(
            q,
            compressed_kv,
            k_pe,
            attn_metadata,
            output,
            position_ids=position_ids,
            latent_cache=latent_cache,
            topk_indices=topk_indices,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
            dsv4_epilogue_output=dsv4_epilogue_output,
        )
    else:
        assert not self.is_deepseek_v4, "DeepSeek-V4 is not supported on pre-blackwell GPUs."
        return forward_sparse_mla_kvcache_bf16(
            self, q, latent_cache, attn_metadata, output, topk_indices, is_generation=True
        )
