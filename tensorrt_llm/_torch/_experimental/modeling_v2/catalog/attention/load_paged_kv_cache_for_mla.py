# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather each context sequence's full MLA latent KV from the paged cache."""

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def load_paged_kv_cache_for_mla(
    out_dtype: torch.dtype,
    num_contexts: int,
    num_ctx_kv_tokens: int,
    max_ctx_kv_len: int,
    cu_ctx_kv_lens: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    host_kv_cache_pool_pointers: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    kv_scale_quant_orig: Optional[torch.Tensor],
    layer_idx: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    tokens_per_block: int,
    attention_window_size: int,
    beam_width: int,
    quant_mode: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copy positions [0, L_s) of every context sequence out of the paged MLA
    latent cache into two new contiguous tensors (compressed_kv, k_pe)."""
    compressed_kv, k_pe = torch.ops.trtllm.load_paged_kv_cache_for_mla(
        out_dtype,
        num_contexts,
        num_ctx_kv_tokens,
        max_ctx_kv_len,
        cu_ctx_kv_lens,
        kv_cache_block_offsets,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        kv_scale_quant_orig,
        layer_idx,
        kv_lora_rank,
        qk_rope_head_dim,
        tokens_per_block,
        attention_window_size,
        beam_width,
        quant_mode,
    )
    return compressed_kv, k_pe
