# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA context-phase RoPE of q_pe/k_pe in place + latent paged-KV-cache append."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def mla_rope_append_paged_kv_assign_q(
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    num_contexts: int,
    cu_ctx_cached_kv_lens: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    max_input_uncached_seq_len: int,
    cos_sin_cache: torch.Tensor,
    head_num: int,
    nope_size: int,
    rope_size: int,
    lora_size: int,
    kv_cache_block_offsets: torch.Tensor,
    host_kv_cache_pool_pointers: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    kv_scale_orig_quant: Optional[torch.Tensor],
    # ``residual_dim`` (rc26; absent in rc21) must be 0 or ``rope_size``,
    # and the op rejects non-zero unless the KV pool is FP4. Every caller
    # here runs a bf16 or fp8-e4m3 pool, so 0 is the only legal value.
    residual_dim: int,
    layer_idx: int,
    tokens_per_block: int,
    attention_window_size: int,
    beam_width: int,
    quant_mode: int,
) -> None:
    """RoPE each new context token's q_pe (in q) and k_pe (in latent_cache)
    in place at its absolute position, and append the token's latent row
    [compressed_kv | rope(k_pe)] to the paged MLA KV cache. Returns None."""
    torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
        q,
        latent_cache,
        num_contexts,
        cu_ctx_cached_kv_lens,
        cu_seq_lens,
        max_input_uncached_seq_len,
        cos_sin_cache,
        head_num,
        nope_size,
        rope_size,
        lora_size,
        kv_cache_block_offsets,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        kv_scale_orig_quant,
        residual_dim,
        layer_idx,
        tokens_per_block,
        attention_window_size,
        beam_width,
        quant_mode,
    )
