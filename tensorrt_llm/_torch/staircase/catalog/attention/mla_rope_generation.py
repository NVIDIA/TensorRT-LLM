# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA generation-phase RoPE + latent KV-cache append + FMHA scheduler-buffer fill."""

from typing import List, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def mla_rope_generation(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    rotary_cos_sin: Optional[torch.Tensor],
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    mla_bmm1_scale: Optional[torch.Tensor],
    mla_bmm2_scale: Optional[torch.Tensor],
    quant_q_buffer: Optional[torch.Tensor],
    sequence_length: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    num_contexts: int,
    kv_cache_block_offsets: Optional[torch.Tensor],
    host_kv_cache_pool_pointers: Optional[torch.Tensor],
    host_kv_cache_pool_mapping: Optional[torch.Tensor],
    kv_scale_orig_quant: Optional[torch.Tensor],
    kv_scale_quant_orig: Optional[torch.Tensor],
    # rc26: when None the op falls back to kv_scale_orig_quant, which is what
    # it did before this parameter existed (dsv3RopeOp.cpp:280).
    kv_cache_scale_orig_quant: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    block_ids_per_seq: Optional[torch.Tensor],
    helix_tensor_params: List[Optional[torch.Tensor]],
    predicted_tokens_per_seq: int,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    # rc26: 0 or rope_size, and non-zero requires an FP4 KV pool.
    residual_dim: int,
    tokens_per_block: int,
    attention_window_size: int,
    beam_width: int,
    quant_mode: int,
    q_scaling: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    rope_append: bool,
    # Added in rc26; every default below reproduces the op's pre-rc26
    # behaviour. kv_norm_weight non-None would fold the kv_a_layernorm into
    # this kernel, which then reads latent_cache RAW -- a caller that already
    # normalized would be normalizing twice.
    kv_norm_weight: Optional[torch.Tensor] = None,
    kv_norm_eps: float = 1e-6,
    precomputed_cu_seqlens: bool = False,
    precomputed_fmha_scheduler: bool = False,
    kv_only: bool = False,
    kv_done_elsewhere: bool = False,
    quant_scale_qkv: Optional[torch.Tensor] = None,
) -> None:
    """RoPE q_pe, append latent_cache rows to the paged MLA KV cache, and fill
    the decode-FMHA scheduler buffers. Returns None.

    Where the roped q_pe lands depends on the pool: into fused_q's tail over a
    bf16 pool, into quant_q_buffer over an fp8 one (where fused_q is read, not
    written). See the contract."""
    torch.ops.trtllm.mla_rope_generation(
        fused_q,
        q_pe,
        latent_cache,
        rotary_cos_sin,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        quant_q_buffer,
        sequence_length,
        host_past_key_value_lengths,
        host_context_lengths,
        num_contexts,
        kv_cache_block_offsets,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        kv_scale_orig_quant,
        kv_scale_quant_orig,
        kv_cache_scale_orig_quant,
        out_scale,
        block_ids_per_seq,
        helix_tensor_params,
        predicted_tokens_per_seq,
        layer_idx,
        num_heads,
        num_kv_heads,
        head_size,
        residual_dim,
        tokens_per_block,
        attention_window_size,
        beam_width,
        quant_mode,
        q_scaling,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rope_append,
        kv_norm_weight,
        kv_norm_eps,
        precomputed_cu_seqlens,
        precomputed_fmha_scheduler,
        kv_only,
        kv_done_elsewhere,
        quant_scale_qkv,
    )
