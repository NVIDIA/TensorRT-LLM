# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer sparse-MLA helpers for DeepSeek-V4 on SM120/SM121."""

import math

import torch

from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding

from ...interface import AttentionForwardArgs, AttentionInputType
from ..flashinfer_utils import (
    SPARSE_MLA_SPLIT_KV_TILE,
    allocate_sparse_mla_split_workspace,
    get_sparse_mla_op,
)
from . import footer_scale_kv
from .metadata import DeepseekV4TrtllmAttentionMetadata
from .params import DeepseekV4AttentionType


def run_flashinfer_sparse_mla(
    attn,
    q: torch.Tensor,
    metadata: DeepseekV4TrtllmAttentionMetadata,
    forward_args: AttentionForwardArgs,
    rotary_emb: RotaryEmbedding,
) -> None:
    """Run DeepSeek-V4 attention for ``FlashInferSparseMlaFmha``."""
    if forward_args.enable_dsv4_epilogue_fusion:
        raise NotImplementedError(
            "DSv4 epilogue fusion is fused into the C++ attention op; the "
            "FlashInfer SM120 FMHA does not provide it."
        )
    if metadata.max_draft_tokens > 0:
        raise NotImplementedError(
            "MTP / speculative decoding is not supported by the FlashInfer DSv4 FMHA yet."
        )

    sinks = forward_args.attention_sinks
    if sinks is not None:
        sinks = sinks.to(torch.float32)

    attention_input_type = forward_args.attention_input_type
    if attention_input_type == AttentionInputType.context_only:
        start_idx, end_idx = 0, metadata.num_ctx_tokens
    elif attention_input_type == AttentionInputType.generation_only:
        start_idx, end_idx = metadata.num_ctx_tokens, metadata.num_tokens
    else:
        raise NotImplementedError(
            "The FlashInfer DSv4 FMHA expects phase-split calls "
            "(context_only / generation_only), matching the DSv4 model forward structure."
        )
    num_tokens = end_idx - start_idx

    output = forward_args.output
    if output is None:
        raise RuntimeError("FlashInfer DSv4 FMHA requires a preallocated output.")
    if num_tokens == 0:
        return

    q_view = q.view(num_tokens, attn.num_heads, attn.head_dim)
    if q_view.dtype != torch.bfloat16:
        raise NotImplementedError(
            "FlashInfer SM120 sparse MLA takes bf16 queries; the fused "
            "FP8-Q path must stay disabled for this FMHA."
        )
    latent_cache = forward_args.latent_cache
    latent_rows = (
        latent_cache.view(num_tokens, attn.head_dim)
        if latent_cache is not None and forward_args.update_kv_cache
        else None
    )
    footer_scale_kv.apply_rope_and_append_swa(
        attn,
        metadata,
        q,
        latent_rows,
        start_idx,
        end_idx,
        rotary_emb.rotary_cos_sin,
        rotary_emb.is_neox,
        apply_rope=not forward_args.skip_mla_rope_generation,
    )
    positions = metadata.token_positions_cuda[start_idx:end_idx]

    sparse_runtime_params = forward_args.sparse_runtime_params
    swa_indices = sparse_runtime_params.sparse_attn_indices
    extra_indices = sparse_runtime_params.sparse_attn_offsets
    if swa_indices is None:
        raise RuntimeError("FlashInfer DSv4 FMHA requires sparse attention indices.")

    window = attn.sparse_attention_config.window_size
    swa_lens = (positions + 1).clamp(max=window).to(torch.int32)

    if attn.compress_ratio > 1:
        if extra_indices is None:
            raise RuntimeError("Compressed DeepSeek-V4 layers require extra sparse indices.")
        extra_lens = (
            metadata.sparse_mla_topk_lens[attn.compress_ratio][start_idx:end_idx] - window
        ).clamp(min=0)
        extra_pool = footer_scale_kv.get_pool_2d(
            metadata, DeepseekV4AttentionType.COMPRESS, attn.compress_ratio
        )
        num_splits = (window + SPARSE_MLA_SPLIT_KV_TILE - 1) // SPARSE_MLA_SPLIT_KV_TILE + (
            extra_indices.shape[1] + SPARSE_MLA_SPLIT_KV_TILE - 1
        ) // SPARSE_MLA_SPLIT_KV_TILE
    else:
        extra_indices = None
        extra_lens = None
        extra_pool = None
        num_splits = (window + SPARSE_MLA_SPLIT_KV_TILE - 1) // SPARSE_MLA_SPLIT_KV_TILE

    out_view = output.view(num_tokens, attn.num_heads, attn.head_dim)
    out_lse = torch.empty(num_tokens, attn.num_heads, dtype=torch.float32, device=q.device)
    mid_out, mid_lse = allocate_sparse_mla_split_workspace(
        num_tokens=num_tokens,
        num_heads=attn.num_heads,
        num_splits=num_splits,
        value_dim=attn.head_dim,
        device=q.device,
    )

    sm_scale = 1.0 / (attn.q_scaling * math.sqrt(attn.head_dim))
    swa_pool = footer_scale_kv.get_pool_2d(metadata, DeepseekV4AttentionType.SWA, 1)
    swa_pool_paged = swa_pool.view(-1, footer_scale_kv.PAGE_SIZE, footer_scale_kv.TOKEN_BYTES)
    extra_pool_paged = None
    if extra_pool is not None:
        extra_pool_paged = extra_pool.view(extra_pool.shape[0], -1, footer_scale_kv.TOKEN_BYTES)

    get_sparse_mla_op()(
        q_view.contiguous(),
        swa_pool_paged,
        swa_indices,
        out_view,
        out_lse,
        sm_scale,
        d_v=attn.head_dim,
        topk_length=swa_lens,
        attn_sink=sinks,
        extra_kv_cache=extra_pool_paged,
        extra_indices=extra_indices,
        extra_topk_length=extra_lens,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )
