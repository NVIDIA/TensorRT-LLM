# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer sparse-MLA helpers for DSA models on SM120/SM121."""

import math

import torch

from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding

from ..interface import AttentionForwardArgs, AttentionInputType
from . import inline_scale_kv
from .dsa import DSAtrtllmAttentionMetadata
from .flashinfer_utils import (
    SPARSE_MLA_SPLIT_KV_TILE,
    allocate_sparse_mla_split_workspace,
    get_sparse_mla_op,
)


def _inline_scale_pool_paged(metadata: DSAtrtllmAttentionMetadata) -> torch.Tensor:
    """Return the layer-interleaved primary pool as inline-scale pages."""
    manager = metadata.kv_cache_manager
    cached = getattr(metadata, "_inline_scale_pool", None)
    if cached is not None and cached[0] == id(manager):
        return cached[1]
    pool = manager.get_unique_primary_pool()
    num_blocks, num_layers = pool.shape[0], pool.shape[1]
    page_bytes = manager.tokens_per_block * inline_scale_kv.TOKEN_BYTES
    view = pool.view(torch.uint8).reshape(num_blocks * num_layers, page_bytes)
    assert view.is_contiguous()
    paged = view.view(-1, manager.tokens_per_block, inline_scale_kv.TOKEN_BYTES)
    metadata._inline_scale_pool = (id(manager), paged)
    return paged


def _latent_append(
    attn,
    metadata: DSAtrtllmAttentionMetadata,
    latent_rows: torch.Tensor,
    start_idx: int,
    end_idx: int,
    is_generation: bool,
) -> None:
    """Quantize the new latent rows and scatter them into the main pool."""
    metadata._ensure_pool_view_cached()
    positions = metadata.token_positions_cuda[start_idx:end_idx]
    if is_generation:
        block_table = metadata._cached_block_table_gen
        req_idx = metadata._cached_req_idx_gen
    else:
        block_table = metadata._cached_block_table_ctx
        req_idx = metadata._cached_req_idx_ctx
    loc = torch.ops.trtllm.convert_req_index_to_global(
        req_idx,
        block_table,
        positions.unsqueeze(1).contiguous(),
        metadata._cached_tokens_per_block,
        1,
        metadata._cached_stride_factor,
        attn.get_local_layer_idx(metadata),
    ).view(-1)
    pool = _inline_scale_pool_paged(metadata)
    inline_scale_kv.quant_scatter(
        pool.view(pool.shape[0], -1),
        loc,
        latent_rows,
        page_size=metadata._cached_tokens_per_block,
    )


def run_flashinfer_sparse_mla(
    attn,
    q: torch.Tensor,
    metadata: DSAtrtllmAttentionMetadata,
    forward_args: AttentionForwardArgs,
    rotary_emb: RotaryEmbedding,
) -> None:
    """Run DSA attention for ``FlashInferSparseMlaFmha``."""
    if metadata.max_draft_tokens > 0:
        raise NotImplementedError(
            "MTP / speculative decoding is not supported by the FlashInfer DSA FMHA yet."
        )

    attention_input_type = forward_args.attention_input_type
    if attention_input_type == AttentionInputType.context_only:
        start_idx, end_idx = 0, metadata.num_ctx_tokens
        is_generation = False
    elif attention_input_type == AttentionInputType.generation_only:
        start_idx, end_idx = metadata.num_ctx_tokens, metadata.num_tokens
        is_generation = True
    else:
        raise NotImplementedError(
            "The FlashInfer DSA FMHA expects phase-split calls "
            "(context_only / generation_only), matching the DSA model forward structure."
        )
    num_tokens = end_idx - start_idx

    kv_lora_rank = attn.mla_params.kv_lora_rank
    output = forward_args.output
    if output is None:
        raise RuntimeError("FlashInfer DSA FMHA requires a preallocated output.")
    if num_tokens == 0:
        return

    q_view = q.view(num_tokens, attn.num_heads, attn.head_dim)
    if q_view.dtype != torch.bfloat16:
        raise NotImplementedError(
            "FlashInfer SM120 sparse MLA takes bf16 queries; fused FP8-Q "
            "paths must stay disabled for this FMHA."
        )

    positions = metadata.token_positions_cuda[start_idx:end_idx]
    latent_cache = forward_args.latent_cache
    latent_rows = (
        latent_cache.view(num_tokens, attn.head_dim)
        if latent_cache is not None and forward_args.update_kv_cache
        else None
    )
    if not forward_args.skip_mla_rope_generation:
        torch.ops.trtllm.mla_rope_inplace(
            q_view,
            positions,
            rotary_emb.rotary_cos_sin,
            attn.num_heads,
            attn.mla_params.kv_lora_rank,
            attn.mla_params.qk_rope_head_dim,
            False,
            rotary_emb.is_neox,
        )
        if latent_rows is not None:
            torch.ops.trtllm.mla_rope_inplace(
                latent_rows.view(num_tokens, 1, attn.head_dim),
                positions,
                rotary_emb.rotary_cos_sin,
                1,
                attn.mla_params.kv_lora_rank,
                attn.mla_params.qk_rope_head_dim,
                False,
                rotary_emb.is_neox,
            )

    if latent_rows is not None:
        _latent_append(
            attn,
            metadata,
            latent_rows,
            start_idx,
            end_idx,
            is_generation,
        )

    topk_indices_global = forward_args.sparse_runtime_params.sparse_attn_indices
    if topk_indices_global is None:
        raise RuntimeError("FlashInfer DSA FMHA requires sparse attention indices.")
    if not topk_indices_global.is_contiguous():
        topk_indices_global = topk_indices_global.contiguous()
    topk = topk_indices_global.shape[-1]

    out_view = output.view(num_tokens, attn.num_heads, kv_lora_rank)
    out_lse = torch.empty(num_tokens, attn.num_heads, dtype=torch.float32, device=q.device)
    num_splits = (topk + SPARSE_MLA_SPLIT_KV_TILE - 1) // SPARSE_MLA_SPLIT_KV_TILE
    mid_out, mid_lse = allocate_sparse_mla_split_workspace(
        num_tokens=num_tokens,
        num_heads=attn.num_heads,
        num_splits=num_splits,
        value_dim=kv_lora_rank,
        device=q.device,
    )

    qk_head_dim = attn.mla_params.qk_nope_head_dim + attn.mla_params.qk_rope_head_dim
    sm_scale = 1.0 / (attn.q_scaling * math.sqrt(qk_head_dim))

    get_sparse_mla_op()(
        q_view.contiguous(),
        _inline_scale_pool_paged(metadata),
        topk_indices_global,
        out_view,
        out_lse,
        sm_scale,
        d_v=kv_lora_rank,
        kv_scale_format="arbitrary_fp32",
        mid_out=mid_out,
        mid_lse=mid_lse,
    )
