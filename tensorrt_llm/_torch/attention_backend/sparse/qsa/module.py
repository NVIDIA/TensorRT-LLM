# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module-layer integration for QSA sparse attention."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm.logger import logger

from ..hooks import AttentionSparseHooks, register_attention_sparse_hooks
from .indexer import QSAIndexer, qsa_sparse_gqa, select_qsa_paged_tokens
from .metadata import QSAAttentionMetadata
from .params import QSASparseParams

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.attention import Attention


_DEFAULT_SCORE_WORKSPACE_BYTES = 128 * 1024 * 1024
_FP32_BYTES = 4


def _query_chunk_size(query_len: int, score_columns: int) -> int:
    raw = os.environ.get("TRTLLM_QSA_SPARSE_QUERY_CHUNK")
    if raw is not None:
        try:
            return max(int(raw), 1)
        except ValueError:
            pass

    # Paged QSA selection materializes one FP32 score per compressed block.
    # Bound its workspace while keeping all packed rows in one launch whenever
    # practical.  A small fixed chunk leaves long prefills launch-bound.
    score_bytes_per_row = max(score_columns * _FP32_BYTES, 1)
    workspace_rows = max(_DEFAULT_SCORE_WORKSPACE_BYTES // score_bytes_per_row, 1)
    return max(min(query_len, workspace_rows), 1)


class QSASparseHooks(AttentionSparseHooks):
    """Populate the side cache and run sparse GQA above the token budget."""

    def initialize(self, attention: "Attention") -> None:
        params = attention.sparse_params
        if not isinstance(params, QSASparseParams):
            raise TypeError("QSASparseHooks requires QSASparseParams")
        attention.indexer = QSAIndexer(attention, params)

    def forward(
        self,
        attention: "Attention",
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: QSAAttentionMetadata,
        attention_mask,
        attention_window_size,
        attention_mask_data,
        mrope_config,
        attention_sinks,
        relative_attention_bias,
        relative_attention_max_distance,
        has_lora,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        del (
            attention_mask,
            attention_window_size,
            attention_mask_data,
            mrope_config,
            attention_sinks,
            relative_attention_bias,
            relative_attention_max_distance,
            has_lora,
        )
        if not isinstance(attn_metadata, QSAAttentionMetadata):
            raise TypeError("QSA sparse attention received incompatible metadata")
        hidden_states = kwargs.get("qsa_index_hidden_states")
        position_ids = kwargs.get("qsa_position_ids")
        if hidden_states is None or position_ids is None:
            raise ValueError("QSA sparse attention requires hidden states and position IDs")

        num_tokens = attn_metadata.num_tokens
        hidden_states = hidden_states[:num_tokens]
        q_index, token_k, coordinates = attention.indexer.project(
            hidden_states,
            position_ids,
        )
        attention.indexer.update_cache_and_compress(
            attention.layer_idx,
            token_k,
            coordinates,
            attn_metadata,
        )

        params = attention.sparse_params
        max_kv_len = int(attn_metadata.kv_lens_runtime.max())
        if max_kv_len <= params.dense_seq_len_threshold:
            return None
        logger.info_once(
            "QSA exact sparse attention is active above the dense token budget",
            key="qsa_exact_sparse_attention_active",
        )

        q, k, v = attention.split_qkv(q, k, v)
        q = q[:num_tokens].reshape(num_tokens, attention.num_heads, attention.head_dim)
        k = k[:num_tokens].reshape(
            num_tokens,
            attention.num_key_value_heads,
            attention.head_dim,
        )
        v = v[:num_tokens].reshape_as(k)

        kv_pool = attn_metadata.kv_cache_manager.get_buffers(
            attention.layer_idx,
            kv_layout="HND",
        )
        if kv_pool is None or kv_pool.ndim != 5 or kv_pool.shape[1] != 2:
            raise RuntimeError("QSA sparse attention requires paged NHD K/V buffers")
        k_cache = kv_pool[:, 0]
        v_cache = kv_pool[:, 1]
        req_idx = attn_metadata.qsa_req_idx_per_token[:num_tokens]
        logical = attn_metadata.qsa_logical_positions[:num_tokens]
        tokens_per_block = attn_metadata.kv_cache_manager.tokens_per_block
        page_columns = logical // tokens_per_block
        pages = attn_metadata.qsa_block_table[
            req_idx.to(torch.long),
            page_columns,
        ].to(torch.long)
        within = (logical % tokens_per_block).to(torch.long)
        k_cache[pages, :, within, :] = k.to(k_cache.dtype)
        v_cache[pages, :, within, :] = v.to(v_cache.dtype)

        if attn_metadata.num_contexts == 0:
            index_cache = attn_metadata.kv_cache_manager.get_index_k_buffer(attention.layer_idx)
            if index_cache is None:
                raise RuntimeError(
                    f"QSA index cache is unavailable for layer {attention.layer_idx}"
                )
            # Speculative verification contributes multiple query rows per
            # request.  Keep that generation-only path on the same fixed-shape,
            # device-resident selection kernel as ordinary decode.  In
            # particular, do not derive its visible cache range from the host
            # KV-length mirror: speculative decoding advances the authoritative
            # lengths on device and the host mirror is not updated between its
            # sub-steps.
            sequence_lengths = attn_metadata.kv_lens_cuda_runtime[req_idx.to(torch.long)]
            selected = select_qsa_paged_tokens(
                q_index,
                index_cache,
                logical,
                sequence_lengths,
                req_idx,
                attn_metadata,
                params,
                top_k=attention.indexer.top_k,
                top_k_output=attn_metadata.qsa_topk_indices,
                top_k_row_starts=attn_metadata.qsa_topk_row_starts,
            )
            output = qsa_sparse_gqa(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                selected_tokens=selected,
                request_indices=req_idx,
                metadata=attn_metadata,
                softmax_scale=1.0 / (attention.q_scaling * attention.head_dim**0.5),
            )
            return output.reshape(num_tokens, -1)

        index_cache = attn_metadata.kv_cache_manager.get_index_k_buffer(attention.layer_idx)
        if index_cache is None:
            raise RuntimeError(f"QSA index cache is unavailable for layer {attention.layer_idx}")
        sequence_lengths = attn_metadata.kv_lens_cuda_runtime[req_idx.to(torch.long)]
        score_columns = (
            attn_metadata.qsa_block_table.shape[1] * tokens_per_block // params.compress_ratio
        )
        chunk_size = _query_chunk_size(num_tokens, score_columns)
        output = torch.empty_like(q)
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            packed_slice = slice(start, end)
            selected = select_qsa_paged_tokens(
                q_index[packed_slice],
                index_cache,
                logical[packed_slice],
                sequence_lengths[packed_slice],
                req_idx[packed_slice],
                attn_metadata,
                params,
                top_k=attention.indexer.top_k,
                top_k_output=attn_metadata.qsa_topk_indices[packed_slice],
                top_k_row_starts=attn_metadata.qsa_topk_row_starts[packed_slice],
            )
            output[packed_slice] = qsa_sparse_gqa(
                q=q[packed_slice],
                k_cache=k_cache,
                v_cache=v_cache,
                selected_tokens=selected,
                request_indices=req_idx[packed_slice],
                metadata=attn_metadata,
                softmax_scale=1.0 / (attention.q_scaling * attention.head_dim**0.5),
            )
        return output.reshape(num_tokens, -1)


register_attention_sparse_hooks("qsa", QSASparseHooks)


__all__ = ["QSASparseHooks"]
