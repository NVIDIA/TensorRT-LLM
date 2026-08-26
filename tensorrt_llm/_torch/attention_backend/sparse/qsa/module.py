# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module-layer integration for QSA sparse attention."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm.logger import logger

from ..hooks import AttentionSparseHooks, register_attention_sparse_hooks
from .indexer import QSAIndexer, qsa_sparse_gqa, select_qsa_decode_tokens, select_qsa_tokens
from .metadata import QSAAttentionMetadata
from .params import QSASparseParams

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.attention import Attention


_DEFAULT_QUERY_CHUNK = 32


def _query_chunk_size() -> int:
    raw = os.environ.get("TRTLLM_QSA_SPARSE_QUERY_CHUNK")
    if raw is None:
        return _DEFAULT_QUERY_CHUNK
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_QUERY_CHUNK
    return max(value, 1)


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
            selected = select_qsa_decode_tokens(
                q_index,
                index_cache,
                logical,
                sequence_lengths,
                req_idx,
                attn_metadata,
                params,
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

        output = torch.empty_like(q)
        seq_lens = attn_metadata.seq_lens[: attn_metadata.num_seqs].tolist()
        kv_lens = attn_metadata.kv_lens_runtime.tolist()
        token_offset = 0
        chunk_size = _query_chunk_size()
        for request_idx, (query_len, sequence_len) in enumerate(zip(seq_lens, kv_lens)):
            query_len = int(query_len)
            sequence_len = int(sequence_len)
            request_slice = slice(token_offset, token_offset + query_len)
            request_positions = logical[request_slice]
            complete_blocks = sequence_len // params.compress_ratio
            compressed_keys = attention.indexer.gather_compressed_keys(
                attention.layer_idx,
                request_idx,
                complete_blocks,
                attn_metadata,
            )
            for start in range(0, query_len, chunk_size):
                end = min(start + chunk_size, query_len)
                packed_slice = slice(token_offset + start, token_offset + end)
                chunk_positions = request_positions[start:end]
                selected = select_qsa_tokens(
                    q_index[packed_slice],
                    compressed_keys,
                    chunk_positions,
                    sequence_len,
                    params,
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
            token_offset += query_len
        if token_offset != num_tokens:
            raise RuntimeError(
                f"QSA packed token accounting mismatch: {token_offset} != {num_tokens}"
            )
        return output.reshape(num_tokens, -1)


register_attention_sparse_hooks("qsa", QSASparseHooks)


__all__ = ["QSASparseHooks"]
