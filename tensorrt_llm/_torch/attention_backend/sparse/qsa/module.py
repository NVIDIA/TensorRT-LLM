# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module-layer integration for QSA sparse attention."""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm.logger import logger

from ...interface import AttentionMask, PredefinedAttentionMask
from ..hooks import AttentionSparseHooks, register_attention_sparse_hooks
from .constants import (
    QSA_KEY_ROLE_INDEX,
    QSA_MAIN_KV_ROLES,
    QSA_SPARSE_KV_CACHE_DTYPES,
    QSA_VALUE_ROLE_INDEX,
)
from .indexer import QSAIndexer, qsa_sparse_gqa, select_qsa_paged_tokens
from .metadata import QSAAttentionMetadata
from .params import QSASparseParams

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.attention import Attention


_DEFAULT_SCORE_WORKSPACE_BYTES = 128 * 1024 * 1024
_FP32_BYTES = torch.empty((), dtype=torch.float32).element_size()


def _query_chunk_size(query_len: int, score_columns: int) -> int:
    # Paged QSA selection materializes one FP32 score per compressed block.
    # Bound its workspace while keeping all packed rows in one launch whenever
    # practical.  A small fixed chunk leaves long prefills launch-bound.
    score_bytes_per_row = max(score_columns * _FP32_BYTES, 1)
    workspace_rows = max(_DEFAULT_SCORE_WORKSPACE_BYTES // score_bytes_per_row, 1)
    return max(min(query_len, workspace_rows), 1)


def _store_paged_kv_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
) -> None:
    """Store rows whose request, table column, and physical page are valid."""
    page_columns = logical_positions // tokens_per_block
    request_rows = request_indices.to(torch.long)
    table_rows, table_columns = block_table.shape
    if table_rows == 0 or table_columns == 0:
        valid_lookup = torch.zeros_like(request_rows, dtype=torch.bool)
        pages = torch.full_like(request_rows, -1)
    else:
        valid_lookup = (
            (request_rows >= 0)
            & (request_rows < table_rows)
            & (page_columns >= 0)
            & (page_columns < table_columns)
        )
        safe_rows = request_rows.clamp(0, table_rows - 1)
        safe_columns = page_columns.clamp(0, table_columns - 1)
        pages = block_table[safe_rows, safe_columns].to(torch.long)
    within = (logical_positions % tokens_per_block).to(torch.long)
    valid_rows = valid_lookup & (pages >= 0) & (pages < k_cache.shape[0])
    pages = pages[valid_rows]
    within = within[valid_rows]
    k_cache[pages, :, within, :] = k[valid_rows].to(k_cache.dtype)
    v_cache[pages, :, within, :] = v[valid_rows].to(v_cache.dtype)


class QSASparseHooks(AttentionSparseHooks):
    """Keep the QSA side cache current and replace dense attention when useful."""

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
        attention_mask: AttentionMask,
        attention_window_size: Optional[int],
        attention_mask_data: Optional[torch.Tensor],
        mrope_config: Optional[dict[str, object]],
        attention_sinks: Optional[torch.Tensor],
        relative_attention_bias: Optional[torch.Tensor],
        relative_attention_max_distance: int,
        has_lora: bool,
        **kwargs: object,
    ) -> Optional[torch.Tensor]:
        if attention_mask is not PredefinedAttentionMask.CAUSAL:
            raise NotImplementedError("QSA sparse attention supports only a causal mask")
        unsupported = {
            "attention_window_size": attention_window_size,
            "attention_mask_data": attention_mask_data,
            "attention_sinks": attention_sinks,
            "relative_attention_bias": relative_attention_bias,
        }
        active = [name for name, value in unsupported.items() if value is not None]
        if relative_attention_max_distance:
            active.append("relative_attention_max_distance")
        if active:
            raise NotImplementedError("QSA sparse attention does not support: " + ", ".join(active))
        if has_lora:
            # LoRA currently updates only the regular QKV projection, not the
            # replicated index Q/K projection that determines token selection.
            raise NotImplementedError("QSA sparse attention does not support LoRA")
        # The model's QKNormRoPEAttention preprocesses q/k before this hook;
        # mrope_config is therefore backend state, not work for the QSA kernel.
        del mrope_config
        if not isinstance(attn_metadata, QSAAttentionMetadata):
            raise TypeError("QSA sparse attention received incompatible metadata")
        num_tokens = attn_metadata.num_tokens
        if num_tokens == 0:
            # Empty ADP ranks have no QSA side state to advance. Let the regular
            # backend preserve its established empty-output contract.
            return None
        hidden_states = kwargs.get("qsa_index_hidden_states")
        position_ids = kwargs.get("qsa_position_ids")
        if hidden_states is None or position_ids is None:
            raise ValueError("QSA sparse attention requires hidden states and position IDs")

        # The standard TRT-LLM backend owns quantized formats with auxiliary
        # scale pages, such as NVFP4. Use it until QSA kernels consume the data
        # and scale pages together; the cache manager itself remains generic.
        kv_cache_dtype = attn_metadata.kv_cache_manager.dtype
        if kv_cache_dtype not in QSA_SPARSE_KV_CACHE_DTYPES:
            logger.warning_once(
                f"QSA sparse K/V kernels do not support {kv_cache_dtype}; "
                "using the regular attention backend",
                key=f"qsa_dense_fallback_{kv_cache_dtype}",
            )
            return None

        hidden_states = hidden_states[:num_tokens]
        q_index = attention.indexer.project_and_update_cache(
            hidden_states,
            position_ids,
            attention.layer_idx,
            attn_metadata,
        )

        params = attention.sparse_params
        max_kv_len = int(attn_metadata.kv_lens_runtime[: attn_metadata.num_seqs].max())
        # The index cache is updated even below this cutoff so a request can
        # switch to sparse attention later without rebuilding its prefix.
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
        if kv_pool is None or kv_pool.ndim != 5 or kv_pool.shape[1] != QSA_MAIN_KV_ROLES:
            raise RuntimeError("QSA sparse attention requires adjacent paged HND K/V buffers")
        k_cache = kv_pool[:, QSA_KEY_ROLE_INDEX]
        v_cache = kv_pool[:, QSA_VALUE_ROLE_INDEX]
        req_idx = attn_metadata.qsa_req_idx_per_token[:num_tokens]
        logical = attn_metadata.qsa_logical_positions[:num_tokens]
        tokens_per_block = attn_metadata.kv_cache_manager.tokens_per_block
        # The direct row-wise kernel removes launch overhead for small decode
        # batches. Larger batches and prefill chunks have enough work for
        # PyTorch's vectorized store path.
        fused_kv_store = (
            attn_metadata.num_contexts == 0
            and k.is_cuda
            and v.is_cuda
            and k_cache.is_cuda
            and v_cache.is_cuda
        )
        if fused_kv_store:
            from .kernels import triton_qsa_paged_kv_store

            logger.info_once(
                "QSA fused decode paged K/V store Triton kernel is active",
                key="qsa_fused_decode_paged_kv_store_active",
            )
            triton_qsa_paged_kv_store(
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                request_indices=req_idx,
                logical_positions=logical,
                block_table=attn_metadata.qsa_block_table,
                tokens_per_block=tokens_per_block,
            )
        else:
            _store_paged_kv_reference(
                k,
                v,
                k_cache,
                v_cache,
                req_idx,
                logical,
                attn_metadata.qsa_block_table,
                tokens_per_block,
            )

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
            sequence_lengths = attn_metadata.qsa_sequence_lengths[:num_tokens]
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
                visible_blocks=attn_metadata.qsa_visible_blocks[:num_tokens],
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
        sequence_lengths = attn_metadata.qsa_sequence_lengths[:num_tokens]
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
                visible_blocks=attn_metadata.qsa_visible_blocks[packed_slice],
                context_rows=True,
            )
            output[packed_slice] = qsa_sparse_gqa(
                q=q[packed_slice],
                k_cache=k_cache,
                v_cache=v_cache,
                selected_tokens=selected,
                request_indices=req_idx[packed_slice],
                metadata=attn_metadata,
                softmax_scale=1.0 / (attention.q_scaling * attention.head_dim**0.5),
                query_positions=logical[packed_slice],
                compress_ratio=params.compress_ratio,
            )
        return output.reshape(num_tokens, -1)


register_attention_sparse_hooks("qsa", QSASparseHooks)


__all__ = ["QSASparseHooks"]
