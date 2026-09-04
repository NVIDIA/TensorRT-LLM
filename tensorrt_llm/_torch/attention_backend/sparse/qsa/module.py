# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module-layer integration for QSA sparse attention."""

import math
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.modules.top_k import TopK
from tensorrt_llm.logger import logger

from ...interface import AttentionMask, PredefinedAttentionMask
from ..hooks import AttentionSparseHooks, register_attention_sparse_hooks
from .constants import (
    QSA_KEY_ROLE_INDEX,
    QSA_MAIN_KV_ROLES,
    QSA_SPARSE_KV_CACHE_DTYPES,
    QSA_VALUE_ROLE_INDEX,
)
from .indexer import _is_power_of_two, _logical_to_pages
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


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand selected complete groups and append the incomplete causal tail.

    CUDA callers pass Top-K output, whose valid block IDs form a contiguous
    prefix followed by ``-1`` padding. The fused expansion relies on that
    layout to append the causal tail without a separate compaction launch.
    """
    block_topk = token_topk // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"Expected block indices [rows, {block_topk}], got {tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("QSA query positions and sequence lengths must match rows")

    if block_indices.is_cuda:
        from .kernels import triton_expand_qsa_block_indices

        return triton_expand_qsa_block_indices(
            block_indices.contiguous(),
            query_positions.to(device=block_indices.device).contiguous(),
            sequence_lengths.to(device=block_indices.device).contiguous(),
            compress_ratio=compress_ratio,
            token_topk=token_topk,
        )

    device = block_indices.device
    blocks = block_indices.to(torch.long)
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0,
        expanded,
        torch.full_like(expanded, -1),
    ).reshape(rows, token_topk)

    query_positions = query_positions.to(device=device, dtype=torch.long)
    sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(
        compress_ratio - 1,
        device=device,
        dtype=torch.long,
    )
    visible_tokens = query_positions + 1
    tail_start = (visible_tokens // compress_ratio) * compress_ratio
    tail_count = visible_tokens - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(final_topk, device=device).unsqueeze(0).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + final_topk)
    return result.gather(
        1,
        torch.argsort(sort_key, dim=1, stable=True),
    ).to(torch.int32)


def qsa_sparse_gqa(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_idx: int | None = None,
    request_indices: torch.Tensor | None = None,
    metadata: "QSAAttentionMetadata",
    softmax_scale: float,
    query_positions: torch.Tensor | None = None,
    compress_ratio: int | None = None,
) -> torch.Tensor:
    """Run sparse GQA over V2 paged K/V, using Triton on CUDA by default."""
    if request_indices is None:
        if request_idx is None:
            raise ValueError("QSA sparse GQA requires request indices")
        request_indices = torch.full(
            (q.shape[0],),
            request_idx,
            dtype=torch.int32,
            device=q.device,
        )
    if request_indices.shape != (q.shape[0],):
        raise ValueError("QSA sparse GQA request indices must match query rows")
    if q.is_cuda and _is_power_of_two(q.shape[-1]):
        from .kernels import triton_qsa_paged_sparse_gqa

        logger.info_once(
            "QSA fused paged sparse GQA Triton kernel is active",
            key="qsa_fused_paged_sparse_gqa_active",
        )
        return triton_qsa_paged_sparse_gqa(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=metadata.qsa_block_table,
            selected_tokens=selected_tokens,
            request_indices=request_indices.contiguous(),
            tokens_per_block=metadata.kv_cache_manager.tokens_per_block,
            softmax_scale=softmax_scale,
            query_positions=query_positions,
            compress_ratio=compress_ratio,
        )
    logger.info_once(
        "QSA sparse GQA reference path is active",
        key="qsa_sparse_gqa_reference_active",
    )
    return qsa_sparse_gqa_reference(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        selected_tokens=selected_tokens,
        request_indices=request_indices,
        metadata=metadata,
        softmax_scale=softmax_scale,
    )


def qsa_sparse_gqa_reference(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    metadata: "QSAAttentionMetadata",
    softmax_scale: float,
) -> torch.Tensor:
    """Torch reference sparse GQA over V2 paged K/V."""
    valid = selected_tokens >= 0
    safe_tokens = selected_tokens.clamp_min(0).to(torch.long)
    req = request_indices[:, None].expand_as(safe_tokens).to(torch.int32)
    pages, within = _logical_to_pages(metadata, req, safe_tokens)
    keys = k_cache[pages, :, within, :]
    values = v_cache[pages, :, within, :]

    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = keys.shape[2]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("QSA query heads must be divisible by local KV heads")
    groups = num_q_heads // num_kv_heads
    q_grouped = q.reshape(rows, num_kv_heads, groups, head_dim)
    scores = (
        torch.einsum(
            "bhgd,bkhd->bhgk",
            q_grouped.float(),
            keys.float(),
        )
        * softmax_scale
    )
    scores.masked_fill_(~valid[:, None, None, :], -float("inf"))
    # A padded request can have no visible token. Softmax over all -inf is NaN;
    # such a row contributes zero attention output.
    probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    output = torch.einsum(
        "bhgk,bkhd->bhgd",
        probabilities,
        values.float(),
    )
    return output.to(q.dtype).reshape(rows, num_q_heads, head_dim)


def select_qsa_tokens(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_length: int,
    params: QSASparseParams,
    *,
    top_k: TopK | None = None,
    top_k_output: torch.Tensor | None = None,
    top_k_row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score complete groups and return fixed-width logical token indices."""
    rows = q.shape[0]
    total_blocks = compressed_keys.shape[0]
    block_indices = torch.full(
        (rows, params.block_topk),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    if total_blocks:
        scores = torch.einsum(
            "mhd,nd->mnh",
            q.float(),
            compressed_keys.float(),
        )
        scores = torch.relu(scores).sum(dim=-1) / math.sqrt(params.index_head_dim)
        visible_blocks = ((query_positions + 1) // params.compress_ratio).to(torch.long)
        if top_k is not None and scores.is_cuda:
            if top_k_output is None or top_k_row_starts is None:
                raise ValueError("QSA CUDA Top-K requires caller-owned output and row starts")
            block_indices = top_k_output[:rows]
            top_k(
                scores,
                block_indices,
                is_prefill=True,
                row_starts=top_k_row_starts[:rows],
                row_ends=visible_blocks.to(torch.int32),
            )
        else:
            columns = torch.arange(total_blocks, device=q.device).unsqueeze(0)
            scores.masked_fill_(columns >= visible_blocks.unsqueeze(1), -float("inf"))
            width = min(params.block_topk, total_blocks)
            values, indices = torch.topk(scores, width, dim=-1)
            indices = torch.where(
                torch.isfinite(values),
                indices,
                torch.full_like(indices, -1),
            )
            block_indices[:, :width] = indices.to(torch.int32)
    sequence_lengths = torch.full_like(query_positions, sequence_length)
    return expand_qsa_block_indices(
        block_indices,
        query_positions,
        sequence_lengths,
        compress_ratio=params.compress_ratio,
        token_topk=params.token_topk,
    )


def select_qsa_paged_tokens(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    request_indices: torch.Tensor,
    metadata: "QSAAttentionMetadata",
    params: QSASparseParams,
    *,
    top_k: TopK | None = None,
    top_k_output: torch.Tensor | None = None,
    top_k_row_starts: torch.Tensor | None = None,
    visible_blocks: torch.Tensor | None = None,
    context_rows: bool = False,
) -> torch.Tensor:
    """Select tokens with packed, fixed-width paged scoring.

    ``context_rows`` marks the caller that packs many consecutive query rows
    per request; scoring can then share one gather of compressed keys across a
    tile of rows.
    """
    from .kernels import triton_qsa_paged_index_scores

    logits = triton_qsa_paged_index_scores(
        q=q,
        index_cache=index_cache,
        block_table=metadata.qsa_block_table,
        query_positions=query_positions,
        request_indices=request_indices,
        tokens_per_block=metadata.kv_cache_manager.tokens_per_block,
        compress_ratio=params.compress_ratio,
        # CUDA radix Top-K scans only [row_starts, row_ends), so score tiles
        # outside the causal boundary need not be materialized. Preserve fully
        # initialized logits for the Torch fallback below.
        only_visible_blocks=top_k is not None and q.is_cuda,
        context_rows=context_rows,
    )
    if top_k is not None and logits.is_cuda:
        if top_k_output is None or top_k_row_starts is None:
            raise ValueError("QSA CUDA Top-K requires caller-owned output and row starts")
        indices = top_k_output[: q.shape[0]]
        if visible_blocks is None:
            visible_blocks = ((query_positions + 1) // params.compress_ratio).to(torch.int32)
        # QSA always has explicit per-row compressed-block bounds, including
        # generation and speculative rows. Use TopK's row-range API rather
        # than its request-grouped decode API; this is a layout choice, not a
        # declaration that the request is in prefill.
        top_k(
            logits,
            indices,
            is_prefill=True,
            row_starts=top_k_row_starts[: q.shape[0]],
            row_ends=visible_blocks,
        )
    else:
        # `triton_qsa_paged_index_scores` applies each row's causal block bound
        # in both `only_visible_blocks` modes; the flag only decides whether
        # out-of-range columns are left unspecified or written as -inf. So the
        # scores reaching this Torch fallback are already causally masked, and
        # the -inf entries below are what mark the padding slots.
        width = min(params.block_topk, logits.shape[1])
        values, indices = torch.topk(logits, width, dim=-1)
        indices = torch.where(
            torch.isfinite(values),
            indices,
            torch.full_like(indices, -1),
        ).to(torch.int32)
        if width < params.block_topk:
            indices = torch.nn.functional.pad(
                indices,
                (0, params.block_topk - width),
                value=-1,
            )
    return expand_qsa_block_indices(
        indices,
        query_positions,
        sequence_lengths,
        compress_ratio=params.compress_ratio,
        token_topk=params.token_topk,
    )


class QSASparseHooks(AttentionSparseHooks):
    """Keep the QSA side cache current and replace dense attention when useful."""

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


__all__ = [
    "QSASparseHooks",
    "expand_qsa_block_indices",
    "qsa_sparse_gqa",
    "qsa_sparse_gqa_reference",
    "select_qsa_paged_tokens",
    "select_qsa_tokens",
]
