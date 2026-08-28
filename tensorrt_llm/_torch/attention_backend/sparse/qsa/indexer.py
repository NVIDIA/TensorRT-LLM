# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 reference indexer and sparse GQA transforms for QSA."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
from torch import nn

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rotary_embedding import MRotaryEmbedding, RotaryEmbedding
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.logger import logger

from .params import QSASparseParams

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.attention import Attention

    from .metadata import QSAAttentionMetadata


def average_pool_qsa_keys(key_groups: torch.Tensor) -> torch.Tensor:
    """Average complete raw-key groups in FP32 and cast back to input dtype."""
    if key_groups.ndim != 4:
        raise ValueError(
            "QSA key groups must be [groups, ratio, kv_heads, head_dim], "
            f"got {tuple(key_groups.shape)}"
        )
    return key_groups.float().mean(dim=1).to(key_groups.dtype)


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand selected complete groups and append the incomplete causal tail."""
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


def _position_coordinates(position_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if position_ids is None:
        raise ValueError("QSA sparse attention requires position_ids")
    if position_ids.dim() == 3:
        positions = position_ids.reshape(3, -1)[:, :num_tokens].transpose(0, 1)
    else:
        positions = position_ids.reshape(-1)[:num_tokens, None].expand(-1, 3)
    if positions.shape[0] != num_tokens:
        raise ValueError(
            f"QSA position count {positions.shape[0]} does not match {num_tokens} tokens"
        )
    # The fused pre-indexer accepts arbitrary row/axis strides and converts
    # positions while storing its int32 side cache. Preserve the scheduler's
    # tensor view here instead of materializing the same coordinates once per
    # sparse layer.
    return positions


def _logical_to_pages(
    metadata: "QSAAttentionMetadata",
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens_per_block = metadata.kv_cache_manager.tokens_per_block
    logical_positions = logical_positions.to(torch.long)
    page_columns = logical_positions // tokens_per_block
    pages = metadata.qsa_block_table[
        request_indices.to(torch.long),
        page_columns,
    ].to(torch.long)
    within = logical_positions % tokens_per_block
    return pages, within


class QSAIndexer(nn.Module):
    """Checkpoint-defined replicated Q/K projection and compressed side cache."""

    def __init__(self, attention: "Attention", params: QSASparseParams) -> None:
        super().__init__()
        config = attention.pretrained_config
        self.params = params
        self.index_qk_proj = Linear(
            attention.hidden_size,
            (params.index_n_heads + params.index_kv_heads) * params.index_head_dim,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=None,
            skip_create_weights_in_init=False,
            use_custom_cublas_mm=True,
        )
        self.q_layernorm = RMSNorm(
            hidden_size=params.index_head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        self.k_layernorm = RMSNorm(
            hidden_size=params.index_head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        pos = attention.pos_embd_params
        if pos is None:
            raise ValueError("QSA sparse attention requires rotary parameters")
        if pos.mrope_section is not None:
            self.rotary_emb = MRotaryEmbedding(
                pos.rope,
                head_dim=params.index_head_dim,
                mrope_section=pos.mrope_section,
                is_neox=pos.is_neox,
                mrope_interleaved=pos.mrope_interleaved,
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                pos.rope,
                head_dim=params.index_head_dim,
                is_neox=pos.is_neox,
            )
        self._pending_speculative_cache = None
        use_cute_dsl_prefill_topk = (
            os.environ.get("TRTLLM_QSA_CUTE_DSL_PREFILL_TOPK", "1") != "0"
            and IS_CUTLASS_DSL_AVAILABLE
            and is_sm_100f()
        )
        self.top_k = TopK(
            params.block_topk,
            prefill_implementation=(
                TopKImplementation.CUTE_DSL_RADIX
                if use_cute_dsl_prefill_topk
                else TopKImplementation.CUDA_RADIX
            ),
            decode_implementation=TopKImplementation.CUDA_RADIX,
            compress_ratio=params.compress_ratio,
        )

    def _apply_rope(
        self,
        tensor: torch.Tensor,
        position_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        flat = tensor.reshape(tensor.shape[0], -1)
        if isinstance(self.rotary_emb, MRotaryEmbedding):
            positions = position_coordinates.transpose(0, 1).reshape(
                3,
                1,
                tensor.shape[0],
            )
        else:
            positions = position_coordinates[:, 0]
        return self.rotary_emb(positions, [flat])[0].reshape_as(tensor)

    def project(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        qk = self.index_qk_proj(hidden_states)
        q_width = self.params.index_n_heads * self.params.index_head_dim
        q_raw, k_raw = qk.split(
            (q_width, self.params.index_kv_heads * self.params.index_head_dim),
            dim=-1,
        )
        q = self.q_layernorm(q_raw.reshape(-1, self.params.index_head_dim)).reshape(
            num_tokens,
            self.params.index_n_heads,
            self.params.index_head_dim,
        )
        coordinates = _position_coordinates(position_ids, num_tokens)
        q = self._apply_rope(q, coordinates)
        return (
            q,
            k_raw.reshape(
                num_tokens,
                self.params.index_kv_heads,
                self.params.index_head_dim,
            ),
            coordinates,
        )

    def project_and_update_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_idx: int,
        metadata: "QSAAttentionMetadata",
    ) -> torch.Tensor:
        """Project the index Q/K and update its paged side cache.

        Ordinary CUDA-graph decode uses a single Triton launch for Q
        normalization/RoPE plus raw/compressed K cache updates. Prefill,
        speculative verification, CPU execution, and unsupported RoPE layouts
        retain the reference path below.
        """
        num_tokens = hidden_states.shape[0]
        qk = self.index_qk_proj(hidden_states)
        q_width = self.params.index_n_heads * self.params.index_head_dim
        q_raw, k_raw = qk.split(
            (q_width, self.params.index_kv_heads * self.params.index_head_dim),
            dim=-1,
        )
        q_raw = q_raw.reshape(
            num_tokens,
            self.params.index_n_heads,
            self.params.index_head_dim,
        )
        token_k = k_raw.reshape(
            num_tokens,
            self.params.index_kv_heads,
            self.params.index_head_dim,
        )
        coordinates = _position_coordinates(position_ids, num_tokens)

        index_cache = metadata.kv_cache_manager.get_index_k_buffer(layer_idx)
        position_cache = metadata.kv_cache_manager.get_qsa_position_buffer()
        if index_cache is None or position_cache is None:
            raise RuntimeError("QSA sparse side-cache buffers are unavailable")
        rotary_cache = self.rotary_emb.rotary_cos_sin
        mrope_section = None
        supported_mrope = True
        if isinstance(self.rotary_emb, MRotaryEmbedding):
            supported_mrope = self.rotary_emb.mrope_interleaved
            if supported_mrope:
                mrope_section = tuple(self.rotary_emb.mrope_section)
        fused_decode = (
            os.environ.get("TRTLLM_QSA_FUSED_DECODE_PRE_INDEXER", "1") != "0"
            and metadata.is_cuda_graph
            and metadata.num_contexts == 0
            and num_tokens == metadata.num_seqs
            and q_raw.is_cuda
            and q_raw.dtype == torch.bfloat16
            and token_k.dtype == torch.bfloat16
            and index_cache.dtype == torch.bfloat16
            and position_cache.is_cuda
            and supported_mrope
            and rotary_cache.ndim == 3
            and rotary_cache.shape[1] == 2
            and rotary_cache.shape[2] * 4 == self.params.index_head_dim
            and self.params.index_head_dim > 0
            and (self.params.index_head_dim & (self.params.index_head_dim - 1)) == 0
            and (self.params.compress_ratio & (self.params.compress_ratio - 1)) == 0
        )
        if fused_decode:
            from .kernels import triton_qsa_decode_pre_indexer

            logger.info_once(
                "QSA fused decode pre-indexer Triton kernel is active",
                key="qsa_fused_decode_pre_indexer_active",
            )
            return triton_qsa_decode_pre_indexer(
                q=q_raw,
                token_k=token_k,
                position_coordinates=coordinates,
                request_indices=metadata.qsa_req_idx_per_token[:num_tokens],
                logical_positions=metadata.qsa_logical_positions[:num_tokens],
                block_table=metadata.qsa_block_table,
                index_cache=index_cache,
                position_cache=position_cache,
                q_norm_weight=self.q_layernorm.weight,
                k_norm_weight=self.k_layernorm.weight,
                cos_sin=rotary_cache.view(rotary_cache.shape[0], -1),
                eps=self.q_layernorm.variance_epsilon,
                tokens_per_block=metadata.kv_cache_manager.tokens_per_block,
                compress_ratio=self.params.compress_ratio,
                mrope_section=mrope_section,
            )

        q = self.q_layernorm(q_raw.reshape(-1, self.params.index_head_dim)).reshape_as(q_raw)
        q = self._apply_rope(q, coordinates)
        self.update_cache_and_compress(
            layer_idx,
            token_k,
            coordinates,
            metadata,
        )
        return q

    def update_cache_and_compress(
        self,
        layer_idx: int,
        token_k: torch.Tensor,
        position_coordinates: torch.Tensor,
        metadata: "QSAAttentionMetadata",
    ) -> None:
        num_tokens = token_k.shape[0]
        req_idx = metadata.qsa_req_idx_per_token[:num_tokens]
        logical = metadata.qsa_logical_positions[:num_tokens]
        pages, within = _logical_to_pages(metadata, req_idx, logical)
        index_cache = metadata.kv_cache_manager.get_index_k_buffer(layer_idx)
        position_cache = metadata.kv_cache_manager.get_qsa_position_buffer()
        if index_cache is None or position_cache is None:
            raise RuntimeError("QSA sparse side-cache buffers are unavailable")
        self._capture_speculative_cache_state(
            layer_idx,
            req_idx,
            pages,
            within,
            index_cache,
            position_cache,
            metadata,
        )
        index_cache[pages, within, 0] = token_k[:, 0].to(index_cache.dtype)
        position_cache[pages, within] = position_coordinates.to(position_cache.dtype)

        rotary_cache = self.rotary_emb.rotary_cos_sin
        mrope_section = None
        supported_mrope = True
        if isinstance(self.rotary_emb, MRotaryEmbedding):
            supported_mrope = self.rotary_emb.mrope_interleaved
            if supported_mrope:
                mrope_section = tuple(self.rotary_emb.mrope_section)
        fused_prefill_compress = (
            os.environ.get("TRTLLM_QSA_FUSED_PREFILL_COMPRESS", "1") != "0"
            and token_k.is_cuda
            and token_k.dtype == torch.bfloat16
            and token_k.shape[1] == 1
            and index_cache.dtype == torch.bfloat16
            and position_cache.is_cuda
            and supported_mrope
            and rotary_cache.ndim == 3
            and rotary_cache.shape[1] == 2
            and rotary_cache.shape[2] * 4 == self.params.index_head_dim
            and self.params.index_head_dim > 0
            and (self.params.index_head_dim & (self.params.index_head_dim - 1)) == 0
            and (self.params.compress_ratio & (self.params.compress_ratio - 1)) == 0
        )
        if fused_prefill_compress:
            from .kernels import triton_qsa_prefill_compress

            logger.info_once(
                "QSA fused prefill compression Triton kernel is active",
                key="qsa_fused_prefill_compress_active",
            )
            triton_qsa_prefill_compress(
                logical_positions=logical,
                request_indices=req_idx,
                block_table=metadata.qsa_block_table,
                index_cache=index_cache,
                position_cache=position_cache,
                k_norm_weight=self.k_layernorm.weight,
                cos_sin=rotary_cache.view(rotary_cache.shape[0], -1),
                eps=self.k_layernorm.variance_epsilon,
                tokens_per_block=metadata.kv_cache_manager.tokens_per_block,
                compress_ratio=self.params.compress_ratio,
                mrope_section=mrope_section,
            )
            return

        boundaries = ((logical + 1) % self.params.compress_ratio) == 0
        if metadata.is_cuda_graph and metadata.num_contexts == 0:
            group_offsets = torch.arange(
                self.params.compress_ratio - 1,
                -1,
                -1,
                device=logical.device,
                dtype=torch.long,
            )
            group_positions = (logical[:, None] - group_offsets[None, :]).clamp_min(0)
            group_req = req_idx[:, None].expand_as(group_positions)
            group_pages, group_within = _logical_to_pages(
                metadata,
                group_req,
                group_positions,
            )
            raw_groups = index_cache[group_pages, group_within]
            compressed = average_pool_qsa_keys(raw_groups)
            compressed = self.k_layernorm(
                compressed.reshape(-1, self.params.index_head_dim)
            ).reshape(
                -1,
                self.params.index_kv_heads,
                self.params.index_head_dim,
            )
            compressed_coordinates = position_cache[
                group_pages[:, 0],
                group_within[:, 0],
            ]
            compressed = self._apply_rope(compressed, compressed_coordinates)
            stored = torch.where(
                boundaries[:, None, None],
                compressed,
                token_k,
            )
            index_cache[pages, within, 0] = stored[:, 0].to(index_cache.dtype)
            return
        if not torch.any(boundaries):
            return
        boundary_req = req_idx[boundaries]
        boundary_pos = logical[boundaries]
        group_offsets = torch.arange(
            self.params.compress_ratio - 1,
            -1,
            -1,
            device=logical.device,
            dtype=torch.long,
        )
        group_positions = boundary_pos[:, None] - group_offsets[None, :]
        group_req = boundary_req[:, None].expand_as(group_positions)
        group_pages, group_within = _logical_to_pages(
            metadata,
            group_req,
            group_positions,
        )
        raw_groups = index_cache[group_pages, group_within]
        compressed = average_pool_qsa_keys(raw_groups)
        compressed = self.k_layernorm(compressed.reshape(-1, self.params.index_head_dim)).reshape(
            -1,
            self.params.index_kv_heads,
            self.params.index_head_dim,
        )
        group_positions_coordinates = position_cache[
            group_pages[:, 0],
            group_within[:, 0],
        ]
        compressed = self._apply_rope(compressed, group_positions_coordinates)
        anchor_pages = group_pages[:, -1]
        anchor_within = group_within[:, -1]
        index_cache[anchor_pages, anchor_within, 0] = compressed[:, 0].to(index_cache.dtype)

    def _capture_speculative_cache_state(
        self,
        layer_idx: int,
        request_indices: torch.Tensor,
        pages: torch.Tensor,
        within: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        metadata: "QSAAttentionMetadata",
    ) -> None:
        """Snapshot QSA side-cache entries that target verification may reject."""
        self._pending_speculative_cache = None
        num_contexts = metadata.num_contexts
        num_seqs = metadata.num_seqs
        if metadata.is_cuda_graph and num_contexts == 0 and metadata.num_tokens == num_seqs:
            return
        if num_contexts >= num_seqs:
            return
        # Mixed-IFB forwards are eager. Derive this branch once from the host
        # query lengths rather than synchronizing a device reduction per layer.
        if not metadata.is_cuda_graph and not metadata.qsa_needs_speculative_snapshot:
            return
        seq_lens = metadata.seq_lens_cuda[:num_seqs].to(torch.long)
        generation_lens = seq_lens[num_contexts:]
        if generation_lens.numel() == 0:
            return

        token_ids = torch.arange(
            request_indices.numel(),
            device=request_indices.device,
            dtype=torch.long,
        )
        seq_starts = torch.cumsum(seq_lens, dim=0) - seq_lens
        token_ordinals = token_ids - seq_starts[request_indices.to(torch.long)]
        if num_contexts:
            generation_mask = request_indices >= num_contexts
            request_indices = request_indices[generation_mask].to(torch.long)
            token_ordinals = token_ordinals[generation_mask]
            pages = pages[generation_mask]
            within = within[generation_mask]
            if request_indices.numel() == 0:
                return
        else:
            # Boolean indexing would create a data-dependent shape during a
            # CUDA-graph target verification batch.
            request_indices = request_indices.to(torch.long)

        pending = {
            "request_indices": request_indices,
            "token_ordinals": token_ordinals,
            "pages": pages,
            "within": within,
            "index_values": index_cache[pages, within, 0].clone(),
            "index_cache": index_cache,
        }
        if layer_idx == metadata.kv_cache_manager.qsa_position_layer_id:
            pending["position_values"] = position_cache[pages, within].clone()
            pending["position_cache"] = position_cache
        self._pending_speculative_cache = pending

    def commit_speculative_states(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        num_contexts: int,
    ) -> None:
        """Restore QSA side-cache entries written for rejected verify tokens."""
        del state_indices, num_contexts
        pending = self._pending_speculative_cache
        self._pending_speculative_cache = None
        if pending is None:
            return

        request_indices = pending["request_indices"]
        accepted = num_accepted_tokens.to(request_indices.device)[request_indices]
        restore = pending["token_ordinals"] >= accepted
        pages = pending["pages"]
        within = pending["within"]
        index_cache = pending["index_cache"]
        current_index = index_cache[pages, within, 0]
        index_cache[pages, within, 0] = torch.where(
            restore[:, None], pending["index_values"], current_index
        )
        position_values = pending.get("position_values")
        if position_values is not None:
            position_cache = pending["position_cache"]
            current_positions = position_cache[pages, within]
            position_cache[pages, within] = torch.where(
                restore[:, None], position_values, current_positions
            )

    def gather_compressed_keys(
        self,
        layer_idx: int,
        request_idx: int,
        complete_blocks: int,
        metadata: "QSAAttentionMetadata",
    ) -> torch.Tensor:
        cache = metadata.kv_cache_manager.get_index_k_buffer(layer_idx)
        if cache is None:
            raise RuntimeError(f"QSA index cache is unavailable for layer {layer_idx}")
        if complete_blocks == 0:
            return cache.new_empty((0, self.params.index_head_dim))
        anchors = (
            torch.arange(complete_blocks, device=cache.device, dtype=torch.long)
            * self.params.compress_ratio
            + self.params.compress_ratio
            - 1
        )
        req = torch.full_like(anchors, request_idx, dtype=torch.int32)
        pages, within = _logical_to_pages(metadata, req, anchors)
        return cache[pages, within, 0]


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
    if q.is_cuda and os.environ.get("TRTLLM_QSA_SPARSE_FUSED", "1") != "0":
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
    probabilities = torch.softmax(scores, dim=-1)
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
) -> torch.Tensor:
    """Select tokens with packed, fixed-width paged scoring."""
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
    )
    if top_k is not None and logits.is_cuda:
        if top_k_output is None or top_k_row_starts is None:
            raise ValueError("QSA CUDA Top-K requires caller-owned output and row starts")
        indices = top_k_output[: q.shape[0]]
        if visible_blocks is None:
            visible_blocks = ((query_positions + 1) // params.compress_ratio).to(torch.int32)
        top_k(
            logits,
            indices,
            is_prefill=True,
            row_starts=top_k_row_starts[: q.shape[0]],
            row_ends=visible_blocks,
        )
    else:
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


__all__ = [
    "QSAIndexer",
    "average_pool_qsa_keys",
    "qsa_sparse_gqa",
    "qsa_sparse_gqa_reference",
    "expand_qsa_block_indices",
    "select_qsa_paged_tokens",
    "select_qsa_tokens",
]
