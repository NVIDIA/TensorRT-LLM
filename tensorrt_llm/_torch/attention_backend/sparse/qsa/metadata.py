# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention metadata for QSA sparse attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

from .cache_manager import QSAMambaHybridCacheManagerV2
from .kernels import triton_qsa_unscale_block_table
from .params import QSASparseMetadataParams


@dataclass(init=False)
class QSAAttentionMetadata(TrtllmAttentionMetadata):
    """Graph-stable request mapping used by the QSA index and GQA paths."""

    sparse_metadata_params: Optional[QSASparseMetadataParams] = None

    def __init__(self, *args, **kwargs) -> None:
        sparse_attention_config = kwargs.pop("sparse_attention_config", None)
        if kwargs.get("sparse_metadata_params") is None and sparse_attention_config is not None:
            kwargs["sparse_metadata_params"] = sparse_attention_config.to_sparse_metadata_params()
        super().__init__(*args, **kwargs)
        if not isinstance(self.sparse_metadata_params, QSASparseMetadataParams):
            raise ValueError("QSA sparse metadata parameters are not set")

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.kv_cache_manager, QSAMambaHybridCacheManagerV2):
            raise TypeError("QSA sparse attention requires QSAMambaHybridCacheManagerV2")
        capture_graph = self.is_cuda_graph
        buffers = self.cuda_graph_buffers
        self.qsa_block_table = self.get_empty(
            buffers,
            (
                self.max_num_sequences,
                self.kv_cache_manager.max_blocks_per_seq,
            ),
            cache_name="qsa_block_table",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self._qsa_attention_pool_id, self._qsa_page_index_scale = (
            self.kv_cache_manager.get_qsa_attention_pool_layout()
        )
        self.qsa_req_idx_per_token = self.get_empty(
            buffers,
            (self.max_num_tokens,),
            cache_name="qsa_req_idx_per_token",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.qsa_logical_positions = self.get_empty(
            buffers,
            (self.max_num_tokens,),
            cache_name="qsa_logical_positions",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self._qsa_token_arange = self.get_empty(
            buffers,
            (self.max_num_tokens,),
            cache_name="qsa_token_arange",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self._qsa_token_arange.copy_(
            torch.arange(self.max_num_tokens, device="cuda", dtype=torch.int64)
        )
        self._qsa_cu_seq_lens = self.get_empty(
            buffers,
            (self.max_num_sequences + 1,),
            cache_name="qsa_cu_seq_lens",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self._qsa_cu_seq_lens.zero_()
        self.qsa_topk_indices = self.get_empty(
            buffers,
            (
                self.max_num_tokens,
                self.sparse_metadata_params.token_topk
                // self.sparse_metadata_params.compress_ratio,
            ),
            cache_name="qsa_topk_indices",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.qsa_topk_row_starts = self.get_empty(
            buffers,
            (self.max_num_tokens,),
            cache_name="qsa_topk_row_starts",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.qsa_topk_row_starts.zero_()

    def _refresh_qsa_token_mapping(self) -> None:
        num_seqs = self.num_seqs
        num_tokens = self.num_tokens
        if num_seqs <= 0 or num_tokens <= 0:
            return
        seq_lens = self.seq_lens_cuda[:num_seqs]
        if self.num_contexts == 0 and num_tokens == num_seqs:
            from .kernels import triton_qsa_decode_token_mapping

            triton_qsa_decode_token_mapping(
                kv_lens=self.kv_lens_cuda_runtime[:num_seqs],
                seq_lens=seq_lens,
                request_indices=self.qsa_req_idx_per_token[:num_tokens],
                logical_positions=self.qsa_logical_positions[:num_tokens],
            )
            return
        torch.cumsum(
            seq_lens,
            dim=0,
            dtype=torch.int64,
            out=self._qsa_cu_seq_lens[1 : num_seqs + 1],
        )
        token_ids = self._qsa_token_arange[:num_tokens]
        req_idx = torch.searchsorted(
            self._qsa_cu_seq_lens[1 : num_seqs + 1],
            token_ids,
            right=True,
        )
        self.qsa_req_idx_per_token[:num_tokens].copy_(req_idx.to(torch.int32))
        seq_starts = self._qsa_cu_seq_lens[:num_seqs]
        cached_lens = self.kv_lens_cuda_runtime[:num_seqs].to(torch.int64) - seq_lens.to(
            torch.int64
        )
        logical = cached_lens[req_idx] + token_ids - seq_starts[req_idx]
        self.qsa_logical_positions[:num_tokens].copy_(logical)

    def _refresh_qsa_block_table(self) -> None:
        num_seqs = self.num_seqs
        if num_seqs <= 0:
            return
        scaled = self.kv_cache_block_offsets[
            self._qsa_attention_pool_id,
            :num_seqs,
            0,
            :,
        ]
        triton_qsa_unscale_block_table(
            scaled_block_table=scaled,
            block_table=self.qsa_block_table[:num_seqs],
            page_index_scale=self._qsa_page_index_scale,
        )

    def prepare(self) -> None:
        super().prepare()
        self._refresh_qsa_block_table()
        self._refresh_qsa_token_mapping()

    def on_update_kv_lens(self) -> None:
        super().on_update_kv_lens()
        self._refresh_qsa_token_mapping()


__all__ = ["QSAAttentionMetadata"]
