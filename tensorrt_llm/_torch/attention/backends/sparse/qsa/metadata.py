# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention metadata for QSA sparse attention."""

from dataclasses import dataclass, field
from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

from .cache_manager import QSAMambaHybridCacheManagerV2
from .constants import QSA_KEY_ROLE_INDEX
from .kernels import triton_qsa_unscale_block_table
from .params import QSASparseMetadataParams


# Keep inherited dataclass fields while intercepting one QSA-only constructor arg.
@dataclass(init=False)
class QSAAttentionMetadata(TrtllmAttentionMetadata):
    """Extend TRTLLM metadata with graph-stable QSA page and row mappings."""

    sparse_metadata_params: Optional[QSASparseMetadataParams] = None
    qsa_needs_speculative_snapshot: bool = field(
        init=False,
        default=False,
        repr=False,
    )
    qsa_has_local_layers: bool = field(
        init=False,
        default=False,
        repr=False,
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        # Forwards the parent dataclass's full field set, so the signature
        # stays open rather than restating it.
        """Allocate the fixed-shape QSA buffers updated before each graph replay."""
        sparse_attention_config = kwargs.pop("sparse_attention_config", None)
        if kwargs.get("sparse_metadata_params") is None and sparse_attention_config is not None:
            kwargs["sparse_metadata_params"] = sparse_attention_config.to_sparse_metadata_params()
        # The generated parent initializer dispatches to this class's
        # ``__post_init__``; keep QSA validation there so both constructor forms
        # (lowered params or the legacy user config) enforce the same contract.
        super().__init__(*args, **kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.kv_cache_manager, QSAMambaHybridCacheManagerV2):
            raise TypeError("QSA sparse attention requires QSAMambaHybridCacheManagerV2")
        if not isinstance(self.sparse_metadata_params, QSASparseMetadataParams):
            raise TypeError("QSA sparse attention requires QSASparseMetadataParams")
        self.qsa_has_local_layers = self.kv_cache_manager.qsa_position_layer_id is not None
        if not self.qsa_has_local_layers:
            # A PP rank may own only recurrent layers. It still uses the hybrid
            # manager and parent metadata, but needs no QSA page views.
            return
        capture_graph = self.is_cuda_graph
        buffers = self.cuda_graph_buffers

        # The sparse kernels index raw V2 pool pages, while the parent metadata
        # stores attention-operator encoded offsets. Keep an unscaled table.
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

        # Packed-row mapping shared by compression, selection, and sparse GQA.
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
        self.qsa_sequence_lengths = self.get_empty(
            buffers,
            (self.max_num_tokens,),
            cache_name="qsa_sequence_lengths",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.qsa_visible_blocks = self.get_empty(
            buffers,
            (self.max_num_tokens,),
            cache_name="qsa_visible_blocks",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

        # Reuse graph-stable row IDs instead of allocating arange each step.
        self._qsa_token_arange = self.get_empty(
            buffers,
            (self.max_num_tokens,),
            cache_name="qsa_token_arange",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self._qsa_token_arange.copy_(
            torch.arange(
                self.max_num_tokens,
                device=self._qsa_token_arange.device,
                dtype=torch.int64,
            )
        )
        # The leading zero turns request lengths into packed-row boundaries.
        self._qsa_cu_seq_lens = self.get_empty(
            buffers,
            (self.max_num_sequences + 1,),
            cache_name="qsa_cu_seq_lens",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self._qsa_cu_seq_lens.zero_()

        # TopK writes in place; caller ownership keeps the address graph-stable.
        self.qsa_topk_indices = self.get_empty(
            buffers,
            (
                self.max_num_tokens,
                self.sparse_metadata_params.block_topk,
            ),
            cache_name="qsa_topk_indices",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Per-row Top-K lower bounds; QSA searches every row from zero.
        self.qsa_topk_row_starts = self.get_empty(
            buffers,
            (self.max_num_tokens,),
            cache_name="qsa_topk_row_starts",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.qsa_topk_row_starts.zero_()

    def _refresh_qsa_token_mapping(self) -> None:
        """Map packed rows to requests, logical positions, and causal limits."""
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
                sequence_lengths=self.qsa_sequence_lengths[:num_tokens],
                visible_blocks=self.qsa_visible_blocks[:num_tokens],
                compress_ratio=self.sparse_metadata_params.compress_ratio,
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
        self.qsa_sequence_lengths[:num_tokens].copy_(self.kv_lens_cuda_runtime[:num_seqs][req_idx])
        self.qsa_visible_blocks[:num_tokens].copy_(
            ((logical + 1) // self.sparse_metadata_params.compress_ratio).to(torch.int32)
        )

    def _refresh_qsa_block_table(self) -> None:
        """Decode V2 attention offsets into raw page IDs used by QSA kernels."""
        num_seqs = self.num_seqs
        if num_seqs <= 0:
            return
        scaled = self.kv_cache_block_offsets[
            self._qsa_attention_pool_id,
            :num_seqs,
            QSA_KEY_ROLE_INDEX,
            :,
        ]
        triton_qsa_unscale_block_table(
            scaled_block_table=scaled,
            block_table=self.qsa_block_table[:num_seqs],
            page_index_scale=self._qsa_page_index_scale,
        )

    def _refresh_qsa_speculative_snapshot_state(self) -> None:
        """Cache whether eager generation contains multi-token verification rows.

        Host lengths are already available during prepare; using them avoids a
        device reduction and synchronization in every QSA layer.
        """
        generation_lens = self.seq_lens[self.num_contexts : self.num_seqs]
        self.qsa_needs_speculative_snapshot = any(length > 1 for length in generation_lens.tolist())

    def _refresh_qsa_device_state(self) -> None:
        """Rebuild the device-side block table and token mapping for this step."""
        if not self.qsa_has_local_layers:
            return
        self._refresh_qsa_block_table()
        self._refresh_qsa_token_mapping()

    def prepare(self) -> None:
        """Refresh host decisions, then rebuild graph-stable device mappings."""
        super().prepare()
        if not self.qsa_has_local_layers:
            return
        self._refresh_qsa_speculative_snapshot_state()
        self._refresh_qsa_device_state()

    def on_update_kv_lens(self) -> None:
        """Refresh row limits after speculative decoding advances KV lengths."""
        super().on_update_kv_lens()
        if not self.qsa_has_local_layers:
            return
        # Page allocation is unchanged within the step, so only the row mapping
        # depends on the updated lengths.
        self._refresh_qsa_token_mapping()


__all__ = ["QSAAttentionMetadata"]
