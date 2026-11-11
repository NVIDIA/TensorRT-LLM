# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention metadata for QSA sparse attention."""

from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.sparse.params import (
    MTPIndexShareUser,
    SparseAlgorithmExtension,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

from .constants import QSA_KEY_ROLE_INDEX
from .kernels import (
    QSA_GENERATION_MAPPING_MAX_SEQS,
    triton_qsa_generation_token_mapping,
    triton_qsa_unscale_block_table,
)
from .params import QSASparseMetadataParams


def _index_share_field(name: str) -> property:
    """A property reading and writing ``name`` of the ``MTPIndexShareExtension``."""

    def get(self):
        return getattr(self.index_share, name)

    def set(self, value):
        setattr(self.index_share, name, value)

    return property(get, set, doc=f"MTPIndexShareExtension.{name}")


class QSAMetadata(SparseAlgorithmExtension, MTPIndexShareUser):
    """Graph-stable QSA page and row mappings of a TRTLLM metadata."""

    fields = frozenset(
        {
            "_attention_pool_id",
            "_cu_seq_lens",
            "_page_index_scale",
            "_token_arange",
            "qsa_block_table",
            "qsa_has_local_layers",
            "qsa_in_mtp_draft_loop",
            "qsa_indexer_skip_topk",
            "qsa_logical_positions",
            "qsa_mtp_num_accepted",
            "qsa_needs_speculative_snapshot",
            "qsa_req_idx_per_token",
            "qsa_sequence_lengths",
            "qsa_shared_topk_captured_slots",
            "qsa_shared_topk_indices",
            "qsa_shared_topk_visible_blocks",
            "qsa_topk_indices",
            "qsa_topk_row_starts",
            "qsa_visible_blocks",
        }
    )
    methods = frozenset(
        {
            "_refresh_block_table",
            "_refresh_speculative_snapshot_state",
            "_refresh_token_mapping",
        }
    )

    # The page table, packed-row request map and draft-loop state live in the
    # MTPIndexShareExtension; these read and write them under QSA's names.
    qsa_block_table = _index_share_field("block_table")
    qsa_req_idx_per_token = _index_share_field("req_idx_per_token")
    qsa_shared_topk_indices = _index_share_field("shared_topk_indices")
    qsa_shared_topk_captured_slots = _index_share_field("shared_topk_captured_slots")
    qsa_in_mtp_draft_loop = _index_share_field("in_mtp_draft_loop")
    qsa_indexer_skip_topk = _index_share_field("indexer_skip_topk")
    qsa_mtp_num_accepted = _index_share_field("mtp_num_accepted")

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import Qwen4ExpHybridCacheManager

        md = self.metadata
        self.qsa_needs_speculative_snapshot = False
        # The shared selection is allocated below when used.
        self.index_share.bind_buffers(shared_topk_indices=None)
        self.qsa_shared_topk_visible_blocks: Optional[torch.Tensor] = None
        if not isinstance(md.kv_cache_manager, Qwen4ExpHybridCacheManager):
            raise TypeError("QSA sparse attention requires Qwen4ExpHybridCacheManager")
        params = md.sparse_metadata_params
        if not isinstance(params, QSASparseMetadataParams):
            raise TypeError("QSA sparse attention requires QSASparseMetadataParams")
        self.qsa_has_local_layers = md.kv_cache_manager.qsa_position_layer_id is not None
        if not self.qsa_has_local_layers:
            # A PP rank may own only recurrent layers. It still uses the hybrid
            # manager and parent metadata, but needs no QSA page views.
            return

        # The sparse kernels index raw V2 pool pages, while the parent metadata
        # stores attention-operator encoded offsets. Keep an unscaled table.
        self.index_share.bind_buffers(
            block_table=md.get_buffer(
                (md.max_num_sequences, md.kv_cache_manager.max_blocks_per_seq),
                torch.int32,
                "qsa_block_table",
            )
        )
        self._attention_pool_id, self._page_index_scale = (
            md.kv_cache_manager.get_qsa_attention_pool_layout()
        )

        # Packed-row mapping shared by compression, selection, and sparse GQA.
        self.index_share.bind_buffers(
            req_idx_per_token=md.get_buffer(
                (md.max_num_tokens,), torch.int32, "qsa_req_idx_per_token"
            )
        )
        self.qsa_logical_positions = md.get_buffer(
            (md.max_num_tokens,), torch.int64, "qsa_logical_positions"
        )
        self.qsa_sequence_lengths = md.get_buffer(
            (md.max_num_tokens,), torch.int32, "qsa_sequence_lengths"
        )
        self.qsa_visible_blocks = md.get_buffer(
            (md.max_num_tokens,), torch.int32, "qsa_visible_blocks"
        )

        # Reuse graph-stable row IDs instead of allocating arange each step.
        self._token_arange = md.get_buffer((md.max_num_tokens,), torch.int64, "qsa_token_arange")
        self._token_arange.copy_(
            torch.arange(
                md.max_num_tokens,
                device=self._token_arange.device,
                dtype=torch.int64,
            )
        )
        # The leading zero turns request lengths into packed-row boundaries.
        self._cu_seq_lens = md.get_buffer(
            (md.max_num_sequences + 1,), torch.int64, "qsa_cu_seq_lens"
        )
        self._cu_seq_lens.zero_()

        # TopK writes in place; caller ownership keeps the address graph-stable.
        self.qsa_topk_indices = md.get_buffer(
            (md.max_num_tokens, params.block_topk), torch.int32, "qsa_topk_indices"
        )
        # Per-row Top-K lower bounds; QSA searches every row from zero.
        self.qsa_topk_row_starts = md.get_buffer(
            (md.max_num_tokens,), torch.int32, "qsa_topk_row_starts"
        )
        self.qsa_topk_row_starts.zero_()

        # One captured selection per sparse layer and request, reused by the
        # MTP draft loop. The layer axis is required because each layer runs
        # its own indexer projection over its own index-K cache, so their
        # selections differ. ``shared_topk_visible_blocks`` records how many
        # compressed groups were complete when the row was captured, so the
        # reuse can tell which groups formed afterwards.
        if params.mtp_index_share:
            # Slots are numbered by the bound manager (the draft manager on a
            # draft metadata over a separate draft KV cache).
            num_slots = len(md.kv_cache_manager.qsa_shared_topk_slots)
            self.index_share.bind_buffers(
                shared_topk_indices=md.get_buffer(
                    (num_slots, md.max_num_sequences, params.block_topk),
                    torch.int32,
                    "qsa_shared_topk_indices",
                )
            )
            self.qsa_shared_topk_visible_blocks = md.get_buffer(
                (num_slots, md.max_num_sequences),
                torch.int32,
                "qsa_shared_topk_visible_blocks",
            )

    def _refresh_token_mapping(self) -> None:
        """Map packed rows to requests, logical positions, and causal limits."""
        md = self.metadata
        num_seqs = md.num_seqs
        num_tokens = md.num_tokens
        if num_seqs <= 0 or num_tokens <= 0:
            return
        seq_lens = md.seq_lens_cuda[:num_seqs]
        if md.num_contexts == 0 and num_tokens == num_seqs:
            from .kernels import triton_qsa_decode_token_mapping

            triton_qsa_decode_token_mapping(
                kv_lens=md.kv_lens_cuda_runtime[:num_seqs],
                seq_lens=seq_lens,
                request_indices=self.index_share.req_idx_per_token[:num_tokens],
                logical_positions=self.qsa_logical_positions[:num_tokens],
                sequence_lengths=self.qsa_sequence_lengths[:num_tokens],
                visible_blocks=self.qsa_visible_blocks[:num_tokens],
                compress_ratio=self.sparse_metadata_params.compress_ratio,
            )
            return
        if md.num_contexts == 0 and num_seqs <= QSA_GENERATION_MAPPING_MAX_SEQS:
            triton_qsa_generation_token_mapping(
                kv_lens=md.kv_lens_cuda_runtime[:num_seqs],
                seq_lens=seq_lens,
                request_indices=self.index_share.req_idx_per_token,
                logical_positions=self.qsa_logical_positions,
                sequence_lengths=self.qsa_sequence_lengths,
                visible_blocks=self.qsa_visible_blocks,
                num_rows=num_tokens,
                compress_ratio=self.sparse_metadata_params.compress_ratio,
            )
            return
        torch.cumsum(
            seq_lens,
            dim=0,
            dtype=torch.int64,
            out=self._cu_seq_lens[1 : num_seqs + 1],
        )
        token_ids = self._token_arange[:num_tokens]
        req_idx = torch.searchsorted(
            self._cu_seq_lens[1 : num_seqs + 1],
            token_ids,
            right=True,
        )
        self.index_share.req_idx_per_token[:num_tokens].copy_(req_idx.to(torch.int32))
        seq_starts = self._cu_seq_lens[:num_seqs]
        cached_lens = md.kv_lens_cuda_runtime[:num_seqs].to(torch.int64) - seq_lens.to(torch.int64)
        logical = cached_lens[req_idx] + token_ids - seq_starts[req_idx]
        self.qsa_logical_positions[:num_tokens].copy_(logical)
        self.qsa_sequence_lengths[:num_tokens].copy_(md.kv_lens_cuda_runtime[:num_seqs][req_idx])
        self.qsa_visible_blocks[:num_tokens].copy_(
            ((logical + 1) // self.sparse_metadata_params.compress_ratio).to(torch.int32)
        )

    def _unscale_block_table(
        self,
        block_offsets: torch.Tensor,
        pool_id: int,
        page_index_scale: int,
        block_table: torch.Tensor,
    ) -> None:
        """Decode one manager's V2 attention offsets into raw page IDs."""
        num_seqs = self.metadata.num_seqs
        if num_seqs <= 0:
            return
        triton_qsa_unscale_block_table(
            scaled_block_table=block_offsets[pool_id, :num_seqs, QSA_KEY_ROLE_INDEX, :],
            block_table=block_table[:num_seqs],
            page_index_scale=page_index_scale,
        )

    def _refresh_block_table(self) -> None:
        """Decode the bound manager's offsets into the table QSA kernels read."""
        self._unscale_block_table(
            self.metadata.kv_cache_block_offsets,
            self._attention_pool_id,
            self._page_index_scale,
            self.index_share.block_table,
        )

    def _refresh_speculative_snapshot_state(self) -> None:
        """Cache whether eager generation contains multi-token verification rows.

        Host lengths are already available during prepare; using them avoids a
        device reduction and synchronization in every QSA layer.
        """
        md = self.metadata
        generation_lens = md.seq_lens[md.num_contexts : md.num_seqs]
        self.qsa_needs_speculative_snapshot = any(length > 1 for length in generation_lens.tolist())

    def _refresh_device_state(self) -> None:
        """Rebuild the device-side block table and token mapping for this step."""
        if not self.qsa_has_local_layers:
            return
        self._refresh_block_table()
        self._refresh_token_mapping()

    def prepare(self) -> None:
        """Refresh host decisions, then rebuild graph-stable device mappings."""
        if not self.qsa_has_local_layers:
            return
        self._refresh_speculative_snapshot_state()
        self._refresh_device_state()

    def refresh_from_draft_lengths(self) -> None:
        super().refresh_from_draft_lengths()
        # A draft step also repacks the batch from one row per accepted token
        # to a single row per request.
        if self.qsa_has_local_layers:
            self._refresh_token_mapping()


class QSAAttentionMetadata(TrtllmAttentionMetadata):
    """TRTLLM attention metadata with the QSA extension attached."""

    backend_extension_types = (QSAMetadata,)


__all__ = ["QSAAttentionMetadata", "QSAMetadata"]
