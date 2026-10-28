# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention metadata state for QSA sparse attention."""

from typing import Any, Optional

import torch

from tensorrt_llm._torch.attention.backends.features import DraftKVMetadata
from tensorrt_llm._torch.attention.backends.sparse.params import (
    MTPIndexShareExtension,
    SparseAlgorithmExtension,
)

from .constants import QSA_KEY_ROLE_INDEX
from .kernels import triton_qsa_unscale_block_table
from .params import QSASparseMetadataParams


class QSAMetadata(SparseAlgorithmExtension, MTPIndexShareExtension):
    """Graph-stable QSA page and row mappings of a TRTLLM metadata."""

    def allocate(self, buffers) -> None:
        """Allocate the fixed-shape QSA buffers updated before each graph replay."""
        from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import Qwen4ExpHybridCacheManager

        md = self.metadata
        self.needs_speculative_snapshot = False
        # Draft-loop state; set by the speculative driver, read by the indexer.
        self.in_mtp_draft_loop = False
        self.indexer_skip_topk = False
        self.mtp_num_accepted: Optional[torch.Tensor] = None
        self.shared_topk_indices: Optional[torch.Tensor] = None
        self.shared_topk_visible_blocks: Optional[torch.Tensor] = None
        self.shared_topk_captured_slots: set[int] = set()
        self.draft_block_table: Optional[torch.Tensor] = None
        if not isinstance(md.kv_cache_manager, Qwen4ExpHybridCacheManager):
            raise TypeError("QSA sparse attention requires Qwen4ExpHybridCacheManager")
        params = md.sparse_metadata_params
        if not isinstance(params, QSASparseMetadataParams):
            raise TypeError("QSA sparse attention requires QSASparseMetadataParams")
        self.params = params
        self.has_local_layers = md.kv_cache_manager.qsa_position_layer_id is not None
        if not self.has_local_layers:
            # A PP rank may own only recurrent layers. It still uses the hybrid
            # manager and core metadata, but needs no QSA page views.
            return

        # The sparse kernels index raw V2 pool pages, while the core metadata
        # stores attention-operator encoded offsets. Keep an unscaled table.
        self.block_table = md.get_buffer(
            (md.max_num_sequences, md.kv_cache_manager.max_blocks_per_seq),
            torch.int32,
            "qsa_block_table",
        )
        self._attention_pool_id, self._page_index_scale = (
            md.kv_cache_manager.get_qsa_attention_pool_layout()
        )
        self._init_draft_block_table()

        # Packed-row mapping shared by compression, selection, and sparse GQA.
        self.req_idx_per_token = md.get_buffer(
            (md.max_num_tokens,), torch.int32, "qsa_req_idx_per_token"
        )
        self.logical_positions = md.get_buffer(
            (md.max_num_tokens,), torch.int64, "qsa_logical_positions"
        )
        self.sequence_lengths = md.get_buffer(
            (md.max_num_tokens,), torch.int32, "qsa_sequence_lengths"
        )
        self.visible_blocks = md.get_buffer((md.max_num_tokens,), torch.int32, "qsa_visible_blocks")

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
        self.topk_indices = md.get_buffer(
            (md.max_num_tokens, params.block_topk), torch.int32, "qsa_topk_indices"
        )
        # Per-row Top-K lower bounds; QSA searches every row from zero.
        self.topk_row_starts = md.get_buffer(
            (md.max_num_tokens,), torch.int32, "qsa_topk_row_starts"
        )
        self.topk_row_starts.zero_()

        # One captured selection per sparse layer and request, reused by the
        # MTP draft loop. The layer axis is required because each layer runs
        # its own indexer projection over its own index-K cache, so their
        # selections differ. ``shared_topk_visible_blocks`` records how many
        # compressed groups were complete when the row was captured, so the
        # reuse can tell which groups formed afterwards.
        if params.mtp_index_share:
            # Slots are numbered by the bound manager; the draft loop binds the
            # separate draft manager.
            num_slots = len(md.kv_cache_manager.qsa_shared_topk_slots)
            if self.draft_block_table is not None:
                num_slots = max(num_slots, len(md.draft_kv_cache_manager.qsa_shared_topk_slots))
            self.shared_topk_indices = md.get_buffer(
                (num_slots, md.max_num_sequences, params.block_topk),
                torch.int32,
                "qsa_shared_topk_indices",
            )
            self.shared_topk_visible_blocks = md.get_buffer(
                (num_slots, md.max_num_sequences),
                torch.int32,
                "qsa_shared_topk_visible_blocks",
            )

    def _init_draft_block_table(self) -> None:
        """Allocate the page table of a separate one-model draft KV cache.

        The draft layers' pages live in the draft manager's pools, which the
        target table does not address. This table is its own graph-stable
        buffer, so a captured graph reads the target table in the target
        forward and this one in the draft loop; ``prepare()`` refreshes both
        before every replay.
        """
        from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import Qwen4ExpHybridCacheManager

        md = self.metadata
        draft_manager = md.draft_kv_cache_manager
        if (
            not isinstance(draft_manager, Qwen4ExpHybridCacheManager)
            or draft_manager.qsa_position_layer_id is None
        ):
            return
        self.draft_block_table = md.get_buffer(
            (md.max_num_sequences, draft_manager.max_blocks_per_seq),
            torch.int32,
            "qsa_draft_block_table",
        )
        self._draft_attention_pool_id, self._draft_page_index_scale = (
            draft_manager.get_qsa_attention_pool_layout()
        )

    def set_skip_topk(self, skip: bool) -> None:
        """Mark a draft step that reuses the captured selection."""
        self.indexer_skip_topk = skip

    def set_in_mtp_draft_loop(self, active: bool) -> None:
        """Bracket the MTP draft loop so the indexer captures once and reuses."""
        self.in_mtp_draft_loop = active
        # Readiness is per layer slot, not global: a layer may reuse only the
        # row it captured itself, and the layers do not all capture in the same
        # pass. A loop whose first step is a context pass captures nothing, so
        # the next step captures layer by layer.
        self.shared_topk_captured_slots.clear()

    def set_mtp_num_accepted(self, num_accepted: Optional[torch.Tensor]) -> None:
        """Supply the per-request accepted counts used to pick the capture row."""
        self.mtp_num_accepted = num_accepted

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
                request_indices=self.req_idx_per_token[:num_tokens],
                logical_positions=self.logical_positions[:num_tokens],
                sequence_lengths=self.sequence_lengths[:num_tokens],
                visible_blocks=self.visible_blocks[:num_tokens],
                compress_ratio=self.params.compress_ratio,
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
        self.req_idx_per_token[:num_tokens].copy_(req_idx.to(torch.int32))
        seq_starts = self._cu_seq_lens[:num_seqs]
        cached_lens = md.kv_lens_cuda_runtime[:num_seqs].to(torch.int64) - seq_lens.to(torch.int64)
        logical = cached_lens[req_idx] + token_ids - seq_starts[req_idx]
        self.logical_positions[:num_tokens].copy_(logical)
        self.sequence_lengths[:num_tokens].copy_(md.kv_lens_cuda_runtime[:num_seqs][req_idx])
        self.visible_blocks[:num_tokens].copy_(
            ((logical + 1) // self.params.compress_ratio).to(torch.int32)
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
        """Decode the bound manager's offsets into the table QSA kernels read.

        While a draft forward is in effect every operand here is the draft
        manager's, so this rebuilds the draft table.
        """
        self._unscale_block_table(
            self.metadata.kv_cache_block_offsets,
            self._attention_pool_id,
            self._page_index_scale,
            self.block_table,
        )

    def _refresh_draft_block_table(self) -> None:
        """Decode the separate draft manager's offsets into the draft table."""
        if self.draft_block_table is None:
            return
        self._unscale_block_table(
            self.metadata.ext(DraftKVMetadata).kv_cache_block_offsets,
            self._draft_attention_pool_id,
            self._draft_page_index_scale,
            self.draft_block_table,
        )

    def _refresh_speculative_snapshot_state(self) -> None:
        """Cache whether eager generation contains multi-token verification rows.

        Host lengths are already available during prepare; using them avoids a
        device reduction and synchronization in every QSA layer.
        """
        md = self.metadata
        generation_lens = md.seq_lens[md.num_contexts : md.num_seqs]
        self.needs_speculative_snapshot = any(length > 1 for length in generation_lens.tolist())

    def _refresh_device_state(self) -> None:
        """Rebuild the device-side block table and token mapping for this step."""
        if not self.has_local_layers:
            return
        self._refresh_block_table()
        self._refresh_draft_block_table()
        self._refresh_token_mapping()

    def draft_swap(self) -> Any:
        """Point the QSA page table at the separate draft manager's pages.

        The caller has already swapped ``kv_cache_manager`` and
        ``kv_cache_block_offsets`` to the draft manager's. The draft table was
        decoded in ``prepare()``, so this only rebinds fields and launches
        nothing. The row mapping is manager-independent and stays as it is.
        """
        if self.draft_block_table is None:
            return None
        md = self.metadata
        if md.kv_cache_manager is not md.draft_kv_cache_manager:
            raise RuntimeError(
                "QSA draft forward expects the metadata's draft KV cache manager "
                f"to be bound; got {type(md.kv_cache_manager).__name__}"
            )
        return self.swap_attrs_for_draft(
            {
                "block_table": self.draft_block_table,
                "_attention_pool_id": self._draft_attention_pool_id,
                "_page_index_scale": self._draft_page_index_scale,
            }
        )

    def draft_restore(self, saved: Any) -> None:
        if saved is not None:
            self.restore_swapped_attrs(saved)

    def prepare(self) -> None:
        """Refresh host decisions, then rebuild graph-stable device mappings."""
        if not self.has_local_layers:
            return
        self._refresh_speculative_snapshot_state()
        self._refresh_device_state()

    def on_kv_lens_changed(self, spec_dec_step: bool) -> None:
        """Refresh row limits after speculative decoding advances KV lengths.

        Page allocation is unchanged within the step, so only the row mapping
        depends on the updated lengths. A draft step also repacks the batch
        from one row per accepted token to a single row per request.
        """
        if self.has_local_layers:
            self._refresh_token_mapping()


__all__ = ["QSAMetadata"]
