# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention metadata for QSA sparse attention."""

from dataclasses import dataclass, field
from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.runtime_hooks import QsaRuntimePolicy

from .constants import QSA_KEY_ROLE_INDEX
from .kernels import (
    QSA_GENERATION_MAPPING_MAX_SEQS,
    triton_qsa_copy_block_offsets,
    triton_qsa_generation_token_mapping,
    triton_qsa_unscale_block_table,
)
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
    # The model's RuntimeHooks.qsa, set by the decoder metadata (CUDA-graph copies inherit it).
    qsa_policy: QsaRuntimePolicy = field(
        init=False,
        default=QsaRuntimePolicy(),
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
        from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import Qwen4ExpHybridCacheManager

        super().__post_init__()
        # Draft-loop state; set by the speculative driver, read by the indexer.
        self.qsa_in_mtp_draft_loop = False
        self.qsa_indexer_skip_topk = False
        self.qsa_mtp_num_accepted: Optional[torch.Tensor] = None
        self.qsa_shared_topk_indices: Optional[torch.Tensor] = None
        self.qsa_shared_topk_visible_blocks: Optional[torch.Tensor] = None
        self.qsa_shared_topk_captured_slots: set[int] = set()
        self.qsa_draft_block_table: Optional[torch.Tensor] = None
        if not isinstance(self.kv_cache_manager, Qwen4ExpHybridCacheManager):
            raise TypeError("QSA sparse attention requires Qwen4ExpHybridCacheManager")
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
        self._init_qsa_draft_block_table()

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

        # One captured selection per sparse layer and request, reused by the
        # MTP draft loop. The layer axis is required because each layer runs
        # its own indexer projection over its own index-K cache, so their
        # selections differ. ``shared_topk_visible_blocks`` records how many
        # compressed groups were complete when the row was captured, so the
        # reuse can tell which groups formed afterwards.
        if self.sparse_metadata_params.mtp_index_share:
            # Slots are numbered by the bound manager; the draft loop binds the
            # separate draft manager.
            num_slots = len(self.kv_cache_manager.qsa_shared_topk_slots)
            if self.qsa_draft_block_table is not None:
                num_slots = max(num_slots, len(self.draft_kv_cache_manager.qsa_shared_topk_slots))
            self.qsa_shared_topk_indices = self.get_empty(
                buffers,
                (
                    num_slots,
                    self.max_num_sequences,
                    self.sparse_metadata_params.block_topk,
                ),
                cache_name="qsa_shared_topk_indices",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.qsa_shared_topk_visible_blocks = self.get_empty(
                buffers,
                (num_slots, self.max_num_sequences),
                cache_name="qsa_shared_topk_visible_blocks",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )

    def _init_qsa_draft_block_table(self) -> None:
        """Allocate the page table of a separate one-model draft KV cache.

        The draft layers' pages live in the draft manager's pools, which the
        target table does not address. This table is its own graph-stable
        buffer, so a captured graph reads the target table in the target
        forward and this one in the draft loop; ``prepare()`` refreshes both
        before every replay.
        """
        from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import Qwen4ExpHybridCacheManager

        draft_manager = self.draft_kv_cache_manager
        if (
            not isinstance(draft_manager, Qwen4ExpHybridCacheManager)
            or draft_manager.qsa_position_layer_id is None
        ):
            return
        self.qsa_draft_block_table = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences, draft_manager.max_blocks_per_seq),
            cache_name="qsa_draft_block_table",
            dtype=torch.int32,
            capture_graph=self.is_cuda_graph,
        )
        self._qsa_draft_attention_pool_id, self._qsa_draft_page_index_scale = (
            draft_manager.get_qsa_attention_pool_layout()
        )

    def set_skip_topk(self, skip: bool) -> None:
        """Mark a draft step that reuses the captured selection."""
        self.qsa_indexer_skip_topk = skip

    def set_in_mtp_draft_loop(self, active: bool) -> None:
        """Bracket the MTP draft loop so the indexer captures once and reuses."""
        self.qsa_in_mtp_draft_loop = active
        # Readiness is per layer slot, not global: a layer may reuse only the
        # row it captured itself, and the layers do not all capture in the same
        # pass. A loop whose first step is a context pass captures nothing, so
        # the next step captures layer by layer.
        self.qsa_shared_topk_captured_slots.clear()

    def set_mtp_num_accepted(self, num_accepted: Optional[torch.Tensor]) -> None:
        """Supply the per-request accepted counts used to pick the capture row."""
        self.qsa_mtp_num_accepted = num_accepted

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
        if self.num_contexts == 0 and num_seqs <= QSA_GENERATION_MAPPING_MAX_SEQS:
            triton_qsa_generation_token_mapping(
                kv_lens=self.kv_lens_cuda_runtime[:num_seqs],
                seq_lens=seq_lens,
                request_indices=self.qsa_req_idx_per_token,
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

    def _unscale_qsa_block_table(
        self,
        block_offsets: torch.Tensor,
        pool_id: int,
        page_index_scale: int,
        block_table: torch.Tensor,
    ) -> None:
        """Decode one manager's V2 attention offsets into raw page IDs."""
        num_seqs = self.num_seqs
        if num_seqs <= 0:
            return
        triton_qsa_unscale_block_table(
            scaled_block_table=block_offsets[pool_id, :num_seqs, QSA_KEY_ROLE_INDEX, :],
            block_table=block_table[:num_seqs],
            page_index_scale=page_index_scale,
        )

    def _refresh_qsa_block_table(self) -> None:
        """Decode the bound manager's offsets into the table QSA kernels read.

        While ``prepare_for_draft_forward()`` is in effect every operand here is
        the draft manager's, so this rebuilds the draft table.
        """
        self._unscale_qsa_block_table(
            self.kv_cache_block_offsets,
            self._qsa_attention_pool_id,
            self._qsa_page_index_scale,
            self.qsa_block_table,
        )

    def _refresh_qsa_draft_block_table(self) -> None:
        """Decode the separate draft manager's offsets into the draft table."""
        if self.qsa_draft_block_table is None:
            return
        self._unscale_qsa_block_table(
            self.draft_kv_cache_block_offsets,
            self._qsa_draft_attention_pool_id,
            self._qsa_draft_page_index_scale,
            self.qsa_draft_block_table,
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
        self._refresh_qsa_draft_block_table()
        self._refresh_qsa_token_mapping()

    def prepare_for_draft_forward(self) -> None:
        """Point the QSA page table at the separate draft manager's pages.

        The caller has already swapped ``kv_cache_manager`` and
        ``kv_cache_block_offsets`` to the draft manager's. The draft table was
        decoded in ``prepare()``, so this only rebinds fields and launches
        nothing; ``restore_draft_swaps()`` restores the target bindings. The
        row mapping is manager-independent and stays as it is.
        """
        if self.qsa_draft_block_table is None:
            return
        if self.kv_cache_manager is not self.draft_kv_cache_manager:
            raise RuntimeError(
                "QSA draft forward expects the metadata's draft KV cache manager "
                f"to be bound; got {type(self.kv_cache_manager).__name__}"
            )
        self.swap_for_draft("qsa_block_table", self.qsa_draft_block_table)
        self.swap_for_draft("_qsa_attention_pool_id", self._qsa_draft_attention_pool_id)
        self.swap_for_draft("_qsa_page_index_scale", self._qsa_draft_page_index_scale)

    def prepare(self) -> None:
        """Refresh host decisions, then rebuild graph-stable device mappings."""
        super().prepare()
        if not self.qsa_has_local_layers:
            return
        self._refresh_qsa_speculative_snapshot_state()
        self._refresh_qsa_device_state()

    @property
    def supports_steady_decode_step(self) -> bool:
        """With ``qsa_policy.steady_decode``: whether ``prepare_steady_decode_step`` below stands in for this
        class's ``prepare()`` (the parent's conditions, with QSA's own prepare). Without it, the parent's
        answer, which is False for this class."""
        if not self.qsa_policy.steady_decode:
            return TrtllmAttentionMetadata.supports_steady_decode_step.fget(self)
        return (
            type(self).prepare is QSAAttentionMetadata.prepare
            and self.qsa_has_local_layers
            and self._steady_decode_config_supported()
        )

    def prepare_steady_decode_step(self, kv_lens, prompt_lens=None) -> None:
        """``prepare()`` of a steady decode step (requires ``supports_steady_decode_step``): the parent's
        steady step (KV lengths from the staging kernel, a manager's block offsets copied only when its
        ``batch_block_table_key`` changed; ``prompt_lens`` as the parent takes them), then QSA's device
        state with an unscale only of a table that was copied (an unchanged offset table unscales to the
        same slot table) and the row mapping over the new lengths. With ``qsa_policy.steady_replay_gap``
        on a CUDA-graph decode step: a changed table is copied and unscaled in one launch where the
        manager's copy is the plain one, and the row mapping is left to the forward, which rewrites it
        before any reader (the overlap scheduler's KV-length correction and ``on_update_kv_lens``, or a
        decode prologue that writes it)."""
        if not self.qsa_policy.steady_decode:
            super().prepare_steady_decode_step(kv_lens, prompt_lens)
            return
        replay_gap = (
            self.qsa_policy.steady_replay_gap
            and self.qsa_has_local_layers
            and self.is_cuda_graph
            and self.num_contexts == 0
        )
        last = tuple(self._steady_block_table_keys)
        fused = self._copy_changed_block_offsets(last) if replay_gap else (False, False)
        super().prepare_steady_decode_step(kv_lens, prompt_lens)
        if not self.qsa_has_local_layers:
            return
        new = tuple(self._steady_block_table_keys)
        self._refresh_qsa_speculative_snapshot_state()
        if (new[0] is None or new[0] != last[0]) and not fused[0]:
            self._refresh_qsa_block_table()
        if (
            self.draft_kv_cache_manager is not None
            and (new[1] is None or new[1] != last[1])
            and not fused[1]
        ):
            self._refresh_qsa_draft_block_table()
        if not replay_gap:
            self._refresh_qsa_token_mapping()

    def _copy_changed_block_offsets(self, last: tuple) -> tuple[bool, bool]:
        """Before the parent's steady step: for each manager whose ``batch_block_table_key`` changed (and is
        not None), its block-offset copy and QSA unscale in one launch (``triton_qsa_copy_block_offsets``),
        where its copy is ``KVCacheManagerV2``'s plain one. The key is then recorded, so the parent skips its
        own copy of that table. Returns, per manager (target, draft), whether its table was taken here."""
        managers = (self.kv_cache_manager, self.draft_kv_cache_manager)
        offsets = (self.kv_cache_block_offsets, self.draft_kv_cache_block_offsets)
        tables = (
            (self.qsa_block_table, self._qsa_attention_pool_id, self._qsa_page_index_scale),
            (
                (
                    self.qsa_draft_block_table,
                    self._qsa_draft_attention_pool_id,
                    self._qsa_draft_page_index_scale,
                )
                if self.qsa_draft_block_table is not None
                else None
            ),
        )
        keys = list(last)
        taken = [False, False]
        for i, (manager, block_offsets, table) in enumerate(zip(managers, offsets, tables)):
            if (
                manager is None
                or table is None
                or not _plain_block_offsets_copy(manager, block_offsets)
            ):
                continue
            key = manager.batch_block_table_key(self.request_ids)
            if key is None or key == last[i]:
                continue
            qsa_table, pool, scale = table
            num_seqs = self.num_seqs
            copy_index = manager.index_mapper.get_copy_index(self.request_ids, 0, 1, False)
            assert copy_index.shape[0] == num_seqs
            triton_qsa_copy_block_offsets(
                host_block_offsets=manager.host_kv_cache_block_offsets,
                block_offsets=block_offsets,
                copy_index=copy_index,
                index_scales=manager.index_scales,
                kv_offset=manager.kv_offset,
                num_seqs=num_seqs,
                qsa_table=qsa_table,
                qsa_pool=pool,
                qsa_page_index_scale=scale,
                qsa_role=QSA_KEY_ROLE_INDEX,
            )
            keys[i] = key
            taken[i] = True
        self._steady_block_table_keys = tuple(keys)
        return taken[0], taken[1]

    def on_update_kv_lens(self) -> None:
        """Refresh row limits after speculative decoding advances KV lengths."""
        super().on_update_kv_lens()
        if not self.qsa_has_local_layers:
            return
        # Page allocation is unchanged within the step, so only the row mapping
        # depends on the updated lengths.
        self._refresh_qsa_token_mapping()

    def update_for_spec_dec(self) -> None:
        """Refresh the row mapping after a draft step repacks the batch.

        The first draft step rewrites seq_lens from one row per accepted token
        to a single row per request and advances kv_lens_cuda in place, so the
        packed-row mapping built during prepare() no longer describes the rows
        the next draft step attends over.
        """
        super().update_for_spec_dec()
        self.on_update_kv_lens()


def _plain_block_offsets_copy(manager, block_offsets: Optional[torch.Tensor]) -> bool:
    """Whether ``manager.copy_batch_block_offsets`` of a beam-1 generation batch into ``block_offsets`` is
    ``KVCacheManagerV2``'s plain host-table copy (``copy_batch_block_offsets_to_device`` on the current
    stream), which ``triton_qsa_copy_block_offsets`` reproduces: not overridden, no per-layer page tables,
    no device expansion, not cross KV, contiguous int32 tables of one width, a width whole native 4-entry
    packs cover (the native copy's last partial pack is not a plain copy)."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
        CacheTypeCpp,
        KVCacheManagerV2,
    )

    if type(manager).copy_batch_block_offsets is not KVCacheManagerV2.copy_batch_block_offsets:
        return False
    materializer = getattr(manager, "_page_table_materializer", None)
    if (
        getattr(manager, "_use_per_layer_page_tables", True)
        or materializer is None
        or materializer.uses_device_expansion
        or manager.kv_cache_type == CacheTypeCpp.CROSS
    ):
        return False
    host = manager.host_kv_cache_block_offsets
    return (
        block_offsets is not None
        and block_offsets.dtype == torch.int32
        and host.dtype == torch.int32
        and block_offsets.dim() == 4
        and host.dim() == 4
        and block_offsets.is_contiguous()
        and host.is_contiguous()
        and block_offsets.shape[0] == host.shape[0]
        and block_offsets.shape[2] == 2
        and host.shape[2] == 2
        and block_offsets.shape[3] == host.shape[3]
        and host.shape[3] % 4 == 0
        and manager._stream.cuda_stream == torch.cuda.current_stream().cuda_stream
    )


__all__ = ["QSAAttentionMetadata"]
