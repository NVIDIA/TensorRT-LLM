# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash worker's side of the unified draft cache, ahead of the engine ``DFlashWorker`` in every K3.1 worker's
MRO (after the target's own worker mixins: ``DFlash2DraftPagesWorkerMixin`` on the MXFP4 targets,
``RubinDFlash2WorkerMixin`` on the NVFP4 ones).

With a unified draft cache the drafter's context lives in the KV cache manager's draft layers
(``ops/ssm/unified_draft_cache.py``) instead of the worker's private arena. The worker then:

* takes the target's KV cache manager as its draft manager (``get_draft_kv_cache_manager``) and binds the draft layers'
  pages as its context pool (``_managed_ctx_pool``);
* before each forward (eager, capture or replay), binds newly resident requests to slots and restores their draft
  histories without rewinding overlapping iterations, then decodes the requests' draft block tables
  (``prepare_managed_draft_cache``, ``_refresh_ctx_block_tables``);
* after each step, queues the histories on the manager, which publishes them once the step's forward is done
  (``snapshot_managed_draft_history``);
* keeps, per slot, the history position offset past the context length capped at ``_max_ctx``: the absolute position
  of a request's next draft entry is its slot's length plus that offset. The engine's slot assignment, length
  advance and rollback, context store (``_drop_overflowing_chunk``, ``_after_context_chunk``) and fused
  generation-step preparation (``_prepare_paged_latent_gen_inputs``) are wrapped to keep it;
* under Helix, stores and reads only this rank's pages (``_helix_shard``, from the drafter's ``_helix_mapping``).
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.helix_draft_shard import (
    HelixDraftShard,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.helix_paged_latent import (
    helix_paged_latent_prepare,
    helix_store_context_latents,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.unified_draft_cache import (
    DraftHistoryUpdate,
    StandaloneDraftHistory,
)
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.speculative.dflash import (
    dflash_noise_block_embedding,
    dflash_rebased_ctx_rows,
)
from tensorrt_llm._torch.speculative.spec_tail_kernels import DraftBlockPlacement
from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned
from tensorrt_llm.logger import logger


def _has_standalone_draft_cache(manager) -> bool:
    return bool(getattr(manager, "has_standalone_draft_cache", False))


class UnifiedDraftCacheWorkerMixin:
    """The unified draft cache methods of the K3.1 DFlash worker (see the module docstring)."""

    # Bound by _lazy_init_ctx_buffers: the draft KV cache manager; this iteration's draft block counts on the host when
    # the unified draft cache built them there (_refresh_ctx_block_tables), else None.
    _ctx_kv_manager = None
    _ctx_block_counts_host: Optional[list[int]] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Per slot: the history position offset (allocated with the slots) and its pre-forward snapshot.
        self._ctx_position_offset: Optional[torch.Tensor] = None
        self._saved_ctx_position_offset: Optional[torch.Tensor] = None
        # The KV cache each request's slot was bound to, and the step's requests and their slots for the snapshot.
        self._managed_cache_bindings = {}
        self._managed_request_ids = ()
        self._managed_snapshot_slots: Optional[torch.Tensor] = None
        self._ctx_block_counts_host: Optional[list[int]] = None
        # The position offsets a context store records, uploaded once after it.
        self._context_position_offsets: dict[int, int] = {}

    def set_draft_model(self, draft_model) -> None:
        super().set_draft_model(draft_model)
        self._helix_mapping = getattr(draft_model, "_helix_mapping", None)

    def _helix_shard(self) -> Optional[HelixDraftShard]:
        """This rank's share of the context pages when the drafter is helix-sharded."""
        helix = getattr(self, "_helix_mapping", None)
        if helix is None:
            return None
        return HelixDraftShard(helix.cp_size, helix.cp_rank, self._ctx_page_size)

    def get_draft_kv_cache_manager(self, resource_manager):
        """The target's KV cache manager when it holds the drafter's context; the engine's choice otherwise."""
        if resource_manager is not None:
            manager = resource_manager.get_resource_manager(ResourceManagerType.KV_CACHE_MANAGER)
            if _has_standalone_draft_cache(manager):
                return manager
        return super().get_draft_kv_cache_manager(resource_manager)

    def _has_unified_draft_cache(self) -> bool:
        return _has_standalone_draft_cache(self._ctx_kv_manager)

    def _write_slot_values(self, target: torch.Tensor, values: dict[int, int]) -> None:
        """Scatter a slot-to-value mapping into a device tensor with one upload."""
        if not values:
            return
        slots, vals = zip(*values.items())
        pinned = prefer_pinned()
        target.index_copy_(
            0,
            torch.tensor(slots, dtype=torch.long, pin_memory=pinned).to(
                target.device, non_blocking=True
            ),
            torch.tensor(vals, dtype=target.dtype, pin_memory=pinned).to(
                target.device, non_blocking=True
            ),
        )

    def _alloc_ctx_block_tables(self, num_slots: int, max_blocks: int) -> None:
        """Allocate the per-iteration block table (and its counts) of a unified draft cache."""
        self._ctx_block_tables = torch.zeros(
            (num_slots, max_blocks), dtype=torch.int32, device="cuda"
        )
        self._ctx_block_tables_read = torch.zeros_like(self._ctx_block_tables)
        self._ctx_block_indptr = torch.arange(
            0, (num_slots + 1) * max_blocks, max_blocks, dtype=torch.int32, device="cuda"
        )
        self._ctx_block_counts = torch.zeros(num_slots, dtype=torch.long, device="cuda")
        logger.info(f"DFlash: ctx block tables {tuple(self._ctx_block_tables.shape)}")

    def _reads_shared_manager_offsets(self, draft_kv_cache_manager, attn_metadata) -> bool:
        """Never for a unified draft cache, which keeps its own draft block tables; the engine's answer otherwise."""
        if _has_standalone_draft_cache(draft_kv_cache_manager):
            return False
        return super()._reads_shared_manager_offsets(draft_kv_cache_manager, attn_metadata)

    def _lazy_init_ctx_buffers(
        self, draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager=None
    ):
        """The engine's context buffers. Bound to a unified draft cache, the slots' position offsets and snapshot
        slots are allocated with them, and no prefix is reuse-addressed: the manager restores each request's draft
        history itself."""
        unified = _has_standalone_draft_cache(draft_kv_cache_manager)
        if unified:
            self._validate_draft_attention_backend(draft_model)
        ctx_len = self._ctx_len
        super()._lazy_init_ctx_buffers(
            draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
        )
        if self._ctx_len is ctx_len:
            return
        self._ctx_position_offset = torch.zeros_like(self._ctx_len)
        self._managed_snapshot_slots = (
            torch.empty(spec_metadata.num_seq_slots, dtype=torch.long, device="cuda")
            if unified
            else None
        )
        self._managed_cache_bindings = {}
        self._managed_request_ids = ()
        self._ctx_block_counts_host = None
        if unified:
            self._ctx_reuse_addressable = False

    def _report_prefix_reuse(self, target_cache_manager, joint_fallback: bool) -> None:
        """With a unified draft cache, report once that KV-cache prefix hits keep the drafter's context: its layers
        share each request's reuse match with the target's, and the manager restores the request's draft history.
        The engine's report otherwise."""
        if not _has_standalone_draft_cache(target_cache_manager):
            super()._report_prefix_reuse(target_cache_manager, joint_fallback)
            return
        if getattr(target_cache_manager, "enable_block_reuse", False):
            logger.info_once(
                f"{type(self).__name__}: KV-cache block reuse is on; the drafter's context lives in the target's KV "
                "cache manager, so a prefix-cache hit keeps it (reused draft-layer pages and the request's draft "
                "history).",
                key="dflash_prefix_reuse_report",
            )

    def _init_ctx_block_tables(
        self, draft_kv_cache_manager, pool, num_slots, L, nkv, hd, page_size
    ) -> bool:
        """A unified draft cache's block table, sized by the manager's draft blocks per sequence (each step's block
        indices come from the manager: ``_refresh_ctx_block_tables``); the engine's sizing otherwise."""
        if not _has_standalone_draft_cache(draft_kv_cache_manager):
            return super()._init_ctx_block_tables(
                draft_kv_cache_manager, pool, num_slots, L, nkv, hd, page_size
            )
        self._alloc_ctx_block_tables(num_slots, draft_kv_cache_manager.draft_max_blocks_per_seq)
        return True

    def _assign_slot(
        self,
        req_id: int,
        reset: bool = False,
        updates: Optional[dict] = None,
        bases: Optional[dict] = None,
    ) -> Optional[int]:
        """The engine's slot assignment. With a unified draft cache a resident request keeps its slot (the managed
        preparation binds fresh histories and records their snapshot slots), a new slot starts unbound at position
        offset 0, and running out of slots fails."""
        if not self._has_unified_draft_cache():
            return super()._assign_slot(req_id, reset=reset, updates=updates, bases=bases)
        is_new = req_id not in self._req_to_slot
        if is_new and not self._free_slots:
            raise RuntimeError("Unified draft cache has no free request staging slot")
        slot = super()._assign_slot(req_id, updates=updates, bases=bases)
        if is_new:
            self._managed_cache_bindings.pop(req_id, None)
            self._ctx_position_offset[slot].zero_()
        return slot

    def _advance_ctx_len(self, slots: torch.Tensor, num_accepted: torch.Tensor) -> None:
        """The engine's advance. With a unified draft cache, what the ``_max_ctx`` cap drops moves the slots' position
        offsets, so positions keep advancing past the capped length."""
        if self._has_unified_draft_cache():
            self._ctx_position_offset[slots] += (
                self._ctx_len[slots] + num_accepted - self._max_ctx
            ).clamp_min(0)
        super()._advance_ctx_len(slots, num_accepted)

    def _restore_ctx_len_host(self) -> None:
        """The engine's rollback, and the unified draft cache's position offsets with it."""
        super()._restore_ctx_len_host()
        if self._has_unified_draft_cache() and self._saved_ctx_position_offset is not None:
            self._ctx_position_offset.copy_(self._saved_ctx_position_offset)

    def _store_prefill_context(
        self, draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
    ):
        """The engine's context store, then one upload of the position offsets it recorded
        (``_drop_overflowing_chunk``, ``_after_context_chunk``)."""
        self._context_position_offsets = {}
        super()._store_prefill_context(
            draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
        )
        self._write_slot_values(self._ctx_position_offset, self._context_position_offsets)

    def _drop_overflowing_chunk(
        self, req_id: int, slot: int, chunk_end: int, ctx_len_updates: dict
    ) -> bool:
        """With a unified draft cache a chunk its history cannot hold is dropped: the slot is emptied and the request
        keeps it, at the absolute position ``chunk_end``, so later chunks stay aligned. The engine's otherwise."""
        if not self._has_unified_draft_cache():
            return super()._drop_overflowing_chunk(req_id, slot, chunk_end, ctx_len_updates)
        ctx_len_updates[slot] = 0
        self._req_ctx_pos.pop(req_id, None)
        self._context_position_offsets[slot] = chunk_end
        self._managed_cache_bindings[req_id] = self._ctx_kv_manager.kv_cache_map[req_id]
        return True

    def _after_context_chunk(self, req_id: int, slot: int, end: int) -> None:
        """With a unified draft cache, the slot's position offset after the chunk and the request's binding."""
        if not self._has_unified_draft_cache():
            super()._after_context_chunk(req_id, slot, end)
            return
        self._context_position_offsets[slot] = self._req_ctx_pos[req_id] - end
        self._managed_cache_bindings[req_id] = self._ctx_kv_manager.kv_cache_map[req_id]

    def _prepare_paged_latent_gen_inputs(
        self,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata,
        spec_metadata,
        draft_model,
        total_target_tokens: int,
        mask_token_id: int,
        trained_mask_embedding: Optional[torch.Tensor],
    ) -> dict:
        """The engine's fused generation-step preparation; under Helix, its copy over this rank's pages
        (``_prepare_helix_paged_latent_gen_inputs``). With a unified draft cache, what the ``_max_ctx`` cap drops from
        this step's lengths first moves the slots' position offsets (on the device, before the preparation advances
        the lengths; the dummy slot's stays)."""
        if self._has_unified_draft_cache():
            gens = slice(attn_metadata.num_contexts, attn_metadata.num_seqs)
            slots = self._batch_to_slot_buffer.tensor[gens]
            dropped = self._ctx_len[slots] + num_accepted_tokens[gens] - self._max_ctx
            dropped = torch.where(slots == self._dummy_slot, 0, dropped.clamp_min(0))
            self._ctx_position_offset.index_add_(0, slots, dropped)
        args = (accepted_tokens, num_accepted_tokens, attn_metadata, spec_metadata, draft_model)
        args += (total_target_tokens, mask_token_id, trained_mask_embedding)
        shard = self._helix_shard()
        if shard is None:
            return super()._prepare_paged_latent_gen_inputs(*args)
        return self._prepare_helix_paged_latent_gen_inputs(*args, shard)

    def _prepare_helix_paged_latent_gen_inputs(
        self,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata,
        spec_metadata,
        draft_model,
        total_target_tokens: int,
        mask_token_id: int,
        trained_mask_embedding: Optional[torch.Tensor],
        shard: HelixDraftShard,
    ) -> dict:
        """``DFlashWorker._prepare_paged_latent_gen_inputs`` (``speculative/dflash.py``) with the Helix kernels."""
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        width = self._draft_tokens_per_req
        block_size = self._compute_block_size
        prep = helix_paged_latent_prepare(
            num_accepted_tokens,
            accepted_tokens,
            self._batch_to_slot_buffer.tensor[num_contexts : num_contexts + num_gens],
            self._ctx_len,
            self._ctx_base_pages,
            self._ctx_block_counts,
            self._ctx_block_tables,
            attn_metadata._seq_lens_cuda,
            num_contexts,
            self._dummy_slot,
            self._max_ctx,
            self._ctx_page_size,
            block_size,
            width,
            shard.cp_size,
            shard.cp_rank,
        )
        attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = width

        noise_embedding = dflash_noise_block_embedding(
            draft_model.draft_model_full.model.embed_tokens,
            prep.bonus,
            mask_token_id,
            block_size,
            trained_mask_embedding,
        )
        gen_start = attn_metadata.num_ctx_tokens
        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        gen_hs = captured_hs[gen_start : gen_start + num_gens * width]
        projected = draft_model.project_target_hidden(gen_hs.reshape(-1, gen_hs.shape[-1]))
        latent, _ = draft_model.precompute_context_kv(
            projected.to(self._ctx_kv_buf[0].dtype), prep.ctx_positions
        )
        helix_store_context_latents(
            latent[:, :, 0],
            self._ctx_kv_buf,
            self._ctx_kv_ptrs,
            prep.store_pages,
            prep.store_offsets,
            prep.store_mask,
        )
        gen_rows = torch.arange(
            num_contexts, num_contexts + num_gens, dtype=torch.long, device="cuda"
        )
        self._ctx_block_tables_read[gen_rows] = dflash_rebased_ctx_rows(
            self._ctx_block_tables[gen_rows], prep.rebase_pages
        )
        return {
            "lilicorr_anchor": None,
            "noise_embedding": noise_embedding,
            "query_positions": prep.query_positions,
            "num_ctx_per_req": prep.num_ctx,
            "ctx_k_cache": self._ctx_k_buf,
            "ctx_v_cache": self._ctx_v_buf,
            "ctx_cache_batch_idx": gen_rows,
            "first_prev_tokens": prep.bonus,
            "ctx_kv_cache": self._ctx_kv_buf,
            "ctx_page_table": self._ctx_block_tables_read,
            "block_placement": DraftBlockPlacement(
                prep.blk_pages, prep.blk_offsets, prep.blk_seq_lens, prep.blk_kv_bounds
            ),
        }

    def _restored_ctx_len(self, history: Optional[StandaloneDraftHistory]) -> int:
        """The context length a slot binding ``history`` resumes at.

        The draft block table is the manager's, addressed by absolute position. A history that holds every position
        the drafter's window reads (its last ``min(position, draft_window_size)``; all of them without a window)
        resumes at its position, capped at ``_max_ctx``, so the window reads the pages the request holds. A shorter
        history (a chunk its pages could not hold was dropped) resumes at its valid length. Either way the rest of the
        position is the slot's position offset.
        """
        if history is None:
            return 0
        window = self._ctx_kv_manager.draft_window_size
        held = history.position if window is None else min(history.position, window)
        if history.valid_length < held:
            return history.valid_length
        return min(history.position, self._max_ctx)

    def _prepare_managed_history(self, request_ids: list[int]) -> list[int]:
        """Restore newly resident histories without rewinding overlapping iterations."""
        updates = {self._dummy_slot: 0}
        offsets = {self._dummy_slot: 0}
        restore_rows = []
        real_request_ids = {
            request_id
            for request_id in request_ids
            if request_id != ATTENTION_DP_DUMMY_REQUEST_ID
            and request_id < self._graph_dummy_id_floor
            and request_id not in self._ctx_kv_manager._draft_dummy_request_ids
        }
        for request_id in list(self._req_to_slot):
            if request_id not in real_request_ids:
                slot = self._req_to_slot.pop(request_id)
                updates[slot] = 0
                offsets[slot] = 0
                self._req_ctx_pos.pop(request_id, None)
                self._managed_cache_bindings.pop(request_id, None)
                self._free_slots.append(slot)
        for row, request_id in enumerate(request_ids):
            if request_id in real_request_ids:
                self._assign_slot(request_id)
            slot = self._req_to_slot.get(request_id)
            if slot is not None:
                cache = self._ctx_kv_manager.kv_cache_map[request_id]
                if self._managed_cache_bindings.get(request_id) is cache:
                    continue
                history = self._ctx_kv_manager.get_draft_history(request_id)
                updates[slot] = self._restored_ctx_len(history)
                self._req_ctx_pos[request_id] = history.position if history is not None else 0
                offsets[slot] = self._req_ctx_pos[request_id] - updates[slot]
                self._managed_cache_bindings[request_id] = cache
                restore_rows.append(row)
        self._write_ctx_len(updates)
        self._write_slot_values(self._ctx_position_offset, offsets)
        self._rebuild_batch_to_slot(request_ids)
        self._managed_request_ids = tuple(
            request_id for request_id in request_ids if request_id in real_request_ids
        )
        # Each upload owns fresh host storage. PyTorch's pinned allocator defers
        # recycling it until the asynchronous copy completes.
        self._managed_snapshot_slots[: len(self._managed_request_ids)].copy_(
            torch.tensor(
                [self._req_to_slot[request_id] for request_id in self._managed_request_ids],
                dtype=torch.long,
                device="cpu",
                pin_memory=prefer_pinned(),
            ),
            non_blocking=True,
        )
        return restore_rows

    def _gather_managed_context(self, request_ids: list[int], restore_rows: list[int]) -> None:
        """Refresh VANILLA's dense staging; manager pages remain authoritative."""
        # A drafter that reads the pages itself (MLA) has no dense staging.
        if self._dflash_attention_backend != "VANILLA" or self._ctx_k_buf is None:
            return
        for row in restore_rows:
            request_id = request_ids[row]
            slot = self._req_to_slot.get(request_id)
            if slot is None:
                continue
            length = self._ctx_len_host[slot]
            positions = torch.arange(length, device=self._ctx_k_buf.device)
            pages = self._ctx_block_tables[row, positions // self._ctx_page_size].long()
            offsets = positions % self._ctx_page_size
            for layer_idx, pool in enumerate(self._ctx_kv_buf):
                self._ctx_k_buf[slot, layer_idx, :length] = pool[pages, 0, :, offsets, :]
                self._ctx_v_buf[slot, layer_idx, :length] = pool[pages, 1, :, offsets, :]

    def prepare_managed_draft_cache(
        self, draft_model, spec_metadata, attn_metadata, resource_manager
    ) -> None:
        """Refresh managed inputs before eager execution, capture, or graph replay."""
        manager = self.get_draft_kv_cache_manager(resource_manager)
        if not _has_standalone_draft_cache(manager):
            return
        self._lazy_init_ctx_buffers(draft_model, spec_metadata, attn_metadata, manager)
        restore_rows = self._prepare_managed_history(spec_metadata.request_ids)
        self._refresh_ctx_block_tables(
            attn_metadata, attn_metadata.num_seqs, spec_metadata.request_ids
        )
        self._gather_managed_context(spec_metadata.request_ids, restore_rows)

    def snapshot_managed_draft_history(self) -> None:
        """Queue this forward's draft histories on the manager, which publishes them once the forward is done."""
        if not self._has_unified_draft_cache() or not self._managed_request_ids:
            return
        # Preparation, forward, snapshot gathers and the next preparation run
        # on the execution stream, so its next upload cannot overtake this read.
        slots = self._managed_snapshot_slots[: len(self._managed_request_ids)]
        lengths = self._ctx_len[slots]
        positions = lengths + self._ctx_position_offset[slots]
        window = self._ctx_kv_manager.draft_window_size
        if window is not None:
            # A windowed draft pool keeps only the last `window` positions before
            # `position`; the decode reads no further back.
            lengths = lengths.clamp(max=window)
        self._ctx_kv_manager.queue_draft_history_update(
            DraftHistoryUpdate.capture(
                self._ctx_kv_manager,
                self._managed_request_ids,
                torch.stack((lengths, positions), dim=1),
            )
        )

    def _managed_ctx_pool(self, draft_kv_cache_manager, L, nkv, hd, dtype, kv_factor=2):
        """Per-layer views of the unified draft cache's pages, which the manager funds and addresses (its
        ``tokens_per_block`` is the page size, and it supplies the draft block indices); the engine's pool
        binding for any other draft KV cache manager.

        The pool may store the drafter's K/V in FP8; writes convert. A pool that does not match the drafter's
        layout fails: the drafter's context has no private arena to fall back to here.
        """
        if not _has_standalone_draft_cache(draft_kv_cache_manager):
            return super()._managed_ctx_pool(draft_kv_cache_manager, L, nkv, hd, dtype, kv_factor)
        self._ctx_layer_ids = tuple(range(L))
        layers = [draft_kv_cache_manager.get_draft_buffers(i, kv_layout="HND") for i in range(L)]
        # kv_factor 1 is the MLA drafter's SELFKONLY pool: one latent, no V.
        expected = (kv_factor, nkv, draft_kv_cache_manager.tokens_per_block, hd)
        for layer in layers:
            if tuple(layer.shape[1:]) != expected or layer.dtype not in (
                dtype,
                torch.float8_e4m3fn,
            ):
                raise ValueError("Unified draft cache pool does not match the drafter KV layout")
        base = layers[0]
        logger.info(
            f"DFlash: binding the drafter ctx cache to the draft KV cache manager pool: "
            f"{L} x {tuple(base.shape)} {base.dtype}, page_size={base.size(-2)}"
        )
        return layers

    def _refresh_ctx_block_tables(
        self, attn_metadata, num_seqs: int, request_ids: list[int] | None = None
    ) -> bool:
        """Upload this iteration's draft block table and block counts from the unified draft cache into the
        persistent buffers (the host counts kept in ``_ctx_block_counts_host``); the engine's decode of the
        draft block offsets otherwise.

        In place and fixed-shape so a captured graph keeps reading the same tensor. Returns False when there is
        no table to fill.
        """
        self._ctx_block_counts_host = None
        if self._ctx_block_tables is None or num_seqs <= 0:
            return False
        if not self._has_unified_draft_cache():
            return super()._refresh_ctx_block_tables(attn_metadata, num_seqs)
        table = self._ctx_kv_manager.get_draft_block_table(request_ids[:num_seqs])
        # Pinned: a pageable upload would wait for the step in flight.
        table = maybe_pin_memory(table)
        self._ctx_block_tables[:num_seqs].copy_(table, non_blocking=True)
        counts = [self._ctx_kv_manager.get_draft_num_blocks(rid) for rid in request_ids[:num_seqs]]
        self._ctx_block_counts[:num_seqs].copy_(
            torch.tensor(counts, dtype=torch.long, device="cpu", pin_memory=prefer_pinned()),
            non_blocking=True,
        )
        self._ctx_block_counts_host = counts
        return True


__all__ = ["UnifiedDraftCacheWorkerMixin"]
