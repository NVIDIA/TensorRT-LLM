# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedded DSpark over the CSA2 manager's explicit draft layers."""

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, ReusePolicy

from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..modules.multi_stream_utils import private_stream
from ..pyexecutor import device_copy
from .dspark import DSparkSpecMetadata, DSv4DSparkWorker

if TYPE_CHECKING:
    from ..attention.backends.interface import AttentionMetadata
    from ..models.modeling_dspark_v41 import DSv41DSparkForCausalLM
    from ..pyexecutor.resource_manager import ResourceManager


class DSparkCopySite(device_copy.CopySiteEnum):
    PAGE_TABLE_OFFSETS = ("dspark.page_table_offsets", device_copy.Direction.H2D)
    PAGE_TABLE_ENTRIES = ("dspark.page_table_entries", device_copy.Direction.H2D)
    REQUEST_SLOTS = ("dspark.request_slots", device_copy.Direction.H2D)
    ROW_TOKEN_CAPACITIES = ("dspark.row_token_capacities", device_copy.Direction.H2D)


class DSv41DSparkWorker(DSv4DSparkWorker):
    """Seed and consume embedded draft KV owned by the CSA2 cache manager."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._context_kv_only = False
        self._draft_kv_manager = None
        self._draft_kv_buffers = ()
        self._draft_layer_ids = ()
        self._draft_block_tables = None
        self._draft_managed_block_tables = None
        self._draft_fallback_block_tables = None
        self._draft_slot_block_tables = None
        self._draft_table_storage = None
        self._draft_capacities = None
        self._draft_page_versions = {}
        self._draft_page_indices = {}
        self._draft_prepared_batch = None
        self._managed_residency = {}
        self._chunking_ids: set[int] = set()
        self._draft_metadata_host_buffers = []
        self._table_slots = None
        self._draft_updates_device = None
        self._draft_indices_device = None

    def _allocate_context_storage(self, rows: int, stages: int, window: int, head_dim: int) -> None:
        return None

    def _can_backfill_main_kv(self, draft_model) -> bool:
        return (
            bool(self._draft_kv_buffers)
            and not self.mapping.enable_attention_dp
            and draft_model.dspark_model._tp_attention_ready
        )

    def prepare_draft_context(
        self,
        draft_model: "DSv41DSparkForCausalLM",
        spec_metadata: DSparkSpecMetadata | None,
        attn_metadata: "AttentionMetadata",
        resource_manager: "ResourceManager",
    ) -> None:
        """Bind the draft cache and publish page tables for this target step.

        Args:
            draft_model: Embedded drafter that reads the managed pages.
            spec_metadata: Current step's requests and capture state.
            attn_metadata: Target metadata defining the context and generation rows.
            resource_manager: Resolver for the target's KV cache manager.
        """
        if spec_metadata is None:
            return
        self._context_kv_only = draft_model.context_kv_only
        if draft_model.is_dspark_context_worker:
            # Confidence calibration and verification planning belong to GEN.
            self.return_confidence = False
        self._lazy_init(draft_model, spec_metadata, attn_metadata)
        manager = self.get_draft_kv_cache_manager(resource_manager)
        if manager is None:
            raise ValueError("V4.1 DSpark requires explicit draft layers in its KV cache manager")
        if manager is not self._draft_kv_manager:
            layers = manager.draft_layer_ids
            if len(layers) != draft_model.num_stages:
                raise ValueError("DSpark stage count does not match the manager's draft layers")
            specs = [manager.get_layer_cache_spec(layer) for layer in layers]
            if any(
                spec.num_kv_heads != 1
                or spec.kv_factor != 1
                or spec.head_dim != int(draft_model._attn_params["head_dim"])
                or spec.sliding_window_size != self._win + 1
                for spec in specs
            ):
                raise ValueError("V4.1 DSpark draft storage does not match the checkpoint")
            shape = (self._batch_to_slot.shape[0], manager.max_blocks_per_seq)
            managed_table = manager.draft_page_table()
            if managed_table is None or managed_table.shape[0] < shape[0]:
                raise ValueError("The manager's draft page table does not cover the worker batch")
            self._prewarm_draft_pages(draft_model, manager)
            # An allocation failure can be retried during warmup. A partial
            # bind must not be reused, including when switching back to another manager.
            self._draft_kv_manager = None
            self._draft_layer_ids = layers
            self._draft_reuse_policy = specs[0].reuse_policy
            self._draft_kv_buffers = tuple(
                manager.get_buffers(layer, kv_layout="HND") for layer in layers
            )
            slots = self._ctx_len.numel()
            self._draft_table_storage = torch.full(
                (slots + shape[0], shape[1]),
                BAD_PAGE_INDEX,
                dtype=torch.int32,
                device=self._ctx_len.device,
            )
            self._draft_slot_block_tables = self._draft_table_storage[:slots]
            self._draft_fallback_block_tables = self._draft_table_storage[slots:]
            self._draft_block_tables = self._draft_fallback_block_tables
            self._draft_managed_block_tables = managed_table[: shape[0]]
            self._draft_capacities = torch.zeros(
                shape[0], dtype=torch.long, device=self._ctx_len.device
            )
            self._slot_upload = device_copy.DeviceCopyBuffer(
                self._batch_to_slot, site=DSparkCopySite.REQUEST_SLOTS, pin_memory=prefer_pinned()
            )
            self._capacity_upload = device_copy.DeviceCopyBuffer(
                self._draft_capacities,
                site=DSparkCopySite.ROW_TOKEN_CAPACITIES,
                pin_memory=prefer_pinned(),
            )
            self._host_slots = torch.empty(shape[0], dtype=torch.long, device="cpu")
            self._host_capacities = torch.empty_like(self._host_slots)
            self._draft_page_versions.clear()
            self._draft_page_indices.clear()
            self._draft_prepared_batch = None
            self._managed_residency.clear()
            self._chunking_ids.clear()
            self._table_slots = None
            self._draft_kv_manager = manager
        spec_metadata._dspark_worker = self
        self._prepare_page_tables(
            spec_metadata.request_ids,
            attn_metadata.num_contexts,
            paused_in_place_ids=getattr(spec_metadata, "paused_in_place_ids", frozenset()),
            context_chunking_ids=spec_metadata.context_chunking_ids,
        )

    def _prewarm_draft_pages(self, draft_model, manager) -> None:
        params = draft_model._attn_params
        if (
            not IS_CUTLASS_DSL_AVAILABLE
            or int(params["head_dim"]) != 512
            or int(params["rope_head_dim"]) != 64
        ):
            return
        from ..custom_ops.dspark_attention_custom_op import warmup_paged_dspark

        heads = set()
        if not self._context_kv_only and self._win == 128:
            inner = draft_model.dspark_model
            heads.add(int(params["n_heads"]))
            if inner._tp_attention_ready:
                heads.update(stage.self_attn.num_heads_tp for stage in inner.mtp_layers)
        warmup_paged_dspark(
            self._ctx_len.device,
            manager.tokens_per_block,
            draft_model.num_stages,
            float(params["eps"]),
            self.max_draft_len,
            heads,
        )

    def _page_indices(self, cache, is_context: bool) -> tuple[int, tuple[int, ...]]:
        manager = self._draft_kv_manager
        first = max(0, cache.history_length - self._win) // manager.tokens_per_block
        count = cache.num_blocks
        if count > manager.max_blocks_per_seq:
            raise ValueError("DSpark page table exceeds its allocated width")
        tables = []
        pools = set()
        for layer in self._draft_layer_ids:
            pool = manager.layer_to_pool_mapping_dict[manager.layer_offsets[layer]]
            if pool in pools:
                continue
            pools.add(pool)
            indices = cache.get_base_page_indices(pool)[first:count]
            if not len(indices) or all(index == BAD_PAGE_INDEX for index in indices):
                raise ValueError("DSpark request has no writable draft pages")
            if not is_context and any(index == BAD_PAGE_INDEX for index in indices):
                raise ValueError("DSpark generation is missing retained draft KV pages")
            scale = manager.get_layer_page_index_scale(layer)
            stride = manager._get_cache_view_page_stride(scale, 1)
            tables.append(
                tuple(
                    BAD_PAGE_INDEX if index == BAD_PAGE_INDEX else int(index) * scale // stride
                    for index in indices
                )
            )
        if any(table != tables[0] for table in tables[1:]):
            raise ValueError("DSpark stages require identical page-table geometry")
        return first, tables[0]

    def _prepare_page_tables(
        self,
        request_ids: list[int],
        num_contexts: int,
        *,
        paused_in_place_ids: frozenset[int] = frozenset(),
        context_chunking_ids: frozenset[int] = frozenset(),
    ) -> None:
        manager = self._draft_kv_manager
        private = self._draft_reuse_policy == ReusePolicy.PRIVATE
        native = private and not num_contexts
        managed = native and manager.draft_page_table_prepared(request_ids)
        # CTX keeps a snapshot: its encoder/decoder passes can prepare inner
        # batches after this call. GEN's single target pass publishes its draft
        # table directly, including the stable address captured by each graph.
        table = self._draft_managed_block_tables if native else self._draft_fallback_block_tables
        if self._draft_block_tables is not table:
            self._draft_block_tables = table
            self._table_slots = None
        batch_state = manager.page_table_conversion_state() if managed else None
        batch_snapshot = (
            (batch_state, tuple(request_ids), paused_in_place_ids)
            if batch_state is not None
            else None
        )
        previous_batch = self._draft_prepared_batch
        if (
            batch_snapshot is not None
            and previous_batch is not None
            and (previous_batch[0] is batch_state and previous_batch[1:] == batch_snapshot[1:])
            and self._slot_upload.host_values is not None
            and self._capacity_upload.host_values is not None
            and not device_copy.CHECK_MODE
        ):
            return
        self._draft_prepared_batch = None
        versioned_window = private and not num_contexts and manager.tokens_per_block == self._win
        real_rows = [
            row
            for row, rid in enumerate(request_ids)
            if rid != 0
            and rid < self._graph_dummy_id_floor
            and rid in manager.kv_cache_map
            and manager.kv_cache_map[rid].is_active
        ]
        real_ids = {request_ids[row] for row in real_rows}
        # An absent context can still be between chunks on a resident cache.
        # Finished contexts release their slot even while their pages are pinned for transfer.
        retained = {
            rid
            for rid in self._chunking_ids - real_ids
            if (cache := manager.kv_cache_map.get(rid)) is not None
            and cache.is_active
            and self._managed_residency.get(rid) is cache
        }
        released = self._req_to_slot.keys() - real_ids - paused_in_place_ids - retained
        for rid in released:
            self._free_slots.append(self._req_to_slot.pop(rid))
            self._managed_residency.pop(rid, None)
        self._chunking_ids.difference_update(released | real_ids)
        self._chunking_ids.update(context_chunking_ids & real_ids)
        slots = [self._scratch_slot] * len(request_ids)
        updates = []
        page_records = []
        capacities = [0] * self._draft_capacities.shape[0]
        # Unmapped native pages are BAD_PAGE_INDEX. The published table's fixed
        # width bounds addressing; zero capacities keep synthetic rows masked.
        native_capacity = table.shape[1] * manager.tokens_per_block if managed else 0
        for row in real_rows:
            rid = request_ids[row]
            cache = manager.kv_cache_map[rid]
            slot = self._req_to_slot.get(rid)
            if slot is None:
                slot = self._assign_slot(rid, reset=False)
            slots[row] = slot
            if managed:
                capacities[row] = native_capacity
            else:
                previous = self._draft_page_versions.get(slot)
                page_indices = self._draft_page_indices.get(slot)
                page_version = cache.page_storage_version
                # A page-sized window moves only at a history-block boundary,
                # which the native storage version tracks alongside page changes.
                if (
                    versioned_window
                    and previous is not None
                    and previous[0] is cache
                    and previous[1] == page_version
                    and previous[-1] is False
                ):
                    version = previous
                else:
                    first = max(0, cache.history_length - self._win) // manager.tokens_per_block
                    version = (cache, page_version, first, cache.num_blocks, row < num_contexts)
                if previous != version:
                    old_indices = page_indices
                    page_indices = self._page_indices(cache, row < num_contexts)
                    if old_indices != page_indices:
                        first, indices = page_indices
                        if old_indices is not None:
                            # Clear retired columns before exposing the updated row.
                            old_first, old_pages = old_indices
                            old_end = old_first + len(old_pages)
                            left_end = min(first, old_end)
                            right_start = max(old_first, first + len(indices))
                            if left_end > old_first:
                                updates.append(
                                    (
                                        slot,
                                        row,
                                        old_first,
                                        (BAD_PAGE_INDEX,) * (left_end - old_first),
                                    )
                                )
                            if old_end > right_start:
                                updates.append(
                                    (
                                        slot,
                                        row,
                                        right_start,
                                        (BAD_PAGE_INDEX,) * (old_end - right_start),
                                    )
                                )
                        updates.append((slot, row, first, indices))
                    page_records.append((slot, version, page_indices))
                first, indices = page_indices
                capacities[row] = (first + len(indices)) * manager.tokens_per_block
            if self._managed_residency.get(rid) is not cache:
                if managed:
                    # Validate a new/resumed cache once. Later page changes are
                    # published by the same native batch conversion as target KV.
                    self._page_indices(cache, False)
                position = cache.history_length
                length = min(position, self._win)
                if row < num_contexts and self._draft_reuse_policy == ReusePolicy.PRIVATE:
                    length = 0
                self._ctx_len[slot].fill_(position)
                self._valid_len[slot].fill_(length)
                # Cache history can lag when overlap suspends a request before
                # its resource update. Fresh GEN slots use this step's positions.
                self._position_initialized[slot].fill_(row < num_contexts)
                self._managed_residency[rid] = cache
        # Generation preparation resets scratch state inside the graph before
        # consuming it. Context seeding still needs the eager reset.
        if num_contexts or not private:
            self._ctx_len[self._scratch_slot].zero_()
            self._valid_len[self._scratch_slot].zero_()
        slot_values = tuple(slots)
        host_slots = self._host_slots.numpy()
        host_slots.fill(self._scratch_slot)
        host_slots[: len(slots)] = slots
        self._host_capacities.numpy()[:] = capacities
        self._slot_upload.update(self._host_slots)
        self._capacity_upload.update(self._host_capacities)
        if managed:
            self._table_slots = None
        gather_table = not managed and (native or self._table_slots != slot_values)
        direct = private and not native and self._table_slots == slot_values
        count = sum(len(indices) for _, _, _, indices in updates) * (2 if direct else 1)
        if count:
            host = next(
                (item for item in self._draft_metadata_host_buffers if item["event"].query()), None
            )
            if host is None:
                host = {"event": torch.cuda.Event()}
                self._draft_metadata_host_buffers.append(host)
            for name, dtype in (("indices", torch.long), ("values", torch.int32)):
                if name not in host or host[name].numel() < count:
                    host[name] = torch.empty(count, dtype=dtype, pin_memory=prefer_pinned())
            width = self._draft_slot_block_tables.shape[1]
            destinations = host["indices"].numpy()
            values = host["values"].numpy()
            offset = 0
            for slot, row, first, indices in updates:
                end = offset + len(indices)
                begin = slot * width + first
                destinations[offset:end] = range(begin, begin + len(indices))
                values[offset:end] = indices
                offset = end
                if direct:
                    end = offset + len(indices)
                    begin = (self._ctx_len.numel() + row) * width + first
                    destinations[offset:end] = range(begin, begin + len(indices))
                    values[offset:end] = indices
                    offset = end
            if self._draft_updates_device is None or self._draft_updates_device.numel() < count:
                self._draft_updates_device, self._draft_indices_device = (
                    torch.empty(count, dtype=torch.int32, device=self._ctx_len.device),
                    torch.empty(count, dtype=torch.long, device=self._ctx_len.device),
                )
            indices_device = self._draft_indices_device[:count]
            values_device = self._draft_updates_device[:count]
            try:
                device_copy.copy(
                    DSparkCopySite.PAGE_TABLE_OFFSETS, indices_device, host["indices"][:count]
                )
                device_copy.copy(
                    DSparkCopySite.PAGE_TABLE_ENTRIES, values_device, host["values"][:count]
                )
                self._draft_table_storage.view(-1).index_copy_(0, indices_device, values_device)
            finally:
                host["event"].record()
        # These records describe the per-slot storage, even if the batch gather fails.
        for slot, version, page_indices in page_records:
            self._draft_page_versions[slot] = version
            self._draft_page_indices[slot] = page_indices
        if not managed and not direct and (updates or gather_table):
            self._table_slots = None
            with manager.overwrite_draft_page_table(table.shape[0]) if native else nullcontext():
                torch.index_select(self._draft_slot_block_tables, 0, self._batch_to_slot, out=table)
            if not native:
                self._table_slots = slot_values
        self._draft_prepared_batch = batch_snapshot

    def _seed_context_windows(
        self,
        draft_model,
        spec_metadata: "DSparkSpecMetadata",
        attn_metadata,
        position_ids: torch.Tensor,
        total_target_tokens: int,
    ) -> None:
        """Seed context chunks using their absolute positions.

        A request can arrive in multiple prefill chunks; continuation chunks
        append to the same request's pages.
        """
        captured = spec_metadata.get_hidden_states(total_target_tokens)
        capture_lens = spec_metadata.context_capture_lens
        if capture_lens is not None and len(capture_lens) != attn_metadata.num_contexts:
            raise ValueError("DSpark capture lengths must match the context requests")
        flat_position_ids = position_ids.reshape(-1)
        private = self._draft_reuse_policy == ReusePolicy.PRIVATE
        private_rows = []
        end = 0
        for i in range(attn_metadata.num_contexts):
            chunk_len = int(attn_metadata._seq_lens[i])
            capture_len = chunk_len if capture_lens is None else capture_lens[i]
            if not 0 <= capture_len <= chunk_len:
                raise ValueError("DSpark capture length must be a suffix of its context chunk")
            end += chunk_len
            if chunk_len == 0:
                continue
            slot = self._req_to_slot.get(spec_metadata.request_ids[i], self._scratch_slot)
            if slot == self._scratch_slot:
                continue
            if captured is None:
                raise RuntimeError(
                    "Embedded DSpark requires captured context to initialize KV history"
                )
            keep = min(self._win, capture_len)
            if private and keep:
                private_rows.append((i, end, keep))
                continue
            if keep:
                count = capture_len if self._draft_kv_manager.enable_block_reuse else keep
                frames = (flat_position_ids[end - count : end].long() + 1).unsqueeze(0)
                draft_model.write_context_pages(
                    captured[end - count : end].unsqueeze(0),
                    frames,
                    torch.ones_like(frames, dtype=torch.bool),
                    self._draft_kv_buffers,
                    self._draft_block_tables[i : i + 1],
                    self._draft_capacities[i : i + 1],
                )
                self._valid_len[slot] = (self._valid_len[slot] + keep).clamp(max=self._win)
            self._ctx_len[slot] = flat_position_ids[end - 1] + 1

        # Group similar suffix lengths to limit padding. Each projection is
        # bounded by the work in eight full windows.
        private_rows.sort(key=lambda row: row[2], reverse=True)
        first = 0
        while first < len(private_rows):
            width = private_rows[first][2]
            last = first + 1
            while (
                last < min(first + 8 * self._win // width, len(private_rows))
                and 2 * private_rows[last][2] >= width
            ):
                last += 1
            rows = private_rows[first:last]
            batch_rows = len(rows)
            row_start, first_end, _ = rows[0]
            capture_start = first_end - width
            if all(
                row == row_start + j and keep == width and end == first_end + j * width
                for j, (row, end, keep) in enumerate(rows)
            ):
                capture = slice(capture_start, rows[-1][1])
                row_slice = slice(row_start, row_start + batch_rows)
                hidden = captured[capture].reshape(batch_rows, width, -1)
                frames = flat_position_ids[capture].long().reshape(batch_rows, width) + 1
                ends = frames[:, -1]
                mask = torch.ones_like(frames, dtype=torch.bool)
                tables = self._draft_block_tables[row_slice]
                capacities = self._draft_capacities[row_slice]
                slots = self._batch_to_slot[row_slice]
                num_valid = width
            else:
                hidden_rows, frame_rows, last_frames = [], [], []
                for _, end, keep in rows:
                    suffix = captured[end - keep : end]
                    positions = flat_position_ids[end - keep : end].long() + 1
                    last_frames.append(positions[-1])
                    if keep != width:
                        suffix = F.pad(suffix, (0, 0, 0, width - keep))
                        positions = F.pad(positions, (0, width - keep))
                    hidden_rows.append(suffix)
                    frame_rows.append(positions)
                hidden, frames, ends = map(torch.stack, (hidden_rows, frame_rows, last_frames))
                mask = frames != 0
                tables = torch.stack([self._draft_block_tables[row] for row, _, _ in rows])
                capacities = torch.stack([self._draft_capacities[row] for row, _, _ in rows])
                slots = torch.stack([self._batch_to_slot[row] for row, _, _ in rows])
                num_valid = mask.sum(dim=1)
            draft_model.write_context_pages(
                hidden, frames, mask, self._draft_kv_buffers, tables, capacities
            )
            self._ctx_len.index_copy_(0, slots, ends)
            valid_len = (self._valid_len.index_select(0, slots) + num_valid).clamp(max=self._win)
            self._valid_len.index_copy_(0, slots, valid_len)
            first = last

    def _follow_target_step(
        self,
        draft_model,
        spec_metadata: "DSparkSpecMetadata",
        attn_metadata,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        position_ids: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        total_target_tokens: int,
    ) -> None:
        """Keep managed draft history current when this step proposes no drafts."""
        if self._context_kv_only:
            return
        captured = spec_metadata.get_hidden_states(total_target_tokens)
        if captured is None:
            raise RuntimeError(
                "DSpark generation requires captured target hidden states; "
                "check target_layer_ids and the target model's capture hooks."
            )
        slots = self._batch_to_slot[num_contexts:batch_size]
        base = attn_metadata.num_ctx_tokens + torch.arange(slots.shape[0], device=slots.device)
        _bonus, rows, start_pos, *_ = self._advance_ragged_generation_state(
            slots,
            accepted_tokens[num_contexts:batch_size],
            num_accepted_tokens[num_contexts:batch_size],
            position_ids.reshape(-1),
            base,
            captured.shape[0],
        )
        draft_model.write_context_pages(
            None,
            start_pos.unsqueeze(1),
            (slots != self._scratch_slot).unsqueeze(1),
            self._draft_kv_buffers,
            self._draft_block_tables[num_contexts:batch_size],
            self._draft_capacities[num_contexts:batch_size],
            main_x=draft_model.project_captured_rows(captured, rows.unsqueeze(1)),
        )

    def _draft_gen_block_batched(
        self,
        draft_model,
        spec_metadata: "DSparkSpecMetadata",
        attn_metadata,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        total_target_tokens: int,
        position_ids: torch.Tensor,
        all_rank_num_tokens: list[int] | None = None,
        return_tokens: bool = False,
        prepared: tuple | None = None,
    ) -> torch.Tensor | None:
        """CUDA-graph-safe batched gen draft (all gen requests in one forward).

        Free of host syncs and data-dependent shapes: per-request quantities
        (``nacc``, the bonus, ``main_hidden``, ``start_pos``, the multi-accept
        back-fill) are gathered as tensors, slots come from the host-built
        ``_batch_to_slot`` mirror, and the backbone runs once via
        ``DSv4DSparkDraftModel.forward_batched``. Returns the per-position corrected
        block logits ``[num_gens, K, vocab]`` (or ``None`` when there is nothing to
        draft); the worker feeds them to ``SpecWorkerBase.sample_draft_tokens``.
        With ``return_tokens``, returns the model's greedy ``[num_gens, K]``
        int32 proposals instead; the caller must first check sampler eligibility.
        Confidence truncation stays disabled — the full block is proposed. When
        enabled, fixed-shape confidence scores are stored separately for
        verification scheduling; they never truncate the proposal.
        ``prepared`` is ``_advance_generation_state``'s result when the acceptance launch ran it.
        """
        num_gens = batch_size - num_contexts
        if num_gens == 0:
            return None
        if self._context_kv_only:
            # Shared warmup includes synthetic decode batches on context workers.
            if return_tokens:
                return torch.zeros(
                    (num_gens, self.max_draft_len), dtype=torch.int32, device=accepted_tokens.device
                )
            return None
        captured = spec_metadata.get_hidden_states(total_target_tokens)
        if captured is None:
            raise RuntimeError(
                "DSpark generation requires captured target hidden states; "
                "check target_layer_ids and the target model's capture hooks."
            )
        gen_start = attn_metadata.num_ctx_tokens
        slots = self._batch_to_slot[num_contexts:batch_size]
        if prepared is None:
            prepared = self._advance_unprepared_generation_state(
                spec_metadata,
                slots,
                accepted_tokens[num_contexts:batch_size],
                num_accepted_tokens[num_contexts:batch_size],
                position_ids.reshape(-1),
                gen_start,
                captured.shape[0],
            )
        bonus, main_rows, start_pos, valid_len, interim_rows, interim_pos, interim_valid = prepared
        K = interim_rows.shape[1]
        fork = self._take_prep_fork(
            spec_metadata, gen_start, accepted_tokens.shape[1], num_gens, captured.shape[0]
        )
        tables = self._draft_block_tables[num_contexts:batch_size]
        capacities = self._draft_capacities[num_contexts:batch_size]

        def write_backfill(**projection):
            draft_model.write_context_pages(
                None,
                interim_pos,
                interim_valid,
                self._draft_kv_buffers,
                tables,
                capacities,
                **projection,
            )

        windows_ready = main_x_rows = None
        main_kv_written = False
        if fork is not None:
            main_x, kv = fork
            if kv.shape[1] > K:
                from .dspark_backfill_inputs import main_kv_backfill_inputs

                main = torch.cuda.current_stream()
                side = private_stream(kv.device.index, "dspark_prep_fork")
                # Both streams read this table. Publish it on main before the fork.
                draft_model.dspark_model._dspark_freqs_table(start_pos.device)
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    positions, valid = main_kv_backfill_inputs(
                        start_pos,
                        num_accepted_tokens[num_contexts:batch_size],
                        kv.shape[1],
                    )
                    draft_model.write_context_pages(
                        None, positions, valid, self._draft_kv_buffers, tables, capacities, kv=kv
                    )
                    windows_ready = torch.cuda.Event()
                    windows_ready.record(side)
                for tensor in (
                    num_accepted_tokens,
                    start_pos,
                    kv,
                    tables,
                    capacities,
                ):
                    tensor.record_stream(side)
                main_kv_written = True
                bonus_x = main_x.flatten(0, 1)
                main_x_rows = main_rows
            else:
                write_backfill(kv=kv)
                bonus_x = main_x.flatten(0, 1).index_select(0, main_rows)
        else:
            row_index = torch.cat((interim_rows, main_rows.unsqueeze(1)), dim=1)
            projected = draft_model.project_captured_rows(captured, row_index)
            write_backfill(main_x=projected[:, :K])
            bonus_x = projected[:, K]
        draft_output = draft_model.forward_batched(
            None,
            bonus,
            start_pos,
            kv_windows=self._kv_windows,
            kv_pages=self._draft_kv_buffers,
            block_tables=tables,
            capacities=capacities,
            slots=slots,
            valid_len=valid_len,
            temperature=0.0,
            return_confidence=self.return_confidence,
            return_logits=not return_tokens,
            all_rank_num_tokens=all_rank_num_tokens,
            main_x=bonus_x,
            windows_ready=windows_ready,
            main_kv_written=main_kv_written,
            main_x_rows=main_x_rows,
        )
        confidence = draft_output[1]
        if confidence is not None:
            self._confidence_logits[slots] = confidence.detach()
            self._confidence_stamp[slots] = self._draft_seq_cuda
        return draft_output[0].to(torch.int32) if return_tokens else draft_output[2]
