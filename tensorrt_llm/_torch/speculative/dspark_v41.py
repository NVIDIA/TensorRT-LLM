# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedded DSpark over the CSA2 manager's explicit draft layers."""

import torch

from tensorrt_llm._utils import is_blackwell_datacenter, prefer_pinned
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, ReusePolicy

from ..modules.multi_stream_utils import private_stream
from ..pyexecutor import device_copy
from .dspark import DSparkSpecMetadata, DSv4DSparkWorker


class DSparkCopySite(device_copy.CopySiteEnum):
    PAGE_INDICES = ("dspark.page_indices", device_copy.Direction.H2D)
    PAGE_VALUES = ("dspark.page_values", device_copy.Direction.H2D)
    REQUEST_SLOTS = ("dspark.request_slots", device_copy.Direction.H2D)
    PAGE_CAPACITIES = ("dspark.page_capacities", device_copy.Direction.H2D)


class DSv41DSparkWorker(DSv4DSparkWorker):
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
        self._draft_metadata_host_buffers = []
        self._uploaded_slots = None
        self._uploaded_capacities = None
        self._table_slots = None
        self._draft_updates_device = None
        self._draft_indices_device = None

    def _allocate_context_storage(self, rows: int, stages: int, window: int, head_dim: int) -> None:
        return None

    def _can_backfill_main_kv(self, draft_model) -> bool:
        return (
            is_blackwell_datacenter()
            and bool(self._draft_kv_buffers)
            and not self.mapping.enable_attention_dp
            and draft_model.dspark_model._tp_attention_ready
        )

    def prepare_draft_context(
        self, draft_model, spec_metadata, attn_metadata, resource_manager
    ) -> None:
        if spec_metadata is None:
            return
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
            self._draft_kv_manager = manager
            self._context_kv_only = getattr(draft_model.dspark_model, "context_kv_only", False)
            self._draft_layer_ids = layers
            self._draft_reuse_policy = specs[0].reuse_policy
            self._draft_kv_buffers = tuple(
                manager.get_buffers(layer, kv_layout="HND") for layer in layers
            )
            shape = (self._batch_to_slot.shape[0], manager.max_blocks_per_seq)
            slots = self._ctx_len.numel()
            self._draft_table_storage = torch.zeros(
                (slots + shape[0], shape[1]), dtype=torch.int32, device=self._ctx_len.device
            )
            self._draft_slot_block_tables = self._draft_table_storage[:slots]
            self._draft_fallback_block_tables = self._draft_table_storage[slots:]
            self._draft_block_tables = self._draft_fallback_block_tables
            managed_table = manager.draft_page_table()
            if managed_table is None or managed_table.shape[0] < shape[0]:
                raise ValueError("The manager's draft page table does not cover the worker batch")
            self._draft_managed_block_tables = managed_table[: shape[0]]
            self._draft_capacities = torch.zeros(
                shape[0], dtype=torch.long, device=self._ctx_len.device
            )
            self._draft_page_versions.clear()
            self._draft_page_indices.clear()
            self._draft_prepared_batch = None
            self._managed_residency.clear()
            self._uploaded_slots = None
            self._uploaded_capacities = None
            self._table_slots = None
        spec_metadata._dspark_worker = self
        self._prepare_page_tables(spec_metadata.request_ids, attn_metadata.num_contexts)

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

    def _prepare_page_tables(self, request_ids: list[int], num_contexts: int) -> None:
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
        batch_state = manager.draft_page_table_state() if managed else None
        previous_batch = self._draft_prepared_batch
        if (
            batch_state is not None
            and previous_batch is not None
            and previous_batch[0] is batch_state
            and previous_batch[1] == tuple(request_ids)
        ):
            return
        batch_snapshot = None if batch_state is None else (batch_state, tuple(request_ids))
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
        for rid in self._req_to_slot.keys() - real_ids:
            self._free_slots.append(self._req_to_slot.pop(rid))
            self._managed_residency.pop(rid, None)
        slots = [self._scratch_slot] * len(request_ids)
        updates = []
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
                    table = self._page_indices(cache, row < num_contexts)
                    if self._draft_page_indices.get(slot) != table:
                        updates.append((slot, row, *table))
                        self._draft_page_indices[slot] = table
                    self._draft_page_versions[slot] = version
                first, indices = self._draft_page_indices[slot]
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
                self._position_initialized[slot].fill_(True)
                self._managed_residency[rid] = cache
        # Generation preparation resets scratch state inside the graph before
        # consuming it. Context seeding still needs the eager reset.
        if num_contexts or not private:
            self._ctx_len[self._scratch_slot].zero_()
            self._valid_len[self._scratch_slot].zero_()
        slot_values = tuple(slots)
        capacity_values = tuple(capacities)
        slots_changed = self._uploaded_slots != slot_values
        capacities_changed = self._uploaded_capacities != capacity_values
        if managed:
            self._table_slots = None
        gather_table = not managed and self._table_slots != slot_values
        if not updates and not slots_changed and not capacities_changed and not gather_table:
            self._draft_prepared_batch = batch_snapshot
            return
        direct = private and not native and self._table_slots == slot_values
        count = sum(len(indices) for _, _, _, indices in updates) * (2 if direct else 1)
        host = next(
            (item for item in self._draft_metadata_host_buffers if item["event"].query()), None
        )
        if host is None:
            host = {"event": torch.cuda.Event()}
            self._draft_metadata_host_buffers.append(host)
        for name, size, dtype in (
            ("indices", count, torch.long),
            ("values", count, torch.int32),
            ("slots", self._batch_to_slot.numel(), torch.long),
            ("capacities", len(capacities), torch.long),
        ):
            if name not in host or host[name].numel() < size:
                host[name] = torch.empty(size, dtype=dtype, pin_memory=prefer_pinned())
        if count:
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
                self._draft_updates_device = torch.empty(
                    count, dtype=torch.int32, device=self._ctx_len.device
                )
                self._draft_indices_device = torch.empty(
                    count, dtype=torch.long, device=self._ctx_len.device
                )
            indices_device = self._draft_indices_device[:count]
            values_device = self._draft_updates_device[:count]
            device_copy.copy(DSparkCopySite.PAGE_INDICES, indices_device, host["indices"][:count])
            device_copy.copy(DSparkCopySite.PAGE_VALUES, values_device, host["values"][:count])
            self._draft_table_storage.view(-1).index_copy_(0, indices_device, values_device)
        if slots_changed:
            host_slots = host["slots"].numpy()
            host_slots.fill(self._scratch_slot)
            host_slots[: len(slots)] = slots
            device_copy.copy(DSparkCopySite.REQUEST_SLOTS, self._batch_to_slot, host["slots"])
            self._uploaded_slots = slot_values
        if capacities_changed:
            host["capacities"].numpy()[:] = capacities
            device_copy.copy(
                DSparkCopySite.PAGE_CAPACITIES, self._draft_capacities, host["capacities"]
            )
            self._uploaded_capacities = capacity_values
        if not managed and not direct and (updates or gather_table):
            torch.index_select(
                self._draft_slot_block_tables, 0, self._batch_to_slot, out=self._draft_block_tables
            )
            self._table_slots = slot_values
        host["event"].record()
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
        contexts = []
        context_offset = 0
        for i in range(attn_metadata.num_contexts):
            chunk_len = int(attn_metadata._seq_lens[i])
            capture_len = chunk_len if capture_lens is None else capture_lens[i]
            if not 0 <= capture_len <= chunk_len:
                raise ValueError("DSpark capture length must be a suffix of its context chunk")
            contexts.append((context_offset, chunk_len, capture_len))
            context_offset += chunk_len
        batched = set()
        if private and captured is not None:
            eligible = [
                capture_len >= self._win
                and self._req_to_slot.get(spec_metadata.request_ids[i], self._scratch_slot)
                != self._scratch_slot
                for i, (_, _, capture_len) in enumerate(contexts)
            ]
            first = 0
            while first < len(contexts):
                last = first + 1
                if eligible[first]:
                    while last < min(first + 8, len(contexts)) and eligible[last]:
                        last += 1
                if last - first > 1:
                    ends = [offset + length for offset, length, _ in contexts[first:last]]
                    hidden = torch.stack([captured[end - self._win : end] for end in ends])
                    frames = (
                        torch.stack(
                            [flat_position_ids[end - self._win : end] for end in ends]
                        ).long()
                        + 1
                    )
                    draft_model.write_context_pages(
                        hidden,
                        frames,
                        torch.ones_like(frames, dtype=torch.bool),
                        self._draft_kv_buffers,
                        self._draft_block_tables[first:last],
                        self._draft_capacities[first:last],
                    )
                    slots = self._batch_to_slot[first:last]
                    self._ctx_len.index_copy_(0, slots, frames[:, -1])
                    self._valid_len.index_fill_(0, slots, self._win)
                    batched.update(range(first, last))
                first = last
        for i, (context_offset, chunk_len, capture_len) in enumerate(contexts):
            chunk_positions = flat_position_ids[context_offset : context_offset + chunk_len]
            if chunk_len == 0:
                continue
            req_id = spec_metadata.request_ids[i]
            slot = self._req_to_slot.get(req_id, self._scratch_slot)
            if captured is None and slot != self._scratch_slot:
                raise RuntimeError(
                    "Embedded DSpark requires captured context to initialize KV history"
                )
            if captured is not None and capture_len:
                keep = min(self._win, capture_len)
                count = (
                    capture_len
                    if self._draft_kv_manager.enable_block_reuse and (not private)
                    else keep
                )
                if i not in batched:
                    positions = (chunk_positions[-count:].long() + 1).unsqueeze(0)
                    draft_model.write_context_pages(
                        captured[
                            context_offset + chunk_len - count : context_offset + chunk_len
                        ].unsqueeze(0),
                        positions,
                        torch.ones_like(positions, dtype=torch.bool),
                        self._draft_kv_buffers,
                        self._draft_block_tables[i : i + 1],
                        self._draft_capacities[i : i + 1],
                    )
                    self._valid_len[slot] = torch.clamp(self._valid_len[slot] + keep, max=self._win)
            if i not in batched:
                self._ctx_len[slot] = chunk_positions[-1] + 1

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
            qo_indptr = spec_metadata.qo_indptr
            if spec_metadata.verify_lens is not None and qo_indptr is not None:
                base = gen_start + qo_indptr[:num_gens].to(device=slots.device, dtype=torch.long)
                prepared = self._advance_ragged_generation_state(
                    slots,
                    accepted_tokens[num_contexts:batch_size],
                    num_accepted_tokens[num_contexts:batch_size],
                    position_ids.reshape(-1),
                    base,
                    captured.shape[0],
                )
            else:
                prepared = self._advance_generation_state(
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

        windows_ready = main_kv = None
        if fork is not None:
            main_x, kv = fork
            if kv.shape[1] > K:
                from .dspark_backfill_inputs import main_kv_backfill_inputs

                main = torch.cuda.current_stream()
                side = private_stream(kv.device.index, "dspark_prep_fork")
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    positions, valid, slots32, start32, bonus_x = main_kv_backfill_inputs(
                        start_pos,
                        num_accepted_tokens[num_contexts:batch_size],
                        slots,
                        main_rows,
                        main_x.flatten(0, 1),
                        kv.shape[1],
                    )
                    draft_model.write_context_pages(
                        None, positions, valid, self._draft_kv_buffers, tables, capacities, kv=kv
                    )
                    main_kv = (slots32, start32)
                    windows_ready = torch.cuda.Event()
                    windows_ready.record(side)
                for tensor in (
                    num_accepted_tokens,
                    start_pos,
                    slots,
                    kv,
                    main_x,
                    main_rows,
                    tables,
                    capacities,
                ):
                    tensor.record_stream(side)
                for tensor in (bonus_x, *main_kv):
                    tensor.record_stream(main)
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
            main_kv=main_kv,
        )
        confidence = draft_output[1]
        if confidence is not None:
            self._confidence_logits[slots] = confidence.detach()
            self._confidence_stamp[slots] = self._draft_seq_cuda
        return draft_output[0].to(torch.int32) if return_tokens else draft_output[2]
