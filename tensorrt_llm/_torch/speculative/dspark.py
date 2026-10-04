# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DSpark worker / metadata mirror the DFlash plumbing (capture target-layer
# hidden states, accept the previous block with standard verification, draft a
# new block in one backbone forward), adapted to DSpark's draft model which
# produces the whole block (and its confidence-truncated length) inside a single
# ``DSv4DSparkDraftModel.forward`` rather than via mask-token cross-attention.

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from ..pyexecutor.resource_manager import ResourceManagerType
from .dflash import DFlashWorker, dflash_draft_slot_ids
from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from ...llmapi.llm_args import DSparkDecodingConfig


def _dspark_position_ceiling(max_ctx: int, block_size: int, max_draft_len: int) -> int:
    """Return the number of RoPE entries needed by the DSv4 block drafter.

    ``start_pos`` is a FRAME index, one above the absolute token position: the
    prompt token at position p occupies frame p+1 (``_seed_context_pages``), so a
    request served to ``max_ctx`` bootstraps at ``max_ctx + 1``. A verification then
    accepts up to ``max_draft_len + 1`` tokens and the block drafter indexes
    ``block_size`` further positions, making the largest index
    ``max_ctx + 1 + max_draft_len + 1 + block_size``; the length is one greater.

    The frame +1 is load-bearing: without it the drafter reached start_pos 4112 at
    max_ctx 4105 and max_draft_len 5, one past the table [measured job 3097207].
    """
    return int(max_ctx) + int(max_draft_len) + int(block_size) + 3


@dataclass
class DSparkSpecMetadata(SpecMetadata):
    """Metadata for DSpark speculative decoding.

    Captures hidden states from the target model's ``layers_to_capture`` during
    the target forward pass. DSpark captures the *mean over the multi-head
    (mHC) residual streams* at each captured layer (handled by the target-side
    capture hook), concatenated across layers, and feeds them to the draft
    model's ``main_proj`` + ``main_norm`` (inside ``DSv4DSparkDraftModel.forward``)
    as the captured-context attention input (``main_x``).

    Mirrors :class:`DFlashSpecMetadata`; the only DSpark-specific detail is that
    the per-layer captured width is the model hidden size (post hc-mean), so the
    buffer is ``[max_num_tokens, hidden_size * num_capture_layers]``.
    """

    batch_indices_cuda: Optional[torch.Tensor] = None

    # Hidden state capture fields
    layers_to_capture: Optional[List[int]] = None
    hidden_size: int = 0
    max_num_tokens: int = 0
    dtype: torch.dtype = torch.bfloat16
    captured_hidden_states: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False

        # Set up hidden state capture buffer
        if self.layers_to_capture is not None and len(self.layers_to_capture) > 0:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
            self.num_capture_layers = len(self.layers_to_capture)
            # O(1) lookups for is_layer_capture() and maybe_capture_hidden_states()
            self._capture_layer_set = frozenset(self.layers_to_capture)
            self._layer_to_idx = {lid: i for i, lid in enumerate(self.layers_to_capture)}
            # As in DFlash, graph buckets share full token-budget scratch
            # storage because forwards and consumers use one execution stream.
            expected_shape = (self.max_num_tokens, self.hidden_size * self.num_capture_layers)
            if (
                self.captured_hidden_states is None
                or self.captured_hidden_states.shape != expected_shape
                or self.captured_hidden_states.dtype != self.dtype
                or self.captured_hidden_states.device != self.batch_indices_cuda.device
            ):
                self.captured_hidden_states = torch.empty(
                    expected_shape, dtype=self.dtype, device=self.batch_indices_cuda.device
                )
                logger.info(
                    f"DSpark: capturing hidden states from layers {self.layers_to_capture}, "
                    f"buffer shape {self.captured_hidden_states.shape}"
                )
        else:
            self.num_capture_layers = 0
            self._capture_layer_set = frozenset()
            self._layer_to_idx = {}

    def prepare(self):
        assert self.request_ids is not None
        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(
            num_seqs, dtype=torch.int, device="cpu", pin_memory=prefer_pinned()
        )
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)

    def is_layer_capture(self, layer_id: int) -> bool:
        return layer_id in self._capture_layer_set

    def maybe_capture_hidden_states(
        self, layer_id: int, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> None:
        """Capture hidden states from a target model layer into the buffer.

        DeepSeek-V4 keeps the multi-head (mHC) residual stream flattened as
        ``[num_tokens, hc_mult * hidden]``; DSpark captures the *mean over the hc
        streams* (reference ``h.mean(dim=2)`` with ``h`` shaped
        ``[*, hc_mult, hidden]``). We reduce here so the V4 decoder layer's
        existing capture call is unchanged. A ``[num_tokens, hidden]`` input
        (already reduced / non-mHC) is stored as-is.
        """
        if self.captured_hidden_states is None:
            return
        i = self._layer_to_idx.get(layer_id)
        if i is not None:
            num_tokens = hidden_states.shape[0]
            to_save = hidden_states + residual if residual is not None else hidden_states
            # mHC residual -> mean over the hc_mult streams.
            if to_save.shape[-1] != self.hidden_size:
                hc_mult = to_save.shape[-1] // self.hidden_size
                to_save = to_save.reshape(num_tokens, hc_mult, self.hidden_size).mean(dim=1)
            self.captured_hidden_states[
                :num_tokens, i * self.hidden_size : (i + 1) * self.hidden_size
            ].copy_(to_save, non_blocking=True)

    def get_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Get captured hidden states (all layers concatenated)."""
        if self.captured_hidden_states is None:
            return None
        return self.captured_hidden_states[
            :num_tokens, : self.hidden_size * self.num_capture_layers
        ]


class DSv4DSparkWorker(SpecWorkerBase):
    """Worker for DSpark speculative decoding.

    DSpark drafts a whole block of ``block_size`` tokens in one backbone forward
    (``DSv4DSparkDraftModel.forward``): it projects the captured target-layer hidden
    states (``main_proj`` + ``main_norm``) into the draft's captured-context
    attention, runs the ``num_stages`` DSpark blocks over a captured
    window, refines the per-position logits with the Markov head, and predicts a
    per-position acceptance confidence used to truncate the proposed prefix.

    Attention reads a sliding window of projected captured context directly
    from manager-owned KV pages, alongside the current draft block.
    Acceptance of the previous block goes through the unified
    :meth:`SpecWorkerBase.sample_and_accept_draft_tokens` (strict target-verify,
    or rejection sampling for a non-greedy batch), so greedy parity with no-spec
    is preserved regardless of draft quality.

    The context window is kept consistent across the whole decode: it is seeded
    from the prompt's captured context at prefill and back-filled with the
    intermediate accepted tokens of a multi-accept step (both via
    ``DSv4DSparkDraftModel.write_context_pages``), in addition to the per-step bonus
    write done by the generation path. These affect draft acceptance rate only,
    not correctness, which the standard target verify guarantees.

    Naming: workers are classified by *deployment form*, not by draft
    backbone (see :class:`DSparkWorker`). This one is form-specific
    because it tracks captured-context history and drives the draft
    through attributes only an embedded DeepSeek-V4-Pro draft has --
    ``num_stages``, ``_attn_params``, ``write_context_pages`` and
    ``forward``. A standalone
    drafter has none of them and is served by :class:`DSparkWorker`.

    Reference: DeepSeek DeepSpec (https://github.com/deepseek-ai/DeepSpec).
    """

    def __init__(
        self,
        spec_config: "DSparkDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping

        # Request progress is slot-indexed; managed KV lives only in its pages.
        self._ctx_len: Optional[torch.Tensor] = None  # [max_batch] abs decode position
        self._valid_len: Optional[torch.Tensor] = None  # [max_batch] written window entries
        self._win = 0
        self._draft_kv_manager = None
        self._draft_kv_buffers = ()
        self._draft_block_tables = None
        self._draft_capacities = None
        self._managed_residency = {}
        # Set in _lazy_init from the RoPE table the drafter will build; None
        # leaves positions unbounded (direct construction in tests).
        self._position_cap: Optional[int] = None

        # Slot management. ``_req_to_slot`` (python dict) + ``_free_slots`` are the
        # source of truth, updated in prepare()/forward(); ``_batch_to_slot`` is the
        # CUDA mirror (request-order -> slot) read by the CUDA-graph-safe batched
        # gen path (set on the host in prepare(), so the captured forward indexes
        # request progress through a tensor instead of a python dict lookup).
        self._req_to_slot = {}  # request_id -> slot index
        self._free_slots = deque()  # available slot indices
        self._batch_to_slot: Optional[torch.Tensor] = None  # [max_batch] long, cuda
        # Index of the throwaway "scratch" progress row that absorbs padded /
        # unknown request IDs (set in ``_lazy_init`` to ``max_batch``); it is
        # never handed out through ``_free_slots``.
        self._scratch_slot = 0

        # The generation draft path is the batched, host-sync-free
        # ``_draft_gen_block_batched`` + ``DSv4DSparkDraftModel.forward`` +
        # ``dspark_attention_forward``: it is correct in eager mode AND safe
        # to capture into the target's CUDA graph (DSpark is a one-engine drafter —
        # its worker forward runs inside that graph, so the draft path MUST be
        # capture-safe whenever ``cuda_graph_config`` is set).

        logger.info(
            f"DSv4DSparkWorker initialized with "
            f"use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    def _publish_position_ceiling(self, draft_model, attn_metadata, block_size: int) -> None:
        """Size the drafter's RoPE table from the runtime bound, and bound positions by it.

        Refreshes the bound for every batch. The first forward can
        land on a KV-estimation probe manager whose ``max_seq_len`` is below the real
        one, and a table pinned to that probe would index out of range for the rest
        of the process; ``DFlashWorker._lazy_init_ctx_buffers`` re-publishes for that
        reason. Grows only, and the table cache is keyed on the cap.

        Two bounds exist and neither dominates: ``attn_metadata`` carries the KV
        manager's ``max_seq_len`` while ``_freqs_cap`` carries the user's, and
        ``_create_cuda_graph_warmup_request`` sizes its dummy request at whichever is
        larger. Covering both is what keeps warmup off the end of the table.
        """
        inner_model = getattr(draft_model, "dspark_model", None) or draft_model
        config_cap = int(getattr(inner_model, "_freqs_cap", 0) or 0)
        # _freqs_cap is max_seq_len + block_size + 2; undo that so the config
        # bound and the KV manager's go through one formula.
        config_ctx = max(0, config_cap - block_size - 2)
        max_ctx = getattr(attn_metadata, "max_seq_len", None)
        if max_ctx is not None:
            ceiling = max(
                _dspark_position_ceiling(
                    max(int(max_ctx), config_ctx), block_size, self.max_draft_len
                ),
                int(getattr(inner_model, "_runtime_position_ceiling", 0) or 0),
            )
            draft_model._runtime_position_ceiling = ceiling
            if inner_model is not draft_model:
                inner_model._runtime_position_ceiling = ceiling
        # Largest absolute position the block drafter may hold: forward
        # gathers ``freqs[start_pos + block_size]`` and the interim back-fill
        # ``freqs[old + block_size]``, so keep one block of headroom.
        table_len = int(getattr(inner_model, "_runtime_position_ceiling", 0) or config_cap)
        self._position_cap = (table_len - 1 - block_size) if table_len else None

    def _lazy_init(self, draft_model, spec_metadata, attn_metadata=None) -> None:
        block_size = int(draft_model.block_size)
        if block_size != self.max_draft_len:
            raise ValueError(
                "DSpark draft model block_size must equal worker max_draft_len; "
                f"got block_size={block_size} and max_draft_len={self.max_draft_len}"
            )

        self._publish_position_ceiling(draft_model, attn_metadata, block_size)

        if self._ctx_len is None:
            max_batch = spec_metadata.max_num_requests
            self._num_stages = draft_model.num_stages
            self._win = int(draft_model._attn_params["window_size"])
            self._head_dim = int(draft_model._attn_params["head_dim"])

            # Padding requests share an extra progress slot, never a real request slot.
            self._scratch_slot = max_batch
            num_rows = max_batch + 1

            # CUDA-graph padding requests carry ids in
            # ``[CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len, CUDA_GRAPH_DUMMY_REQUEST_ID]``,
            # while real request ids start at ``max_batch_size`` and grow, so a simple
            # floor cleanly separates them. Together with ``ATTENTION_DP_DUMMY_REQUEST_ID``
            # (0) these dummies must route to the scratch row (see ``prepare()``) and
            # never consume a real slot. Imported lazily to break the
            # dspark -> cuda_graph_runner -> speculative.utils -> dspark import cycle.
            from ..pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

            self._graph_dummy_id_floor = CUDA_GRAPH_DUMMY_REQUEST_ID - self.max_draft_len

            self._ctx_len = torch.zeros(num_rows, dtype=torch.long, device="cuda")
            self._valid_len = torch.zeros(num_rows, dtype=torch.long, device="cuda")
            self._batch_to_slot = torch.full(
                (max_batch,), self._scratch_slot, dtype=torch.long, device="cuda"
            )
            self._free_slots = deque(range(max_batch))
            self._req_to_slot = {}

    def _assign_slot(self, req_id: int) -> int:
        """Get the persistent progress slot for a request."""
        if req_id not in self._req_to_slot:
            if not self._free_slots:
                raise RuntimeError(
                    "DSpark has no free progress slots for request "
                    f"{req_id}; increase max_num_requests"
                )
            slot = self._free_slots.popleft()
            self._req_to_slot[req_id] = slot
            self._ctx_len[slot] = 0
            self._valid_len[slot] = 0
        return self._req_to_slot[req_id]

    def _release_inactive_slots(self, request_ids: list[int]) -> None:
        current = set(request_ids)
        for request_id in list(self._req_to_slot):
            if request_id not in current:
                slot = self._req_to_slot.pop(request_id)
                self._managed_residency.pop(request_id, None)
                self._ctx_len[slot] = 0
                self._valid_len[slot] = 0
                self._free_slots.append(slot)

    def _is_managed_request(self, request_id: int) -> bool:
        return (
            request_id != ATTENTION_DP_DUMMY_REQUEST_ID
            and request_id < self._graph_dummy_id_floor
            and request_id not in self._draft_kv_manager._draft_dummy_request_ids
        )

    def _bind_managed_history(self, resource_manager) -> None:
        manager = (
            resource_manager.get_resource_manager(ResourceManagerType.KV_CACHE_MANAGER)
            if resource_manager is not None
            else None
        )
        if not getattr(manager, "draft_layer_ids", ()):
            raise ValueError("Embedded DSpark requires manager-owned draft KV pages")
        if manager is self._draft_kv_manager:
            return
        if (
            manager.draft_window_size != self._win
            or len(manager.draft_layer_ids) != self._num_stages
            or any(
                manager.kv_factor_per_layer[manager.layer_offsets[layer_id]] != 1
                or manager.head_dim_per_layer[manager.layer_offsets[layer_id]] != self._head_dim
                for layer_id in manager.draft_layer_ids
            )
        ):
            raise ValueError("Embedded DSpark draft cache does not match its attention dimensions")
        buffers = tuple(
            manager.get_draft_buffers(stage) for stage in range(len(manager.draft_layer_ids))
        )
        self._draft_kv_manager = manager
        self._draft_kv_buffers = buffers
        self._managed_residency.clear()
        self._draft_block_tables = torch.zeros(
            (self._batch_to_slot.shape[0], manager.draft_max_blocks_per_seq),
            dtype=torch.int32,
            device=self._ctx_len.device,
        )
        self._draft_capacities = torch.zeros(
            self._batch_to_slot.shape[0], dtype=torch.long, device=self._ctx_len.device
        )

    def prepare_managed_draft_cache(
        self, draft_model, spec_metadata, attn_metadata, resource_manager
    ) -> None:
        """Refresh persistent draft inputs before eager execution or graph replay."""
        self._lazy_init(draft_model, spec_metadata, attn_metadata)
        self._bind_managed_history(resource_manager)
        self._prepare_managed_history(spec_metadata.request_ids, attn_metadata.num_contexts)

    def _prepare_managed_history(self, request_ids: list[int], num_contexts: int) -> None:
        """Bind live page tables and initialize progress for newly resident histories."""
        # Startup probes and graph/ADP padding never publish synthetic history.
        real_rows = [
            row
            for row, request_id in enumerate(request_ids)
            if self._is_managed_request(request_id)
        ]
        real_ids = [request_ids[row] for row in real_rows]
        self._release_inactive_slots(real_ids)
        table = self._draft_kv_manager.get_draft_block_table(real_ids)
        tables_host = torch.zeros_like(self._draft_block_tables, device="cpu")
        tables_host[real_rows] = table
        self._draft_block_tables.copy_(tables_host, non_blocking=True)
        capacities_host = torch.zeros_like(self._draft_capacities, device="cpu")
        capacities_host[real_rows] = torch.tensor(
            [self._draft_kv_manager.kv_cache_map[rid].capacity for rid in real_ids],
            dtype=torch.long,
        )
        self._draft_capacities.copy_(capacities_host, non_blocking=True)
        self._ctx_len[self._scratch_slot] = 0
        self._valid_len[self._scratch_slot] = 0
        batch_slots = [self._scratch_slot] * len(request_ids)
        for row in real_rows:
            request_id = request_ids[row]
            cache = self._draft_kv_manager.kv_cache_map[request_id]
            slot = self._assign_slot(request_id)
            batch_slots[row] = slot
            if self._managed_residency.get(request_id) is cache:
                continue
            position = cache.history_length
            if row >= num_contexts and position == 0:
                raise ValueError(
                    f"Embedded DSpark generation request {request_id} has no committed KV history"
                )
            self._ctx_len[slot] = position
            self._valid_len[slot] = min(self._win, position)
            self._managed_residency[request_id] = cache
        self._batch_to_slot.fill_(self._scratch_slot)
        self._batch_to_slot[: len(request_ids)].copy_(
            torch.tensor(batch_slots, dtype=torch.long, device=self._batch_to_slot.device)
        )

    def _seed_context_pages(
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
        flat_position_ids = position_ids.reshape(-1)
        context_offset = 0
        for i in range(attn_metadata.num_contexts):
            chunk_len = int(attn_metadata._seq_lens[i])
            chunk_positions = flat_position_ids[context_offset : context_offset + chunk_len].long()
            if chunk_len == 0:
                context_offset += chunk_len
                continue

            req_id = spec_metadata.request_ids[i]
            slot = self._req_to_slot.get(req_id, self._scratch_slot)
            if captured is None and slot != self._scratch_slot:
                raise RuntimeError(
                    "Embedded DSpark requires captured context to initialize KV history"
                )
            if captured is not None:
                keep = min(self._win, chunk_len)
                # Prefix reuse can publish any complete page in this chunk.
                count = chunk_len if self._draft_kv_manager.enable_block_reuse else keep
                positions = (chunk_positions[-count:] + 1).unsqueeze(0)
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
            self._ctx_len[slot] = chunk_positions[-1] + 1
            context_offset += chunk_len

    def _advance_generation_state(
        self,
        slots: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        input_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bootstrap and advance per-slot decode state without host synchronization."""
        # Reset inside the captured region: many padding / ADP-idle rows share this
        # one scratch slot within a single forward, so its position has to restart
        # from ``input_positions`` rather than accumulate across capture shapes.
        scratch = self._scratch_slot
        self._ctx_len[scratch].zero_()
        self._valid_len[scratch].zero_()
        old = torch.where(slots == scratch, input_positions, self._ctx_len[slots])
        start_pos = old + num_accepted_tokens
        # Keep dummy graph positions within the RoPE table.
        if self._position_cap is not None:
            old = torch.clamp(old, max=self._position_cap)
            start_pos = torch.clamp(start_pos, max=self._position_cap)
        self._ctx_len[slots] = start_pos
        self._valid_len[slots] = torch.clamp(
            self._valid_len[slots] + num_accepted_tokens, max=self._win
        )
        return old, start_pos

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
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """CUDA-graph-safe batched gen draft (all gen requests in one forward).

        Free of host syncs and data-dependent shapes: per-request quantities
        (``nacc``, the bonus, ``main_hidden``, ``start_pos``, the multi-accept
        back-fill) are gathered as tensors, slots come from the host-built
        ``_batch_to_slot`` mirror, and the backbone runs once via
        ``DSv4DSparkDraftModel.forward``. Returns the per-position corrected
        block logits ``[num_gens, K, vocab]`` (or ``None`` when there is nothing to
        draft); the worker feeds them to ``SpecWorkerBase.sample_draft_tokens``.
        Confidence truncation stays disabled — the full block is proposed.
        """
        num_gens = batch_size - num_contexts
        K = self.max_draft_len
        device = accepted_tokens.device

        if num_gens == 0:
            return None
        captured = spec_metadata.get_hidden_states(total_target_tokens)
        if captured is None:
            raise RuntimeError("Embedded DSpark requires captured context to advance KV history")

        # gen-only graph batches have num_ctx_tokens == 0; mixed eager batches put
        # the gen tokens after the context tokens.
        gen_start = attn_metadata.num_ctx_tokens
        slots = self._batch_to_slot[num_contexts:batch_size]  # [G]
        # Bootstrap iterations can process one target token per request, while
        # normal speculative verification processes K+1. Use the actual accepted
        # row width to index both captured hidden states and position IDs.
        target_width = accepted_tokens.shape[1]
        nacc = num_accepted_tokens[num_contexts:batch_size].long()  # [G]
        gidx = nacc - 1  # [G] index of the bonus within each verified prefix

        # Bonus token = last accepted token of the verified prefix.
        bonus = (
            accepted_tokens[num_contexts:batch_size].gather(1, gidx.unsqueeze(1)).squeeze(1).long()
        )  # [G]

        arange_g = torch.arange(num_gens, device=device)
        base = gen_start + arange_g * target_width  # [G]
        main_hidden = captured[base + gidx]  # [G, ncap*hidden]

        # Back-fill intermediate accepted tokens (everything but the bonus) at
        # frames old+1 .. old+nacc-1. Fixed [G, K] tensors keep this capture-safe;
        # j >= nacc-1 is masked before any page store.
        input_positions = position_ids.reshape(-1)[base].long()
        old, start_pos = self._advance_generation_state(slots, nacc, input_positions)
        j = torch.arange(K, device=device)  # [K]
        interim_valid = j.unsqueeze(0) < (nacc.unsqueeze(1) - 1)  # [G, K]
        interim_pos = old.unsqueeze(1) + 1 + j.unsqueeze(0)  # [G, K]
        interim_base = (base.unsqueeze(1) + j.unsqueeze(0)).clamp(
            min=0, max=captured.shape[0] - 1
        )  # [G, K] (clamped; invalid entries are masked out anyway)
        interim_hidden = captured[interim_base]  # [G, K, ncap*hidden]
        tables = self._draft_block_tables[num_contexts:batch_size]
        capacities = self._draft_capacities[num_contexts:batch_size]
        draft_model.write_context_pages(
            interim_hidden, interim_pos, interim_valid, self._draft_kv_buffers, tables, capacities
        )

        # Surface the per-position corrected block logits ([num_gens, K, vocab])
        # and let SpecWorkerBase.sample_draft_tokens do the (greedy or rejection)
        # sampling + TP gather + draft_probs scatter, rather than argmaxing here.
        _toks, _num_proposed, block_logits = draft_model.forward(
            main_hidden,
            bonus,
            start_pos,
            kv_pages=self._draft_kv_buffers,
            block_tables=tables,
            capacities=capacities,
            valid_len=self._valid_len[slots],
            temperature=0.0,
            confidence_threshold=0.0,
            return_logits=True,
            all_rank_num_tokens=all_rank_num_tokens,
        )
        return block_logits

    def _sample_draft_tokens_guided(
        self,
        gen_logits: torch.Tensor,
        spec_metadata: "DSparkSpecMetadata",
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        K: int,
    ):
        """
        Grammar-constrained draft sampling for the guided-decoding path.
        """
        vocab = gen_logits.shape[-1]
        # Lay the block out step-major ([K, batch, vocab]) so each step's slice is
        # a contiguous [batch, vocab] tensor.
        if num_contexts > 0:
            full_logits = gen_logits.new_zeros((K, batch_size, vocab))
            full_logits[:, num_contexts:, :] = gen_logits.transpose(0, 1)
        else:
            full_logits = gen_logits.transpose(0, 1).contiguous()

        gidx = (num_accepted_tokens - 1).clamp(min=0).unsqueeze(1).long()
        new_tokens = accepted_tokens.gather(1, gidx).squeeze(1).to(torch.int32)

        gen_draft_tokens = []
        for k in range(K):
            self.guided_decoder.add_draft_batch(new_tokens, num_accepted_tokens, draft_step=k)
            step_logits = full_logits[k]
            self.guided_decoder.execute_draft_batch(step_logits, draft_step=k)
            step_tokens = self.sample_draft_tokens(
                step_logits, spec_metadata, batch_size, draft_step=k
            )
            gen_draft_tokens.append(step_tokens[num_contexts:])
            new_tokens = step_tokens
        gen_draft_tokens = torch.stack(gen_draft_tokens, dim=1)
        return gen_draft_tokens

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        raw_logits = logits
        K = self.max_draft_len

        if self._draft_kv_manager is None:
            raise RuntimeError("DSpark KV pages must be prepared before worker forward")

        self._execute_guided_decoder_if_present(logits)

        # Target-verify acceptance via the unified SpecWorkerBase entry: it
        # reshapes the stored draft tokens (default (num_gens, runtime_draft_len)
        # hook), then routes to strict or rejection sampling. Greedy parity with
        # the previous hand-rolled path is preserved (rejection only engages for a
        # non-greedy batch with valid draft_probs).
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        total_target_tokens = input_ids.shape[0]

        if num_contexts > 0:
            self._seed_context_pages(
                draft_model,
                spec_metadata,
                attn_metadata,
                position_ids,
                total_target_tokens,
            )

        # FUSED_COMM MoE backends (DeepGEMM MegaMoE) synchronize EP ranks with an
        # in-kernel phase-flip NVLink barrier that flips on every kernel call, so
        # every rank must invoke the draft MoE the same number of times and with
        # the same globally-gathered per-rank token list, or the barrier desyncs
        # (hang / "unspecified launch failure"). The draft runs over generation
        # requests only, each expanded to ``block`` positions, so the per-rank
        # draft-MoE token count is ``num_gens * block``. ``all_rank_num_gens`` is
        # gathered at metadata-prep time (model_engine, outside any CUDA-graph
        # capture region); it is None for non-ADP / single-rank runs, where the
        # local ``[num_tokens]`` fallback in ``_forward_stage`` is correct.
        block = int(draft_model.block_size)
        all_rank_num_gens = getattr(spec_metadata, "all_rank_num_gens", None)
        # A rank with zero local gen requests still has to cross the draft MoE's
        # cross-rank barrier, but DeepseekV4MoE's router / shared-expert dense
        # GEMMs reject a 0-row input (cuBLAS CUBLAS_STATUS_INVALID_VALUE), so such
        # a rank runs a single 1-row dummy through the MoE (like ADP padding).
        # Encode that as ``1`` in the globally-shared per-rank token list so every
        # rank agrees on the FUSED_COMM chunk count and per-rank slice.
        all_rank_draft_tokens = (
            [max(1, int(g) * block) for g in all_rank_num_gens]
            if all_rank_num_gens is not None
            else None
        )
        global_has_gen = (
            max(all_rank_num_gens) > 0 if all_rank_num_gens is not None else num_gens > 0
        )

        if num_gens > 0:
            # The batched gen-block draft returns the per-position corrected block
            # logits [num_gens, K, vocab] and is CUDA-graph-safe.
            gen_logits = self._draft_gen_block_batched(
                draft_model,
                spec_metadata,
                attn_metadata,
                accepted_tokens,
                num_accepted_tokens,
                num_contexts,
                batch_size,
                total_target_tokens,
                position_ids,
                all_rank_num_tokens=all_rank_draft_tokens,
            )
            if gen_logits is not None:
                if self.guided_decoder is not None:
                    gen_draft_tokens = self._sample_draft_tokens_guided(
                        gen_logits,
                        spec_metadata,
                        accepted_tokens,
                        num_accepted_tokens,
                        num_contexts,
                        batch_size,
                        K,
                    )
                else:
                    # SpecWorkerBase samples the draft tokens.
                    gen_draft_tokens = self.sample_draft_tokens(
                        gen_logits, spec_metadata, batch_size, num_contexts=num_contexts
                    )
                # The context one-hot must match the width the gen scatter just
                # published to draft_probs, NOT gen_logits.shape[-1]: under TP the
                # draft logits are vocab-sharded and sample_draft_tokens gathers
                # them to full vocab before scattering, so the pre-gather shard
                # width would leave stale columns and corrupt rejection.
                gen_vocab = spec_metadata.draft_probs_last_dim
            else:
                gen_draft_tokens = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")
                gen_vocab = None
        else:
            # No local generation requests: if any peer EP rank has some, we must
            # still cross the draft MoE's cross-rank barrier the same number of
            # times (zero-token) so a FUSED_COMM phase-flip barrier stays lockstep.
            if global_has_gen:
                draft_model.run_moe_lockstep_noop(all_rank_draft_tokens, accepted_tokens.device)
            gen_draft_tokens = torch.empty((0, K), dtype=torch.int32, device="cuda")
            gen_vocab = None

        # Context requests are not drafted by the block worker (zero placeholder
        # token); fill their draft-prob slot rows with a legal one-hot so they are
        # a valid distribution when they become gen requests next iteration.
        self.write_context_onehot_draft_probs(spec_metadata, num_contexts, num_gens, K, gen_vocab)

        if num_contexts > 0:
            ctx_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        else:
            next_draft_tokens = gen_draft_tokens

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }


class DSparkWorker(DFlashWorker):
    """Worker for a *standalone* DSpark drafter (DFlash lineage).

    DSpark is DFlash plus two extra heads, so the drafting plumbing is
    inherited wholesale from :class:`DFlashWorker` -- paged context K/V,
    slot management, the mask-token block forward -- and only the two
    head-driven policies are overridden here: the block-output slot
    convention (``shift_label``) and the Markov intra-block logit bias.

    Mirrors the model side, where ``GQADSparkForCausalLM`` extends
    ``DFlashForCausalLM`` with the same two heads.

    Naming: this is the unqualified DSpark worker because a separately
    shipped drafter is the ordinary case; :class:`DSv4DSparkWorker` carries
    the qualifier because a draft embedded in the target checkpoint is the
    special one. Workers are classified by *deployment form*, never by draft
    backbone -- so there is no ``Qwen3DSparkWorker``. Note the name meant the
    embedded worker before this split; both the rebind and the rename to
    ``DSv4DSparkWorker`` land in one commit so the swap reads as a unit.

    A worker is agnostic to the draft backbone: everything backbone-shaped is
    supplied by the draft model, which reports its own shapes
    (``_num_attn_layers``, ``_num_heads``, ``_num_kv_heads``, ``_head_dim``)
    and owns the operators (``_build_fused_kv_buffers``,
    ``precompute_context_kv``, ``dflash_forward``,
    ``apply_markov_chain_logits``, ``project_target_hidden``). The worker only
    allocates against the reported shapes and sequences the calls. An MLA
    drafter therefore reuses this class unchanged; its differences (fused-QKV
    assumptions, a 576-latent K/V layout) land in its own draft-model
    subclass. Naming workers by backbone would produce N classes with
    identical bodies.

    Embedded DSpark uses :class:`DSv4DSparkWorker` to manage captured-context
    progress and the draft block forward.
    """

    def set_draft_model(self, draft_model) -> None:
        """Reject an unsupported vocab mapping here rather than mid-decode.

        ``d2t`` is model-static, so a config mistake should surface at load and
        not as a ``NotImplementedError`` raised per decode step, possibly during
        CUDA-graph capture.
        """
        super().set_draft_model(draft_model)
        if self._d2t is not None and getattr(draft_model, "has_markov_head", False):
            raise NotImplementedError(
                "DSpark Markov head requires a shared draft/target vocab "
                "(d2t vocab mapping is not supported); drafter "
                f"{type(draft_model).__name__} declares one."
            )

    def _draft_block_width(self, draft_model) -> int:
        """Block width under the dspark ``shift_label`` convention.

        shift_label reads slots 0..K-1, so K draft tokens fit in K slots and
        the base class' K+1 over-demands by one -- enough to reject a block-7
        checkpoint at max_draft_len=7, which is how both published DSpark
        drafters are meant to run.
        """
        if getattr(draft_model, "_dspark_shift_label", False):
            return self.max_draft_len
        return super()._draft_block_width(draft_model)

    def _draft_slot_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int
    ) -> torch.Tensor:
        """Block-output slots under the dspark ``shift_label`` convention.

        The drafter checkpoint declares the convention, so it is read off the
        draft model rather than assumed: a DSpark drafter trained with the
        legacy DFlash slot layout keeps the base class' slots 1..K.
        """
        shift_label = getattr(draft_model, "_dspark_shift_label", False)
        return dflash_draft_slot_ids(
            num_gens, block_size, num_draft_tokens, shift_label, device="cuda"
        )

    def _refine_block_logits(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        inputs: dict,
        spec_metadata,
    ) -> torch.Tensor:
        """Add the greedy-chained Markov intra-block bias to the block logits.

        A DSpark drafter checkpoint may omit the Markov head (``markov_rank``
        0), which loads as a drafter without one; that case falls through to
        the unmodified backbone logits.
        """
        if not getattr(draft_model, "has_markov_head", False):
            return gen_logits
        return self._apply_dspark_markov_bias(
            draft_model, gen_logits, inputs["first_prev_tokens"], spec_metadata
        )

    def _apply_dspark_markov_bias(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        spec_metadata,
    ) -> torch.Tensor:
        """Apply the dspark vanilla-Markov intra-block bias to block logits.

        Reference (DeepSpec VanillaMarkov.sample_block_tokens, temperature 0):
        step i adds bias = markov_w2 @ markov_w1[prev_i] to the shared-lm_head
        logits, where prev_0 is the anchor (last accepted) token and prev_{i>0}
        is the greedy token from step i-1's biased logits. Greedy per-position
        argmax of the returned logits therefore reproduces the reference
        sampled chain; the rejection-sampling path samples from the same
        biased distributions (proposal conditioned on the greedy chain).

        Handles a TP vocab-sharded draft lm_head by slicing markov_w2's rows
        to this rank's contiguous shard and chaining through the TP-aware
        global argmax.
        """
        # The d2t guard lives in set_draft_model: it is model-static, so raising
        # it here would surface a load-time config error per decode step.
        # Unlike the d2t guard this one cannot move to set_draft_model: it
        # keys on the runtime logits width, and reproducing that at init would
        # duplicate the draft head's sharding rules. A standalone drafter
        # borrows the target lm_head, whose gather_output defaults to True, so
        # the logits normally arrive full-vocab and this branch is skipped.
        full_vocab = draft_model.markov_w2.shape[0]
        shard = gen_logits.shape[-1]
        vocab_slice = None
        if shard != full_vocab:
            mapping = self.mapping
            if (
                mapping is None
                or getattr(mapping, "enable_attention_dp", False)
                or shard * mapping.tp_size != full_vocab
            ):
                raise NotImplementedError(
                    f"DSpark Markov head: draft logits width {shard} does not "
                    f"match the drafter vocab {full_vocab} and is not a plain "
                    "TP column shard of it."
                )
            vocab_slice = slice(mapping.tp_rank * shard, (mapping.tp_rank + 1) * shard)

        def argmax_fn(step_logits):
            # Full-vocab token ids (TP-aware when sharded); tokens stay in
            # draft-vocab space, which is what markov_w1 indexes.
            return self.greedy_sample_draft_with_tp_gather(step_logits, spec_metadata).long()

        return draft_model.apply_markov_chain_logits(
            gen_logits,
            first_prev_tokens,
            argmax_fn=argmax_fn,
            vocab_slice=vocab_slice,
        )
