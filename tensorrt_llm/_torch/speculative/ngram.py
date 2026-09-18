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
"""
NGram (prompt lookup) speculative decoding that runs inside the target model forward.

Every request's token history lives in a GPU slot pool owned by ``NGramPoolManager``. After the target
model has verified the previous drafts, ``NGramWorker`` launches one fused kernel that appends the
accepted tokens to each history and proposes the tokens that followed the longest earlier occurrence of
the sequence's suffix (up to ``max_matching_ngram_size`` tokens). Nothing about the lookup touches the
host, so the whole step is CUDA graph capturable and compatible with the overlap scheduler.

Key components:
- NGramPoolManager: resource manager owning the per-request token histories
- NGramSpecMetadata: per-batch slot ids and row masks handed to the kernel
- NGramWorker: spec worker that verifies drafts and produces the next ones
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from ..pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager, SlotManager
from ..pyexecutor.scheduler import ScheduledRequests
from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from tensorrt_llm.llmapi import NGramDecodingConfig


class NGramPoolManager(BaseResourceManager):
    """
    GPU token-history pool for one-model NGram drafting.

    Each admitted request owns one slot holding its full token sequence (prompt plus every token the
    target model accepted so far). The prompt is seeded from the host when the request enters its first
    context chunk; from then on the history is only ever extended on the device, by the kernel that
    ``NGramWorker`` launches inside the forward. Rows the kernel must leave alone (CUDA graph padding
    dummies and context chunks that are not the last one, whose sampled token is discarded) are masked
    out per iteration.

    Args:
        spec_config: NGram decoding configuration.
        max_num_requests: Maximum batch size.
        max_seq_len: Longest sequence a request can reach; sizes each history row.
        num_seq_slots: Size of the executor's sequence-slot pool when it is larger than
            ``max_num_requests`` (e.g. overlap headroom). Defaults to ``max_num_requests``.
    """

    def __init__(
        self,
        spec_config: "NGramDecodingConfig",
        max_num_requests: int,
        max_seq_len: int,
        num_seq_slots: Optional[int] = None,
    ) -> None:
        self.max_num_requests = max_num_requests
        self.max_seq_len = max_seq_len
        self.max_draft_len = spec_config.max_draft_len
        self.max_matching_ngram_size = spec_config.max_matching_ngram_size
        self.is_use_oldest = spec_config.is_use_oldest
        self.is_public_pool = spec_config.is_public_pool

        self.pool_size = max(num_seq_slots or 0, max_num_requests)
        # Padding dummies share one reserved slot past the pool. It is never seeded or extended, so it
        # stays empty and the public-pool search skips it.
        self.dummy_slot = self.pool_size
        self.slot_manager = SlotManager(self.pool_size)

        num_rows = self.pool_size + 1
        self.history_tokens = torch.zeros((num_rows, max_seq_len),
                                          dtype=torch.int32,
                                          device="cuda")
        self.history_lens = torch.zeros((num_rows, ),
                                        dtype=torch.int32,
                                        device="cuda")
        # Flat so that a [batch, runtime_draft_len] view stays contiguous for any runtime draft length.
        self._draft_tokens_flat = torch.zeros(
            (max_num_requests * self.max_draft_len, ),
            dtype=torch.int32,
            device="cuda")
        self.match_lens = torch.zeros((max_num_requests, ),
                                      dtype=torch.int32,
                                      device="cuda")

        self._dummy_request_ids: set[int] = set()
        self._seeded_request_ids: set[int] = set()
        # Context requests scheduled for a chunk that is not their last one this iteration.
        self._skip_extend_request_ids: set[int] = set()

        logger.info(
            f"NGram pool: {self.pool_size} slots x {max_seq_len} tokens "
            f"({self.history_tokens.numel() * self.history_tokens.element_size() / 1024 / 1024:.1f} MB)"
        )

    # --- BaseResourceManager interface ---

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0

    def prepare_resources(self, scheduled_batch: ScheduledRequests) -> None:
        """Seed the histories of newly admitted requests and record which rows must not be extended."""
        self._skip_extend_request_ids = {
            req.request_id
            for req in scheduled_batch.context_requests_chunking
        }
        for req in scheduled_batch.context_requests:
            # Disaggregated serving routes DISAGG_GENERATION_INIT requests through here as context
            # requests before the context server's first generated token has been appended. Defer to
            # the generation loop below, which sees the full history.
            if req.is_generation_only_request:
                continue
            if req.is_first_context_chunk and req.request_id not in self._seeded_request_ids:
                self._seed_history(req)
        for req in scheduled_batch.generation_requests:
            if (req.is_generation_only_request and not req.is_dummy
                    and req.request_id not in self._seeded_request_ids):
                self._seed_history(req)

    def update_resources(self, scheduled_batch: ScheduledRequests) -> None:
        pass

    def free_resources(self, request: LlmRequest) -> None:
        request_id = request.request_id
        if request_id in self._dummy_request_ids:
            self._dummy_request_ids.discard(request_id)
            return
        slot = self.slot_manager.get_slot(request_id)
        if slot is None:
            return
        # Stream ordered after any forward that still references the slot, so the history is cleared
        # before a new request is seeded into it and before the public-pool search can see it.
        self.history_lens[slot].fill_(0)
        self.slot_manager.remove_slot(request_id)
        self._seeded_request_ids.discard(request_id)

    def add_dummy_requests(self, request_ids: list[int]) -> None:
        """Register CUDA graph padding dummies; they map to the reserved dummy slot and are masked."""
        self._dummy_request_ids.update(request_ids)

    def shutdown(self) -> None:
        self.slot_manager.shutdown()
        self._dummy_request_ids.clear()
        self._seeded_request_ids.clear()
        self._skip_extend_request_ids.clear()

    # --- Per-iteration hooks used by NGramSpecMetadata / NGramWorker ---

    def prepare(self, request_ids: list[int], slot_ids_cuda: torch.Tensor,
                row_mask_cuda: torch.Tensor) -> None:
        """Fill the batch's slot ids and row mask. Runs outside the CUDA graph, before the forward.

        Args:
            request_ids: Request id of every batch row, in batch order.
            slot_ids_cuda: [max_num_requests] int32 device buffer receiving the history slot per row.
            row_mask_cuda: [max_num_requests] int32 device buffer receiving 1 for rows the kernel
                extends and drafts, 0 for rows it must skip.
        """
        num_rows = len(request_ids)
        slots = []
        mask = []
        for request_id in request_ids:
            slot = self.slot_manager.get_slot(request_id)
            if slot is None:
                if request_id not in self._dummy_request_ids:
                    logger.debug(
                        f"NGram: request {request_id} has no history slot; drafting skipped"
                    )
                slots.append(self.dummy_slot)
                mask.append(0)
            else:
                slots.append(slot)
                mask.append(0 if request_id in
                            self._skip_extend_request_ids else 1)
        # A fresh pinned staging tensor per call: the caching host allocator keeps it alive until the
        # copies below have run, which avoids a stream synchronization on buffer reuse.
        staging = torch.tensor([slots, mask],
                               dtype=torch.int32,
                               pin_memory=prefer_pinned())
        slot_ids_cuda[:num_rows].copy_(staging[0], non_blocking=True)
        row_mask_cuda[:num_rows].copy_(staging[1], non_blocking=True)

    def extend_and_draft(
        self,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        slot_ids_cuda: torch.Tensor,
        row_mask_cuda: torch.Tensor,
        batch_size: int,
        draft_len: int,
    ) -> torch.Tensor:
        """Append the accepted tokens to each row's history and look up the next draft tokens.

        Fully on device; CUDA graph capturable.

        Args:
            accepted_tokens: [batch_size, runtime_draft_len + 1] int32 tokens accepted this step.
            num_accepted_tokens: [batch_size] int32 accepted count per row.
            slot_ids_cuda: [>= batch_size] int32 slot per row, filled by ``prepare``.
            row_mask_cuda: [>= batch_size] int32 row mask, filled by ``prepare``.
            batch_size: Number of rows in the batch.
            draft_len: Draft tokens to propose per row; 0 only extends the histories.

        Returns:
            [batch_size, draft_len] int32 draft tokens, zero padded where no continuation exists.
        """
        draft_tokens = self._draft_tokens_flat[:batch_size * draft_len].view(
            batch_size, draft_len)
        match_lens = self.match_lens[:batch_size]
        torch.ops.trtllm.ngram_extend_and_draft_op(
            self.history_tokens,
            self.history_lens,
            slot_ids_cuda[:batch_size],
            row_mask_cuda[:batch_size],
            accepted_tokens.to(torch.int32).contiguous(),
            num_accepted_tokens.to(torch.int32).contiguous(),
            draft_tokens,
            match_lens,
            self.max_matching_ngram_size,
            self.is_use_oldest,
            self.is_public_pool,
        )
        return draft_tokens

    def _seed_history(self, request: LlmRequest) -> None:
        tokens = request.get_tokens(0)
        num_tokens = min(len(tokens), self.max_seq_len)
        slot = self.slot_manager.add_slot(request.request_id)
        if num_tokens > 0:
            staging = torch.tensor(tokens[:num_tokens],
                                   dtype=torch.int32,
                                   pin_memory=prefer_pinned())
            self.history_tokens[slot, :num_tokens].copy_(staging,
                                                         non_blocking=True)
        self.history_lens[slot].fill_(num_tokens)
        self._seeded_request_ids.add(request.request_id)


@dataclass
class NGramSpecMetadata(SpecMetadata):
    """
    Metadata for NGram speculative decoding.

    Holds the pool manager reference and the per-batch device buffers the kernel reads.
    """

    # Reference to the pool manager (history state lives outside the graph)
    ngram_pool_manager: Optional[NGramPoolManager] = None

    # Pre-allocated GPU buffers for CUDA graph compatibility
    batch_indices_cuda: Optional[torch.Tensor] = field(default=None, repr=False)
    slot_ids_cuda: Optional[torch.Tensor] = field(default=None, repr=False)
    row_mask_cuda: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_num_requests > 0:
            self.batch_indices_cuda = torch.arange(self.max_num_requests,
                                                   dtype=torch.int32,
                                                   device="cuda")
            self.slot_ids_cuda = torch.zeros(self.max_num_requests,
                                             dtype=torch.int32,
                                             device="cuda")
            self.row_mask_cuda = torch.zeros(self.max_num_requests,
                                             dtype=torch.int32,
                                             device="cuda")

    def prepare(self) -> None:
        """Called before the forward, outside the CUDA graph, to map batch rows to history slots."""
        assert self.request_ids is not None, "request_ids must be set before prepare()"
        if self.ngram_pool_manager is None:
            raise ValueError("NGram pool manager is not set")
        self.ngram_pool_manager.prepare(self.request_ids, self.slot_ids_cuda,
                                        self.row_mask_cuda)

    def create_cuda_graph_metadata(self, max_batch_size: int):
        """Creates metadata for CUDA graph execution."""
        if self.is_cuda_graph:
            return self

        import copy

        cuda_graph_metadata = copy.copy(self)
        cuda_graph_metadata.is_cuda_graph = True
        cuda_graph_metadata.max_num_requests = max_batch_size
        cuda_graph_metadata.__post_init__()
        return cuda_graph_metadata


class NGramWorker(SpecWorkerBase):
    """
    NGram speculative decoding worker that runs inside the model forward.

    Verifies the previous drafts against the target logits, then extends the request histories and
    looks up the next drafts with a single fused GPU kernel. No neural draft model is involved.
    """

    def __init__(self,
                 spec_config: "NGramDecodingConfig",
                 model_config=None) -> None:
        super().__init__()
        self.spec_config = spec_config
        self._max_draft_len = spec_config.max_draft_len

    @property
    def max_draft_len(self) -> int:
        return self._max_draft_len

    def _forward_impl(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            hidden_states: torch.Tensor,
            logits: torch.Tensor,
            attn_metadata,
            spec_metadata: NGramSpecMetadata,
            draft_model=None,  # Not used for NGram
            resource_manager=None,  # Not used for NGram
    ):
        """
        Steps:
        1. Sample the target tokens and accept/reject the previous draft tokens
        2. Promote accepted recurrent states on hybrid (SSM) models
        3. Extend the histories and look up the next drafts (one kernel launch)
        4. Assemble the outputs the SpecSampler expects

        Returns:
            Dict with:
            - logits: Raw logits from the target model
            - new_tokens: Accepted tokens [batch_size, runtime_draft_len + 1]
            - new_tokens_lens: Number of accepted tokens [batch_size]
            - next_draft_tokens: Draft tokens for the next iteration [batch_size, runtime_draft_len]
            - next_new_tokens: Input tokens for the next iteration [batch_size, runtime_draft_len + 1]
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        runtime_draft_len = spec_metadata.runtime_draft_len
        pool_manager = spec_metadata.ngram_pool_manager
        if pool_manager is None:
            raise ValueError("NGram pool manager is not set")

        if runtime_draft_len == 0:
            outputs = self.skip_drafting(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
            )
            # Speculation is off for this batch, but the histories must still follow the sequence so
            # that drafting resumes from the right suffix once it is re-enabled.
            pool_manager.extend_and_draft(
                outputs["new_tokens"],
                outputs["new_tokens_lens"],
                spec_metadata.slot_ids_cuda,
                spec_metadata.row_mask_cuda,
                batch_size,
                draft_len=0,
            )
            return outputs

        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata)

        # Hybrid (SSM/recurrent) models: promote the accepted step's recurrent state from the
        # verification scratch buffers into the live pools. Same call site as the other one-engine
        # workers; no-op for pure-attention models via the isinstance gate.
        if num_gens > 0 and isinstance(attn_metadata.kv_cache_manager,
                                       MambaHybridCacheManager):
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
            )

        next_draft_tokens = pool_manager.extend_and_draft(
            accepted_tokens,
            num_accepted_tokens,
            spec_metadata.slot_ids_cuda,
            spec_metadata.row_mask_cuda,
            batch_size,
            runtime_draft_len,
        )

        self._rollback_guided_decoder_after_verify(num_accepted_tokens)

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
