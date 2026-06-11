# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Base class for speculative decoding samplers.

This module provides a common base class for MTPSampler, SASampler, and
Eagle3OneModelSampler.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.sampler import (
    DEFAULT_BEAM_IDX,
    AsyncWorkerMixin,
    Sampler,
    SampleState,
    SampleStateTensors,
    TorchSampler,
    add_token,
    int_tensor,
)
from ..pyexecutor.scheduler import ScheduledRequests


@dataclass(kw_only=True)
class SampleStateTensorsSpec(SampleStateTensors):
    """Tensors for speculative decoding sample state."""

    new_tokens_lens: torch.Tensor
    next_draft_tokens: torch.Tensor


@dataclass(kw_only=True)
class SampleStateSpec(SampleState):
    """Sample state for speculative decoding."""

    device: SampleStateTensorsSpec
    host: SampleStateTensorsSpec


class SpecSamplerBase(Sampler[SampleStateSpec], AsyncWorkerMixin):
    """
    Base class for speculative decoding samplers (MTP, NGram, Eagle3, SA).

    Provides common functionality:
    - Pre-allocated GPU storage buffers
    - Async GPU->CPU copy in sample_async
    - Request state updates in update_requests

    Subclasses can customize behavior by overriding:
    - _get_max_tokens(): How to calculate max_tokens for storage
    - _get_draft_tokens_storage_size(): Size of next_draft_tokens tensor
    - _add_dummy_draft_tokens(): Whether to add dummy drafts for context requests
    """

    SampleState = SampleStateSpec

    def is_generation_model(self) -> bool:
        return True

    @dataclass(kw_only=True)
    class Store:
        """Storage for speculative decoding tensors."""

        new_tokens: torch.Tensor
        next_new_tokens: torch.Tensor
        next_draft_tokens: torch.Tensor
        new_tokens_lens: torch.Tensor

    def __init__(self, args: TorchSampler.Args, *, draft_len: int):
        """
        Initialize the speculative sampler.

        Args:
            args: TorchSampler.Args with max_num_sequences, max_seq_len, etc.
            draft_len: Maximum number of draft tokens per iteration.
        """
        self.mapping = None
        self.draft_len = draft_len
        self.max_seq_len = args.max_seq_len

        seq_slots = args.max_num_sequences
        max_tokens = self._get_max_tokens(args, draft_len)
        draft_tokens_size = self._get_draft_tokens_storage_size(args, draft_len)
        self.max_beam_width = args.max_beam_width
        assert self.max_beam_width == 1, "beam width must be 1 for speculative decoding"

        self.store = self.Store(
            new_tokens=int_tensor((max_tokens, seq_slots, self.max_beam_width)),
            next_new_tokens=int_tensor((max_tokens, seq_slots, self.max_beam_width)),
            next_draft_tokens=int_tensor((seq_slots, draft_tokens_size)),
            new_tokens_lens=int_tensor((seq_slots,)),
        )
        self.store.new_tokens.zero_()
        self.store.next_new_tokens.zero_()
        self.store.next_draft_tokens.zero_()
        self.store.new_tokens_lens.zero_()

    def _get_max_tokens(self, args: TorchSampler.Args, draft_len: int) -> int:
        """
        Calculate max_tokens for storage allocation.

        Override in subclasses if needed. Default: draft_len + 1.
        MTP uses args.max_total_draft_tokens + 1 for tree-based speculation.
        """
        return draft_len + 1

    def _get_draft_tokens_storage_size(self, args: TorchSampler.Args, draft_len: int) -> int:
        """
        Calculate storage size for next_draft_tokens tensor.

        Override in subclasses if needed. Default: draft_len.
        MTP uses args.max_total_draft_tokens for tree-based speculation.
        """
        return draft_len

    def _add_dummy_draft_tokens(self) -> bool:
        """
        Whether to add dummy draft tokens for context requests.

        Override in subclasses. Default: True (needed for KV cache preparation).
        """
        return True

    @staticmethod
    def _zero_negative_token_ids(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.numel() == 0:
            return tokens
        return torch.where(tokens < 0, torch.zeros_like(tokens), tokens)

    def _request_common_handling(
        self,
        request: LlmRequest,
        next_draft_tokens: list[list[int]],
        runtime_draft_len: Optional[int],
    ) -> None:
        """Common handling for both context and generation requests."""
        if request.py_return_context_logits:
            logger.warning(
                "return_context_logits not supported with speculative decoding, "
                "skipping for request %s",
                request.py_request_id,
            )
        if request.py_return_generation_logits:
            logger.warning(
                "return_generation_logits not supported with speculative decoding, "
                "skipping for request %s",
                request.py_request_id,
            )
        if request.py_return_log_probs:
            logger.warning(
                "return_log_probs not supported with speculative decoding, skipping for request %s",
                request.py_request_id,
            )
        request.py_draft_tokens = next_draft_tokens[request.py_seq_slot][:runtime_draft_len]
        request.py_decoding_iter += 1

    def update_requests(
        self,
        state: SampleStateSpec,
        resource_manager: Optional[BaseResourceManager] = None,
    ) -> None:
        """
        CPU-side request updates after GPU->CPU sync.

        Waits for async copy to complete, then updates request state with:
        - Accepted tokens
        - Stop criteria checks
        - Next iteration draft tokens
        """
        assert isinstance(state, SampleStateSpec)

        state.sampler_event.synchronize()
        new_tokens = state.host.new_tokens.tolist()
        new_tokens_lens_list = state.host.new_tokens_lens.tolist()
        next_draft_tokens_list = state.host.next_draft_tokens.tolist()
        beam_idx = DEFAULT_BEAM_IDX
        runtime_draft_len = getattr(state, "runtime_draft_len", self.draft_len)

        for req in state.requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            num_new_tokens = new_tokens_lens_list[req.py_seq_slot]
            num_accepted_draft_tokens = max(num_new_tokens - 1, 0)
            for i in range(num_new_tokens):
                new_token = add_token(req, new_tokens, beam_idx=beam_idx, step=i)
                if TorchSampler._handle_stop_criteria(
                    req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
                ):
                    break
            req.py_num_accepted_draft_tokens = num_accepted_draft_tokens
            req.py_rewind_len = runtime_draft_len - num_accepted_draft_tokens
            self._request_common_handling(req, next_draft_tokens_list, runtime_draft_len)

    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        outputs: dict[str, torch.Tensor],
        num_context_logits_prefix_sum: list[int],
    ) -> SampleStateSpec:
        """
        Async sampling - schedules GPU->CPU copy.
        Called after CUDA graph replay.

        Args:
            scheduled_requests: Batch of scheduled requests
            outputs: Dict from worker forward() containing:
                - new_tokens: [batch, max_draft_len + 1] accepted tokens
                - new_tokens_lens: [batch] number of accepted tokens
                - next_draft_tokens: [batch, max_draft_len] draft tokens for next iter
                - next_new_tokens: [batch, max_draft_len + 1] input for next iter
            num_context_logits_prefix_sum: Prefix sum of context logits (unused)

        Returns:
            SampleStateSpec with device and host tensors
        """
        num_skip = len(scheduled_requests.context_requests_chunking)
        finished_context_requests = scheduled_requests.context_requests_last_chunk
        all_sampling_requests = finished_context_requests + scheduled_requests.generation_requests

        # CUDA graph padding appends dummy requests as a suffix. Keep the common
        # real-row path as a contiguous slice and only use advanced indexing for
        # unexpected inner dummy rows.
        num_real_prefix_rows = len(all_sampling_requests)
        while num_real_prefix_rows > 0 and all_sampling_requests[num_real_prefix_rows - 1].is_dummy:
            num_real_prefix_rows -= 1

        has_inner_dummy = any(req.is_dummy for req in all_sampling_requests[:num_real_prefix_rows])
        if has_inner_dummy:
            sampling_row_indices = [
                idx for idx, req in enumerate(all_sampling_requests) if not req.is_dummy
            ]
            sampling_requests = [all_sampling_requests[idx] for idx in sampling_row_indices]
        else:
            sampling_row_indices = list(range(num_real_prefix_rows))
            sampling_requests = all_sampling_requests[:num_real_prefix_rows]
        seq_slots = [req.py_seq_slot for req in sampling_requests]
        max_seq_slots = self.store.new_tokens_lens.shape[0]
        invalid_slots = [
            (req.py_request_id, slot)
            for req, slot in zip(sampling_requests, seq_slots)
            if slot is None or slot < 0 or slot >= max_seq_slots
        ]
        if invalid_slots:
            raise RuntimeError(
                f"Invalid speculative sampler seq slots {invalid_slots}; "
                f"max_seq_slots={max_seq_slots}"
            )

        slots = torch.as_tensor(seq_slots, dtype=torch.long)
        slots = slots.to(device="cuda", non_blocking=True)

        output_rows = outputs["new_tokens"].shape[0]
        for output_name in ("new_tokens_lens", "next_draft_tokens", "next_new_tokens"):
            if outputs[output_name].shape[0] != output_rows:
                raise RuntimeError(
                    "Speculative sampler output row mismatch: "
                    f"new_tokens rows={output_rows}, "
                    f"{output_name} rows={outputs[output_name].shape[0]}"
                )

        first_output_row = num_skip
        row_indices = [first_output_row + idx for idx in sampling_row_indices]
        if row_indices and max(row_indices) >= output_rows:
            raise RuntimeError(
                "Speculative sampler output row mismatch: "
                f"row_indices={row_indices}, output_rows={output_rows}, "
                f"num_skip={num_skip}, "
                f"num_all_sampling_requests={len(all_sampling_requests)}"
            )

        if len(row_indices) == 0:
            row_indexer = slice(first_output_row, first_output_row)
        elif row_indices == list(range(first_output_row, first_output_row + len(row_indices))):
            row_indexer = slice(first_output_row, first_output_row + len(row_indices))
        else:
            row_indexer = torch.as_tensor(
                row_indices,
                dtype=torch.long,
                device=outputs["new_tokens"].device,
            )

        o_new_tokens = outputs["new_tokens"][row_indexer]
        o_new_tokens_lens = outputs["new_tokens_lens"][row_indexer]
        o_next_draft_tokens = outputs["next_draft_tokens"][row_indexer]
        o_next_new_tokens = outputs["next_new_tokens"][row_indexer]
        runtime_draft_len = o_next_draft_tokens.shape[1]
        o_new_tokens_lens = o_new_tokens_lens.clamp(min=0, max=o_new_tokens.shape[1])

        if o_new_tokens.numel() > 0:
            o_new_tokens = self._zero_negative_token_ids(o_new_tokens)
            token_cols = torch.arange(
                o_new_tokens.shape[1],
                device=o_new_tokens.device,
            ).unsqueeze(0)
            accepted_mask = token_cols < o_new_tokens_lens.unsqueeze(1)
            o_new_tokens = torch.where(accepted_mask, o_new_tokens, torch.zeros_like(o_new_tokens))
        o_next_draft_tokens = self._zero_negative_token_ids(o_next_draft_tokens)
        o_next_new_tokens = self._zero_negative_token_ids(o_next_new_tokens)

        # Pad to match fixed-size store buffers for index_copy_.
        if o_new_tokens.shape[1] < (self.draft_len + 1):
            o_new_tokens = torch.nn.functional.pad(
                o_new_tokens, (0, (self.draft_len + 1) - o_new_tokens.shape[1])
            )
        if o_next_draft_tokens.shape[1] < self.draft_len:
            o_next_draft_tokens = torch.nn.functional.pad(
                o_next_draft_tokens, (0, self.draft_len - o_next_draft_tokens.shape[1])
            )
        if o_next_new_tokens.shape[1] < (self.draft_len + 1):
            o_next_new_tokens = torch.nn.functional.pad(
                o_next_new_tokens, (0, (self.draft_len + 1) - o_next_new_tokens.shape[1])
            )

        # Use index_copy_ for efficient copying (slots are unique)
        self.store.new_tokens.squeeze(-1).T.index_copy_(0, slots, o_new_tokens)
        self.store.next_new_tokens.squeeze(-1).T.index_copy_(0, slots, o_next_new_tokens)
        self.store.new_tokens_lens.index_copy_(0, slots, o_new_tokens_lens)
        self.store.next_draft_tokens.index_copy_(0, slots, o_next_draft_tokens)

        # Create sample state with async D2H copy
        device_tensors = SampleStateTensorsSpec(
            new_tokens=self.store.next_new_tokens,
            new_tokens_lens=self.store.new_tokens_lens,
            next_draft_tokens=self.store.next_draft_tokens,
        )

        host_tensors = SampleStateTensorsSpec(
            new_tokens=self._copy_to_host(self.store.new_tokens),
            new_tokens_lens=self._copy_to_host(self.store.new_tokens_lens),
            next_draft_tokens=self._copy_to_host(self.store.next_draft_tokens),
        )
        sampler_event = self._record_sampler_event()

        # Add dummy draft tokens to context requests for KV cache preparation
        if self._add_dummy_draft_tokens():
            for request in finished_context_requests:
                request.py_draft_tokens = [1] * self.draft_len

        return SampleStateSpec(
            requests=sampling_requests,
            device=device_tensors,
            host=host_tensors,
            sampler_event=sampler_event,
            runtime_draft_len=runtime_draft_len,
        )
