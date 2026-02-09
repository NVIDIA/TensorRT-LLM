# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

This module provides a common base class for MTPSampler, NGramSampler, and
Eagle3OneModelSampler.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from ..pyexecutor.llm_request import LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.sampler import (
    DEFAULT_BEAM_IDX,
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


class SpecSamplerBase(TorchSampler):
    """
    Base class for speculative decoding samplers (MTP, NGram, Eagle3).

    Provides common functionality:
    - Pre-allocated GPU storage buffers
    - Async GPU->CPU copy in sample_async
    - Request state updates in update_requests

    Subclasses can customize behavior by overriding:
    - _get_max_tokens(): How to calculate max_tokens for storage
    - _add_dummy_draft_tokens(): Whether to add dummy drafts for context requests
    """

    SampleState = SampleStateSpec

    @dataclass(kw_only=True)
    class Store(TorchSampler.Store):
        new_tokens: torch.Tensor
        next_new_tokens: torch.Tensor
        next_draft_tokens: torch.Tensor
        new_tokens_lens: torch.Tensor
        # Necessary to satisfy the interface of TorchSampler.Store
        finish_reasons: None = None

        def __post_init__(self):
            pass  # finish_reasons has no size to compare against new_tokens

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
        self.max_beam_width = args.max_beam_width
        assert self.max_beam_width == 1, "beam width must be 1 for speculative decoding"

        self.store = self.Store(
            new_tokens=int_tensor((max_tokens, seq_slots, self.max_beam_width)),
            next_new_tokens=int_tensor((max_tokens, seq_slots, self.max_beam_width)),
            next_draft_tokens=int_tensor((seq_slots, draft_len)),
            new_tokens_lens=int_tensor((seq_slots,)),
        )

    def _get_max_tokens(self, args: TorchSampler.Args, draft_len: int) -> int:
        """
        Calculate max_tokens for storage allocation.

        Override in subclasses if needed. Default: draft_len + 1.
        MTP uses args.max_total_draft_tokens + 1 for tree-based speculation.
        """
        return draft_len + 1

    def _add_dummy_draft_tokens(self) -> bool:
        """
        Whether to add dummy draft tokens for context requests.

        Override in subclasses. Default: True (needed for KV cache preparation).
        """
        return True

    def _request_common_handling(
        self,
        request: LlmRequest,
        next_draft_tokens: list[list[int]],
    ) -> None:
        """Common handling for both context and generation requests."""
        assert not request.py_return_context_logits, (
            "return_context_logits not implemented for speculative sampler"
        )
        assert not request.py_return_generation_logits, (
            "return_generation_logits not implemented for speculative sampler"
        )
        assert not request.py_return_log_probs, (
            "return_log_probs not implemented for speculative sampler"
        )
        request.py_draft_tokens = next_draft_tokens[request.py_seq_slot]
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

        # Handle context requests (prefill phase)
        for req in state.scheduled_requests.context_requests:
            if (
                req.state == LlmRequestState.GENERATION_COMPLETE
                or req.context_remaining_length != 0
            ):
                continue
            new_token = add_token(req, new_tokens, beam_idx=beam_idx)
            TorchSampler._handle_stop_criteria(
                req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
            )
            self._request_common_handling(req, next_draft_tokens_list)

        # Handle generation requests (decode phase)
        for req in state.scheduled_requests.generation_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            num_new_tokens = new_tokens_lens_list[req.py_seq_slot]
            for i in range(num_new_tokens):
                new_token = add_token(req, new_tokens, beam_idx=beam_idx, step=i)
                if TorchSampler._handle_stop_criteria(
                    req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
                ):
                    break
            req.py_num_accepted_draft_tokens = num_new_tokens - 1
            req.py_rewind_len = self.draft_len - req.py_num_accepted_draft_tokens
            self._request_common_handling(req, next_draft_tokens_list)

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
        requests = scheduled_requests.all_requests()
        slots = torch.as_tensor([r.py_seq_slot for r in requests])
        slots = slots.to(device="cuda", non_blocking=True)

        o_new_tokens = outputs["new_tokens"][: len(requests)]
        o_new_tokens_lens = outputs["new_tokens_lens"][: len(requests)]
        o_next_draft_tokens = outputs["next_draft_tokens"][: len(requests)]
        o_next_new_tokens = outputs["next_new_tokens"][: len(requests)]

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
            new_tokens=self.store.new_tokens.to("cpu", non_blocking=True),
            new_tokens_lens=self.store.new_tokens_lens.to("cpu", non_blocking=True),
            next_draft_tokens=self.store.next_draft_tokens.to("cpu", non_blocking=True),
        )

        sampler_event = torch.cuda.Event()
        sampler_event.record()

        # Add dummy draft tokens to context requests for KV cache preparation
        if self._add_dummy_draft_tokens():
            for request in scheduled_requests.context_requests:
                request.py_draft_tokens = [1] * self.draft_len

        return SampleStateSpec(
            scheduled_requests=scheduled_requests,
            device=device_tensors,
            host=host_tensors,
            sampler_event=sampler_event,
        )
