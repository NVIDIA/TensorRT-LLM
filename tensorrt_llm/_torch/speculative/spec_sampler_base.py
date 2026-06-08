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

import os
from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm.bindings.executor import FinishReason
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
        self.max_num_sequences = seq_slots
        max_tokens = self._get_max_tokens(args, draft_len)
        draft_tokens_size = self._get_draft_tokens_storage_size(args, draft_len)
        self.max_beam_width = args.max_beam_width
        assert self.max_beam_width == 1, "beam width must be 1 for speculative decoding"
        self.loop_guard_tokens = int(
            os.environ.get("TRTLLM_SPEC_LOOP_GUARD_TOKENS", "0"))
        self.loop_guard_period = max(
            int(os.environ.get("TRTLLM_SPEC_LOOP_GUARD_PERIOD", "64")), 1)
        self.loop_guard_repeats = max(
            int(os.environ.get("TRTLLM_SPEC_LOOP_GUARD_REPEATS", "4")), 2)
        self.loop_guard_window = max(
            int(os.environ.get("TRTLLM_SPEC_LOOP_GUARD_WINDOW", "1024")), 128)
        self.count_debug_tokens = int(
            os.environ.get("TRTLLM_SPEC_COUNT_DEBUG_TOKENS", "0"))
        self.count_debug_period = max(
            int(os.environ.get("TRTLLM_SPEC_COUNT_DEBUG_PERIOD", "256")), 1)
        self.count_debug_max_rows = max(
            int(os.environ.get("TRTLLM_SPEC_COUNT_DEBUG_MAX_ROWS", "4")), 1)
        self.count_debug_path_logged = False
        self.count_debug_order_logged = False

        self.store = self.Store(
            new_tokens=int_tensor((max_tokens, seq_slots, self.max_beam_width)),
            next_new_tokens=int_tensor((max_tokens, seq_slots, self.max_beam_width)),
            next_draft_tokens=int_tensor((seq_slots, draft_tokens_size)),
            new_tokens_lens=int_tensor((seq_slots,)),
        )

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

    def _is_valid_slot(self, request: LlmRequest) -> bool:
        slot = request.py_seq_slot
        return (slot is not None and 0 <= int(slot) < self.max_num_sequences
                and not request.is_dummy)

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
        request.py_draft_tokens = next_draft_tokens[
            request.py_seq_slot][:runtime_draft_len]
        request.py_decoding_iter += 1

    def _ordered_sampling_requests_from_outputs(
            self,
            fallback_requests: list[LlmRequest],
            outputs: dict[str, torch.Tensor],
            num_skip: int) -> list[LlmRequest]:
        request_ids = outputs.get("penalty_sampling_request_ids")
        if request_ids is None:
            return fallback_requests

        output_request_ids = [int(request_id) for request_id in request_ids]
        output_seq_slots = outputs.get("penalty_sampling_seq_slots")
        if output_seq_slots is not None:
            output_seq_slots = [int(slot) for slot in output_seq_slots]

        row_request_ids = output_request_ids[
            num_skip:num_skip + len(fallback_requests)]
        row_seq_slots = None
        if output_seq_slots is not None:
            row_seq_slots = output_seq_slots[
                num_skip:num_skip + len(fallback_requests)]
        if len(row_request_ids) != len(fallback_requests):
            return fallback_requests

        available = list(fallback_requests)
        ordered_requests: list[LlmRequest] = []
        for row, request_id in enumerate(row_request_ids):
            row_slot = row_seq_slots[row] if row_seq_slots is not None else None
            match_idx = None
            for idx, request in enumerate(available):
                if int(request.py_request_id) != request_id:
                    continue
                if row_slot is not None:
                    slot = request.py_seq_slot
                    request_slot = int(slot) if slot is not None else -1
                    if request_slot != row_slot:
                        continue
                match_idx = idx
                break
            if match_idx is None and row_slot is not None:
                for idx, request in enumerate(available):
                    if int(request.py_request_id) == request_id:
                        match_idx = idx
                        break
            if match_idx is None:
                return fallback_requests
            ordered_requests.append(available.pop(match_idx))

        if (self.count_debug_tokens > 0 and not self.count_debug_order_logged):
            fallback_head = [
                (int(request.py_request_id),
                 int(request.py_seq_slot) if request.py_seq_slot is not None else -1)
                for request in fallback_requests[:16]
            ]
            ordered_head = [
                (int(request.py_request_id),
                 int(request.py_seq_slot) if request.py_seq_slot is not None else -1)
                for request in ordered_requests[:16]
            ]
            if ordered_head != fallback_head:
                self.count_debug_order_logged = True
                logger.info(
                    "Spec sampler remapped output row request order "
                    "fallback_head=%s output_head=%s",
                    fallback_head, ordered_head)

        return ordered_requests

    @staticmethod
    def _prompt_len(request: LlmRequest) -> int:
        for attr in ("py_orig_prompt_len", "orig_prompt_len", "py_prompt_len",
                     "prompt_len"):
            value = getattr(request, attr, None)
            if value is not None:
                return max(int(value), 0)
        return 0

    @staticmethod
    def _has_repeated_suffix(tokens: list[int], repeats: int) -> bool:
        if len(tokens) >= 256 and len(set(tokens[-256:])) <= 8:
            return True
        for ngram in (16, 8, 4):
            needed = ngram * repeats
            if len(tokens) < needed:
                continue
            suffix = tokens[-ngram:]
            repeated = True
            for i in range(2, repeats + 1):
                start = -ngram * i
                end = start + ngram
                if tokens[start:end] != suffix:
                    repeated = False
                    break
            if repeated:
                return True
        return False

    def _has_repeated_window(self, tokens: list[int]) -> bool:
        window = tokens[-self.loop_guard_window:]
        if len(window) < 512:
            return False
        if len(set(window[-512:])) <= 32:
            return True
        for ngram, min_count in ((16, 3), (12, 4), (8, 8)):
            if len(window) < ngram * min_count:
                continue
            counts: dict[tuple[int, ...], int] = {}
            for start in range(0, len(window) - ngram + 1):
                key = tuple(window[start:start + ngram])
                count = counts.get(key, 0) + 1
                if count >= min_count:
                    return True
                counts[key] = count
        return False

    def _maybe_finish_repetition_loop(self, request: LlmRequest, beam_idx: int,
                                      prev_generated_len: int,
                                      generated_len: int) -> bool:
        if self.loop_guard_tokens <= 0:
            return False
        if generated_len < self.loop_guard_tokens:
            return False
        if (prev_generated_len >= self.loop_guard_tokens
                and (generated_len // self.loop_guard_period
                     == prev_generated_len // self.loop_guard_period)):
            return False
        tokens = request.get_tokens(beam_idx)
        generated_tokens = tokens[-generated_len:]
        if (not self._has_repeated_suffix(generated_tokens,
                                          self.loop_guard_repeats)
                and not self._has_repeated_window(generated_tokens)):
            return False

        logger.warning(
            "Speculative decoding repetition guard stopped request_id=%s "
            "generated_len=%s",
            request.py_request_id, generated_len)
        request.finish_by(FinishReason.STOP_WORDS, beam_idx)
        return True

    def _should_debug_count_row(self, request: LlmRequest,
                                accepted_len: int) -> tuple[bool, int, int]:
        if self.count_debug_tokens <= 0 or accepted_len <= 0:
            return False, 0, 0
        prompt_len = self._prompt_len(request)
        prev_generated_len = max(len(request.get_tokens(0)) - prompt_len, 0)
        next_generated_len = prev_generated_len + accepted_len
        if next_generated_len < self.count_debug_tokens:
            return False, prev_generated_len, next_generated_len
        prev_bucket = max(prev_generated_len - self.count_debug_tokens,
                          0) // self.count_debug_period
        next_bucket = max(next_generated_len - self.count_debug_tokens,
                          0) // self.count_debug_period
        return (prev_generated_len < self.count_debug_tokens
                or next_bucket != prev_bucket), prev_generated_len, next_generated_len

    def _maybe_log_count_debug_path(self, outputs: dict[str, torch.Tensor]) -> None:
        if self.count_debug_tokens <= 0 or self.count_debug_path_logged:
            return
        self.count_debug_path_logged = True
        logger.info(
            "Spec count debug path env_mode=%s dense=%s sparse=%s "
            "history=%s history_appended=%s",
            os.environ.get("TRTLLM_SPEC_COUNT_MODE", ""),
            outputs.get("penalty_token_counts") is not None,
            (outputs.get("penalty_sparse_token_ids") is not None
             and outputs.get("penalty_sparse_token_counts") is not None
             and outputs.get("penalty_sparse_count_lens") is not None),
            (outputs.get("penalty_history_tokens") is not None
             and outputs.get("penalty_history_lens") is not None),
            outputs.get("penalty_history_appended", False))

    def _debug_dense_counts_before_append(
            self,
            sampling_requests: list[LlmRequest],
            penalty_slot_values: list[int],
            o_new_tokens: torch.Tensor,
            o_new_tokens_lens: torch.Tensor,
            token_counts: torch.Tensor,
    ) -> None:
        if self.count_debug_tokens <= 0:
            return

        accepted_lens = o_new_tokens_lens.detach().cpu().tolist()
        rows_logged = 0
        for row, request in enumerate(sampling_requests):
            if rows_logged >= self.count_debug_max_rows:
                return
            if row >= len(penalty_slot_values):
                continue
            slot = penalty_slot_values[row]
            if slot < 0:
                continue
            accepted_len = int(accepted_lens[row])
            should_log, prev_generated_len, next_generated_len = (
                self._should_debug_count_row(request, accepted_len))
            if not should_log:
                continue

            accepted_tokens = o_new_tokens[row, :accepted_len].detach().cpu().tolist()
            history_tail = [
                int(token) for token in request.get_tokens(0)[-16:]
                if int(token) >= 0
            ]
            interesting_tokens = list(dict.fromkeys(
                [int(token) for token in accepted_tokens] + history_tail))
            if interesting_tokens:
                interesting_tensor = torch.tensor(
                    interesting_tokens,
                    dtype=torch.long,
                    device=token_counts.device)
                interesting_counts = token_counts[slot].index_select(
                    0, interesting_tensor).detach().cpu().tolist()
                count_by_token = {
                    token: int(count)
                    for token, count in zip(interesting_tokens,
                                            interesting_counts)
                }
            else:
                count_by_token = {}
            accepted_counts = [(int(token), count_by_token.get(int(token), 0))
                               for token in accepted_tokens]
            history_tail_counts = [(token, count_by_token.get(token, 0))
                                   for token in history_tail]

            logger.info(
                "Spec dense count debug before append request_id=%s slot=%s "
                "prev_generated_len=%s next_generated_len=%s accepted_len=%s "
                "accepted_counts=%s history_tail_counts=%s",
                request.py_request_id, slot, prev_generated_len,
                next_generated_len, accepted_len, accepted_counts,
                history_tail_counts)
            rows_logged += 1

    def _debug_sparse_counts_before_append(
            self,
            sampling_requests: list[LlmRequest],
            penalty_slot_values: list[int],
            o_new_tokens: torch.Tensor,
            o_new_tokens_lens: torch.Tensor,
            token_ids: torch.Tensor,
            token_counts: torch.Tensor,
            count_lens: torch.Tensor,
            count_vocab_size: int,
    ) -> None:
        if self.count_debug_tokens <= 0:
            return

        accepted_lens = o_new_tokens_lens.detach().cpu().tolist()
        rows_logged = 0
        for row, request in enumerate(sampling_requests):
            if rows_logged >= self.count_debug_max_rows:
                return
            if row >= len(penalty_slot_values):
                continue
            slot = penalty_slot_values[row]
            if slot < 0:
                continue
            accepted_len = int(accepted_lens[row])
            should_log, prev_generated_len, next_generated_len = (
                self._should_debug_count_row(request, accepted_len))
            if not should_log:
                continue

            count_len = int(count_lens[slot].detach().cpu().item())
            count_len = min(max(count_len, 0), int(token_ids.shape[1]))
            sparse_ids = token_ids[slot, :count_len].detach().cpu().tolist()
            sparse_counts = token_counts[slot, :count_len].detach().cpu().tolist()
            count_by_token = {
                int(token): int(count)
                for token, count in zip(sparse_ids, sparse_counts)
            }

            accepted_tokens = o_new_tokens[row, :accepted_len].detach().cpu().tolist()
            accepted_counts = [(int(token), count_by_token.get(int(token), 0))
                               for token in accepted_tokens]
            history_tail = [
                int(token) for token in request.get_tokens(0)[-16:]
                if int(token) >= 0
            ]
            history_tail_counts = [(token, count_by_token.get(token, 0))
                                   for token in history_tail]

            logger.info(
                "Spec sparse count debug before append request_id=%s slot=%s "
                "prev_generated_len=%s next_generated_len=%s accepted_len=%s "
                "count_len=%s count_vocab_size=%s accepted_counts=%s "
                "history_tail_counts=%s",
                request.py_request_id, slot, prev_generated_len,
                next_generated_len, accepted_len, count_len, count_vocab_size,
                accepted_counts, history_tail_counts)
            rows_logged += 1

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
            for i in range(num_new_tokens):
                new_token = add_token(req, new_tokens, beam_idx=beam_idx, step=i)
                if TorchSampler._handle_stop_criteria(
                    req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
                ):
                    break
            if (self.loop_guard_tokens > 0
                    and req.state != LlmRequestState.GENERATION_COMPLETE):
                guard_len_attr = "_spec_loop_guard_generated_len"
                prev_generated_len = getattr(req, guard_len_attr, 0)
                generated_len = prev_generated_len + num_new_tokens
                setattr(req, guard_len_attr, generated_len)
                self._maybe_finish_repetition_loop(req, beam_idx,
                                                   prev_generated_len,
                                                   generated_len)
            req.py_num_accepted_draft_tokens = num_new_tokens - 1
            req.py_rewind_len = runtime_draft_len - req.py_num_accepted_draft_tokens
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
        fallback_sampling_requests = (
            finished_context_requests + scheduled_requests.generation_requests)
        sampling_requests = self._ordered_sampling_requests_from_outputs(
            fallback_sampling_requests, outputs, num_skip)
        num_sampling_requests = len(sampling_requests)

        valid_positions: list[int] = []
        valid_slot_values: list[int] = []
        penalty_slot_values: list[int] = []
        valid_sampling_requests: list[LlmRequest] = []
        for pos, request in enumerate(sampling_requests):
            if self._is_valid_slot(request):
                slot = int(request.py_seq_slot)
                valid_positions.append(pos)
                valid_slot_values.append(slot)
                penalty_slot_values.append(slot)
                valid_sampling_requests.append(request)
            else:
                penalty_slot_values.append(-1)

        slots = torch.as_tensor(valid_slot_values, dtype=torch.long)
        slots = slots.to(device="cuda", non_blocking=True)
        penalty_slots: Optional[torch.Tensor] = None

        def fallback_penalty_slots() -> torch.Tensor:
            nonlocal penalty_slots
            if penalty_slots is None:
                penalty_slots = torch.as_tensor(penalty_slot_values,
                                                dtype=torch.int32)
                penalty_slots = penalty_slots.to(device="cuda",
                                                 non_blocking=True)
            return penalty_slots

        def output_slot_slice(name: str) -> Optional[torch.Tensor]:
            seq_slots = outputs.get(name)
            if seq_slots is None:
                return None
            end = num_skip + num_sampling_requests
            if seq_slots.numel() < end:
                return None
            return seq_slots[num_skip:end].contiguous()

        count_penalty_slots = output_slot_slice("penalty_count_seq_slots")
        history_penalty_slots = output_slot_slice("penalty_history_seq_slots")
        valid_positions_cuda = torch.as_tensor(valid_positions,
                                               dtype=torch.long,
                                               device="cuda")

        o_new_tokens = outputs["new_tokens"][num_skip : num_skip + num_sampling_requests]
        o_new_tokens_lens = outputs["new_tokens_lens"][num_skip : num_skip + num_sampling_requests]
        o_next_draft_tokens = outputs["next_draft_tokens"][
            num_skip : num_skip + num_sampling_requests
        ]
        o_next_new_tokens = outputs["next_new_tokens"][num_skip : num_skip + num_sampling_requests]
        runtime_draft_len = o_next_draft_tokens.shape[1]

        self._maybe_log_count_debug_path(outputs)

        penalty_token_counts = outputs.get("penalty_token_counts")
        if penalty_token_counts is not None:
            from .one_model_sampler import append_accepted_tokens_to_counts
            self._debug_dense_counts_before_append(
                sampling_requests, penalty_slot_values, o_new_tokens,
                o_new_tokens_lens, penalty_token_counts)
            append_accepted_tokens_to_counts(
                penalty_token_counts,
                count_penalty_slots
                if count_penalty_slots is not None else fallback_penalty_slots(),
                o_new_tokens.contiguous(),
                o_new_tokens_lens.contiguous())
        elif (outputs.get("penalty_sparse_token_ids") is not None
              and outputs.get("penalty_sparse_token_counts") is not None
              and outputs.get("penalty_sparse_count_lens") is not None):
            from .one_model_sampler import append_accepted_tokens_to_sparse_counts
            self._debug_sparse_counts_before_append(
                sampling_requests, penalty_slot_values, o_new_tokens,
                o_new_tokens_lens, outputs["penalty_sparse_token_ids"],
                outputs["penalty_sparse_token_counts"],
                outputs["penalty_sparse_count_lens"],
                int(outputs.get("penalty_count_vocab_size", 0)))
            append_accepted_tokens_to_sparse_counts(
                outputs["penalty_sparse_token_ids"],
                outputs["penalty_sparse_token_counts"],
                outputs["penalty_sparse_count_lens"],
                count_penalty_slots
                if count_penalty_slots is not None else fallback_penalty_slots(),
                o_new_tokens.contiguous(),
                o_new_tokens_lens.contiguous(),
                int(outputs.get("penalty_count_vocab_size", 0)))

        penalty_history_tokens = outputs.get("penalty_history_tokens")
        penalty_history_lens = outputs.get("penalty_history_lens")
        if (penalty_history_tokens is not None
                and penalty_history_lens is not None
                and not outputs.get("penalty_history_appended", False)):
            from .one_model_sampler import append_accepted_tokens_to_history
            append_accepted_tokens_to_history(
                penalty_history_tokens,
                penalty_history_lens,
                history_penalty_slots
                if history_penalty_slots is not None else fallback_penalty_slots(),
                o_new_tokens.contiguous(),
                o_new_tokens_lens.contiguous())

        o_new_tokens = o_new_tokens.index_select(0, valid_positions_cuda)
        o_new_tokens_lens = o_new_tokens_lens.index_select(0, valid_positions_cuda)
        o_next_draft_tokens = o_next_draft_tokens.index_select(0, valid_positions_cuda)
        o_next_new_tokens = o_next_new_tokens.index_select(0, valid_positions_cuda)

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
                if self._is_valid_slot(request):
                    request.py_draft_tokens = [1] * self.draft_len

        return SampleStateSpec(
            requests=valid_sampling_requests,
            device=device_tensors,
            host=host_tensors,
            sampler_event=sampler_event,
            runtime_draft_len=runtime_draft_len,
        )
