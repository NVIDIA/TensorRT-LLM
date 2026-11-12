# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finish-reason handling for ``TorchSampler``.

Owns the stop criteria evaluated on the device each step -- end-id match,
max-length, and stop words -- along with the per-slot buffers they need
(packed stop-word rules and per-request past-token windows).
``TorchSampler`` holds one :class:`FinishReasonsHandler` and drives it
through request admission, release, and the per-step write.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch

from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.bindings.executor import FinishReason

from ..llm_request import LlmRequest
from .ops.triton import (
    MAX_FUSED_ELEMENTS_PER_REQUEST,
    compact_stop_words,
    fused_write_finish_reasons,
)
from .sampler_common import int_tensor

__all__ = ["FinishReasonsHandler"]


class _StopWordAllocation(NamedTuple):
    request: LlmRequest
    tensor: torch.Tensor


class FinishReasonsHandler:
    @dataclass(kw_only=True)
    class _FinishReasonsStore:
        """Auxiliary data structures used for finish reasons handling."""

        # Per-request dynamic data
        finish_reasons_cuda: torch.Tensor
        """Shape: [max_tokens, batch_size, beam_width]
        Usage: Stores the determined finish reasons for all sampled tokens
        for each request. Some (draft) tokens and corresponding
        finish reasons might still be discarded."""

        # Per-request static data
        max_lengths_cuda: torch.Tensor
        """Shape: [batch_size]
        Usage: Stores the maximum sequence lengths for each request"""
        end_ids_cuda: torch.Tensor
        """Shape: batch_size
        Usage: Stores the end ids for each request"""
        stop_words_metadata_cuda: torch.Tensor
        """Shape: [batch_size, 5], int64.
        Per-slot allocation pointer, rule count, history offset, window length,
        and beam stride. The handler owns the pointed-to allocations."""
        max_stop_word_lengths_host: torch.Tensor
        """Shape: [batch_size]
        Usage: Stores the size of the longest stop word for each request."""
        num_accepted_draft_tokens_host: torch.Tensor
        """Shape: [batch_size]
        Usage: Stores the number of accepted tokens for each request."""

    def __init__(
        self,
        *,
        max_num_sequences: int,
        max_beam_width: int,
        max_tokens: int,
        max_seq_len: int,
    ):
        self._update_sizes(
            max_num_sequences=max_num_sequences,
            max_beam_width=max_beam_width,
            max_tokens=max_tokens,
            max_seq_len=max_seq_len,
        )
        self._stop_word_allocations: dict[int, _StopWordAllocation] = {}
        self._stop_word_history_slots: set[int] = set()
        self._stop_word_allocation_generation = 0
        self._stop_word_stream_generations: dict[torch.cuda.Stream, int] = {}
        self._setup_store()
        self._setup_helper_tensors()
        self._temp_data: FinishReasonsHandler._TemporaryData = self._TemporaryData()

    @property
    def has_stop_word_history(self) -> bool:
        """Whether any retained request needs a multi-token history window."""
        return bool(self._stop_word_history_slots)

    @property
    def _use_speculative_decoding(self) -> bool:
        return self._max_tokens > 1

    @property
    def new_max_lens(self) -> list[int]:
        return self._temp_data.max_lens

    @property
    def new_end_ids(self) -> list[int]:
        return self._temp_data.end_ids

    def _update_sizes(
        self,
        *,
        max_num_sequences: int,
        max_beam_width: int,
        max_tokens: int,
        max_seq_len: int,
    ) -> None:
        """Updates the sizes of the finish reasons handler

        Sets member variables to store the current sizes.
        These sizes are used to initialize the buffer tensors.
        """
        self._max_num_sequences: int = max_num_sequences
        self._max_beam_width: int = max_beam_width
        self._max_tokens: int = max_tokens
        self._max_seq_len: int = max_seq_len
        self._fused_tile_fits: bool = (
            self._max_tokens * self._max_beam_width <= MAX_FUSED_ELEMENTS_PER_REQUEST
        )

    def _setup_store(self) -> None:
        """Sets up the store for the finish reasons handler by initializing all buffer tensors."""
        finish_reasons_cuda = int_tensor(
            (self._max_tokens, self._max_num_sequences, self._max_beam_width)
        )
        max_lengths_cuda = int_tensor((self._max_num_sequences,))
        end_ids_cuda = int_tensor((self._max_num_sequences,))
        stop_words_metadata_cuda = torch.zeros(
            (self._max_num_sequences, 5), dtype=torch.int64, device="cuda"
        )
        max_stop_word_lengths_host = torch.zeros(
            self._max_num_sequences, device="cpu", dtype=torch.int32
        )
        num_accepted_draft_tokens_host = torch.empty(
            self._max_num_sequences, device="cpu", dtype=torch.int32
        )
        self.store: FinishReasonsHandler._FinishReasonsStore = self._FinishReasonsStore(
            finish_reasons_cuda=finish_reasons_cuda,
            max_lengths_cuda=max_lengths_cuda,
            end_ids_cuda=end_ids_cuda,
            stop_words_metadata_cuda=stop_words_metadata_cuda,
            max_stop_word_lengths_host=max_stop_word_lengths_host,
            num_accepted_draft_tokens_host=num_accepted_draft_tokens_host,
        )

    def _setup_helper_tensors(self) -> None:
        """Preallocate the token offsets used by the maximum-length check."""
        self._max_tokens_offset_cuda: torch.Tensor = torch.arange(
            1, self._max_tokens + 1, device="cuda", dtype=torch.int32
        ).view(-1, 1, 1)

    @dataclass(kw_only=True)
    class _TemporaryData:
        """Data structure to store the temporary data during setup_sampler_step for new requests"""

        def __init__(self) -> None:
            self.metadata: list[list[int]] = []
            self.max_lens: list[int] = []
            self.end_ids: list[int] = []
            self.max_stop_word_lengths: list[int] = []

        def clear(self) -> None:
            self.metadata.clear()
            self.max_lens.clear()
            self.end_ids.clear()
            self.max_stop_word_lengths.clear()

    def release_request(self, request: LlmRequest) -> None:
        """Release a request's allocation before its sequence slot is returned."""
        slot = request.py_seq_slot
        if slot is None:
            return
        allocation = self._stop_word_allocations.get(slot)
        if allocation is None or allocation.request is not request:
            return
        del self._stop_word_allocations[slot]
        self._stop_word_history_slots.discard(slot)
        # Outstanding kernels may still read the metadata on another stream.
        # Admission overwrites host and device metadata before the slot is reused.

    def setup_new_request_handling(self) -> None:
        """Setup the new request handling for the finish reasons handler

        Clears the temporary data for the new request handling.
        This should be called before processing new requests, to avoid
        stale data from previous requests.
        """
        self._temp_data.clear()

    def prepare_for_new_request(self, request: LlmRequest) -> None:
        """Fill _temp_data with the corresponding data from new requests to be used during setup_sampler_step

        Args:
            request: The request to prepare for.
        """

        self._temp_data.max_lens.append(
            min(self._max_seq_len, request.py_orig_prompt_len + request.py_max_new_tokens)
        )
        # Beam-search context-only (disaggregated prefill) requests hand off
        # after their single step, so their end id is masked to the "no end
        # token" sentinel (< 0). Two things depend on it: an end candidate is
        # not pooled by the CBA op, so it stays in its beam slot and travels to
        # the generation server as first_gen_tokens; and the request is not
        # marked finished here, so it still reaches the disagg-transmission
        # state that builds ContextPhaseParams (llmRequest.cpp) -- an END_ID
        # finish leaves it in GENERATION_COMPLETE, which is neither
        # isContextFinished() nor finished-due-to-length, so the handoff is
        # never started and no tokens are produced at all. The generation
        # server sees the end id itself and pools the beam there
        # (TRTLLM-14792). Scoped to beam search: single-beam disaggregation
        # keeps its current end-id behaviour.
        end_id = request.py_end_id
        if end_id is None or (request.is_context_only_request and request.py_beam_width > 1):
            end_id = -1
        self._temp_data.end_ids.append(end_id)

        assert (seq_slot := request.py_seq_slot) is not None
        self._stop_word_allocations.pop(seq_slot, None)
        self._stop_word_history_slots.discard(seq_slot)
        words = request.py_stop_words_list or []
        if not words:
            self._temp_data.metadata.append([0, 0, 0, 0, self._max_beam_width])
            self._temp_data.max_stop_word_lengths.append(0)
            return
        if any(not word for word in words):
            raise ValueError("Stop words must contain at least one token")

        max_length = max(map(len, words))
        offsets = [0]
        flat_tokens: list[int] = []
        for word in words:
            flat_tokens.extend(word)
            offsets.append(len(flat_tokens))
        history_offset = len(offsets) + len(flat_tokens)
        window_length = max_length - 1 + self._max_tokens if max_length > 1 else 0
        history_size = window_length * self._max_beam_width
        host = torch.zeros(
            history_offset + 2 * history_size,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        host[:history_offset] = torch.tensor(offsets + flat_tokens, dtype=torch.int32)
        if window_length:
            history = host[history_offset : history_offset + history_size].view(
                window_length, self._max_beam_width
            )
            # The first sampling step shifts by one before appending new tokens.
            for beam, tokens in enumerate(request.get_tokens()[: self._max_beam_width]):
                count = min(max_length - 1, len(tokens))
                if count:
                    history[max_length - count : max_length, beam] = torch.tensor(
                        tokens[-count:], dtype=torch.int32
                    )
        allocation = host.to("cuda", non_blocking=True)
        self._stop_word_allocations[seq_slot] = _StopWordAllocation(
            request=request, tensor=allocation
        )
        self._stop_word_allocation_generation += 1
        if window_length:
            self._stop_word_history_slots.add(seq_slot)
        self._temp_data.metadata.append(
            [
                allocation.data_ptr(),
                len(words),
                history_offset,
                window_length,
                self._max_beam_width,
            ]
        )
        self._temp_data.max_stop_word_lengths.append(max_length)

    def update_for_new_request(
        self,
        *,
        seq_slots_cuda_long: torch.Tensor,
        max_lengths_cuda: torch.Tensor,
        end_ids_cuda: torch.Tensor,
        seq_slots_host: torch.Tensor,
    ) -> None:
        """Publish new-request metadata without resizing existing requests."""
        temp_data = self._temp_data
        store = self.store
        store.max_lengths_cuda.index_copy_(0, seq_slots_cuda_long, max_lengths_cuda)
        store.end_ids_cuda.index_copy_(0, seq_slots_cuda_long, end_ids_cuda)
        store.max_stop_word_lengths_host[seq_slots_host] = torch.tensor(
            temp_data.max_stop_word_lengths, device="cpu", dtype=torch.int32
        )

        store.num_accepted_draft_tokens_host[seq_slots_host] = 0
        metadata = (
            torch.tensor(
                temp_data.metadata, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
            )
            .reshape(-1, 5)
            .to("cuda", non_blocking=True)
        )
        store.stop_words_metadata_cuda.index_copy_(0, seq_slots_cuda_long, metadata)

    def record_stop_word_stream(self) -> None:
        """Register streams that consume the indirectly referenced allocations."""
        stream = torch.cuda.current_stream()
        generation = self._stop_word_allocation_generation
        if self._stop_word_stream_generations.get(stream) == generation:
            return
        for allocation in self._stop_word_allocations.values():
            allocation.tensor.record_stream(stream)
        # The allocator records completion events when storage is released;
        # repeated uses on an already registered stream need no registration.
        self._stop_word_stream_generations[stream] = generation

    def write_finish_reasons(
        self,
        seq_slots_host: torch.Tensor,
        seq_slots_cuda: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        new_tokens_cuda: torch.Tensor,
        first_finish_reasons_cuda: torch.Tensor | None = None,
        pending_harvest_cuda: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculates the finish reasons for each request and returns the finish reasons tensor.

        Prepares stop word handling for each requests and processes all newly generated tokens
        per request to determine if any finish reason is met. Returns the device finish reasons
        tensor from the store, which is updated with the calculated finish reason for each newly
        generated token.

        Args:
            seq_slots_host: The sequence slots of the processed requests. Used to determine which
            requests need stop word processing on the host.
              Shape: [len(requests)]
            seq_slots_cuda: The sequence slots of the processed requests. Used for accessing device buffers.
              Shape: [len(requests)]
            seq_lens_cuda: The sequence lengths of the processed requests.
              Shape: [len(requests)]
            new_tokens_cuda: A buffer containing the newly generated tokens.
              Shape: [max_tokens, max_batch_size, max_beam_width]
            first_finish_reasons_cuda: The first finish reason of each beam. Used only for beam search.
              Shape: [max_batch_size, max_beam_width]
        Returns:
            finish_reasons_cuda: The finish reasons tensor.
              Shape: [max_tokens, max_batch_size, max_beam_width]
        """
        num_accepted_tokens_cuda, stop_word_indices_cuda, single_token_stop_words_only = (
            self._prepare_stop_word_handling_for_finish_reasons(seq_slots_host)
        )
        self._write_finish_reasons(
            seq_slots=seq_slots_cuda,
            seq_lens=seq_lens_cuda,
            new_tokens=new_tokens_cuda,
            num_accepted_tokens=num_accepted_tokens_cuda,
            stop_word_indices=stop_word_indices_cuda,
            single_token_stop_words_only=single_token_stop_words_only,
            first_finish_reasons=first_finish_reasons_cuda,
            pending_harvest=pending_harvest_cuda,
        )
        return self.store.finish_reasons_cuda

    def _prepare_stop_word_handling_for_finish_reasons(
        self,
        seq_slots_host: torch.Tensor,
    ) -> tuple[torch.Tensor | int | None, torch.Tensor | None, bool]:
        """Prepare stop word handling for finish reasons.

        Args:
            seq_slots_host: The sequence slots of the processed requests. Used for accessing host buffers.
              Shape: [len(requests)]
        Returns:
            num_accepted_tokens_cuda: The number of accepted draft tokens +1 for each request.
              Shape: [len(requests)] if torch.Tensor
            stop_word_indices_cuda: The indices of the requests that have stop words in the current batch.
              Shape: [len(requests_with_stop_words)]
            single_token_stop_words_only: Whether all stop words in this batch are of length 1.
        """
        # Filter all requests, that have stop words
        store = self.store
        num_accepted_tokens_cuda: torch.Tensor | int | None = None
        stop_word_indices_cuda: torch.Tensor | None = None
        single_token_stop_words_only: bool = False

        stop_word_mask = store.max_stop_word_lengths_host[seq_slots_host] > 0
        batch_has_stop_words = stop_word_mask.any()

        if batch_has_stop_words:
            num_accepted_tokens_cuda = 1
            # Only calculate num_accepted_tokens from the accepted draft tokens if speculative decoding is enabled
            if self._use_speculative_decoding:
                num_accepted_tokens_cuda = (
                    store.num_accepted_draft_tokens_host[seq_slots_host].to(
                        device="cuda", non_blocking=True
                    )
                    + 1
                )
            stop_word_indices_cuda = torch.nonzero(stop_word_mask)[:, 0].to(
                device="cuda", non_blocking=True
            )
            single_token_stop_words_only = (
                store.max_stop_word_lengths_host[seq_slots_host].max().item() == 1
            )
        return num_accepted_tokens_cuda, stop_word_indices_cuda, single_token_stop_words_only

    def _can_fuse_finish_reasons(
        self,
        *,
        seq_slots: torch.Tensor,
        seq_lens: torch.Tensor,
        new_tokens: torch.Tensor,
        stop_word_indices: torch.Tensor | None,
        first_finish_reasons: torch.Tensor | None,
    ) -> bool:
        """Return whether only end-ID and maximum-length checks are active.

        The tensor layouts below are invariants the caller already guarantees,
        so they are asserted rather than tested: a layout regression should be
        loud, not a silent fallback to the ten-kernel path.
        """
        if stop_word_indices is not None or first_finish_reasons is not None:
            return False
        assert seq_slots.dim() == 1 and seq_slots.is_contiguous()
        assert seq_lens.dim() == 1 and seq_lens.is_contiguous()
        assert seq_lens.numel() == seq_slots.numel()
        assert seq_slots.dtype == torch.int64 and seq_lens.dtype in (torch.int32, torch.int64)
        assert new_tokens.is_contiguous()
        # One row past the real slots: the sampler's new_tokens buffer carries a
        # scratch row for CUDA graph padding (see TorchSampler.dummy_slot_row).
        assert new_tokens.shape == (
            self._max_tokens,
            self._max_num_sequences + 1,
            self._max_beam_width,
        )
        return self._fused_tile_fits and seq_slots.numel() > 0

    @nvtx_range("_write_finish_reasons")
    def _write_finish_reasons(
        self,
        *,
        seq_slots: torch.Tensor,
        seq_lens: torch.Tensor,
        new_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor | int | None = None,
        stop_word_indices: torch.Tensor | None = None,
        single_token_stop_words_only: bool = False,
        first_finish_reasons: torch.Tensor | None = None,
        pending_harvest: torch.Tensor | None = None,
    ) -> None:
        """Writes the finish reasons to the finish_reasons tensor.

        The finish reasons are written to the finish_reasons tensor in the following order:
        - Stop words
        - Max length
        - End ID
        Later finish reasons overwrite earlier ones, in reverse precedence order.

        Args:
            seq_slots: The sequence slots of the processed requests. Used for accessing device buffers.
              Shape: [len(requests)]
            seq_lens: The sequence lengths of the processed requests.
              Shape: [len(requests)]
            new_tokens: A buffer containing the newly generated tokens.
              Shape: [max_tokens, max_batch_size, max_beam_width]
            num_accepted_tokens: A buffer containing the number of accepted draft tokens +1 for each request.
              Shape: [max_batch_size] if torch.Tensor
            stop_word_indices: The indices of the requests that have stop words in the current batch.
              Shape: [len(requests_with_stop_words)]
            single_token_stop_words_only: Whether all stop words in this batch are of length 1
            first_finish_reasons: The first finish reason of each beam.
              Shape: [max_batch_size, max_beam_width]
        """

        # Seq Slots should be on the same device as new_tokens
        assert seq_slots.device == new_tokens.device
        assert seq_lens.device == new_tokens.device

        store = self.store
        if self._can_fuse_finish_reasons(
            seq_slots=seq_slots,
            seq_lens=seq_lens,
            new_tokens=new_tokens,
            stop_word_indices=stop_word_indices,
            first_finish_reasons=first_finish_reasons,
        ):
            fused_write_finish_reasons(
                finish_reasons=store.finish_reasons_cuda,
                new_tokens=new_tokens,
                seq_slots=seq_slots,
                seq_lens=seq_lens,
                max_lengths=store.max_lengths_cuda,
                end_ids=store.end_ids_cuda,
                max_tokens=self._max_tokens,
                beam_width=self._max_beam_width,
                not_finished_value=FinishReason.NOT_FINISHED.value,
                length_value=FinishReason.LENGTH.value,
                end_id_value=FinishReason.END_ID.value,
            )
            return

        tokens = new_tokens[:, seq_slots]
        finish_reasons = store.finish_reasons_cuda

        # we need to fill with NOT_FINISHED so we can differentiate between
        # previous requests that had the same seq slot
        _ = finish_reasons.index_fill_(1, seq_slots, FinishReason.NOT_FINISHED.value)
        batched_finish_reasons = finish_reasons[:, seq_slots]

        if stop_word_indices is not None:
            assert num_accepted_tokens is not None, "draft_lengths is required for stop words"
            stop_seq_slots = seq_slots[stop_word_indices]
            stop_tokens = new_tokens[:, stop_seq_slots]
            stop_words_func = (
                self._are_stop_words
                if not single_token_stop_words_only
                else self._are_stop_words_single_token
            )
            batched_finish_reasons_stop_words = batched_finish_reasons[:, stop_word_indices]
            _ = batched_finish_reasons_stop_words.masked_fill_(
                stop_words_func(
                    stop_seq_slots,
                    stop_tokens,
                    num_accepted_tokens[stop_word_indices]
                    if isinstance(num_accepted_tokens, torch.Tensor)
                    else num_accepted_tokens,
                ),
                FinishReason.STOP_WORDS.value,
            )
            batched_finish_reasons[:, stop_word_indices] = batched_finish_reasons_stop_words

        _ = batched_finish_reasons.masked_fill_(
            self._are_max_length(seq_lens, store.max_lengths_cuda[seq_slots]),
            FinishReason.LENGTH.value,
        )

        _ = batched_finish_reasons.masked_fill_(
            self._are_end_id(store.end_ids_cuda[seq_slots], tokens),
            FinishReason.END_ID.value,
        )

        finish_reasons[:, seq_slots] = batched_finish_reasons
        if first_finish_reasons is not None:
            # store the first stop reason for each beam of a seq_slot.
            batched_first_finish_reasons = first_finish_reasons[seq_slots]
            newly_finished = (batched_first_finish_reasons == FinishReason.NOT_FINISHED.value) & (
                batched_finish_reasons != FinishReason.NOT_FINISHED.value
            )
            first_finish_reasons[seq_slots, ...] = torch.where(
                batched_first_finish_reasons == FinishReason.NOT_FINISHED.value,
                batched_finish_reasons,
                batched_first_finish_reasons,
            )
            if pending_harvest is not None:
                # Raise the beam-search harvest latch for beams that finished on
                # *this* step. The CBA step lowers it once it has pooled them;
                # first_finish_reasons itself cannot serve as the latch because
                # it must outlive the harvest to be reported to the caller.
                pending_harvest[seq_slots, ...] |= newly_finished.any(dim=0)

    def _are_end_id(self, end_ids_cuda: torch.Tensor, tokens_cuda: torch.Tensor) -> torch.Tensor:
        """Checks if the tokens are the end id

        Args:
            end_ids_cuda: The end ids of the requests to check the end id of.
              Shape: [len(requests)]
            tokens_cuda: A buffer containing the newly generated tokens.
              Shape: [max_tokens, len(requests), max_beam_width]
        Returns:
            A tensor where each element is True if the corresponding token is the end id, False otherwise
            Shape: [max_tokens, len(requests), max_beam_width]
        """
        return tokens_cuda == end_ids_cuda.view(1, -1, 1).expand(
            self._max_tokens, -1, self._max_beam_width
        )

    def _are_max_length(
        self, seq_lens_cuda: torch.Tensor, max_seq_lens_cuda: torch.Tensor
    ) -> torch.Tensor:
        """Checks which sequences are at or beyond the max length

        Args:
            seq_lens_cuda: The sequence lengths of the requests to check the max length of.
              Shape: [len(requests)]
            max_seq_lens_cuda: The maximum sequence lengths of the requests to check the max length of.
              Shape: [len(requests)]
        Returns:
            A tensor where each element is True if the sequence at the corresponding token
            is at or beyond the max length, False otherwise
            Shape: [max_tokens, len(requests), max_beam_width]
        """
        lengths_tensor_cuda = (seq_lens_cuda.view(1, -1, 1) + self._max_tokens_offset_cuda).expand(
            self._max_tokens, -1, self._max_beam_width
        )
        max_lengths_tensor_cuda = max_seq_lens_cuda.view(1, -1, 1).expand(
            self._max_tokens, -1, self._max_beam_width
        )
        return lengths_tensor_cuda >= max_lengths_tensor_cuda

    @nvtx_range("_are_stop_words")
    def _are_stop_words(
        self,
        seq_slots: torch.Tensor,
        tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor | int,
    ) -> torch.Tensor:
        """Checks if the tokens are stop words

        Args:
            seq_slots: The sequence slots of the processed requests. Used for accessing device buffers.
              Shape: [len(requests)]
            tokens: A buffer containing the newly generated tokens.
              Shape: [max_tokens, len(requests), max_beam_width]
            num_accepted_tokens: The number of accepted draft tokens +1 for each request.
              Shape: [len(requests)] if torch.Tensor
        Returns:
            A tensor where each element is True if the sequence at the corresponding token
            ends with a stop word, False otherwise
            Shape: [max_tokens, len(requests), max_beam_width]
        """
        self.record_stop_word_stream()
        return compact_stop_words(
            self.store.stop_words_metadata_cuda, seq_slots, tokens, num_accepted_tokens
        )

    @nvtx_range("_are_stop_words_single_token")
    def _are_stop_words_single_token(
        self,
        seq_slots: torch.Tensor,
        tokens: torch.Tensor,
        _num_accepted_tokens: torch.Tensor | int,
    ) -> torch.Tensor:
        """Checks if the tokens are stop words (single token per stop word only)

        Args:
            seq_slots: The sequence slots of the processed requests. Used for accessing device buffers.
              Shape: [len(requests)]
            tokens: A buffer containing the newly generated tokens.
              Shape: [max_tokens, len(requests), max_beam_width]
            _num_accepted_tokens: Unused
        Returns:
            A tensor where each element is True if the sequence at the corresponding token
            ends with a stop word, False otherwise
            Shape: [max_tokens, len(requests), max_beam_width]
        """
        self.record_stop_word_stream()
        return compact_stop_words(
            self.store.stop_words_metadata_cuda,
            seq_slots,
            tokens,
            _num_accepted_tokens,
            single_token_only=True,
        )
