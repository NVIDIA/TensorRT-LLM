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
(stop-word rules, the past-token window) and their resize/refresh logic.
``TorchSampler`` holds one :class:`FinishReasonsHandler` and drives it
through request admission and the per-step write.
"""

from dataclasses import dataclass

import torch

from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.bindings.executor import FinishReason

from ...utils import torch_multi_arange
from ..llm_request import LlmRequest
from .sampler_common import int_tensor

__all__ = ["FinishReasonsHandler"]


class FinishReasonsHandler:
    _EMPTY_STOP_WORD_TOKEN_ID: int = -2
    _PAD_STOP_WORD_TOKEN_ID: int = -1

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
        stop_words_cuda: torch.Tensor
        """Shape: [max_num_stop_words, max_stop_word_length, batch_size]
        Usage: Stores the stop words for each request as a padded tensor."""
        past_tokens_cuda: torch.Tensor
        """Shape: [max_stop_word_length,batch_size, beam_width]
        Usage: Stores the last max_stop_word_length tokens for each beam."""
        max_stop_word_lengths_host: torch.Tensor
        """Shape: [batch_size]
        Usage: Stores the size of the longest stop word for each request."""
        num_accepted_draft_tokens_host: torch.Tensor
        """Shape: [batch_size]
        Usage: Stores the number of accepted tokens for each request."""

    def __init__(
        self,
        *,
        max_stop_word_length: int,
        max_num_stop_words: int,
        max_num_sequences: int,
        max_beam_width: int,
        max_tokens: int,
        max_seq_len: int,
    ):
        self._update_sizes(
            max_stop_word_length=max_stop_word_length,
            max_num_stop_words=max_num_stop_words,
            max_num_sequences=max_num_sequences,
            max_beam_width=max_beam_width,
            max_tokens=max_tokens,
            max_seq_len=max_seq_len,
        )
        self._setup_store()
        self._setup_helper_tensors()
        self._temp_data: FinishReasonsHandler._TemporaryData = self._TemporaryData()

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
        max_stop_word_length: int,
        max_num_stop_words: int,
        max_num_sequences: int,
        max_beam_width: int,
        max_tokens: int,
        max_seq_len: int,
    ) -> None:
        """Updates the sizes of the finish reasons handler

        Sets member variables to store the current sizes.
        These sizes are used to initialize the buffer tensors.
        """
        self._max_stop_word_length: int = max_stop_word_length
        self._max_num_stop_words: int = max_num_stop_words
        self._max_num_sequences: int = max_num_sequences
        self._max_beam_width: int = max_beam_width
        self._max_tokens: int = max_tokens
        self._max_seq_len: int = max_seq_len
        self._stop_words_shape: tuple[int, int, int] = (
            self._max_num_stop_words,
            self._max_stop_word_length,
            self._max_num_sequences,
        )
        self._past_tokens_shape: tuple[int, int, int] = (
            self._max_stop_word_length - 1 + self._max_tokens,
            self._max_num_sequences,
            self._max_beam_width,
        )

    def _setup_store(self) -> None:
        """Sets up the store for the finish reasons handler by initializing all buffer tensors."""
        finish_reasons_cuda = int_tensor(
            (self._max_tokens, self._max_num_sequences, self._max_beam_width)
        )
        max_lengths_cuda = int_tensor((self._max_num_sequences,))
        end_ids_cuda = int_tensor((self._max_num_sequences,))
        stop_words_cuda = int_tensor(self._stop_words_shape)
        past_tokens_cuda = int_tensor(self._past_tokens_shape)
        max_stop_word_lengths_host = torch.empty(
            self._max_num_sequences, device="cpu", dtype=torch.int32
        )
        num_accepted_draft_tokens_host = torch.empty(
            self._max_num_sequences, device="cpu", dtype=torch.int32
        )
        self.store: FinishReasonsHandler._FinishReasonsStore = self._FinishReasonsStore(
            finish_reasons_cuda=finish_reasons_cuda,
            max_lengths_cuda=max_lengths_cuda,
            end_ids_cuda=end_ids_cuda,
            stop_words_cuda=stop_words_cuda,
            past_tokens_cuda=past_tokens_cuda,
            max_stop_word_lengths_host=max_stop_word_lengths_host,
            num_accepted_draft_tokens_host=num_accepted_draft_tokens_host,
        )

    def _setup_helper_tensors(self) -> None:
        # Helper tensors for finish_reasons:
        """Preallocate buffer needed for torch.nonzero_static(..., out=finish_reasons_nonzero_static_buffer).
        See `def _write_reason`."""
        # setup local buffer for max tokens checking
        self._max_tokens_offset_cuda: torch.Tensor = torch.arange(
            1, self._max_tokens + 1, device="cuda", dtype=torch.int32
        ).view(-1, 1, 1)

        self._stop_words_index_offset_cuda: torch.Tensor = torch.arange(
            max(0, self._max_stop_word_length - 1), device="cuda"
        ).unsqueeze(1)

        self._past_token_buffer_cuda: torch.Tensor = torch.empty(
            self._past_tokens_shape, device="cuda", dtype=torch.int32
        )
        starts = torch.arange(self._max_tokens, device="cuda")
        ends = starts + self._max_stop_word_length
        self._multi_arange_indexing: torch.Tensor = torch_multi_arange(
            ends=ends,
            starts=starts,
            output_length=self._max_tokens * self._max_stop_word_length,
        )

    def _resize_stop_word_buffers(self) -> None:
        self._stop_words_index_offset_cuda = torch.arange(
            max(0, self._max_stop_word_length - 1), device="cuda"
        ).unsqueeze(1)

        self._past_tokens_shape = (
            self._max_stop_word_length - 1 + self._max_tokens,
            self._max_num_sequences,
            self._max_beam_width,
        )
        self._stop_words_shape = (
            self._max_num_stop_words,
            self._max_stop_word_length,
            self._max_num_sequences,
        )
        self._past_token_buffer_cuda = torch.empty(
            self._past_tokens_shape, device="cuda", dtype=torch.int32
        )
        starts = torch.arange(self._max_tokens, device="cuda")
        ends = starts + self._max_stop_word_length
        self._multi_arange_indexing = torch_multi_arange(
            ends=ends,
            starts=starts,
            output_length=self._max_tokens * self._max_stop_word_length,
        )
        # resize the stop words buffer if necessary
        # if the sizes are constant, this does nothing
        store = self.store
        _ = store.stop_words_cuda.resize_(self._stop_words_shape)
        _ = store.past_tokens_cuda.resize_(self._past_tokens_shape)

    @dataclass(kw_only=True)
    class _TemporaryData:
        """Data structure to store the temporary data during setup_sampler_step for new requests"""

        def __init__(self) -> None:
            # list of device tensors
            self.stop_words_cuda_list: list[torch.Tensor] = []
            self.past_tokens_cuda_list: list[torch.Tensor] = []
            # list of integers
            self.stop_word_seq_slots: list[int] = []
            self.max_lens: list[int] = []
            self.end_ids: list[int] = []
            self.max_stop_word_lengths: list[int] = []
            # integers
            self.total_max_length: int = 0
            self.total_max_num_stop_words: int = 0

        def clear(self) -> None:
            self.stop_words_cuda_list = []
            self.past_tokens_cuda_list = []
            self.stop_word_seq_slots = []
            self.max_lens = []
            self.end_ids = []
            self.max_stop_word_lengths = []
            self.total_max_length = 0
            self.total_max_num_stop_words = 0

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

        if (stop_words_list := request.py_stop_words_list) is not None:
            assert (seq_slot := request.py_seq_slot) is not None
            self._temp_data.stop_word_seq_slots.append(seq_slot)
            extracted_stop_words_cuda, max_length, num_stop_words = self._extract_stop_words(
                stop_words_list
            )
            self._temp_data.stop_words_cuda_list.append(extracted_stop_words_cuda)
            self._temp_data.past_tokens_cuda_list.append(self._get_past_tokens(request))
            self._temp_data.total_max_length = max(self._temp_data.total_max_length, max_length)
            self._temp_data.total_max_num_stop_words = max(
                self._temp_data.total_max_num_stop_words, num_stop_words
            )
            self._temp_data.max_stop_word_lengths.append(max_length)
        else:
            # max stop word length is used to determine if a request has stop words
            # explicitly set it to 0 here to avoid stale data from previous requests
            self._temp_data.max_stop_word_lengths.append(0)

    def update_for_new_request(
        self,
        *,
        seq_slots_cuda_long: torch.Tensor,
        max_lengths_cuda: torch.Tensor,
        end_ids_cuda: torch.Tensor,
        seq_slots_host: torch.Tensor,
        all_sampling_requests: list[LlmRequest],
    ) -> None:
        """Update tensors of this store with the new request data.

        If stop words are present, also update the stop words buffers.
        If the new stop words exceed either the current max_num_stop_words or max_stop_word_length values,
        a resize of the stop words buffers is triggered. If a resize is necessary, all requests in the batch
        need to be re-processed.

        Args:
            seq_slots_cuda_long: The sequence slots of the processed requests, as int64
              CUDA indices (required by ``index_copy_``). Shape: [len(requests)]
            max_lengths_cuda: The maximum lengths for each request.
              Shape: [len(requests)]
            end_ids_cuda: The end ids for each request.
              Shape: [len(requests)]
            seq_slots_host: The sequence slots of the processed requests. Used for accessing host buffers.
              Shape: [len(requests)]
            all_sampling_requests: If a resize of the stop words related buffers is necessary, all sampling requests
                need to be re-processed.
        """

        temp_data = self._temp_data
        store = self.store
        store.max_lengths_cuda.index_copy_(0, seq_slots_cuda_long, max_lengths_cuda)
        store.end_ids_cuda.index_copy_(0, seq_slots_cuda_long, end_ids_cuda)
        store.max_stop_word_lengths_host[seq_slots_host] = torch.tensor(
            temp_data.max_stop_word_lengths, device="cpu", dtype=torch.int32
        )

        # Handle stop words only if any new ones are added
        if temp_data.stop_word_seq_slots:
            self._update_stop_words_buffer(
                all_sampling_requests,
                temp_data.total_max_length,
                temp_data.total_max_num_stop_words,
                temp_data.stop_words_cuda_list,
                temp_data.past_tokens_cuda_list,
                temp_data.stop_word_seq_slots,
            )

    def _maybe_resize_stop_words_buffer(
        self, total_max_length: int, total_max_num_stop_words: int
    ) -> bool:
        """Checks if the stop words buffer needs to be resized and resizes it if necessary

        If the total maximum length or number of stop words exceeds the current maximum values,
        the stop words buffer is resized to the new maximum values.

        Args:
            total_max_length: The maximum length of the stop words in this batch.
            total_max_num_stop_words: The maximum number of stop words of a request in this batch.
        Returns:
            True if the stop words buffer needs to be resized, False otherwise.
        """
        if (
            total_max_length > self._max_stop_word_length
            or total_max_num_stop_words > self._max_num_stop_words
        ):
            self._max_stop_word_length = max(total_max_length, self._max_stop_word_length)
            self._max_num_stop_words = max(total_max_num_stop_words, self._max_num_stop_words)
            self._resize_stop_word_buffers()
            return True
        return False

    def _reprocess_stop_words_buffer(
        self, requests: list[LlmRequest]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        """Reprocesses the stop words buffer with the new maximum values

        If a resize of the stop words buffer is necessary, all requests in the batch need to be re-processed.

        Args:
            requests: The requests to reprocess the stop words buffer for.
        Returns:
            stop_words_cuda_list: A list of device tensors containing the stop words per request with stop words.
            past_tokens_cuda_list: A list of device tensors containing the past tokens per request with stop words.
            stop_word_seq_slots: A list of sequence slot indices (int) per request with stop words.
        """
        stop_words_cuda_list: list[torch.Tensor] = []
        past_tokens_cuda_list: list[torch.Tensor] = []
        stop_word_seq_slots: list[int] = []
        # Rerun with the new size. Set the stop words and past tokens for all the requests.
        for request in requests:
            if (stop_words_list := request.py_stop_words_list) is not None:
                extracted_stop_words_cuda, _, _ = self._extract_stop_words(stop_words_list)
                assert (seq_slot := request.py_seq_slot) is not None
                stop_word_seq_slots.append(seq_slot)
                stop_words_cuda_list.append(extracted_stop_words_cuda)
                past_tokens_cuda_list.append(self._get_past_tokens(request))
        return stop_words_cuda_list, past_tokens_cuda_list, stop_word_seq_slots

    def _update_stop_words_buffer(
        self,
        all_sampling_requests: list[LlmRequest],
        total_max_length: int,
        total_max_num_stop_words: int,
        stop_words_cuda_list: list[torch.Tensor],
        past_tokens_cuda_list: list[torch.Tensor],
        stop_word_seq_slots: list[int],
    ) -> None:
        """Updates the stop words buffer with the new maximum values

        Args:
            all_sampling_requests: If a resize of the stop words related buffers is necessary, all sampling requests
                need to be re-processed.
            total_max_length: The maximum length of the stop words in this batch.
            total_max_num_stop_words: The maximum number of stop words of a request in this batch.
            stop_words_cuda_list: A list of device tensors containing the stop words per request with stop words.
            past_tokens_cuda_list: A list of device tensors containing the past tokens per request with stop words.
            stop_word_seq_slots: Sequence slot index (int) per request with stop words;
              same order as the lists above.
        """
        # Potentially resize the buffers and update
        # stop_words, past_tokens and stop_word_seq_slots
        # In case of a resize all requests in the batch need to be re-processed
        if self._maybe_resize_stop_words_buffer(total_max_length, total_max_num_stop_words):
            stop_words_cuda_list, past_tokens_cuda_list, stop_word_seq_slots = (
                self._reprocess_stop_words_buffer(all_sampling_requests)
            )

        # Host Tensor for host access of self.store.num_accepted_draft_tokens
        stop_word_seq_slots_tensor_host = torch.tensor(
            stop_word_seq_slots, device="cpu", dtype=torch.int32, pin_memory=prefer_pinned()
        )
        # Device Tensor for device access of self.store.stop_words and self.store.past_tokens
        stop_word_seq_slots_tensor_cuda = stop_word_seq_slots_tensor_host.to(
            device="cuda", non_blocking=True
        )
        # stop_word_seq_slots x max_num_stop_words x max_stop_word_length
        stop_words_cuda_tensor = torch.stack(stop_words_cuda_list)
        # stop_word_seq_slots x max_stop_word_length x beam_width
        past_tokens_cuda_tensor = torch.stack(past_tokens_cuda_list)

        store = self.store
        # Reset the accepted tokens buffer for the stop word sequence slots
        store.num_accepted_draft_tokens_host[stop_word_seq_slots_tensor_host] = 0

        store.stop_words_cuda[..., stop_word_seq_slots_tensor_cuda] = (
            stop_words_cuda_tensor.permute(1, 2, 0)
        )
        # Past tokens will be shifted by 1 to the left on their first sampling iteration
        # We need to consider this here.
        store.past_tokens_cuda[1 : self._max_stop_word_length, stop_word_seq_slots_tensor_cuda] = (
            past_tokens_cuda_tensor.permute(1, 0, 2)
        )

    def _extract_stop_words(
        self, stop_words_list: list[list[int]]
    ) -> tuple[torch.Tensor, int, int]:
        """Extract the stop words and size information from the stop words list

        Processes the stop words list and stores the stop words in a padded device tensor.
        Stop words shorter than FinishReasonsHandler.max_stop_word_length
        are padded with _PAD_STOP_WORD_TOKEN_ID to max_stop_word_length.
        Unused stop word slots are padded with _EMPTY_STOP_WORD_TOKEN_ID.
        This function additionally returns the maximum stop word length and the number of stop words
        in the processed stop words list.


        Args:
            stop_words_list: A list of stop sequences, each a list of token ids.

        Returns:
            stop_words: A padded device tensor containing the stop words
              Shape: [max_num_stop_words, max_stop_word_length]
            max_stop_word_length: The maximum stop word length in the stop words list
            num_stop_words: The number of stop words in the stop words list
        """
        stop_words_host = torch.empty(
            self._max_num_stop_words,
            self._max_stop_word_length,
            device="cpu",
            dtype=torch.int32,
        )
        _ = stop_words_host.fill_(self._EMPTY_STOP_WORD_TOKEN_ID)
        max_stop_word_length = 0
        num_stop_words = 0
        for idx, word in enumerate(stop_words_list):
            length = len(word)
            max_stop_word_length = max(max_stop_word_length, length)
            num_stop_words += 1
            # skip processing if either the length or the index is greater than the current max values.
            # These will be updated outside this function.
            if length > self._max_stop_word_length or idx >= self._max_num_stop_words:
                continue
            stop_words_host[idx, -length:] = torch.tensor(word, dtype=torch.int32)
            stop_words_host[idx, :-length] = self._PAD_STOP_WORD_TOKEN_ID
        return (
            stop_words_host.to("cuda", non_blocking=True),
            max_stop_word_length,
            num_stop_words,
        )

    def _get_past_tokens(self, request: LlmRequest) -> torch.Tensor:
        """Get the past tokens from the request and return the past tokens device tensor

        Args:
            request: The request to get the past tokens for

        Returns:
            past_tokens: The past tokens device tensor
              Shape: [max_stop_word_length - 1, max_beam_width]
        """
        past_tokens_host = torch.zeros(
            max(0, self._max_stop_word_length - 1),
            self._max_beam_width,
            device="cpu",
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )
        tokens = request.get_tokens()
        for beam_idx in range(self._max_beam_width):
            max_len = min(past_tokens_host.shape[0], len(tokens[beam_idx]))
            past_tokens_host[past_tokens_host.shape[0] - max_len :, beam_idx] = torch.tensor(
                tokens[beam_idx][len(tokens[beam_idx]) - max_len :],
                device="cpu",
                dtype=torch.int32,
            )
        return past_tokens_host.to("cuda", non_blocking=True)

    def write_finish_reasons(
        self,
        seq_slots_host: torch.Tensor,
        is_draft_batch: bool,
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
            is_draft_batch: Whether the batch consists of draft requests.
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
            self._prepare_stop_word_handling_for_finish_reasons(
                seq_slots_host,
                is_draft_batch,
            )
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
        is_draft_batch: bool,
    ) -> tuple[torch.Tensor | int | None, torch.Tensor | None, bool]:
        """Prepare stop word handling for finish reasons.

        Args:
            seq_slots_host: The sequence slots of the processed requests. Used for accessing host buffers.
              Shape: [len(requests)]
            is_draft_batch: Whether the batch consists of draft requests.
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

        # NB: is_draft_batch is a workaround
        # as draft requests can be in the sampler
        # without having setup a slot in the FinishReasonsHandler.
        # These can be removed once this can be avoided.
        if is_draft_batch:
            # Do not process stop words for draft requests
            return (
                num_accepted_tokens_cuda,
                stop_word_indices_cuda,
                single_token_stop_words_only,
            )

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
        tokens = new_tokens[:, seq_slots]

        store = self.store
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
        store = self.store
        # num_words, len_words, batch_size
        # unsqueeze the beam_width dimension to match the past tokens tensor
        stop_words = (
            store.stop_words_cuda[..., seq_slots]
            .unsqueeze(3)
            .expand(-1, -1, -1, self._max_beam_width)
        )
        # Get the past tokens
        # num_steps, batch_size, beam_width
        past_tokens_batch = store.past_tokens_cuda[:, seq_slots]
        # Shift the past tokens to the left by the number of accepted draft tokens

        full_tokens = self._past_token_buffer_cuda[:, : seq_slots.shape[0]]

        index_tensor = (
            (self._stop_words_index_offset_cuda + num_accepted_tokens)
            .unsqueeze(2)
            .expand(-1, seq_slots.shape[0], self._max_beam_width)
        )
        _ = torch.gather(
            past_tokens_batch,
            dim=0,
            index=index_tensor,
            out=full_tokens[: index_tensor.shape[0]],
        )
        # Fill in the new tokens at the end of the past tokens buffer
        full_tokens[-self._max_tokens :] = tokens
        # short words are padded with _PAD_STOP_WORD_TOKEN_ID, so we need to mask them
        mask = stop_words == self._PAD_STOP_WORD_TOKEN_ID
        matches = torch.empty(
            (
                self._max_tokens,
                stop_words.shape[0],
                stop_words.shape[1],
                stop_words.shape[2],
                stop_words.shape[3],
            ),
            device="cuda",
            dtype=torch.bool,
        )

        # Get the comparison sequence for each step
        full_tokens_for_match = full_tokens[self._multi_arange_indexing].view(
            self._max_tokens,
            1,  # Unsqueeze on dimension 1 to match the num_stop_words dimension of stop words
            self._max_stop_word_length,
            seq_slots.shape[0],
            self._max_beam_width,
        )
        # Unsqueeze on dimension 0 to match the max_tokens dimension of full tokens
        stop_words_for_match = stop_words.unsqueeze(0)
        _ = torch.eq(full_tokens_for_match, stop_words_for_match, out=matches)
        # Mask the padding tokens
        _ = matches.masked_fill_(mask.unsqueeze(0).expand(self._max_tokens, -1, -1, -1, -1), True)
        # Update the past tokens storage for the next iteration
        store.past_tokens_cuda[:, seq_slots] = full_tokens
        # Return the result
        word_len_dim = 2
        num_words_dim = 1
        return torch.any(matches.all(dim=word_len_dim), dim=num_words_dim)

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
        per_step = torch.zeros(
            (self._max_tokens, seq_slots.shape[0], self._max_beam_width),
            dtype=torch.bool,
            device="cuda",
        )
        # num_words, 1, batch_size
        stop_words = self.store.stop_words_cuda[:, -1:, seq_slots].unsqueeze(3)
        full_tokens = tokens.unsqueeze(0)
        matches = full_tokens == stop_words
        _ = torch.any(matches, dim=0, out=per_step)
        return per_step
