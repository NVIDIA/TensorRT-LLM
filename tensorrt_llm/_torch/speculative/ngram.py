from itertools import chain
from typing import List, Optional, Tuple

import torch

from tensorrt_llm.llmapi import NGramDecodingConfig
from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import BaseResourceManager, ResourceManager
from ..pyexecutor.scheduler import ScheduledRequests
from .drafter import Drafter
from .suffix_automaton import SAConfig, SuffixAutomatonManager


class NGramPoolManager(BaseResourceManager):
    """
    Manager for NGram speculative decoding using suffix automaton.

    Uses a suffix automaton (SA) to efficiently find matching patterns in
    previously generated tokens. This provides CUDA-compatible pattern matching
    with O(L) lookup complexity where L is the match length.

    The SA maintains per-request state and supports two matching modes:
    - Fixed-size ngram: Tries ngram sizes N, N-1, ..., 1 until a match is found
    - Longest match: Returns the longest matching suffix (max_matching_ngram_size=-1)

    Arguments:
        max_total_draft_tokens: int
            The maximum number of draft tokens to generate.

        max_matching_ngram_size: int
            The ngram size for pattern matching:
            - Positive value (e.g., 3): Fixed-size ngram matching
            - -1: Use suffix automaton for longest possible match
    """

    def __init__(self, spec_config: "NGramDecodingConfig",
                 max_num_requests: int):
        self.max_total_draft_tokens = spec_config.max_total_draft_tokens
        self.max_matching_ngram_size = spec_config.max_matching_ngram_size
        self.max_num_requests = max_num_requests

        # Initialize suffix automaton manager
        sa_config = SAConfig(
            max_seq_len=262144,  # Reasonable default for LLM sequences
            max_slots=max_num_requests,
            threshold=1  # Minimum match length
        )
        self._sa_manager = SuffixAutomatonManager(sa_config, max_num_requests)

        # Track token counts for incremental updates
        self._token_counts: dict[int, int] = {}

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def update_resources(self, scheduled_batch: ScheduledRequests):
        # Remove completed requests
        for request in chain(scheduled_batch.context_requests,
                             scheduled_batch.generation_requests):
            if request.state == LlmRequestState.GENERATION_COMPLETE:
                request_id = request.request_id
                self._sa_manager.remove_request(request_id)
                self._token_counts.pop(request_id, None)

    def get_draft_tokens(
        self,
        prefix: list[int],
        request_id: int,
        max_sequence_length: int,
    ):
        """
        Get draft tokens using suffix automaton pattern matching.

        Uses the native CUDA kernel via extend_ngram() with batch size 1.
        This ensures we use the native implementation rather than Python fallback.

        Args:
            prefix: List of token IDs generated so far
            request_id: Unique identifier for this request
            max_sequence_length: Maximum allowed sequence length

        Returns:
            List of draft token IDs
        """
        prefix_len = len(prefix)
        max_draft_len = min(
            self.max_total_draft_tokens,
            max_sequence_length - 1 - prefix_len
        )

        if max_draft_len <= 0:
            return []

        # Initialize or update the SA state for this request
        if request_id not in self._token_counts:
            # New request - build SA from scratch
            self._sa_manager.add_request(request_id, prefix)
            self._token_counts[request_id] = prefix_len
            # No new tokens to extend, just do lookup
            new_tokens = []
        else:
            # Existing request - get new tokens to extend
            prev_count = self._token_counts[request_id]
            if prefix_len > prev_count:
                new_tokens = prefix[prev_count:]
                self._token_counts[request_id] = prefix_len
            else:
                new_tokens = []

        # Prepare batch for single request
        self._sa_manager.prepare([request_id], max_draft_len)

        # Create tensors for single-request batch
        # accepted_tokens contains the new tokens to extend the SA with
        num_new = len(new_tokens)
        accepted_tokens = torch.zeros((1, max(num_new, 1)),
                                      dtype=torch.int32,
                                      device='cuda')
        if num_new > 0:
            accepted_tokens[0, :num_new] = torch.tensor(new_tokens,
                                                        dtype=torch.int32)
        num_accepted_tokens = torch.tensor([num_new],
                                           dtype=torch.int32,
                                           device='cuda')

        # Use native kernel via extend_ngram with batch size 1
        match_len, draft_tokens = self._sa_manager.extend_ngram(
            [request_id],
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=self.max_matching_ngram_size
        )

        # Extract results from tensors
        match_length = match_len[0].item()
        if match_length == 0:
            return []

        # Convert draft tokens tensor to list
        draft_list = draft_tokens[0, :max_draft_len].tolist()

        # Trim trailing zeros (padding)
        while draft_list and draft_list[-1] == 0:
            draft_list.pop()

        return draft_list

    def print_pool(self):  # For debug
        """Print debug information about the SA states."""
        logger.debug(f"NGramPoolManager using SuffixAutomaton")
        logger.debug(f"Active requests: {len(self._token_counts)}")
        for request_id, token_count in self._token_counts.items():
            logger.debug(f"  Request {request_id}: {token_count} tokens")

    def prepare_batch_draft_tokens(
        self,
        request_ids: List[int],
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        max_draft_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched draft token generation using the CUDA kernel.

        This method is CUDA graph compatible when the native kernel is available.
        It extends the SA states with accepted tokens and performs ngram matching
        in a single batched GPU operation.

        Args:
            request_ids: List of request IDs in the batch
            accepted_tokens: [batch_size, max_draft_len + 1] tensor of accepted tokens
            num_accepted_tokens: [batch_size] tensor of number of accepted tokens per request
            max_draft_len: Maximum number of draft tokens to generate

        Returns:
            Tuple of (match_len, draft_tokens) tensors:
            - match_len: [batch_size] tensor of match lengths
            - draft_tokens: [batch_size, max_draft_len] tensor of draft token IDs
        """
        # Prepare batch indices for the SA manager
        self._sa_manager.prepare(request_ids, max_draft_len)

        # Use the batched extend_ngram method
        return self._sa_manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=self.max_matching_ngram_size
        )


class NGramDrafter(Drafter):

    def __init__(
        self,
        spec_config: NGramDecodingConfig,
        ngram_pool_manager: NGramPoolManager = None,
    ):
        super().__init__(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            max_concurrency=spec_config.max_concurrency,
            draft_len_schedule=spec_config.draft_len_schedule)
        assert ngram_pool_manager is not None, "NGram needs a resource manager to maintain the pool."
        self.spec_resource_manager = ngram_pool_manager
        self.spec_config = spec_config
        self.max_draft_len = spec_config.max_draft_len
        self.max_total_draft_tokens = spec_config.max_total_draft_tokens
        assert self.max_draft_len == self.max_total_draft_tokens, "NGram only supports linear tree."

    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        # Sort by request_id when py_batch_idx is None as a fallback.
        # This happens in the disagg case: for a set of new requests, we draft
        # before forward_step, so py_batch_idx is not assigned.
        for request in sorted(
                scheduled_requests.generation_requests,
                key=lambda r:
            (r.py_batch_idx is None, r.py_batch_idx or r.request_id),
        ):
            # Add new token to a copy of the generated tokens to find new draft tokens
            prefix = list(request.get_tokens(0))  # Get a copy

            # Generate draft tokens
            draft_tokens = self.spec_resource_manager.get_draft_tokens(
                prefix,
                request.request_id,
                max_sequence_length=request.py_orig_prompt_len +
                request.py_max_new_tokens,
            )
            request.py_draft_tokens = draft_tokens

    def update_max_total_draft_tokens(self,
                                      new_max_total_draft_tokens: int) -> None:
        """Override to propagate to NGramPoolManager."""
        super().update_max_total_draft_tokens(new_max_total_draft_tokens)
        self.spec_resource_manager.max_total_draft_tokens = new_max_total_draft_tokens
