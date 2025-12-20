# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import TYPE_CHECKING, Dict, Optional

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager, ResourceManager
from ..pyexecutor.scheduler import ScheduledRequests
from .drafter import Drafter

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SuffixDecodingConfig


class SuffixResourceManager(BaseResourceManager):
    """
    Resource manager for Suffix Decoding. This class maintains the SuffixDecodingCache
    for managing request outputs, evicting old requests, and managing per-prompt suffix trees.
    Based on Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    """

    def __init__(self, spec_config: "SuffixDecodingConfig", max_num_requests: int):
        self.spec_config = spec_config
        self.max_num_requests = max_num_requests
        self.max_tree_depth = spec_config.max_tree_depth

        # Lazy import to avoid error when Suffix Decoding is not used.
        try:
            from arctic_inference.suffix_decoding import SuffixDecodingCache
        except ImportError:
            raise ImportError(
                "arctic_inference package is required for Suffix Decoding. "
                "Please install it with: pip install arctic-inference"
            )

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=spec_config.max_tree_depth,
            max_cached_requests=spec_config.max_cached_requests,
        )
        # Track the last token count for each request to determine newly sampled tokens
        self.last_token_counts: Dict[int, int] = {}

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        """
        Prepare resources for Suffix Decoding for requests in the scheduled batch.
        """
        for req in scheduled_batch.generation_requests:
            req_id = req.request_id
            token_ids = list(req.get_tokens(0))
            num_prompt_tokens = req.py_orig_prompt_len
            num_tokens = len(token_ids)

            if req_id not in self.suffix_cache.active_requests:
                # TODO: Check if we need to evict cache response when the same req_id is reused,
                # e.g., when a request is interrupted and then restarted with different content.
                # Not sure if this case can actually happen.
                if req_id in self.suffix_cache.cached_requests:
                    self.suffix_cache.evict_cached_response(req_id)

                # Get prompt tokens for initialization
                prompt_token_ids = token_ids[:num_prompt_tokens]

                # Start a new request, which will build a suffix tree for this prompt
                self.suffix_cache.start_request(req_id, prompt_token_ids)

                # Initialize the token count for new requests
                # If there are already generated tokens, they will be added in the unified logic below
                self.last_token_counts[req_id] = num_prompt_tokens

            # update the suffix cache with newly sampled tokens
            last_token_count = self.last_token_counts.get(req_id, num_prompt_tokens)
            if num_tokens > last_token_count:
                # Get the newly sampled token(s) (tokens added since last call)
                newly_sampled_ids = token_ids[last_token_count:num_tokens]
                # Append the newly sampled ids to the suffix cache for this request
                self.suffix_cache.add_active_response(req_id, newly_sampled_ids)
                # Update the last token count
                self.last_token_counts[req_id] = num_tokens
            elif num_tokens == last_token_count:
                # No new tokens were sampled (might happen in edge cases)
                pass
            else:
                # This shouldn't happen, but handle gracefully
                logger.warning(
                    f"Request {req_id}: token count decreased from {last_token_count} "
                    f"to {num_tokens}. Resetting token count."
                )
                self.last_token_counts[req_id] = num_tokens

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        """
        Free resources occupied by the specified request.
        """
        req_id = request.request_id
        if req_id in self.suffix_cache.active_requests:
            self.suffix_cache.stop_request(req_id)
        # Clean up last_token_counts for completed requests
        if req_id in self.last_token_counts:
            del self.last_token_counts[req_id]


class SuffixDrafter(Drafter):
    """
    Drafter for Suffix Decoding. This class uses SuffixCache to propose
    speculative tokens for each request.
    """

    def __init__(
        self,
        spec_config,
        suffix_cache_manager: SuffixResourceManager = None,
    ):
        super().__init__(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_draft_len,
            max_concurrency=spec_config.max_concurrency,
            draft_len_schedule=spec_config.draft_len_schedule,
        )
        assert suffix_cache_manager is not None, (
            "SuffixDecoding needs a resource manager to maintain the suffix cache."
        )
        self.spec_resource_manager = suffix_cache_manager
        self.spec_config = spec_config
        self.max_draft_len = spec_config.max_draft_len
        self.max_total_draft_tokens = spec_config.max_draft_len
        self.max_tree_depth = spec_config.max_tree_depth
        self.max_spec_factor = spec_config.max_spec_factor
        self.min_token_prob = spec_config.min_token_prob

    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        """
        Prepare draft tokens for the scheduled requests using Suffix Decoding.
        Suffix Decoding will speculate a dynamic number of tokens for each request
        every decoding step, so each entry may have different lengths.
        """
        # Sort by request_id when py_batch_idx is None as a fallback.
        # This happens in the disagg case: for a set of new requests, we draft
        # before forward_step, so py_batch_idx is not assigned.
        for request in sorted(
            scheduled_requests.generation_requests,
            key=lambda r: (r.py_batch_idx is None, r.py_batch_idx or r.request_id),
        ):
            req_id = request.request_id

            # Skip requests that have already reached the max sequence length.
            num_tokens = len(request.get_tokens(0))
            max_sequence_length = request.py_orig_prompt_len + request.py_max_new_tokens
            if num_tokens >= max_sequence_length:
                request.py_draft_tokens = []
                continue

            # Get the current token sequence
            token_ids = list(request.get_tokens(0))

            # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
            # we extract the pattern from the end of the input.
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = token_ids[start:num_tokens]

            # Speculate draft tokens
            draft = self.spec_resource_manager.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(self.max_draft_len, max_sequence_length - num_tokens - 1),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            draft_token_ids = draft.token_ids if draft else []
            request.py_draft_tokens = draft_token_ids

    def update_max_total_draft_tokens(self, new_max_total_draft_tokens: int) -> None:
        """Override to propagate to SuffixResourceManager."""
        super().update_max_total_draft_tokens(new_max_total_draft_tokens)
        # Suffix Decoding uses dynamic draft lengths, so we don't need to update
        # the resource manager's max_total_draft_tokens
