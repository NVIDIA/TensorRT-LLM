from dataclasses import dataclass
from itertools import chain
from typing import List

from ordered_set import OrderedSet

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import *
from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.sampler import SampleState, TorchSampler
from ..pyexecutor.scheduler import ScheduledRequests
from .interface import SpecConfig, SpecMetadata, SpeculativeDecodingMode


@dataclass
class NGramConfig(SpecConfig):
    """
    Configuration for NGram drafter.
    """
    # The name of speculative decoding.
    spec_dec_name = "NGRAM"

    num_extra_kv_tokens: int = 0
    max_draft_tokens: int = 0

    prompt_lookup_num_tokens: int = 5
    max_matching_ngram_size: int = 5
    end_id: int = -1
    is_keep_all: bool = True
    is_use_oldest: bool = True
    is_public_pool: bool = True

    def __post_init__(self) -> None:
        self.spec_dec_mode = SpeculativeDecodingMode.from_string(
            self.spec_dec_name)
        self.max_draft_tokens = self.prompt_lookup_num_tokens

    def update_from_model_config(self, model_config):
        pass


class NGramHiddenStatesManager(BaseResourceManager):

    def __init__(self):
        pass

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        for req in chain(scheduled_batch.context_requests,
                         scheduled_batch.generation_requests):
            if req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                continue
            if req.py_next_draft_tokens is not None:
                req.py_draft_tokens = req.py_next_draft_tokens
                req.py_next_draft_tokens = None

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        pass

    def add_dummy_requests(self, request_ids: List[int]):
        pass

    def shutdown(self):
        pass

    def get_max_resource_count(self) -> int:
        return 0

    def get_needed_resource_to_completion(self, request: LlmRequest):
        # Do not need to allocate any blocks for the drafts?
        return 0


@dataclass
class NGramSpecMetadata(SpecMetadata):  # Remove this?
    """
    Metadata for NGram.
    """
    max_draft_tokens = 10

    def __post_init__(self) -> None:
        pass

    def prepare(self):
        pass


class NGramSampler(TorchSampler):
    """
    Sampler for NGram. This class maintains the pattern-matches pairs for NGram drafter.

    For example, one of the existed pairs could be: ["I","love"] -> [["apple", "because", "it", "is"], ["banana", "and"]].

    Here we call ["I","love"] as `pattern`, and [["apple", "because", "it", "is"], ["banana", "and"]] as `matches`.

    `pattern` is a list of token_ids. The pool provides corresponding draft tokens from the matches if the pattern appears at the tail of the sentence during generation.

    `matches` is a list of candidate draft token_ids attaching to a pattern.

    Arguments:
        prompt_lookup_num_tokens: int
            The length maximum of draft tokens (can be understood as length maximum of output draft tokens).

        max_matching_ngram_size: int
            The length maximum of searching tokens (can be understood as length maximum of input tokens to search).

        is_keep_all: bool = True
            Whether to keep all candidate pattern-matches pairs, only one match is kept for each pattern if False.

        is_use_oldest: bool = True
            Whether to provide the oldest match when pattern is hit, the newest one is provided if False.

        is_public_pool: bool = True
            Whether to use a common pool for all requests, or the pool is private for each request if False.

    Members:
        pool: dict[tuple[int], OrderedSet[int]] or dict[int, dict[tuple[int], OrderedSet[int]]]
            If is_public_pool == True, it maps from patterns to matches
            If is_public_pool == False, it maps from request ID to the request-specific pool

        start_index: dict[int, int]
            It maps from request ID to the index of the prompt to update the pool in the next step.
    """

    def __init__(self, max_seq_len: int, spec_config: SpecConfig):
        super().__init__(max_seq_len, False)

        self.max_num_draft_tokens = spec_config.max_draft_tokens
        self.prompt_lookup_num_tokens = spec_config.prompt_lookup_num_tokens
        self.max_matching_ngram_size = spec_config.max_matching_ngram_size
        self.is_keep_all = spec_config.is_keep_all
        self.is_use_oldest = spec_config.is_use_oldest  # TODO: remove this if updating strategy is supported
        self.is_public_pool = spec_config.is_public_pool
        #self.is_overlap_scheduler = is_overlap_scheduler
        self.pool = {}
        self.start_index = {}

    def update_requests(self, state: SampleState):
        super().update_requests(state)
        if self.is_public_pool:  # TODO: need a strategy to swap out the out-of-date pairs
            return

        for request in chain(state.scheduled_requests.context_requests,
                             state.scheduled_requests.generation_requests):
            if request.state == LlmRequestState.GENERATION_COMPLETE:
                request_id = request.request_id
                if request_id in self.pool:
                    self.pool.pop(request_id)
                    self.start_index.pop(request_id)
        return

    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs) -> SampleState:

        base_sample_state = super().sample_async(scheduled_requests,
                                                 model_outputs)
        new_tokens = base_sample_state.host.new_tokens.tolist()
        index = 0  # Index for each request to get corresponding tokens from `new_tokens`.

        sorted_requests = sorted(chain(scheduled_requests.context_requests,
                                       scheduled_requests.generation_requests),
                                 key=lambda x: x.py_batch_idx)
        for py_batch_idx, request in enumerate(sorted_requests):
            # TODO: remove this check
            assert request.py_batch_idx == py_batch_idx
            # TODO: move this to `NGramHiddenStatesManager.prepare_resources()`
            request.py_draft_pages_allocated = self.max_num_draft_tokens
            request.py_next_draft_tokens = None
            # Add new token to a copy of the generated tokens to find new daft tokens
            prefix = list(request.get_tokens()[0])  # Get a copy
            if request.py_draft_tokens is not None:
                draft_length = len(request.py_draft_tokens)
                output_tokens = new_tokens[index:index + draft_length + 1]
                # Simulate the process of acception
                num_accept_tokens = 0
                for i in range(draft_length):
                    if request.py_draft_tokens[i] == output_tokens[i]:
                        num_accept_tokens += 1
                    else:
                        break
                prefix += output_tokens[:num_accept_tokens + 1]
                index += draft_length + 1
            else:
                prefix += [new_tokens[index]]
                index += 1

            # Generate draft tokens, return None or a list
            draft_tokens = self._get_draft_tokens(
                prefix,
                request.request_id,
                request.py_end_id,
                request.py_orig_prompt_len + request.py_max_new_tokens,
            )

            # Pad to max_draft_tokens
            if draft_tokens is not None:
                pad_length = self.max_num_draft_tokens - len(draft_tokens)
                draft_tokens.extend([request.py_end_id] * pad_length)
            request.py_next_draft_tokens = draft_tokens

        return base_sample_state

    def print_pool(self):  # For debug
        if self.is_public_pool:
            logger.debug(f"Using public pool, size = {len(self.pool)}")
            self._print_line(self.pool)
        else:
            logger.debug(f"Using private pool")
            for request_id, request_map in self.pool.items():
                logger.debug(f"Request {request_id}, size={len(request_map)}")
                self._print_line(request_map, 4)

    def _print_line(self, local_map, indentation=0):  # For debug
        for pattern, matches in local_map.items():
            output = " " * indentation + str(pattern) + "->"
            for match in matches:
                output += str(match) + ", "
            logger.debug(output)

    def _get_draft_tokens(
        self,
        prefix: list[int],
        request_id: int,
        end_id: int,
        max_sequence_length: int,
    ):
        prefix_len = len(prefix)
        max_draft_token_length = max_sequence_length - 1 - prefix_len
        if max_draft_token_length <= 0:  # Skip search if prefix is long enough
            return None

        if request_id not in self.start_index:  # A new request
            self.start_index[request_id] = 0
            if not self.is_public_pool:
                self.pool[request_id] = {}
        pool = (self.pool if self.is_public_pool else self.pool[request_id])

        # Update pool
        sequence = prefix[self.start_index[request_id]:]
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            # Find each possible pattern-match combination, and use tuple for hash
            for l in range(len(sequence) - size):
                r = min(l + size + self.prompt_lookup_num_tokens, len(sequence))
                pattern = tuple(sequence[l:l + size])
                new_match = tuple(sequence[l + size:r])
                if pattern not in pool or \
                    (not self.is_keep_all and len(match) > pool[pattern][0]):
                    # Replace the match if
                    # 1. the pattern does not exist in the pool
                    # 2. only one match is kept, and the new match is longer (MRU)
                    pool[pattern] = OrderedSet((new_match, ))
                elif new_match not in pool[pattern]:
                    # Update the matches if the pattern is already existed:
                    # TODO: need a strategy to maintain the short candidates, now we just remove them
                    # Drop all existed matches with small length
                    for match in pool[pattern]:
                        if len(match) < len(new_match):
                            pool[pattern].remove(match)
                    pool[pattern].add(new_match)

        # Find match
        draft_tokens = [end_id]
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            pattern = tuple(prefix[-size:])
            if pattern not in pool:
                continue
            draft_tokens = pool[pattern][0 if self.is_use_oldest else -1]
            draft_tokens = list(draft_tokens)[:max_draft_token_length]
            break
        self.start_index[request_id] = max(
            0, prefix_len -
            (self.prompt_lookup_num_tokens + self.max_matching_ngram_size - 1))

        return draft_tokens
