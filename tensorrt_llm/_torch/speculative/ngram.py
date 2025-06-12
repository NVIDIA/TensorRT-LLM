import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Tuple

from ordered_set import OrderedSet

from tensorrt_llm.bindings.executor import IterationStats
from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import *
from ..pyexecutor.sampler import SampleState, SampleStateTensors, TorchSampler
from ..pyexecutor.scheduler import ScheduledRequests
from .interface import SpecConfig, SpecMetadata, SpeculativeDecodingMode


@dataclass(frozen=True, kw_only=True)
class SampleStateTensorsNGram(SampleStateTensors):
    # Map from `request.py_batch_id` to index in `new_tokens_device`
    replace_dict: dict
    # Map from `request.request_id` to number of tokens generated in this step.
    length_dict: OrderedDict


@dataclass(kw_only=True)
class SampleStateNGram(SampleState):
    device: SampleStateTensorsNGram
    # host: SampleStateTensorsNGram  # Useless yet


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


@dataclass
class NGramSpecMetadata(SpecMetadata):
    """
    Metadata for NGram.
    """

    def __post_init__(self) -> None:
        return

    def prepare(self):
        super().prepare()
        return

    def create_cuda_graph_metadata(self, max_batch_size: int):
        return super().create_cuda_graph_metadata(max_batch_size)


class NGramSampler(TorchSampler):
    """
    Sampler for NGram. This class maintains the pattern-matches pairs for NGram drafter.

    For example, one of the existed pairs could be: ["I","love"] -> [["apple", "because", "it", "is"], ["banana", "and"]].

    Here we call ["I","love"] as `pattern`, and [["apple", "because", "it", "is"], ["banana", "and"]] as `matches`.

    `pattern` is a list of token ids. The pool emits corresponding draft tokens from the matches if the pattern appears at the tail of the generated sentence.

    `matches` is a list of candidate draft token ids attaching to a pattern.

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

        is_overlap_scheduler: bool = True
            Whether overlap scheduler is used since the sampler has different behaviors in those modes.

    Members:
        pool: dict[tuple[int], OrderedSet[int]] or dict[int, dict[tuple[int], OrderedSet[int]]]
            If is_public_pool == True, it maps from patterns to matches
            If is_public_pool == False, it maps from request ID to the request-specific pool

        start_index: dict[int, int]
            It maps from request ID to the index of the prompt to update the pool in the next step.
    """

    def __init__(
        self,
        max_seq_len: int,
        spec_config: SpecConfig,
        disable_overlap_scheduler: bool = False,
    ):
        super().__init__(max_seq_len, False)

        self.max_num_draft_tokens = spec_config.max_draft_tokens
        self.prompt_lookup_num_tokens = spec_config.prompt_lookup_num_tokens
        self.max_matching_ngram_size = spec_config.max_matching_ngram_size
        self.is_keep_all = spec_config.is_keep_all
        self.is_use_oldest = spec_config.is_use_oldest  # TODO: remove this if updating strategy is supported
        self.is_public_pool = spec_config.is_public_pool
        self.is_overlap_scheduler = not disable_overlap_scheduler
        self.pool = {}
        self.start_index = {}

    def update_requests(self, state: SampleState):
        if self.is_overlap_scheduler:
            for request in state.scheduled_requests.generation_requests:
                # In overlap-scheduler mode, `py_draft_tokens` contains draft tokens of next forward computation,
                # while `py_last_draft_tokens` contains draft tokens of last forward computation.
                # So they needs to be swapped since `TorchSampler.TorchSampler()` uses `py_draft_tokens` to update requests.
                # After updating, `py_draft_tokens` must be restored and `py_last_draft_tokens` is useless anymore.
                # In non-overlap-scheduler mode, `py_draft_tokens` contains draft tokens of last forward computation,
                # So, this swap is not needed.
                request.py_draft_tokens, request.py_last_draft_tokens = \
                    request.py_last_draft_tokens, request.py_draft_tokens

        super().update_requests(state)  # Reuse `TorchSampler.TorchSampler()`

        if self.is_overlap_scheduler:
            for request in state.scheduled_requests.generation_requests:
                request.py_draft_tokens = request.py_last_draft_tokens

        if self.is_public_pool:  # TODO: need an updating strategy to swap out the out-of-date pairs
            return

        # Swap out the out-of-date pairs if the request is completed.
        for request in chain(state.scheduled_requests.context_requests,
                             state.scheduled_requests.generation_requests):
            if request.state == LlmRequestState.GENERATION_COMPLETE:
                request_id = request.request_id
                if request_id in self.pool:
                    self.pool.pop(request_id)
                    self.start_index.pop(request_id)
        return

    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs) -> SampleStateNGram:
        base_state = super().sample_async(scheduled_requests, model_outputs)

        # Note which tokens in the `base_state.device.new_tokens` belong to each request
        # For example,
        # Here are requests below and `base_state.device.new_tokens = [2,3,5,7,11,13,15,17,19,23,29,31]`
        #     | request_id | state | py_batch_idx | num_draft_tokens |
        #     |    2048    |  ctx  |      0       |        0         |
        #     |    2049    |  ctx  |      1       |        0         |
        #     |    2052    |  gen  |      2       |        4         |
        #     |    2051    |  gen  |      3       |        4         |
        # Finally,
        # replace_dict = {0:0, 1:1, 2:2, 3:7} (start index of the tokens of the request)
        # length_dict = {2048:1, 2049:1, 2052:5, 2051: 5} (length of the tokens of the request)
        # TODO: merge the two dictionaries
        accumulate_index = 0
        replace_dict = {}
        length_dict = OrderedDict()
        for request in sorted(chain(scheduled_requests.context_requests,
                                    scheduled_requests.generation_requests),
                              key=lambda x: x.py_batch_idx):
            num_tokens = (1 if request in scheduled_requests.context_requests
                          else self.max_num_draft_tokens + 1)
            replace_dict[request.py_batch_idx] = accumulate_index
            accumulate_index += num_tokens
            length_dict[request.request_id] = num_tokens

        device = SampleStateTensorsNGram(
            new_tokens=base_state.device.new_tokens,
            replace_dict=replace_dict,
            length_dict=length_dict,
        )
        sampler_event = torch.cuda.Event()
        sampler_event.record()

        return SampleStateNGram(
            scheduled_requests=scheduled_requests,
            logits=model_outputs['logits'],
            device=device,
            host=base_state.host,
            sampler_event=sampler_event,
        )

    def prepare_forward(
        self,
        scheduled_requests: ScheduledRequests,
        state: SampleState,
        iter_stats: IterationStats,
    ) -> None:
        if iter_stats is not None:
            before = time.time()

        self._prepare_forward(scheduled_requests, state)

        if iter_stats is not None:
            self._insert_ngram_iter_stats(scheduled_requests, iter_stats)
            iter_stats.specdec_stats.iter_latency_ms = (time.time() -
                                                        before) * 1e3
        return

    def _prepare_forward(self, scheduled_requests: ScheduledRequests,
                         state: SampleState) -> None:
        if state is None:  # Skip in the first step
            return

        state.sampler_event.synchronize()
        new_tokens = state.host.new_tokens.tolist()
        length_dict = state.device.length_dict
        for request in sorted(scheduled_requests.generation_requests,
                              key=lambda r: r.py_batch_idx):
            assert request.request_id in length_dict, f"request {request.request_id} not in length_dict {length_dict}"
            index = 0  # Index for each request to get corresponding tokens from `new_tokens`.
            for last_request_id, length in length_dict.items():
                if request.request_id == last_request_id:
                    break
                index += length
            # Add new token to a copy of the generated tokens to find new daft tokens
            prefix = list(request.get_tokens()[0])  # Get a copy
            accepted_tokens_list = []
            if self.is_overlap_scheduler:
                # In overlap-scheduler, the generated tokens (as well as the accepted tokens) in the
                # last forward computation has not been updated into `request` yet.
                # So here we simulate the process of acception to find new draft tokens
                # In non-overlap-scheduler, the generated tokens (as well as the accepted tokens) in the
                # last forward computation has been updated into `request`, so the simulation is not needed.
                num_accepted = 0
                if len(request.py_last_draft_tokens) > 0:
                    draft_length = len(request.py_last_draft_tokens)
                    output_tokens = new_tokens[index:index + draft_length + 1]
                    for i in range(draft_length):
                        if request.py_last_draft_tokens[i] == output_tokens[i]:
                            num_accepted += 1
                        else:
                            break
                    accepted_tokens_list = output_tokens[:num_accepted + 1]
                else:
                    accepted_tokens_list = [new_tokens[index]]
                request.py_num_accepted_draft_tokens = num_accepted

            # Generate draft tokens
            draft_tokens = self._get_draft_tokens(
                prefix + accepted_tokens_list,
                request.request_id,
                request.py_end_id,
                request.py_orig_prompt_len + request.py_max_new_tokens,
            )
            # Pad length to `self.max_num_draft_tokens`
            if len(draft_tokens) > 0:
                pad_length = self.max_num_draft_tokens - len(draft_tokens)
                draft_tokens.extend([request.py_end_id] * pad_length)
            request.py_draft_tokens = draft_tokens

        return

    def _insert_ngram_iter_stats(
        self, scheduled_requests: ScheduledRequests, iter_stats: IterationStats
    ) -> Tuple[ScheduledRequests, Dict[int, LlmRequest]]:
        """
        Get statistic information from the draft tokens in NGram drafter
        """
        assert iter_stats is not None

        total_num_draft_tokens = 0
        total_num_accepted_tokens = 0
        num_requests_with_draft_tokens = 0
        for request in chain(scheduled_requests.context_requests,
                             scheduled_requests.generation_requests):
            num_draft_tokens = 0 if request.py_last_draft_tokens is None else len(
                request.py_last_draft_tokens)
            num_accepted_tokens = getattr(request,
                                          "py_num_accepted_draft_tokens", 0)
            if num_draft_tokens > 0:
                total_num_draft_tokens = total_num_draft_tokens + num_draft_tokens
                total_num_accepted_tokens = total_num_accepted_tokens + num_accepted_tokens
                num_requests_with_draft_tokens = num_requests_with_draft_tokens + 1

        if num_requests_with_draft_tokens > 0:
            iter_stats.specdec_stats.iter_latency_ms = 0.0  # We do not coutn time in this method
            iter_stats.specdec_stats.num_draft_tokens = total_num_draft_tokens
            iter_stats.specdec_stats.num_accepted_tokens = total_num_accepted_tokens
            iter_stats.specdec_stats.num_requests_with_draft_tokens = num_requests_with_draft_tokens
            iter_stats.specdec_stats.acceptance_length = float(
                (total_num_accepted_tokens + num_requests_with_draft_tokens
                 )) / float(num_requests_with_draft_tokens)
        else:
            iter_stats.specdec_stats.iter_latency_ms = 0.0
            iter_stats.specdec_stats.num_draft_tokens = 0
            iter_stats.specdec_stats.num_accepted_tokens = 0
            iter_stats.specdec_stats.num_requests_with_draft_tokens = 0
            iter_stats.specdec_stats.acceptance_length = 1.0

        return

    def _get_draft_tokens(
        self,
        prefix: list[int],
        request_id: int,
        end_id: int,
        max_sequence_length: int,
    ):
        prefix_len = len(prefix)
        max_draft_token_length_this_step = max_sequence_length - 1 - prefix_len
        if max_draft_token_length_this_step <= 0:  # No draft tokens is need if the prefix is long enough
            return [end_id]

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
        draft_tokens = [end_id]  # fallback value
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            pattern = tuple(prefix[-size:])
            if pattern not in pool:
                continue
            draft_tokens = pool[pattern][0 if self.is_use_oldest else -1]
            draft_tokens = list(draft_tokens)[:max_draft_token_length_this_step]
            break

        # Update start_index
        self.start_index[request_id] = max(
            0, prefix_len -
            (self.prompt_lookup_num_tokens + self.max_matching_ngram_size - 1))

        return draft_tokens

    def print_pool(self):  # For debug
        if self.is_public_pool:
            logger.debug(f"Using public pool, size = {len(self.pool)}")
            self._print_line(self.pool)
        else:
            logger.debug(f"Using private pool")
            for request_id, request_map in self.pool.items():
                logger.debug(f"Request {request_id}, size={len(request_map)}")
                self._print_line(request_map, 4)

    def _print_line(self, local_map, indentation=0):
        for pattern, matches in local_map.items():
            output = " " * indentation + str(pattern) + "->"
            for match in matches:
                output += str(match) + ", "
            logger.debug(output)
