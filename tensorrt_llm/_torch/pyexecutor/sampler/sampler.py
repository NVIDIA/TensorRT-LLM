# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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

import enum
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent import futures
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from itertools import repeat
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Type,
    TypeAlias,
    TypeVar,
    cast,
)

import numpy as np
import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.pyexecutor.make_decoding_batch_input_output import (
    MakeDecodingBatchInputOutput,
)
from tensorrt_llm._utils import (
    maybe_pin_memory,
    mpi_disabled,
    nvtx_range,
    prefer_pinned,
    torch_dtype_to_binding,
)
from tensorrt_llm.bindings import (
    CudaStream,
    DataType,
    ModelConfig,
    SamplingConfigVector,
    WorldConfig,
    make_sampling_config,
)
from tensorrt_llm.bindings.executor import DecodingConfig, DecodingMode, FinishReason
from tensorrt_llm.bindings.internal.algorithms import CreateNewDecoderRequests
from tensorrt_llm.bindings.internal.batch_manager import (
    DecoderInputBuffers,
    add_new_tokens_to_requests,
    make_decoding_batch_input,
)
from tensorrt_llm.bindings.internal.runtime import (
    BufferManager,
    CudaEvent,
    DecoderState,
    GptDecoderBatched,
)
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import (
    MAX_TOP_LOGPROBS,
    LogprobMode,
    SamplingParams,
    check_logprobs_limit,
)

from ...speculative.interface import get_force_num_accepted_tokens
from ...speculative.spec_tree_manager import SpecTreeManager
from ...utils import torch_multi_arange
from ..finish_reason import FinishedState
from ..llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from ..resource_manager import ResourceManager, ResourceManagerType
from ..scheduler import ScheduledRequests
from .finish_reasons import FinishReasonsHandler
from .logprobs import (
    LogProbsState,
    LogProbsStateList,
    LogProbsStore,
    convert_logprobs_tensor_to_list,
    get_logprobs_from_request,
    store_logprobs_list_to_request,
)
from .sampler_common import (
    DEFAULT_BEAM_IDX,
    DEFAULT_STEP_IDX,
    FinishReasonsList,
    _get_max_beam_width,
    _request_get_sampling_params,
    _unwrap_singleton,
    add_token,
    int_tensor,
)
from .sampler_strategy import (
    BEAM_SEARCH_PAD_TOKEN,
    GREEDY,
    BeamSearchMetadata,
    FlashInferGroupedStrategySampler,
    Fusions,
    GenericStrategyKeyType,
    Strategy,
    StrategyMetadata,
    TopPDecayMetadata,
    _request_strategy,
    get_rejected_indices,
    sample,
    sample_rejected,
)
from .token_ban import OverlappedTokenBanHandler, SynchronousTokenBanHandler, TokenBanHandler
from .top_p_decay import TopPDecayHandler

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM

    _ModelType = TypeVar("_ModelType", bound=DecoderModel)
    _ConfigType = TypeVar("_ConfigType", bound=PretrainedConfig)

T = TypeVar("T")


@dataclass(kw_only=True)
class SampleStateTensors:
    new_tokens: torch.Tensor
    log_probs: torch.Tensor | None = None


@dataclass(kw_only=True)
class SamplerEvent:
    cuda_event: torch.cuda.Event
    # Side-stream D2H completion, synced host-side without gating the main stream.
    side_stream_event: Optional[torch.cuda.Event] = None
    worker_futures: Optional[list[futures.Future[Any]]] = None

    def synchronize(self) -> None:
        if self.worker_futures:
            futures.wait(self.worker_futures)
        self.cuda_event.synchronize()
        if self.side_stream_event is not None:
            self.side_stream_event.synchronize()


GenericSampleStateTensorsHost = TypeVar("GenericSampleStateTensorsHost", bound=SampleStateTensors)
GenericSampleStateTensorsDevice = TypeVar(
    "GenericSampleStateTensorsDevice", bound=SampleStateTensors
)


@dataclass(kw_only=True)
class SampleState(Generic[GenericSampleStateTensorsHost, GenericSampleStateTensorsDevice]):
    requests: list[LlmRequest]
    device: Optional[GenericSampleStateTensorsDevice] = None
    host: Optional[GenericSampleStateTensorsHost] = None
    sampler_event: Optional[SamplerEvent] = None
    runtime_draft_len: Optional[int] = None


# Generic bounds not supported, https://github.com/python/typing/issues/548
GenericSampleState = TypeVar("GenericSampleState", bound=SampleState)  # type: ignore


class Sampler(ABC, Generic[GenericSampleState]):
    def setup_sampler_step(self, scheduled_requests: ScheduledRequests) -> None:
        pass

    def get_cache_indirection(self) -> torch.Tensor | None:
        return None

    @abstractmethod
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> GenericSampleState:
        raise NotImplementedError

    @abstractmethod
    def update_requests(
        self,
        state: GenericSampleState,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def beam_width(requests: Iterable[LlmRequest]) -> int:
        for req in requests:
            return req.py_beam_width
        return 0

    @abstractmethod
    def is_generation_model(self) -> bool:
        raise NotImplementedError

    def validate_request(self, request: LlmRequest) -> None:
        """Validate that the request can be processed by the sampler.

        If the request is not supported by the sampler, this should raise an
        appropriate exception.

        Args:
            request: The request to validate

        Returns:
            None if request is valid

        Raises:
            Appropriate exception if request is not supported by sampler.
        """

    def should_provide_draft_probs(self, request: LlmRequest) -> bool:
        """Check if sampler wants to receive draft token probabilities."""
        return True  # conservative default


class EarlyStopSampler(Sampler[SampleState[SampleStateTensors, SampleStateTensors]]):
    """
    Use for skipping decoding step for non generation model,
    such as encoder-only model (e.g., BERT) or reward models that only need context phase.
    """

    SampleState: TypeAlias = SampleState[SampleStateTensors, SampleStateTensors]

    @override
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleState:
        host = SampleStateTensors(new_tokens=torch.empty(0))
        assert not scheduled_requests.generation_requests
        return self.SampleState(requests=scheduled_requests.context_requests, host=host)

    @override
    def update_requests(
        self,
        state: SampleState,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        assert isinstance(state, SampleState)
        requests = state.requests
        for idx, request in enumerate(requests):
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)

    @override
    def is_generation_model(self) -> bool:
        return False


@dataclass(kw_only=True)
class MultimodalResult:
    mm_embeddings: List[torch.Tensor]
    # needed to torch.split the mm_embeddings into item-wise chunks
    mm_embedding_lengths: List[List[int]]
    # needed when requests mix text-only and multimodal ones
    mm_embedding_request_indices: List[int]
    # number of context requests in the batch
    num_context_requests: int
    # Can be used to include e.g. `mrope_position_ids`, etc.
    extra_data: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        num_embeddings = len(self.mm_embeddings)
        num_lengths = len(self.mm_embedding_lengths)
        if num_lengths != num_embeddings:
            raise ValueError(
                "mm_embedding_lengths batch size does not match mm_embeddings: "
                f"{num_lengths} != {num_embeddings}"
            )
        num_request_indices = len(self.mm_embedding_request_indices)
        if num_request_indices != num_embeddings:
            raise ValueError(
                "mm_embedding_request_indices batch size does not match "
                f"mm_embeddings: {num_request_indices} != {num_embeddings}"
            )
        for result_index, (mm_embedding, mm_embedding_lengths) in enumerate(
            zip(self.mm_embeddings, self.mm_embedding_lengths, strict=True)
        ):
            actual_rows = len(mm_embedding)
            expected_rows = sum(mm_embedding_lengths)
            if actual_rows != expected_rows:
                raise ValueError(
                    f"mm_embedding shape mismatch for result {result_index}: "
                    f"{actual_rows} != {expected_rows}"
                )
        for request_index in self.mm_embedding_request_indices:
            if request_index < 0 or request_index >= self.num_context_requests:
                raise ValueError(
                    "mm_embedding_request_indices contains an invalid request "
                    f"index: {request_index} not in [0, {self.num_context_requests})"
                )

    @classmethod
    def from_model_outputs(
        cls, model_outputs: Dict[str, Any], num_context_requests: int
    ) -> "MultimodalResult":
        result_keys = {
            "mm_embeddings",
            "mm_embedding_lengths",
            "mm_embedding_request_indices",
        }
        return cls(
            mm_embeddings=model_outputs["mm_embeddings"],
            mm_embedding_lengths=model_outputs["mm_embedding_lengths"],
            mm_embedding_request_indices=model_outputs["mm_embedding_request_indices"],
            num_context_requests=num_context_requests,
            extra_data={
                key: value for key, value in model_outputs.items() if key not in result_keys
            },
        )


@dataclass(kw_only=True)
class SampleStateWithMMResult(SampleState[SampleStateTensors, SampleStateTensors]):
    data: MultimodalResult


@dataclass(kw_only=True, frozen=True, slots=True)
class RequestGroupKey(Generic[GenericStrategyKeyType]):
    strategy_key: GenericStrategyKeyType
    needs_probs: bool


@dataclass(kw_only=True, frozen=True)
class RequestGroupValue:
    indices: torch.Tensor
    strategies: list[Strategy]
    speculation_needs_probs_indices: torch.Tensor
    need_processed_logprobs: torch.Tensor
    need_raw_logprobs: torch.Tensor


@dataclass(kw_only=True, frozen=True)
class RequestGroupValueWithMetadata(RequestGroupValue):
    metadata: StrategyMetadata | None


class EarlyStopWithMMResult(Sampler[SampleStateWithMMResult]):
    """
    Use for skipping decoding step for non generation model, and return the batch_output (such as mm_embeddings)
    """

    SampleState: TypeAlias = SampleStateWithMMResult

    @override
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleState:
        # from model_outputs to MultimodalResult
        assert not scheduled_requests.generation_requests
        data = MultimodalResult.from_model_outputs(
            model_outputs, scheduled_requests.num_context_requests
        )
        return self.SampleState(requests=scheduled_requests.context_requests, data=data)

    @override
    def update_requests(
        self,
        state: SampleState,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        # resource_manager will not be used in this function, just for interface consistency.
        assert isinstance(state, SampleState)
        requests = state.requests
        mm_embeddings = state.data.mm_embeddings
        extra_data = state.data.extra_data or {}
        mrope_position_ids = extra_data.get("mrope_position_ids", None)
        mrope_position_deltas = extra_data.get("mrope_position_deltas", None)
        for request in requests:
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)

        request_indices = state.data.mm_embedding_request_indices
        for result_index, (request_index, mm_embedding) in enumerate(
            zip(request_indices, mm_embeddings, strict=True)
        ):
            request = requests[request_index]
            mm_embedding_lengths = state.data.mm_embedding_lengths[result_index]

            request.py_result.append_mm_embeddings(mm_embedding, mm_embedding_lengths)

            # Store mrope data if available
            if mrope_position_ids is not None and mrope_position_deltas is not None:
                mrope_index = (
                    request_index if len(mrope_position_ids) == len(requests) else result_index
                )
                request.py_result.set_mrope_position(
                    mrope_position_ids[mrope_index],
                    mrope_position_deltas[mrope_index],
                )

    @override
    def is_generation_model(self) -> bool:
        return False


def _has_occurrence_penalty(request: LlmRequest) -> bool:
    sampling_config = request.sampling_config
    repetition = _unwrap_singleton(sampling_config.repetition_penalty)
    presence = _unwrap_singleton(sampling_config.presence_penalty)
    frequency = _unwrap_singleton(sampling_config.frequency_penalty)
    return (
        (repetition is not None and repetition != 1.0)
        or (presence is not None and presence != 0.0)
        or (frequency is not None and frequency != 0.0)
    )


class _CachingRequestGrouper(Generic[GenericStrategyKeyType]):
    """Efficiently groups requests for batched sampling."""

    @dataclass(kw_only=True)
    class _Store:
        """Auxiliary data structures used for efficiently grouping requests for batched sampling."""

        slots_needing_recompute: set[int]
        """Slots where strategy needs (re)computation. Populated in setup_sampler_step."""
        non_greedy_slots: set[int]
        """Slots with non-greedy strategies. Used to limit draft-token checks."""
        need_processed_logprobs: list[bool]
        """Length: max_num_sequences. True if logprob mode is PROCESSED and return_log_probs is set."""
        need_raw_logprobs: list[bool]
        """Length: max_num_sequences. True if logprob mode is RAW and return_log_probs is set."""
        speculation_needs_probs: list[bool]
        """Length: max_num_sequences. True if request has draft tokens and non-greedy sampling."""
        needs_probs: list[bool]
        """Length: max_num_sequences. True if speculation_needs_probs or need_processed_logprobs."""
        strategies: list[Strategy | None]
        """Length: max_num_sequences. Stores cached Strategy tuple for each seq_slot."""
        uses_beam_search: list[bool]
        """Length: max_num_sequences. True if max_beam_width > 1 for this slot."""

    def __init__(self, max_num_sequences: int):
        # Use Python lists instead of tensors to avoid .item() overhead in hot loops
        speculation_needs_probs = [False] * max_num_sequences
        need_processed_logprobs = [False] * max_num_sequences
        need_raw_logprobs = [False] * max_num_sequences
        needs_probs = [False] * max_num_sequences
        strategies: list[Strategy | None] = [None] * max_num_sequences
        uses_beam_search = [False] * max_num_sequences
        slots_needing_recompute: set[int] = set()
        non_greedy_slots: set[int] = set()

        self._store = self._Store(
            speculation_needs_probs=speculation_needs_probs,
            need_processed_logprobs=need_processed_logprobs,
            need_raw_logprobs=need_raw_logprobs,
            needs_probs=needs_probs,
            strategies=strategies,
            uses_beam_search=uses_beam_search,
            slots_needing_recompute=slots_needing_recompute,
            non_greedy_slots=non_greedy_slots,
        )

    def prepare_for_new_request(self, request: LlmRequest, slot: int) -> None:
        store = self._store
        # Initialize cached data for this slot (prevents stale data from previous request)
        store.strategies[slot] = None
        store.uses_beam_search[slot] = _get_max_beam_width(request) > 1
        # Mark slot for strategy recomputation in _group_requests_by_strategy_key
        store.slots_needing_recompute.add(slot)
        store.non_greedy_slots.discard(slot)  # reset until strategy is computed

    def group_requests_by_strategy_key(
        self,
        requests: Iterable[LlmRequest],
        *,
        strategy_to_key: Callable[[Strategy], GenericStrategyKeyType],
        pin_memory: bool = False,
        seq_slots: torch.Tensor,
        vocab_size: int,
    ) -> dict[RequestGroupKey[GenericStrategyKeyType], RequestGroupValue]:
        """
        Optimized implementation with vectorized boolean operations and efficient grouping.

        NB: Client code relies on request indices in returned torch.Tensor being sorted.
        """
        store = self._store

        # Convert to list for efficient indexing
        requests_list = list(requests) if not isinstance(requests, list) else requests
        num_requests = len(requests_list)

        if num_requests == 0:
            return {}

        assert not seq_slots.is_cuda, "seq_slots is expected to be a host tensor"
        seq_slots_list = seq_slots.tolist()

        # Get strategies from cache, only recomputing for slots that need it.
        # Recompute is needed for:
        #   - Uncached slots (strategy is None) — recorded in store.slots_needing_recompute
        #   - Beam search (beam_width_in changes) — kept in slots_needing_recompute permanently
        #   - Speculative decoding (draft_tokens can change) — checked for non-greedy slots only

        # Build strategies from cache in one shot (C-level list comprehension, ~50ns/elem)
        s_strategies = store.strategies
        batch_strategies = [s_strategies[slot] for slot in seq_slots_list]

        # Build slot→request_index mapping for targeted access
        slot_to_idx = {slot: i for i, slot in enumerate(seq_slots_list)}
        active_slots = set(slot_to_idx)

        # 1) Slots pre-recorded for recompute (context-phase or beam search)
        recompute_batch_slots = store.slots_needing_recompute & active_slots

        # 2) Non-greedy slots where draft-token status may have changed
        #    (For greedy: current_has_draft is always False, matching cached, so never stale)
        draft_check_slots = (store.non_greedy_slots & active_slots) - recompute_batch_slots
        for slot in draft_check_slots:
            batch_index = slot_to_idx[slot]
            has_draft = bool(requests_list[batch_index].py_draft_tokens)
            if store.speculation_needs_probs[slot] != has_draft:
                # Draft-token status changed — only update the affected flags.
                # The strategy itself doesn't depend on draft tokens (only on sampling params).
                store.speculation_needs_probs[slot] = has_draft
                store.needs_probs[slot] = has_draft or store.need_processed_logprobs[slot]

        # 3) Full recompute for the pre-recorded slots.
        #    Every slot with a None strategy must already be in slots_needing_recompute
        #    (populated by setup_sampler_step when a new request arrives).
        assert None not in batch_strategies or all(
            seq_slots_list[batch_index] in recompute_batch_slots
            for batch_index in range(num_requests)
            if batch_strategies[batch_index] is None
        ), (
            "Found slots with uncached strategies not registered in slots_needing_recompute. "
            "Ensure setup_sampler_step is called before sample_async for new requests."
        )

        for slot in recompute_batch_slots:
            batch_index = slot_to_idx[slot]
            request = requests_list[batch_index]
            has_draft_tokens = bool(request.py_draft_tokens)

            strategy = _request_strategy(request, vocab_size=vocab_size)
            store.strategies[slot] = strategy
            batch_strategies[batch_index] = strategy

            is_greedy = strategy == GREEDY
            current_speculation_needs_probs = has_draft_tokens and not is_greedy
            store.speculation_needs_probs[slot] = current_speculation_needs_probs
            current_need_processed_logprobs = (
                request.py_logprobs_mode == LogprobMode.PROCESSED and request.return_log_probs
            )
            store.need_processed_logprobs[slot] = current_need_processed_logprobs
            store.need_raw_logprobs[slot] = (
                request.py_logprobs_mode == LogprobMode.RAW and request.return_log_probs
            )
            store.needs_probs[slot] = (
                current_speculation_needs_probs or current_need_processed_logprobs
            )

            # Track non-greedy slots for future draft-token checks
            if is_greedy:
                store.non_greedy_slots.discard(slot)
            else:
                store.non_greedy_slots.add(slot)

            # Keep beam-search slots in the recompute set (they always need it);
            # remove everything else (strategy is now cached).
            if not store.uses_beam_search[slot]:
                store.slots_needing_recompute.discard(slot)

        # Gather flags using list comprehension (faster than append in loop)
        needs_probs = torch.tensor(
            [store.needs_probs[slot] for slot in seq_slots_list], dtype=torch.bool, device="cpu"
        )
        speculation_needs_probs = torch.tensor(
            [store.speculation_needs_probs[slot] for slot in seq_slots_list],
            dtype=torch.bool,
            device="cpu",
        )
        need_processed_logprobs = torch.tensor(
            [store.need_processed_logprobs[slot] for slot in seq_slots_list],
            dtype=torch.bool,
            device="cpu",
        )
        need_raw_logprobs = torch.tensor(
            [store.need_raw_logprobs[slot] for slot in seq_slots_list],
            dtype=torch.bool,
            device="cpu",
        )
        # Build strategy ID mapping for vectorized comparison (all on CPU).
        # NB: set() does not preserve insertion order, so we use dict.fromkeys() to deduplicate while preserving order.
        unique_strategies = list(dict.fromkeys(batch_strategies))
        strategy_to_id = {s: idx for idx, s in enumerate(unique_strategies)}
        strategy_ids = torch.tensor(
            [strategy_to_id[s] for s in batch_strategies], dtype=torch.int32, device="cpu"
        )

        # Pre-allocate group_ids array
        group_ids = torch.empty(num_requests, dtype=torch.int32, device="cpu")

        _next_gid = 0

        def _provision_gid() -> int:
            nonlocal _next_gid
            gid = _next_gid
            _next_gid += 1
            return gid

        unique_keys: defaultdict[tuple[GenericStrategyKeyType, bool], int] = defaultdict(
            _provision_gid
        )

        # Vectorized assignment: loop over unique combinations instead of all requests
        for sid, strategy in enumerate(unique_strategies):
            strat_mask = strategy_ids == sid

            for needs_probs_val in (False, True):
                # Vectorized mask for this (strategy, needs_probs) group
                mask = strat_mask & (needs_probs if needs_probs_val else ~needs_probs)

                if torch.any(mask):
                    strategy_key = strategy_to_key(strategy)  # Called once per group!
                    key = (strategy_key, needs_probs_val)
                    group_ids[mask] = unique_keys[key]  # Vectorized assignment

        # Efficient grouping using sort
        sorted_group_ids, sorted_order = torch.sort(group_ids, stable=True)
        # Use prepend to detect a "change" at position 0, giving us group_starts directly
        group_starts = torch.nonzero(
            torch.diff(sorted_group_ids, prepend=torch.tensor([-1], device="cpu")) != 0
        ).squeeze(1)
        group_ends = torch.cat([group_starts[1:], torch.tensor([num_requests], device="cpu")])
        # Since groups are assigned in request order, gid → key is just list indexing
        id_to_key = list(unique_keys)

        # Build result dictionary efficiently
        result: dict[RequestGroupKey[GenericStrategyKeyType], RequestGroupValue] = {}

        for gid, (start, end) in enumerate(zip(group_starts.tolist(), group_ends.tolist())):
            group_sorted_indices = sorted_order[start:end]
            strategy_key, needs_probs_bool = id_to_key[gid]

            indices_arr = group_sorted_indices.to(torch.int32)
            # Convert to list for Python list indexing
            group_sorted_indices_list = group_sorted_indices.tolist()
            group_strategies = [
                batch_strategies[batch_index] for batch_index in group_sorted_indices_list
            ]
            spec_mask = speculation_needs_probs[group_sorted_indices]
            spec_indices = indices_arr[spec_mask]
            processed_flags = need_processed_logprobs[group_sorted_indices]
            raw_flags = need_raw_logprobs[group_sorted_indices]

            if pin_memory:
                indices_tensor = maybe_pin_memory(indices_arr)
                spec_tensor = maybe_pin_memory(spec_indices)
                processed_tensor = maybe_pin_memory(processed_flags)
                raw_tensor = maybe_pin_memory(raw_flags)
            else:
                indices_tensor = indices_arr
                spec_tensor = spec_indices
                processed_tensor = processed_flags
                raw_tensor = raw_flags

            result[RequestGroupKey(strategy_key=strategy_key, needs_probs=needs_probs_bool)] = (
                RequestGroupValue(
                    indices=indices_tensor,
                    strategies=group_strategies,
                    speculation_needs_probs_indices=spec_tensor,
                    need_processed_logprobs=processed_tensor,
                    need_raw_logprobs=raw_tensor,
                )
            )

        return result


@dataclass(kw_only=True, frozen=True)
class _BatchedSamplingResult:
    # Original request indices for all requests (permuted due to batching by strategy):
    batch_req_indices: torch.Tensor
    # Next tokens for all requests:
    batch_next_tokens_cuda_int: torch.Tensor
    # Logits for all requests used for logprobs:
    batch_logits_for_logprobs_cuda: torch.Tensor | None = None


# Helper class for _PackedStepIndexer and _UnpackedStepIndexer, facilitating the
# selection of memory locations of tokens associated with given sets of requests.
class _StepIndexTranslator(ABC):
    def __init__(
        self,
        *,
        num_steps: torch.Tensor,
        req_offsets: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        index_dtype: Optional[torch.dtype] = None,
    ):
        """Build the index.

        Arguments:
            index_dtype: torch.dtype to use for indices (defaults to torch.int32).
            num_steps (index_dtype): Number of steps/tokens for each request
            req_offsets (index_dtype): Index offset at which the data for each request starts.
                                       If not provided, it is computed using calculate_request_offsets(),
                                       which assumes dense packing.
            max_steps (int): The largest value allowed to occur in num_steps.
                             If not provided, it is computed from num_steps.
        """
        if req_offsets is None:
            req_offsets, _ = self.calculate_request_offsets(num_steps)
        if max_steps is None:
            max_steps = cast(int, num_steps.max().item())
        self._index_map, self._index_mask = self._build_index(
            req_offsets=req_offsets,
            num_steps=num_steps,
            max_steps=max_steps,
            index_dtype=(index_dtype or torch.int32),
        )

    @staticmethod
    def calculate_request_offsets(
        req_num_steps: torch.Tensor,
        pin_memory: bool = False,
    ) -> tuple[torch.Tensor, int]:
        if req_num_steps.numel():
            req_offsets = torch.cumsum(req_num_steps, 0)
            sum_steps = int(req_offsets[-1].item())
            req_offsets_rolled = torch.empty_like(req_offsets, pin_memory=pin_memory)
            req_offsets_rolled[1:] = req_offsets[:-1]
            req_offsets_rolled[0] = 0
            req_offsets = req_offsets_rolled
        else:
            req_offsets = torch.empty_like(req_num_steps, pin_memory=pin_memory)
            sum_steps = 0
        return req_offsets, sum_steps

    def _build_index(
        self,
        req_offsets: torch.Tensor,
        num_steps: torch.Tensor,
        max_steps: int,
        index_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps_dim = torch.arange(max_steps, device=num_steps.device, dtype=index_dtype)
        valid_mask = steps_dim.unsqueeze(0) < num_steps.unsqueeze(-1)
        indices = self._compute_index_map(
            index_dtype=index_dtype,
            steps_dim=steps_dim,
            req_offsets=req_offsets,
        )
        # NB: steps_dim and req_offsets may have been overwritten by this point.
        return indices, valid_mask

    @abstractmethod
    def _compute_index_map(
        self,
        index_dtype: torch.dtype,
        steps_dim: torch.Tensor,
        req_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute full tensor index map.

        Should return a tensor of shape (len(num_steps), max_steps) containing the linear
        token index (index_dtype) corresponding to a given request and decoding step.
        Each row corresponds to a request (same ordering as 'req_offsets' and 'num_steps'),
        and the columns correspond to decoding steps 0, ..., num_steps[i]. Entries corresponding
        to decoding steps which are invalid for the given request are masked elsewhere within
        _StepIndexTranslator.

        This method is allowed to repurpose/overwrite 'steps_dim' and 'req_offsets'.

        Arguments:
            num_steps (index_dtype): Number of steps/tokens for each request
            req_offsets (index_dtype): Index offset at which the data for each request starts.
            steps_dim (index_dtype): arange(max_steps)
            index_dtype: torch.dtype to use for indices
        """

    def __getitem__(self, req_indices: Any) -> torch.Tensor:
        """Gather indices for a given set of requests.

        Arguments:
            req_indices: Any 1d torch-compatible indexing expression to select requests, corresponds
                         to the linear indices of the entries in 'num_steps' and 'req_offsets' (cf. __init__).
        Returns:
            Array of linear indices (index_dtype) selecting the tokens/steps associated
            with the requests identified by req_indices, in the same order as
            req_indices.
        """
        indices = self._index_map[req_indices].view(-1)
        mask = self._index_mask[req_indices].view(-1)
        # NB: Return value has dynamic shape (depends on mask nnz), which
        #     implies stream sync if CUDA is used.
        return indices[mask]


# Helper class for _PackedStepIndexer and _UnpackedStepIndexer, facilitating the
# selection of memory locations of tokens associated with given sets of requests,
# for memory layouts that can be parametrized via request offsets and step stride.
class _StridedStepIndexTranslator(_StepIndexTranslator):
    def __init__(
        self,
        *,
        num_steps: torch.Tensor,
        req_offsets: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        index_dtype: Optional[torch.dtype] = None,
        step_stride: Optional[int] = None,
    ):
        """Build the index.

        Allows to specify a custom stride for steps dimension.

        Arguments:
            index_dtype: torch.dtype to use for indices (defaults to torch.int32).
            num_steps (index_dtype): Number of steps/tokens for each request
            req_offsets (index_dtype): Index offset at which the data for each request starts.
                                       If not provided, it is computed using calculate_request_offsets(),
                                       assuming dense packing of tokens (grouped by request). Overriding
                                       this also allows for "request major" indexing into rectangular
                                       tensors.
            max_steps (int): The largest value allowed to occur in num_steps.
                             If not provided, it is computed from 'num_steps'.
            step_stride: Additional stride to multiply 'steps_dim' with (defaults to 1). Allows,
                         e.g., "step major" indexing into rectangular tensors.
        """
        self._step_stride = step_stride
        super().__init__(
            num_steps=num_steps,
            req_offsets=req_offsets,
            max_steps=max_steps,
            index_dtype=index_dtype,
        )

    @override
    def _compute_index_map(
        self,
        index_dtype: torch.dtype,
        steps_dim: torch.Tensor,
        req_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if self._step_stride is not None:
            steps_dim *= self._step_stride  # in-place OK
        return req_offsets.unsqueeze(-1) + steps_dim.unsqueeze(0)


# In sample_async(), each request contains a different number of output positions
# (a.k.a. 'steps') and 'logits_cuda' (and other tensors derived from it) packs those
# tokens into a single contiguous array, with the 'step' axis being the rapidly
# changing one.
#
# The class below builds an index to simplify selecting the linear indices of the
# tokens associated with a given set of requests.
#
# NB: Consider switching to torch.nested (cf. https://github.com/pytorch/pytorch/issues/80577)
class _PackedStepIndexer(_StridedStepIndexTranslator):
    def __init__(
        self,
        *,
        num_steps: torch.Tensor,
        req_offsets: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        index_dtype: Optional[torch.dtype] = None,
    ):
        """Build the index.

        Arguments:
            index_dtype: torch.dtype to use for indices (defaults to torch.int32).
            num_steps (index_dtype): Number of steps/tokens for each request
            req_offsets (index_dtype): Index offset at which the data for each request starts.
                                       If not provided, it is computed using calculate_request_offsets().
            max_steps (int): The largest value allowed to occur in num_steps.
                             If not provided, it is computed from 'num_steps'.
        """
        super().__init__(
            num_steps=num_steps,
            req_offsets=req_offsets,
            max_steps=max_steps,
            index_dtype=index_dtype,
        )


# After gathering results with _PackedStepIndexer in TorchSampler._sample_batched_by_strategy,
# they need to be scattered into result buffers in TorchSampler._unbatch_sampling_results.
# This helper class provides the translation from linear packed request + step/token indices
# to unpacked / rectangular-tensor (but still linearized) request + step/token indices.
#
# NB: Consider switching to torch.nested (cf. https://github.com/pytorch/pytorch/issues/80577)
class _UnpackedStepIndexer(_StridedStepIndexTranslator):
    class DimOrder(enum.Enum):
        SLOT_MAJOR = enum.auto()
        STEP_MAJOR = enum.auto()

    def __init__(
        self,
        *,
        seq_slots: torch.Tensor,
        num_steps: torch.Tensor,
        dim_order: DimOrder = DimOrder.SLOT_MAJOR,
        steps_dim_size: int,
        slots_dim_size: Optional[int] = None,
        index_dtype: Optional[torch.dtype] = None,
    ):
        """Build the index.

        Arguments:
            index_dtype: torch.dtype to use for indices (defaults to torch.int32).
            seq_slots (index_dtype): Request indices in unpacked tensor, enumerated in packed tensor
                                     request order.
            num_steps (index_dtype): Number of steps/tokens for each request
            dim_order: Memory layout of indexed tensor.
            steps_dim_size (int): The extent of the step dimension in the unpacked tensor.
            slots_dim_size (int): The extent of the slot dimension in the unpacked tensor.
                                  Required if dim_order is DimOrder.STEP_MAJOR.
        """
        if dim_order is self.DimOrder.SLOT_MAJOR:
            super().__init__(
                num_steps=num_steps,
                req_offsets=(steps_dim_size * seq_slots),
                max_steps=steps_dim_size,
                index_dtype=index_dtype,
            )
        elif dim_order is self.DimOrder.STEP_MAJOR:
            if slots_dim_size is None:
                raise ValueError("slots_dim_size required for step-major order")
            super().__init__(
                num_steps=num_steps,
                req_offsets=seq_slots,  # no need for stride here
                max_steps=steps_dim_size,
                index_dtype=index_dtype,
                step_stride=slots_dim_size,
            )
        else:
            raise ValueError(f"Invalid dim_order: {dim_order}")


@dataclass(kw_only=True)
class BeamHistory:
    """
    Beam history class for beam search.
    This class is used to store the corrected tokens and logprobs for each beam.
    It is used to update the beam history for each beam.
    """

    tokens: torch.Tensor
    logprobs: torch.Tensor | None = None
    logprobs_indices: torch.Tensor | None = None
    cum_logprobs: torch.Tensor | None = None


BeamHistoryBuilder: TypeAlias = Callable[[], BeamHistory | None]
"""Builder for BeamHistory.

Used to defer possibly unnecessary host-tensor construction until update_requests().
"""


@dataclass(kw_only=True)
class SamplingRequestsMetadata:
    """Metadata for the sampling requests."""

    req_num_generated_tokens: torch.Tensor
    """The number of generated tokens for each sampling request.
    In beam search, this uses the incoming beam width."""
    req_num_generated_tokens_output: torch.Tensor
    """The number of generated tokens for each sampling request.
    In beam search, this uses the outgoing beam width."""
    req_num_beams: torch.Tensor
    """The number of beams for each sampling request."""
    req_num_steps: torch.Tensor
    """The number of generation steps for each sampling request."""
    req_offsets: torch.Tensor
    """The start offsets of the sampling requests in the raw logits."""


@dataclass(kw_only=True)
class SampleStateTensorsHostTorch(SampleStateTensors):
    finish_reasons: torch.Tensor | None
    first_finish_reasons: torch.Tensor | None
    logprobs_state: LogProbsState | None = None

    def finish_reasons_list(self) -> FinishReasonsList:
        """`(num_seq_slots, num_steps)`"""
        # step, slot, beam => slot, step, beam
        finish_reasons = self.finish_reasons
        if finish_reasons is None:
            return []
        else:
            return finish_reasons.permute(1, 0, 2).tolist()


@dataclass(kw_only=True)
class SampleStateTorch(SampleState[SampleStateTensorsHostTorch, SampleStateTensors]):
    beam_history_builders: list[BeamHistoryBuilder | None] | None = None


@dataclass(kw_only=True, frozen=True)
class _BeamHistoryLogProbsSlices:
    """Correlated beam-history log-prob tensors; all three fields are bound together."""

    sampled_log_probs: torch.Tensor
    sampled_logprobs_indices: torch.Tensor
    cum_logprobs: torch.Tensor


@dataclass(kw_only=True, frozen=True)
class _BeamHistoryTensors:
    """Beam-history tensor slices.

    Used to carry both device-side views (before D2H) and host-side
    snapshots (after D2H). `log_probs` is bound iff log-probs are
    requested.
    """

    cache_indirection: torch.Tensor
    current_path: torch.Tensor
    log_probs: _BeamHistoryLogProbsSlices | None


def _gather_beam_path(
    *, current_path: torch.Tensor, cache_indirection: torch.Tensor
) -> torch.Tensor:
    """Gather the correct tokens for each beam from current_path."""
    new_path = torch.zeros_like(current_path)
    torch.gather(input=current_path, dim=0, index=cache_indirection, out=new_path)
    return new_path


class _SideStreamCopier:
    """Batch non-blocking D2H copies onto a private side stream.

    Inside the `with` block, stage_copy_to_host(src) stages a copy and
    returns a pinned-CPU destination. commit() then issues all staged
    copies on the side stream in a single stream-context and records
    (and returns) an event after them, or None when nothing was staged.

    Caller contract: src must not be mutated on the main stream, and
    the returned host tensor must not be read, until the event has
    been synced host-side. Each copier is single-use.
    """

    def __init__(
        self,
        side_stream: torch.cuda.Stream,
        side_stream_ctx: torch.cuda.StreamContext,
    ) -> None:
        self._side_stream = side_stream
        self._side_stream_ctx = side_stream_ctx
        self._tasks: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.event: torch.cuda.Event | None = None

    def stage_copy_to_host(self, src: torch.Tensor) -> torch.Tensor:
        """Stage a non-blocking D2H copy of src and return its pinned-CPU dst.

        The copy is not issued until `commit()` runs; the returned host
        tensor is only valid after the resulting event has been synced.
        """
        dst = torch.empty_like(src, device="cpu", pin_memory=prefer_pinned())
        self._tasks.append((dst, src))
        return dst

    def commit(self) -> torch.cuda.Event | None:
        """Issue all staged copies and record (and return) an event after them, or None if none were staged."""
        if not self._tasks:
            self.event = None
            return None
        self._side_stream.wait_stream(torch.cuda.current_stream())
        with self._side_stream_ctx:
            for dst, src in self._tasks:
                dst.copy_(src, non_blocking=True)
        self._tasks.clear()
        event = torch.cuda.Event()
        event.record(self._side_stream)
        self.event = event
        return event


class AsyncWorkerMixin:
    """
    Mixin that adds the ability to fork off operations to run on a worker
    thread (particularly D2H copies). If the async worker isn't active,
    operations will seamlessly run on the main thread.

    Also owns a lazily-allocated private D2H side stream, handed out via
    _make_side_stream_copier for batched non-blocking D2H copies.
    """

    MAX_WORKERS = 1

    def _async_worker_active(self) -> bool:
        return getattr(self, "_async_worker", None) is not None

    def _async_worker_init(self, enable_async_worker: bool) -> None:
        self._enable_async_worker = enable_async_worker
        self._async_worker: futures.ThreadPoolExecutor | None = None
        self._async_worker_futures: list[futures.Future[Any]] = []
        # Private D2H side stream + cached stream context shared by all
        # speculative beam-history copiers.
        self._d2h_side_stream: torch.cuda.Stream = torch.cuda.Stream()
        self._d2h_side_stream_ctx: torch.cuda.StreamContext = torch.cuda.stream(
            self._d2h_side_stream
        )

    def async_worker_enabled(self) -> bool:
        return getattr(self, "_enable_async_worker", False)

    def async_worker_start(self) -> None:
        assert self.async_worker_enabled()
        if not self._async_worker_active():

            def _async_worker_initializer(device_id: int) -> None:
                # The current device is set per thread, so we need to set it
                # again here
                torch.cuda.set_device(device_id)
                # Submit the host copies in a separate stream to prevent the
                # blocking copies from gating subsequent async work
                torch.cuda.set_stream(torch.cuda.Stream())

            self._async_worker = futures.ThreadPoolExecutor(
                max_workers=self.MAX_WORKERS,
                initializer=_async_worker_initializer,
                initargs=(torch.cuda.current_device(),),
            )

    def async_worker_stop(self) -> None:
        assert self.async_worker_enabled()
        if self._async_worker_active():
            assert self._async_worker is not None
            self._async_worker.shutdown(wait=True)
            self._async_worker = None

    @torch.inference_mode()
    def _async_copy_to_host(
        self, copy_ready: torch.cuda.Event, dest: torch.Tensor, src: torch.Tensor
    ) -> None:
        # Make sure the async work takes place after all prior operations on
        # the primary stream. synchronize() is intentionally chosen instead of
        # wait() here; otherwise, blocking copies will stall subsequent CUDA
        # API calls on the main stream/thread
        copy_ready.synchronize()

        # Note that the omission of non_blocking=True here is intentional; Work
        # submitted to the async worker is expected to block at the end,
        # consistent with the semantics of futures
        dest.copy_(src)

    def _copy_to_host(self, src: torch.Tensor) -> torch.Tensor:
        dest = torch.empty_like(src, device="cpu", pin_memory=prefer_pinned())
        if self._async_worker_active():
            # Create a snapshot of the source on the main stream, so as to
            # guarantee that the tensor data hasn't been modified before the
            # copy. This precaution is only needed because the copy will
            # execute on a side stream and thus there is no guarantee that
            # future operations on the main stream won't race to modify the
            # tensor data before we copy it.
            src_snapshot = src.clone()

            # Record an event on the main thread/stream that we will
            # synchronize with on the worker thread/stream
            copy_ready = torch.cuda.Event()
            copy_ready.record()

            # Submit the copy to the async worker thread
            assert self._async_worker is not None
            result = self._async_worker.submit(
                self._async_copy_to_host, copy_ready, dest, src_snapshot
            )

            # Save the future, so that we can await it later
            self._async_worker_futures.append(result)
        else:
            # If the async worker is not in use, just copy as usual
            dest.copy_(src, non_blocking=True)
        return dest

    @contextmanager
    def _make_side_stream_copier(self) -> Iterator[_SideStreamCopier]:
        """Yield a fresh copier bound to the shared D2H side stream.

        Staged copies are committed on normal exit; the resulting event
        is exposed as `copier.event` (`None` if nothing was staged or if
        the `with` body raised).
        """
        copier = _SideStreamCopier(self._d2h_side_stream, self._d2h_side_stream_ctx)
        yield copier
        copier.commit()

    def _record_sampler_event(
        self, side_stream_event: torch.cuda.Event | None = None
    ) -> SamplerEvent:
        """Record a SamplerEvent on the main stream.

        side_stream_event, if given, is forwarded so SamplerEvent.synchronize
        also awaits the side-stream copies host-side.
        """
        cuda_event = torch.cuda.Event()
        cuda_event.record()

        # Transfer ownership to worker_futures and re-initialize
        if self._async_worker_active():
            worker_futures = self._async_worker_futures
            self._async_worker_futures = []
        else:
            worker_futures = None

        return SamplerEvent(
            cuda_event=cuda_event,
            side_stream_event=side_stream_event,
            worker_futures=worker_futures,
        )


class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
    DEFAULT_MAX_STOP_WORD_LENGTH: Final[int] = 20
    DEFAULT_MAX_STOP_WORDS: Final[int] = 10

    SampleState = SampleStateTorch

    @override
    def get_cache_indirection(self) -> torch.Tensor | None:
        if (beam_search_store := self.store.beam_search_store) is not None:
            return beam_search_store.cache_indirection
        return None

    @override
    def is_generation_model(self) -> bool:
        return True

    class PenaltyHandler:
        """Applies the occurrence penalties: repetition, presence and frequency.

        These rescale or subtract from a token's logit based on how often it has already
        occurred, and run before the sampling strategy divides by temperature. Bans that
        force a logit to -inf (min_length, bad words, no-repeat-ngram) are a different
        kind of transform and live in ``TokenBanHandler``.

        The implementation follows the C++ ``batchApplyPenalty`` kernel
        (``cpp/tensorrt_llm/kernels/penaltyKernels.cu``) as driven by ``PenaltyLayer``.
        Its persistent device state lives in :class:`PenaltyStore`, which documents the
        workspace semantics. Per-slot parameter buffers are filled once per request,
        batched across all requests admitted in a step (``prepare_for_new_request``
        accumulates on the host, ``update_for_new_requests`` issues the device updates).
        Vocab-sized workspaces are allocated lazily and skipped entirely when no matching
        request uses an occurrence penalty.
        """

        @dataclass(kw_only=True)
        class PenaltyStore:
            """Persistent device state: penalty-parameter buffers + occurrence workspace.

            This is the torch counterpart of the tensors ``PenaltyLayer`` allocates, and
            the anchor for the workspace semantics the ops and the handler rely on:

            * The **parameter buffers** (``repetition_cuda`` / ``presence_cuda`` /
              ``frequency_cuda``, plus the ``active_cuda`` gate) are the counterpart of
              ``allocateBuffer`` + ``fillBuffers``: one entry per sequence slot, written
              once per request and gathered every step, never rebuilt on the host.
            * The **occurrence workspace** (``counts_cuda`` and ``presence_prefix_cuda``)
              is the counterpart of ``allocateWorkspace`` / ``mPenaltyWorkspaceDevice``,
              updated incrementally each step. A token in the ignored prompt prefix
              ``[0, prompt_ignore_length)`` only sets ``presence_prefix_cuda``, so it
              contributes to the repetition penalty but not to presence/frequency; every
              other token (the rest of the prompt plus each generated token) increments
              ``counts_cuda``, which drives presence/frequency and -- via ``counts > 0`` --
              repetition as well.
            """

            max_num_sequences: int
            device: torch.device

            # --- Penalty parameters (allocateBuffer counterpart), shape [max_num_sequences] ---
            repetition_cuda: torch.Tensor
            """float32; per-slot repetition penalty (default 1.0)."""
            presence_cuda: torch.Tensor
            """float32; per-slot presence penalty (default 0.0)."""
            frequency_cuda: torch.Tensor
            """float32; per-slot frequency penalty (default 0.0)."""
            active_cuda: torch.Tensor
            """bool[slots]; whether a slot has an active occurrence penalty."""
            has_previous_token_cuda: torch.Tensor
            """bool[slots]; whether ``new_tokens`` contains a token to accumulate."""

            # --- Occurrence workspace (allocateWorkspace counterpart), allocated lazily ---
            counts_cuda: torch.Tensor | None = None
            """int32[slots, vocab_size] or None; occurrence counts (see class docstring)."""
            presence_prefix_cuda: torch.Tensor | None = None
            """bool[slots, vocab_size] or None; ignored-prompt-prefix presence mask."""

            # Per-step request metadata, staged into persistent device buffers by
            # ``stage_request_metadata`` so the hot path does not allocate per step.
            request_offsets_cuda: torch.Tensor | None = None
            request_num_steps_cuda: torch.Tensor | None = None

            @classmethod
            def create(
                cls, *, max_num_sequences: int, device: torch.device
            ) -> "TorchSampler.PenaltyHandler.PenaltyStore":
                """Allocate the vocab-independent buffers with their no-op defaults.

                ``inference_mode(False)`` guards every allocation in this class: the
                buffers persist across sampler steps and are mutated in place later, which
                inference-mode tensors forbid.
                """
                with torch.inference_mode(False):
                    return cls(
                        max_num_sequences=max_num_sequences,
                        device=device,
                        repetition_cuda=torch.ones(
                            max_num_sequences, dtype=torch.float32, device=device
                        ),
                        presence_cuda=torch.zeros(
                            max_num_sequences, dtype=torch.float32, device=device
                        ),
                        frequency_cuda=torch.zeros(
                            max_num_sequences, dtype=torch.float32, device=device
                        ),
                        active_cuda=torch.zeros(max_num_sequences, dtype=torch.bool, device=device),
                        has_previous_token_cuda=torch.zeros(
                            max_num_sequences, dtype=torch.bool, device=device
                        ),
                    )

            def ensure_workspace(self, *, vocab_size: int, needs_prefix: bool) -> None:
                """Allocate the vocab-sized workspace on first use.

                Deferred because ``vocab_size`` is only known once logits arrive, mirroring
                ``PenaltyLayer::allocateWorkspace`` being gated on penalty usage. The prefix
                mask is allocated only if some request has used ``prompt_ignore_length``.
                """
                with torch.inference_mode(False):
                    if self.counts_cuda is None:
                        self.counts_cuda = torch.zeros(
                            (self.max_num_sequences, vocab_size),
                            dtype=torch.int32,
                            device=self.device,
                        )
                    if needs_prefix and self.presence_prefix_cuda is None:
                        self.presence_prefix_cuda = torch.zeros(
                            (self.max_num_sequences, vocab_size),
                            dtype=torch.bool,
                            device=self.device,
                        )

            def stage_request_metadata(
                self, request_offsets_host: torch.Tensor, request_num_steps_host: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Copy this step's ``[R]`` request metadata into persistent device buffers.

                The host tensors are already pinned by the caller, so each step costs two
                small async H2D copies into a reused allocation rather than two fresh
                device tensors. Returned views are only valid until the next call.
                """
                num_requests = request_offsets_host.numel()
                with torch.inference_mode(False):
                    if (
                        self.request_offsets_cuda is None
                        or self.request_offsets_cuda.numel() < num_requests
                    ):
                        capacity = max(num_requests, self.max_num_sequences)
                        self.request_offsets_cuda = torch.empty(
                            capacity, dtype=request_offsets_host.dtype, device=self.device
                        )
                        self.request_num_steps_cuda = torch.empty(
                            capacity, dtype=request_num_steps_host.dtype, device=self.device
                        )
                assert self.request_num_steps_cuda is not None
                offsets = self.request_offsets_cuda[:num_requests]
                num_steps = self.request_num_steps_cuda[:num_requests]
                offsets.copy_(request_offsets_host, non_blocking=True)
                num_steps.copy_(request_num_steps_host, non_blocking=True)
                return offsets, num_steps

        @dataclass(kw_only=True)
        class _SlotState:
            """Per-slot host-only bookkeeping (never read by the ops)."""

            prompt_ignore_length: int
            initialized: bool = False

        def __init__(
            self,
            *,
            max_num_sequences: int,
            device: torch.device | str,
        ):
            self._max_num_sequences = max_num_sequences
            self._device = torch.device(device)
            # Whether any (past or current) active request uses prompt_ignore_length > 0,
            # which requires allocating the presence-prefix mask.
            self._needs_prefix = False
            self._num_active_slots = 0
            # Per-slot state; None marks a slot without active occurrence penalties.
            self._slots: list[TorchSampler.PenaltyHandler._SlotState | None] = [
                None
            ] * max_num_sequences
            # Slots admitted this step that carry an occurrence penalty, with their
            # parameters; drained by ``update_for_new_requests``.
            self._new_slots: list[int] = []
            self._new_repetition: list[float] = []
            self._new_presence: list[float] = []
            self._new_frequency: list[float] = []
            self.store = self.PenaltyStore.create(
                max_num_sequences=max_num_sequences, device=self._device
            )

        def _to_device(self, values: list[int], dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor(values, dtype=dtype, pin_memory=prefer_pinned()).to(
                self._device, non_blocking=True
            )

        def prepare_for_new_request(self, request: LlmRequest, slot: int) -> None:
            """Record the slot's penalty parameters for this step's batched flush.

            Called from ``TorchSampler.setup_sampler_step`` for each new request, mirroring
            ``PenaltyLayer::setup`` (``fillBuffers`` + per-``batchSlot`` ``setZero``). This
            only touches host state; ``update_for_new_requests`` issues the device updates
            for all requests admitted in the step at once. Inactive slots are never
            gathered, so their stale parameters/counts are left untouched.
            """
            was_active = self._slots[slot] is not None
            if not (_get_max_beam_width(request) == 1 and _has_occurrence_penalty(request)):
                self._slots[slot] = None
                if was_active:
                    self._num_active_slots -= 1
                return

            sampling_config = request.sampling_config
            repetition = _unwrap_singleton(sampling_config.repetition_penalty)
            presence = _unwrap_singleton(sampling_config.presence_penalty)
            frequency = _unwrap_singleton(sampling_config.frequency_penalty)
            prompt_ignore_length = _unwrap_singleton(sampling_config.prompt_ignore_length)
            # min(prompt_ignore_length, inputLen), matching the C++ kernel.
            prompt_ignore_length = min(
                prompt_ignore_length if prompt_ignore_length is not None else 0,
                request.py_orig_prompt_len,
            )
            if prompt_ignore_length > 0:
                self._needs_prefix = True

            self._slots[slot] = self._SlotState(prompt_ignore_length=prompt_ignore_length)
            if not was_active:
                self._num_active_slots += 1

            self._new_slots.append(slot)
            self._new_repetition.append(repetition if repetition is not None else 1.0)
            self._new_presence.append(presence if presence is not None else 0.0)
            self._new_frequency.append(frequency if frequency is not None else 0.0)

        def update_for_new_requests(self, *, new_seq_slots_cuda_long: torch.Tensor) -> None:
            """Flush this step's admissions to the device in a handful of batched updates.

            ``new_seq_slots_cuda_long`` holds *every* slot admitted this step. Clearing the
            active gate and the pending-token flag across all of them also covers slot
            reuse: a slot whose prior occupant was penalized but whose new occupant is not
            must read False.
            """
            store = self.store
            store.active_cuda.index_fill_(0, new_seq_slots_cuda_long, False)
            store.has_previous_token_cuda.index_fill_(0, new_seq_slots_cuda_long, False)

            if not self._new_slots:
                return

            slots_cuda = self._to_device(self._new_slots, torch.int64)
            # One [3, N] host tensor -> one H2D for all three parameter buffers.
            params_cuda = torch.tensor(
                [self._new_repetition, self._new_presence, self._new_frequency],
                dtype=torch.float32,
                pin_memory=prefer_pinned(),
            ).to(self._device, non_blocking=True)
            store.repetition_cuda.index_copy_(0, slots_cuda, params_cuda[0])
            store.presence_cuda.index_copy_(0, slots_cuda, params_cuda[1])
            store.frequency_cuda.index_copy_(0, slots_cuda, params_cuda[2])
            store.active_cuda.index_fill_(0, slots_cuda, True)

            # Re-zero the workspace rows so a prior occupant's counts do not leak in.
            if store.counts_cuda is not None:
                store.counts_cuda.index_fill_(0, slots_cuda, 0)
            if store.presence_prefix_cuda is not None:
                store.presence_prefix_cuda.index_fill_(0, slots_cuda, False)

            self._new_slots.clear()
            self._new_repetition.clear()
            self._new_presence.clear()
            self._new_frequency.clear()

        def _initialize_workspace(
            self,
            request: LlmRequest,
            state: "TorchSampler.PenaltyHandler._SlotState",
            vocab_size: int,
        ) -> None:
            """Initialize one regular slot from its prompt exactly once."""
            if state.initialized:
                return

            slot = request.py_seq_slot
            assert slot is not None
            counts_cuda = self.store.counts_cuda
            assert counts_cuda is not None

            prompt = request.get_tokens(0)[: request.py_orig_prompt_len]
            state.initialized = True
            if not prompt:
                return

            # One conversion for the whole prompt; the split point is just
            # prompt_ignore_length, so the two groups are plain slices.
            tokens = self._to_device(prompt, torch.int64)
            prefix_tokens = tokens[: state.prompt_ignore_length]
            counted_tokens = tokens[state.prompt_ignore_length :]

            # Multimodal models place placeholder ids >= vocab_size in the prompt (see
            # _torch/models/modeling_multimodal_utils.py), so out-of-range ids reach us
            # here and must be dropped before they index the workspace.
            counted_tokens = counted_tokens[(counted_tokens >= 0) & (counted_tokens < vocab_size)]
            prefix_tokens = prefix_tokens[(prefix_tokens >= 0) & (prefix_tokens < vocab_size)]

            Fusions.update_occurrence_workspace(
                counts_cuda,
                self.store.presence_prefix_cuda,
                torch.full_like(counted_tokens, slot),
                counted_tokens,
                torch.full_like(prefix_tokens, slot),
                prefix_tokens,
            )

        def update_token_counts(
            self,
            updates: list[tuple[int, list[int]]],
        ) -> None:
            """Commit finalized sampled tokens that replaced the device pending token.

            This is used after sampler-side postprocessing has finalized a multi-token
            result. The complete confirmed sequence is counted here, then the raw first
            token left in ``new_tokens`` is marked consumed so the next kernel cannot count
            it again. Regular one-token sampling never calls this method and keeps its
            fused device-pending fast path.
            """
            if not updates or self._num_active_slots == 0:
                return

            counts_cuda = self.store.counts_cuda
            assert counts_cuda is not None
            vocab_size = counts_cuda.size(-1)
            consumed_slots: list[int] = []
            counted_slots: list[int] = []
            counted_tokens: list[int] = []

            for slot, tokens in updates:
                if self._slots[slot] is None:
                    continue
                consumed_slots.append(slot)
                for token in tokens:
                    if 0 <= token < vocab_size:
                        counted_slots.append(slot)
                        counted_tokens.append(token)

            if consumed_slots:
                self.store.has_previous_token_cuda.index_fill_(
                    0, self._to_device(consumed_slots, torch.int64), False
                )

            if not counted_tokens:
                return

            Fusions.update_occurrence_workspace(
                counts_cuda,
                self.store.presence_prefix_cuda,
                self._to_device(counted_slots, torch.int64),
                self._to_device(counted_tokens, torch.int64),
            )

        @nvtx_range("apply_penalties")
        @torch.inference_mode()
        def apply(
            self,
            logits: torch.Tensor,
            requests: list[LlmRequest],
            *,
            new_tokens: torch.Tensor,
            seq_slots: torch.Tensor,
            request_offsets: torch.Tensor,
            request_num_steps: torch.Tensor,
            is_draft_batch: bool = False,
        ) -> None:
            """Apply the occurrence penalties to ``logits`` in place.

            ``logits`` is the packed generated-token logits ``[sum(num_steps * num_beams),
            vocab_size]``; request ``r`` owns ``request_num_steps[r]`` consecutive rows
            starting at ``request_offsets[r]``, in beam-major / step-minor order.
            ``request_offsets`` / ``request_num_steps`` are the caller's pinned host
            tensors and are staged to the device here.

            Args:
                is_draft_batch: draft batches share this sampler but draw ``py_seq_slot``
                    from a separate numbering space that collides with target slots, so
                    penalizing them would read/write an unrelated target request's
                    occurrence state; skip them like the pending-steps tracking.
            """
            if is_draft_batch or not requests or self._num_active_slots == 0:
                return

            # Cheap per-batch scan so the vocab-sized workspace is only allocated when this
            # batch actually contains a penalized request.
            active_requests: list[tuple[LlmRequest, "TorchSampler.PenaltyHandler._SlotState"]] = []
            for request in requests:
                slot = request.py_seq_slot
                assert slot is not None
                state = self._slots[slot]
                if state is not None:
                    active_requests.append((request, state))
            if not active_requests:
                return

            store = self.store
            store.ensure_workspace(vocab_size=logits.size(-1), needs_prefix=self._needs_prefix)
            counts_cuda = store.counts_cuda
            assert counts_cuda is not None
            for request, state in active_requests:
                self._initialize_workspace(request, state, logits.size(-1))

            request_offsets_cuda, request_num_steps_cuda = store.stage_request_metadata(
                request_offsets, request_num_steps
            )
            Fusions.apply_batched_occurrence_penalties(
                logits,
                counts_cuda,
                store.presence_prefix_cuda,
                store.active_cuda,
                store.has_previous_token_cuda,
                new_tokens,
                seq_slots,
                request_offsets_cuda,
                request_num_steps_cuda,
                store.repetition_cuda,
                store.presence_cuda,
                store.frequency_cuda,
            )
            # Arm has_previous_token for the slots this call penalized (active, num_steps > 0)
            # so the next apply folds their sampled new_tokens. Done here rather than in the
            # compiled op because the op's fold reads the flag for every request row; flipping
            # it in the same graph would make the result depend on execution order within the
            # kernel.
            #
            # The scan is kept on the host deliberately. The same thing can be expressed on
            # device as active_cuda[seq_slots] & (num_steps > 0), avoiding this loop and the
            # H2D, but that costs several extra kernel launches and measured 5-7us slower for
            # batches up to 32 and no better at 64-256: the loop overlaps with the model
            # forward, the launches do not.
            pending_token_slots: list[int] = []
            for request, num_steps in zip(requests, request_num_steps.tolist()):
                slot = request.py_seq_slot
                if slot is None:
                    continue
                if self._slots[slot] is not None and num_steps > 0:
                    pending_token_slots.append(slot)
            if pending_token_slots:
                store.has_previous_token_cuda.index_fill_(
                    0, self._to_device(pending_token_slots, torch.int64), True
                )

    @dataclass(kw_only=True)
    class BeamSearchStore:
        """Auxiliary data structures required for beam search."""

        cache_indirection: torch.Tensor
        """Shape: batch_size, beam_width, attention_size
           Usage: Stores the cache indirection necessary for beam search sampling"""
        cache_indirection_buffer: torch.Tensor
        """Shape: batch_size, beam_width, attention_size
           Usage: A second buffer used to update the cache indirection during sampling"""
        cum_log_probs: torch.Tensor
        """Shape: batch_size, beam_width
           Usage: Stores the current cumulative logprob of each active beam for faster sampling"""
        first_finish_reasons: torch.Tensor
        """Shape: batch_size, beam_width
           Usage: Stores the first finish reason for each beam"""
        predecessor_beams: torch.Tensor
        """Shape: batch_size, beam_width
           Usage: Stores the predecessor beams for each beam used for stop word detection"""
        original_tokens: torch.Tensor
        """Shape: batch_size, beam_width, sequence_length
           Usage: Stores the original tokens for each beam.
           This is used to recover the original tokens for each beam when streaming is enabled"""
        seq_offsets: torch.Tensor
        """Shape: (max_num_sequences,), dtype int64
           Usage: Cached `arange(max_num_sequences) * max_beam_width` used by
           ``beam_search_sampling_batch`` to flatten (batch_idx, beam_idx) pairs."""
        beam_idx_arange: torch.Tensor
        """Shape: (max_beam_width,), dtype int32
           Usage: Cached `arange(max_beam_width)` used as the scatter source in the
           per-step ``cache_indirection.scatter_``."""

    @dataclass(kw_only=True)
    class Store:
        new_tokens: torch.Tensor
        """Device tensor containing latest sampled tokens.

        Shape: See cpp DecoderState.getAllNewTokens().
        """
        beam_search_store: "TorchSampler.BeamSearchStore | None" = None
        """Holds data related to beam search."""
        log_probs_store: LogProbsStore
        """Holds data related to log-probs handling."""

    def _create_store(self) -> Store:
        # Tensors necessary for all sampling methods
        new_tokens = int_tensor(self.NEW_TOKENS_SHAPE)

        # Only used for logprobs processing or beam search
        sampled_log_probs = torch.empty(self.LOGPROBS_SHAPE, device="cuda", dtype=torch.float32)
        # Only used for logprobs processing
        sampled_log_prob_indices = torch.empty(
            self.LOGPROBS_SHAPE, device="cuda", dtype=torch.int32
        )
        sampled_log_prob_ranks = torch.empty(self.LOGPROBS_SHAPE, device="cuda", dtype=torch.int32)
        # These are 0 sized tensors, if topk-logprobs are not used
        topk_indices = torch.empty(self.TOPK_LOGPROBS_SHAPE, device="cuda", dtype=torch.int32)
        topk_vals = torch.empty(self.TOPK_LOGPROBS_SHAPE, device="cuda", dtype=torch.float32)
        log_probs_store = LogProbsStore(
            sampled_log_prob_indices=sampled_log_prob_indices,
            sampled_log_probs=sampled_log_probs,
            sampled_log_prob_ranks=sampled_log_prob_ranks,
            topk_indices=topk_indices,
            topk_vals=topk_vals,
        )

        beam_search_store = None
        if self._use_beam_search:
            cache_indirection = torch.empty(
                self.CACHE_INDIRECTION_SHAPE, device="cuda", dtype=torch.int
            )
            cache_indirection_buffer = int_tensor(self.CACHE_INDIRECTION_SHAPE)
            cum_log_probs = torch.empty(
                self.CACHE_INDIRECTION_SHAPE[:-1], device="cuda", dtype=torch.float32
            )
            predecessor_beams = int_tensor(self.CACHE_INDIRECTION_SHAPE[:-1])
            original_tokens = int_tensor(self.CACHE_INDIRECTION_SHAPE)
            first_finish_reasons = int_tensor(self.CACHE_INDIRECTION_SHAPE[:-1])
            seq_offsets = (
                torch.arange(self.max_num_sequences, device="cuda", dtype=torch.int64)
                * self.max_beam_width
            )
            beam_idx_arange = torch.arange(self.max_beam_width, device="cuda", dtype=torch.int32)
            beam_search_store = self.BeamSearchStore(
                cache_indirection=cache_indirection,
                cache_indirection_buffer=cache_indirection_buffer,
                cum_log_probs=cum_log_probs,
                predecessor_beams=predecessor_beams,
                original_tokens=original_tokens,
                first_finish_reasons=first_finish_reasons,
                seq_offsets=seq_offsets,
                beam_idx_arange=beam_idx_arange,
            )
        return self.Store(
            new_tokens=new_tokens,
            log_probs_store=log_probs_store,
            beam_search_store=beam_search_store,
        )

    @dataclass(frozen=True, kw_only=True)
    class Args:
        max_seq_len: int
        max_draft_len: int
        max_num_sequences: int
        max_beam_width: int
        max_total_draft_tokens: int
        disable_overlap_scheduler: bool = False
        enable_async_worker: bool = False
        enable_speculative_beam_history_d2h: bool = False

    def __init__(self, args: Args):
        self.max_seq_len = args.max_seq_len
        self.max_tokens = args.max_total_draft_tokens + 1
        self.max_beam_width = args.max_beam_width
        # Snapshot of `not self._use_beam_search` so the update_requests
        # fast-path avoids a property call per iteration.
        self._batch_fastpath_eligible: bool = self.max_beam_width == 1
        # The current maximum number of topk logprobs which can be stored in the sampler's store
        self.max_topk_logprobs = MAX_TOP_LOGPROBS
        # The maximum number of topk logprobs for the current batch of requests
        self.batch_max_topk_logprobs = 0
        if args.max_total_draft_tokens > 0 and args.max_beam_width > 1:
            raise ValueError("TorchSampler does not support beam search with speculative decoding")
        self.max_num_sequences = args.max_num_sequences
        # With the overlap scheduler, sample_async for step i runs before
        # update_requests for step i-1, so the host-side token lists lag the
        # device state. Track, per seq slot, how many sampled steps have not
        # been folded back into the request yet; bad-words handling uses this
        # to decide whether the newest token must be read device-side.
        self._track_pending_steps = not args.disable_overlap_scheduler
        self._pending_steps = [0] * self.max_num_sequences
        self.NEW_TOKENS_SHAPE = (self.max_tokens, self.max_num_sequences, self.max_beam_width)
        self.CACHE_INDIRECTION_SHAPE = (
            self.max_num_sequences,
            self.max_beam_width,
            self.max_seq_len,
        )
        self.LOGPROBS_SHAPE = (self.max_num_sequences, self.max_beam_width, self.max_tokens)
        self.TOPK_LOGPROBS_SHAPE = (self.max_num_sequences, self.max_tokens, self.max_topk_logprobs)

        # The Torch sampler hard-depends on flashinfer. Enforce it once here, at
        # construction, so the check stays out of the CUDA-graph-captured
        # sampling loop.
        if not IS_FLASHINFER_AVAILABLE:
            raise ImportError(
                "flashinfer is not available, please install the version pinned "
                "in requirements.txt."
            )
        self._grouped_sampler_cls = FlashInferGroupedStrategySampler
        # Per-slot Top-P Decay runtime state (FlashInfer path). Allocated for all
        # sampler instances; only decay-admitted slots are ever read.
        self._top_p_decay = TopPDecayHandler(self.max_num_sequences)

        # Token-ban handling (bad words, no-repeat ngram). The overlap-aware
        # variant is selected once here from whether the overlap scheduler is
        # enabled; only it produces the conditional (stale-host) bans.
        self._token_ban_handler: TokenBanHandler = (
            OverlappedTokenBanHandler()
            if self._track_pending_steps
            else SynchronousTokenBanHandler()
        )

        # AutoDeploy build creates the sampler in inference mode,
        # which would disallow in-place mutating of new_tokens.
        # So, we temporarily exit inference mode.
        with torch.inference_mode(False):
            self.store = self._create_store()
            self._request_grouper: _CachingRequestGrouper[Any] = _CachingRequestGrouper(
                self.max_num_sequences
            )
            self._finish_reasons_handler = FinishReasonsHandler(
                max_stop_word_length=self.DEFAULT_MAX_STOP_WORD_LENGTH,
                max_num_stop_words=self.DEFAULT_MAX_STOP_WORDS,
                max_num_sequences=self.max_num_sequences,
                max_beam_width=self.max_beam_width,
                max_tokens=self.max_tokens,
                max_seq_len=self.max_seq_len,
            )
            assert (
                self.store.new_tokens.shape
                == self._finish_reasons_handler.store.finish_reasons_cuda.shape
            )
            self._penalty_handler = self.PenaltyHandler(
                max_num_sequences=self.max_num_sequences,
                device="cuda",
            )

        # Initialize seed for multi-GPU consistency
        self._global_seed = 42
        self._generator: torch.Generator | None = None

        # Force number of accepted tokens for speculative decoding testing
        self._force_num_accepted_tokens = get_force_num_accepted_tokens()

        self._async_worker_init(args.enable_async_worker)

        # The speculative path bypasses _copy_to_host, so it cannot coexist
        # with the async worker. LlmArgs validation rejects the explicit
        # conflict with sampler_force_async_worker. Confidential compute may
        # also enable the async worker at runtime; if so, disable the path
        # here with a warning.
        self._use_speculative_beam_history_d2h: bool = args.enable_speculative_beam_history_d2h
        if self._use_speculative_beam_history_d2h and self.async_worker_enabled():
            logger.warning(
                "enable_speculative_beam_history_d2h is incompatible with the "
                "sampler async worker (likely auto-enabled by confidential "
                "compute); disabling the speculative beam-history D2H path."
            )
            self._use_speculative_beam_history_d2h = False

        # 1-step-lagged host mirror of first_finish_reasons used by the
        # speculative predictor, indexed by py_seq_slot. None for unoccupied
        # slots or before the first step; all-None in default mode.
        self._prev_first_finish_reasons_host: list[torch.Tensor | None] = [
            None
        ] * self.max_num_sequences

    @staticmethod
    def _is_draft_batch(requests: list[LlmRequest]) -> bool:
        """Whether this batch belongs to the draft model.

        Batches are homogeneous by construction: ModelDrafter builds all-draft
        batches for its sample_async/update_requests calls on this shared
        sampler, and PyExecutor's batches are all-target. The pending-steps
        accounting relies on this to skip draft batches wholesale; assert it so
        a mixed batch fails loudly instead of silently corrupting the counters.
        """
        is_draft: bool = requests[0].py_is_draft
        assert all(r.py_is_draft == is_draft for r in requests), (
            "sampler batch must be homogeneous (all-draft or all-target)"
        )
        return is_draft

    def get_generator(self, device: torch.device) -> torch.Generator:
        """Get a deterministic generator for the specified device.

        Args:
            device: The device to create the generator on

        Returns:
            A torch.Generator with the global seed set
        """
        if self._generator is None:
            # Fallback to a default seed if not set
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self._global_seed)
        assert self._generator.device == device
        return self._generator

    def get_spec_tree_manager(
        self, resource_manager: Optional[ResourceManager]
    ) -> Optional[SpecTreeManager]:
        if resource_manager is None:
            return None
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        if spec_resource_manager is None or not hasattr(spec_resource_manager, "spec_tree_manager"):
            return None
        return spec_resource_manager.spec_tree_manager  # type: ignore

    @property
    def _use_beam_search(self) -> bool:
        return self.max_beam_width > 1

    def _can_use_fast_greedy_path(self, requests: list[LlmRequest]) -> bool:
        """
        Check if we can use the fast argmax path for greedy sampling.
        """
        # Check if all requests use greedy sampling and don't require features
        # that the fast path skips
        for req in requests:
            # vocab_size doesn't affect greediness check
            if _request_strategy(req, vocab_size=2**31) != GREEDY:
                return False

            # Fast path skips logprobs handling
            if req.py_return_log_probs:
                return False
        return True

    @staticmethod
    def _meet_max_token_stop_criteria(
        request: LlmRequest, max_seq_len: int, beam_idx: int = DEFAULT_BEAM_IDX
    ) -> bool:
        num_tokens = request.get_num_tokens(beam_idx)
        return (num_tokens - request.py_orig_prompt_len >= request.py_max_new_tokens) or (
            num_tokens >= max_seq_len
        )

    @staticmethod
    def _meet_stop_token_criteria(
        request: LlmRequest, new_token: int, beam_idx: int = DEFAULT_BEAM_IDX
    ) -> bool:
        if request.py_stop_words_list:
            assert isinstance(request.py_stop_words_list, list), (
                "request.py_stop_words_list should be a list"
            )
            stop_words_list, prefix_sum = request.py_stop_words_list

            # Determine max stop word length to decide optimization path
            max_stop_word_length = prefix_sum[0] if prefix_sum else 0
            for i in range(1, len(prefix_sum)):
                word_length = prefix_sum[i] - prefix_sum[i - 1]
                max_stop_word_length = max(max_stop_word_length, word_length)

            # Fast path: all stop words are single tokens
            if max_stop_word_length == 1:
                return new_token in stop_words_list

            # Slow path: at least one multi-token stop word exists
            tokens = request.get_tokens(beam_idx)
            offset = 0
            for i, offset_end in enumerate(prefix_sum):
                if i > 0:
                    offset = prefix_sum[i - 1]
                stop_word = stop_words_list[offset:offset_end]
                if len(stop_word) > len(tokens):
                    continue
                if tokens[-len(stop_word) :] == stop_word:
                    return True
        return False

    @classmethod
    def _handle_stop_criteria(
        cls, request: LlmRequest, new_token: int, *, max_seq_len: int, beam_idx: int
    ) -> bool:
        """Handle stop criteria and set appropriate finish reasons and state.
        Returns True if generation should stop."""
        if new_token == request.py_end_id:
            request.finish_by(FinishReason.END_ID, beam_idx)
            return True

        if cls._meet_max_token_stop_criteria(request, max_seq_len, beam_idx):
            request.finish_by(FinishReason.LENGTH, beam_idx)
            return True

        if cls._meet_stop_token_criteria(request, new_token, beam_idx):
            request.finish_by(FinishReason.STOP_WORDS, beam_idx)
            return True

        return False

    def _handle_finish_reasons_impl(
        self,
        request: LlmRequest,
        beam_width: int,
        finish_reasons: torch.Tensor,
        finish_reasons_list: list[int],
    ) -> bool:
        """Check if all beams of a request have finished and set the request state accordingly

        Args:
            request: LlmRequest. The request to check.
            beam_width: int. The beam width of the request.
            finish_reasons: torch.Tensor. Shape: (beam_width)
                            The finish reasons for each beam.
            finish_reasons_list: list[int]. The finish reasons for each beam.
        Returns:
            True if all beams have finished, False otherwise.
        """
        if (finish_reasons[:beam_width] != FinishReason.NOT_FINISHED.value).sum() == beam_width:
            request.state = LlmRequestState.GENERATION_COMPLETE
            for beam_idx in range(beam_width):
                request.set_finished_reason(
                    FinishReason(finish_reasons_list[beam_idx]),
                    beam_idx,
                )
            return True
        return False

    def _handle_first_finish_reasons(
        self,
        request: LlmRequest,
        finish_reasons: torch.Tensor,
        finish_reasons_list: list[list[int]],
    ) -> bool:
        """Check if all beams of a request have finished and set the request state accordingly

        Args:
            request: LlmRequest. The request to check.
            finish_reasons: torch.Tensor. Shape: (max_batch_size, max_beam_width)
                            The finish reasons for each beam.
            finish_reasons_list: list[list[int]]. The finish reasons for each beam.
        Returns:
            True if all beams have finished, False otherwise.
        """
        assert request.py_seq_slot is not None
        beam_width = request.py_beam_width
        return self._handle_finish_reasons_impl(
            request,
            beam_width,
            finish_reasons[request.py_seq_slot, :beam_width],
            finish_reasons_list[request.py_seq_slot],
        )

    @staticmethod
    @nvtx_range("update_original_tokens")
    def _update_original_tokens(
        original_tokens: torch.Tensor,
        seq_slots: torch.Tensor,
        seq_lens: torch.Tensor,
        new_tokens: torch.Tensor,
    ) -> None:
        """Update the original tokens storage for the request with the newly sampled tokens

        When using streaming a requests tokens may be altered, leading to wrong results when called multiple times.
        Store the original tokens in a separate buffer to use them as a consistent basis
        when updating the tokens in a request."""
        assert new_tokens.device == original_tokens.device, (
            "new_tokens and original_tokens must be on the same device"
        )
        original_tokens[seq_slots, :, seq_lens] = new_tokens[0, seq_slots, :]

    def handle_logprobs(
        self,
        request: LlmRequest,
        logprobs_state_list: LogProbsStateList | None,
        *,
        count: int,
    ) -> None:
        if request.py_return_log_probs:
            beam_width = request.py_beam_width
            assert request.py_num_logprobs is not None, "request.py_num_logprobs must be provided"
            assert logprobs_state_list is not None, "logprobs_state_list must be provided"
            assert request.py_seq_slot is not None
            token_log_probs = store_logprobs_list_to_request(
                logprobs_state_list,
                request.py_seq_slot,
                beam_width,
                count,
                request.py_num_logprobs,
                simple_format=request.py_logprobs_simple_format,
            )
            request.py_result.append_log_probs(token_log_probs)

    def finish_if_reason(
        self, request: LlmRequest, finish_reasons: FinishReasonsList, *, step: int, beam_idx: int
    ) -> bool:
        assert request.py_seq_slot is not None
        reason = FinishReason(finish_reasons[request.py_seq_slot][step][beam_idx])
        valid_reasons = {FinishReason.END_ID, FinishReason.LENGTH, FinishReason.STOP_WORDS}
        if reason in valid_reasons:
            request.finish_by(reason, beam_idx)
            return True
        return False

    def _process_draft_tokens_greedy(
        self,
        request: LlmRequest,
        new_tokens: list[list[list[int]]],
        finish_reasons: FinishReasonsList,
    ) -> int:
        new_token = add_token(request, new_tokens, beam_idx=DEFAULT_BEAM_IDX)
        stop = self.finish_if_reason(request, finish_reasons, step=0, beam_idx=DEFAULT_BEAM_IDX)
        if stop or get_draft_token_length(request) == 0:
            return 0
        num_accepted = 0

        if self._force_num_accepted_tokens != 0:
            # Force acceptance of up to force_num_accepted_tokens draft tokens
            force_limit = min(self._force_num_accepted_tokens, len(request.py_draft_tokens))
            for _ in request.py_draft_tokens[:force_limit]:
                num_accepted += 1
                new_token = add_token(
                    request, new_tokens, beam_idx=DEFAULT_BEAM_IDX, step=num_accepted
                )
                if self.finish_if_reason(
                    request, finish_reasons, step=num_accepted, beam_idx=DEFAULT_BEAM_IDX
                ):
                    break
        else:
            for draft_token in request.py_draft_tokens:
                if draft_token != new_token:
                    # Reject.
                    break

                num_accepted += 1
                new_token = add_token(
                    request, new_tokens, beam_idx=DEFAULT_BEAM_IDX, step=num_accepted
                )
                if self.finish_if_reason(
                    request, finish_reasons, step=num_accepted, beam_idx=DEFAULT_BEAM_IDX
                ):
                    break
        return num_accepted

    def _process_draft_tokens_tree(
        self,
        request: LlmRequest,
        new_tokens_tensor: torch.Tensor,
        new_tokens_list: list[list[list[int]]],
        finish_reasons: FinishReasonsList,
        spec_tree_manager: SpecTreeManager,
    ) -> int:
        """Tree verification for draft token tree based speculative decoding.

        This function will only be called for the target model.

        Verification logic:
            Find the longest prefix match. Since each node in the tree has a related path,
            we can find the longest match by comparing all the paths.
        Args:
            request: LlmRequest. The request with draft tokens.
            new_tokens: torch.Tensor. [max_total_draft_tokens + 1, max_num_sequences, max_beam_width], host buffer.
                        The tokens generated by the target model
                        The relationship between [max_total_draft_tokens + 1] and the draft token tree:
                        If the current node is accepted, what is the NEXT token_id that the target model will generate?
                        For example, new_tokens[0, req_idx, 1] indicates the NEXT token_id sampled from the root
                        node in the draft token tree if it is accepted.
                        We know that the root node in the draft token tree is always accepted. Therefore,
                        new_tokens[0, req_idx, 1] indicates the token_id following the root node,
                        corresponding to the first layer in the draft token tree (the root node is the 0th layer).
                        Similarly, new_tokens[1, req_idx, 1] represents the NEXT token_id if the first token in the
                        first layer of the draft tokens tree is accepted.
            spec_tree_manager: SpecTreeManager. which contains the tree structure and other meta information
                               of the tree.
        """
        # handle the target model request
        # For the target model, we will do the tree verification logic.
        seq_slot = request.py_seq_slot
        assert seq_slot is not None
        eagle_paths = spec_tree_manager.get_eagle_paths(seq_slot)

        all_draft_tokens = torch.tensor(request.py_draft_tokens)  # [max_total_draft_tokens]
        all_target_tokens = new_tokens_tensor[:, seq_slot, :].squeeze(
            -1
        )  # [max_total_draft_tokens]
        assert all_target_tokens.shape[0] == spec_tree_manager.max_total_draft_tokens + 1

        longest_accepted_len = 0
        longest_match_path_idx = -1

        for path_idx, path in enumerate(eagle_paths):
            path_exclude_root = (
                path[1:] - 1
            )  # [max_draft_len], '[1:]' since the new_tokens does not contain the root node.
            # '-1' is the index shift after exclude the root node.
            draft_tokens_indices = path_exclude_root[path_exclude_root >= 0]  # [max_draft_len]
            target_tokens_indices = path[path >= 0]  # [max_draft_len + 1]

            assert len(draft_tokens_indices) == len(target_tokens_indices) - 1

            cur_draft_tokens = all_draft_tokens[draft_tokens_indices]
            cur_target_tokens = all_target_tokens[target_tokens_indices]

            cur_accepted_len = cast(
                int,
                torch.cumprod((cur_draft_tokens == cur_target_tokens[:-1]).int(), dim=-1)
                .sum()
                .item(),
            )

            # Accepted one more token from the target model.
            cur_accepted_len += 1

            if cur_accepted_len > longest_accepted_len:
                longest_accepted_len = cur_accepted_len
                longest_match_path_idx = path_idx

        assert longest_accepted_len >= 1
        if longest_accepted_len == 1:
            assert longest_match_path_idx == 0

        # Take the longest accepted path as the next new token.
        num_accepted_draft_tokens = 0
        for idx in eagle_paths[longest_match_path_idx][:longest_accepted_len]:
            step = cast(int, idx.item())
            add_token(request, new_tokens_list, beam_idx=DEFAULT_BEAM_IDX, step=step)
            num_accepted_draft_tokens += 1
            if self.finish_if_reason(
                request,
                finish_reasons,
                step=step,
                beam_idx=DEFAULT_BEAM_IDX,
            ):
                break

        assert num_accepted_draft_tokens <= longest_accepted_len

        tree_node_indices = eagle_paths[longest_match_path_idx][1:num_accepted_draft_tokens]
        request.py_num_accepted_draft_tokens_indices = (tree_node_indices - 1).tolist()

        return num_accepted_draft_tokens - 1

    @classmethod
    def _collect_new_requests_for_setup(
        cls, scheduled_requests: ScheduledRequests
    ) -> list[LlmRequest]:
        # ADP can inject generation-phase dummy requests after request activation.
        # Those still need sampler-side slot initialization before grouping/sampling.
        # The two source lists are disjoint by construction.
        context_new_requests = [
            request
            for request in scheduled_requests.context_requests_last_chunk
            if not request.is_finished and not request.py_is_draft
        ]
        adp_dummy_generation_requests = [
            request
            for request in scheduled_requests.generation_requests
            if request.is_attention_dp_dummy
        ]
        return context_new_requests + adp_dummy_generation_requests

    @override
    def validate_request(self, request: LlmRequest) -> None:
        # Reject unsupported top-p-decay combinations at admission, so only the
        # offending request fails (raising later, inside setup_sampler_step or
        # sampling, would abort the whole executor step).
        self._top_p_decay.validate_request(request)
        if self._use_beam_search:
            if _get_max_beam_width(request) > 1 and _has_occurrence_penalty(request):
                raise ValueError(
                    "TorchSampler does not support repetition, presence, or frequency "
                    "penalties with beam search."
                )
            if request.py_return_log_probs:
                if request.py_num_logprobs > 1:
                    raise ValueError(
                        "Beam search does not support returning multiple logprobs per request"
                    )
                if request.py_num_logprobs != 0:
                    raise ValueError(
                        "Beam search only supports returning the sampled logprob per token"
                    )

    @override
    @nvtx_range("setup_sampler_step")
    def setup_sampler_step(self, scheduled_requests: ScheduledRequests) -> None:
        """Setup the sampler step for the requests

        Args:
            scheduled_requests: The scheduled requests to set up the sampler step for.
        """
        new_requests = self._collect_new_requests_for_setup(scheduled_requests)

        if not new_requests:
            return

        # Used for all store updates
        seq_slots: list[int] = []
        # Used for beam search updates
        max_prompt_len: int = 0

        # Prepare finish reasons handler
        self._finish_reasons_handler.setup_new_request_handling()
        for request in new_requests:
            slot = request.py_seq_slot
            assert slot is not None
            seq_slots.append(slot)
            # update temp_data with this requests data
            self._finish_reasons_handler.prepare_for_new_request(request)

            if self._use_beam_search:
                assert not (request.py_return_log_probs and request.py_num_logprobs > 1), (
                    "Beam search does not support returning multiple logprobs per request"
                )
                max_prompt_len = max(max_prompt_len, request.py_prompt_len)
                if self._use_speculative_beam_history_d2h:
                    # Drop stale predictor state from any prior occupant of this slot.
                    self._prev_first_finish_reasons_host[slot] = None

            self._request_grouper.prepare_for_new_request(request, slot)
            self._penalty_handler.prepare_for_new_request(request, slot)

        max_lens = self._finish_reasons_handler.new_max_lens
        end_ids = self._finish_reasons_handler.new_end_ids
        # Perform updates to the stores
        full_list = [seq_slots, max_lens, end_ids]
        # perform only a single copy
        full_list_tensor_host = torch.tensor(
            full_list, device="cpu", dtype=torch.int32, pin_memory=prefer_pinned()
        )
        full_list_tensor_cuda = full_list_tensor_host.to(device="cuda", non_blocking=True)
        seq_slots_tensor_host = full_list_tensor_host[0]
        seq_slots_tensor_cuda = full_list_tensor_cuda[0]
        max_lens_tensor_cuda = full_list_tensor_cuda[1]
        end_ids_tensor_cuda = full_list_tensor_cuda[2]

        # Cast to int64 once for downstream ``index_copy_`` / ``index_fill_`` calls.
        seq_slots_tensor_cuda_long = seq_slots_tensor_cuda.long()

        self._finish_reasons_handler.update_for_new_request(
            seq_slots_cuda_long=seq_slots_tensor_cuda_long,
            max_lengths_cuda=max_lens_tensor_cuda,
            end_ids_cuda=end_ids_tensor_cuda,
            seq_slots_host=seq_slots_tensor_host,
            all_sampling_requests=new_requests + scheduled_requests.generation_requests,
        )

        self._top_p_decay.setup_for_new_requests(
            new_requests, new_seq_slots_cuda_long=seq_slots_tensor_cuda_long
        )

        self._penalty_handler.update_for_new_requests(
            new_seq_slots_cuda_long=seq_slots_tensor_cuda_long
        )

        if self._use_beam_search:
            beam_search_store = self.store.beam_search_store
            assert beam_search_store is not None
            self._prepare_beam_search(
                beam_search_store,
                self.store.log_probs_store,
                seq_slots_long=seq_slots_tensor_cuda_long,
                max_prompt_len=max_prompt_len,
            )

    @staticmethod
    def _prepare_beam_search(
        beam_search_store: BeamSearchStore,
        log_probs_store: LogProbsStore,
        seq_slots_long: torch.Tensor,
        max_prompt_len: int,
    ) -> None:
        """Prepare the beam search buffers for the requests

        If the last context chunk is being processed,
        initialize/reset the buffers for the request.

        ``seq_slots_long`` must be int64 (required by ``index_fill_``).
        """
        beam_search_store.cache_indirection.narrow(2, 0, max_prompt_len).index_fill_(
            0, seq_slots_long, 0
        )
        beam_search_store.cum_log_probs.index_fill_(0, seq_slots_long, 0)
        log_probs_store.sampled_log_probs.index_fill_(0, seq_slots_long, 0)
        log_probs_store.sampled_log_prob_ranks.index_fill_(0, seq_slots_long, 0)
        beam_search_store.predecessor_beams.index_fill_(0, seq_slots_long, 0)
        beam_search_store.first_finish_reasons.index_fill_(
            0, seq_slots_long, FinishReason.NOT_FINISHED.value
        )
        beam_search_store.original_tokens.index_fill_(0, seq_slots_long, 0)

    @torch.inference_mode()
    def _process_draft_tokens_rejection_sampling(
        self,
        request: LlmRequest,
        new_tokens_list: list[list[list[int]]],
        new_tokens_tensor: torch.Tensor,
    ) -> int:
        """We cannot use finish_if_reason in _process_draft_tokens_rejection_sampling because it *writes to new_tokens*,
        rendering the finish reason calculation in sample_async stale (incorrect) for this batch"""
        assert request.py_draft_logits is not None
        # FIXME: Passing a dummy vocab_size could result in unnecessary
        #        filtering of vocab_size logits, out of vocab_size in
        #        total. The 'sample' below should generally be avoided
        #        by retaining the draft_probs during drafting (TRTLLM-7772).
        draft_sampling_strategy = (
            ("greedy", None)
            if request.py_draft_use_greedy_sampling
            else _request_strategy(request, vocab_size=2**31)
        )
        generator = self.get_generator(request.py_draft_logits.device)
        _, draft_probs, _ = sample(
            draft_sampling_strategy,
            request.py_draft_logits,
            generator=generator,
        )
        assert draft_probs is not None
        target_probs = request.py_target_probs
        assert target_probs is not None
        d2t = getattr(request, "d2t", None)
        if d2t is not None:
            vocab_d = draft_probs.shape[-1]
            vocab_t = target_probs.shape[-1]
            assert d2t.numel() == vocab_d, f"d2t size mismatch: {d2t.numel()} != {vocab_d}"
            assert d2t.device == draft_probs.device, (
                f"d2t device mismatch: {d2t.device} != {draft_probs.device}"
            )
            aligned_draft_probs = torch.zeros(
                (*draft_probs.shape[:-1], vocab_t),
                device=draft_probs.device,
                dtype=draft_probs.dtype,
            )
            source_indices = torch.arange(vocab_d, device=draft_probs.device)
            target_indices = (source_indices + d2t) % vocab_t
            aligned_draft_probs[..., target_indices] = draft_probs
            draft_probs = aligned_draft_probs
        rejected_indices = get_rejected_indices(
            draft_probs,
            target_probs,
            generator,
            request.py_draft_tokens,
        )
        sample_last = True
        if rejected_indices.numel() == 0:
            num_initially_accepted = get_draft_token_length(request)
            sample_last = False
        else:
            num_initially_accepted = cast(int, rejected_indices[0].item())
        num_accepted = num_initially_accepted
        for i in range(num_accepted):
            new_token = request.py_draft_tokens[i]
            new_tokens_tensor[i, request.seq_slot, DEFAULT_BEAM_IDX] = new_token
            request.add_new_token(new_token, DEFAULT_BEAM_IDX)
            if self._handle_stop_criteria(
                request, new_token, beam_idx=DEFAULT_BEAM_IDX, max_seq_len=self.max_seq_len
            ):
                num_accepted = i + 1
                return num_accepted
        if sample_last:
            new_token = sample_rejected(draft_probs, target_probs, generator, num_accepted)
            new_tokens_tensor[num_accepted, request.seq_slot, DEFAULT_BEAM_IDX] = new_token
            request.add_new_token(new_token, DEFAULT_BEAM_IDX)
        else:
            new_token = add_token(
                request, new_tokens_list, beam_idx=DEFAULT_BEAM_IDX, step=num_accepted
            )
        self._handle_stop_criteria(
            request, new_token, beam_idx=DEFAULT_BEAM_IDX, max_seq_len=self.max_seq_len
        )

        return num_accepted

    @staticmethod
    def _speculation_could_use_rejection_sampling(
        request: LlmRequest, strategy: Optional[Strategy] = None
    ) -> bool:
        if strategy is None:
            strategy = _request_strategy(
                request,
                vocab_size=2**31,  # vocab_size does not affect greediness
            )
        return strategy != GREEDY and get_draft_token_length(request) > 0

    def process_draft_tokens(
        self,
        request: LlmRequest,
        new_tokens_tensor: torch.Tensor,
        new_tokens_list: list[list[list[int]]],
        finish_reasons: FinishReasonsList,
        resource_manager: Optional[ResourceManager] = None,
    ) -> int:
        if not (
            self._speculation_could_use_rejection_sampling(request)
            # NB: '_speculation_could_use_rejection_sampling' is called in sample_async, which precludes
            #     inspection of .py_draft_logits, because it is not set yet when the overlap path
            #     is used.
            #
            #     OTOH, some drafters (e.g. NGram) do not provide draft logits, precluding rejection
            #     sampling. The current solution accepts that .py_target_probs may sometimes be
            #     computed, even though .py_draft_logits may never be set and the target probs
            #     may ultimately not be required.
            and request.py_draft_logits is not None
        ):
            spec_tree_manager = self.get_spec_tree_manager(resource_manager)
            if spec_tree_manager is not None:
                num_accepted = self._process_draft_tokens_tree(
                    request,
                    new_tokens_tensor=new_tokens_tensor,
                    new_tokens_list=new_tokens_list,
                    finish_reasons=finish_reasons,
                    spec_tree_manager=spec_tree_manager,
                )
            else:
                num_accepted = self._process_draft_tokens_greedy(
                    request, new_tokens=new_tokens_list, finish_reasons=finish_reasons
                )
            return num_accepted
        else:
            return self._process_draft_tokens_rejection_sampling(
                request, new_tokens_list=new_tokens_list, new_tokens_tensor=new_tokens_tensor
            )

    def _prepare_beam_history(
        self,
        request: LlmRequest,
        *,
        finish_reasons: torch.Tensor,
        d2h_copier: Callable[[torch.Tensor], torch.Tensor],
    ) -> BeamHistoryBuilder | None:
        """Correct the stored tokens for each beam and return it as a BeamHistory object.

        Beam Search sampling only adds new tokens to the beam.
        However during beam search, a beam may change its previously sampled tokens.
        This function corrects the stored tokens for each beam to match the expected tokens.
        If logprobs are requested, the function also corrects the stored logprobs for each beam.
        The function returns a BeamHistory object that contains the corrected tokens and logprobs for each beam.

        D2H copies are issued through `d2h_copier`. When
        `_use_speculative_beam_history_d2h` is set, a host-side predictor
        decides per step whether to stage copies via `d2h_copier`;
        predictor misses fall back to a synchronous `.cpu()` inside
        `_builder`. Otherwise, copies are issued unconditionally.

        Note: To defer the decision whether or not to skip BeamHistory construction until update_requests(), only
              a builder (BeamHistoryBuilder) is returned here. The builder contains host tensors which are
              being populated asynchronously. Hence, it can only be invoked after async D2H copies have completed,
              e.g., after awaiting state.sampler_event in update_requests.

        arguments:
            request: The request to create the beam history for
            finish_reasons: The first finish reason encountered for each beam of the request.
                            Shape: (max_tokens, max_beam_width)
            d2h_copier: Callable performing the D2H copy.
        """

        # Gather data used for skipping beam history processing
        need_finalize_due_to_stop_words = self._check_stop_words_length(request)
        if need_finalize_due_to_stop_words:
            need_history = torch.tensor(True)
        else:
            should_stop = self._check_beam_search_stop_criteria(
                request,
                finish_reasons=finish_reasons,
            )
            need_history = should_stop
            # enqueue async D2H copy
            need_history = self._copy_to_host(need_history)

        num_tokens = request.max_beam_num_tokens + 1  # last token is not yet added
        prompt_length = request.py_prompt_len
        num_generated_tokens = num_tokens - prompt_length
        num_beams = request.py_beam_width

        if num_generated_tokens == 0 or request.state == LlmRequestState.GENERATION_COMPLETE:
            # early return if no tokens have been generated yet or the request is already finished
            return None

        beam_search_store = self.store.beam_search_store
        assert beam_search_store is not None

        log_probs_device: _BeamHistoryLogProbsSlices | None = None
        if request.py_return_log_probs:
            log_probs_store = self.store.log_probs_store
            log_probs_device = _BeamHistoryLogProbsSlices(
                sampled_log_probs=log_probs_store.sampled_log_probs[
                    request.py_seq_slot, :num_beams
                ].view(-1, 1),
                sampled_logprobs_indices=self.store.new_tokens[
                    0, request.py_seq_slot, :num_beams
                ].view(-1, 1),
                cum_logprobs=beam_search_store.cum_log_probs[request.py_seq_slot, :num_beams],
            )
        device_slices = _BeamHistoryTensors(
            cache_indirection=beam_search_store.cache_indirection[
                request.py_seq_slot, :num_beams, prompt_length:num_tokens
            ],
            current_path=beam_search_store.original_tokens[
                request.py_seq_slot, :num_beams, prompt_length:num_tokens
            ],
            log_probs=log_probs_device,
        )

        # In speculative mode, the predictor may skip the copy; otherwise
        # always copy. `host_snapshot is None` triggers the .cpu() fallback
        # in `_builder`, which can only happen on a predictor miss.
        issue_copy = (
            not self._use_speculative_beam_history_d2h
            or self._predict_beam_search_is_likely_finishing(
                request,
                num_generated_tokens=num_generated_tokens,
                num_tokens=num_tokens,
            )
        )

        host_snapshot: _BeamHistoryTensors | None = None
        if issue_copy:
            log_probs_host: _BeamHistoryLogProbsSlices | None = None
            if device_slices.log_probs is not None:
                log_probs_host = _BeamHistoryLogProbsSlices(
                    sampled_log_probs=d2h_copier(device_slices.log_probs.sampled_log_probs),
                    sampled_logprobs_indices=d2h_copier(
                        device_slices.log_probs.sampled_logprobs_indices
                    ),
                    cum_logprobs=d2h_copier(device_slices.log_probs.cum_logprobs),
                )
            host_snapshot = _BeamHistoryTensors(
                cache_indirection=d2h_copier(device_slices.cache_indirection),
                current_path=d2h_copier(device_slices.current_path),
                log_probs=log_probs_host,
            )

        def _builder() -> BeamHistory | None:
            if not need_history.item():
                return None

            if host_snapshot is not None:
                cache_indirection = host_snapshot.cache_indirection
                current_path = host_snapshot.current_path
                log_probs_host = host_snapshot.log_probs
            else:
                # Predictor-miss fallback: synchronous .cpu() on the main stream.
                cache_indirection = device_slices.cache_indirection.cpu()
                current_path = device_slices.current_path.cpu()
                log_probs_host = None
                if device_slices.log_probs is not None:
                    log_probs_host = _BeamHistoryLogProbsSlices(
                        sampled_log_probs=device_slices.log_probs.sampled_log_probs.cpu(),
                        sampled_logprobs_indices=(
                            device_slices.log_probs.sampled_logprobs_indices.cpu()
                        ),
                        cum_logprobs=device_slices.log_probs.cum_logprobs.cpu(),
                    )

            new_path = _gather_beam_path(
                current_path=current_path, cache_indirection=cache_indirection
            )
            new_logprobs: torch.Tensor | None = None
            new_logprobs_indices: torch.Tensor | None = None
            cum_logprobs_out: torch.Tensor | None = None
            if log_probs_host is not None:
                new_logprobs, new_logprobs_indices, cum_logprobs_out = (
                    self._postprocess_beam_logprobs(
                        request,
                        cache_indirection=cache_indirection,
                        log_probs_host=log_probs_host,
                    )
                )

            return BeamHistory(
                tokens=new_path,
                logprobs=new_logprobs,
                logprobs_indices=new_logprobs_indices,
                cum_logprobs=cum_logprobs_out,
            )

        return _builder

    def _postprocess_beam_logprobs(
        self,
        request: LlmRequest,
        *,
        cache_indirection: torch.Tensor,
        log_probs_host: _BeamHistoryLogProbsSlices,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder per-step beam logprobs along the cache-indirection axis.

        Concatenates the freshly-sampled per-step entries onto the
        request's existing host-side logprobs buffer and gathers each
        beam's history through `cache_indirection`. Returns the gathered
        (logprobs, logprobs_indices, cum_logprobs) triple.
        """
        current_logprobs, current_logprobs_indices = get_logprobs_from_request(
            request, preallocate_extra_steps=1
        )
        # concatenate the newly generated logprobs and newly
        # generated tokens to the current logprobs and logprobs indices
        current_logprobs[:, -1, :].copy_(log_probs_host.sampled_log_probs)
        current_logprobs_indices[:, -1, :].copy_(log_probs_host.sampled_logprobs_indices)

        # Gather the correct logprobs for each beam.
        new_logprobs = torch.zeros_like(current_logprobs)
        new_logprobs_indices = torch.zeros_like(current_logprobs_indices)
        cache_indirection_for_logprobs = cache_indirection.unsqueeze(-1).expand(
            -1, -1, current_logprobs.shape[2]
        )
        torch.gather(
            input=current_logprobs,
            dim=0,
            index=cache_indirection_for_logprobs,
            out=new_logprobs,
        )
        torch.gather(
            input=current_logprobs_indices,
            dim=0,
            index=cache_indirection_for_logprobs,
            out=new_logprobs_indices,
        )
        return new_logprobs, new_logprobs_indices, log_probs_host.cum_logprobs

    def _finalize_beam(
        self,
        request: LlmRequest,
        beam_history: BeamHistory,
    ) -> None:
        """Update the request with the corrected tokens and logprobs for each beam.

        Args:
            request: The request to update
            beam_history: The beam history used to update the request
        """

        beam_width = request.py_beam_width
        assert beam_history.tokens.shape[0] == beam_width, (
            f"Beam_history.tokens.shape[0] should equal beam width: \
                {beam_history.tokens.shape[0]} != {beam_width}"
        )
        if request.py_return_log_probs:
            assert beam_history.logprobs is not None
            assert beam_history.logprobs_indices is not None
            assert beam_history.cum_logprobs is not None
            assert beam_history.logprobs.shape[0] == beam_width, (
                f"Beam_history.logprobs.shape[0] should equal beam width: \
                    {beam_history.logprobs.shape[0]} != {beam_width}"
            )
            assert beam_history.logprobs_indices.shape[0] == beam_width, (
                f"Beam_history.logprobs_indices.shape[0] should equal beam width: \
                    {beam_history.logprobs_indices.shape[0]} != {beam_width}"
            )
            assert beam_history.cum_logprobs.shape[0] == beam_width, (
                f"Beam_history.cum_logprobs.shape[0] should equal beam width: \
                    {beam_history.cum_logprobs.shape[0]} != {beam_width}"
            )
        valid_tokens = (beam_history.tokens != BEAM_SEARCH_PAD_TOKEN).sum(dim=-1).tolist()
        gen_token_list = []
        gen_log_probs_list = []
        for beam_idx in range(beam_width):
            beam_valid_tokens = valid_tokens[beam_idx]
            gen_token_list.append(beam_history.tokens[beam_idx, :beam_valid_tokens].tolist())
            if request.py_return_log_probs:
                assert beam_history.logprobs_indices is not None
                assert beam_history.logprobs is not None
                gen_log_probs_list.append(
                    convert_logprobs_tensor_to_list(
                        beam_history.logprobs_indices[beam_idx : beam_idx + 1, :beam_valid_tokens],
                        beam_history.logprobs[beam_idx : beam_idx + 1, :beam_valid_tokens],
                    )[0]
                )
        request.set_generated_tokens(gen_token_list)
        if request.py_return_log_probs:
            # cum_log_probs will not change when padding with end tokens.
            # Therefore, we do not need to correct it
            assert beam_history.cum_logprobs is not None
            request.py_result.set_log_probs(
                gen_log_probs_list, cum_log_probs=beam_history.cum_logprobs.tolist()
            )

    def _add_metadata_to_grouped_requests(
        self,
        requests: list[LlmRequest],
        grouped_requests: dict[RequestGroupKey[GenericStrategyKeyType], RequestGroupValue],
        seq_slots: torch.Tensor,
        seq_lens: torch.Tensor | None,
        get_metadata_type_for_group_fn: Callable[
            [GenericStrategyKeyType], Type[StrategyMetadata] | None
        ],
        *,
        seq_slots_cuda: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        req_num_steps: torch.Tensor,
    ) -> dict[RequestGroupKey[GenericStrategyKeyType], RequestGroupValueWithMetadata]:
        grouped_requests_with_metadata: dict[
            RequestGroupKey[GenericStrategyKeyType], RequestGroupValueWithMetadata
        ] = {}
        beam_search_store = self.store.beam_search_store
        log_probs_store = self.store.log_probs_store
        num_requests = len(requests)
        for key, value in grouped_requests.items():
            metadata_type = get_metadata_type_for_group_fn(key.strategy_key)
            metadata: StrategyMetadata | None
            if metadata_type is BeamSearchMetadata:
                assert beam_search_store is not None
                assert seq_lens is not None, "seq_lens is required for beam search"
                # Reuse the precomputed CUDA tensors when the strategy group
                # covers the full batch (typical single-strategy case);
                # otherwise fall back to a per-group H2D for the subset.
                if value.indices.size(0) == num_requests:
                    group_seq_slots_cuda = seq_slots_cuda
                    group_seq_lens_cuda = seq_lens_cuda
                else:
                    group_seq_slots_cuda = seq_slots[value.indices].to(
                        device="cuda", dtype=torch.int64, non_blocking=True
                    )  # Should be on device for beam search, need long for index_copy_
                    group_seq_lens_cuda = seq_lens[value.indices].to(
                        device="cuda", non_blocking=True
                    )  # Should be on device for beam search
                metadata = BeamSearchMetadata(
                    cache_indirection=beam_search_store.cache_indirection,
                    cache_indirection_buffer=beam_search_store.cache_indirection_buffer,
                    cum_log_probs=beam_search_store.cum_log_probs,
                    new_log_probs=log_probs_store.sampled_log_probs[..., DEFAULT_STEP_IDX],
                    seq_slots=group_seq_slots_cuda,
                    seq_lens=group_seq_lens_cuda,
                    finished_beams=beam_search_store.first_finish_reasons,
                    predecessor_beams=beam_search_store.predecessor_beams,
                    seq_offsets=beam_search_store.seq_offsets,
                    beam_idx_arange=beam_search_store.beam_idx_arange,
                )
            elif metadata_type is TopPDecayMetadata:
                metadata = self._top_p_decay.build_metadata(
                    group_req_indices=value.indices,
                    req_num_steps=req_num_steps,
                    seq_slots=seq_slots,
                    seq_slots_cuda=seq_slots_cuda,
                )
            elif metadata_type is None:
                metadata = None
            else:
                raise ValueError(f"Unsupported metadata type: {metadata_type}")
            grouped_requests_with_metadata[key] = RequestGroupValueWithMetadata(
                indices=value.indices,
                strategies=value.strategies,
                speculation_needs_probs_indices=value.speculation_needs_probs_indices,
                need_processed_logprobs=value.need_processed_logprobs,
                need_raw_logprobs=value.need_raw_logprobs,
                metadata=metadata,
            )
        return grouped_requests_with_metadata

    @staticmethod
    def _check_beam_search_stop_criteria(
        request: LlmRequest,
        finish_reasons: torch.Tensor,
    ) -> torch.Tensor:
        """Check if the stop criteria is met for the request.

        Returns a boolean tensor of shape (), whose value is computed asynchronously.
        """
        return (finish_reasons[: request.py_beam_width] > 0).sum() == request.py_beam_width

    @staticmethod
    def _check_stop_words_length(request: LlmRequest) -> bool:
        """Check if the stop words length is greater than 1"""
        # TODO: cache this on the request (e.g. as `request._py_has_multi_token_stop_words`)
        # so we don't recompute it per step from `py_stop_words_list`.
        if request.py_stop_words_list is not None:
            _, cumsum = request.py_stop_words_list
            if -1 in cumsum:
                cumsum = cumsum[: cumsum.index(-1)]
            cumsum_arr = np.asarray(cumsum, dtype=np.int32)
            longest_stop_word_len = cast(
                int, np.max(np.diff(cumsum_arr, prepend=0), initial=0).item()
            )
            return longest_stop_word_len > 1
        return False

    def _predict_beam_search_is_likely_finishing(
        self,
        request: LlmRequest,
        *,
        num_generated_tokens: int,
        num_tokens: int,
    ) -> bool:
        """Predict whether this step is likely to trigger beam history finalization.

        Returns True if any of:
          1. Length budget reached (max_new_tokens or max_seq_len).
          2. Multi-token stop_words configured (forces finalization).
          3. Lagged first_finish_reasons shows any beam finished previously.

        Known miss: all beams hit end_id on the same step from a clean state.
        """
        if num_generated_tokens >= request.py_max_new_tokens or num_tokens >= self.max_seq_len:
            return True
        if self._check_stop_words_length(request):
            return True
        assert request.py_seq_slot is not None
        prev = self._prev_first_finish_reasons_host[request.py_seq_slot]
        # FinishReason.NOT_FINISHED == 0, so a nonzero entry implies that
        # some beam has already finished.
        if prev is not None and prev.any().item():
            return True
        return False

    @nvtx_range("maybe_create_beam_histories")
    def _prepare_beam_histories(
        self,
        requests: list[LlmRequest],
        finish_reasons: torch.Tensor,
    ) -> tuple[list[BeamHistoryBuilder | None], torch.cuda.Event | None]:
        """Create the corrected tokens and logprobs for each beam of a request.

        The builders returned by this function create a beam history object containing
        the corrected tokens and logprobs for each beam of a request.

        Returns (builders, side_stream_event). side_stream_event is set
        only when the speculative path queued copies; the caller must
        forward it to _record_sampler_event so SamplerEvent.synchronize
        awaits the side stream before any builder is invoked.
        """
        # Single `with` for both modes; nullcontext yields None.
        copier_ctx: AbstractContextManager[_SideStreamCopier | None] = (
            self._make_side_stream_copier()
            if self._use_speculative_beam_history_d2h
            else nullcontext()
        )
        with copier_ctx as copier:
            d2h_copier: Callable[[torch.Tensor], torch.Tensor] = (
                copier.stage_copy_to_host if copier is not None else self._copy_to_host
            )
            builders = [
                self._prepare_beam_history(
                    req,
                    finish_reasons=finish_reasons[req.py_seq_slot],
                    d2h_copier=d2h_copier,
                )
                for req in requests
            ]
        side_stream_event = copier.event if copier is not None else None
        return builders, side_stream_event

    @override
    @nvtx_range("update_requests")
    @torch.inference_mode()
    def update_requests(
        self,
        state: SampleStateTorch,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        if state.sampler_event:
            state.sampler_event.synchronize()

        if not state.requests:
            return

        if self._track_pending_steps and not self._is_draft_batch(state.requests):
            for req in state.requests:
                slot = req.py_seq_slot
                if slot is not None and self._pending_steps[slot] > 0:
                    self._pending_steps[slot] -= 1

        assert state.host is not None
        new_tokens = state.host.new_tokens
        finish_reasons = state.host.finish_reasons_list()
        first_finish_reasons = (
            state.host.first_finish_reasons.tolist()
            if state.host.first_finish_reasons is not None
            else []
        )

        new_tokens_list = new_tokens.tolist()

        logprobs_state_list: LogProbsStateList | None = None
        if state.host.logprobs_state is not None:
            logprobs_state_list = LogProbsStateList.from_logprobs_state(state.host.logprobs_state)

        beam_history_builders = state.beam_history_builders
        assert (beam_history_builders is not None) == self._use_beam_search

        def _maybe_build_beam_history(req_idx: int) -> BeamHistory | None:
            if (
                beam_history_builders is not None
                and (beam_history_builder := beam_history_builders[req_idx]) is not None
            ):
                return beam_history_builder()
            else:
                return None

        finalized_token_updates: list[tuple[int, list[int]]] = []
        # Fast-path (batched pybind): when the batch is greedy with no beam
        # search, no logprobs, no draft tokens, no stop-words, and no
        # speculative tree, collapse per-request pybind chatter into one
        # batched add_new_tokens_to_requests call. Single-pass eligibility
        # check with early-break; falls through when any invariant breaks.
        if (
            self._batch_fastpath_eligible
            and logprobs_state_list is None
            and self.get_spec_tree_manager(resource_manager) is None
        ):
            alive_reqs: list[LlmRequest] = []
            tokens_flat: list[int] = []
            fastpath_ok = True
            new_tokens_step0 = new_tokens_list[0]
            for req in state.requests:
                if req.state == LlmRequestState.GENERATION_COMPLETE:
                    continue
                if get_draft_token_length(req) != 0 or req.py_stop_words_list:
                    fastpath_ok = False
                    break
                assert req.py_seq_slot is not None
                alive_reqs.append(req)
                tokens_flat.append(new_tokens_step0[req.py_seq_slot][DEFAULT_BEAM_IDX])
            if fastpath_ok and alive_reqs:
                add_new_tokens_to_requests(alive_reqs, tokens_flat, DEFAULT_BEAM_IDX)
                _valid_finish_reasons = {
                    FinishReason.END_ID,
                    FinishReason.LENGTH,
                    FinishReason.STOP_WORDS,
                }
                for req in alive_reqs:
                    assert req.py_seq_slot is not None
                    reason_val = finish_reasons[req.py_seq_slot][0][DEFAULT_BEAM_IDX]
                    if reason_val != 0:
                        reason = FinishReason(reason_val)
                        if reason in _valid_finish_reasons:
                            req.finish_by(reason, DEFAULT_BEAM_IDX)
                    req.py_num_accepted_draft_tokens = 0
                    req.py_rewind_len = 0
                    req.py_decoding_iter += 1
                return

        for req_idx, req in enumerate(state.requests):
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                self._top_p_decay.retire_slot(req)
                continue

            if req.py_beam_width > 1:
                if (beam_history := _maybe_build_beam_history(req_idx)) is not None:
                    self._finalize_beam(req, beam_history)
                else:
                    for beam_idx in range(req.py_beam_width):
                        # Beam search does not support speculative decoding.
                        add_token(req, new_tokens_list, beam_idx=beam_idx)
                    self.handle_logprobs(req, logprobs_state_list=logprobs_state_list, count=1)
                first_finish_reasons_host = state.host.first_finish_reasons
                assert first_finish_reasons_host is not None
                self._handle_first_finish_reasons(
                    req, first_finish_reasons_host, first_finish_reasons
                )
                if self._use_speculative_beam_history_d2h:
                    # Snapshot for the next step's predictor.
                    assert req.py_seq_slot is not None
                    self._prev_first_finish_reasons_host[req.py_seq_slot] = (
                        first_finish_reasons_host[req.py_seq_slot]
                    )
                if req.is_context_only_request:
                    beam_search_store = self.store.beam_search_store
                    assert beam_search_store is not None
                    assert req.py_seq_slot is not None
                    beam_width = req.py_beam_width
                    first_gen_scores = (
                        beam_search_store.cum_log_probs[req.py_seq_slot, :beam_width]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    first_gen_tokens = [
                        new_tokens_list[0][req.py_seq_slot][beam_idx]
                        for beam_idx in range(beam_width)
                    ]
                    first_gen_log_probs = [
                        {token_id: Logprob(logprob=log_prob, rank=None)}
                        for token_id, log_prob in zip(first_gen_tokens, first_gen_scores)
                    ]
                    req.py_result.set_first_gen_log_probs(first_gen_log_probs)
                req.py_num_accepted_draft_tokens = 0
                req.py_rewind_len = 0
            else:
                processed = 1
                num_tokens_before = req.get_num_tokens(DEFAULT_BEAM_IDX)
                num_accepted = self.process_draft_tokens(
                    req,
                    new_tokens_tensor=new_tokens,
                    new_tokens_list=new_tokens_list,
                    finish_reasons=finish_reasons,
                    resource_manager=resource_manager,
                )
                if (actual_draft_len := get_draft_token_length(req)) > 0:
                    req.py_num_accepted_draft_tokens = num_accepted
                    req.py_rewind_len = actual_draft_len - num_accepted
                else:
                    req.py_num_accepted_draft_tokens = 0
                    req.py_rewind_len = 0
                processed += num_accepted
                if actual_draft_len > 0:
                    num_new_tokens = req.get_num_tokens(DEFAULT_BEAM_IDX) - num_tokens_before
                    if num_new_tokens > 0:
                        assert req.py_seq_slot is not None
                        confirmed_tokens = req.get_tokens(DEFAULT_BEAM_IDX)[-num_new_tokens:]
                        finalized_token_updates.append((req.py_seq_slot, confirmed_tokens))
                self.handle_logprobs(req, logprobs_state_list=logprobs_state_list, count=processed)
            req.py_decoding_iter += 1
            # Check None or empty list
            if req.py_stop_words_list:
                self._finish_reasons_handler.store.num_accepted_draft_tokens_host[
                    req.py_seq_slot
                ] = req.py_num_accepted_draft_tokens
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                self._top_p_decay.retire_slot(req)

        self._penalty_handler.update_token_counts(finalized_token_updates)

    def _return_log_probs(self, requests: list[LlmRequest]) -> bool:
        return any(req.py_return_log_probs for req in requests)

    def _prepare_log_probs(self, requests: list[LlmRequest]) -> None:
        self.batch_max_topk_logprobs = max(
            (req.py_num_logprobs or 0 for req in requests),
            default=0,
        )
        check_logprobs_limit("batch_max_logprobs", self.batch_max_topk_logprobs, MAX_TOP_LOGPROBS)
        if self.max_topk_logprobs < self.batch_max_topk_logprobs:
            self.max_topk_logprobs = self.batch_max_topk_logprobs
            self.TOPK_LOGPROBS_SHAPE = (
                self.max_num_sequences,
                self.max_tokens,
                self.max_topk_logprobs,
            )
            log_probs_store = self.store.log_probs_store
            log_probs_store.topk_vals.resize_(self.TOPK_LOGPROBS_SHAPE)
            log_probs_store.topk_indices.resize_(self.TOPK_LOGPROBS_SHAPE)

    @override
    @torch.inference_mode()
    @nvtx_range("sample_async")
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleStateTorch:
        # NB: The sampler is either called directly by PyExecutor, for the target model,
        #     or by ModelDrafter.prepare_draft_tokens(), for the draft model. In the former
        #     case there are 1 + get_draft_token_length(request) tokens per request. In the
        #     latter case, there is always only 1 token per request because draft
        #     tokens are sampled one-by-one.
        self.setup_sampler_step(scheduled_requests)
        new_tokens = self.store.new_tokens

        if self._track_pending_steps:
            # A context request claims a (possibly reused) slot: clear any
            # counter leaked by a prior occupant that never got its final
            # update_requests. Must happen before _process_requests, which
            # reads the counters for bad-words staleness.
            for r in scheduled_requests.context_requests:
                if not r.py_is_draft:
                    assert r.py_seq_slot is not None
                    self._pending_steps[r.py_seq_slot] = 0

        # seq_slots_cuda / seq_lens_cuda are cast once inside
        # _process_requests and shared with the beam-search metadata builder.
        (
            requests,
            seq_slots_host,
            seq_lens_host,
            seq_slots_cuda,
            seq_lens_cuda,
            new_tokens_host,
        ) = self._process_requests(
            scheduled_requests,
            model_outputs,
            new_tokens,
            num_context_logits_prefix_sum,
        )

        if self._track_pending_steps and requests and not self._is_draft_batch(requests):
            for r in requests:
                assert r.py_seq_slot is not None
                self._pending_steps[r.py_seq_slot] += 1

        finish_reasons_host: torch.Tensor | None = None
        first_finish_reasons_host: torch.Tensor | None = None
        beam_history_builders: list[BeamHistoryBuilder | None] | None = None
        # Forwarded to _record_sampler_event so SamplerEvent.synchronize
        # awaits any side-stream D2H copies host-side.
        side_stream_event: torch.cuda.Event | None = None
        if requests:
            beam_search_store = self.store.beam_search_store
            assert self._use_beam_search == (beam_search_store is not None)
            # Prepare stop word handling
            # Draft requests need to be ignored for stop word handling as they never set up
            # their buffers in the store.
            # Assume that either all requests are drafts or none are drafts
            is_draft_batch = requests[0].py_is_draft
            finish_reasons_device = self._finish_reasons_handler.write_finish_reasons(
                seq_slots_host=seq_slots_host,
                is_draft_batch=is_draft_batch,
                seq_slots_cuda=seq_slots_cuda,
                seq_lens_cuda=seq_lens_cuda,
                new_tokens_cuda=new_tokens,
                first_finish_reasons_cuda=(
                    beam_search_store.first_finish_reasons
                    if beam_search_store is not None
                    else None
                ),
            )
            finish_reasons_host = self._copy_to_host(finish_reasons_device)

            if self._use_beam_search:
                assert beam_search_store is not None
                first_finish_reasons = beam_search_store.first_finish_reasons
                first_finish_reasons_host = self._copy_to_host(first_finish_reasons)
                self._update_original_tokens(
                    beam_search_store.original_tokens, seq_slots_cuda, seq_lens_cuda, new_tokens
                )
                beam_history_builders, side_stream_event = self._prepare_beam_histories(
                    requests, finish_reasons=first_finish_reasons
                )

        # copy logprobs to host
        logprobs_state: LogProbsState | None = None
        if self._return_log_probs(requests):
            log_probs_store = self.store.log_probs_store
            host_topk_vals = self._copy_to_host(
                log_probs_store.topk_vals[..., : self.batch_max_topk_logprobs]
            )
            host_topk_indices = self._copy_to_host(
                log_probs_store.topk_indices[..., : self.batch_max_topk_logprobs]
            )
            host_sampled_vals = self._copy_to_host(log_probs_store.sampled_log_probs)
            host_sampled_indices = self._copy_to_host(log_probs_store.sampled_log_prob_indices)
            host_sampled_rank = self._copy_to_host(log_probs_store.sampled_log_prob_ranks)
            logprobs_state = LogProbsState(
                topk_vals=host_topk_vals,
                topk_indices=host_topk_indices,
                sampled_vals=host_sampled_vals,
                sampled_indices=host_sampled_indices,
                sampled_rank=host_sampled_rank,
            )

        sampler_event = self._record_sampler_event(side_stream_event=side_stream_event)
        return SampleStateTorch(
            requests=requests,
            device=SampleStateTensors(new_tokens=new_tokens),
            host=SampleStateTensorsHostTorch(
                new_tokens=new_tokens_host,
                finish_reasons=finish_reasons_host,
                first_finish_reasons=first_finish_reasons_host,
                logprobs_state=logprobs_state,
            ),
            sampler_event=sampler_event,
            beam_history_builders=beam_history_builders,
        )

    @staticmethod
    def _apply_d2t(tokens: torch.Tensor, model_outputs: dict[str, Any]) -> None:
        """Applies draft-to-target token translation table.

        Modifies tokens in-place.
        """
        if "d2t" in model_outputs:
            d2t = model_outputs["d2t"][tokens]
            tokens += d2t

    @staticmethod
    @nvtx_range("fast_greedy_sample_kernel")
    def _fast_greedy_sample_kernel(
        logits_cuda: torch.Tensor,
        new_tokens_cuda: torch.Tensor,
        batch_dest_indices: torch.Tensor,
        max_beam_width: int,
        d2t: torch.Tensor | None,
    ) -> None:
        """Applies fast greedy sampling to the logits.

        Performs argmax, applies d2t translation if present, and scatters
        tokens into the output buffer. All operations are in-place.
        """
        # Simple argmax for greedy sampling
        next_tokens = torch.argmax(logits_cuda, dim=-1).to(dtype=new_tokens_cuda.dtype)

        # Apply draft-to-target token translation if present (for Eagle3)
        if d2t is not None:
            next_tokens += d2t[next_tokens]

        # Scatter tokens into output buffer
        batch_dest_indices_expanded = batch_dest_indices.unsqueeze(1).expand(-1, max_beam_width)
        next_tokens_expanded = next_tokens.unsqueeze(1).expand(-1, max_beam_width)
        new_tokens_cuda.view(-1, *new_tokens_cuda.shape[2:]).scatter_(
            0, batch_dest_indices_expanded, next_tokens_expanded
        )

    @staticmethod
    def _apply_embedding_bias(
        logits: torch.Tensor,
        requests: list[LlmRequest],
        request_steps: torch.Tensor,
    ) -> None:
        """Apply embedding bias (aka logit bias) to logits.

        Arguments:
          request_steps: Number of steps/tokens for each request.

        Modifies logits in-place.
        """
        # NB: Unfortunately, Torch provides no combination of torch.index_select (similar to
        #     torch.Tensor.gather -- allows one-to-many mapping) and addition, analogous to how
        #     torch.Tensor.scatter_add_ (and its variant torch.Tensor.index_add_ -- allows
        #     many-to-one mapping) combine addition with torch.Tensor.scatter_.
        #
        #     Notwithstanding the previous point, there are two options:
        #         (i)  materialize a permuted bias tensor with repeated consecutive rows via
        #              torch.repeat_interleave and then use torch.Tensor.index_add_ (poor write
        #              locality / risk of false sharing)
        #        (ii)  materialize the correctly ordered bias tensor via torch.index_select and then
        #              perform a masked addition (poor read locality for request batches randomly
        #              mixing uniform and heterogeneous bias tensors, i.e., mixing slices with high
        #              and low reuse).
        #     Since read-caching is expected to help in typical cases, option (ii) is implemented here.

        # Track which logits require logit bias application
        request_steps_list = request_steps.tolist()
        logits_bias_masks = [False] * logits.size(0)
        _next_bias_index = 0

        def provision_bias_index() -> int:
            nonlocal _next_bias_index
            bias_index = _next_bias_index
            _next_bias_index += 1
            return bias_index

        # Indices of unique bias tensors
        #
        # NB: hash(torch.Tensor) is equivalent to id(torch.Tensor), and does not
        #     depend on tensor contents, cf. https://github.com/pytorch/pytorch/issues/2569
        bias_to_index: dict[torch.Tensor, int] = defaultdict(provision_bias_index)

        # Source indices for bias application
        bias_gather_indices: list[int] = []

        # Collect bias information
        req_bias = None
        for i, (req, steps) in enumerate(zip(requests, request_steps_list)):
            req_bias = req._py_embedding_bias_1d
            if req_bias is not None:
                for j in range(i, i + steps):
                    logits_bias_masks[j] = True
                req_bias_index = bias_to_index[req_bias]
                bias_gather_indices.extend(repeat(req_bias_index, steps))

        if not bias_to_index:
            return
        assert req_bias is not None  # otherwise bias_to_index is empty

        bias_gather_indices_cuda = torch.tensor(
            bias_gather_indices, pin_memory=prefer_pinned(), dtype=torch.int32
        ).to(logits.device, non_blocking=True)
        logits_bias_mask_cuda = torch.tensor(
            logits_bias_masks, pin_memory=prefer_pinned(), dtype=torch.bool
        ).to(logits.device, non_blocking=True)
        biases_tensor = torch.empty(
            (len(bias_to_index), *req_bias.shape), pin_memory=prefer_pinned()
        )
        biases_tensor = torch.stack(
            tuple(bias_to_index.keys()),
            out=biases_tensor,
        )
        biases_tensor_cuda = biases_tensor.to(logits.device, non_blocking=True)

        biases_tensor_cuda = torch.index_select(biases_tensor_cuda, 0, bias_gather_indices_cuda)
        # NB: Avoiding logits[bias_scatter_indices] += biases_tensor (and torch.Tensor.scatter_add_), because it
        #     is unclear if this allows for repeated indices, cf.
        #         https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.index_put_.html#torch-tensor-index-put
        #     and thus introduces read-after-write dependencies (including possible false
        #     sharing).
        logits[logits_bias_mask_cuda] += biases_tensor_cuda

    @nvtx_range("sample_batched_by_strategy")
    @torch.inference_mode()
    def _sample_batched_by_strategy(
        self,
        logits_cuda: torch.Tensor,
        requests: list[LlmRequest],
        model_outputs: dict[str, Any],
        *,
        logits_cuda_indexer: _PackedStepIndexer,
        req_num_generated_tokens: torch.Tensor,
        req_num_steps: torch.Tensor,
        req_offsets: torch.Tensor,
        seq_slots: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        seq_slots_cuda: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        token_dtype: torch.dtype,
        return_log_probs: bool,
    ) -> _BatchedSamplingResult:
        cuda_device = logits_cuda.device

        grouped_requests = self._request_grouper.group_requests_by_strategy_key(
            requests,
            pin_memory=prefer_pinned(),
            strategy_to_key=self._grouped_sampler_cls.strategy_grouping_key,
            seq_slots=seq_slots,
            vocab_size=logits_cuda.size(1),  # Dummy value; strategy should already be cached
        )
        grouped_requests_with_metadata = self._add_metadata_to_grouped_requests(
            requests,
            grouped_requests,
            seq_slots,
            seq_lens,
            get_metadata_type_for_group_fn=self._grouped_sampler_cls.get_metadata_type_for_group,
            seq_slots_cuda=seq_slots_cuda,
            seq_lens_cuda=seq_lens_cuda,
            req_num_steps=req_num_steps,
        )
        generator_cuda = self.get_generator(cuda_device)

        # NB: Currently, "d2t" is applied to draft tokens, but not to draft logits,
        #     breaking _process_draft_tokens_rejection_sampling.
        needs_d2t = "d2t" in model_outputs
        if needs_d2t and (
            len(grouped_requests_with_metadata) > 1
            or (
                grouped_requests_with_metadata
                and next(iter(grouped_requests_with_metadata.values())).strategies[0] != GREEDY
            )
        ):
            raise ValueError("d2t does not yet support non-greedy sampling")

        # Tensors for collecting sampling results (in batch ordering)
        batch_req_indices = torch.empty((len(requests),), dtype=torch.int32)
        batch_next_tokens_cuda_int = torch.empty(
            (logits_cuda.size(0), self.max_beam_width), device=cuda_device, dtype=token_dtype
        )
        batch_logits_for_logprobs_cuda = (
            torch.empty(
                (logits_cuda.size(0), logits_cuda.size(1)), device=cuda_device, dtype=torch.float32
            )
            if return_log_probs
            else None
        )
        batch_req_idx_offset_start = 0
        batch_next_tokens_offset_start = 0
        for group_key, group_val_with_metadata in grouped_requests_with_metadata.items():
            strategy_key = group_key.strategy_key
            needs_probs = group_key.needs_probs
            group_req_indices = group_val_with_metadata.indices
            group_strategies = group_val_with_metadata.strategies
            group_speculation_needs_probs_indices = (
                group_val_with_metadata.speculation_needs_probs_indices
            )
            group_need_processed_logprobs = group_val_with_metadata.need_processed_logprobs
            group_need_raw_logprobs = group_val_with_metadata.need_raw_logprobs
            group_metadata = group_val_with_metadata.metadata

            # group_req_indices: Indices of 'requests' entries having the same sampling
            # strategy, ordered ascending.
            batch_req_idx_offset_end = batch_req_idx_offset_start + group_req_indices.size(0)
            batch_req_indices[batch_req_idx_offset_start:batch_req_idx_offset_end] = (
                group_req_indices
            )

            need_processed_logprobs_indices = torch.nonzero(group_need_processed_logprobs)
            need_raw_logprobs_indices = torch.nonzero(group_need_raw_logprobs)
            any_request_needs_processed_logprobs = need_processed_logprobs_indices.size(0) > 0
            any_request_needs_raw_logprobs = need_raw_logprobs_indices.size(0) > 0
            any_request_needs_logprobs = (
                any_request_needs_processed_logprobs or any_request_needs_raw_logprobs
            )

            if any_request_needs_logprobs:
                # indices for accessing logits within the current group
                group_logit_indexer = _PackedStepIndexer(
                    num_steps=req_num_generated_tokens[group_req_indices],
                    max_steps=cast(
                        int, req_num_generated_tokens.max().item() * self.max_beam_width
                    ),
                )
                logit_indices_for_processed_logprobs_cuda = group_logit_indexer[
                    need_processed_logprobs_indices
                ].to(logits_cuda.device, non_blocking=True)
                logit_indices_for_raw_logprobs_cuda = group_logit_indexer[
                    need_raw_logprobs_indices
                ].to(logits_cuda.device, non_blocking=True)
            else:
                logit_indices_for_processed_logprobs_cuda = None
                logit_indices_for_raw_logprobs_cuda = None

            group_logits_cuda_indices = logits_cuda_indexer[group_req_indices]
            # NB: Assuming that group_req_indices are sorted
            group_req_1st_index, group_req_last_index = group_req_indices[0], group_req_indices[-1]
            group_logits_cuda_indices_cuda: torch.Tensor | slice
            logit_indices_for_sampler: Optional[torch.Tensor]
            if group_req_last_index - group_req_1st_index + 1 == len(group_req_indices):
                # Avoid data movement if indices are contiguous
                group_logits_cuda_indices_cuda = slice(
                    req_offsets[group_req_1st_index],
                    req_offsets[group_req_last_index]
                    + req_num_generated_tokens[group_req_last_index],
                )
                group_logits_cuda = logits_cuda[group_logits_cuda_indices_cuda]
                logit_indices_for_sampler = None
                # group_logits_cuda already contains only logits for the group
                group_logits_indices_for_processed_logprobs_cuda = (
                    logit_indices_for_processed_logprobs_cuda
                )
                group_logits_indices_for_raw_logprobs_cuda = logit_indices_for_raw_logprobs_cuda
            else:
                group_logits_cuda_indices_cuda = group_logits_cuda_indices.to(
                    device=logits_cuda.device, non_blocking=True
                )
                group_logits_cuda = logits_cuda
                logit_indices_for_sampler = group_logits_cuda_indices_cuda
                # group_logits_cuda contains logits for the whole batch
                # Therefore, we need indices corresponding to the whole batch
                group_logits_indices_for_processed_logprobs_cuda = (
                    None
                    if not any_request_needs_processed_logprobs
                    else logits_cuda_indexer[group_req_indices[group_need_processed_logprobs]].to(
                        logits_cuda.device, non_blocking=True
                    )
                )
                group_logits_indices_for_raw_logprobs_cuda = (
                    None
                    if not any_request_needs_raw_logprobs
                    else logits_cuda_indexer[group_req_indices[group_need_raw_logprobs]].to(
                        logits_cuda.device, non_blocking=True
                    )
                )

            group_strategies_per_step = [  # convert from per-request to per-step
                strat
                for strat, steps in zip(group_strategies, req_num_steps[group_req_indices].tolist())
                for _ in range(steps)
            ]

            group_next_tokens_cuda, group_softmax_cuda, group_temperature_cuda = (
                self._grouped_sampler_cls.sample_grouped_strategies(
                    strategy_key,
                    group_strategies_per_step,
                    group_logits_cuda,
                    generator=generator_cuda,
                    return_probs=needs_probs,
                    group_logit_indices=logit_indices_for_sampler,
                    group_metadata=group_metadata,
                )
            )
            batch_next_tokens_offset_end = (
                batch_next_tokens_offset_start + group_next_tokens_cuda.size(0)
            )
            # if no beam search is used, the shape is (batch_size,), so we need to unsqueeze it to (batch_size, 1)
            if group_next_tokens_cuda.dim() == 1:
                group_next_tokens_cuda = group_next_tokens_cuda.unsqueeze(1)
            batch_next_tokens_cuda_int[
                batch_next_tokens_offset_start:batch_next_tokens_offset_end
            ].copy_(group_next_tokens_cuda, non_blocking=True)

            if any_request_needs_processed_logprobs:
                assert group_logits_indices_for_processed_logprobs_cuda is not None
                assert logit_indices_for_processed_logprobs_cuda is not None
                assert group_softmax_cuda is not None
                assert batch_logits_for_logprobs_cuda is not None
                # NB: The logits copy could be avoided by instead counting (and storing):
                #        -  the number of unmasked tokens 'nu'
                #        -  r := log(max(probs)) - max(logits)
                #   Later, processed logprobs can be reconstructed from raw logits _after_ applying
                #   top-k: Add 'r' and mask smallest entries so that only min(k, nu) tokens remain.
                current_logits_cuda = group_logits_cuda[
                    group_logits_indices_for_processed_logprobs_cuda
                ]
                current_softmax_cuda = group_softmax_cuda[logit_indices_for_processed_logprobs_cuda]
                # processed_logits_cuda is an alias to current_logits_cuda after this operation
                processed_logits_cuda = current_logits_cuda.masked_fill_(
                    current_softmax_cuda == 0, float("-inf")
                )
                temperature_for_processed_logprobs = group_temperature_cuda
                if isinstance(temperature_for_processed_logprobs, torch.Tensor):
                    temperature_for_processed_logprobs = cast(torch.Tensor, group_temperature_cuda)[
                        logit_indices_for_processed_logprobs_cuda
                    ].unsqueeze(-1)
                if temperature_for_processed_logprobs is not None:
                    processed_logits_cuda /= temperature_for_processed_logprobs
                logit_indices_for_processed_logprobs_cuda += batch_next_tokens_offset_start
                batch_logits_for_logprobs_cuda[logit_indices_for_processed_logprobs_cuda] = (
                    processed_logits_cuda
                )

            if any_request_needs_raw_logprobs:
                assert group_logits_indices_for_raw_logprobs_cuda is not None
                assert logit_indices_for_raw_logprobs_cuda is not None
                assert batch_logits_for_logprobs_cuda is not None
                if (
                    group_logits_indices_for_raw_logprobs_cuda
                    is logit_indices_for_raw_logprobs_cuda
                ):
                    group_logits_indices_for_raw_logprobs_cuda = (
                        group_logits_indices_for_raw_logprobs_cuda.clone()
                    )
                logit_indices_for_raw_logprobs_cuda += batch_next_tokens_offset_start
                # NB: Copy could be avoided by storing logit indices (and temperature) instead (cf. comment on
                #     processed logprobs above).
                Fusions.gather_scatter(
                    batch_logits_for_logprobs_cuda,
                    logit_indices_for_raw_logprobs_cuda,
                    group_logits_cuda,
                    group_logits_indices_for_raw_logprobs_cuda,
                )

            # Set LlmRequest.py_target_probs
            if group_speculation_needs_probs_indices.size(0) > 0:
                assert group_softmax_cuda is not None
                current_offset = 0
                for req_idx, steps in zip(
                    group_speculation_needs_probs_indices.tolist(),
                    req_num_steps[group_speculation_needs_probs_indices].tolist(),
                ):
                    next_offset = current_offset + steps
                    # using view avoids copy
                    requests[req_idx].py_target_probs = group_softmax_cuda[
                        current_offset:next_offset
                    ]
                    current_offset = next_offset

            batch_next_tokens_offset_start = batch_next_tokens_offset_end
            batch_req_idx_offset_start = batch_req_idx_offset_end

        # NB: 'd2t' contains offsets for transforming draft vocab token IDs into
        #     the target vocab. This is used by Eagle3ForCausalLM, whose input domain
        #     is the target vocab, whereas the output logits correspond to the draft
        #     vocab. Since the inputs/outputs are linked by TorchSampler.update_requests,
        #     they currently need to be handled within TorchSampler.
        if needs_d2t:
            self._apply_d2t(batch_next_tokens_cuda_int, model_outputs)

        return _BatchedSamplingResult(
            batch_req_indices=batch_req_indices,
            batch_next_tokens_cuda_int=batch_next_tokens_cuda_int,
            batch_logits_for_logprobs_cuda=batch_logits_for_logprobs_cuda,
        )

    def _unbatch_sampling_results(
        self,
        batched_sampling_result: _BatchedSamplingResult,
        *,
        new_tokens_cuda: torch.Tensor,
        req_num_generated_tokens: torch.Tensor,
        seq_slots: torch.Tensor,
        seq_slots_cuda: torch.Tensor,
    ) -> torch.Tensor:
        batch_req_indices = batched_sampling_result.batch_req_indices
        batch_next_tokens_cuda_int = batched_sampling_result.batch_next_tokens_cuda_int

        def _dims_canonically_ordered(t: torch.Tensor) -> bool:
            return len(t.dim_order(ambiguity_check=[torch.contiguous_format])) == t.ndim

        # Assert destination tensor dimensions are canonically ordered ("row"-major); this
        # matters for element ordering in the .view(...).scatter_(...) calls below.
        assert _dims_canonically_ordered(new_tokens_cuda)

        # Construct index mapping from slice indices of computed tensors
        # (packed request_idx and step dimensions) to linearized indices
        # in (steps, seq_slot).
        batch_destination_cuda_indexer = _UnpackedStepIndexer(
            seq_slots=seq_slots[batch_req_indices],
            num_steps=req_num_generated_tokens[batch_req_indices],
            steps_dim_size=new_tokens_cuda.size(0),
            slots_dim_size=new_tokens_cuda.size(1),
            dim_order=_UnpackedStepIndexer.DimOrder.STEP_MAJOR,
            index_dtype=torch.int64,  # enforced by Tensor.scatter_
        )

        # Batch update output tensors
        batch_dest_indices_1d_cuda = (
            batch_destination_cuda_indexer[:]
            .to(new_tokens_cuda.device, non_blocking=True)
            .unsqueeze(1)
            .expand(-1, self.max_beam_width)
        )
        new_tokens_cuda.view(-1, *new_tokens_cuda.shape[2:]).scatter_(
            0, batch_dest_indices_1d_cuda, batch_next_tokens_cuda_int
        )
        # Post-sample: decay the runtime top-p for any decay-active slots that were
        # sampled this iteration (must run after tokens land in new_tokens_cuda).
        # batch_req_indices is a permutation of all sampled requests, so the set of
        # sampled slots is exactly seq_slots (the kernel updates each slot
        # independently; order is irrelevant) -- pass the resident device copy
        # instead of gathering seq_slots[batch_req_indices] on host and copying it.
        self._top_p_decay.update_after_sample(
            step_tokens=new_tokens_cuda[DEFAULT_STEP_IDX, :, DEFAULT_BEAM_IDX],
            sampled_slots_cuda=seq_slots_cuda,
        )
        return self._copy_to_host(new_tokens_cuda)

    def _compute_pending_steps(self, requests: list[LlmRequest]) -> list[int] | None:
        """Per-request count of tokens sampled but not yet written back, or None.

        With the overlap scheduler ``sample_async`` for step ``i`` runs before
        ``update_requests`` for step ``i - 1``, so the host token list — and
        hence ``get_num_tokens()`` — lags the true sequence by this many tokens.
        Length-based bans (min_length) add it back to recover the real generated
        length. Unlike the suffix-rule staleness this needs no device-side
        lookup, so it is not restricted to the single-step / single-beam case.
        Returns None when the overlap scheduler is off, on a draft batch, or
        when nothing is pending, so callers can skip the correction entirely.
        """
        if not self._track_pending_steps or self._is_draft_batch(requests):
            return None
        pending = [
            self._pending_steps[r.py_seq_slot] if r.py_seq_slot is not None else 0 for r in requests
        ]
        return pending if any(pending) else None

    def _compute_stale_by_one(self, requests: list[LlmRequest]) -> list[bool] | None:
        """Per-request overlap-scheduler stale flags, or None when not applicable.

        Returns a list where entry ``i`` is True when request ``i``'s host token
        history lags the device state by exactly one token (the previous step's
        token was sampled but not yet written back). Only the single-step,
        single-beam overlap case is reconstructible on the device side; under
        speculative decoding or beam search the missing history cannot be
        recovered, so bans are matched against the lagging host history and may
        be enforced one step late (warned once). Returns None when the overlap
        scheduler is off, on a draft batch, or when nothing is pending.
        """
        if not self._track_pending_steps or self._is_draft_batch(requests):
            return None
        pending = [
            self._pending_steps[r.py_seq_slot] if r.py_seq_slot is not None else 0 for r in requests
        ]
        if not any(pending):
            return None
        if self.max_tokens == 1 and self.max_beam_width == 1 and max(pending) == 1:
            return [p > 0 for p in pending]
        logger.warning_once(
            "bad_words / no_repeat_ngram_size with the overlap scheduler and "
            "speculative decoding or beam search: bans are matched against a "
            "host token history that lags the device state and may be enforced "
            "inexactly.",
            key="bad_words_stale_overlap",
        )
        return None

    @staticmethod
    def _select_generated_logits(
        scheduled_requests: ScheduledRequests,
        raw_logits_cuda: torch.Tensor,
        *,
        num_context_logits_prefix_sum: list[int],
    ) -> tuple[list[LlmRequest], SamplingRequestsMetadata, torch.Tensor]:
        """Select the sampling requests and the corresponding logits from the raw logits.

        Args:
            scheduled_requests: The scheduled requests. Sampling requests will be selected from this list.
            raw_logits_cuda: The raw logits corresponding to the scheduled requests.
            num_context_logits_prefix_sum: The prefix sum of the number of logits for each context request.

        Returns:
            A tuple containing the following:
            - sampling requests: The requests that are selected for sampling.
            - sampling requests metadata: The metadata for the sampling requests.
            - logits: The logits for the sampling requests.
        """
        finished_context_requests = scheduled_requests.context_requests_last_chunk
        sampling_requests = finished_context_requests + scheduled_requests.generation_requests

        req_num_generation_steps_list = [
            1 + get_draft_token_length(req) for req in sampling_requests
        ]
        req_num_generation_steps = torch.tensor(
            req_num_generation_steps_list, dtype=torch.int32, pin_memory=prefer_pinned()
        )

        # context requests do not have multiple beams yet, so beam width may differ in mixed batches
        req_num_beams_list = [1] * len(finished_context_requests) + [
            req.get_beam_width_by_iter(False) for req in scheduled_requests.generation_requests
        ]
        req_num_beams = torch.tensor(
            req_num_beams_list, dtype=torch.int32, pin_memory=prefer_pinned()
        )
        # context requests do not have multiple beams yet, so beam width may differ after sampling
        req_num_output_beams_list = [req.get_beam_width_by_iter(True) for req in sampling_requests]
        req_num_beams_output = torch.tensor(
            req_num_output_beams_list, dtype=torch.int32, pin_memory=prefer_pinned()
        )

        req_num_generated_tokens = req_num_generation_steps * req_num_beams
        req_num_generated_tokens_output = req_num_generation_steps * req_num_beams_output
        # NB: These offsets consider generated tokens _only_ (draft and target, but not context).
        #     Filter out the context tokens below.
        req_offsets, sum_num_generated_tokens = _PackedStepIndexer.calculate_request_offsets(
            req_num_generated_tokens, pin_memory=prefer_pinned()
        )

        generation_requests_total_steps = (
            # NB: requests == finished_context_requests + scheduled_requests.generation_requests
            sum_num_generated_tokens - cast(int, req_offsets[len(finished_context_requests)].item())
            if scheduled_requests.generation_requests
            else 0
        )

        sampling_requests_metadata = SamplingRequestsMetadata(
            req_num_generated_tokens=req_num_generated_tokens,
            req_num_generated_tokens_output=req_num_generated_tokens_output,
            req_num_beams=req_num_beams,
            req_num_steps=req_num_generation_steps,
            req_offsets=req_offsets,
        )

        # logits_cuda should contain only the generated logits for the sampling requests.
        # If return context logits is requested, select only the generated logits of the context requests.
        #
        # NB: context_requests_chunking precede finished_context_requests, which precede generation requests.
        #     context_requests_chunking do not sample new tokens, so they should be skipped.
        #     sampling_requests == finished_context_requests + scheduled_requests.generation_requests
        num_skipped_requests = len(scheduled_requests.context_requests_chunking)
        if any(r.py_return_context_logits for r in finished_context_requests):
            assert len(num_context_logits_prefix_sum) == scheduled_requests.num_context_requests + 1
            logits_end_offsets = num_context_logits_prefix_sum[num_skipped_requests + 1 :]

            if scheduled_requests.generation_requests:
                # Since logits for generation requests are densely packed, add them all as one contiguous block
                logits_end_offsets.append(
                    num_context_logits_prefix_sum[-1] + generation_requests_total_steps
                )
                num_logits = req_num_generated_tokens[: len(finished_context_requests) + 1].clone()
                num_logits[-1] = generation_requests_total_steps
            else:
                num_logits = req_num_generated_tokens[: len(finished_context_requests)]

            logits_end_offsets_cuda = torch.tensor(
                logits_end_offsets, dtype=torch.int32, pin_memory=prefer_pinned()
            ).to(device=raw_logits_cuda.device, non_blocking=True)
            num_logits_cuda = num_logits.to(raw_logits_cuda.device, non_blocking=True)

            num_logits_to_keep = sum_num_generated_tokens

            # Now, the generated tokens for context request i are at indices
            #    range(logits_end_offsets_cuda[i] - num_logits_cuda[i],
            #          logits_end_offsets_cuda[i])
            # And if generation requests are present, those tensors each include a trailing entry selecting
            # all tokens/logits generated by all generation requests.
            indices_to_keep_cuda = torch_multi_arange(
                starts=(logits_end_offsets_cuda - num_logits_cuda),
                ends=logits_end_offsets_cuda,
                output_length=num_logits_to_keep,
            )

            logits_cuda = raw_logits_cuda[indices_to_keep_cuda]
        else:
            logits_begin_offset = num_context_logits_prefix_sum[num_skipped_requests]
            logits_cuda = raw_logits_cuda[logits_begin_offset:]

        return sampling_requests, sampling_requests_metadata, logits_cuda

    @nvtx_range("_process_logprobs")
    def _process_logprobs(
        self,
        batched_sampling_result: _BatchedSamplingResult,
        seq_slots: torch.Tensor,
        requests: list[LlmRequest],
        req_num_steps: torch.Tensor,
        req_num_generated_tokens: torch.Tensor,
    ) -> None:
        assert batched_sampling_result.batch_logits_for_logprobs_cuda is not None, (
            "batch_logits_for_logprobs_cuda must be a Tensor for _process_logprobs"
        )

        all_req_indices: list[int] = batched_sampling_result.batch_req_indices.tolist()

        local_group_req_indices_list: list[int] = []
        max_num_logprobs_no_beam_search = 0

        local_group_req_indices_with_beam_search_list: list[int] = []

        for req_id, req_gid in enumerate(all_req_indices):
            req = requests[req_gid]
            num_logprobs = req.py_num_logprobs
            if num_logprobs is None:
                continue
            if req.py_beam_width == 1:
                local_group_req_indices_list.append(req_id)
                max_num_logprobs_no_beam_search = max(max_num_logprobs_no_beam_search, num_logprobs)
            else:
                local_group_req_indices_with_beam_search_list.append(req_id)

        # Index the positions of each token in the padded 2d tensors
        # NB: Using all_req_indices to allow reuse for beam search requests
        padded_indexer = _PackedStepIndexer(
            num_steps=req_num_generated_tokens[batched_sampling_result.batch_req_indices],
            max_steps=cast(int, req_num_generated_tokens.max().item()),
            req_offsets=seq_slots[batched_sampling_result.batch_req_indices]
            * self.max_tokens
            * self.max_beam_width,  # NB: Currently either max_tokens or max_beam_width is 1
        )
        # indexer for shuffled logits after grouping
        logits_cuda_indexer = _PackedStepIndexer(
            num_steps=req_num_steps[batched_sampling_result.batch_req_indices],
            max_steps=cast(int, req_num_steps.max().item()),
        )

        log_probs_store = self.store.log_probs_store
        sampled_log_prob_indices = log_probs_store.sampled_log_prob_indices
        sampled_log_prob_ranks = log_probs_store.sampled_log_prob_ranks

        if local_group_req_indices_list:
            # The request indices in the shuffled batch after grouping (NB: Beam search request are handled separately)
            local_group_req_indices = torch.tensor(local_group_req_indices_list, dtype=torch.int32)
            sampled_log_probs = log_probs_store.sampled_log_probs
            # NB: Already begin copy here, to overlap with the remaining host code
            padded_indices_cuda = padded_indexer[local_group_req_indices].to(
                device=sampled_log_probs.device, non_blocking=True
            )

            # get indices of the logits after grouping
            group_logits_indices_cuda = logits_cuda_indexer[local_group_req_indices].to(
                device=batched_sampling_result.batch_logits_for_logprobs_cuda.device,
                non_blocking=True,
            )

            # (batch_size, vocab_size)
            group_logprobs_cuda = Fusions.gather_log_softmax(
                batched_sampling_result.batch_logits_for_logprobs_cuda, group_logits_indices_cuda
            )

            # Process the topk logprobs
            if self.batch_max_topk_logprobs > 0:
                # Get the topk logprobs
                topk_vals_cuda, topk_indices_cuda = torch.topk(
                    group_logprobs_cuda,
                    k=max_num_logprobs_no_beam_search,
                    dim=-1,
                )
                expanded_indices_cuda = padded_indices_cuda.view(-1, 1).expand(
                    -1, topk_vals_cuda.shape[-1]
                )
                log_probs_store.topk_vals[..., : self.batch_max_topk_logprobs].view(
                    self.max_num_sequences * self.max_tokens, self.batch_max_topk_logprobs
                ).scatter_(dim=0, index=expanded_indices_cuda, src=topk_vals_cuda)
                log_probs_store.topk_indices[..., : self.batch_max_topk_logprobs].view(
                    self.max_num_sequences * self.max_tokens, self.batch_max_topk_logprobs
                ).scatter_(
                    dim=0, index=expanded_indices_cuda, src=topk_indices_cuda.to(torch.int32)
                )

            # Process the sampled logprobs
            # (batch_size, max_beam_width)
            group_next_tokens_cuda = batched_sampling_result.batch_next_tokens_cuda_int[
                group_logits_indices_cuda
            ][:, :1]
            # Get the sampled logprobs
            sampled_vals_cuda = torch.gather(
                group_logprobs_cuda, dim=-1, index=group_next_tokens_cuda.view(-1, 1)
            )
            # Get the sampled logprobs indices
            sampled_indices_cuda = group_next_tokens_cuda.squeeze(1)

            # sampled_rank_cuda contains the 0-based rank, it will be corrected to 1-based in handle_logprobs
            # NB: Computation of sampled rank could be lowered into FlashInferGroupedStrategySampler, s.t., e.g., for
            #     greedy sampling, logits management and log_softmax could be completely skipped (sampled rank
            #     computation is trivial in this case).
            sampled_rank_cuda = Fusions.determine_sampled_rank(
                group_logprobs_cuda, sampled_vals_cuda
            )

            sampled_vals_cuda = sampled_vals_cuda.squeeze(1)

            sampled_log_prob_indices.view(
                self.max_num_sequences * self.max_tokens * self.max_beam_width
            ).scatter_(dim=0, index=padded_indices_cuda, src=sampled_indices_cuda)
            sampled_log_probs.view(
                self.max_num_sequences * self.max_tokens * self.max_beam_width
            ).scatter_(dim=0, index=padded_indices_cuda, src=sampled_vals_cuda)
            sampled_log_prob_ranks.view(
                self.max_num_sequences * self.max_tokens * self.max_beam_width
            ).scatter_(dim=0, index=padded_indices_cuda, src=sampled_rank_cuda)

        if local_group_req_indices_with_beam_search_list:
            local_group_req_indices_with_beam_search = torch.tensor(
                local_group_req_indices_with_beam_search_list, dtype=torch.int32
            )
            group_logits_indices_with_beam_search = logits_cuda_indexer[
                local_group_req_indices_with_beam_search
            ]
            group_logits_indices_with_beam_search_cuda = group_logits_indices_with_beam_search.to(
                device=batched_sampling_result.batch_next_tokens_cuda_int.device,
                non_blocking=True,
            )
            group_next_tokens_with_beam_search_cuda = (
                batched_sampling_result.batch_next_tokens_cuda_int[
                    group_logits_indices_with_beam_search_cuda
                ].view(-1)
            )
            padded_indices_with_beam_search_cuda = padded_indexer[
                local_group_req_indices_with_beam_search
            ].to(device=sampled_log_prob_indices.device, non_blocking=True)
            sampled_log_prob_indices.view(-1).scatter_(
                dim=0,
                index=padded_indices_with_beam_search_cuda,
                src=group_next_tokens_with_beam_search_cuda,
            )

    @nvtx_range("_process_requests")
    def _process_requests(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        new_tokens_cuda: torch.Tensor,
        num_context_logits_prefix_sum: list[int],
    ) -> tuple[
        list[LlmRequest], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        raw_logits_cuda = model_outputs["logits"]

        sampling_requests, sampling_requests_metadata, logits_cuda = self._select_generated_logits(
            scheduled_requests,
            raw_logits_cuda,
            num_context_logits_prefix_sum=num_context_logits_prefix_sum,
        )
        return_log_probs = self._return_log_probs(sampling_requests)
        if return_log_probs:
            self._prepare_log_probs(sampling_requests)

        seq_slots_host = torch.tensor(
            [r.py_seq_slot for r in sampling_requests],
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )

        # necessary for beam search and max_length checks
        seq_lens_host = torch.tensor(
            [r.max_beam_num_tokens for r in sampling_requests],
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )

        # Cast seq_slots / seq_lens to CUDA exactly once; consumed by both
        # the per-group beam-search metadata builder and the finish-reasons
        # handler in sample_async. int64 is required for the index_*_ ops
        # downstream.
        seq_slots_cuda = seq_slots_host.to(device="cuda", dtype=torch.int64, non_blocking=True)
        seq_lens_cuda = seq_lens_host.to(device="cuda", non_blocking=True)

        # Handle embedding bias
        self._apply_embedding_bias(
            logits_cuda, sampling_requests, sampling_requests_metadata.req_num_steps
        )

        # Apply repetition/presence/frequency penalties in place, before the greedy fast
        # path, so both greedy and grouped-sampling logits are penalized.
        self._penalty_handler.apply(
            logits_cuda,
            sampling_requests,
            new_tokens=new_tokens_cuda,
            seq_slots=seq_slots_cuda,
            request_offsets=sampling_requests_metadata.req_offsets,
            request_num_steps=sampling_requests_metadata.req_num_steps,
            # _is_draft_batch reads requests[0]; an empty batch has no penalties to apply
            # anyway, so short-circuit rather than index into it.
            is_draft_batch=bool(sampling_requests) and self._is_draft_batch(sampling_requests),
        )

        has_min_length = any(getattr(r, "py_min_length", None) for r in sampling_requests)
        has_bad_words = any(getattr(r, "py_bad_words", None) for r in sampling_requests)
        # Normalized in executor_request_to_llm_request: a positive int, or
        # None when the restriction is disabled for the request.
        ngram_sizes = [getattr(r, "py_no_repeat_ngram_size", None) for r in sampling_requests]
        has_no_repeat_ngram = any(size is not None for size in ngram_sizes)
        if has_min_length or has_bad_words or has_no_repeat_ngram:
            # Overlap-scheduler stale flags (per request): True when the host
            # token history lags the device by one token. Only the overlap
            # handler consumes them; computed here as it needs sampler state.
            # Only the suffix-matching bans (bad words, no-repeat ngram) care:
            # min_length bans EOS from a length count, never from token values,
            # so a lagging history cannot mismatch it. Skipping the call for
            # min-length-only batches also avoids emitting the stale-history
            # warning, which names features such a batch does not use.
            stale_by_one = (
                self._compute_stale_by_one(sampling_requests)
                if (has_bad_words or has_no_repeat_ngram)
                else None
            )
            # min_length compares against get_num_tokens(), which counts only
            # the host history; add back the tokens still pending write-back so
            # the generated length is exact under the overlap scheduler.
            pending_steps = (
                self._compute_pending_steps(sampling_requests) if has_min_length else None
            )
            bans = self._token_ban_handler.generate_ban_list(
                sampling_requests,
                sampling_requests_metadata.req_num_steps.tolist(),
                sampling_requests_metadata.req_num_beams.tolist(),
                ngram_sizes,
                stale_by_one=stale_by_one,
                pending_steps=pending_steps,
            )
            self._token_ban_handler.apply_ban_list(
                logits_cuda, bans, new_tokens_cuda=new_tokens_cuda
            )

        # Fast path for greedy sampling
        if self._can_use_fast_greedy_path(sampling_requests):
            # Compute destination indices on CPU (same pattern as _unbatch_sampling_results)
            batch_destination_indexer = _UnpackedStepIndexer(
                seq_slots=seq_slots_host,
                num_steps=sampling_requests_metadata.req_num_generated_tokens,
                steps_dim_size=new_tokens_cuda.size(0),
                slots_dim_size=new_tokens_cuda.size(1),
                dim_order=_UnpackedStepIndexer.DimOrder.STEP_MAJOR,
                index_dtype=torch.int64,
            )
            batch_dest_indices_cuda = batch_destination_indexer[:].to(
                new_tokens_cuda.device, non_blocking=True
            )

            # Get d2t tensor if present
            d2t = model_outputs.get("d2t", None)

            # Run compiled kernel for argmax, d2t application, and scatter
            self._fast_greedy_sample_kernel(
                logits_cuda,
                new_tokens_cuda,
                batch_dest_indices_cuda,
                self.max_beam_width,
                d2t,
            )

            new_tokens_host = self._copy_to_host(new_tokens_cuda)
            return (
                sampling_requests,
                seq_slots_host,
                seq_lens_host,
                seq_slots_cuda,
                seq_lens_cuda,
                new_tokens_host,
            )

        # Indexer for accessing tokens in 'logits_cuda', corresponding to the
        # requests in 'requests'.
        steps_dim_size = new_tokens_cuda.size(0)
        logits_cuda_indexer = _PackedStepIndexer(
            num_steps=sampling_requests_metadata.req_num_generated_tokens,
            max_steps=steps_dim_size * self.max_beam_width,
            req_offsets=sampling_requests_metadata.req_offsets,
        )

        # Perform sampling in batches
        batched_sampling_result = self._sample_batched_by_strategy(
            logits_cuda,
            sampling_requests,
            model_outputs,
            logits_cuda_indexer=logits_cuda_indexer,
            req_offsets=sampling_requests_metadata.req_offsets,
            seq_slots=seq_slots_host,
            seq_lens=seq_lens_host,
            seq_slots_cuda=seq_slots_cuda,
            seq_lens_cuda=seq_lens_cuda,
            req_num_generated_tokens=sampling_requests_metadata.req_num_generated_tokens,
            req_num_steps=sampling_requests_metadata.req_num_steps,
            token_dtype=new_tokens_cuda.dtype,
            return_log_probs=return_log_probs,
        )

        if return_log_probs:
            self._process_logprobs(
                batched_sampling_result,
                seq_slots_host,
                sampling_requests,
                sampling_requests_metadata.req_num_steps,
                sampling_requests_metadata.req_num_generated_tokens_output,
            )

        # Fill results into output buffers
        new_tokens_host = self._unbatch_sampling_results(
            batched_sampling_result,
            new_tokens_cuda=new_tokens_cuda,
            req_num_generated_tokens=sampling_requests_metadata.req_num_generated_tokens,
            seq_slots=seq_slots_host,
            seq_slots_cuda=seq_slots_cuda,
        )

        # NB: update_requests syncs w/ device computation and async D2H copies
        return (
            sampling_requests,
            seq_slots_host,
            seq_lens_host,
            seq_slots_cuda,
            seq_lens_cuda,
            new_tokens_host,
        )

    @override
    def should_provide_draft_probs(self, request: LlmRequest) -> bool:
        params = _request_get_sampling_params(request)
        temperature = params.temperature
        top_p = params.top_p
        top_k = params.top_k

        # Do not request draft probs when sampling is greedy.
        return not SamplingParams.params_imply_greedy_decoding(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_beam_search=self._use_beam_search,
        )


class Algorithms:
    def defined_algorithms(self) -> list[str]:
        return [attr for attr in dir(self) if not attr.startswith("__")]

    def __repr__(self) -> str:
        algs = self.defined_algorithms()
        return f"Algs({', '.join(algs)})"


@dataclass(kw_only=True)
class SampleStateTensorsHostTRTLLM(SampleStateTensors):
    finished_sum: torch.Tensor
    finish_reasons: torch.Tensor
    sequence_lengths: torch.Tensor
    cum_log_probs: torch.Tensor | None = None
    gathered_ids: torch.Tensor | None = None


@dataclass(kw_only=True)
class SampleStateTRTLLM(SampleState[SampleStateTensorsHostTRTLLM, SampleStateTensors]):
    finalize_events: dict[int, CudaEvent] | None = None
    """`Optional` to accommodate `_forward_step_inter_pp` which creates a `SampleState` without `finalize_events`"""


class TRTLLMSampler(Sampler[SampleStateTRTLLM], AsyncWorkerMixin):
    MAX_DECODING_TOKENS = 1  # It must be 1 when not in speculative decoding
    SampleState = SampleStateTRTLLM

    @override
    def is_generation_model(self) -> bool:
        return True

    def __init__(
        self,
        model: "DecoderModelForCausalLM[_ModelType, _ConfigType]",
        model_dtype: torch.dtype,
        mapping: Mapping,
        decoding_mode: DecodingMode,
        disable_overlap_scheduler: bool,
        max_seq_len: int,
        max_batch_size: int,
        max_beam_width: int,
        decoding_config: Optional[DecodingConfig] = None,
        kv_cache_config: Optional[KvCacheConfig] = None,
        enable_async_worker: bool = False,
        max_num_sequences: Optional[int] = None,
    ):
        assert model.config is not None
        vocab_size = model.config.vocab_size
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads

        self.model_datatype = torch_dtype_to_binding(model_dtype)
        self.logits_datatype = DataType.FLOAT
        self.decoding_mode = decoding_mode
        self.decoding_config = decoding_config if decoding_config else DecodingConfig(decoding_mode)
        max_attn_window = kv_cache_config.max_attention_window  # type: ignore
        self.max_seq_len = max_seq_len
        self.max_attention_window = (
            max(max_attn_window) if max_attn_window is not None else max_seq_len
        )
        self.max_batch_size = max_batch_size
        self.max_beam_width = max_beam_width
        self.max_seq_idle_microseconds = 180 * 1000 * 1000
        self.is_trt_overlap = not disable_overlap_scheduler
        self.num_micro_batches = (
            mapping.pp_size if mapping.pp_size > 1 else (2 if self.is_trt_overlap else 1)
        )
        # Decoder state is indexed by sequence slot and must match the
        # executor's SeqSlotManager. The fallback preserves the established
        # sizing for direct callers outside the PyExecutor creator.
        self.max_num_sequences = (
            max_num_sequences if max_num_sequences is not None else mapping.pp_size * max_batch_size
        )
        self.micro_batch_idx = 0

        if mpi_disabled():
            self.world_config = WorldConfig(
                mapping.tp_size,
                mapping.pp_size,
                mapping.cp_size,
                rank=mapping.rank,
                gpus_per_node=mapping.gpus_per_node,
            )
        else:
            self.world_config = WorldConfig.mpi(
                mapping.gpus_per_node, mapping.tp_size, mapping.pp_size
            )
        self.model_config = ModelConfig(
            vocab_size,
            num_hidden_layers,
            num_hidden_layers,
            0,
            num_heads,
            hidden_size,
            self.model_datatype,
        )

        self._initialize_store()
        self._instantiate_algorithms()

        self._async_worker_init(enable_async_worker)

    def _initialize_store(self) -> None:
        torch_stream = torch.cuda.current_stream().cuda_stream
        cuda_stream = CudaStream(torch_stream)
        buffer_manager = BufferManager(stream=torch_stream)

        self.store = {
            "torch_stream": torch_stream,
            "cuda_stream": cuda_stream,
            "buffer_manager": buffer_manager,
            "decoder_input_buffers": [
                DecoderInputBuffers(self.max_batch_size, self.MAX_DECODING_TOKENS, buffer_manager)
                for _ in range(self.num_micro_batches)
            ],
            "sequence_lengths_host": torch.empty(
                (
                    self.max_num_sequences,
                    self.max_beam_width,
                ),
                dtype=torch.int,
            ),
            "decoder_state": DecoderState(),
        }

        cast(DecoderState, self.store["decoder_state"]).setup(
            max_num_sequences=self.max_num_sequences,
            max_beam_width=self.max_beam_width,
            max_attention_window=self.max_attention_window,
            sink_token_length=0,
            max_sequence_length=self.max_seq_len,
            dtype=self.logits_datatype,
            model_config=self.model_config,
            world_config=self.world_config,
            buffer_manager=buffer_manager,
        )

    def _instantiate_algorithms(self) -> None:
        self.algs = Algorithms()
        self.algs.decoder = GptDecoderBatched(stream=self.store["torch_stream"])  # type: ignore
        self.algs.decoder.setup(  # type: ignore
            mode=self.decoding_mode,
            max_num_sequences=self.max_num_sequences,
            max_beam_width=self.max_beam_width,
            dtype=self.logits_datatype,
            model_config=self.model_config,
            world_config=self.world_config,
        )
        self.algs.create_new_decoder_requests = CreateNewDecoderRequests(  # type: ignore
            speculative_decoding_fast_logits=False,
            is_leader_in_orch_mode=False,
            is_normalize_log_probs=False,
        )
        self.algs.make_decoding_batch_input_output = MakeDecodingBatchInputOutput()  # type: ignore

    @torch.inference_mode()
    @nvtx_range("setup_sampler_step")
    def setup_sampler_step(self, scheduled_requests: ScheduledRequests) -> None:
        batch_slots, sampling_configs, lookahead_prompt, lookahead_algo_configs = (
            self.algs.create_new_decoder_requests(  # type: ignore
                self.model_config,
                self.world_config,
                self.decoding_config,
                scheduled_requests.context_requests,
                self.logits_datatype,
                self.store["decoder_input_buffers"][self.micro_batch_idx],  # type: ignore
                self.store["decoder_state"],
                self.store["cuda_stream"],
                self.algs.decoder.decoder_stream,  # type: ignore
                self.max_seq_len,
                self.beam_width(scheduled_requests.context_requests),
            )
        )

        local_batch_size = len(batch_slots)
        if local_batch_size > 0:
            sampling_config = make_sampling_config(sampling_configs)
            self.algs.decoder.underlying_decoder().setup(  # type: ignore
                sampling_config,
                local_batch_size,
                batch_slots,
                self.store["decoder_state"].joint_decoding_output,  # type: ignore
                self.model_config.data_type,
                lookahead_prompt,
                lookahead_algo_configs,
            )

        adp = [r for r in scheduled_requests.generation_requests if r.is_attention_dp_dummy]
        batch_size = len(adp)
        if batch_size == 0:
            return
        config = make_sampling_config(cast(SamplingConfigVector, [r.sampling_config for r in adp]))
        slots = torch.tensor([r.py_seq_slot for r in adp], dtype=torch.int32)
        self.algs.decoder.underlying_decoder().setup(config, batch_size, slots)  # type: ignore

    def get_cache_indirection(self) -> torch.Tensor | None:
        return self.store["decoder_state"].cache_indirection_output  # type: ignore

    def _update_cache_indirection_buffer(self, scheduled_requests: ScheduledRequests) -> None:
        # Copy cache indirection output to input
        for request in scheduled_requests.generation_requests:
            self.store["decoder_state"].cache_indirection_input[request.py_seq_slot].copy_(  # type: ignore
                self.store["decoder_state"].cache_indirection_output[request.py_seq_slot],  # type: ignore
                non_blocking=True,
            )

    @override
    def validate_request(self, request: LlmRequest) -> None:
        if (
            self.max_batch_size > 1
            and self.beam_width([request]) > 1
            and request.py_return_log_probs
        ):
            raise ValueError("Beam search only supports logprobs when batch size is 1")

    @torch.inference_mode()
    @nvtx_range("sample_async")
    @override
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleStateTRTLLM:
        batch_size = scheduled_requests.batch_size
        beam_width = self.beam_width(scheduled_requests.all_requests())
        assert not (
            batch_size > 1
            and beam_width > 1
            and any(request.py_return_log_probs for request in scheduled_requests.all_requests())
        ), "Beam search only supports logprobs when batch size is 1"

        self.setup_sampler_step(scheduled_requests)

        # For beam search, cache indirection needs to be updated
        if beam_width > 1:
            self._update_cache_indirection_buffer(scheduled_requests)

        decoder_input_buffers = self.store["decoder_input_buffers"][self.micro_batch_idx]  # type: ignore
        decoder_state = self.store["decoder_state"]

        make_decoding_batch_input(
            decoder_input_buffers,
            decoder_state,
            scheduled_requests.context_requests,
            scheduled_requests.generation_requests,
            model_outputs["logits"],
            beam_width,
            num_context_logits_prefix_sum,
            self.store["buffer_manager"],
        )

        self.algs.decoder.forward_async(  # type: ignore
            decoder_state,
            self.store["decoder_input_buffers"][self.micro_batch_idx],  # type: ignore
        )

        sampling_requests = (
            scheduled_requests.context_requests_last_chunk + scheduled_requests.generation_requests
        )

        finalize_events = {}
        gathered_ids = None
        if beam_width > 1:
            finished_sum_device = decoder_state.finished_sum  # type: ignore[attr-defined]

            for request in sampling_requests:
                if request.is_context_init_state:
                    continue
                if finished_sum_device[request.seq_slot] == beam_width:
                    finalize_events[request.request_id] = self._finalize_request(request, False)
                elif request.streaming:
                    finalize_events[request.request_id] = self._finalize_request(request, True)
            gathered_ids = self._copy_to_host(decoder_state.gathered_ids)  # type: ignore[attr-defined]
        new_output_tokens = self._copy_to_host(decoder_state.all_new_tokens)  # type: ignore[attr-defined]
        finished_sum = self._copy_to_host(decoder_state.finished_sum)  # type: ignore[attr-defined]
        finish_reasons = self._copy_to_host(decoder_state.finish_reasons)  # type: ignore[attr-defined]
        sequence_lengths = self._copy_to_host(decoder_state.sequence_lengths)  # type: ignore[attr-defined]

        log_probs = None
        cum_log_probs = None
        if any(request.py_return_log_probs for request in sampling_requests):
            log_probs = self._copy_to_host(decoder_state.log_probs)  # type: ignore[attr-defined]
            cum_log_probs = self._copy_to_host(decoder_state.cum_log_probs)  # type: ignore[attr-defined]

        device = SampleStateTensors(new_tokens=decoder_state.all_new_tokens)  # type: ignore[attr-defined]

        host = SampleStateTensorsHostTRTLLM(
            new_tokens=new_output_tokens,
            finished_sum=finished_sum,
            finish_reasons=finish_reasons,
            sequence_lengths=sequence_lengths,
            log_probs=log_probs,
            cum_log_probs=cum_log_probs,
            gathered_ids=gathered_ids,
        )

        sampler_event = self._record_sampler_event()

        self.micro_batch_idx = (self.micro_batch_idx + 1) % self.num_micro_batches

        return SampleStateTRTLLM(
            requests=sampling_requests,
            device=device,
            host=host,
            sampler_event=sampler_event,
            finalize_events=finalize_events,
        )

    @torch.inference_mode()
    @override
    def update_requests(
        self,
        state: SampleStateTRTLLM,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        # resource_manager will not be used in this function, just for interface consistency.
        assert isinstance(state, SampleStateTRTLLM)

        if state.sampler_event:
            state.sampler_event.synchronize()

        if not state.requests:
            return

        beam_width = self.beam_width(state.requests)

        if beam_width == 1 and self.MAX_DECODING_TOKENS == 1:
            self.update_requests_single_beam_single_step(state)
        else:
            self.update_requests_multiple_beams_or_drafting(state, beam_width)

    @torch.inference_mode()
    @nvtx_range("update_requests_single_beam_single_step")
    def update_requests_single_beam_single_step(self, state: SampleStateTRTLLM) -> None:
        """Specialization of update_requests for single beam and single step"""
        assert state.host is not None
        sequence_lengths_host_data = state.host.sequence_lengths.flatten().tolist()
        finish_reasons = state.host.finish_reasons.flatten().tolist()

        reqs = [r for r in state.requests if not r.is_generation_complete_state]

        # NB: To ensure good performance, we must
        #  1. Avoid accessing torch.Tensor object inside the for-each-request loops
        #  2. Convert only necessary data to Python list

        # Add new tokens
        reqs_with_new_tokens = []
        seq_slots = []
        seq_slots_need_log_probs = []
        for request in reqs:
            assert request.py_seq_slot is not None
            if sequence_lengths_host_data[request.py_seq_slot] <= request.get_num_tokens(0):
                continue

            reqs_with_new_tokens.append(request)
            seq_slots.append(request.py_seq_slot)

            if request.py_return_log_probs:
                seq_slots_need_log_probs.append(request.py_seq_slot)

        # [maxTokensPerStep, batchSize, maxBeamWidth]
        new_tokens = state.host.new_tokens[0, seq_slots, 0].tolist()
        add_new_tokens_to_requests(reqs_with_new_tokens, new_tokens, 0)

        # Log probs
        assert state.host is not None
        if state.host.log_probs is not None:
            # [batchSize, maxBeamWidth]
            seq_last_idx = state.host.sequence_lengths[seq_slots_need_log_probs, 0] - 1
            # [batchSize, maxBeamWidth, maxSequenceLength]
            log_probs_host = state.host.log_probs[
                seq_slots_need_log_probs, 0, seq_last_idx
            ].tolist()
            # [batchSize, maxBeamWidth]
            assert state.host.cum_log_probs is not None
            cum_log_probs_host = state.host.cum_log_probs[seq_slots_need_log_probs, 0].tolist()

            log_probs_idx = 0
            for request, new_token in zip(reqs_with_new_tokens, new_tokens):
                if request.py_return_log_probs:
                    log_probs = [
                        {
                            new_token: Logprob(
                                logprob=log_probs_host[log_probs_idx],
                                rank=1,
                            )
                        }
                    ]
                    cum_log_probs = [cum_log_probs_host[log_probs_idx]]
                    request.py_result.append_log_probs([log_probs], cum_log_probs)
                    log_probs_idx += 1

        for request in reqs:
            request.py_decoding_iter += 1
            assert request.py_seq_slot is not None
            finished_state = FinishedState(finish_reasons[request.py_seq_slot])
            if finished_state.is_finished:
                request.state = LlmRequestState.GENERATION_COMPLETE
                finish_reason = finished_state.to_finish_reason()
                request.set_finished_reason(finish_reason, 0)

    @torch.inference_mode()
    @nvtx_range("update_requests_multiple_beams_or_drafting")
    def update_requests_multiple_beams_or_drafting(
        self,
        state: SampleStateTRTLLM,
        beam_width: int,
    ) -> None:
        assert state.host is not None
        new_tokens_host = state.host.new_tokens.tolist()
        finished_sum_host = state.host.finished_sum.tolist()
        finish_reasons = state.host.finish_reasons.flatten().tolist()
        sequence_lengths_host_data = state.host.sequence_lengths.flatten().tolist()
        cum_log_probs_host = (
            state.host.cum_log_probs.tolist() if state.host.cum_log_probs is not None else None
        )
        log_probs_host = state.host.log_probs.tolist() if state.host.log_probs is not None else None
        finalize_events = state.finalize_events

        reqs = [r for r in state.requests if not r.is_generation_complete_state]

        for request in reqs:
            seq_slot = request.py_seq_slot
            assert seq_slot is not None
            num_generated_tokens = request.num_draft_tokens + 1
            current_num_of_tokens = request.max_beam_num_tokens
            num_new_tokens = [0] * beam_width

            log_probs: list[list[dict[int, Logprob]]] = [[] for _ in range(beam_width)]
            cum_log_probs = []

            for beam_idx in range(beam_width):
                seq_len = sequence_lengths_host_data[seq_slot * beam_width + beam_idx]
                num_new_tokens[beam_idx] = min(
                    num_generated_tokens, seq_len - request.get_num_tokens(beam_idx)
                )

                for step in range(num_new_tokens[beam_idx]):
                    new_token = add_token(request, new_tokens_host, beam_idx=beam_idx, step=step)

                    if request.py_return_log_probs:
                        assert state.host.log_probs is not None
                        assert log_probs_host is not None
                        # NOTE: Log probs with drafting has not been tested yet.
                        begin_log_probs_offset = (
                            request.prompt_len if request.py_beam_width == 1 else 0
                        )
                        current_token = (
                            seq_len - request.prompt_len - num_new_tokens[beam_idx] + step
                        )
                        log_probs[beam_idx].append(
                            {
                                new_token: Logprob(
                                    logprob=log_probs_host[seq_slot][beam_idx][
                                        begin_log_probs_offset + current_token
                                    ],
                                    rank=1,
                                )
                            }
                        )

                if request.py_return_log_probs:
                    assert cum_log_probs_host is not None
                    cum_log_probs.append(cum_log_probs_host[seq_slot][beam_idx])

                finished_state = FinishedState(finish_reasons[seq_slot * beam_width + beam_idx])
                if finished_state.is_finished:
                    finish_reason = finished_state.to_finish_reason()
                    request.set_finished_reason(finish_reason, beam_idx)

            if request.py_return_log_probs:
                request.py_result.append_log_probs(log_probs, cum_log_probs)

            # Set number of tokens predicted per runtime iteration. Will be > 1 for speculative decoding.
            request.update_num_tokens_per_iteration(
                request.max_beam_num_tokens - current_num_of_tokens, self.model_config
            )

            # Increment the decoding iteration counter
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                request.py_decoding_iter += 1

            if finished_sum_host[seq_slot] == beam_width:
                request.state = LlmRequestState.GENERATION_COMPLETE
        for request in reqs:
            if finalize_events is not None and request.request_id in finalize_events:
                self._post_process_request(request, state)

    def _finalize_request(
        self,
        request: LlmRequest,
        streaming: bool,
    ) -> CudaEvent:
        """Finalizes the request. This is necessary for beam search."""
        seq_slot = request.py_seq_slot
        event = cast(
            CudaEvent,
            self.algs.decoder.finalize(  # type: ignore
                self.store["decoder_state"], seq_slot, request.sampling_config, streaming
            ),
        )
        return event

    def _post_process_request(self, request: LlmRequest, state: SampleStateTRTLLM) -> None:
        """Post Process the request. Updates the sequence according to the beam search results.
        request: LlmRequest which shall be post processed
        finalize_event: CudaEvent to wait for the finalize step to finish
        """
        assert state.host is not None
        seq_slot = request.py_seq_slot
        beam_width = request.py_beam_width
        # synchronize on the finalize event before continuing the post processing.
        # should be unnecessary, as already wait for the sampler event in update_requests
        assert state.finalize_events is not None
        state.finalize_events[request.request_id].synchronize()

        # Get these values again, as they might have changed during the finalize step
        output_ids_host = state.host.gathered_ids
        assert output_ids_host is not None
        sequence_lengths_host = state.host.sequence_lengths

        if request.py_return_log_probs:
            log_probs_host = state.host.log_probs
            cum_log_probs_host = state.host.cum_log_probs
        else:
            log_probs_host = None
            cum_log_probs_host = None

        generated_tokens = [[0]] * beam_width
        log_probs: list[list[dict[int, Logprob]]] = [[] for _ in range(beam_width)]
        cum_log_probs = []

        for beam_idx in range(beam_width):
            # get the correct generated tokens for beam search
            begin = request.py_prompt_len
            end = cast(int, sequence_lengths_host[seq_slot, beam_idx].item())
            generated_tokens[beam_idx] = output_ids_host[seq_slot, beam_idx][begin:end].tolist()

            # get the correct log probs for beam search
            if request.py_return_log_probs:
                assert log_probs_host is not None
                assert cum_log_probs_host is not None
                cum_log_probs.append(cum_log_probs_host[seq_slot, beam_idx].item())

                begin_log_probs_offset = request.prompt_len if request.py_beam_width == 1 else 0
                for current_token, token in enumerate(generated_tokens[beam_idx]):
                    log_probs[beam_idx].append(
                        {
                            token: Logprob(
                                logprob=log_probs_host[seq_slot, beam_idx][
                                    begin_log_probs_offset + current_token
                                ].item(),
                                rank=1,
                            )
                        }
                    )
        if request.py_return_log_probs:
            request.py_result.set_log_probs(log_probs, cum_log_probs)

        request.set_generated_tokens(generated_tokens)
