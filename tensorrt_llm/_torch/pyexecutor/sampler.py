# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from concurrent import futures
from dataclasses import dataclass
from functools import cached_property
from itertools import repeat
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.pyexecutor.make_decoding_batch_input_output import (
    MakeDecodingBatchInputOutput,
)
from tensorrt_llm._utils import mpi_disabled, nvtx_range, torch_dtype_to_binding
from tensorrt_llm.bindings import (
    CudaStream,
    DataType,
    ModelConfig,
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
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import LogprobMode, SamplingParams

from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..speculative.interface import get_force_num_accepted_tokens
from ..speculative.spec_tree_manager import SpecTreeManager
from .finish_reason import FinishedState
from .llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from .resource_manager import ResourceManager, ResourceManagerType
from .sampling_utils import (
    BEAM_SEARCH_PAD_TOKEN,
    GREEDY,
    BeamSearchMetadata,
    GenericStrategyKeyType,
    GroupedStrategySampler,
    SimpleGroupedStrategySampler,
    Strategy,
    StrategyMetadata,
    UtilsSamplingParams,
    get_rejected_indices,
    resolve_sampling_strategy,
    sample,
    sample_rejected,
    torch_multi_arange,
)
from .scheduler import ScheduledRequests

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

T = TypeVar("T")


@dataclass(kw_only=True)
class LogProbsState:
    sampled_vals: torch.Tensor
    sampled_indices: torch.Tensor
    sampled_rank: torch.Tensor
    topk_vals: torch.Tensor
    topk_indices: torch.Tensor


@dataclass(kw_only=True)
class LogProbsStateList:
    FloatState = list[list[list[float]]]
    IntState = list[list[list[int]]]

    sampled_vals: FloatState
    sampled_indices: IntState
    sampled_rank: IntState
    topk_vals: FloatState
    topk_indices: IntState

    @staticmethod
    def from_logprobs_state(logprobs_state: LogProbsState) -> "LogProbsStateList":
        return LogProbsStateList(
            sampled_vals=logprobs_state.sampled_vals.tolist(),
            sampled_indices=logprobs_state.sampled_indices.tolist(),
            topk_vals=logprobs_state.topk_vals.tolist(),
            topk_indices=logprobs_state.topk_indices.tolist(),
            sampled_rank=logprobs_state.sampled_rank.tolist(),
        )


@dataclass(kw_only=True)
class SampleStateTensors:
    new_tokens: torch.Tensor
    log_probs: torch.Tensor | None = None

    def values(self):
        return vars(self).values()


@dataclass(kw_only=True)
class SamplerEvent:
    cuda_event: torch.cuda.Event
    worker_futures: Optional[list[futures.Future[Any]]] = None

    def synchronize(self):
        if self.worker_futures:
            futures.wait(self.worker_futures)
        self.cuda_event.synchronize()


@dataclass(kw_only=True)
class SampleState:
    scheduled_requests: ScheduledRequests

    device: Optional[SampleStateTensors] = None
    host: Optional[SampleStateTensors] = None

    sampler_event: Optional[SamplerEvent] = None


class Sampler(ABC):
    SampleState = SampleState

    def setup_sampler_step(self, scheduled_requests: ScheduledRequests):
        pass

    def get_cache_indirection(self) -> torch.Tensor | None:
        return None

    @abstractmethod
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs,
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleState:
        raise NotImplementedError

    @abstractmethod
    def update_requests(
        self,
        state: SampleState,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def beam_width(scheduled_requests: Iterable[LlmRequest]) -> int:
        for req in scheduled_requests:
            return req.sampling_config.beam_width
        return 0

    @abstractmethod
    def is_generation_model(self) -> bool:
        raise NotImplementedError

    def should_provide_draft_probs(self, request: LlmRequest) -> bool:
        """Check if sampler wants to receive draft token probabilities."""
        return True  # conservative default


class EarlyStopSampler(Sampler):
    """
    Use for skipping decoding step for non generation model,
    such as encoder-only model (e.g., BERT) or reward models that only need context phase.
    """

    @override
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs,
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleState:
        host = SampleStateTensors(new_tokens=torch.empty(0))
        return SampleState(scheduled_requests=scheduled_requests, host=host)

    @override
    def update_requests(
        self,
        state: SampleState,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        assert isinstance(state, SampleState)
        scheduled_requests = state.scheduled_requests
        assert not scheduled_requests.generation_requests
        for idx, request in enumerate(scheduled_requests.context_requests):
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)

    @override
    def is_generation_model(self) -> bool:
        return False


@dataclass(kw_only=True)
class MultimodalResult:
    mm_embeddings: List[torch.Tensor]
    # Can be used to include e.g. `mrope_position_ids`, etc.
    extra_data: Optional[Dict[str, Any]] = None

    def values(self):
        return vars(self).values()


@dataclass(kw_only=True)
class SampleStateWithMMResult:
    scheduled_requests: ScheduledRequests

    data: MultimodalResult


@dataclass(kw_only=True, frozen=True, slots=True)
class RequestGroupKey(Generic[GenericStrategyKeyType]):
    strategy_key: GenericStrategyKeyType
    needs_probs: bool

    def __iter__(self):
        return iter((self.strategy_key, self.needs_probs))

    def __len__(self):
        return 2


@dataclass(kw_only=True, frozen=True)
class RequestGroupValue:
    indices: torch.Tensor
    strategies: list[Strategy]
    speculation_needs_probs_indices: torch.Tensor
    need_processed_logprobs: torch.Tensor
    need_raw_logprobs: torch.Tensor

    def __iter__(self):
        return iter(
            (
                self.indices,
                self.strategies,
                self.speculation_needs_probs_indices,
                self.need_processed_logprobs,
                self.need_raw_logprobs,
            )
        )

    def __len__(self):
        return 5


@dataclass(kw_only=True, frozen=True)
class RequestGroupValueWithMetadata(RequestGroupValue):
    metadata: StrategyMetadata | None

    @override
    def __iter__(self):
        return iter(
            (
                self.indices,
                self.strategies,
                self.speculation_needs_probs_indices,
                self.need_processed_logprobs,
                self.need_raw_logprobs,
                self.metadata,
            )
        )

    @override
    def __len__(self):
        return 6


class EarlyStopWithMMResult(Sampler):
    """
    Use for skipping decoding step for non generation model, and return the batch_output (such as mm_embeddings)
    """

    @override
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs,
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleStateWithMMResult:
        # from model_outputs to MultimodalResult
        data = MultimodalResult(
            mm_embeddings=model_outputs.pop("mm_embeddings"),
            extra_data={**model_outputs},
        )
        return SampleStateWithMMResult(scheduled_requests=scheduled_requests, data=data)

    @override
    def update_requests(
        self,
        state: SampleStateWithMMResult,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        # resource_manager will not be used in this function, just for interface consistency.
        assert isinstance(state, SampleStateWithMMResult)
        scheduled_requests = state.scheduled_requests
        assert not scheduled_requests.generation_requests
        mm_embeddings = state.data.mm_embeddings
        extra_data = state.data.extra_data or {}
        mrope_position_ids = extra_data.get("mrope_position_ids", None)
        mrope_position_deltas = extra_data.get("mrope_position_deltas", None)
        for i, (request, mm_embedding) in enumerate(
            zip(scheduled_requests.context_requests, mm_embeddings)
        ):
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)
            if len(mm_embedding) != sum(request.multimodal_lengths):
                raise ValueError(
                    f"mm_embedding shape mismatch: {len(mm_embedding)} != {sum(request.multimodal_lengths)}"
                )

            request.py_result.append_mm_embeddings(mm_embedding)

            # Store mrope data if available
            if mrope_position_ids is not None and mrope_position_deltas is not None:
                request.py_result.set_mrope_position(
                    mrope_position_ids[i], mrope_position_deltas[i]
                )

    @override
    def is_generation_model(self) -> bool:
        return False


# Due to tensorrt_llm::runtime::SamplingConfig using vectors, params
# in LlmRequest.sampling_params are either None or single-element lists.
# This helper method simplifies code using such params.
def _unwrap_singleton(p: Optional[List[T]]) -> Optional[T]:
    if p is None:
        return None
    (t,) = p
    return t


def _get_beam_width_in(request: LlmRequest) -> int:
    return (
        1
        if request.is_context_init_state
        else request.get_beam_width_by_iter(for_next_iteration=False)
    )


def _get_beam_width_out(request: LlmRequest) -> int:
    return request.get_beam_width_by_iter(for_next_iteration=True)


def _get_max_beam_width(request: LlmRequest) -> int:
    sampling_config = request.sampling_config
    max_beam_width = sampling_config.beam_width
    if sampling_config.beam_width_array is not None:
        max_beam_width = max(max_beam_width, sampling_config.beam_width_array.max())
    return max_beam_width


def _request_sampling_params_cachable(params: UtilsSamplingParams) -> bool:
    return not params.use_beam_search


def _request_get_sampling_params(request: LlmRequest) -> UtilsSamplingParams:
    sampling_config = request.sampling_config
    temperature = _unwrap_singleton(cast(Optional[list[float]], sampling_config.temperature))
    top_p = _unwrap_singleton(cast(Optional[list[float]], sampling_config.top_p))
    top_k = _unwrap_singleton(cast(Optional[list[int]], sampling_config.top_k))
    beam_width_out = _get_beam_width_out(request)
    beam_width_in = _get_beam_width_in(request)
    use_beam_search = _get_max_beam_width(request) > 1

    return UtilsSamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        beam_width_in=beam_width_in,
        beam_width_out=beam_width_out,
        use_beam_search=use_beam_search,
    )


def _request_strategy(request: LlmRequest, *, vocab_size: int) -> Strategy:
    # We try to cache the resolved strategy on the request object, as it's not cheap enough to
    # resolve it on every iteration.
    if hasattr(request, "py_sampling_strategy"):
        return request.py_sampling_strategy

    params = _request_get_sampling_params(request)
    sampling_strategy = resolve_sampling_strategy(params, vocab_size=vocab_size)
    if _request_sampling_params_cachable(params):
        request.py_sampling_strategy = resolve_sampling_strategy(params, vocab_size=vocab_size)
    return sampling_strategy


def _group_requests_by_strategy_key(
    requests: Iterable[LlmRequest],
    *,
    strategy_to_key: Callable[[Strategy, bool], GenericStrategyKeyType],
    pin_memory: bool = False,
    vocab_size: int,
) -> dict[RequestGroupKey[GenericStrategyKeyType], RequestGroupValue]:
    # NB: Client code relies on request indices in returned torch.Tensor being sorted.
    RequestGroupValueBuilder = namedtuple(
        "RequestGroupValueBuilder",
        [
            "indices",
            "strategies",
            "speculation_needs_probs_list",
            "need_processed_logprobs_list",
            "need_raw_logprobs_list",
        ],
    )

    group_dict: dict[RequestGroupKey, RequestGroupValueBuilder] = defaultdict(
        lambda: RequestGroupValueBuilder([], [], [], [], [])
    )

    for req_index, req in enumerate(requests):
        strategy = _request_strategy(req, vocab_size=vocab_size)
        speculation_needs_probs = (
            # NB: This criterion needs to be consistent with the gating of rejection sampling in
            #     process_draft_tokens.
            TorchSampler._speculation_could_use_rejection_sampling(req, strategy)
        )
        need_processed_logprobs = (
            req.py_logprobs_mode == LogprobMode.PROCESSED and req.return_log_probs
        )
        need_raw_logprobs = req.py_logprobs_mode == LogprobMode.RAW and req.return_log_probs
        needs_probs = speculation_needs_probs or need_processed_logprobs
        strategy_key = strategy_to_key(strategy, needs_probs)
        group_dict_entry = group_dict[
            RequestGroupKey(strategy_key=strategy_key, needs_probs=needs_probs)
        ]
        group_dict_entry.indices.append(req_index)
        group_dict_entry.strategies.append(strategy)
        if speculation_needs_probs:
            group_dict_entry.speculation_needs_probs_list.append(req_index)
        group_dict_entry.need_processed_logprobs_list.append(need_processed_logprobs)
        group_dict_entry.need_raw_logprobs_list.append(need_raw_logprobs)
    return {
        group_key: RequestGroupValue(
            indices=torch.tensor(group_value.indices, pin_memory=pin_memory, dtype=torch.int32),
            strategies=group_value.strategies,
            speculation_needs_probs_indices=torch.tensor(
                group_value.speculation_needs_probs_list, pin_memory=pin_memory, dtype=torch.int32
            ),
            need_processed_logprobs=torch.tensor(
                group_value.need_processed_logprobs_list, pin_memory=pin_memory, dtype=torch.bool
            ),
            need_raw_logprobs=torch.tensor(
                group_value.need_raw_logprobs_list, pin_memory=pin_memory, dtype=torch.bool
            ),
        )
        for group_key, group_value in group_dict.items()
    }


def add_token(
    request: LlmRequest, new_tokens: list[list[list[int]]], *, beam_idx: int, step: int = 0
) -> int:
    # NB: Accessing nested lists faster than torch.Tensor or numpy.ndarray
    seq_slot = request.py_seq_slot
    assert seq_slot is not None
    new_token = new_tokens[step][seq_slot][beam_idx]
    request.add_new_token(new_token, beam_idx)
    return new_token


def int_tensor(shape: tuple[int, ...], device: str = "cuda") -> torch.Tensor:
    return torch.empty(shape, dtype=torch.int, device=device)


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


# Beam index to use when no beam search is used but a beam index is required
DEFAULT_BEAM_IDX = 0
# Step index to use when no speculative decoding is used but a step index is required
DEFAULT_STEP_IDX = 0

FinishReasonsList = list[list[int]]


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


@dataclass(kw_only=True)
class SamplingRequestsMetadata:
    req_num_generated_tokens: torch.Tensor
    req_num_generated_tokens_output: torch.Tensor
    req_num_beams: torch.Tensor
    req_num_steps: torch.Tensor
    req_offsets: torch.Tensor


@dataclass(kw_only=True)
class SampleStateTensorsHostTorch(SampleStateTensors):
    finish_reasons: torch.Tensor
    first_finish_reasons: torch.Tensor
    logprobs_state: LogProbsState | None = None

    def finish_reasons_list(self) -> FinishReasonsList:
        """`(num_seq_slots, num_steps)`"""
        # step, slot, beam => slot, step, beam
        return self.finish_reasons.permute(1, 0, 2).tolist()


@dataclass(kw_only=True)
class SampleStateTorch(SampleState):
    host: SampleStateTensorsHostTorch
    beam_histories: list[BeamHistory | None] | None = None


class AsyncWorkerMixin:
    """
    Mixin that adds the ability to fork off operations to run on a worker
    thread (particularly D2H copies). If the async worker isn't active,
    operations will seamlessly run on the main thread.
    """

    MAX_WORKERS = 1

    def _async_worker_active(self) -> bool:
        return getattr(self, "_async_worker", None) is not None

    def _async_worker_init(self, enable_async_worker: bool):
        self._enable_async_worker = enable_async_worker
        self._async_worker = None
        self._async_worker_futures: list[futures.Future[any]] = []

    def async_worker_enabled(self):
        return getattr(self, "_enable_async_worker", False)

    def async_worker_start(self):
        assert self.async_worker_enabled()
        if not self._async_worker_active():

            def _async_worker_initializer(device_id):
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

    def async_worker_stop(self):
        assert self.async_worker_enabled()
        if self._async_worker_active():
            self._async_worker.shutdown(wait=True)
            self._async_worker = None

    @torch.inference_mode()
    def _async_copy_to_host(
        self, copy_ready: torch.cuda.Event, dest: torch.Tensor, src: torch.Tensor
    ):
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
        dest = torch.empty_like(src, device="cpu", pin_memory=True)
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
            result = self._async_worker.submit(
                self._async_copy_to_host, copy_ready, dest, src_snapshot
            )

            # Save the future, so that we can await it later
            self._async_worker_futures.append(result)
        else:
            # If the async worker is not in use, just copy as usual
            dest.copy_(src, non_blocking=True)
        return dest

    def _record_sampler_event(self) -> SamplerEvent:
        cuda_event = torch.cuda.Event()
        cuda_event.record()

        # Transfer ownership to worker_futures and re-initialize
        if self._async_worker_active():
            worker_futures = self._async_worker_futures
            self._async_worker_futures = []
        else:
            worker_futures = None

        return SamplerEvent(cuda_event=cuda_event, worker_futures=worker_futures)


class TorchSampler(Sampler, AsyncWorkerMixin):
    SampleState = SampleStateTorch
    DEFAULT_MAX_TOPK_LOGPROBS = 20

    @override
    def get_cache_indirection(self) -> torch.Tensor | None:
        return self.store.cache_indirection

    @override
    def is_generation_model(self) -> bool:
        return True

    @dataclass(kw_only=True)
    class Store:
        new_tokens: torch.Tensor
        """Shape: See cpp DecoderState.getAllNewTokens()"""
        max_lengths_tensor: torch.Tensor
        """Shape: batch_size
           Usage: Stores the maximum lengths for each request"""
        end_ids: torch.Tensor
        """Shape: batch_size
           Usage: Stores the end ids for each request"""
        finish_reasons: torch.Tensor
        """Shape: max_tokens, batch_size, beam_width
           Usage: Stores the currently estimated finish_reasons for each request"""
        cache_indirection: torch.Tensor | None = None
        """Shape: batch_size, beam_width, attention_size
           Usage: Stores the cache indirection necessary for beam search sampling"""
        cache_indirection_buffer: torch.Tensor | None = None
        """Shape: batch_size, beam_width, attention_size
           Usage: A second buffer used to update the cache indirection during sampling"""
        cum_log_probs: torch.Tensor | None = None
        """Shape: batch_size, beam_width
           Usage: Stores the current cumulative logprob of each active beam for faster sampling"""
        sampled_log_prob_indices: torch.Tensor | None = None
        """Shape: batch_size, beam_width, max_tokens
           Usage: Stores the token indices of the sampled logprobs"""
        sampled_log_probs: torch.Tensor | None = None
        """Shape: batch_size, beam_width, max_tokens
           Usage: Stores the values of the sampled logprobs"""
        sampled_log_prob_ranks: torch.Tensor | None = None
        """Shape: batch_size, beam_width, max_tokens
           Usage: Stores the ranks of the sampled logprobs"""
        topk_indices: torch.Tensor | None = None
        """Shape: batch_size, max_tokens, max_topk_logprobs
           Usage: Stores the token indices of the topk logprobs"""
        topk_vals: torch.Tensor | None = None
        """Shape: batch_size, max_tokens, max_topk_logprobs
           Usage: Stores the values of the topk logprobs"""
        first_finish_reasons: torch.Tensor | None = None
        """Shape: batch_size, beam_width
           Usage: Stores the first finish reason for each beam"""
        predecessor_beams: torch.Tensor | None = None
        """Shape: batch_size, beam_width
           Usage: Stores the predecessor beams for each beam used for stop word detection"""
        original_tokens: torch.Tensor | None = None
        """Shape: batch_size, beam_width, sequence_length
           Usage: Stores the original tokens for each beam.
           This is used to recover the original tokens for each beam when streaming is enabled"""

        def __post_init__(self):
            assert self.new_tokens.shape == self.finish_reasons.shape

    def _create_store(self) -> Store:
        # Tensors necessary for all sampling methods
        new_tokens = int_tensor(self.NEW_TOKENS_SHAPE)
        finish_reasons = int_tensor(self.NEW_TOKENS_SHAPE)
        max_lengths_tensor = int_tensor(self.max_num_sequences)
        end_ids = int_tensor(self.max_num_sequences)

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

        # Only used for beam search
        cache_indirection: torch.Tensor | None = None
        cache_indirection_buffer: torch.Tensor | None = None
        cum_log_probs: torch.Tensor | None = None
        predecessor_beams: torch.Tensor | None = None
        original_tokens: torch.Tensor | None = None
        first_finish_reasons: torch.Tensor | None = None
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
        return self.Store(
            new_tokens=new_tokens,
            finish_reasons=finish_reasons,
            max_lengths_tensor=max_lengths_tensor,
            end_ids=end_ids,
            cache_indirection=cache_indirection,
            cache_indirection_buffer=cache_indirection_buffer,
            cum_log_probs=cum_log_probs,
            sampled_log_prob_indices=sampled_log_prob_indices,
            sampled_log_probs=sampled_log_probs,
            sampled_log_prob_ranks=sampled_log_prob_ranks,
            topk_indices=topk_indices,
            topk_vals=topk_vals,
            predecessor_beams=predecessor_beams,
            original_tokens=original_tokens,
            first_finish_reasons=first_finish_reasons,
        )

    @dataclass(frozen=True, kw_only=True)
    class Args:
        max_seq_len: int
        max_draft_len: int
        max_num_sequences: int
        max_beam_width: int
        max_total_draft_tokens: int
        disable_overlap_scheduler: bool = False
        disable_flashinfer_sampling: bool = False
        enable_async_worker: bool = False

    def __init__(self, args: Args):
        self.max_seq_len = args.max_seq_len
        self.max_tokens = args.max_total_draft_tokens + 1
        self.max_beam_width = args.max_beam_width
        # The current maximum number of topk logprobs which can be stored in the sampler's store
        self.max_topk_logprobs = self.DEFAULT_MAX_TOPK_LOGPROBS
        # The maximum number of topk logprobs for the current batch of requests
        self.batch_max_topk_logprobs = 0
        if args.max_total_draft_tokens > 0 and args.max_beam_width > 1:
            raise ValueError("TorchSampler does not support beam search with speculative decoding")
        self.max_num_sequences = args.max_num_sequences

        self.NEW_TOKENS_SHAPE = (self.max_tokens, self.max_num_sequences, self.max_beam_width)
        self.CACHE_INDIRECTION_SHAPE = (
            self.max_num_sequences,
            self.max_beam_width,
            self.max_seq_len + (0 if args.disable_overlap_scheduler else 1),
        )
        self.LOGPROBS_SHAPE = (self.max_num_sequences, self.max_beam_width, self.max_tokens)
        self.TOPK_LOGPROBS_SHAPE = (self.max_num_sequences, self.max_tokens, self.max_topk_logprobs)
        # AutoDeploy build creates the sampler in inference mode,
        # which would disallow in-place mutating of new_tokens.
        # So, we temporarily exit inference mode.
        with torch.inference_mode(False):
            self.store = self._create_store()
            # Helper tensors for finish_reasons:
            """Preallocate buffer needed for torch.nonzero_static(..., out=finish_reasons_nonzero_static_buffer).
            See `def _write_reason`."""
            self._reason_tensors = {
                reason: torch.tensor(
                    reason.value, dtype=self.store.finish_reasons.dtype, device="cuda"
                )
                for reason in [
                    FinishReason.NOT_FINISHED,
                    FinishReason.END_ID,
                    FinishReason.STOP_WORDS,
                    FinishReason.LENGTH,
                    FinishReason.TIMED_OUT,
                    FinishReason.CANCELLED,
                ]  # `in FinishReason` clashes with PyBind11: `TypeError: 'pybind11_type' object is not iterable`
            }
            self._max_tokens_offset = torch.arange(
                1, self.max_tokens + 1, device="cuda", dtype=torch.int32
            ).view(-1, 1, 1)

        self._grouped_sampler_cls: Type[GroupedStrategySampler]
        if IS_FLASHINFER_AVAILABLE and not args.disable_flashinfer_sampling:
            from .sampling_utils_flashinfer import FlashInferGroupedStrategySampler

            self._grouped_sampler_cls = FlashInferGroupedStrategySampler
        else:
            self._grouped_sampler_cls = SimpleGroupedStrategySampler

        # Initialize seed for multi-GPU consistency
        self._global_seed = 42
        self._generator = None

        # Force number of accepted tokens for speculative decoding testing
        self._force_num_accepted_tokens = get_force_num_accepted_tokens()

        self._async_worker_init(args.enable_async_worker)

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

    def _handle_finish_reasons(
        self,
        request: LlmRequest,
        finish_reasons: torch.Tensor,
        finish_reasons_list: list[list[list[int]]],
    ):
        """Check if all beams of a request have finished and set the request state accordingly

        Args:
            request: LlmRequest. The request to check.
            finish_reasons: torch.Tensor. Shape: (max_tokens, max_batch_size, max_beam_width)
                            The finish reasons for each beam.
        Returns:
            True if all beams have finished, False otherwise.
        """
        if (
            finish_reasons[
                DEFAULT_STEP_IDX, request.py_seq_slot, : request.sampling_config.beam_width
            ]
            != FinishReason.NOT_FINISHED.value
        ).sum() == request.sampling_config.beam_width:
            request.state = LlmRequestState.GENERATION_COMPLETE
            for beam_idx in range(request.sampling_config.beam_width):
                request.set_finished_reason(
                    FinishReason(
                        finish_reasons_list[request.py_seq_slot][DEFAULT_STEP_IDX][beam_idx]
                    ),
                    beam_idx,
                )
            return True
        return False

    @nvtx_range("update_original_tokens")
    def _update_original_tokens(
        self, seq_slots: torch.Tensor, seq_lens: torch.Tensor, new_tokens: torch.Tensor
    ):
        """Update the original tokens storage for the request with the newly sampled tokens

        When using streaming a requests tokens may be altered, leading to wrong results when called multiple times.
        Store the original tokens in a separate buffer to use them as a consistent basis
        when updating the tokens in a request."""
        assert new_tokens.device == self.store.original_tokens.device, (
            "new_tokens and original_tokens must be on the same device"
        )
        self.store.original_tokens[seq_slots, :, seq_lens] = new_tokens[0, seq_slots, :]

    def _convert_logprobs_tensor_to_list(
        self,
        token_tensor: torch.Tensor,
        logprobs_tensor: torch.Tensor,
        sampled_log_probs_indices: torch.Tensor | None,
        sampled_log_probs_vals: torch.Tensor | None,
        sampled_log_probs_rank: torch.Tensor | None,
    ) -> list[list[dict[int, Logprob]]]:
        """Convert the logprobs tensor to a list of lists of dictionaries of Logprob objects

        Logprobs storage expects logprobs as a list[list[dict[int, Logprob]]] object

        args:
            token_tensor: torch.Tensor. Shape: beam_width, num_tokens, num_logprobs
            logprobs_tensor: torch.Tensor. Shape: beam_width, num_tokens, num_logprobs
            sampled_log_probs_indices: torch.Tensor | None. Shape: num_tokens
            sampled_log_probs_vals: torch.Tensor | None. Shape: num_tokens
            sampled_log_probs_rank: torch.Tensor | None. Shape: num_tokens
        output:
            list[list[dict[int, Logprob]]]. Shape: (beam_width, num_tokens)
        """
        assert token_tensor.dim() == 3 and logprobs_tensor.dim() == 3, (
            f"Token and logprobs tensors must have 3 dimensions (beam_width, num_tokens, num_logprobs). \
            Got shapes (token_tensor) {token_tensor.shape} and (logprobs_tensor) {logprobs_tensor.shape} instead"
        )

        token_log_probs: list[list[dict[int, Logprob]]] = []
        token_list = token_tensor.tolist()
        logprobs_list = logprobs_tensor.tolist()
        sampled_log_probs_indices_list: list[int] | None = None
        sampled_log_probs_vals_list: list[float] | None = None
        sampled_log_probs_rank_list: list[int] | None = None
        if sampled_log_probs_indices is not None:
            sampled_log_probs_indices_list = sampled_log_probs_indices.tolist()
            assert sampled_log_probs_vals is not None, "sampled_log_probs_vals must be provided"
            assert sampled_log_probs_rank is not None, "sampled_log_probs_rank must be provided"
            sampled_log_probs_vals_list = sampled_log_probs_vals.tolist()
            sampled_log_probs_rank_list = sampled_log_probs_rank.tolist()
        for beam_idx in range(token_tensor.shape[0]):
            beam_token_log_probs: list[dict[int, Logprob]] = []
            for step_idx, (topk_token, topk_logprob) in enumerate(
                zip(token_list[beam_idx], logprobs_list[beam_idx])
            ):
                logprobs = {
                    token: Logprob(logprob=logprob, rank=rank + 1)
                    for rank, (token, logprob) in enumerate(zip(topk_token, topk_logprob))
                }
                if sampled_log_probs_indices is not None:
                    assert beam_idx == DEFAULT_BEAM_IDX, (
                        "beam search does not need to explicitly handle sampled log probs"
                    )
                    if sampled_log_probs_indices_list[step_idx] not in logprobs:
                        logprobs[sampled_log_probs_indices_list[step_idx]] = Logprob(
                            logprob=sampled_log_probs_vals_list[step_idx],
                            rank=max(
                                token_tensor.shape[2] + 1, sampled_log_probs_rank_list[step_idx]
                            ),
                        )
                beam_token_log_probs.append(logprobs)
            token_log_probs.append(beam_token_log_probs)

        return token_log_probs

    def _store_logprobs_list_to_request(
        self,
        logprobs_state_list: LogProbsStateList,
        req_seq_slot: int,
        beam_width: int,
        count: int,
        num_topk_logprobs: int,
    ) -> list[list[dict[int, Logprob]]]:
        """Convert the LogProbsStateList object to a list of lists of dictionaries of Logprob objects

        Logprobs storage expects logprobs as a list[list[dict[int, Logprob]]] object

        args:
            logprobs_state_list: LogProbsStateList. Contains the topk indices, topk values,
                sampled indices, sampled values, and sampled ranks.
            req_seq_slot: int. The sequence slot of the request.
            beam_width: int. The beam width of the request.
            count: int. The number of tokens to store.
            num_topk_logprobs: int. The number of topk logprobs of each token.
        output:
            list[list[dict[int, Logprob]]]. Shape: (beam_width, count)
        """

        token_list = logprobs_state_list.topk_indices[req_seq_slot]
        logprobs_list = logprobs_state_list.topk_vals[req_seq_slot]
        sampled_log_probs_indices_list = logprobs_state_list.sampled_indices[req_seq_slot]
        sampled_log_probs_vals_list = logprobs_state_list.sampled_vals[req_seq_slot]
        sampled_log_probs_rank_list = logprobs_state_list.sampled_rank[req_seq_slot]

        token_log_probs: list[list[dict[int, Logprob]]] = []
        for beam_idx in range(beam_width):
            beam_token_log_probs: list[dict[int, Logprob]] = []
            for step_idx, (topk_token, topk_logprob) in enumerate(
                zip(token_list[:count], logprobs_list[:count])
            ):
                logprobs = {
                    token: Logprob(logprob=logprob, rank=rank + 1)
                    for rank, (token, logprob) in enumerate(
                        zip(topk_token[:num_topk_logprobs], topk_logprob[:num_topk_logprobs])
                    )
                }
                if sampled_log_probs_indices_list[beam_idx][step_idx] not in logprobs:
                    logprobs[sampled_log_probs_indices_list[beam_idx][step_idx]] = Logprob(
                        logprob=sampled_log_probs_vals_list[beam_idx][step_idx],
                        rank=max(
                            len(token_list[step_idx]) + 1,
                            sampled_log_probs_rank_list[beam_idx][step_idx] + 1,
                        ),
                    )
                beam_token_log_probs.append(logprobs)
            token_log_probs.append(beam_token_log_probs)

        return token_log_probs

    def handle_logprobs(
        self,
        request: LlmRequest,
        logprobs_state_list: LogProbsStateList | None,
        *,
        count: int,
    ):
        if request.py_return_log_probs:
            beam_width = request.sampling_config.beam_width
            assert request.py_num_logprobs is not None, "request.py_num_logprobs must be provided"
            assert logprobs_state_list is not None, "logprobs_state_list must be provided"
            token_log_probs = self._store_logprobs_list_to_request(
                logprobs_state_list, request.py_seq_slot, beam_width, count, request.py_num_logprobs
            )
            request.py_result.append_log_probs(token_log_probs)

    def finish_if_reason(
        self, request: LlmRequest, finish_reasons: FinishReasonsList, *, step: int, beam_idx: int
    ) -> bool:
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

            cur_accepted_len = torch.cumprod(
                (cur_draft_tokens == cur_target_tokens[:-1]).int(), dim=-1
            ).sum()

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

    def _is_new_request(self, request: LlmRequest) -> bool:
        return (
            not request.is_finished
            and not request.py_is_draft
            and (
                (request.is_context_init_state and request.is_last_context_chunk)
                or request.is_disagg_generation_transmission_complete
            )
        )

    @override
    def setup_sampler_step(self, scheduled_requests: ScheduledRequests):
        """Setup the sampler step for the requests

        Args:
            requests: list[LlmRequest]. The requests to setup the sampler step for
        """
        if self._use_beam_search:
            self._prepare_beam_search(scheduled_requests.all_requests())

        seq_slots: list[int] = []
        max_lens: list[int] = []
        end_ids: list[int] = []
        for request in scheduled_requests.context_requests:
            if self._is_new_request(request):
                seq_slots.append(request.py_seq_slot)
                max_lens.append(
                    min(self.max_seq_len, request.orig_prompt_len + request.py_max_new_tokens)
                )
                end_ids.append(request.py_end_id if request.py_end_id is not None else -1)
        if len(seq_slots) > 0:
            full_list = [seq_slots, max_lens, end_ids]
            # perform only a single copy
            full_list_tensor = torch.tensor(full_list, device="cpu", dtype=torch.int32).to(
                device="cuda", non_blocking=True
            )
            seq_slots_tensor = full_list_tensor[0]
            max_lens_tensor = full_list_tensor[1]
            end_ids_tensor = full_list_tensor[2]
            self.store.max_lengths_tensor[seq_slots_tensor] = max_lens_tensor
            self.store.end_ids[seq_slots_tensor] = end_ids_tensor

    def _prepare_beam_search(
        self,
        requests: list[LlmRequest],
    ):
        """Prepare the beam search buffers for the requests

        If the last context chunk is being processed,
        initialize/reset the buffers for the request
        """
        for request in requests:
            if self._is_new_request(request):
                if request.py_return_log_probs and request.py_num_logprobs > 1:
                    raise ValueError("Beam search does not support multiple logprobs")
                self.store.cache_indirection[request.py_seq_slot, :, request.py_prompt_len].fill_(0)
                self.store.cum_log_probs[request.py_seq_slot].fill_(0)
                self.store.sampled_log_probs[request.py_seq_slot].fill_(0)
                self.store.sampled_log_prob_ranks[request.py_seq_slot].fill_(0)
                self.store.predecessor_beams[request.py_seq_slot].fill_(0)
                self.store.first_finish_reasons[request.py_seq_slot].fill_(
                    FinishReason.NOT_FINISHED.value
                )
                self.store.original_tokens[request.py_seq_slot].fill_(0)

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
        return get_draft_token_length(request) > 0 and strategy != GREEDY

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

    def _get_logprobs_from_request(self, request: LlmRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the logprobs from the request

        Returns:
            logprobs_tensor: A tensor of shape (beam_width, num_generated_tokens, num_logprobs)
            logprobs_indices_tensor: A tensor of shape (beam_width, num_generated_tokens, num_logprobs)
        """
        num_generated_tokens = request.max_beam_num_tokens - request.py_prompt_len
        assert request.py_num_logprobs == 0, (
            "Beam search only supports returning the sampled logprob per token"
        )
        logprobs_tensor = torch.empty(
            (
                request.sampling_config.beam_width,
                num_generated_tokens,
                request.py_num_logprobs + 1,
            ),
            device="cuda",
            dtype=torch.float32,
        )
        logprobs_indices_tensor = torch.empty(
            (
                request.sampling_config.beam_width,
                num_generated_tokens,
                request.py_num_logprobs + 1,
            ),
            device="cuda",
            dtype=torch.int32,
        )
        if hasattr(request.py_result._log_probs, "log_probs"):
            logprobs_list = request.py_result.log_probs
            for beam_idx, beam_logprobs in enumerate(logprobs_list):
                for token_idx, token_logprobs in enumerate(beam_logprobs):
                    for key, value in token_logprobs.items():
                        logprobs_tensor[beam_idx, token_idx, value.rank - 1] = value.logprob
                        logprobs_indices_tensor[beam_idx, token_idx, value.rank - 1] = key
        return logprobs_tensor, logprobs_indices_tensor

    def _create_beam_history(
        self,
        request: LlmRequest,
    ) -> BeamHistory | None:
        """Correct the stored tokens for each beam and return it as a BeamHistory object.

        Beam Search sampling only adds new tokens to the beam.
        However during beam search, a beam may change its previously sampled tokens.
        This function corrects the stored tokens for each beam to match the expected tokens.
        If logprobs are requested, the function also corrects the stored logprobs for each beam.
        The function returns a BeamHistory object that contains the corrected tokens and logprobs for each beam.

        arguments:
            request: The request to create the beam history for
        """
        num_tokens = request.max_beam_num_tokens + 1  # last token is not yet added
        prompt_length = request.py_prompt_len
        num_generated_tokens = num_tokens - prompt_length
        num_beams = request.sampling_config.beam_width

        if num_generated_tokens == 0 or request.state == LlmRequestState.GENERATION_COMPLETE:
            # early return if no tokens have been generated yet or the request is already finished
            return None
        cache_indirection = self.store.cache_indirection[
            request.py_seq_slot, :num_beams, prompt_length:num_tokens
        ]
        current_path = self.store.original_tokens[
            request.py_seq_slot, :num_beams, prompt_length:num_tokens
        ]
        new_path = torch.zeros_like(current_path)
        if request.py_return_log_probs:
            current_logprobs, current_logprobs_indices = self._get_logprobs_from_request(request)
            # concatenate the newly generated logprobs and newly
            # generated tokens to the current logprobs and logprobs indices
            current_logprobs = torch.cat(
                [
                    current_logprobs,
                    self.store.sampled_log_probs[request.py_seq_slot, :num_beams].view(-1, 1, 1),
                ],
                dim=1,
            )
            current_logprobs_indices = torch.cat(
                [
                    current_logprobs_indices,
                    self.store.new_tokens[0, request.py_seq_slot, :num_beams].view(-1, 1, 1),
                ],
                dim=1,
            )
            # Initialize the buffers to store the results
            new_logprobs = torch.zeros_like(current_logprobs)
            new_logprobs_indices = torch.zeros_like(current_logprobs_indices)
        # initialize each beam with its own index

        # Gather the correct tokens and logprobs for each beam
        torch.gather(input=current_path, dim=0, index=cache_indirection, out=new_path)
        if request.py_return_log_probs:
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
            cum_logprobs = self.store.cum_log_probs[request.py_seq_slot, :num_beams]
            return BeamHistory(
                tokens=new_path,
                logprobs=new_logprobs,
                logprobs_indices=new_logprobs_indices,
                cum_logprobs=cum_logprobs,
            )
        else:
            return BeamHistory(
                tokens=new_path,
                logprobs=None,
                logprobs_indices=None,
                cum_logprobs=None,
            )

    def _finalize_beam(
        self,
        request: LlmRequest,
        beam_history: BeamHistory,
    ) -> None:
        """Update the request with the corrected tokens and logprobs for each beam.

        arguments:
            request: The request to update
            beam_history: The beam history used to update the request
            finish_reasons: The finish reasons to use to check if the beam is finished (Shape: (beam_width,))
        """

        beam_width = request.sampling_config.beam_width
        assert beam_history.tokens.shape[0] == beam_width, (
            f"Beam_history.tokens.shape[0] should equal beam width: \
                {beam_history.tokens.shape[0]} != {beam_width}"
        )
        if request.py_return_log_probs:
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
        valid_tokens = (beam_history.tokens != BEAM_SEARCH_PAD_TOKEN).sum(dim=-1)
        gen_token_list = []
        gen_log_probs_list = []
        for beam_idx in range(beam_width):
            gen_token_list.append(beam_history.tokens[beam_idx, : valid_tokens[beam_idx]].tolist())
            if request.py_return_log_probs:
                gen_log_probs_list.append(
                    self._convert_logprobs_tensor_to_list(
                        beam_history.logprobs_indices[
                            beam_idx : beam_idx + 1, : valid_tokens[beam_idx]
                        ],
                        beam_history.logprobs[beam_idx : beam_idx + 1, : valid_tokens[beam_idx]],
                        None,
                        None,
                        None,
                    )[0]
                )
        request.set_generated_tokens(gen_token_list)
        if request.py_return_log_probs:
            # cum_log_probs will not change when padding with end tokens.
            # Therefore, we do not need to correct it
            request.py_result.set_log_probs(
                gen_log_probs_list, cum_log_probs=beam_history.cum_logprobs.tolist()
            )

    def _add_metadata_to_grouped_requests(
        self,
        requests: list[LlmRequest],
        grouped_requests: dict[RequestGroupKey[GenericStrategyKeyType], RequestGroupValue],
        seq_slots: torch.Tensor,
        seq_lens: torch.Tensor | None,
        get_metadata_type_for_group_fn: Callable[[GenericStrategyKeyType], Type[StrategyMetadata]],
    ) -> dict[RequestGroupKey[GenericStrategyKeyType], RequestGroupValueWithMetadata]:
        grouped_requests_with_metadata: dict[
            RequestGroupKey[GenericStrategyKeyType], RequestGroupValueWithMetadata
        ] = {}
        for key, value in grouped_requests.items():
            metadata_type = get_metadata_type_for_group_fn(key.strategy_key)
            if metadata_type is BeamSearchMetadata:
                assert seq_lens is not None, "seq_lens is required for beam search"
                metadata = BeamSearchMetadata(
                    cache_indirection=self.store.cache_indirection,
                    cache_indirection_buffer=self.store.cache_indirection_buffer,
                    cum_log_probs=self.store.cum_log_probs,
                    new_log_probs=self.store.sampled_log_probs[..., DEFAULT_STEP_IDX],
                    seq_slots=seq_slots[grouped_requests[key].indices].to(
                        device="cuda", dtype=torch.int64, non_blocking=True
                    ),  # Should be on device for beam search, need long for index_copy_
                    seq_lens=seq_lens[grouped_requests[key].indices].to(
                        device="cuda", non_blocking=True
                    ),  # Should be on device for beam search
                    finished_beams=self.store.first_finish_reasons,
                    predecessor_beams=self.store.predecessor_beams,
                    end_ids=torch.tensor(
                        [
                            requests[request_idx].py_end_id
                            for request_idx in grouped_requests[key].indices
                        ],
                        dtype=torch.int32,
                    ).to(
                        device="cuda", non_blocking=True
                    ),  # end_ids should be on device for beam search
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

    def _check_beam_search_stop_criteria(
        self,
        request: LlmRequest,
        finish_reasons: torch.Tensor,
    ) -> bool:
        """Check if the stop criteria is met for the request"""
        return (
            finish_reasons[: request.sampling_config.beam_width] > 0
        ).sum().item() == request.sampling_config.beam_width  # NB: This syncs

    def _check_stop_words_length(self, request: LlmRequest) -> bool:
        """Check if the stop words length is greater than 1"""
        if request.py_stop_words_list is not None:
            _, cumsum = request.py_stop_words_list
            if -1 in cumsum:
                cumsum = cumsum[: cumsum.index(-1)]
            longest_stop_word_len = np.max(np.diff(cumsum, prepend=0), initial=0)
            return longest_stop_word_len > 1
        return False

    @nvtx_range("maybe_create_beam_histories")
    def _maybe_create_beam_histories(
        self,
        requests: list[LlmRequest],
        finish_reasons: torch.Tensor,
        beam_histories: list[BeamHistory | None],
    ) -> None:
        """Create the corrected tokens and logprobs for each beam of a request

        This function creates a beam history object containing the corrected
        tokens and logprobs for each beam of a request"""
        for req_idx, req in enumerate(requests):
            should_stop = self._check_beam_search_stop_criteria(
                req, finish_reasons=finish_reasons[req.py_seq_slot]
            )
            need_finalize_due_to_stop_words = self._check_stop_words_length(req)
            if should_stop or req.streaming or need_finalize_due_to_stop_words:
                beam_histories[req_idx] = self._create_beam_history(req)

    @override
    @nvtx_range("update_requests")
    @torch.inference_mode()
    def update_requests(
        self,
        state: SampleStateTorch,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        assert isinstance(state, SampleStateTorch)
        if state.sampler_event:
            state.sampler_event.synchronize()

        assert state.host is not None
        new_tokens = state.host.new_tokens
        finish_reasons = state.host.finish_reasons_list()

        new_tokens_list = new_tokens.tolist()
        beam_histories = state.beam_histories
        logprobs_state_list: LogProbsStateList | None = None
        if state.host.logprobs_state is not None:
            logprobs_state_list = LogProbsStateList.from_logprobs_state(state.host.logprobs_state)

        for req_idx, req in enumerate(state.scheduled_requests.context_requests):
            if (
                req.state == LlmRequestState.GENERATION_COMPLETE
                or req.context_remaining_length != 0
            ):
                continue
            if beam_histories is not None and beam_histories[req_idx] is not None:
                self._finalize_beam(
                    req,
                    beam_histories[req_idx],
                )
            else:
                for beam_idx in range(req.sampling_config.beam_width):
                    add_token(req, new_tokens_list, beam_idx=beam_idx)
                self.handle_logprobs(req, logprobs_state_list=logprobs_state_list, count=1)
            self._handle_finish_reasons(req, state.host.finish_reasons, finish_reasons)
            req.py_decoding_iter += 1

        for req_idx, req in enumerate(
            state.scheduled_requests.generation_requests,
            len(state.scheduled_requests.context_requests),
        ):
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            if req.sampling_config.beam_width > 1:
                if beam_histories is not None and beam_histories[req_idx] is not None:
                    self._finalize_beam(
                        req,
                        beam_histories[req_idx],
                    )
                else:
                    for beam_idx in range(req.sampling_config.beam_width):
                        # Beam search does not support speculative decoding.
                        add_token(req, new_tokens_list, beam_idx=beam_idx)
                    self.handle_logprobs(req, logprobs_state_list=logprobs_state_list, count=1)
                self._handle_finish_reasons(req, state.host.finish_reasons, finish_reasons)
                req.py_num_accepted_draft_tokens = 0
                req.py_rewind_len = 0

            else:
                processed = 1
                num_accepted = self.process_draft_tokens(
                    req,
                    new_tokens_tensor=new_tokens,
                    new_tokens_list=new_tokens_list,
                    finish_reasons=finish_reasons,
                    resource_manager=resource_manager,
                )
                if get_draft_token_length(req) > 0:
                    req.py_num_accepted_draft_tokens = num_accepted
                    actual_draft_len = get_draft_token_length(req)
                    req.py_rewind_len = actual_draft_len - num_accepted
                else:
                    req.py_num_accepted_draft_tokens = 0
                    req.py_rewind_len = 0
                processed += num_accepted
                self.handle_logprobs(req, logprobs_state_list=logprobs_state_list, count=processed)
            req.py_decoding_iter += 1

    def _return_log_probs(self, requests: list[LlmRequest]) -> bool:
        return any(req.py_return_log_probs for req in requests)

    def _prepare_log_probs(self, requests: list[LlmRequest]) -> None:
        self.batch_max_topk_logprobs = max(
            (req.py_num_logprobs or 0 for req in requests), default=0
        )
        if self.max_topk_logprobs < self.batch_max_topk_logprobs:
            self.max_topk_logprobs = self.batch_max_topk_logprobs
            self.TOPK_LOGPROBS_SHAPE = (
                self.max_num_sequences,
                self.max_tokens,
                self.max_topk_logprobs,
            )
            self.store.topk_vals.resize_(self.TOPK_LOGPROBS_SHAPE)
            self.store.topk_indices.resize_(self.TOPK_LOGPROBS_SHAPE)

    @override
    @torch.inference_mode()
    @nvtx_range("sample_async")
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, torch.Tensor],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleStateTorch:
        # NB: The sampler is either called directly by PyExecutor, for the target model,
        #     or by ModelDrafter.prepare_draft_tokens(), for the draft model. In the former
        #     case there are 1 + get_draft_token_length(request) tokens per request. In the
        #     latter case, there is always only 1 token per request because draft
        #     tokens are sampled one-by-one.
        self.setup_sampler_step(scheduled_requests)
        requests = scheduled_requests.all_requests()
        new_tokens = self.store.new_tokens
        seq_slots_host = torch.tensor(
            [r.py_seq_slot for r in requests],
            dtype=torch.int64,  # for index_fill_
            pin_memory=True,
        )
        # necessary for beam search and max_length checks
        seq_lens_host = torch.tensor(
            [r.max_beam_num_tokens for r in requests], dtype=torch.int32, pin_memory=True
        )
        new_tokens_host = self._process_requests(
            scheduled_requests,
            model_outputs,
            new_tokens,
            num_context_logits_prefix_sum,
            seq_slots=seq_slots_host,
            seq_lens=seq_lens_host,
        )

        finish_reasons = self.store.finish_reasons
        seq_slots = seq_slots_host.to(device="cuda", non_blocking=True)
        seq_lens = seq_lens_host.to(device="cuda", non_blocking=True)
        first_finish_reasons = self.store.first_finish_reasons if self._use_beam_search else None

        self._write_finish_reasons(
            requests,
            finish_reasons=finish_reasons,
            seq_slots=seq_slots,
            seq_lens=seq_lens,
            new_tokens=new_tokens,
            first_finish_reasons=first_finish_reasons,
            predecessor_beams=self.store.predecessor_beams,
        )
        finish_reasons_host = self._copy_to_host(finish_reasons)

        beam_histories = [None] * len(requests)
        if self._use_beam_search:
            assert seq_lens_host is not None, "seq_lens is required for beam search"
            assert self.store.first_finish_reasons is not None, (
                "first_finish_reasons must be provided"
            )
            seq_lens = seq_lens_host.to(device="cuda", non_blocking=True)
            first_finish_reasons_host = self._copy_to_host(self.store.first_finish_reasons)
            self._update_original_tokens(seq_slots, seq_lens, new_tokens)
            self._maybe_create_beam_histories(
                requests, finish_reasons=first_finish_reasons, beam_histories=beam_histories
            )

        # copy logprobs to host
        logprobs_state: LogProbsState | None = None
        if self._return_log_probs(requests):
            assert self.store.topk_vals is not None, "topk_vals must be provided"
            assert self.store.topk_indices is not None, "topk_indices must be provided"
            assert self.store.sampled_log_probs is not None, "sampled_log_probs must be provided"
            assert self.store.sampled_log_prob_indices is not None, (
                "sampled_log_prob_indices must be provided"
            )
            assert self.store.sampled_log_prob_ranks is not None, (
                "sampled_log_prob_ranks must be provided"
            )
            host_topk_vals = self._copy_to_host(
                self.store.topk_vals[..., : self.batch_max_topk_logprobs]
            )
            host_topk_indices = self._copy_to_host(
                self.store.topk_indices[..., : self.batch_max_topk_logprobs]
            )
            host_sampled_vals = self._copy_to_host(self.store.sampled_log_probs)
            host_sampled_indices = self._copy_to_host(self.store.sampled_log_prob_indices)
            host_sampled_rank = self._copy_to_host(self.store.sampled_log_prob_ranks)
            logprobs_state = LogProbsState(
                topk_vals=host_topk_vals,
                topk_indices=host_topk_indices,
                sampled_vals=host_sampled_vals,
                sampled_indices=host_sampled_indices,
                sampled_rank=host_sampled_rank,
            )

        sampler_event = self._record_sampler_event()
        return SampleStateTorch(
            scheduled_requests=scheduled_requests,
            device=SampleStateTensors(new_tokens=new_tokens),
            host=SampleStateTensorsHostTorch(
                new_tokens=new_tokens_host,
                finish_reasons=finish_reasons_host,
                first_finish_reasons=None
                if not self._use_beam_search
                else first_finish_reasons_host,
                logprobs_state=logprobs_state,
            ),
            sampler_event=sampler_event,
            beam_histories=beam_histories,
        )

    @staticmethod
    def _apply_d2t(tokens: torch.Tensor, model_outputs) -> None:
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
        #     torch.Tensor.scatter_add_ (and it's variant torch.Tensor.index_add_ -- allows
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
            bias_gather_indices, pin_memory=True, dtype=torch.int32
        ).to(logits.device, non_blocking=True)
        logits_bias_mask_cuda = torch.tensor(
            logits_bias_masks, pin_memory=True, dtype=torch.bool
        ).to(logits.device, non_blocking=True)
        biases_tensor = torch.empty((len(bias_to_index), *req_bias.shape), pin_memory=True)
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
        model_outputs: dict[str, torch.Tensor],
        *,
        cuda_device: torch.device,
        logits_cuda_indexer: _PackedStepIndexer,
        req_num_generated_tokens: torch.Tensor,
        req_num_steps: torch.Tensor,
        req_offsets: torch.Tensor,
        seq_slots: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        token_dtype: torch.dtype,
        return_log_probs: bool,
    ) -> _BatchedSamplingResult:
        grouped_requests = _group_requests_by_strategy_key(
            requests,
            pin_memory=True,
            vocab_size=logits_cuda.size(1),
            strategy_to_key=self._grouped_sampler_cls.strategy_grouping_key,
        )
        grouped_requests_with_metadata = self._add_metadata_to_grouped_requests(
            requests,
            grouped_requests,
            seq_slots,
            seq_lens,
            get_metadata_type_for_group_fn=self._grouped_sampler_cls.get_metadata_type_for_group,
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
        for (strategy_key, needs_probs), (
            group_req_indices,
            group_strategies,
            group_speculation_needs_probs_indices,
            group_need_processed_logprobs,
            group_need_raw_logprobs,
            group_metadata,
        ) in grouped_requests_with_metadata.items():
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
                    max_steps=req_num_generated_tokens.max() * self.max_beam_width,
                )
            logit_indices_for_processed_logprobs_cuda = (
                None
                if not any_request_needs_processed_logprobs
                else group_logit_indexer[need_processed_logprobs_indices].to(
                    logits_cuda.device, non_blocking=True
                )
            )
            logit_indices_for_raw_logprobs_cuda = (
                None
                if not any_request_needs_raw_logprobs
                else group_logit_indexer[need_raw_logprobs_indices].to(
                    logits_cuda.device, non_blocking=True
                )
            )

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
                current_logits_cuda = group_logits_cuda[
                    group_logits_indices_for_processed_logprobs_cuda
                ]
                current_softmax_cuda = group_softmax_cuda[logit_indices_for_processed_logprobs_cuda]
                processed_logits_cuda = torch.where(
                    current_softmax_cuda > 0, current_logits_cuda, float("-inf")
                )
                if group_temperature_cuda is not None:
                    if isinstance(group_temperature_cuda, torch.Tensor):
                        processed_logits_cuda /= group_temperature_cuda[
                            logit_indices_for_processed_logprobs_cuda
                        ]
                    else:
                        processed_logits_cuda /= group_temperature_cuda
                logit_indices_for_processed_logprobs_cuda += batch_next_tokens_offset_start
                batch_logits_for_logprobs_cuda[logit_indices_for_processed_logprobs_cuda] = (
                    processed_logits_cuda
                )

            if any_request_needs_raw_logprobs:
                assert group_logits_indices_for_raw_logprobs_cuda is not None
                assert logit_indices_for_raw_logprobs_cuda is not None
                assert batch_logits_for_logprobs_cuda is not None
                raw_logits_cuda = group_logits_cuda[group_logits_indices_for_raw_logprobs_cuda]
                logit_indices_for_raw_logprobs_cuda += batch_next_tokens_offset_start
                batch_logits_for_logprobs_cuda[logit_indices_for_raw_logprobs_cuda] = (
                    raw_logits_cuda
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
        new_tokens_host = self._copy_to_host(new_tokens_cuda)

        return new_tokens_host

    @staticmethod
    @torch.inference_mode()
    def _apply_min_length_penalty(
        logits: torch.Tensor,
        requests: list[LlmRequest],
        num_steps: list[int],
        num_beams: list[int],
    ) -> torch.Tensor:
        """Inplace apply min_length_penalty to logits.

        Args:
            logits: The logits to apply min length penalty to
            requests: The requests to apply min length penalty to
            num_steps: The number of steps per request

        Returns:
            The logits with min length penalty applied
        """
        if any(r.py_min_length and r.max_beam_num_tokens < r.py_min_length[0] for r in requests):
            current_offset = 0
            for index, r in enumerate(requests):
                if r.py_min_length:
                    for beam_idx in range(num_beams[index]):
                        for step in range(num_steps[index]):
                            if r.get_num_tokens(beam_idx) + step < r.py_min_length[0]:
                                # NOTE(jthomson04): We can NOT just assign logits[...] = float("-inf").
                                # This introduces a pageable HtoD transfer, which wreaks havoc on TPOT (up to ~20%)
                                # Instead, we create a little tensor on device, then assign to that.
                                # This way, we avoid the pageable transfer.
                                neg_inf_tensor = torch.full((), float("-inf"), device=logits.device)
                                logits[
                                    current_offset + num_steps[index] * beam_idx + step, r.py_end_id
                                ] = neg_inf_tensor
                            else:
                                # early exit
                                break
                current_offset += num_steps[index] * num_beams[index]
        return logits

    @staticmethod
    def _select_generated_logits(
        scheduled_requests: ScheduledRequests,
        raw_logits_cuda: torch.Tensor,
        *,
        num_context_logits_prefix_sum: list[int],
    ) -> tuple[SamplingRequestsMetadata, torch.Tensor]:
        requests = scheduled_requests.all_requests()

        req_num_generation_steps_list = [1 + get_draft_token_length(req) for req in requests]
        req_num_generation_steps = torch.tensor(
            req_num_generation_steps_list, dtype=torch.int32, pin_memory=True
        )

        # context requests do not have multiple beams yet, so beam width may differ in mixed batches
        req_num_beams_list = [
            req.get_beam_width_by_iter(False) if not req.is_context_init_state else 1
            for req in requests
        ]
        req_num_beams = torch.tensor(req_num_beams_list, dtype=torch.int32, pin_memory=True)
        # context requests do not have multiple beams yet, so beam width may differ after sampling
        req_num_output_beams_list = [req.get_beam_width_by_iter(True) for req in requests]
        req_num_beams_output = torch.tensor(
            req_num_output_beams_list, dtype=torch.int32, pin_memory=True
        )

        req_num_generated_tokens = req_num_generation_steps * req_num_beams
        req_num_generated_tokens_output = req_num_generation_steps * req_num_beams_output
        # NB: These offsets consider generated tokens _only_ (draft and target, but not context).
        #     Filter out the context tokens below.
        req_offsets, sum_num_generated_tokens = _PackedStepIndexer.calculate_request_offsets(
            req_num_generated_tokens, pin_memory=True
        )

        generation_requests_total_steps = (
            # NB: requests == scheduled_requests.context_requests + scheduled_requests.generation_requests
            sum_num_generated_tokens
            - cast(int, req_offsets[len(scheduled_requests.context_requests)].item())
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

        num_logits_to_keep = sum_num_generated_tokens

        # raw_logits should contain only the generated logits.
        # If return context logits is requested, select only the generated logits.
        #
        # NB: Context request logits always precede generation request logits, also
        #     requests == scheduled_requests.context_requests + scheduled_requests.generation_requests
        if any(r.py_return_context_logits for r in scheduled_requests.context_requests):
            assert (
                len(num_context_logits_prefix_sum) == len(scheduled_requests.context_requests) + 1
            )
            req_num_generated_tokens_cuda = req_num_generated_tokens.to(
                raw_logits_cuda.device, non_blocking=True
            )
            context_req_offsets_cuda = torch.tensor(
                num_context_logits_prefix_sum, dtype=torch.int32, pin_memory=True
            ).to(device=raw_logits_cuda.device, non_blocking=True)

            if scheduled_requests.generation_requests:
                # Since the goal is to keep the req_num_steps[i] last tokens for each requests[i],
                # only end-offsets of the token storage locations matter.
                next_context_req_offsets_cuda = context_req_offsets_cuda.roll(
                    -1
                )  # trailing '0' is overwritten below
                # Since logits for generation requests are densely packed, cover them all by a single
                # fictituous entry in 'context_req_offsets_cuda'.
                req_num_steps_fictitious_cuda = req_num_generated_tokens_cuda[
                    : (len(scheduled_requests.context_requests) + 1)
                ].clone()
                req_num_steps_fictitious_cuda[-1].fill_(generation_requests_total_steps)
                next_context_req_offsets_cuda[-1].copy_(
                    next_context_req_offsets_cuda[-2] + req_num_steps_fictitious_cuda[-1],
                    non_blocking=True,
                )
            else:
                req_num_steps_fictitious_cuda = req_num_generated_tokens_cuda[
                    : len(scheduled_requests.context_requests)
                ]
                # Since the goal is to keep the req_num_steps[i] last tokens for each requests[i],
                # only end-offsets of the token storage locations matter.
                next_context_req_offsets_cuda = context_req_offsets_cuda[1:]

            # Now, the generated tokens for context request i are at indices
            #    range(next_context_req_offsets_cuda[i] - req_num_steps_fictitious_cuda[i],
            #          next_context_req_offsets_cuda[i])
            # And if generation requests are present, those tensors each include a trailing entry selecting
            # all tokens/logits generated by all generation requests.
            indices_to_keep_cuda = torch_multi_arange(
                starts=(next_context_req_offsets_cuda - req_num_steps_fictitious_cuda),
                ends=next_context_req_offsets_cuda,
                output_length=num_logits_to_keep,
            )

            raw_logits_cuda = raw_logits_cuda[indices_to_keep_cuda]

        logits_cuda = raw_logits_cuda[:num_logits_to_keep]

        return sampling_requests_metadata, logits_cuda

    @staticmethod
    def _longest_stop_word_len(requests: Iterable[LlmRequest]) -> int:
        max_stop_word_len = 0
        for req in requests:
            _, cumsum = req.py_stop_words_list
            if -1 in cumsum:
                cumsum = cumsum[: cumsum.index(-1)]
            request_max_stop_word_len = np.max(np.diff(cumsum, prepend=0), initial=0)
            max_stop_word_len = max(max_stop_word_len, request_max_stop_word_len)
        return max_stop_word_len

    @staticmethod
    def _requests_with_stop_words(requests: list[LlmRequest]) -> list[LlmRequest]:
        return [
            r
            for r in requests
            if (r.py_stop_words_list is not None and len(r.py_stop_words_list[0]) > 0)
        ]

    def _request_indices_with_stop_words(self, requests: list[LlmRequest]) -> torch.Tensor:
        return torch.tensor(
            [
                ridx
                for ridx, r in enumerate(requests)
                if (r.py_stop_words_list is not None and len(r.py_stop_words_list[0]) > 0)
            ],
            dtype=torch.int32,
            pin_memory=True,
        ).to(device="cuda", non_blocking=True)

    @nvtx_range("_write_finish_reasons")
    def _write_finish_reasons(
        self,
        requests: list[LlmRequest],
        *,
        finish_reasons: torch.Tensor,
        seq_slots: torch.Tensor,
        seq_lens: torch.Tensor,
        new_tokens: torch.Tensor,
        first_finish_reasons: torch.Tensor | None = None,
        predecessor_beams: torch.Tensor | None = None,
    ) -> None:
        """later end reason overwrites earlier, in reverse precedence order

        writes the finish reasons to the finish_reasons tensor.
        Args:
            requests: the requests to write the finish reasons to
            finish_reasons: the finish reasons tensor to write to. Shape: (max_tokens, max_batch_size, max_beam_width)
            seq_slots: the sequence slots of the processed requests. Used to determine where to
            read and write from the finish_reasons and new_tokens buffers. Shape: (len(requests),)
            new_tokens: a buffer containing the newly generated tokens.
            Shape: (max_tokens, max_batch_size, max_beam_width)
        """

        # Seq Slots should be on the same device as new_tokens
        assert seq_slots.device == new_tokens.device
        assert seq_lens.device == new_tokens.device
        tokens = new_tokens[:, seq_slots]

        # we need to fill with NOT_FINISHED so we can differentiate between previous requests that had the same seq slot
        finish_reasons.index_fill_(1, seq_slots, FinishReason.NOT_FINISHED.value)
        batched_finish_reasons = finish_reasons[:, seq_slots]

        if with_stop_words := self._requests_with_stop_words(requests):
            stop_seq_slots = torch.tensor(
                [r.py_seq_slot for r in with_stop_words], pin_memory=True
            ).to("cuda", non_blocking=True)
            stop_tokens = new_tokens[:, stop_seq_slots]
            stop_indices = self._request_indices_with_stop_words(requests)
            predecessor_beams_batched = (
                predecessor_beams
                if predecessor_beams is None
                else predecessor_beams[stop_seq_slots]
            )
            batched_finish_reasons[:, stop_indices] = torch.where(
                self._are_stop_words(
                    with_stop_words, stop_tokens, predecessor_beams=predecessor_beams_batched
                ),
                self._reason_tensors[FinishReason.STOP_WORDS],
                batched_finish_reasons[:, stop_indices],
            )

        batched_finish_reasons = torch.where(
            self._are_max_length(seq_lens, self.store.max_lengths_tensor[seq_slots]),
            self._reason_tensors[FinishReason.LENGTH],
            batched_finish_reasons,
        )
        batched_finish_reasons = torch.where(
            self._are_end_id(self.store.end_ids[seq_slots], tokens),
            self._reason_tensors[FinishReason.END_ID],
            batched_finish_reasons,
        )

        finish_reasons[:, seq_slots] = batched_finish_reasons
        if first_finish_reasons is not None:
            # store the first stop reason for each beam of a seq_slot.
            batched_first_finish_reasons = first_finish_reasons[seq_slots]
            batched_first_finish_reasons = torch.where(
                batched_first_finish_reasons == FinishReason.NOT_FINISHED.value,
                batched_finish_reasons,
                batched_first_finish_reasons,
            )
            first_finish_reasons[seq_slots] = batched_first_finish_reasons

    def _are_end_id(self, end_ids: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return tokens == end_ids.view(1, -1, 1).expand(self.max_tokens, -1, self.max_beam_width)

    def _are_max_length(self, seq_lens: torch.Tensor, max_seq_lens: torch.Tensor) -> torch.Tensor:
        """Checks which sequences are at or beyond the max length

        Args:
            seq_lens: the sequence lengths of the requests to check the max length of
            max_seq_lens: the maximum sequence lengths of the requests to check the max length of
        Returns:
            A tensor of shape (max_tokens, len(requests), max_beam_width)
            where each element is True if the sequence is at or beyond the max length, False otherwise
        """
        lengths_tensor = (seq_lens.view(1, -1, 1) + self._max_tokens_offset).expand(
            self.max_tokens, -1, self.max_beam_width
        )
        max_lengths_tensor = max_seq_lens.view(1, -1, 1).expand(
            self.max_tokens, -1, self.max_beam_width
        )
        return lengths_tensor >= max_lengths_tensor

    _PAD_ID = -1
    """Pad with negative, doesn't matter what"""

    @cached_property
    def _pad_steps_mask(self):
        square = torch.ones(self.max_tokens, self.max_tokens, dtype=torch.bool)
        pad_id = torch.tensor(self._PAD_ID)
        mask = torch.where(square.tril(), torch.tensor(1), pad_id)
        mask.pin_memory()
        return mask.to("cuda", non_blocking=True)

    def _padded_old_tokens(
        self,
        requests: list[LlmRequest],
        new_tokens: torch.Tensor,
        predecessor_beams: torch.Tensor | None = None,
        pad_id: int = _PAD_ID,
    ) -> torch.Tensor:
        # TODO: make sure only the lookback tokens are pulled into the list
        longest = self._longest_stop_word_len(requests)
        assert longest > 0, f"{longest=}, longest stop word length should be greater than 0"
        lookback = longest - 1
        old_tokens = []
        for request_idx, request in enumerate(requests):
            beam_width = request.sampling_config.beam_width
            old = [
                request.get_tokens(min(beam_idx, beam_width - 1))[-lookback:]
                if lookback > 0
                else []
                for beam_idx in range(self.max_beam_width)
            ]
            padded = [
                [pad_id]
                * max(
                    0,
                    lookback
                    - len(
                        old[
                            beam_idx
                            if predecessor_beams is None
                            else predecessor_beams[request_idx, beam_idx]
                        ]
                    ),
                )
                + old[
                    beam_idx
                    if predecessor_beams is None
                    else predecessor_beams[request_idx, beam_idx]
                ]
                for beam_idx in range(self.max_beam_width)
            ]
            old_tokens.append(padded)
        old_tokens_tensor = torch.tensor(old_tokens, pin_memory=True).to("cuda", non_blocking=True)
        assert old_tokens_tensor.shape == (
            len(requests),
            self.max_beam_width,
            lookback,
        ), f"{old_tokens_tensor.shape} != ({len(requests)=}, {self.max_beam_width=}, {lookback=})"
        new_tokens = new_tokens.permute(1, 2, 0)
        ret = torch.cat((old_tokens_tensor, new_tokens), dim=-1)
        assert ret.shape == (
            len(requests),
            self.max_beam_width,
            lookback + self.max_tokens,
        ), (
            f"{ret.shape} != ({len(requests)=}, {self.max_beam_width=}, {lookback + self.max_tokens=})"
        )
        return ret

    def _are_stop_words(
        self,
        requests: list[LlmRequest],
        tokens: torch.Tensor,
        predecessor_beams: torch.Tensor | None = None,
    ) -> torch.Tensor:
        per_step = torch.zeros(
            (self.max_tokens, len(requests), self.max_beam_width), dtype=torch.bool, pin_memory=True
        ).to("cuda", non_blocking=True)

        padded_tokens = self._padded_old_tokens(requests, tokens, predecessor_beams)

        for request_idx, request in enumerate(requests):
            swl, ends = request.py_stop_words_list
            if -1 in ends:
                ends = ends[: ends.index(-1)]
            lens = np.diff(ends, prepend=0)
            max_len = np.max(lens)

            words = torch.zeros(len(lens), max_len, dtype=torch.int32, pin_memory=True)
            for step, (start, length) in enumerate(zip([0] + ends, lens)):
                words[step, :length] = torch.tensor(swl[start : start + length], dtype=torch.int32)
            words_device = words.to("cuda", non_blocking=True)

            draft_token_length = get_draft_token_length(request)

            for beam_idx in range(self.max_beam_width):
                new_tokens = padded_tokens[request_idx, beam_idx]
                for step_idx in range(draft_token_length + 1):
                    size_per_step = new_tokens.size(0) - draft_token_length + step_idx
                    matches = []
                    for word, L in zip(words_device, lens):
                        truncated_seq = new_tokens[size_per_step - L : size_per_step]
                        match = (truncated_seq == word[:L]).all()
                        matches.append(match)
                    per_step[step_idx, request_idx, beam_idx] = torch.stack(matches).any()

        return per_step

    @nvtx_range("_process_logprobs")
    def _process_logprobs(
        self,
        batched_sampling_result: _BatchedSamplingResult,
        seq_slots: torch.Tensor,
        requests: list[LlmRequest],
        req_num_steps: torch.Tensor,
        req_num_generated_tokens: torch.Tensor,
    ):
        assert batched_sampling_result.batch_logits_for_logprobs_cuda is not None, (
            "batch_logits_for_logprobs_cuda must be a Tensor for _process_logprobs"
        )

        all_req_indices = batched_sampling_result.batch_req_indices.tolist()
        # The request indices in the shuffled batch after grouping (NB: Beam search request are handled separately)
        local_group_req_indices = torch.tensor(
            [
                req_id
                for req_id, req_gid in enumerate(all_req_indices)
                if requests[req_gid].py_num_logprobs is not None
                and requests[req_gid].sampling_config.beam_width == 1
            ],
            dtype=torch.int32,
        )
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

        any_request_without_beam_search = local_group_req_indices.shape[0] > 0

        if any_request_without_beam_search:
            assert self.store.sampled_log_probs is not None, "sampled_log_probs must be provided"
            assert self.store.sampled_log_prob_indices is not None, (
                "sampled_log_prob_indices must be provided"
            )
            assert self.store.sampled_log_prob_ranks is not None, (
                "sampled_log_prob_ranks must be provided"
            )
            # NB: Already begin copy here, to overlap with the remaining host code
            padded_indices_cuda = padded_indexer[local_group_req_indices].to(
                device=self.store.sampled_log_probs.device, non_blocking=True
            )

            # get indices of the logits after grouping
            group_logits_indices_cuda = logits_cuda_indexer[local_group_req_indices].to(
                device=batched_sampling_result.batch_logits_for_logprobs_cuda.device,
                non_blocking=True,
            )

            # (batch_size, vocab_size)
            group_logprobs_cuda = F.log_softmax(
                batched_sampling_result.batch_logits_for_logprobs_cuda[group_logits_indices_cuda],
                dim=-1,
            )

            # Process the topk logprobs
            if self.batch_max_topk_logprobs > 0:
                assert self.store.topk_vals is not None, "topk_vals must be provided"
                assert self.store.topk_indices is not None, "topk_indices must be provided"
                # Get the topk logprobs
                # The request indices in the batch before grouping
                group_req_indices = batched_sampling_result.batch_req_indices[
                    local_group_req_indices
                ]
                topk_vals_cuda, topk_indices_cuda = torch.topk(
                    group_logprobs_cuda,
                    k=max(requests[req_id].py_num_logprobs for req_id in group_req_indices),
                    dim=-1,
                )
                expanded_indices_cuda = padded_indices_cuda.view(-1, 1).expand(
                    -1, topk_vals_cuda.shape[-1]
                )
                self.store.topk_vals[..., : self.batch_max_topk_logprobs].view(
                    self.max_num_sequences * self.max_tokens, self.batch_max_topk_logprobs
                ).scatter_(dim=0, index=expanded_indices_cuda, src=topk_vals_cuda)
                self.store.topk_indices[..., : self.batch_max_topk_logprobs].view(
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

            # NB: group_logprobs_cuda is not needed anymore and the storage can be safely reused.
            # sampled_rank_cuda contains the 0-based rank, it will be corrected to 1-based in handle_logprobs
            group_logprobs_cuda.greater_(sampled_vals_cuda)
            sampled_rank_cuda = group_logprobs_cuda.sum(dim=-1).to(torch.int32)

            sampled_vals_cuda = sampled_vals_cuda.squeeze(1)

            self.store.sampled_log_prob_indices.view(
                self.max_num_sequences * self.max_tokens * self.max_beam_width
            ).scatter_(dim=0, index=padded_indices_cuda, src=sampled_indices_cuda)
            self.store.sampled_log_probs.view(
                self.max_num_sequences * self.max_tokens * self.max_beam_width
            ).scatter_(dim=0, index=padded_indices_cuda, src=sampled_vals_cuda)
            self.store.sampled_log_prob_ranks.view(
                self.max_num_sequences * self.max_tokens * self.max_beam_width
            ).scatter_(dim=0, index=padded_indices_cuda, src=sampled_rank_cuda)

        if self._use_beam_search:
            local_group_req_indices_with_beam_search = torch.tensor(
                [
                    req_id
                    for req_id, req_gid in enumerate(all_req_indices)
                    if requests[req_gid].py_num_logprobs is not None
                    and requests[req_gid].sampling_config.beam_width > 1
                ],
                dtype=torch.int32,
            )
            any_request_has_beam_search = local_group_req_indices_with_beam_search.shape[0] > 0
            if any_request_has_beam_search:
                group_logits_indices_with_beam_search = logits_cuda_indexer[
                    local_group_req_indices_with_beam_search
                ]
                group_logits_indices_with_beam_search_cuda = (
                    group_logits_indices_with_beam_search.to(
                        device=batched_sampling_result.batch_next_tokens_cuda_int.device,
                        non_blocking=True,
                    )
                )
                group_next_tokens_with_beam_search_cuda = (
                    batched_sampling_result.batch_next_tokens_cuda_int[
                        group_logits_indices_with_beam_search_cuda
                    ].view(-1)
                )
                padded_indices_with_beam_search_cuda = padded_indexer[
                    local_group_req_indices_with_beam_search
                ].to(device=self.store.sampled_log_prob_indices.device, non_blocking=True)
                self.store.sampled_log_prob_indices.view(-1).scatter_(
                    dim=0,
                    index=padded_indices_with_beam_search_cuda,
                    src=group_next_tokens_with_beam_search_cuda,
                )

    @nvtx_range("_process_requests")
    def _process_requests(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, torch.Tensor],
        new_tokens_cuda: torch.Tensor,
        num_context_logits_prefix_sum: list[int],
        *,
        seq_slots: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_slots = seq_slots.to(dtype=torch.int32)  # int32 suffices here

        raw_logits_cuda = model_outputs["logits"]

        requests = scheduled_requests.all_requests()
        cuda_device = raw_logits_cuda.device

        sampling_requests_metadata, logits_cuda = self._select_generated_logits(
            scheduled_requests,
            raw_logits_cuda,
            num_context_logits_prefix_sum=num_context_logits_prefix_sum,
        )
        return_log_probs = self._return_log_probs(requests)
        if return_log_probs:
            self._prepare_log_probs(requests)

        # Handle embedding bias
        self._apply_embedding_bias(logits_cuda, requests, sampling_requests_metadata.req_num_steps)

        logits_cuda = self._apply_min_length_penalty(
            logits_cuda,
            requests,
            sampling_requests_metadata.req_num_steps,
            sampling_requests_metadata.req_num_beams,
        )

        # Fast path for greedy sampling
        if self._can_use_fast_greedy_path(requests):
            # Compute destination indices on CPU (same pattern as _unbatch_sampling_results)
            batch_destination_indexer = _UnpackedStepIndexer(
                seq_slots=seq_slots,
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
            return new_tokens_host

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
            requests,
            model_outputs,
            cuda_device=cuda_device,
            logits_cuda_indexer=logits_cuda_indexer,
            req_offsets=sampling_requests_metadata.req_offsets,
            seq_slots=seq_slots,
            seq_lens=seq_lens,
            req_num_generated_tokens=sampling_requests_metadata.req_num_generated_tokens,
            req_num_steps=sampling_requests_metadata.req_num_steps,
            token_dtype=new_tokens_cuda.dtype,
            return_log_probs=return_log_probs,
        )

        if return_log_probs:
            self._process_logprobs(
                batched_sampling_result,
                seq_slots,
                requests,
                sampling_requests_metadata.req_num_steps,
                sampling_requests_metadata.req_num_generated_tokens_output,
            )

        # Fill results into output buffers
        new_tokens_host = self._unbatch_sampling_results(
            batched_sampling_result,
            new_tokens_cuda=new_tokens_cuda,
            req_num_generated_tokens=sampling_requests_metadata.req_num_generated_tokens,
            seq_slots=seq_slots,
        )

        # NB: update_requests syncs w/ device computation and async D2H copies
        return new_tokens_host

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
    def defined_algorithms(self):
        return [attr for attr in dir(self) if not attr.startswith("__")]

    def __repr__(self):
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
class SampleStateTRTLLM(SampleState):
    finalize_events: dict[str, CudaEvent] | None = None
    """`Optional` to accommodate `_forward_step_inter_pp` which creates a `SampleState` without `finalize_events`"""
    host: Optional[SampleStateTensorsHostTRTLLM] = None


class TRTLLMSampler(Sampler, AsyncWorkerMixin):
    MAX_DECODING_TOKENS = 1  # It must be 1 when not in speculative decoding
    SampleState = SampleStateTRTLLM

    @override
    def is_generation_model(self) -> bool:
        return True

    def __init__(
        self,
        model,
        model_dtype,
        mapping: Mapping,
        decoding_mode: DecodingMode,
        disable_overlap_scheduler: bool,
        max_seq_len: int,
        max_batch_size: int,
        max_beam_width: int,
        decoding_config: Optional[DecodingConfig] = None,
        kv_cache_config: Optional[KvCacheConfig] = None,
        enable_async_worker: bool = False,
    ):
        vocab_size = model.config.vocab_size
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads

        self.model_datatype = torch_dtype_to_binding(model_dtype)
        self.logits_datatype = DataType.FLOAT
        self.decoding_mode = decoding_mode
        self.decoding_config = decoding_config if decoding_config else DecodingConfig(decoding_mode)
        max_attn_window = kv_cache_config.max_attention_window
        self.max_seq_len = max_seq_len
        self.max_attention_window = (
            max(max_attn_window) if max_attn_window is not None else max_seq_len
        )
        self.max_batch_size = max_batch_size
        self.max_beam_width = max_beam_width
        self.max_num_sequences = mapping.pp_size * max_batch_size
        self.max_seq_idle_microseconds = 180 * 1000 * 1000
        self.is_trt_overlap = not disable_overlap_scheduler
        self.num_micro_batches = (
            mapping.pp_size if mapping.pp_size > 1 else (2 if self.is_trt_overlap else 1)
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

    def _initialize_store(self):
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

        self.store["decoder_state"].setup(
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

    def _instantiate_algorithms(self):
        self.algs = Algorithms()
        self.algs.decoder = GptDecoderBatched(stream=self.store["torch_stream"])
        self.algs.decoder.setup(
            mode=self.decoding_mode,
            max_num_sequences=self.max_num_sequences,
            max_beam_width=self.max_beam_width,
            dtype=self.logits_datatype,
            model_config=self.model_config,
            world_config=self.world_config,
        )
        self.algs.create_new_decoder_requests = CreateNewDecoderRequests(
            speculative_decoding_fast_logits=False,
            is_leader_in_orch_mode=False,
            is_normalize_log_probs=False,
        )
        self.algs.make_decoding_batch_input_output = MakeDecodingBatchInputOutput()

    @torch.inference_mode()
    @nvtx_range("setup_sampler_step")
    def setup_sampler_step(self, requests):
        batch_slots, sampling_configs, lookahead_prompt, lookahead_algo_configs = (
            self.algs.create_new_decoder_requests(
                self.model_config,
                self.world_config,
                self.decoding_config,
                requests.context_requests,
                self.logits_datatype,
                self.store["decoder_input_buffers"][self.micro_batch_idx],
                self.store["decoder_state"],
                self.store["cuda_stream"],
                self.algs.decoder.decoder_stream,
                self.max_seq_len,
                self.beam_width(requests.context_requests),
            )
        )

        local_batch_size = len(batch_slots)
        if local_batch_size > 0:
            sampling_config = make_sampling_config(sampling_configs)
            self.algs.decoder.underlying_decoder().setup(
                sampling_config,
                local_batch_size,
                batch_slots,
                self.store["decoder_state"].joint_decoding_output,
                self.model_config.data_type,
                lookahead_prompt,
                lookahead_algo_configs,
            )

        adp = [r for r in requests.generation_requests if r.is_attention_dp_dummy]
        batch_size = len(adp)
        if batch_size == 0:
            return
        config = make_sampling_config([r.sampling_config for r in adp])
        slots = torch.tensor([r.py_seq_slot for r in adp], dtype=torch.int32)
        self.algs.decoder.underlying_decoder().setup(config, batch_size, slots)

    def get_cache_indirection(self) -> torch.Tensor | None:
        return self.store["decoder_state"].cache_indirection_output

    def _update_cache_indirection_buffer(self, scheduled_requests: ScheduledRequests):
        # Copy cache indirection output to input
        for request in scheduled_requests.generation_requests:
            self.store["decoder_state"].cache_indirection_input[request.py_seq_slot].copy_(
                self.store["decoder_state"].cache_indirection_output[request.py_seq_slot],
                non_blocking=True,
            )

    @torch.inference_mode()
    @nvtx_range("sample_async")
    @override
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs,
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleStateTRTLLM:
        batch_size = scheduled_requests.batch_size
        beam_width = self.beam_width(scheduled_requests.all_requests())
        if (
            batch_size > 1
            and beam_width > 1
            and any(request.py_return_log_probs for request in scheduled_requests.all_requests())
        ):
            raise ValueError("Beam search is not supported for multiple prompts and logprobs")

        self.setup_sampler_step(scheduled_requests)

        # For beam search, cache indirection needs to be updated
        if beam_width > 1:
            self._update_cache_indirection_buffer(scheduled_requests)

        make_decoding_batch_input(
            self.store["decoder_input_buffers"][self.micro_batch_idx],
            self.store["decoder_state"],
            scheduled_requests.context_requests,
            scheduled_requests.generation_requests,
            model_outputs["logits"],
            beam_width,
            num_context_logits_prefix_sum,
            self.store["buffer_manager"],
        )

        self.algs.decoder.forward_async(
            self.store["decoder_state"],
            self.store["decoder_input_buffers"][self.micro_batch_idx],
        )

        finalize_events = {}
        gathered_ids = None
        if beam_width > 1:
            finished_sum_device = self.store["decoder_state"].finished_sum

            for request in scheduled_requests.all_requests():
                if request.is_context_init_state:
                    continue
                if finished_sum_device[request.seq_slot] == beam_width:
                    finalize_events[request.request_id] = self._finalize_request(request, False)
                elif request.streaming:
                    finalize_events[request.request_id] = self._finalize_request(request, True)
            gathered_ids = self._copy_to_host(self.store["decoder_state"].gathered_ids)
        new_output_tokens = self._copy_to_host(self.store["decoder_state"].all_new_tokens)
        finished_sum = self._copy_to_host(self.store["decoder_state"].finished_sum)
        finish_reasons = self._copy_to_host(self.store["decoder_state"].finish_reasons)
        sequence_lengths = self._copy_to_host(self.store["decoder_state"].sequence_lengths)

        log_probs = None
        cum_log_probs = None
        if any(request.py_return_log_probs for request in scheduled_requests.all_requests()):
            log_probs = self._copy_to_host(self.store["decoder_state"].log_probs)
            cum_log_probs = self._copy_to_host(self.store["decoder_state"].cum_log_probs)

        device = SampleStateTensors(new_tokens=self.store["decoder_state"].all_new_tokens)

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
            scheduled_requests=scheduled_requests,
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
    ):
        # resource_manager will not be used in this function, just for interface consistency.
        assert isinstance(state, SampleStateTRTLLM)
        if state.scheduled_requests.batch_size == 0:
            return

        if state.sampler_event:
            state.sampler_event.synchronize()

        beam_width = self.beam_width(state.scheduled_requests.all_requests())

        if beam_width == 1 and self.MAX_DECODING_TOKENS == 1:
            self.update_requests_single_beam_single_step(state)
        else:
            self.update_requests_multiple_beams_or_drafting(state, beam_width)

    @torch.inference_mode()
    @nvtx_range("update_requests_single_beam_single_step")
    def update_requests_single_beam_single_step(self, state: SampleStateTRTLLM):
        """Specialization of update_requests for single beam and single step"""
        sequence_lengths_host_data = state.host.sequence_lengths.flatten().tolist()
        finish_reasons = state.host.finish_reasons.flatten().tolist()

        reqs = [
            r for r in state.scheduled_requests.context_requests if not r.is_context_init_state
        ] + [
            r
            for r in state.scheduled_requests.generation_requests
            if not r.is_generation_complete_state
        ]

        # NB: To ensure good performance, we must
        #  1. Avoid accessing torch.Tensor object inside the for-each-request loops
        #  2. Convert only necessary data to Python list

        # Add new tokens
        reqs_with_new_tokens = []
        seq_slots = []
        seq_slots_need_log_probs = []
        for request in reqs:
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
        if state.host.log_probs is not None:
            # [batchSize, maxBeamWidth]
            seq_last_idx = state.host.sequence_lengths[seq_slots_need_log_probs, 0] - 1
            # [batchSize, maxBeamWidth, maxSequenceLength]
            log_probs_host = state.host.log_probs[
                seq_slots_need_log_probs, 0, seq_last_idx
            ].tolist()
            # [batchSize, maxBeamWidth]
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
    ):
        new_tokens_host = state.host.new_tokens.tolist()
        finished_sum_host = state.host.finished_sum.tolist()
        finish_reasons = state.host.finish_reasons.flatten().tolist()
        sequence_lengths_host_data = state.host.sequence_lengths.flatten().tolist()
        cum_log_probs_host = (
            state.host.cum_log_probs.tolist() if state.host.cum_log_probs is not None else None
        )
        log_probs_host = state.host.log_probs.tolist() if state.host.log_probs is not None else None
        finalize_events = state.finalize_events

        reqs = [
            r for r in state.scheduled_requests.context_requests if not r.is_context_init_state
        ] + [
            r
            for r in state.scheduled_requests.generation_requests
            if not r.is_generation_complete_state
        ]

        for request in reqs:
            seq_slot = request.py_seq_slot
            num_generated_tokens = request.num_draft_tokens + 1
            current_num_of_tokens = request.max_beam_num_tokens
            num_new_tokens = [0] * beam_width

            log_probs = [[] for _ in range(beam_width)]
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
                        # NOTE: Log probs with drafting has not been tested yet.
                        begin_log_probs_offset = (
                            request.prompt_len if request.sampling_config.beam_width == 1 else 0
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
    ):
        """Finalizes the request. This is necessary for beam search."""
        seq_slot = request.py_seq_slot
        event = self.algs.decoder.finalize(
            self.store["decoder_state"], seq_slot, request.sampling_config, streaming
        )
        return event

    def _post_process_request(self, request: LlmRequest, state: SampleStateTRTLLM):
        """Post Process the request. Updates the sequence according to the beam search results.
        request: LlmRequest which shall be post processed
        finalize_event: CudaEvent to wait for the finalize step to finish
        """
        seq_slot = request.py_seq_slot
        beam_width = request.sampling_config.beam_width
        # synchronize on the finalize event before continuing the post processing.
        # should be unnecessary, as already wait for the sampler event in update_requests
        state.finalize_events[request.request_id].synchronize()

        # Get these values again, as they might have changed during the finalize step
        output_ids_host = state.host.gathered_ids
        sequence_lengths_host = state.host.sequence_lengths

        if request.py_return_log_probs:
            log_probs_host = state.host.log_probs
            cum_log_probs_host = state.host.cum_log_probs

        generated_tokens = [[0]] * beam_width
        log_probs = [[] for _ in range(beam_width)]
        cum_log_probs = []

        for beam_idx in range(beam_width):
            # get the correct generated tokens for beam search
            begin = request.py_prompt_len
            generated_length = (
                sequence_lengths_host[seq_slot, beam_idx].item() - request.py_prompt_len
            )
            end = begin + generated_length
            generated_tokens[beam_idx] = output_ids_host[seq_slot, beam_idx, begin:end].tolist()

            # get the correct log probs for beam search
            if request.py_return_log_probs:
                cum_log_probs.append(cum_log_probs_host[seq_slot, beam_idx].item())

                begin_log_probs_offset = (
                    request.prompt_len if request.sampling_config.beam_width == 1 else 0
                )
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
