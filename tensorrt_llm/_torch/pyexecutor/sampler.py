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
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from itertools import repeat
from typing import Any, Callable, List, Optional, Type, TypeVar, cast

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
from tensorrt_llm.sampling_params import SamplingParams

from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..speculative.spec_tree_manager import SpecTreeManager
from .finish_reason import FinishedState
from .llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from .resource_manager import ResourceManager, ResourceManagerType
from .sampling_utils import (
    GREEDY,
    GenericStrategyKeyType,
    GroupedStrategySampler,
    SimpleGroupedStrategySampler,
    Strategy,
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
class SampleStateTensors:
    new_tokens: torch.Tensor
    log_probs: torch.Tensor | None = None

    def values(self):
        return vars(self).values()


@dataclass(kw_only=True)
class SampleState:
    scheduled_requests: ScheduledRequests

    device: Optional[SampleStateTensors] = None
    host: Optional[SampleStateTensors] = None

    sampler_event: Optional[torch.cuda.Event] = None


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

    def values(self):
        return vars(self).values()


@dataclass(kw_only=True)
class SampleStateWithMMResult:
    scheduled_requests: ScheduledRequests

    data: MultimodalResult


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
        data = MultimodalResult(mm_embeddings=model_outputs["mm_embeddings"])
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
        for request, mm_embedding in zip(scheduled_requests.context_requests, mm_embeddings):
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)
            if len(mm_embedding) != sum(request.multimodal_lengths):
                raise ValueError(
                    f"mm_embedding shape mismatch: {len(mm_embedding)} != {sum(request.multimodal_lengths)}"
                )

            request.py_result.append_mm_embeddings(mm_embedding)

    @override
    def is_generation_model(self) -> bool:
        return False


# Due to tensorrt_llm::runtime::SamplingConfig using vectors, params
# in LlmRequest.sampling_params are either None or single-element lists.
# This helper method simplifies code using such params.
def _unwrap_singleton(p: Optional[list[T]]) -> Optional[T]:
    if p is None:
        return None
    (t,) = p
    return t


def _request_get_sampling_params(request: LlmRequest) -> UtilsSamplingParams:
    sampling_config = request.sampling_config
    temperature = _unwrap_singleton(cast(Optional[list[float]], sampling_config.temperature))
    top_p = _unwrap_singleton(cast(Optional[list[float]], sampling_config.top_p))
    top_k = _unwrap_singleton(cast(Optional[list[int]], sampling_config.top_k))

    return UtilsSamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def _request_strategy(request: LlmRequest, *, vocab_size: int) -> Strategy:
    params = _request_get_sampling_params(request)
    return resolve_sampling_strategy(params, vocab_size=vocab_size)


def _group_requests_by_strategy_key(
    requests: Iterable[LlmRequest],
    *,
    strategy_to_key: Callable[[Strategy, bool], GenericStrategyKeyType],
    pin_memory: bool = False,
    vocab_size: int,
) -> dict[tuple[GenericStrategyKeyType, bool], tuple[torch.Tensor, List[Strategy]]]:
    # NB: Client code relies on request indices in returned torch.Tensor being sorted.
    group_dict: dict[tuple[GenericStrategyKeyType, bool], tuple[list[int], list[Strategy]]] = (
        defaultdict(lambda: ([], []))
    )

    for req_index, req in enumerate(requests):
        strategy = _request_strategy(req, vocab_size=vocab_size)
        speculation_needs_probs = (
            # NB: This criterion needs to be consistent with the gating of rejection sampling in
            #     process_draft_tokens.
            TorchSampler._speculation_could_use_rejection_sampling(req, strategy)
        )
        strategy_key = strategy_to_key(strategy, speculation_needs_probs)
        group_dict_entry = group_dict[(strategy_key, speculation_needs_probs)]
        group_dict_entry[0].append(req_index)
        group_dict_entry[1].append(strategy)
    return {
        group_key: (
            torch.tensor(indices, pin_memory=pin_memory, dtype=torch.int32),
            strategies,
        )
        for group_key, (indices, strategies) in group_dict.items()
    }


def add_token(
    request: LlmRequest, new_tokens: list[list[list[int]]], *, beam: int, step: int = 0
) -> int:
    # NB: Accessing nested lists faster than torch.Tensor or numpy.ndarray
    seq_slot = request.py_seq_slot
    assert seq_slot is not None
    new_token = new_tokens[step][seq_slot][beam]
    request.add_new_token(new_token, beam)
    return new_token


def int_tensor(shape: tuple[int, ...], device: str = "cuda") -> torch.Tensor:
    return torch.empty(shape, dtype=torch.int, device=device)


@dataclass(kw_only=True, frozen=True)
class _BatchedSamplingResult:
    # Original request indices for all requests (permuted due to batching by strategy):
    batch_req_indices: torch.Tensor
    # Next tokens for all requests:
    batch_next_tokens_cuda_int: torch.Tensor


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


BEAM = 0
MAX_BEAM_WIDTH = BEAM + 1

FinishReasonsList = list[list[int]]


@dataclass(kw_only=True)
class SampleStateTensorsHostTorch(SampleStateTensors):
    finish_reasons: torch.Tensor

    def finish_reasons_list(self) -> FinishReasonsList:
        """`(num_seq_slots, num_steps)`"""
        return self.finish_reasons[:, :, BEAM].T.tolist()


@dataclass(kw_only=True)
class SampleStateTorch(SampleState):
    host: SampleStateTensorsHostTorch


class TorchSampler(Sampler):
    SampleState = SampleStateTorch

    @override
    def is_generation_model(self) -> bool:
        return True

    @dataclass(frozen=True, kw_only=True)
    class Store:
        new_tokens: torch.Tensor
        """Shape: See cpp DecoderState.getAllNewTokens()"""
        finish_reasons: torch.Tensor

        def __post_init__(self):
            assert self.new_tokens.shape == self.finish_reasons.shape

    @dataclass(frozen=True, kw_only=True)
    class Args:
        max_seq_len: int
        max_draft_len: int
        max_num_sequences: int
        max_beam_width: int
        max_total_draft_tokens: int
        disable_flash_infer_sampling: bool = False

    def __init__(self, args: Args):
        self.max_seq_len = args.max_seq_len
        self.max_tokens = args.max_total_draft_tokens + 1
        assert args.max_beam_width == MAX_BEAM_WIDTH, "TorchSampler only supports beam_width = 1"
        self.max_num_sequences = args.max_num_sequences

        # AutoDeploy build creates the sampler in inference mode,
        # which would disallow in-place mutating of new_tokens.
        # So, we temporarily exit inference mode.
        with torch.inference_mode(False):
            self.store = self.Store(
                new_tokens=int_tensor((self.max_tokens, self.max_num_sequences, MAX_BEAM_WIDTH)),
                finish_reasons=int_tensor(
                    (self.max_tokens, self.max_num_sequences, MAX_BEAM_WIDTH)
                ),
            )
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

        self._grouped_sampler_cls: Type[GroupedStrategySampler]
        if IS_FLASHINFER_AVAILABLE and not args.disable_flash_infer_sampling:
            from .sampling_utils_flashinfer import FlashInferGroupedStrategySampler

            self._grouped_sampler_cls = FlashInferGroupedStrategySampler
        else:
            self._grouped_sampler_cls = SimpleGroupedStrategySampler

        # Initialize seed for multi-GPU consistency
        self._global_seed = 42
        self._generator = None

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

    @staticmethod
    def _meet_max_token_stop_criteria(request: LlmRequest, max_seq_len: int):
        num_tokens = request.get_num_tokens(BEAM)
        return (num_tokens - request.py_orig_prompt_len >= request.py_max_new_tokens) or (
            num_tokens >= max_seq_len
        )

    @staticmethod
    def _meet_stop_token_criteria(request: LlmRequest):
        if request.py_stop_words_list:
            assert isinstance(request.py_stop_words_list, list), (
                "request.py_stop_words_list should be a list"
            )
            stop_words_list, prefix_sum = request.py_stop_words_list
            tokens = request.get_tokens(0)
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
        cls, request: LlmRequest, new_token: int, *, max_seq_len: int
    ) -> bool:
        """Handle stop criteria and set appropriate finish reasons and state.
        Returns True if generation should stop."""
        if new_token == request.py_end_id:
            request.finish_by(FinishReason.END_ID, BEAM)
            return True

        if cls._meet_max_token_stop_criteria(request, max_seq_len):
            request.finish_by(FinishReason.LENGTH, BEAM)
            return True

        if cls._meet_stop_token_criteria(request):
            request.finish_by(FinishReason.STOP_WORDS, BEAM)
            return True

        return False

    def handle_logprobs(
        self,
        request: LlmRequest,
        state: SampleState,
        *,
        beam: int,
        count: int,
    ):
        if request.py_return_log_probs:
            topk_log_probs_vals = request.py_topk_logprobs_vals[:count]
            topk_log_probs_indices = request.py_topk_logprobs_indices[:count]

            token_log_probs = [
                {
                    token: Logprob(logprob=logprob, rank=rank + 1)
                    for rank, (token, logprob) in enumerate(
                        zip(topk_token.tolist(), topk_logprob.tolist())
                    )
                }
                for topk_token, topk_logprob in zip(topk_log_probs_indices, topk_log_probs_vals)
            ]
            assert beam == 0, (
                "The following call relies on beam_width to be 1 - hence the list with a single element"
            )
            request.py_result.append_log_probs([token_log_probs])

    def finish_if_reason(
        self, request: LlmRequest, finish_reasons: FinishReasonsList, *, step: int
    ) -> bool:
        reason = FinishReason(finish_reasons[request.py_seq_slot][step])
        valid_reasons = {FinishReason.END_ID, FinishReason.LENGTH, FinishReason.STOP_WORDS}
        if reason in valid_reasons:
            request.finish_by(reason, BEAM)
            return True
        return False

    def _process_draft_tokens_greedy(
        self,
        request: LlmRequest,
        new_tokens: list[list[list[int]]],
        finish_reasons: FinishReasonsList,
    ) -> int:
        new_token = add_token(request, new_tokens, beam=BEAM)
        stop = self.finish_if_reason(request, finish_reasons, step=0)
        if stop or get_draft_token_length(request) == 0:
            return 0
        num_accepted = 0

        for draft_token in request.py_draft_tokens:
            if draft_token != new_token:
                # Reject.
                break

            num_accepted += 1
            new_token = add_token(request, new_tokens, beam=BEAM, step=num_accepted)
            if self.finish_if_reason(request, finish_reasons, step=num_accepted):
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
            new_tokens: torch.Tensor. [max_total_draft_tokens + 1, max_num_sequences, MAX_BEAM_WIDTH], host buffer.
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
        # handle the drafter model request
        # For the drafter model, we do not execute the tree verification logic,
        # but only add the draft tokens of the previous layer.
        if get_draft_token_length(request) == 0:
            cur_draft_layer_idx = spec_tree_manager.cur_draft_layer_idx

            # TODO: For the last layer of the dynamic tree, we need to resampling all the draft tokens.
            cur_layer_num_nodes = sum(spec_tree_manager.get_top_k_list(cur_draft_layer_idx))
            for i in range(cur_layer_num_nodes):
                add_token(request, new_tokens_list, beam=0, step=i)
            return 0
        else:
            # handle the target model request
            # For the target model, we will do the tree verification logic.
            seq_slot = request.py_seq_slot
            assert seq_slot is not None
            eagle_paths = spec_tree_manager.get_eagle_paths(seq_slot)

            all_draft_tokens = request.py_draft_tokens  # [max_total_draft_tokens]
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

            if longest_accepted_len == 0:
                # No draft tokens are accepted.
                # Take the top-1 token of the first layer as the next new token.
                add_token(request, new_tokens_list, beam=0, step=0)
                return 0
            else:
                # Take the longest accepted path as the next new token.
                num_accepted_draft_tokens = 0
                for idx in eagle_paths[longest_match_path_idx][:longest_accepted_len]:
                    add_token(request, new_tokens_list, beam=0, step=cast(int, idx.item()))
                    num_accepted_draft_tokens += 1
                    if self.finish_if_reason(
                        request, finish_reasons, step=num_accepted_draft_tokens
                    ):
                        break

                return num_accepted_draft_tokens - 1

    def _tree_sampling_batch(
        self,
        requests: list[LlmRequest],
        max_num_sequences: int,
        seq_slots: torch.Tensor,
        model_outputs: dict[str, torch.Tensor],
        spec_tree_manager: SpecTreeManager,
    ):
        if (
            spec_tree_manager.use_dynamic_tree
            # FIXME: 'draft_layer_id' is undefined
            and draft_layer_id == spec_tree_manager.max_draft_len - 1  # noqa: F821
        ):
            # TODO: Re-sample the draft tokens for the last layer.
            raise NotImplementedError("Dynamic tree is not fully supported yet.")

        raw_logits = model_outputs["logits"]
        num_requests = len(requests)
        assert raw_logits.shape[0] % num_requests == 0
        num_logits_per_request = raw_logits.shape[0] // num_requests
        request_index = torch.arange(num_requests)

        draft_layer_id = spec_tree_manager.cur_draft_layer_idx
        # 1) Get the topK list for the specific draft layer.
        top_k_list = spec_tree_manager.get_top_k_list(draft_layer_id)
        assert len(top_k_list) == num_logits_per_request

        # Considering that the beam_width of spec-dec can only be 1, we ignore this dimension here.
        new_draft_tokens_cuda = torch.empty(
            (max_num_sequences, spec_tree_manager.max_total_draft_tokens + 1),
            dtype=torch.int64,
            device=raw_logits.device,
        )

        top_k_list_cumsum = torch.cumsum(top_k_list, dim=0)
        # Different nodes have different topK value.
        for i, top_k_list_i in enumerate(top_k_list):
            # 2) Extract the logits needed for this layer.
            logits = raw_logits[request_index * num_logits_per_request + i, :]
            assert logits.shape[0] == len(requests)
            # 3) Sample the logits according to the topK value.
            indices = torch.topk(logits, k=top_k_list_i, dim=-1).indices
            # 4) Write to the temporary output tensor.
            new_draft_tokens_cuda[
                seq_slots, top_k_list_cumsum[i] - top_k_list_i : top_k_list_cumsum[i]
            ] = indices[request_index]

        # 5) Append eagle3 d2t.
        self._apply_d2t(new_draft_tokens_cuda, model_outputs)

        # 6) Copy back to the output tensor.
        int_new_draft_tokens = (
            new_draft_tokens_cuda.transpose(0, 1).to(torch.int, non_blocking=True).unsqueeze(dim=-1)
        )

        new_draft_tokens_host = int_new_draft_tokens.to("cpu", non_blocking=True)

        return new_draft_tokens_host

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
        _, draft_probs = sample(
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
            new_tokens_tensor[i, request.seq_slot, BEAM] = new_token
            request.add_new_token(new_token, BEAM)
            if self._handle_stop_criteria(request, new_token, max_seq_len=self.max_seq_len):
                num_accepted = i + 1
                return num_accepted
        if sample_last:
            new_token = sample_rejected(draft_probs, target_probs, generator, num_accepted)
            new_tokens_tensor[num_accepted, request.seq_slot, BEAM] = new_token
            request.add_new_token(new_token, BEAM)
        else:
            new_token = add_token(request, new_tokens_list, beam=BEAM, step=num_accepted)
        self._handle_stop_criteria(request, new_token, max_seq_len=self.max_seq_len)

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

    @override
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

        for req in state.scheduled_requests.context_requests:
            if (
                req.state == LlmRequestState.GENERATION_COMPLETE
                or req.context_remaining_length != 0
            ):
                continue
            add_token(req, new_tokens_list, beam=BEAM)
            self.finish_if_reason(req, finish_reasons, step=0)
            self.handle_logprobs(req, state, beam=BEAM, count=1)
            req.py_decoding_iter += 1

        for req in state.scheduled_requests.generation_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
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
            self.handle_logprobs(req, state, beam=BEAM, count=processed)
            req.py_decoding_iter += 1

    def return_log_probs(self, scheduled_requests: ScheduledRequests) -> bool:
        return any(req.py_return_log_probs for req in scheduled_requests.all_requests())

    @override
    @torch.inference_mode()
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

        requests = scheduled_requests.all_requests()
        new_tokens = self.store.new_tokens
        return_log_probs = self.return_log_probs(scheduled_requests)
        seq_slots_host = torch.tensor(
            [r.py_seq_slot for r in requests],
            dtype=torch.int64,  # for index_fill_
            pin_memory=True,
        )
        new_tokens_host = self._process_requests(
            scheduled_requests,
            model_outputs,
            new_tokens,
            num_context_logits_prefix_sum,
            seq_slots=seq_slots_host,
            return_log_probs=return_log_probs,
            resource_manager=resource_manager,
        )

        finish_reasons = self.store.finish_reasons
        seq_slots = seq_slots_host.to(device="cuda", non_blocking=True)
        self._write_finish_reasons(
            requests, finish_reasons=finish_reasons, seq_slots=seq_slots, new_tokens=new_tokens
        )
        finish_reasons_host = finish_reasons.to(device="cpu", non_blocking=True)

        sampler_event = torch.cuda.Event()
        sampler_event.record()
        return SampleStateTorch(
            scheduled_requests=scheduled_requests,
            device=SampleStateTensors(new_tokens=new_tokens),
            host=SampleStateTensorsHostTorch(
                new_tokens=new_tokens_host, finish_reasons=finish_reasons_host
            ),
            sampler_event=sampler_event,
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
        logits_bias_mask = torch.zeros((logits.size(0),), dtype=torch.bool, pin_memory=True)

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
        for i, (req, steps) in enumerate(zip(requests, request_steps)):
            steps = int(steps.item())
            req_bias = req._py_embedding_bias_1d
            if req_bias is not None:
                logits_bias_mask[i : (i + steps)] = True
                req_bias_index = bias_to_index[req_bias]
                bias_gather_indices.extend(repeat(req_bias_index, steps))

        if not bias_to_index:
            return
        assert req_bias is not None  # otherwise bias_to_index is empty

        bias_gather_indices_cuda = torch.tensor(
            bias_gather_indices, pin_memory=True, dtype=torch.int32
        ).to(logits.device, non_blocking=True)
        logits_bias_mask_cuda = logits_bias_mask.to(logits.device, non_blocking=True)
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

    def _sample_batched_by_strategy(
        self,
        logits_cuda: torch.Tensor,
        requests: list[LlmRequest],
        model_outputs: dict[str, torch.Tensor],
        *,
        cuda_device: torch.device,
        logits_cuda_indexer: _PackedStepIndexer,
        req_num_steps: torch.Tensor,
        req_offsets: torch.Tensor,
        token_dtype: torch.dtype,
    ) -> _BatchedSamplingResult:
        grouped_requests = _group_requests_by_strategy_key(
            requests,
            pin_memory=True,
            vocab_size=logits_cuda.size(1),
            strategy_to_key=self._grouped_sampler_cls.strategy_grouping_key,
        )
        generator_cuda = self.get_generator(cuda_device)

        # NB: Currently, "d2t" is applied to draft tokens, but not to draft logits,
        #     breaking _process_draft_tokens_rejection_sampling.
        needs_d2t = "d2t" in model_outputs
        if needs_d2t and (
            len(grouped_requests) > 1
            or (grouped_requests and next(iter(grouped_requests.values()))[1][0] != GREEDY)
        ):
            raise ValueError("d2t does not yet support non-greedy sampling")

        # Tensors for collecting sampling results (in batch ordering)
        batch_req_indices = torch.empty((len(requests),), dtype=torch.int32)
        batch_next_tokens_cuda_int = torch.empty(
            (logits_cuda.size(0),), device=cuda_device, dtype=token_dtype
        )
        batch_req_idx_offset_start = 0
        batch_next_tokens_offset_start = 0
        for (strategy_key, speculation_needs_probs), (
            group_req_indices,
            group_strategies,
        ) in grouped_requests.items():
            # group_req_indices: Indices of 'requests' entries having the same sampling
            # strategy, ordered ascending.
            batch_req_idx_offset_end = batch_req_idx_offset_start + group_req_indices.size(0)
            batch_req_indices[batch_req_idx_offset_start:batch_req_idx_offset_end] = (
                group_req_indices
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
                    req_offsets[group_req_last_index] + req_num_steps[group_req_last_index],
                )
                group_logits_cuda = logits_cuda[group_logits_cuda_indices_cuda]
                logit_indices_for_sampler = None
            else:
                group_logits_cuda_indices_cuda = group_logits_cuda_indices.to(
                    device=logits_cuda.device, non_blocking=True
                )
                group_logits_cuda = logits_cuda
                logit_indices_for_sampler = group_logits_cuda_indices_cuda

            group_strategies_per_step = [  # convert from per-request to per-step
                strat
                for strat, steps in zip(group_strategies, req_num_steps[group_req_indices])
                for _ in range(steps)
            ]
            group_next_tokens_cuda, group_softmax_cuda = (
                self._grouped_sampler_cls.sample_grouped_strategies(
                    strategy_key,
                    group_strategies_per_step,
                    group_logits_cuda,
                    generator=generator_cuda,
                    return_probs=speculation_needs_probs,
                    group_logit_indices=logit_indices_for_sampler,
                )
            )
            batch_next_tokens_offset_end = (
                batch_next_tokens_offset_start + group_next_tokens_cuda.size(0)
            )
            batch_next_tokens_cuda_int[
                batch_next_tokens_offset_start:batch_next_tokens_offset_end
            ].copy_(group_next_tokens_cuda, non_blocking=True)

            # Set LlmRequest.py_target_probs
            if speculation_needs_probs:
                assert group_softmax_cuda is not None
                current_offset = 0
                for req_idx, steps in zip(
                    group_req_indices, req_num_steps[group_req_indices].tolist()
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
        )

    def _unbatch_sampling_results(
        self,
        batched_sampling_result: _BatchedSamplingResult,
        *,
        new_tokens_cuda: torch.Tensor,
        req_num_steps: torch.Tensor,
        seq_slots: torch.Tensor,
    ) -> torch.Tensor:
        beam = BEAM
        assert beam == 0, "beam_width != 1 not supported"

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
            num_steps=req_num_steps[batch_req_indices],
            steps_dim_size=new_tokens_cuda.size(0),
            slots_dim_size=new_tokens_cuda.size(1),
            dim_order=_UnpackedStepIndexer.DimOrder.STEP_MAJOR,
            index_dtype=torch.int64,  # enforced by Tensor.scatter_
        )

        # Batch update output tensors
        batch_dest_indices_1d_cuda = batch_destination_cuda_indexer[:].to(
            new_tokens_cuda.device, non_blocking=True
        )
        new_tokens_cuda.view(-1, *new_tokens_cuda.shape[2:])[:, beam, ...].scatter_(
            0, batch_dest_indices_1d_cuda, batch_next_tokens_cuda_int
        )
        new_tokens_host = new_tokens_cuda.to("cpu", non_blocking=True)

        return new_tokens_host

    @staticmethod
    @torch.inference_mode()
    def _apply_min_length_penalty(
        logits: torch.Tensor,
        requests: list[LlmRequest],
        num_steps: list[int],
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
                    for step in range(num_steps[index]):
                        if r.max_beam_num_tokens + step < r.py_min_length[0]:
                            logits[current_offset + step, r.py_end_id] = float("-inf")
                        else:
                            # early exit
                            break
                current_offset += num_steps[index]
        return logits

    @staticmethod
    def _select_generated_logits(
        scheduled_requests: ScheduledRequests,
        raw_logits_cuda: torch.Tensor,
        *,
        req_num_generation_steps: torch.Tensor,
        num_context_logits_prefix_sum: list[int],
        generation_requests_total_steps: int,
        num_logits_to_keep: int,
    ) -> torch.Tensor:
        # raw_logits should contain only the generated logits.
        # If return context logits is requested, select only the generated logits.
        #
        # NB: Context request logits always precede generation request logits, also
        #     requests == scheduled_requests.context_requests + scheduled_requests.generation_requests
        if any(r.py_return_context_logits for r in scheduled_requests.context_requests):
            assert (
                len(num_context_logits_prefix_sum) == len(scheduled_requests.context_requests) + 1
            )
            req_num_generation_steps_cuda = req_num_generation_steps.to(
                raw_logits_cuda.device, non_blocking=True
            )
            context_req_offsets_cuda = torch.tensor(
                num_context_logits_prefix_sum, dtype=torch.int32, pin_memory=True
            ).to(device=raw_logits_cuda.device, non_blocking=True)

            # Since the goal is to keep the req_num_steps[i] last tokens for each requests[i],
            # only end-offsets of the token storage locations matter.
            next_context_req_offsets_cuda = context_req_offsets_cuda.roll(
                -1
            )  # trailing '0' is overwritten below
            # Since logits for generation requests are densely packed, cover them all by a single
            # fictituous entry in 'context_req_offsets_cuda'.
            if scheduled_requests.generation_requests:
                req_num_steps_fictitious_cuda = req_num_generation_steps_cuda[
                    : (len(scheduled_requests.context_requests) + 1)
                ].clone()
                req_num_steps_fictitious_cuda[-1].fill_(generation_requests_total_steps)
                next_context_req_offsets_cuda[-1].copy_(
                    next_context_req_offsets_cuda[-2] + req_num_steps_fictitious_cuda[-1],
                    non_blocking=True,
                )
            else:
                req_num_steps_fictitious_cuda = req_num_generation_steps_cuda[
                    : len(scheduled_requests.context_requests)
                ]
                next_context_req_offsets_cuda = next_context_req_offsets_cuda[:-1]

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
        return raw_logits_cuda

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
        return [
            ridx
            for ridx, r in enumerate(requests)
            if (r.py_stop_words_list is not None and len(r.py_stop_words_list[0]) > 0)
        ]

    def _write_finish_reasons(
        self,
        requests: list[LlmRequest],
        *,
        finish_reasons: torch.Tensor,
        seq_slots: torch.Tensor,
        new_tokens: torch.Tensor,
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
        tokens = new_tokens[:, seq_slots, BEAM]

        # we need to fill with NOT_FINISHED so we can differentiate between previous requests that had the same seq slot
        finish_reasons.index_fill_(1, seq_slots, FinishReason.NOT_FINISHED.value)
        batched_finish_reasons = finish_reasons[:, seq_slots, BEAM]

        if with_stop_words := self._requests_with_stop_words(requests):
            stop_seq_slots = torch.tensor(
                [r.py_seq_slot for r in with_stop_words], pin_memory=True
            ).to("cuda", non_blocking=True)
            stop_tokens = new_tokens[:, stop_seq_slots, BEAM]
            stop_indices = self._request_indices_with_stop_words(requests)
            batched_finish_reasons[:, stop_indices] = torch.where(
                self._are_stop_words(with_stop_words, stop_tokens),
                self._reason_tensors[FinishReason.STOP_WORDS],
                batched_finish_reasons[:, stop_indices],
            )

        batched_finish_reasons = torch.where(
            self._are_max_length(requests),
            self._reason_tensors[FinishReason.LENGTH],
            batched_finish_reasons,
        )
        batched_finish_reasons = torch.where(
            self._are_end_id(requests, tokens),
            self._reason_tensors[FinishReason.END_ID],
            batched_finish_reasons,
        )

        finish_reasons[:, seq_slots, BEAM] = batched_finish_reasons

    def _are_end_id(self, requests: list[LlmRequest], tokens: torch.Tensor) -> torch.Tensor:
        end_ids_tensor = torch.tensor(
            [
                ([req.py_end_id if req.py_end_id is not None else -1] * self.max_tokens)
                for req in requests
            ],
            pin_memory=True,
            dtype=tokens.dtype,
        ).T.to(device="cuda", non_blocking=True)
        return tokens == end_ids_tensor

    def _are_max_length(self, requests: list[LlmRequest]) -> torch.Tensor:
        lengths_tensor = torch.tensor(
            [
                [
                    ((req.get_num_tokens(BEAM) + num_tokens) - req.py_orig_prompt_len)
                    for num_tokens in range(1, self.max_tokens + 1)
                ]
                for req in requests
            ]
        )
        max_lengths_tensor = torch.tensor(
            [([min(req.py_max_new_tokens, self.max_seq_len)] * self.max_tokens) for req in requests]
        )
        return (
            (lengths_tensor >= max_lengths_tensor)
            .T.pin_memory()
            .to(device="cuda", non_blocking=True)
        )

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
        self, requests: list[LlmRequest], new_tokens: torch.Tensor, pad_id: int = _PAD_ID
    ) -> torch.Tensor:
        # TODO: make sure only the lookback tokens are pulled into the list
        longest = self._longest_stop_word_len(requests)
        assert longest > 0, f"{longest=}, longest stop word length should be greater than 0"
        lookback = longest - 1
        old_tokens = []
        for request in requests:
            old = request.get_tokens(BEAM)[-lookback:] if lookback > 0 else []
            padded = [pad_id] * max(0, lookback - len(old)) + old
            old_tokens.append([padded] * self.max_tokens)
        old_tokens_tensor = torch.tensor(old_tokens, pin_memory=True).to("cuda", non_blocking=True)
        assert old_tokens_tensor.shape == (len(requests), self.max_tokens, lookback), (
            f"{old_tokens_tensor.shape} != ({len(requests)=}, {self.max_tokens=}, {lookback=})"
        )
        new_tokens = new_tokens.T.unsqueeze(1) * self._pad_steps_mask
        ret = torch.cat((old_tokens_tensor, new_tokens), dim=-1)
        assert ret.shape == (len(requests), self.max_tokens, lookback + self.max_tokens), (
            f"{ret.shape} != ({len(requests)=}, {self.max_tokens=}, {lookback + self.max_tokens=})"
        )
        return ret

    def _are_stop_words(self, requests: list[LlmRequest], tokens: torch.Tensor) -> torch.Tensor:
        per_step = torch.zeros(
            (self.max_tokens, len(requests)), dtype=torch.bool, pin_memory=True
        ).to("cuda", non_blocking=True)

        padded_tokens = self._padded_old_tokens(requests, tokens)

        def request_stop_words(request: LlmRequest, new_tokens: torch.Tensor):
            swl, ends = request.py_stop_words_list
            if -1 in ends:
                ends = ends[: ends.index(-1)]
            lens = np.diff(ends, prepend=0)
            lens_device = torch.tensor(list(lens), pin_memory=True).to("cuda", non_blocking=True)
            max_len = np.max(lens)

            words = torch.zeros(len(lens), max_len, dtype=torch.int32, pin_memory=True)
            for step, (start, length) in enumerate(zip([0] + ends, lens)):
                words[step, :length] = torch.tensor(swl[start : start + length], dtype=torch.int32)
            words_device = words.to("cuda", non_blocking=True)

            draft_token_length = get_draft_token_length(request)
            for step, step_seq in enumerate(new_tokens[: max_len + draft_token_length + 1]):
                size_per_step = step_seq.size(0) - draft_token_length + step
                for word, L in zip(words_device, lens_device):
                    truncated_seq = step_seq[size_per_step - L : size_per_step]
                    if torch.equal(truncated_seq, word[:L]):
                        # We don't care about subsequent steps because we already found a stop word match
                        return step
            return None

        for request_idx, request in enumerate(requests):
            step = request_stop_words(request, padded_tokens[request_idx])
            if step is not None:
                per_step[step][request_idx] = True
        return per_step

    @nvtx_range("_process_requests")
    def _process_requests(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, torch.Tensor],
        new_tokens_cuda: torch.Tensor,
        num_context_logits_prefix_sum: list[int],
        *,
        seq_slots: torch.Tensor,
        return_log_probs: bool,
        resource_manager: Optional[ResourceManager] = None,
    ) -> torch.Tensor:
        seq_slots = seq_slots.to(dtype=torch.int32)  # int32 suffices here
        spec_tree_manager = self.get_spec_tree_manager(resource_manager)

        raw_logits_cuda = model_outputs["logits"]

        requests = scheduled_requests.all_requests()
        cuda_device = raw_logits_cuda.device
        req_num_steps_list = [1 + get_draft_token_length(req) for req in requests]
        req_num_steps = torch.tensor(req_num_steps_list, dtype=torch.int32, pin_memory=True)
        # NB: These offsets consider generated tokens _only_ (draft and target, but not context)
        #     and are thus only correct after _select_generated_logits() below.
        req_offsets, sum_steps = _PackedStepIndexer.calculate_request_offsets(
            req_num_steps, pin_memory=True
        )

        raw_logits_cuda = self._select_generated_logits(
            scheduled_requests,
            raw_logits_cuda,
            req_num_generation_steps=req_num_steps,
            num_context_logits_prefix_sum=num_context_logits_prefix_sum,
            generation_requests_total_steps=(
                # NB: requests == scheduled_requests.context_requests + scheduled_requests.generation_requests
                sum_steps - cast(int, req_offsets[len(scheduled_requests.context_requests)].item())
                if scheduled_requests.generation_requests
                else 0
            ),
            num_logits_to_keep=sum_steps,
        )

        # Handle embedding bias
        logits_cuda = raw_logits_cuda[:sum_steps]
        self._apply_embedding_bias(logits_cuda, requests, req_num_steps)

        logits_cuda = self._apply_min_length_penalty(logits_cuda, requests, req_num_steps_list)

        # Fast path for drafter model's tree sampling.
        if spec_tree_manager is not None and logits_cuda.size(0) == len(
            scheduled_requests.all_requests()
        ):
            new_tokens_host = self._tree_sampling_batch(
                requests,
                self.max_num_sequences,
                seq_slots,
                model_outputs,
                spec_tree_manager,
            )
            return new_tokens_host

        # Indexer for accessing tokens in 'logits_cuda', corresponding to the
        # requests in 'requests'.
        steps_dim_size = new_tokens_cuda.size(0)
        logits_cuda_indexer = _PackedStepIndexer(
            num_steps=req_num_steps,
            max_steps=steps_dim_size,
            req_offsets=req_offsets,
        )

        # Handle top-k logprobs. This is done outside the sampling loop,
        # because the returned logprobs are specified to not reflect temperature scaling,
        # top-k/top-p masking, etc.
        if return_log_probs:
            assert logits_cuda.dim() == 2, "logits should be 2D"
            logprobs_req_indices = [
                req_id for req_id, req in enumerate(requests) if req.py_num_logprobs
            ]
            logprobs_logit_indices = logits_cuda_indexer[logprobs_req_indices]
            logprobs_logit_indices_cuda = logprobs_logit_indices.to(
                device=logits_cuda.device, non_blocking=True
            )
            logprobs_cuda = F.log_softmax(
                logits_cuda[logprobs_logit_indices_cuda].to(dtype=torch.float32, non_blocking=True),
                dim=-1,
            )
            topk_vals_cuda, topk_indices_cuda = torch.topk(
                logprobs_cuda, k=max(req.py_num_logprobs for req in requests), dim=-1
            )
            # Use a single D2H copy to reduce overheads
            topk_vals = torch.empty_like(topk_vals_cuda, device="cpu", pin_memory=True)
            topk_indices = torch.empty_like(topk_indices_cuda, device="cpu", pin_memory=True)
            topk_vals.copy_(topk_vals_cuda, non_blocking=True)
            topk_indices.copy_(topk_indices_cuda, non_blocking=True)
            current_offset = 0
            for req_id, steps in zip(
                logprobs_req_indices, req_num_steps[logprobs_req_indices].tolist()
            ):
                req = requests[req_id]
                next_offset = current_offset + steps
                # NB: Assigning views on memory which is being filled asynchronously
                req.py_topk_logprobs_vals = topk_vals[
                    current_offset:next_offset, : req.py_num_logprobs
                ]
                req.py_topk_logprobs_indices = topk_indices[
                    current_offset:next_offset, : req.py_num_logprobs
                ]
                current_offset = next_offset

        # Perform sampling in batches
        batched_sampling_result = self._sample_batched_by_strategy(
            logits_cuda,
            requests,
            model_outputs,
            cuda_device=cuda_device,
            logits_cuda_indexer=logits_cuda_indexer,
            req_offsets=req_offsets,
            req_num_steps=req_num_steps,
            token_dtype=new_tokens_cuda.dtype,
        )

        # Fill results into output buffers
        new_tokens_host = self._unbatch_sampling_results(
            batched_sampling_result,
            new_tokens_cuda=new_tokens_cuda,
            req_num_steps=req_num_steps,
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


class TRTLLMSampler(Sampler):
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
            gathered_ids = self.store["decoder_state"].gathered_ids.to("cpu", non_blocking=True)
        new_output_tokens = self.store["decoder_state"].all_new_tokens.to("cpu", non_blocking=True)
        finished_sum = self.store["decoder_state"].finished_sum.to("cpu", non_blocking=True)
        finish_reasons = self.store["decoder_state"].finish_reasons.to("cpu", non_blocking=True)
        sequence_lengths = self.store["decoder_state"].sequence_lengths.to("cpu", non_blocking=True)

        log_probs = None
        cum_log_probs = None
        if any(request.py_return_log_probs for request in scheduled_requests.all_requests()):
            log_probs = self.store["decoder_state"].log_probs.to("cpu", non_blocking=True)
            cum_log_probs = self.store["decoder_state"].cum_log_probs.to("cpu", non_blocking=True)

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

        sampler_event = torch.cuda.Event()
        sampler_event.record()

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
        new_tokens_host = state.host.new_tokens.flatten().tolist()
        sequence_lengths_host_data = state.host.sequence_lengths.flatten().tolist()
        finish_reasons = state.host.finish_reasons.flatten().tolist()
        log_probs_host = state.host.log_probs.tolist() if state.host.log_probs is not None else None
        cum_log_probs_host = (
            state.host.cum_log_probs.tolist() if state.host.cum_log_probs is not None else None
        )

        reqs = [
            r for r in state.scheduled_requests.context_requests if not r.is_context_init_state
        ] + [
            r
            for r in state.scheduled_requests.generation_requests
            if not r.is_generation_complete_state
        ]

        reqs_with_new_tokens = [
            r for r in reqs if (sequence_lengths_host_data[r.py_seq_slot] > r.get_num_tokens(0))
        ]

        # Add new tokens
        new_tokens = [new_tokens_host[r.py_seq_slot] for r in reqs_with_new_tokens]
        add_new_tokens_to_requests(reqs_with_new_tokens, new_tokens, 0)

        # Log probs
        for request in reqs_with_new_tokens:
            if request.py_return_log_probs:
                seq_slot = request.py_seq_slot
                seq_len = sequence_lengths_host_data[seq_slot]
                begin_log_probs_offset = request.prompt_len
                current_token = seq_len - request.prompt_len - 1
                log_probs = [
                    {
                        new_tokens_host[seq_slot]: Logprob(
                            logprob=log_probs_host[seq_slot][0][
                                begin_log_probs_offset + current_token
                            ],
                            rank=1,
                        )
                    }
                ]
                cum_log_probs = [cum_log_probs_host[seq_slot]]
                request.py_result.append_log_probs([log_probs], cum_log_probs)

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

            for beam in range(beam_width):
                seq_len = sequence_lengths_host_data[seq_slot * beam_width + beam]
                num_new_tokens[beam] = min(
                    num_generated_tokens, seq_len - request.get_num_tokens(beam)
                )

                for step in range(num_new_tokens[beam]):
                    new_token = add_token(request, new_tokens_host, beam=beam, step=step)

                    if request.py_return_log_probs:
                        assert state.host.log_probs is not None
                        # NOTE: Log probs with drafting has not been tested yet.
                        begin_log_probs_offset = (
                            request.prompt_len if request.sampling_config.beam_width == 1 else 0
                        )
                        current_token = seq_len - request.prompt_len - num_new_tokens[beam] + step
                        log_probs[beam].append(
                            {
                                new_token: Logprob(
                                    logprob=log_probs_host[seq_slot][beam][
                                        begin_log_probs_offset + current_token
                                    ],
                                    rank=1,
                                )
                            }
                        )

                if request.py_return_log_probs:
                    cum_log_probs.append(cum_log_probs_host[seq_slot][beam])

                finished_state = FinishedState(finish_reasons[seq_slot * beam_width + beam])
                if finished_state.is_finished:
                    finish_reason = finished_state.to_finish_reason()
                    request.set_finished_reason(finish_reason, beam)

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

        for beam in range(beam_width):
            # get the correct generated tokens for beam search
            begin = request.py_prompt_len
            generated_length = sequence_lengths_host[seq_slot, beam].item() - request.py_prompt_len
            end = begin + generated_length
            generated_tokens[beam] = output_ids_host[seq_slot, beam, begin:end].tolist()

            # get the correct log probs for beam search
            if request.py_return_log_probs:
                cum_log_probs.append(cum_log_probs_host[seq_slot, beam].item())

                begin_log_probs_offset = (
                    request.prompt_len if request.sampling_config.beam_width == 1 else 0
                )
                for current_token, token in enumerate(generated_tokens[beam]):
                    log_probs[beam].append(
                        {
                            token: Logprob(
                                logprob=log_probs_host[seq_slot, beam][
                                    begin_log_probs_offset + current_token
                                ].item(),
                                rank=1,
                            )
                        }
                    )
        if request.py_return_log_probs:
            request.py_result.set_log_probs(log_probs, cum_log_probs)

        request.set_generated_tokens(generated_tokens)
