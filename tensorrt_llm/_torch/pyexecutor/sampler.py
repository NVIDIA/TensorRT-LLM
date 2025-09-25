import dataclasses
import enum
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import repeat
from typing import Any, List, Literal, Optional, cast

import torch

from tensorrt_llm._torch.pyexecutor.make_decoding_batch_input_output import \
    MakeDecodingBatchInputOutput
from tensorrt_llm._utils import nvtx_range, torch_dtype_to_binding
from tensorrt_llm.bindings import (CudaStream, DataType, ModelConfig,
                                   WorldConfig, make_sampling_config)
from tensorrt_llm.bindings.executor import (DecodingConfig, DecodingMode,
                                            FinishReason, KvCacheConfig)
from tensorrt_llm.bindings.internal.algorithms import CreateNewDecoderRequests
from tensorrt_llm.bindings.internal.batch_manager import (
    DecoderInputBuffers, add_new_tokens_to_requests, make_decoding_batch_input)
from tensorrt_llm.bindings.internal.runtime import (BufferManager, CudaEvent,
                                                    DecoderState,
                                                    GptDecoderBatched)
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.mapping import Mapping

from .finish_reason import FinishedState
from .llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from .scheduler import ScheduledRequests

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


@dataclass(kw_only=True)
class SampleStateTensors:
    new_tokens: torch.Tensor
    log_probs: torch.Tensor | None = None

    def values(self):
        return vars(self).values()


@dataclass(kw_only=True)
class SampleState:
    scheduled_requests: ScheduledRequests

    device: SampleStateTensors = None
    host: SampleStateTensors = None

    sampler_event: torch.cuda.Event = None


class Sampler(ABC):

    SampleState = SampleState

    def setup_sampler_step(self, scheduled_requests: ScheduledRequests):
        pass

    def get_cache_indirection(self) -> torch.Tensor | None:
        return None

    @abstractmethod
    def sample_async(self, scheduled_requests: ScheduledRequests, model_outputs,
                     num_context_logits_prefix_sum: list[int]) -> SampleState:
        raise NotImplementedError

    @abstractmethod
    def update_requests(self, state: SampleState) -> None:
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
    def sample_async(self, scheduled_requests: ScheduledRequests, model_outputs,
                     num_context_logits_prefix_sum: list[int]) -> SampleState:
        host = SampleStateTensors(new_tokens=torch.empty(0))
        return SampleState(scheduled_requests=scheduled_requests, host=host)

    @override
    def update_requests(self, state: SampleState) -> None:
        assert isinstance(state, SampleState)
        scheduled_requests = state.scheduled_requests
        assert (not scheduled_requests.generation_requests)
        for idx, request in enumerate(scheduled_requests.context_requests):
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)

    @override
    def is_generation_model(self) -> bool:
        return False


@dataclass(kw_only=True)
class MultimodalResult:
    mm_embeddings: List[torch.Tensor] = None

    def values(self):
        return vars(self).values()


@dataclass(kw_only=True)
class SampleStateWithMMResult:
    scheduled_requests: ScheduledRequests

    data: MultimodalResult = None


class EarlyStopWithMMResult(Sampler):
    """
    Use for skipping decoding step for non generation model, and return the batch_output (such as mm_embeddings)
    """

    @override
    def sample_async(
            self, scheduled_requests: ScheduledRequests, model_outputs,
            num_context_logits_prefix_sum: list[int]
    ) -> SampleStateWithMMResult:
        # from model_outputs to MultimodalResult
        data = MultimodalResult(mm_embeddings=model_outputs['mm_embeddings'])
        return SampleStateWithMMResult(scheduled_requests=scheduled_requests,
                                       data=data)

    @override
    def update_requests(self, state: SampleStateWithMMResult) -> None:
        assert isinstance(state, SampleStateWithMMResult)
        scheduled_requests = state.scheduled_requests
        assert (not scheduled_requests.generation_requests)
        mm_embeddings = state.data.mm_embeddings
        for request, mm_embedding in zip(scheduled_requests.context_requests,
                                         mm_embeddings):
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


def top_k_sampling_batch(
    logits,
    top_k=50,
    generator: Optional[torch.Generator] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_dim = logits.dim()
    if logits_dim == 1:
        logits = logits.unsqueeze(0)
    # logits should be 2D ：[batch_size, vocab_size]
    batch_size, vocab_size = logits.size()

    # get first top_k logits of each sample and their indices
    if top_k > 0:
        values, indices = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1).expand(batch_size, vocab_size)

        # set the logits who is less than first top_k logits to -inf
        logits = torch.where(logits < min_values,
                             torch.full_like(logits, float('-inf')), logits)

    # compute probability distribution
    softmax = torch.softmax(logits, dim=-1)

    # sample from the distribution and generate result of [batch_size, 1]
    next_tokens = torch.multinomial(softmax, num_samples=1,
                                    generator=generator).squeeze(-1)
    return next_tokens, softmax


def top_p_sampling_batch(
    logits: torch.Tensor,
    *,
    top_p: float = 0.9,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_dim = logits.dim()
    assert logits_dim == 2, "logits should be 2D: [batch_size, vocab_size]"

    if temperature != 0:
        logits = logits / max(temperature, 1e-5)

    # sort the logits of each sample in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # compute  cumulative probability distribution of each sample
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1),
                                    dim=-1)
    # get the location of top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    # set the logits to -inf whose is outside top_p
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))

    # compute probability distribution
    softmax = torch.softmax(logits, dim=-1)

    # sample from the distribution and generate result of [batch_size, 1]
    next_tokens = torch.multinomial(softmax, num_samples=1,
                                    generator=generator).squeeze(-1)
    return next_tokens, softmax


def top_k_top_p_sampling_batch(logits: torch.Tensor,
                               *,
                               top_k: int,
                               top_p: float,
                               temperature: float = 1.0,
                               generator: Optional[torch.Generator] = None):
    logits_dim = logits.dim()
    assert logits_dim == 2, "logits should be 2D: [batch_size, vocab_size]"
    if temperature != 0:
        logits = logits / max(temperature, 1e-5)
    batch_size, vocab_size = logits.size()
    # get first top_k logits of each sample and their indices
    if top_k > 0:
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1).expand(batch_size, vocab_size)

        # set the logits who is less than first top_k logits to -inf
        logits = torch.where(logits < min_values,
                             torch.full_like(logits, float('-inf')), logits)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # compute  cumulative probability distribution of each sample
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1),
                                    dim=-1)

    # get the location of top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    # set the logits to -inf whose is outside top_p
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))

    # compute probability distribution
    softmax = torch.softmax(logits, dim=-1)

    # sample from the distribution and generate result of [batch_size, 1]
    next_tokens = torch.multinomial(softmax, num_samples=1,
                                    generator=generator).squeeze(-1)
    return next_tokens, softmax


def greedy_search_sampling_batch(
    logits,
    *,
    softmax_indices: Optional[torch.IntTensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    next_tokens = torch.argmax(logits, dim=-1)
    if softmax_indices is not None:
        logits = logits[softmax_indices.to(logits.device, non_blocking=True)]
    softmax = torch.softmax(logits, dim=-1)
    return next_tokens, softmax


def get_rejected_indices(draft_probs: torch.Tensor, target_probs: torch.Tensor,
                         generator: torch.Generator,
                         draft_tokens: list[int]) -> torch.Tensor:
    # NB: ModelDrafter._pad_to_max_draft_tokens pads draft_tokens, but
    #     not draft_probs. Relying on shape of draft_probs here.
    num_draft_tokens = draft_probs.size(0)
    draft_tokens = draft_tokens[:num_draft_tokens]
    # NB: torch.arange is needed to enable "advanced indexing",
    #   cf. https://numpy.org/devdocs/user/basics.indexing.html#integer-array-indexing
    token_idx = torch.arange(num_draft_tokens,
                             dtype=torch.int32,
                             device=generator.device)
    draft_tokens_cuda = torch.tensor(draft_tokens,
                                     dtype=torch.int32,
                                     pin_memory=True).to(
                                         device=generator.device,
                                         non_blocking=True)
    p = draft_probs[token_idx, draft_tokens_cuda]
    q = target_probs.squeeze(0)[token_idx, draft_tokens_cuda]
    accept_probs = torch.minimum(
        torch.ones((), device=generator.device, dtype=q.dtype), q / p)
    # Use deterministic random generation for multi-GPU consistency
    rejected_indices = (torch.rand(accept_probs.shape,
                                   generator=generator,
                                   device=accept_probs.device)
                        > accept_probs).nonzero()
    return rejected_indices


def sample_rejected(draft_probs: torch.Tensor, target_probs: torch.Tensor,
                    generator: torch.Generator, num_accepted: int):

    last_draft = draft_probs[num_accepted]
    last_target = target_probs[num_accepted]
    new = last_target - last_draft
    new = torch.where(new > 0, new, 0.0)

    new_token = torch.multinomial(new, num_samples=1,
                                  generator=generator).squeeze(-1)
    return new_token


TopK = tuple[Literal["top_k"], int]
TopP = tuple[Literal["top_p"], float, float]
TopKTopP = tuple[Literal["top_k_top_p"], int, float, float]
Greedy = tuple[Literal["greedy"], None]
GREEDY: Greedy = ("greedy", None)
Strategy = TopK | TopP | Greedy | TopKTopP


def _request_strategy(request: LlmRequest) -> Strategy:
    # top_p and top_K with temperature=0.0 reduces to greedy
    # sampling
    temperature = request.sampling_config.temperature
    if temperature is not None:
        temperature = temperature[0]
        if temperature == 0.0:
            return GREEDY

    if request.sampling_config.top_k is not None and len(
            request.sampling_config.top_k
    ) > 0 and request.sampling_config.top_p is not None and len(
            request.sampling_config.top_p) > 0:
        return ("top_k_top_p", request.sampling_config.top_k[0],
                request.sampling_config.top_p[0], temperature)
    elif request.sampling_config.top_p is not None and len(
            request.sampling_config.top_p) > 0:
        top_p = request.sampling_config.top_p[0]
        return ("top_p", top_p, temperature)
    elif request.sampling_config.top_k is not None and len(
            request.sampling_config.top_k) > 0:
        return ("top_k", request.sampling_config.top_k[0])
    else:
        return GREEDY


def _group_requests_by_sampling_strategy(
        requests: Iterable[LlmRequest],
        *,
        pin_memory: bool = False) -> dict[Strategy, torch.Tensor]:
    strategy_dict: dict[Strategy, list[int]] = defaultdict(list)
    for req_index, req in enumerate(requests):
        strategy_dict[_request_strategy(req)].append(req_index)
    return {
        strategy: torch.tensor(indices,
                               pin_memory=pin_memory,
                               dtype=torch.int32)
        for strategy, indices in strategy_dict.items()
    }


def sample(
    strategy: Strategy,
    logits: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    *,
    softmax_indices: Optional[torch.IntTensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    filter_softmax = True
    match strategy:
        case ("top_k", top_k):
            tokens, softmax = top_k_sampling_batch(logits, top_k, generator)
        case ("top_p", top_p, temperature):
            tokens, softmax = top_p_sampling_batch(
                logits,
                top_p=top_p,
                generator=generator,
                **(dict(temperature=temperature)
                   if temperature is not None else dict()))
        case ("top_k_top_p", top_k, top_p, temperature):
            tokens, softmax = top_k_top_p_sampling_batch(
                logits,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
                **(dict(temperature=temperature)
                   if temperature is not None else dict()))
        case ("greedy", None):
            tokens, softmax = greedy_search_sampling_batch(
                logits, softmax_indices=softmax_indices)
            filter_softmax = False
    if filter_softmax and softmax_indices is not None:
        softmax = softmax[softmax_indices.to(softmax.device, non_blocking=True)]
    return tokens, softmax


def add_token(request: LlmRequest,
              new_tokens: list[list[list[int]]],
              *,
              beam: int,
              step: int = 0) -> int:
    seq_slot = request.py_seq_slot
    assert seq_slot is not None
    new_token = new_tokens[step][seq_slot][beam]
    request.add_new_token(new_token, beam)
    return new_token


def int_tensor(shape: tuple[int, ...], device: str = 'cuda') -> torch.Tensor:
    return torch.empty(shape, dtype=torch.int, device=device)


@dataclass(kw_only=True, frozen=True)
class _BatchedSamplingResult:
    # Original request indices for all requests (permuted due to batching by strategy):
    batch_req_indices: torch.Tensor
    # Next tokens for all requests:
    batch_next_tokens_cuda_int: torch.Tensor
    # Probs for all requests which need them:
    batch_softmax_cuda: torch.Tensor
    # (request, batch_softmax indices), for requests having py_draft_logits / requesting py_target_probs:
    py_draft_logits_indices: list[tuple[LlmRequest,
                                        torch.Tensor]] = dataclasses.field(
                                            default_factory=list)


# Inspired by https://github.com/pytorch/pytorch/issues/80577; note also the
# suggestion to consider torch.nested.
def torch_multi_arange(ends: torch.Tensor,
                       *,
                       starts: Optional[torch.Tensor] = None,
                       steps: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Efficiently compute torch.cat([torch.arange(b, e, d) for b, e, d in zip(starts, ends, steps)]).

    Starts, ends, steps need to share dtype and shape. Invalid ranges like range(1, 2, -1) are
    silently discarded. 'steps' defaults to 1 and 'starts' defaults to 0.
    """
    if steps is not None:
        assert ends.dtype == steps.dtype
        assert ends.shape == steps.shape
    if starts is not None:
        assert ends.dtype == starts.dtype
        assert ends.shape == starts.shape

    # This algorithm combines torch.repeat_interleaved() and torch.cumsum() to
    # construct the result.
    #
    # 1. Given N ranges (characterized by starts, ends, steps), construct a sequence
    #    of 2N numbers, in which the non-overlapping pairs of consecutive numbers
    #    correspond to the ranges. For a given range, the pair (a, b) is chosen such
    #    that upon torch.cumsum() application 'a' turns the last element of the
    #    preceding range into the start element for the current range and 'b' is
    #    simply the step size for the current range.
    #
    repeats = ends  # number of elements in each range
    if starts is not None:
        repeats = repeats.clone()
        repeats -= starts
    if steps is not None:
        repeats = repeats.div(steps, rounding_mode="trunc")
    repeats = repeats.clip(0)  # ignore invalid ranges
    range_ends = repeats - 1  # last element in each range
    if steps is not None:
        range_ends *= steps
    if starts is not None:
        range_ends += starts
    prev_range_ends = range_ends.roll(
        1)  # last element in preceding range (or 0)
    prev_range_ends[0] = 0
    ones = torch.tensor(1, dtype=ends.dtype, pin_memory=True).to(
        device=ends.device, non_blocking=True).broadcast_to(ends.shape)
    if steps is None:
        steps = ones
    jumps = -prev_range_ends  # delta from one range to the next
    if starts is not None:
        jumps += starts
    seq = torch.cat((jumps.unsqueeze(-1), steps.unsqueeze(-1)), dim=1).view(-1)
    #
    # 2. Construct output via torch.repeat_interleave() and torch.cumsum()
    seq_repeats = torch.cat((ones.unsqueeze(-1), (repeats - 1).unsqueeze(-1)),
                            dim=1).view(-1)
    seq = seq.repeat_interleave(seq_repeats)
    seq = seq.cumsum(0)
    return seq


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
            pin_memory: bool = False) -> tuple[torch.Tensor, int]:
        if req_num_steps.numel():
            req_offsets = torch.cumsum(req_num_steps, 0)
            sum_steps = int(req_offsets[-1].item())
            req_offsets_rolled = torch.empty_like(req_offsets,
                                                  pin_memory=pin_memory)
            req_offsets_rolled[1:] = req_offsets[:-1]
            req_offsets_rolled[0] = 0
            req_offsets = req_offsets_rolled
        else:
            req_offsets = torch.empty_like(req_num_steps, pin_memory=pin_memory)
            sum_steps = 0
        return req_offsets, sum_steps

    def _build_index(
            self, req_offsets: torch.Tensor, num_steps: torch.Tensor,
            max_steps: int,
            index_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        steps_dim = torch.arange(max_steps,
                                 device=num_steps.device,
                                 dtype=index_dtype)
        valid_mask = steps_dim.unsqueeze(0) < num_steps.unsqueeze(-1)
        indices = self._compute_index_map(index_dtype=index_dtype,
                                          steps_dim=steps_dim,
                                          req_offsets=req_offsets)
        # NB: steps_dim and req_offsets may have been overwritten by this point.
        return indices, valid_mask

    @abstractmethod
    def _compute_index_map(self, index_dtype: torch.dtype,
                           steps_dim: torch.Tensor,
                           req_offsets: torch.Tensor) -> torch.Tensor:
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
        super().__init__(num_steps=num_steps,
                         req_offsets=req_offsets,
                         max_steps=max_steps,
                         index_dtype=index_dtype)

    @override
    def _compute_index_map(self, index_dtype: torch.dtype,
                           steps_dim: torch.Tensor,
                           req_offsets: torch.Tensor) -> torch.Tensor:
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
        super().__init__(num_steps=num_steps,
                         req_offsets=req_offsets,
                         max_steps=max_steps,
                         index_dtype=index_dtype)


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


class TorchSampler(Sampler):
    BEAM = 0
    MAX_BEAM_WIDTH = BEAM + 1

    @override
    def is_generation_model(self) -> bool:
        return True

    @dataclass(frozen=True, kw_only=True)
    class Store:
        new_tokens: torch.Tensor
        """Shape: See cpp DecoderState.getAllNewTokens()"""

    def create_store(self) -> Store:
        return self.Store(new_tokens=int_tensor(self.NEW_TOKENS_SHAPE))

    @dataclass(frozen=True, kw_only=True)
    class Args:
        max_seq_len: int
        max_draft_len: int
        max_num_sequences: int
        max_beam_width: int

    def __init__(self, args: Args):
        self.max_seq_len = args.max_seq_len
        self.max_tokens = args.max_draft_len + 1
        assert args.max_beam_width == self.MAX_BEAM_WIDTH, "TorchSampler only supports beam_width = 1"
        self.max_num_sequences = args.max_num_sequences

        self.NEW_TOKENS_SHAPE = (self.max_tokens, self.max_num_sequences,
                                 self.MAX_BEAM_WIDTH)
        # AutoDeploy build creates the sampler in inference mode,
        # which would disallow in-place mutating of new_tokens.
        # So, we temporarily exit inference mode.
        with torch.inference_mode(False):
            self.store = self.create_store()

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

    def _meet_max_token_stop_criteria(self, request: LlmRequest):
        num_tokens = request.get_num_tokens(self.BEAM)
        return (num_tokens - request.py_orig_prompt_len
                >= request.py_max_new_tokens) or (num_tokens
                                                  >= self.max_seq_len)

    @staticmethod
    def _meet_stop_token_criteria(request: LlmRequest):
        if request.py_stop_words_list:
            assert isinstance(
                request.py_stop_words_list,
                list), "request.py_stop_words_list should be a list"
            stop_words_list, prefix_sum = request.py_stop_words_list
            tokens = request.get_tokens(0)
            offset = 0
            for i, offset_end in enumerate(prefix_sum):
                if i > 0:
                    offset = prefix_sum[i - 1]
                stop_word = stop_words_list[offset:offset_end]
                if len(stop_word) > len(tokens):
                    continue
                if tokens[-len(stop_word):] == stop_word:
                    return True
        return False

    def _handle_stop_criteria(self, request: LlmRequest,
                              new_token: int) -> bool:
        """Handle stop criteria and set appropriate finish reasons and state.
        Returns True if generation should stop."""
        if new_token == request.py_end_id:
            request.finish_by(FinishReason.END_ID, self.BEAM)
            return True

        if self._meet_max_token_stop_criteria(request):
            request.finish_by(FinishReason.LENGTH, self.BEAM)
            return True

        if self._meet_stop_token_criteria(request):
            request.finish_by(FinishReason.STOP_WORDS, self.BEAM)
            return True

        return False

    def handle_logprobs(self, request: LlmRequest, state: SampleState, *,
                        beam: int, count: int):
        current_slice = slice(0, count), request.py_seq_slot, beam
        if request.py_return_log_probs:
            assert state.host.log_probs is not None
            log_probs = state.host.log_probs[request.py_seq_slot][beam][:count]
            current_tokens = state.host.new_tokens[current_slice]

            token_log_probs = [{
                int(token): Logprob(logprob=logprob, rank=1)
            } for token, logprob in zip(current_tokens, log_probs.tolist())]
            assert beam == 0, "The following call relies on beam_width to be 1 - hence the list with a single element"
            request.py_result.append_log_probs([token_log_probs])

    def _process_draft_tokens_greedy(self, request: LlmRequest,
                                     new_tokens: torch.Tensor) -> int:
        new_token = add_token(request, new_tokens, beam=self.BEAM)
        stop = self._handle_stop_criteria(request, new_token)
        if stop or get_draft_token_length(request) == 0:
            return 0
        num_accepted = 0

        for draft_token in request.py_draft_tokens:
            if draft_token != new_token:
                # Reject.
                break

            num_accepted += 1
            new_token = add_token(request,
                                  new_tokens,
                                  beam=self.BEAM,
                                  step=num_accepted)
            if self._handle_stop_criteria(request, new_token):
                break
        return num_accepted

    def _process_draft_tokens_rejection_sampling(
            self, request: LlmRequest, new_tokens: torch.Tensor) -> int:
        sampling_strategy = _request_strategy(request)
        generator = self.get_generator(request.py_draft_logits.device)
        _, draft_probs = sample(sampling_strategy,
                                request.py_draft_logits,
                                generator=generator)
        draft_probs = draft_probs.squeeze(0)
        target_probs = request.py_target_probs
        rejected_indices = get_rejected_indices(draft_probs, target_probs,
                                                generator,
                                                request.py_draft_tokens)
        sample_last = True
        stop = False
        if rejected_indices.numel() == 0:
            num_initially_accepted = get_draft_token_length(request)
            sample_last = False
        else:
            num_initially_accepted = rejected_indices[0].item()
        num_accepted = num_initially_accepted
        for i in range(num_accepted):
            new_token = request.py_draft_tokens[i]
            new_tokens[i, request.seq_slot, self.BEAM] = new_token
            new_token = add_token(request, new_tokens, beam=self.BEAM, step=i)
            stop = self._handle_stop_criteria(request, new_token)
            if stop:
                num_accepted = i + 1
                return num_accepted
        if sample_last:
            new_token = sample_rejected(draft_probs, target_probs, generator,
                                        num_accepted)
            new_tokens[num_accepted, request.seq_slot, self.BEAM] = new_token
        new_token = add_token(request,
                              new_tokens,
                              beam=self.BEAM,
                              step=num_accepted)
        stop = self._handle_stop_criteria(request, new_token)

        return num_accepted

    def process_draft_tokens(self, request: LlmRequest,
                             new_tokens: torch.Tensor) -> int:
        if request.py_draft_logits is None:
            return self._process_draft_tokens_greedy(request, new_tokens)
        else:
            return self._process_draft_tokens_rejection_sampling(
                request, new_tokens)

    @override
    @torch.inference_mode()
    def update_requests(self, state: SampleState) -> None:
        assert isinstance(state, SampleState)
        if state.sampler_event:
            state.sampler_event.synchronize()

        new_tokens = state.host.new_tokens.tolist()

        for req in state.scheduled_requests.context_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE or req.context_remaining_length != 0:
                continue
            new_token = add_token(req, new_tokens, beam=self.BEAM)
            self._handle_stop_criteria(req, new_token)
            self.handle_logprobs(req, state, beam=self.BEAM, count=1)
            req.py_decoding_iter += 1

        for req in state.scheduled_requests.generation_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            processed = 1
            num_accepted = self.process_draft_tokens(req, new_tokens)
            if get_draft_token_length(req) > 0:
                req.py_num_accepted_draft_tokens = num_accepted
                req.py_rewind_len = req.py_draft_pages_allocated - num_accepted
            else:
                req.py_num_accepted_draft_tokens = 0
                req.py_rewind_len = 0
            processed += num_accepted
            self.handle_logprobs(req, state, beam=self.BEAM, count=processed)
            req.py_decoding_iter += 1

    def log_probs_host(
            self,
            scheduled_requests: ScheduledRequests) -> Optional[torch.Tensor]:
        """Shape: In lockstep with TRTLLMSampler: https://github.com/NVIDIA/TensorRT-LLM/blob/cea5dd1e3883b18bf50901a7f196f50a9544c28c/cpp/include/tensorrt_llm/runtime/decoderState.h#L103"""
        if any(req.py_return_log_probs
               for req in scheduled_requests.all_requests()):
            return torch.empty(
                (self.max_num_sequences, self.MAX_BEAM_WIDTH, self.max_tokens),
                device="cpu",
                pin_memory=True)
        return None

    @override
    @torch.inference_mode()
    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs: dict[str, torch.Tensor],
                     num_context_logits_prefix_sum: list[int]) -> SampleState:
        requests = scheduled_requests.all_requests()
        new_tokens = self.store.new_tokens
        log_probs_host = self.log_probs_host(scheduled_requests)
        seq_slots_host = torch.tensor(
            [r.py_seq_slot for r in requests],
            dtype=torch.int64,  # for index_fill_
            pin_memory=True)
        new_tokens_host = self._process_requests(scheduled_requests,
                                                 model_outputs,
                                                 new_tokens,
                                                 num_context_logits_prefix_sum,
                                                 seq_slots=seq_slots_host,
                                                 log_probs_host=log_probs_host)

        sampler_event = torch.cuda.Event()
        sampler_event.record()
        return SampleState(scheduled_requests=scheduled_requests,
                           device=SampleStateTensors(new_tokens=new_tokens),
                           host=SampleStateTensors(new_tokens=new_tokens_host,
                                                   log_probs=log_probs_host),
                           sampler_event=sampler_event)

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
        logits_bias_mask = torch.zeros((logits.size(0), ),
                                       dtype=torch.bool,
                                       pin_memory=True)

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
        bias_to_index: dict[torch.Tensor,
                            int] = defaultdict(provision_bias_index)

        # Source indices for bias application
        bias_gather_indices: list[int] = []

        # Collect bias information
        req_bias = None
        for i, (req, steps) in enumerate(zip(requests, request_steps)):
            steps = int(steps.item())
            req_bias = req._py_embedding_bias_1d
            if req_bias is not None:
                logits_bias_mask[i:(i + steps)] = True
                req_bias_index = bias_to_index[req_bias]
                bias_gather_indices.extend(repeat(req_bias_index, steps))

        if not bias_to_index:
            return
        assert req_bias is not None  # otherwise bias_to_index is empty

        bias_gather_indices_cuda = torch.tensor(bias_gather_indices,
                                                pin_memory=True,
                                                dtype=torch.int32).to(
                                                    logits.device,
                                                    non_blocking=True)
        logits_bias_mask_cuda = logits_bias_mask.to(logits.device,
                                                    non_blocking=True)
        biases_tensor = torch.empty((len(bias_to_index), *req_bias.shape),
                                    pin_memory=True)
        biases_tensor = torch.stack(
            tuple(bias_to_index.keys()),
            out=biases_tensor,
        )
        biases_tensor_cuda = biases_tensor.to(logits.device, non_blocking=True)

        biases_tensor_cuda = torch.index_select(biases_tensor_cuda, 0,
                                                bias_gather_indices_cuda)
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
        log_probs_host: torch.Tensor | None = None,
        req_num_steps: torch.Tensor,
        req_offsets: torch.Tensor,
        steps_dim_size: int,
        token_dtype: torch.dtype,
    ) -> _BatchedSamplingResult:
        requests_by_strategy = _group_requests_by_sampling_strategy(
            requests, pin_memory=True)
        generator_cuda = self.get_generator(cuda_device)

        # FIXME: This check should/could be performed in ModelDrafter.prepare_draft_tokens
        #
        # NB: Currently, "d2t" is applied to draft tokens, but not to draft logits,
        #     breaking _process_draft_tokens_rejection_sampling.
        needs_d2t = "d2t" in model_outputs
        if needs_d2t and (len(requests_by_strategy) > 1 or
                          (requests_by_strategy
                           and next(iter(requests_by_strategy)) != GREEDY)):
            raise ValueError("d2t does not yet support non-greedy sampling")

        # Indexer for accessing tokens in 'logits_cuda', corresponding to the
        # requests in 'requests'.
        logits_cuda_indexer = _PackedStepIndexer(
            num_steps=req_num_steps,
            max_steps=steps_dim_size,
            req_offsets=req_offsets,
        )

        batched_results: list[tuple[torch.Tensor, torch.Tensor,
                                    torch.Tensor]] = []
        py_draft_logits_indices: list[tuple[LlmRequest, torch.Tensor]] = [
        ]  # see _BatchedSamplingResult for details
        softmax_index_offset = 0
        for strategy, group_req_indices in requests_by_strategy.items():
            # group_req_indices: Indices of 'requests' entries having the same sampling
            # strategy, ordered ascending.

            # Indices of 'group_req_indices' entries corresponding to requests
            # with draft logits.
            speculation_group_indices = [
                grp_idx
                for grp_idx, glob_req_idx in enumerate(group_req_indices)
                if requests[glob_req_idx].py_draft_logits is not None
            ]
            # To skip softmax computation where it is not needed, track
            #   softmax_req_indices: Indices of 'requests' entries requesting probs
            #   softmax_grp_indices: Indices of 'speculation_group_indices' entries requesting probs
            #   speculation_softmax_indices: Indices of 'softmax_grp_indices' entries corresponding
            #                                to requests with draft logits.
            if log_probs_host is not None:
                softmax_req_indices = group_req_indices
                softmax_grp_indices = torch.arange(len(group_req_indices),
                                                   dtype=torch.int32)
                speculation_softmax_indices = torch.tensor(
                    speculation_group_indices, dtype=torch.int32)
            else:
                speculation_group_indices_tensor = torch.tensor(
                    speculation_group_indices, dtype=torch.int32)
                softmax_req_indices = group_req_indices[
                    speculation_group_indices_tensor]
                softmax_grp_indices = speculation_group_indices_tensor
                speculation_softmax_indices = torch.arange(
                    len(speculation_group_indices), dtype=torch.int32)

            group_logits_cuda_indices = logits_cuda_indexer[group_req_indices]
            if group_logits_cuda_indices.numel() != logits_cuda.size(0):
                group_logits_cuda_indices_cuda = group_logits_cuda_indices.to(
                    device=logits_cuda.device, non_blocking=True)
                group_logits_cuda = logits_cuda[group_logits_cuda_indices_cuda]
            else:
                group_logits_cuda = logits_cuda

            # Indexer for accessing tokens in 'group_logits_cuda' (and 'group_next_tokens_cuda')
            # corresponding to the requests in 'group_req_indices'.
            group_logits_cuda_indexer = _PackedStepIndexer(
                num_steps=req_num_steps[group_req_indices],
                max_steps=steps_dim_size)
            softmax_group_indices_cuda = group_logits_cuda_indexer[
                softmax_grp_indices].to(device=logits_cuda.device,
                                        non_blocking=True)

            # Indexer for accessing tokens in 'group_softmax_cuda' corresponding to the
            # requests in 'softmax_req_indices'.
            if softmax_req_indices is not group_req_indices:
                group_softmax_cuda_indexer = _PackedStepIndexer(
                    num_steps=req_num_steps[softmax_req_indices],
                    max_steps=steps_dim_size)
            else:
                group_softmax_cuda_indexer = group_logits_cuda_indexer

            py_draft_logits_indices += [(
                requests[request_idx],
                softmax_index_offset + group_softmax_cuda_indexer[softmax_idx],
            ) for softmax_idx, request_idx in zip(
                speculation_softmax_indices,
                softmax_req_indices[speculation_softmax_indices],
            )]

            group_next_tokens_cuda, group_softmax_cuda = sample(
                strategy,
                group_logits_cuda,
                generator_cuda,
                softmax_indices=cast(torch.IntTensor,
                                     softmax_group_indices_cuda),
            )
            group_next_tokens_cuda_int = group_next_tokens_cuda.to(
                dtype=token_dtype, non_blocking=True)
            batched_results.append(
                (group_req_indices, group_next_tokens_cuda_int,
                 group_softmax_cuda))
            softmax_index_offset += group_softmax_cuda.size(0)
        # Batched sampling results, see _BatchedSamplingResult for details.
        if len(batched_results) > 1:
            batch_req_indices = torch.cat([res[0] for res in batched_results])
            batch_next_tokens_cuda_int = torch.cat(
                [res[1] for res in batched_results])
            batch_softmax_cuda = torch.cat([res[2] for res in batched_results])
        else:
            (batch_req_indices, batch_next_tokens_cuda_int,
             batch_softmax_cuda), = batched_results

        # FIXME: This should be done in ModelDrafter.prepare_draft_tokens, but for performance
        #        parity py_draft_tokens might need to be replaced / backed by a torch.Tensor, so
        #        that d2t can be applied in a batched manner similar to the code below.
        if needs_d2t:
            # NB: The sampler is either called directly by PyExecutor, for the target model,
            #     or by ModelDrafter.prepare_draft_tokens(), for the draft model. In the former
            #     case there are 1 + get_draft_token_length(request) tokens per request. In the
            #     latter case, only there is always only 1 token per request because draft
            #     tokens are sampled one-by-one.
            self._apply_d2t(batch_next_tokens_cuda_int, model_outputs)

        return _BatchedSamplingResult(
            batch_req_indices=batch_req_indices,
            batch_next_tokens_cuda_int=batch_next_tokens_cuda_int,
            batch_softmax_cuda=batch_softmax_cuda,
            py_draft_logits_indices=py_draft_logits_indices,
        )

    def _unbatch_sampling_results(
        self,
        batched_sampling_result: _BatchedSamplingResult,
        *,
        new_tokens_cuda: torch.Tensor,
        req_num_steps: torch.Tensor,
        seq_slots: torch.Tensor,
        log_probs_host: torch.Tensor | None = None,
    ) -> torch.Tensor:
        beam = self.BEAM
        assert beam == 0, "beam_width != 1 not supported"

        batch_req_indices = batched_sampling_result.batch_req_indices
        batch_next_tokens_cuda_int = batched_sampling_result.batch_next_tokens_cuda_int
        batch_softmax_cuda = batched_sampling_result.batch_softmax_cuda
        py_draft_logits_indices = batched_sampling_result.py_draft_logits_indices

        def _dims_canonically_ordered(t: torch.Tensor) -> bool:
            return len(t.dim_order(
                ambiguity_check=[torch.contiguous_format])) == t.ndim

        # Assert destination tensor dimensions are canonically ordered ("row"-major); this
        # matters for element ordering in the .view(...).scatter_(...) calls below.
        assert _dims_canonically_ordered(new_tokens_cuda)
        assert log_probs_host is None or _dims_canonically_ordered(
            log_probs_host)

        #     new_tokens_cuda indexed by
        #         slice(0, steps), slot, beam
        #     log_probs_host indexed by
        #         slot, beam, slice(0, steps)
        #     batch_... tensors indexed by slice(batch_req_index, batch_req_index + steps)
        #
        if log_probs_host is not None:
            assert new_tokens_cuda.size(0) == log_probs_host.size(-2)

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
            new_tokens_cuda.device, non_blocking=True)
        new_tokens_cuda.view(-1,
                             *new_tokens_cuda.shape[2:])[:, beam, ...].scatter_(
                                 0, batch_dest_indices_1d_cuda,
                                 batch_next_tokens_cuda_int)
        new_tokens_host = new_tokens_cuda.to("cpu", non_blocking=True)
        # NB: In order to avoid a scatter_ on the host and the necessary D2H copy + synchronization,
        #     the 'step' and 'seq_slot' dimensions are unpacked on GPU and later asynchronously
        #     copied into the destination buffer. Note that this overwrites all 'step' and token slots for the
        #     requests in 'requests' (passed to _process_requests). In fact, the current implementation
        #     even overwrites the destination tensors completely (including slices corresponding to request
        #     slots not present in 'requests', cf. 'FIXME' below).
        if log_probs_host is not None:
            # FIXME: If log_probs_host were indexed by request indices, rather than request slots, this
            #        tensor could be packed densely along the request axis.
            log_probs_cuda = torch.empty_like(
                log_probs_host, device=batch_dest_indices_1d_cuda.device)
            # FIXME: Needs a separate indexer because tensor layout differs from new_tokens_cuda
            batch_dest_probs_cuda_indexer = _UnpackedStepIndexer(
                seq_slots=seq_slots[batch_req_indices],
                num_steps=req_num_steps[batch_req_indices],
                steps_dim_size=new_tokens_cuda.size(0),
                slots_dim_size=new_tokens_cuda.size(1),
                dim_order=_UnpackedStepIndexer.DimOrder.SLOT_MAJOR,
                index_dtype=torch.int64,  # enforced by Tensor.scatter_
            )
            batch_dest_probs_indices_cuda = batch_dest_probs_cuda_indexer[:].to(
                batch_softmax_cuda.device, non_blocking=True)
            # NB: torch.arange is needed to enable "advanced indexing",
            #   cf. https://numpy.org/devdocs/user/basics.indexing.html#integer-array-indexing
            batch_token_probs = batch_softmax_cuda[
                torch.arange(batch_softmax_cuda.size(0),
                             device=batch_softmax_cuda.device,
                             dtype=torch.int32), batch_next_tokens_cuda_int]
            log_probs_cuda[:, beam,
                           ...].view(-1, *log_probs_cuda.shape[3:]).scatter_(
                               0, batch_dest_probs_indices_cuda,
                               torch.log(batch_token_probs))
            log_probs_host.copy_(log_probs_cuda, non_blocking=True)
        # For requests with LlmRequest.py_draft_logits, return py_target_probs
        for request, batch_softmax_index_cuda in py_draft_logits_indices:
            request.py_target_probs = batch_softmax_cuda[
                batch_softmax_index_cuda].clone()

        return new_tokens_host

    @staticmethod
    @torch.inference_mode()
    def _apply_min_length_penalty(logits: torch.Tensor,
                                  requests: list[LlmRequest],
                                  num_steps: list[int]) -> torch.Tensor:
        """Inplace apply min_length_penalty to logits.

            Args:
                logits: The logits to apply min length penalty to
                requests: The requests to apply min length penalty to
                num_steps: The number of steps per request

            Returns:
                The logits with min length penalty applied
        """
        if any(r.py_min_length and r.max_beam_num_tokens < r.py_min_length[0]
               for r in requests):
            current_offset = 0
            for index, r in enumerate(requests):
                if r.py_min_length:
                    for step in range(num_steps[index]):
                        if r.max_beam_num_tokens + step < r.py_min_length[0]:
                            logits[current_offset + step,
                                   r.py_end_id] = float('-inf')
                        else:
                            #early exit
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
    ) -> torch.Tensor:
        # raw_logits should contain only the generated logits.
        # If return context logits is requested, select only the generated logits.
        #
        # NB: Context request logits always precede generation request logits, also
        #     requests == scheduled_requests.context_requests + scheduled_requests.generation_requests
        if any(r.py_return_context_logits
               for r in scheduled_requests.context_requests):
            assert len(num_context_logits_prefix_sum) == len(
                scheduled_requests.context_requests) + 1
            req_num_generation_steps_cuda = req_num_generation_steps.to(
                raw_logits_cuda.device, non_blocking=True)
            context_req_offsets_cuda = torch.tensor(
                num_context_logits_prefix_sum,
                dtype=torch.int32,
                pin_memory=True).to(device=raw_logits_cuda.device,
                                    non_blocking=True)

            # Since the goal is to keep the req_num_steps[i] last tokens for each requests[i],
            # only end-offsets of the token storage locations matter.
            next_context_req_offsets_cuda = context_req_offsets_cuda.roll(
                -1)  # trailing '0' is overwritten below
            # Since logits for generation requests are densely packed, cover them all by a single
            # fictituous entry in 'context_req_offsets_cuda'.
            if scheduled_requests.generation_requests:
                req_num_steps_fictitious_cuda = req_num_generation_steps_cuda[:(
                    len(scheduled_requests.context_requests) + 1)].clone()
                req_num_steps_fictitious_cuda[
                    -1] = generation_requests_total_steps
                next_context_req_offsets_cuda[-1] = (
                    next_context_req_offsets_cuda[-2] +
                    req_num_steps_fictitious_cuda[-1])
            else:
                req_num_steps_fictitious_cuda = req_num_generation_steps_cuda[:len(
                    scheduled_requests.context_requests)]
                next_context_req_offsets_cuda = next_context_req_offsets_cuda[:
                                                                              -1]

            # Now, the generated tokens for context request i are at indices
            #    range(next_context_req_offsets_cuda[i] - req_num_steps_fictitious_cuda[i], next_context_req_offsets_cuda[i])
            # And if generation requests are present, those tensors each include a trailing entry selecting
            # all tokens/logits generated by all generation requests.
            indices_to_keep_cuda = torch_multi_arange(
                starts=(next_context_req_offsets_cuda -
                        req_num_steps_fictitious_cuda),
                ends=next_context_req_offsets_cuda,
            )

            raw_logits_cuda = raw_logits_cuda[indices_to_keep_cuda]
        return raw_logits_cuda

    @nvtx_range("_process_requests")
    def _process_requests(
            self,
            scheduled_requests: ScheduledRequests,
            model_outputs: dict[str, torch.Tensor],
            new_tokens_cuda: torch.Tensor,
            num_context_logits_prefix_sum: list[int],
            *,
            seq_slots: torch.Tensor,
            log_probs_host: torch.Tensor | None = None) -> torch.Tensor:
        seq_slots = seq_slots.to(dtype=torch.int32)  # int32 suffices here

        raw_logits_cuda = model_outputs["logits"]

        requests = scheduled_requests.all_requests()
        cuda_device = raw_logits_cuda.device
        req_num_steps_list = [
            1 + get_draft_token_length(req) for req in requests
        ]
        req_num_steps = torch.tensor(req_num_steps_list,
                                     dtype=torch.int32,
                                     pin_memory=True)
        # NB: These offsets consider generated tokens _only_ (draft and target, but not context)
        #     and are thus only correct after _select_generated_logits() below.
        req_offsets, sum_steps = _PackedStepIndexer.calculate_request_offsets(
            req_num_steps, pin_memory=True)

        raw_logits_cuda = self._select_generated_logits(
            scheduled_requests,
            raw_logits_cuda,
            req_num_generation_steps=req_num_steps,
            num_context_logits_prefix_sum=num_context_logits_prefix_sum,
            generation_requests_total_steps=(
                # NB: requests == scheduled_requests.context_requests + scheduled_requests.generation_requests
                sum_steps - cast(
                    int, req_offsets[len(
                        scheduled_requests.context_requests)].item())
                if scheduled_requests.generation_requests else 0),
        )

        # Handle embedding bias
        logits_cuda = raw_logits_cuda[:sum_steps]
        self._apply_embedding_bias(logits_cuda, requests, req_num_steps)

        logits_cuda = self._apply_min_length_penalty(logits_cuda, requests,
                                                     req_num_steps_list)

        # Perform sampling in batches
        batched_sampling_result = self._sample_batched_by_strategy(
            logits_cuda,
            requests,
            model_outputs,
            cuda_device=cuda_device,
            log_probs_host=log_probs_host,
            req_offsets=req_offsets,
            steps_dim_size=new_tokens_cuda.size(0),
            req_num_steps=req_num_steps,
            token_dtype=new_tokens_cuda.dtype,
        )

        # Fill results into output buffers
        new_tokens_host = self._unbatch_sampling_results(
            batched_sampling_result,
            new_tokens_cuda=new_tokens_cuda,
            log_probs_host=log_probs_host,
            req_num_steps=req_num_steps,
            seq_slots=seq_slots,
        )

        # NB: update_requests syncs w/ device computation and async D2H copies
        return new_tokens_host

    @override
    def should_provide_draft_probs(self, request: LlmRequest) -> bool:
        # Do not request draft probs when sampling is greedy.
        return _request_strategy(request) is not GREEDY


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
    host: SampleStateTensorsHostTRTLLM


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
        self.decoding_config = decoding_config if decoding_config else DecodingConfig(
            decoding_mode)
        max_attn_window = kv_cache_config.max_attention_window
        self.max_seq_len = max_seq_len
        self.max_attention_window = max(
            max_attn_window) if max_attn_window is not None else max_seq_len
        self.max_batch_size = max_batch_size
        self.max_beam_width = max_beam_width
        self.max_num_sequences = mapping.pp_size * max_batch_size
        self.max_seq_idle_microseconds = 180 * 1000 * 1000
        self.is_trt_overlap = not disable_overlap_scheduler
        self.num_micro_batches = mapping.pp_size if mapping.pp_size > 1 else (
            2 if self.is_trt_overlap else 1)
        self.micro_batch_idx = 0

        self.world_config = WorldConfig.mpi(mapping.gpus_per_node,
                                            mapping.tp_size, mapping.pp_size)
        self.model_config = ModelConfig(vocab_size, num_hidden_layers,
                                        num_hidden_layers, 0, num_heads,
                                        hidden_size, self.model_datatype)

        self._initialize_store()
        self._instantiate_algorithms()

    def _initialize_store(self):
        torch_stream = torch.cuda.current_stream().cuda_stream
        cuda_stream = CudaStream(torch_stream)
        buffer_manager = BufferManager(stream=torch_stream)

        self.store = {
            "torch_stream":
            torch_stream,
            "cuda_stream":
            cuda_stream,
            "buffer_manager":
            buffer_manager,
            "decoder_input_buffers": [
                DecoderInputBuffers(self.max_batch_size,
                                    self.MAX_DECODING_TOKENS, buffer_manager)
                for _ in range(self.num_micro_batches)
            ],
            "sequence_lengths_host":
            torch.empty((
                self.max_num_sequences,
                self.max_beam_width,
            ),
                        dtype=torch.int),
            "decoder_state":
            DecoderState(),
            "decoding_input": [None] * self.num_micro_batches,
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
            is_normalize_log_probs=False)
        self.algs.make_decoding_batch_input_output = MakeDecodingBatchInputOutput(
        )

    @torch.inference_mode()
    @nvtx_range("setup_sampler_step")
    def setup_sampler_step(self, requests):
        batch_slots, sampling_configs, lookahead_prompt, lookahead_algo_configs = self.algs.create_new_decoder_requests(
            self.model_config, self.world_config, self.decoding_config,
            requests.context_requests, self.logits_datatype,
            self.store["decoder_input_buffers"][self.micro_batch_idx],
            self.store["decoder_state"], self.store["cuda_stream"],
            self.algs.decoder.decoder_stream, self.max_seq_len,
            self.beam_width(requests.context_requests))

        local_batch_size = len(batch_slots)
        if local_batch_size > 0:
            sampling_config = make_sampling_config(sampling_configs)
            self.algs.decoder.underlying_decoder().setup(
                sampling_config, local_batch_size, batch_slots,
                self.store["decoder_state"].joint_decoding_output,
                self.model_config.data_type, lookahead_prompt,
                lookahead_algo_configs)

        adp = [
            r for r in requests.generation_requests if r.is_attention_dp_dummy
        ]
        batch_size = len(adp)
        if batch_size == 0:
            return
        config = make_sampling_config([r.sampling_config for r in adp])
        slots = torch.tensor([r.py_seq_slot for r in adp], dtype=torch.int32)
        self.algs.decoder.underlying_decoder().setup(config, batch_size, slots)

    def get_cache_indirection(self) -> torch.Tensor | None:
        return self.store["decoder_state"].cache_indirection_output

    def _update_cache_indirection_buffer(self,
                                         scheduled_requests: ScheduledRequests):
        # Copy cache indirection output to input
        for request in scheduled_requests.generation_requests:
            self.store["decoder_state"].cache_indirection_input[
                request.py_seq_slot].copy_(
                    self.store["decoder_state"].cache_indirection_output[
                        request.py_seq_slot],
                    non_blocking=True)

    @torch.inference_mode()
    @nvtx_range("sample_async")
    @override
    def sample_async(
            self, scheduled_requests: ScheduledRequests, model_outputs,
            num_context_logits_prefix_sum: list[int]) -> SampleStateTRTLLM:

        batch_size = scheduled_requests.batch_size
        beam_width = self.beam_width(scheduled_requests.all_requests())
        if (batch_size > 1 and beam_width > 1
                and any(request.py_return_log_probs
                        for request in scheduled_requests.all_requests())):
            raise ValueError(
                "Beam search is not supported for multiple prompts and logprobs"
            )

        self.setup_sampler_step(scheduled_requests)

        # For beam search, cache indirection needs to be updated
        if beam_width > 1:
            self._update_cache_indirection_buffer(scheduled_requests)

        self.store["decoding_input"][
            self.micro_batch_idx] = make_decoding_batch_input(
                scheduled_requests.context_requests,
                scheduled_requests.generation_requests, model_outputs["logits"],
                beam_width, num_context_logits_prefix_sum,
                self.store["decoder_input_buffers"][self.micro_batch_idx],
                self.store["decoder_state"], self.store["buffer_manager"])

        self.algs.decoder.forward_async(
            self.store["decoder_state"],
            self.store["decoding_input"][self.micro_batch_idx])

        finalize_events = {}
        gathered_ids = None
        if beam_width > 1:
            finished_sum_device = self.store["decoder_state"].finished_sum

            for request in scheduled_requests.all_requests():
                if request.is_context_init_state:
                    continue
                if finished_sum_device[request.seq_slot] == beam_width:
                    finalize_events[
                        request.request_id] = self._finalize_request(
                            request, False)
                elif request.streaming:
                    finalize_events[
                        request.request_id] = self._finalize_request(
                            request, True)
            gathered_ids = self.store["decoder_state"].gathered_ids.to(
                'cpu', non_blocking=True)
        new_output_tokens = self.store["decoder_state"].all_new_tokens.to(
            'cpu', non_blocking=True)
        finished_sum = self.store["decoder_state"].finished_sum.to(
            'cpu', non_blocking=True)
        finish_reasons = self.store["decoder_state"].finish_reasons.to(
            'cpu', non_blocking=True)
        sequence_lengths = self.store["decoder_state"].sequence_lengths.to(
            'cpu', non_blocking=True)

        log_probs = None
        cum_log_probs = None
        if any(request.py_return_log_probs
               for request in scheduled_requests.all_requests()):
            log_probs = self.store["decoder_state"].log_probs.to(
                'cpu', non_blocking=True)
            cum_log_probs = self.store["decoder_state"].cum_log_probs.to(
                'cpu', non_blocking=True)

        device = SampleStateTensors(
            new_tokens=self.store["decoder_state"].all_new_tokens)

        host = SampleStateTensorsHostTRTLLM(new_tokens=new_output_tokens,
                                            finished_sum=finished_sum,
                                            finish_reasons=finish_reasons,
                                            sequence_lengths=sequence_lengths,
                                            log_probs=log_probs,
                                            cum_log_probs=cum_log_probs,
                                            gathered_ids=gathered_ids)

        sampler_event = torch.cuda.Event()
        sampler_event.record()

        self.micro_batch_idx = (self.micro_batch_idx +
                                1) % self.num_micro_batches

        return SampleStateTRTLLM(scheduled_requests=scheduled_requests,
                                 device=device,
                                 host=host,
                                 sampler_event=sampler_event,
                                 finalize_events=finalize_events)

    @torch.inference_mode()
    @override
    def update_requests(self, state: SampleStateTRTLLM):
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
        sequence_lengths_host_data = state.host.sequence_lengths.flatten(
        ).tolist()
        finish_reasons = state.host.finish_reasons.flatten().tolist()
        log_probs_host = state.host.log_probs.tolist(
        ) if state.host.log_probs is not None else None
        cum_log_probs_host = state.host.cum_log_probs.tolist(
        ) if state.host.cum_log_probs is not None else None

        reqs = [
            r for r in state.scheduled_requests.context_requests
            if not r.is_context_init_state
        ] + [
            r for r in state.scheduled_requests.generation_requests
            if not r.is_generation_complete_state
        ]

        reqs_with_new_tokens = [
            r for r in reqs
            if (sequence_lengths_host_data[r.py_seq_slot] > r.get_num_tokens(0))
        ]

        # Add new tokens
        new_tokens = [
            new_tokens_host[r.py_seq_slot] for r in reqs_with_new_tokens
        ]
        add_new_tokens_to_requests(reqs_with_new_tokens, new_tokens, 0)

        # Log probs
        for request in reqs_with_new_tokens:
            if request.py_return_log_probs:
                seq_slot = request.py_seq_slot
                seq_len = sequence_lengths_host_data[seq_slot]
                begin_log_probs_offset = request.prompt_len
                current_token = seq_len - request.prompt_len - 1
                log_probs = [{
                    new_tokens_host[seq_slot]:
                    Logprob(logprob=log_probs_host[seq_slot][0][
                        begin_log_probs_offset + current_token],
                            rank=1)
                }]
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
    def update_requests_multiple_beams_or_drafting(self,
                                                   state: SampleStateTRTLLM,
                                                   beam_width: int):
        new_tokens_host = state.host.new_tokens
        finished_sum_host = state.host.finished_sum.tolist()
        finish_reasons = state.host.finish_reasons.flatten().tolist()
        sequence_lengths_host_data = state.host.sequence_lengths.flatten(
        ).tolist()
        cum_log_probs_host = state.host.cum_log_probs.tolist(
        ) if state.host.cum_log_probs is not None else None
        log_probs_host = state.host.log_probs.tolist(
        ) if state.host.log_probs is not None else None
        finalize_events = state.finalize_events

        reqs = [
            r for r in state.scheduled_requests.context_requests
            if not r.is_context_init_state
        ] + [
            r for r in state.scheduled_requests.generation_requests
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
                seq_len = sequence_lengths_host_data[seq_slot * beam_width +
                                                     beam]
                num_new_tokens[beam] = min(
                    num_generated_tokens,
                    seq_len - request.get_num_tokens(beam))

                for step in range(num_new_tokens[beam]):
                    new_token = add_token(request,
                                          new_tokens_host,
                                          beam=beam,
                                          step=step)

                    if request.py_return_log_probs:
                        assert state.host.log_probs is not None
                        # NOTE: Log probs with drafting has not been tested yet.
                        begin_log_probs_offset = request.prompt_len if request.sampling_config.beam_width == 1 else 0
                        current_token = seq_len - request.prompt_len - num_new_tokens[
                            beam] + step
                        log_probs[beam].append({
                            new_token:
                            Logprob(logprob=log_probs_host[seq_slot][beam][
                                begin_log_probs_offset + current_token],
                                    rank=1)
                        })

                if request.py_return_log_probs:
                    cum_log_probs.append(cum_log_probs_host[seq_slot][beam])

                finished_state = FinishedState(
                    finish_reasons[seq_slot * beam_width + beam])
                if finished_state.is_finished:
                    finish_reason = finished_state.to_finish_reason()
                    request.set_finished_reason(finish_reason, beam)

            if request.py_return_log_probs:
                request.py_result.append_log_probs(log_probs, cum_log_probs)

            # Set number of tokens predicted per runtime iteration. Will be > 1 for speculative decoding.
            request.update_num_tokens_per_iteration(
                request.max_beam_num_tokens - current_num_of_tokens,
                self.model_config)

            # Increment the decoding iteration counter
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                request.py_decoding_iter += 1

            if finished_sum_host[seq_slot] == beam_width:
                request.state = LlmRequestState.GENERATION_COMPLETE
        for request in reqs:
            if finalize_events is not None and request.request_id in finalize_events:
                self._post_process_request(request, state)

    def _finalize_request(self, request: LlmRequest, streaming: bool):
        """ Finalizes the request. This is necessary for beam search. """
        seq_slot = request.py_seq_slot
        event = self.algs.decoder.finalize(self.store["decoder_state"],
                                           seq_slot, request.sampling_config,
                                           streaming)
        return event

    def _post_process_request(self, request: LlmRequest,
                              state: SampleStateTRTLLM):
        """ Post Process the request. Updates the sequence according to the beam search results.
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
            generated_length = sequence_lengths_host[
                seq_slot, beam].item() - request.py_prompt_len
            end = begin + generated_length
            generated_tokens[beam] = output_ids_host[seq_slot, beam,
                                                     begin:end].tolist()

            # get the correct log probs for beam search
            if request.py_return_log_probs:
                cum_log_probs.append(cum_log_probs_host[seq_slot, beam].item())

                begin_log_probs_offset = request.prompt_len if request.sampling_config.beam_width == 1 else 0
                for current_token, token in enumerate(generated_tokens[beam]):
                    log_probs[beam].append({
                        token:
                        Logprob(
                            logprob=log_probs_host[seq_slot,
                                                   beam][begin_log_probs_offset
                                                         +
                                                         current_token].item(),
                            rank=1)
                    })
        if request.py_return_log_probs:
            request.py_result.set_log_probs(log_probs, cum_log_probs)

        request.set_generated_tokens(generated_tokens)
