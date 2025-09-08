from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import List, Literal, Optional, TypeAlias

import numpy as np
import torch

from tensorrt_llm._torch.pyexecutor.make_decoding_batch_input_output import \
    MakeDecodingBatchInputOutput
from tensorrt_llm._torch.pyexecutor.sampler_utils import (
    BEAM_0, SINGLE_BEAM_WIDTH, handle_stop_single_beam)
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


class EarlyStopSampler(Sampler):
    """
    Use for skipping decoding step for non generation model,
    such as encoder-only model (e.g., BERT) or reward models that only need context phase.
    """

    def sample_async(self, scheduled_requests: ScheduledRequests, model_outputs,
                     num_context_logits_prefix_sum: list[int]) -> SampleState:
        host = SampleStateTensors(new_tokens=torch.empty(0))
        return SampleState(scheduled_requests=scheduled_requests, host=host)

    def update_requests(self, state: SampleState) -> None:
        assert isinstance(state, SampleState)
        scheduled_requests = state.scheduled_requests
        assert (not scheduled_requests.generation_requests)
        for idx, request in enumerate(scheduled_requests.context_requests):
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)

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

    def sample_async(
            self, scheduled_requests: ScheduledRequests, model_outputs,
            num_context_logits_prefix_sum: list[int]
    ) -> SampleStateWithMMResult:
        # from model_outputs to MultimodalResult
        data = MultimodalResult(mm_embeddings=model_outputs['mm_embeddings'])
        return SampleStateWithMMResult(scheduled_requests=scheduled_requests,
                                       data=data)

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

    def is_generation_model(self) -> bool:
        return False


def top_k_sampling_batch(logits,
                         top_k=50,
                         generator: Optional[torch.Generator] = None):
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


def top_p_sampling_batch(logits: torch.Tensor,
                         top_p: float = 0.9,
                         temperature: float = 1.0,
                         generator: Optional[torch.Generator] = None):
    logits_dim = logits.dim()
    if logits_dim == 1:
        logits = logits.unsqueeze(0)
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
                               top_k: int,
                               top_p: float,
                               temperature: float = 1.0,
                               generator: Optional[torch.Generator] = None):
    logits_dim = logits.dim()
    if logits_dim == 1:
        logits = logits.unsqueeze(0)
    assert logits_dim == 2, "logits should be 2D: [batch_size, vocab_size]"
    if temperature != 0:
        logits = logits / max(temperature, 1e-5)
    batch_size, vocab_size = logits.size()
    # get first top_k logits of each sample and their indices
    if top_k > 0:
        values, indices = torch.topk(logits, top_k, dim=-1)
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


def greedy_search_sampling_batch(logits):
    next_tokens = torch.argmax(logits, dim=-1)
    softmax = torch.softmax(logits, dim=-1)
    return next_tokens, softmax


def get_rejected_indices(draft_probs: torch.Tensor, target_probs: torch.Tensor,
                         generator: torch.Generator, draft_tokens: list[int]):

    p = draft_probs[torch.arange(len(draft_tokens)), draft_tokens]
    q = target_probs[:-1]
    q = q[torch.arange(len(draft_tokens)), draft_tokens]
    accept_probs = torch.minimum(torch.ones(()), q / p)
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
Strategy = TopK | TopP | Greedy


def request_strategy(request: LlmRequest) -> Strategy:
    if request.sampling_config.top_k is not None and len(
            request.sampling_config.top_k
    ) > 0 and request.sampling_config.top_p is not None and len(
            request.sampling_config.top_p) > 0:
        return ("top_k_top_p", request.sampling_config.top_k[0],
                request.sampling_config.top_p[0],
                request.sampling_config.temperature[0])
    if request.sampling_config.top_p is not None and len(
            request.sampling_config.top_p) > 0:
        return ("top_p", request.sampling_config.top_p[0],
                request.sampling_config.temperature[0])
    elif request.sampling_config.top_k is not None and len(
            request.sampling_config.top_k) > 0:
        return ("top_k", request.sampling_config.top_k[0])
    else:
        return ("greedy", None)


def sampling_strategies(requests: Iterable[LlmRequest]) -> list[Strategy]:
    return [request_strategy(req) for req in requests]


def sample(strategy: Strategy,
           logits: torch.Tensor,
           generator: Optional[torch.Generator] = None):
    match strategy:
        case ("top_k", top_k):
            return top_k_sampling_batch(logits, top_k, generator)
        case ("top_p", top_p, temperature):
            return top_p_sampling_batch(logits, top_p, temperature, generator)
        case ("top_k_top_p", top_k, top_p, temperature):
            return top_k_top_p_sampling_batch(logits, top_k, top_p, temperature,
                                              generator)
        case ("greedy", None):
            return greedy_search_sampling_batch(logits)


def add_token(request: LlmRequest,
              new_tokens: torch.Tensor,
              *,
              beam: int,
              step: int = 0) -> int:
    seq_slot = request.py_seq_slot
    assert seq_slot is not None
    new_token = int(new_tokens[step, seq_slot, beam])
    request.add_new_token(new_token, beam)
    return new_token


def int_tensor(shape: tuple[int, ...], device: str = 'cuda') -> torch.Tensor:
    return torch.empty(shape, dtype=torch.int, device=device)


class TorchStore:

    def __init__(self, *, max_draft_len: int, max_num_sequences: int,
                 max_beam_width: int):
        self.max_draft_len = max_draft_len
        self.max_num_sequences = max_num_sequences
        self.max_beam_width = max_beam_width
        self.max_tokens = max_draft_len + 1
        assert max_beam_width == SINGLE_BEAM_WIDTH, "TorchSampler only supports beam_width = 1"
        self.new_tokens = int_tensor(
            (self.max_tokens, max_num_sequences, max_beam_width))
        """Shape: See cpp DecoderState.getAllNewTokens()"""
        self.finish_reasons = int_tensor(self.new_tokens.shape)

        # Helper tensors for finish_reasons:
        self._finish_reasons_nonzero_static_buffer = torch.empty(
            (self.max_tokens * max_num_sequences, 2),
            device='cuda',
            dtype=torch.int64)
        """Preallocate buffer needed for torch.nonzero_static(..., out=finish_reasons_nonzero_static_buffer), see `def _write_reason`"""
        self._reason_tensors = {
            reason:
            torch.tensor(reason.value,
                         dtype=self.finish_reasons.dtype,
                         device="cuda")
            for reason in [
                FinishReason.NOT_FINISHED, FinishReason.END_ID,
                FinishReason.STOP_WORDS, FinishReason.LENGTH,
                FinishReason.TIMED_OUT, FinishReason.CANCELLED
            ]  # `in FinishReason` clashes with PyBind11: `TypeError: 'pybind11_type' object is not iterable`
        }


@dataclass(kw_only=True)
class SampleStateTensorsHostTorch(SampleStateTensors):
    finish_reasons: torch.Tensor


@dataclass(kw_only=True)
class SampleStateTorch(SampleState):
    host: SampleStateTensorsHostTorch


class TorchSampler(Sampler):
    SampleState = SampleStateTorch

    def is_generation_model(self) -> bool:
        return True

    @dataclass(frozen=True, kw_only=True)
    class Args:
        max_seq_len: int
        max_draft_len: int
        max_num_sequences: int
        max_beam_width: int
        enable_mixed_sampler: bool

    def __init__(self, args: Args):
        self.max_seq_len = args.max_seq_len
        self.enable_mixed_sampler = args.enable_mixed_sampler

        # AutoDeploy build creates the sampler in inference mode,
        # which would disallow in-place mutating of new_tokens.
        # So, we temporarily exit inference mode.
        with torch.inference_mode(False):
            self.store = TorchStore(max_draft_len=args.max_draft_len,
                                    max_num_sequences=args.max_num_sequences,
                                    max_beam_width=args.max_beam_width)
        self.max_num_sequences = args.max_num_sequences
        self.max_tokens = self.store.max_tokens

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
        return self._generator

    def handle_logprobs(self, request: LlmRequest, state: SampleStateTorch, *,
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

    FinishReasons: TypeAlias = list[list[int]]
    """`(num_seq_slots, num_steps)`"""

    @classmethod
    def finish_if_reason(cls, request: LlmRequest,
                         finish_reasons: FinishReasons, *, step: int) -> bool:
        reason = FinishReason(finish_reasons[request.py_seq_slot][step])
        valid_reasons = {
            FinishReason.END_ID, FinishReason.LENGTH, FinishReason.STOP_WORDS
        }
        if reason in valid_reasons:
            request.finish_by(reason, BEAM_0)
            return True
        return False

    def _process_draft_tokens_greedy(self, request: LlmRequest, *,
                                     new_tokens: torch.Tensor,
                                     finish_reasons: FinishReasons) -> int:
        new_token = add_token(request, new_tokens, beam=BEAM_0)
        stop = self.finish_if_reason(request, finish_reasons, step=0)
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
                                  beam=BEAM_0,
                                  step=num_accepted)
            if self.finish_if_reason(request, finish_reasons,
                                     step=num_accepted):
                break
        return num_accepted

    def _process_draft_tokens_rejection_sampling(
            self, request: LlmRequest, new_tokens: torch.Tensor) -> int:
        """We cannot use finish_if_reason in _process_draft_tokens_rejection_sampling because it *writes to new_tokens*,
        rendering the finish reason calculation in sample_async stale (incorrect) for this batch"""
        sampling_strategy = request_strategy(request)
        generator = self.get_generator(request.py_draft_logits.device)
        _, draft_probs = sample(sampling_strategy,
                                request.py_draft_logits[0],
                                generator=generator)
        target_probs = request.py_target_probs
        rejected_indices = get_rejected_indices(draft_probs, target_probs,
                                                generator,
                                                request.py_draft_tokens)
        sample_last = True
        if rejected_indices.numel() == 0:
            num_initially_accepted = get_draft_token_length(request)
            sample_last = False
        else:
            num_initially_accepted = rejected_indices[0].item()
        num_accepted = num_initially_accepted
        for i in range(num_accepted):
            new_token = request.py_draft_tokens[i]
            new_tokens[i, request.seq_slot, BEAM_0] = new_token
            request.add_new_token(new_token, BEAM_0)
            if handle_stop_single_beam(request,
                                       new_token,
                                       max_seq_len=self.max_seq_len):
                num_accepted = i + 1
                return num_accepted
        if sample_last:
            new_token = sample_rejected(draft_probs, target_probs, generator,
                                        num_accepted)
            new_tokens[num_accepted, request.seq_slot, BEAM_0] = new_token
            request.add_new_token(new_token, BEAM_0)
            handle_stop_single_beam(request,
                                    new_token,
                                    max_seq_len=self.max_seq_len)
        else:
            new_token = add_token(request,
                                  new_tokens,
                                  beam=BEAM_0,
                                  step=num_accepted)
            handle_stop_single_beam(request,
                                    new_token,
                                    max_seq_len=self.max_seq_len)

        return num_accepted

    def update_requests(self, state: SampleStateTorch) -> None:
        assert isinstance(state, SampleStateTorch)
        if state.sampler_event:
            state.sampler_event.synchronize()
        new_tokens = state.host.new_tokens
        finish_reasons = state.host.finish_reasons[:, :, BEAM_0].T.tolist()

        for req in state.scheduled_requests.context_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE or req.context_remaining_length != 0:
                continue
            add_token(req, new_tokens, beam=BEAM_0)
            self.finish_if_reason(req, finish_reasons, step=0)
            self.handle_logprobs(req, state, beam=BEAM_0, count=1)
            req.py_decoding_iter += 1

        for req in state.scheduled_requests.generation_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            processed = 1
            if req.py_draft_logits is None:
                num_accepted = self._process_draft_tokens_greedy(
                    req, new_tokens=new_tokens, finish_reasons=finish_reasons)
            else:
                num_accepted = self._process_draft_tokens_rejection_sampling(
                    req, new_tokens)
            if get_draft_token_length(req) > 0:
                req.py_num_accepted_draft_tokens = num_accepted
                req.py_rewind_len = req.py_draft_pages_allocated - num_accepted
            processed += num_accepted
            self.handle_logprobs(req, state, beam=BEAM_0, count=processed)
            req.py_decoding_iter += 1

    def log_probs_host(self, scheduled_requests: ScheduledRequests):
        """Shape: In lockstep with TRTLLMSampler: https://github.com/NVIDIA/TensorRT-LLM/blob/cea5dd1e3883b18bf50901a7f196f50a9544c28c/cpp/include/tensorrt_llm/runtime/decoderState.h#L103"""
        if any(req.py_return_log_probs
               for req in scheduled_requests.all_requests()):
            return torch.empty(
                (self.max_num_sequences, SINGLE_BEAM_WIDTH, self.max_tokens),
                device="cpu",
                pin_memory=True)
        return None

    def sample_async(
            self, scheduled_requests: ScheduledRequests,
            model_outputs: dict[str, torch.Tensor],
            num_context_logits_prefix_sum: list[int]) -> SampleStateTorch:
        requests = scheduled_requests.all_requests()
        new_tokens = self.store.new_tokens
        finish_reasons = self.store.finish_reasons
        log_probs_host = self.log_probs_host(scheduled_requests)
        seq_slots_host = torch.tensor(
            [r.py_seq_slot for r in requests],
            dtype=torch.int64,  # for index_fill_
            pin_memory=True)
        seq_slots = seq_slots_host.to(device="cuda", non_blocking=True)
        self._process_requests(scheduled_requests,
                               model_outputs,
                               new_tokens,
                               num_context_logits_prefix_sum,
                               seq_slots=seq_slots,
                               seq_slots_host=seq_slots_host,
                               log_probs_host=log_probs_host)
        self._write_finish_reasons(requests,
                                   finish_reasons=finish_reasons,
                                   seq_slots=seq_slots,
                                   new_tokens=new_tokens)

        new_tokens_host = new_tokens.to(device="cpu", non_blocking=True)
        finish_reasons_host = finish_reasons.to(device="cpu", non_blocking=True)
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        return SampleStateTorch(
            scheduled_requests=scheduled_requests,
            device=SampleStateTensors(new_tokens=new_tokens),
            host=SampleStateTensorsHostTorch(
                new_tokens=new_tokens_host,
                log_probs=log_probs_host,
                finish_reasons=finish_reasons_host,
            ),
            sampler_event=sampler_event,
        )

    @staticmethod
    def append_eagle3(tokens: torch.Tensor, model_outputs):
        if "d2t" in model_outputs:
            d2t = model_outputs["d2t"][tokens]
            tokens += d2t

    @staticmethod
    def _apply_embedding_bias(
            logits: torch.Tensor,
            requests: list[LlmRequest],
            steps_per_request: list[int] = None) -> torch.Tensor:
        """Apply embedding bias (aka logit bias) to logits.
        If steps_per_request is None, assumes 1 step per request (non-batched path).
        """
        # Collect biases and their associated data
        bias_list = []
        bias_data = []  # Either indices (fast path) or steps (batched path)

        for i, req in enumerate(requests):
            bias = req._py_embedding_bias_1d
            if bias is not None:
                bias_list.append(bias)
                bias_data.append(i if steps_per_request is
                                 None else steps_per_request[i])

        if not bias_list:
            return logits

        bias_tensor = torch.stack(bias_list).to(logits.device,
                                                non_blocking=True)
        logits = logits.clone()

        if steps_per_request is None:
            # Fast path: direct indexing
            indices = torch.tensor(bias_data, device=logits.device)
            logits[indices] += bias_tensor
        else:
            # Batched path: expand biases and use boolean mask
            expanded_biases = torch.repeat_interleave(bias_tensor,
                                                      torch.tensor(
                                                          bias_data,
                                                          device=logits.device),
                                                      dim=0)

            mask = torch.zeros(sum(steps_per_request),
                               dtype=torch.bool,
                               device=logits.device)
            offset = 0
            for i, req in enumerate(requests):
                steps = steps_per_request[i]
                if req._py_embedding_bias_1d is not None:
                    mask[offset:offset + steps] = True
                offset += steps

            logits[mask] += expanded_biases

        return logits

    @staticmethod
    def _longest_stop_word_len(requests: Iterable[LlmRequest]) -> int:
        max_stop_word_len = 0
        for req in requests:
            _, cumsum = req.py_stop_words_list
            if -1 in cumsum:
                cumsum = cumsum[:cumsum.index(-1)]
            request_max_stop_word_len = np.max(np.diff(cumsum, prepend=0),
                                               initial=0)
            max_stop_word_len = max(max_stop_word_len,
                                    request_max_stop_word_len)
        return max_stop_word_len

    @staticmethod
    def _requests_with_stop_words(
            requests: list[LlmRequest]) -> list[LlmRequest]:
        return [
            r for r in requests if (r.py_stop_words_list is not None
                                    and len(r.py_stop_words_list[0]) > 0)
        ]

    def _write_reason(self, finish_reasons: torch.Tensor, reason: FinishReason,
                      *, where: torch.Tensor, seq_slots: torch.Tensor) -> None:
        """Avoid GPU<->CPU syncs via:
        ### `nonzero_static` [REF-A], see: https://ianbarber.blog/2024/12/18/nonzero_static-in-pytorch/.
        - `nonzero` syncs (frontend needs result size).
        - `nonzero_static` pads with dummy entries (`fill_value`), written into a prealloc buffer (max_num_sequences, 2).
        - Need to drop padding, but `buffer[buffer!=fill_value]`, `buffer[:count_nonzero]`, `buffer[:sum]` all sync.

        ### Hack:
        1. Use `fill_value=0`, so padding is `[..., [0,0], [0,0]]`.
        2. Write blindly to `finish_reasons` [REF-B]. Only `[seq_slot[0],0]` might have wrong values written to it, because of the padding entries.
        3. Save `[seq_slot[0],0]` in `before_write` [REF-C], restore if `where[0][0]` is `False` [REF-D].
        """
        assert seq_slots.is_cuda and where.is_cuda
        assert seq_slots.shape[0] == where.shape[1]
        first_slot = seq_slots[0].unsqueeze(0)
        before_write = finish_reasons[0][:].index_select(
            0, first_slot).squeeze()  # REF-C
        reason_tensor = self.store._reason_tensors[reason]
        buffer = self.store._finish_reasons_nonzero_static_buffer
        size = buffer.shape[0]
        torch.nonzero_static(where, size=size, fill_value=0,
                             out=buffer)  # REF-A
        r, c = buffer[:, 0], buffer[:, 1]
        finish_reasons[r, seq_slots[c], BEAM_0] = reason_tensor  # REF-B

        correct = torch.where(~where[0, 0], before_write, reason_tensor).view(1)
        assert correct.is_cuda
        finish_reasons[0, first_slot, BEAM_0] = correct  # REF-D

    def _write_finish_reasons(self, requests: list[LlmRequest], *,
                              finish_reasons: torch.Tensor,
                              seq_slots: torch.Tensor,
                              new_tokens: torch.Tensor) -> None:
        """later _write_reason overwrites earlier, in reverse precedence order"""
        tokens = new_tokens[:, seq_slots, BEAM_0]
        # we need to fill with NOT_FINISHED so we can differentiate between previous requests that had the same seq slot
        finish_reasons.index_fill_(1, seq_slots,
                                   FinishReason.NOT_FINISHED.value)

        if with_stop_words := self._requests_with_stop_words(requests):
            stop_seq_slots = torch.tensor(
                [r.py_seq_slot for r in with_stop_words],
                pin_memory=True).to("cuda", non_blocking=True)
            stop_tokens = new_tokens[:, stop_seq_slots, BEAM_0]
            self._write_reason(
                finish_reasons,
                FinishReason.STOP_WORDS,
                where=self._are_stop_words(with_stop_words, stop_tokens),
                seq_slots=stop_seq_slots,
            )

        self._write_reason(
            finish_reasons,
            FinishReason.LENGTH,
            where=self._are_max_length(requests),
            seq_slots=seq_slots,
        )

        self._write_reason(
            finish_reasons,
            FinishReason.END_ID,
            where=self._are_end_id(requests, tokens),
            seq_slots=seq_slots,
        )

    def _are_end_id(self, requests: list[LlmRequest],
                    tokens: torch.Tensor) -> torch.Tensor:
        end_ids_tensor = torch.tensor(
            [([req.py_end_id if req.py_end_id is not None else -1] *
              self.max_tokens) for req in requests],
            pin_memory=True,
            dtype=tokens.dtype).T.to(device="cuda", non_blocking=True)
        return tokens == end_ids_tensor

    def _are_max_length(self, requests: list[LlmRequest]) -> torch.Tensor:
        lengths_tensor = torch.tensor([[
            ((req.get_num_tokens(BEAM_0) + num_tokens) - req.py_orig_prompt_len)
            for num_tokens in range(1, self.max_tokens + 1)
        ] for req in requests])
        max_lengths_tensor = torch.tensor([
            ([min(req.py_max_new_tokens, self.max_seq_len)] * self.max_tokens)
            for req in requests
        ])
        return (lengths_tensor
                >= max_lengths_tensor).T.pin_memory().to(device="cuda",
                                                         non_blocking=True)

    _PAD_ID = -1
    """Pad with negative, doesn't matter what"""

    @cached_property
    def _pad_steps_mask(self):
        square = torch.ones(self.max_tokens, self.max_tokens, dtype=torch.bool)
        pad_id = torch.tensor(self._PAD_ID)
        mask = torch.where(square.tril(), torch.tensor(1), pad_id)
        mask.pin_memory()
        return mask.to("cuda", non_blocking=True)

    def _padded_old_tokens(self,
                           requests: list[LlmRequest],
                           new_tokens: torch.Tensor,
                           pad_id: int = _PAD_ID) -> torch.Tensor:
        # TODO: make sure only the lookback tokens are pulled into the list
        longest = self._longest_stop_word_len(requests)
        assert longest > 0, f"{longest=}, longest stop word length should be greater than 0, as this code path is only reached with requests with stop words"
        lookback = longest - 1
        old_tokens = []
        for request in requests:
            old = request.get_tokens(BEAM_0)[-lookback:] if lookback > 0 else []
            padded = [pad_id] * max(0, lookback - len(old)) + old
            old_tokens.append([padded] * self.max_tokens)
        old_tokens_tensor = torch.tensor(old_tokens,
                                         pin_memory=True).to("cuda",
                                                             non_blocking=True)
        assert old_tokens_tensor.shape == (
            len(requests), self.max_tokens, lookback
        ), f"{old_tokens_tensor.shape} != ({len(requests)=}, {self.max_tokens=}, {lookback=})"
        new_tokens = new_tokens.T.unsqueeze(1) * self._pad_steps_mask
        ret = torch.cat((old_tokens_tensor, new_tokens), dim=-1)
        assert ret.shape == (
            len(requests), self.max_tokens, lookback + self.max_tokens
        ), f"{ret.shape} != ({len(requests)=}, {self.max_tokens=}, {lookback + self.max_tokens=})"
        return ret

    def _are_stop_words(self, requests: list[LlmRequest],
                        tokens: torch.Tensor) -> torch.Tensor:
        per_step = torch.zeros((self.max_tokens, len(requests)),
                               dtype=torch.bool,
                               pin_memory=True).to("cuda", non_blocking=True)

        padded_tokens = self._padded_old_tokens(requests, tokens)

        def request_stop_words(request: LlmRequest, new_tokens: torch.Tensor):
            swl, ends = request.py_stop_words_list
            if -1 in ends:
                ends = ends[:ends.index(-1)]
            lens = np.diff(ends, prepend=0)
            lens_device = torch.tensor(list(lens),
                                       pin_memory=True).to("cuda",
                                                           non_blocking=True)
            max_len = np.max(lens)

            words = torch.zeros(len(lens),
                                max_len,
                                dtype=torch.int32,
                                pin_memory=True)
            for step, (start, l) in enumerate(zip([0] + ends, lens)):
                words[step, :l] = torch.tensor(swl[start:start + l],
                                               dtype=torch.int32)
            words_device = words.to("cuda", non_blocking=True)

            for step, step_seq in enumerate(new_tokens):
                for word, L in zip(words_device, lens_device):
                    truncated_seq = step_seq[step_seq >= 0][-L:]
                    if torch.equal(truncated_seq, word[-L:]):
                        # We don't care about subsequent steps because we already found a stop word match
                        return step
            return None

        for request_idx, request in enumerate(requests):
            step = request_stop_words(request, padded_tokens[request_idx])
            if step is not None:
                per_step[step][request_idx] = True
        return per_step

    def _process_requests(self,
                          scheduled_requests: ScheduledRequests,
                          model_outputs: dict[str, torch.Tensor],
                          new_tokens: torch.Tensor,
                          num_context_logits_prefix_sum: list[int],
                          *,
                          seq_slots: torch.Tensor,
                          seq_slots_host: torch.Tensor,
                          log_probs_host: torch.Tensor | None = None):

        # raw_logits should contain only the logits from the gen requests.
        # If return context logits is requested, fetch only the logits from gen requests.
        if any(r.py_return_context_logits
               for r in scheduled_requests.context_requests):
            gen_logits_indices = []
            total_context_logits = num_context_logits_prefix_sum[-1]
            for i in range(len(scheduled_requests.context_requests)):
                gen_logits_indices.append(num_context_logits_prefix_sum[i + 1] -
                                          1)
            gen_logits_indices.extend(
                range(
                    total_context_logits, total_context_logits +
                    len(scheduled_requests.generation_requests)))
            raw_logits = model_outputs["logits"][gen_logits_indices]
        else:
            raw_logits = model_outputs["logits"]

        requests = scheduled_requests.all_requests()
        num_steps = [1 + get_draft_token_length(req) for req in requests]
        sum_steps = sum(num_steps)
        no_draft_tokens = len(requests) == sum_steps
        fast_path = not self.enable_mixed_sampler and no_draft_tokens and log_probs_host is None

        if fast_path:
            logits = raw_logits[:len(requests)]
            logits = self._apply_embedding_bias(logits, requests)
            next_tokens = torch.argmax(logits, dim=-1)
            self.append_eagle3(next_tokens, model_outputs)
            int_next_tokens = next_tokens.to(torch.int, non_blocking=True)
            next_tokens = int_next_tokens.view(1, -1, SINGLE_BEAM_WIDTH)
            new_tokens[:1].index_copy_(1, seq_slots, next_tokens)
            return

        strategies = sampling_strategies(requests)
        batched_next_tokens, batched_softmax = None, None
        batched_strategy: Strategy | None = GREEDY
        if self.enable_mixed_sampler:
            assert "d2t" not in model_outputs, "eagle3 does not yet support non-greedy sampling"
            if len(set(strategies)) == 1:
                batched_strategy = strategies[0]
            else:
                batched_strategy = None
        generator = self.get_generator(raw_logits.device)
        if batched_strategy is not None:
            logits = raw_logits[:sum_steps]
            # Collect steps per request for batched strategy
            steps_per_request = [
                1 + get_draft_token_length(req) for req in requests
            ]
            logits = self._apply_embedding_bias(logits, requests,
                                                steps_per_request)
            batched_next_tokens, batched_softmax = sample(
                batched_strategy, logits, generator)
            self.append_eagle3(batched_next_tokens, model_outputs)

        offset = 0
        for i, (strategy, slot, steps, request) in enumerate(
                zip(strategies, seq_slots_host, num_steps, requests)):
            input_slice = slice(offset, offset + steps)
            logits = raw_logits[input_slice]

            req = requests[i]

            if batched_next_tokens is None:
                logits = self._apply_embedding_bias(logits, [req])
                next_tokens, softmax = sample(strategy, logits, generator)
            else:
                # Batched processing already applied bias, just use the results
                next_tokens = batched_next_tokens[input_slice]
                softmax = batched_softmax[input_slice]
            current_slice = slice(0, steps), slot, BEAM_0
            new_tokens[current_slice] = next_tokens
            if request.py_draft_logits is not None:
                request.py_target_probs = softmax.clone()
            if log_probs_host is not None:
                assert BEAM_0 == 0, "The following call relies on beam_width to be 1 - hence the unsqueeze"
                token_probs = torch.gather(
                    softmax, dim=1, index=next_tokens.unsqueeze(1)).squeeze(-1)
                log_probs = torch.log(token_probs)
                log_probs_host[slot, BEAM_0, :steps].copy_(log_probs,
                                                           non_blocking=True)
            offset += steps


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
