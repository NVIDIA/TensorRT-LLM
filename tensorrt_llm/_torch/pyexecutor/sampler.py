from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from tensorrt_llm._torch.pyexecutor.handle_logits import HandleLogits
from tensorrt_llm._torch.pyexecutor.make_decoding_batch_input_output import \
    MakeDecodingBatchInputOutput
from tensorrt_llm._utils import nvtx_range, torch_dtype_to_binding
from tensorrt_llm.bindings import (CudaStream, DataType, ModelConfig,
                                   WorldConfig, make_sampling_config)
from tensorrt_llm.bindings.executor import (DecodingConfig, DecodingMode,
                                            ExecutorConfig, FinishReason)
from tensorrt_llm.bindings.internal.algorithms import CreateNewDecoderRequests
from tensorrt_llm.bindings.internal.batch_manager import (
    DecoderInputBuffers, add_new_tokens_to_requests, make_decoding_batch_input)
from tensorrt_llm.bindings.internal.runtime import (BufferManager, CudaEvent,
                                                    DecoderState,
                                                    GptDecoderBatched)
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.mapping import Mapping

from .finish_reason import FinishedState
from .llm_request import LlmRequest, LlmRequestState
from .scheduler import ScheduledRequests


@dataclass(kw_only=True)
class SampleStateTensors:
    new_tokens: torch.Tensor
    logits: torch.Tensor | None = None
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
    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs) -> SampleState:
        raise NotImplementedError

    @abstractmethod
    def update_requests(self, state: SampleState) -> None:
        raise NotImplementedError


class EarlyStopSampler(Sampler):
    """
    Use for skipping decoding step for non generation model,
    such as encoder-only model (e.g., BERT) or reward models that only need context phase.
    """

    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs) -> SampleState:
        host = SampleStateTensors(logits=model_outputs['logits'],
                                  new_tokens=torch.empty(0))
        return SampleState(scheduled_requests=scheduled_requests, host=host)

    def update_requests(self, state: SampleState) -> None:
        assert isinstance(state, SampleState)
        scheduled_requests = state.scheduled_requests
        assert (not scheduled_requests.generation_requests)
        for idx, request in enumerate(scheduled_requests.context_requests):
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)
            if request.py_return_context_logits:
                logits = state.host.logits[idx]
                if logits.ndim == 1:
                    # For BERT: Add axis to be compatible with LogitsStorage
                    # (LogitsStorage will interpret this dim as the prompt_len which
                    # is not relevant for outputting logits of encoder only model).
                    logits = logits.unsqueeze(0)
                request.py_result.append_context_logits(logits)


def top_k_sampling_batch(logits, top_k=50):
    logits_dim = logits.dim()
    if logits_dim == 1:
        logits = logits.unsqueeze(0)
    # logits should be 2D ：[batch_size, vocab_size]
    batch_size, vocab_size = logits.size()

    # get first top_k logits of each sample and their indices
    values, indices = torch.topk(logits, top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1).expand(batch_size, vocab_size)

    # set the logits who is less than first top_k logits to -inf
    logits = torch.where(logits < min_values,
                         torch.full_like(logits, float('-inf')), logits)

    # compute probability distribution
    softmax = torch.softmax(logits, dim=-1)

    # sample from the distribution and generate result of [batch_size, 1]
    next_tokens = torch.multinomial(softmax, num_samples=1).squeeze(-1)
    return next_tokens, softmax


def top_p_sampling_batch(logits: torch.Tensor, top_p: float = 0.9):
    logits_dim = logits.dim()
    if logits_dim == 1:
        logits = logits.unsqueeze(0)
    assert logits_dim == 2, "logits should be 2D: [batch_size, vocab_size]"

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
    next_tokens = torch.multinomial(softmax, num_samples=1).squeeze(-1)
    return next_tokens, softmax


def greedy_search_sampling_batch(logits):
    next_tokens = torch.argmax(logits, dim=-1)
    softmax = torch.softmax(logits, dim=-1)
    return next_tokens, softmax


TopK = tuple[Literal["top_k"], int]
TopP = tuple[Literal["top_p"], float]
Greedy = tuple[Literal["greedy"], None]
GREEDY: Greedy = ("greedy", None)
Strategy = TopK | TopP | Greedy


def request_strategy(request: LlmRequest) -> Strategy:
    if request.sampling_config.top_p is not None and len(
            request.sampling_config.top_p) > 0:
        return ("top_p", request.sampling_config.top_p[0])
    elif request.sampling_config.top_k is not None and len(
            request.sampling_config.top_k) > 0:
        return ("top_k", request.sampling_config.top_k[0])
    else:
        return ("greedy", None)


def sampling_strategies(requests: Iterable[LlmRequest]) -> list[Strategy]:
    return [request_strategy(req) for req in requests]


def sample(strategy: Strategy, logits: torch.Tensor):
    match strategy:
        case ("top_k", top_k):
            return top_k_sampling_batch(logits, top_k)
        case ("top_p", top_p):
            return top_p_sampling_batch(logits, top_p)
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


class TorchSampler(Sampler):
    BEAM = 0
    MAX_BEAM_WIDTH = BEAM + 1

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
        enable_mixed_sampler: bool

    def __init__(self, args: Args):
        self.max_seq_len = args.max_seq_len
        self.enable_mixed_sampler = args.enable_mixed_sampler
        self.max_tokens = args.max_draft_len + 1
        assert args.max_beam_width == self.MAX_BEAM_WIDTH, "TorchSampler only supports beam_width = 1"
        self.num_seq_slots = args.max_num_sequences

        self.NEW_TOKENS_SHAPE = (self.max_tokens, self.num_seq_slots,
                                 self.MAX_BEAM_WIDTH)
        # AutoDeploy build creates the sampler in inference mode,
        # which would disallow in-place mutating of new_tokens.
        # So, we temporarily exit inference mode.
        with torch.inference_mode(False):
            self.store = self.create_store()

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

    def handle_logits(self, request: LlmRequest, state: SampleState, *,
                      beam: int, count: int):
        current_slice = slice(0, count), request.py_seq_slot, beam
        if request.py_return_generation_logits:
            assert state.host.logits is not None
            current_logits = state.host.logits[current_slice]
            request.py_result.append_generation_logits(current_logits)
        if request.py_return_log_probs:
            assert state.host.log_probs is not None
            log_probs = state.host.log_probs[request.py_seq_slot][beam][:count]
            current_tokens = state.host.new_tokens[current_slice]

            token_log_probs = [{
                int(token): Logprob(logprob=logprob, rank=1)
            } for token, logprob in zip(current_tokens, log_probs.tolist())]
            assert beam == 0, "The following call relies on beam_width to be 1 - hence the list with a single element"
            request.py_result.append_log_probs([token_log_probs])

    def process_draft_tokens(self, request: LlmRequest,
                             new_tokens: torch.Tensor, new_token: int) -> int:
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

    def update_requests(self, state: SampleState) -> None:
        assert isinstance(state, SampleState)
        if state.sampler_event:
            state.sampler_event.synchronize()
        new_tokens = state.host.new_tokens

        for req in state.scheduled_requests.context_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE or req.context_remaining_length != 0:
                continue
            new_token = add_token(req, new_tokens, beam=self.BEAM)
            self._handle_stop_criteria(req, new_token)
            self.handle_logits(req, state, beam=self.BEAM, count=1)
            req.py_decoding_iter += 1

        for req in state.scheduled_requests.generation_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            new_token = add_token(req, new_tokens, beam=self.BEAM)
            stop = self._handle_stop_criteria(req, new_token)
            processed = 1
            if not stop and len(req.py_draft_tokens) > 0:
                num_accepted = self.process_draft_tokens(
                    req, new_tokens, new_token)
                req.py_num_accepted_draft_tokens = num_accepted
                req.py_rewind_len = req.py_draft_pages_allocated - num_accepted
                processed += num_accepted
            self.handle_logits(req, state, beam=self.BEAM, count=processed)
            req.py_decoding_iter += 1

    def log_probs_host(self, requests: Iterable[LlmRequest]):
        """Shape: In lockstep with TRTLLMSampler: https://github.com/NVIDIA/TensorRT-LLM/blob/cea5dd1e3883b18bf50901a7f196f50a9544c28c/cpp/include/tensorrt_llm/runtime/decoderState.h#L103"""
        if any(req.py_return_log_probs for req in requests):
            return torch.empty(
                (self.num_seq_slots, self.MAX_BEAM_WIDTH, self.max_tokens),
                device="cpu",
                pin_memory=True)
        return None

    def gen_logits_host(self, requests: Iterable[LlmRequest], vocab_size: int):
        if any(req.py_return_generation_logits for req in requests):
            return torch.empty((self.max_tokens, self.num_seq_slots,
                                self.MAX_BEAM_WIDTH, vocab_size),
                               device="cpu",
                               pin_memory=True)
        return None

    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs: dict[str, torch.Tensor]) -> SampleState:
        requests = scheduled_requests.all_requests()
        new_tokens = self.store.new_tokens
        vocab_size = model_outputs["logits"].shape[-1]
        log_probs_host = self.log_probs_host(requests)
        gen_logits_host = self.gen_logits_host(requests, vocab_size)
        self._process_requests(requests,
                               model_outputs,
                               new_tokens,
                               gen_logits_host=gen_logits_host,
                               log_probs_host=log_probs_host)
        new_tokens_host = new_tokens.to(device="cpu", non_blocking=True)
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        return SampleState(scheduled_requests=scheduled_requests,
                           device=SampleStateTensors(new_tokens=new_tokens),
                           host=SampleStateTensors(new_tokens=new_tokens_host,
                                                   log_probs=log_probs_host,
                                                   logits=gen_logits_host),
                           sampler_event=sampler_event)

    @staticmethod
    def append_eagle3(tokens: torch.Tensor, model_outputs):
        if "d2t" in model_outputs:
            d2t = model_outputs["d2t"][tokens]
            tokens += d2t

    def _process_requests(self,
                          requests: list[LlmRequest],
                          model_outputs: dict[str, torch.Tensor],
                          new_tokens: torch.Tensor,
                          *,
                          gen_logits_host: torch.Tensor | None = None,
                          log_probs_host: torch.Tensor | None = None):
        beam_width = self.MAX_BEAM_WIDTH
        beam = self.BEAM
        raw_logits = model_outputs["logits"]
        num_steps = [1 + len(req.py_draft_tokens) for req in requests]
        sum_steps = sum(num_steps)
        no_draft_tokens = len(requests) == sum_steps
        fast_path = not self.enable_mixed_sampler and no_draft_tokens and gen_logits_host is None and log_probs_host is None

        seq_slots = torch.as_tensor([r.py_seq_slot for r in requests])
        seq_slots = seq_slots.to(device="cuda", non_blocking=True)

        if fast_path:
            logits = raw_logits[:len(requests)]
            next_tokens = torch.argmax(logits, dim=-1)
            self.append_eagle3(next_tokens, model_outputs)
            int_next_tokens = next_tokens.to(torch.int, non_blocking=True)
            next_tokens = int_next_tokens.view(1, -1, beam_width)
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

        if batched_strategy is not None:
            logits = raw_logits[:sum_steps]
            batched_next_tokens, batched_softmax = sample(
                batched_strategy, logits)
            self.append_eagle3(batched_next_tokens, model_outputs)

        offset = 0
        for strategy, slot, steps in zip(strategies, seq_slots, num_steps):
            input_slice = slice(offset, offset + steps)
            logits = raw_logits[input_slice]
            if batched_next_tokens is None:
                next_tokens, softmax = sample(strategy, logits)
            else:
                next_tokens = batched_next_tokens[input_slice]
                softmax = batched_softmax[input_slice]
            current_slice = slice(0, steps), slot, beam
            new_tokens[current_slice] = next_tokens
            if gen_logits_host is not None:
                gen_logits_host[current_slice].copy_(logits, non_blocking=True)
            if log_probs_host is not None:
                assert beam == 0, "The following call relies on beam_width to be 1 - hence the unsqueeze"
                token_probs = torch.gather(
                    softmax, dim=1, index=next_tokens.unsqueeze(1)).squeeze(-1)
                log_probs = torch.log(token_probs)
                log_probs_host[slot, beam, :steps].copy_(log_probs,
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
    finalize_events: dict[str, CudaEvent]
    host: SampleStateTensorsHostTRTLLM


class TRTLLMSampler(Sampler):
    MAX_DECODING_TOKENS = 1  # It must be 1 when not in speculative decoding
    SampleState = SampleStateTRTLLM

    def __init__(
        self,
        executor_config: ExecutorConfig,
        model,
        model_dtype,
        mapping: Mapping,
        decoding_mode: DecodingMode,
        disable_overlap_scheduler: bool,
    ):

        vocab_size = model.config.vocab_size
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads

        self.model_datatype = torch_dtype_to_binding(model_dtype)
        self.logits_datatype = DataType.FLOAT
        self.decoding_mode = decoding_mode
        self.executor_config = executor_config
        self.decoding_config = self.executor_config.decoding_config if self.executor_config.decoding_config else DecodingConfig(
            decoding_mode)
        max_attn_window = self.executor_config.kv_cache_config.max_attention_window
        self.max_attention_window = max_attn_window if max_attn_window is not None else executor_config.max_seq_len
        self.max_num_sequences = mapping.pp_size * self.executor_config.max_batch_size
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
                DecoderInputBuffers(self.executor_config.max_batch_size,
                                    self.MAX_DECODING_TOKENS, buffer_manager)
                for _ in range(self.num_micro_batches)
            ],
            "sequence_lengths_host":
            torch.empty((
                self.executor_config.max_batch_size,
                self.executor_config.max_beam_width,
            ),
                        dtype=torch.int),
            "decoder_state":
            DecoderState(),
            "decoding_input": [None] * self.num_micro_batches,
        }

        self.store["decoder_state"].setup(
            max_batch_size=self.executor_config.max_batch_size,
            max_beam_width=self.executor_config.max_beam_width,
            max_attention_window=self.max_attention_window,
            sink_token_length=0,
            max_sequence_length=self.executor_config.max_seq_len,
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
            max_batch_size=self.executor_config.max_batch_size,
            max_beam_width=self.executor_config.max_beam_width,
            dtype=self.logits_datatype,
            model_config=self.model_config,
            world_config=self.world_config)
        self.algs.create_new_decoder_requests = CreateNewDecoderRequests(
            speculative_decoding_fast_logits=False,
            is_leader_in_orch_mode=False,
            is_normalize_log_probs=False)
        self.algs.handle_logits = HandleLogits()
        self.algs.make_decoding_batch_input_output = MakeDecodingBatchInputOutput(
        )

    @torch.inference_mode()
    @nvtx_range("setup_sampler_step")
    def setup_sampler_step(self, requests):
        batch_slots, sampling_configs, lookahead_prompt, lookahead_algo_configs = self.algs.create_new_decoder_requests(
            self.model_config, self.world_config, self.decoding_config,
            requests, self.store["buffer_manager"], self.logits_datatype,
            self.store["decoder_input_buffers"][self.micro_batch_idx],
            self.store["decoder_state"], self.store["cuda_stream"],
            self.algs.decoder.decoder_stream, self.executor_config.max_seq_len,
            self.beam_width(requests))

        local_batch_size = len(batch_slots)
        if local_batch_size > 0:
            sampling_config = make_sampling_config(sampling_configs)
            self.algs.decoder.underlying_decoder().setup(
                sampling_config, local_batch_size, batch_slots,
                self.store["decoder_state"].joint_decoding_output,
                self.model_config.data_type, lookahead_prompt,
                lookahead_algo_configs)

    @staticmethod
    @torch.inference_mode()
    def beam_width(scheduled_requests: Iterable[LlmRequest]) -> int:
        for req in scheduled_requests:
            return req.sampling_config.beam_width
        return 0

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
    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs) -> SampleStateTRTLLM:

        batch_size = scheduled_requests.batch_size
        beam_width = self.beam_width(scheduled_requests.all_requests())
        if (batch_size > 1 and beam_width > 1
                and any(request.py_return_log_probs
                        for request in scheduled_requests.all_requests())):
            raise ValueError(
                "Beam search is not supported for multiple prompts and logprobs"
            )

        self.setup_sampler_step(scheduled_requests.context_requests)

        num_context_logits_prefix_sum = [0]
        prefix_sum = 0
        for request in scheduled_requests.context_requests:
            prefix_sum += request.context_chunk_size if request.py_return_context_logits else 1
            num_context_logits_prefix_sum.append(prefix_sum)

        if any(r.py_return_context_logits or r.py_return_generation_logits
               for r in scheduled_requests.all_requests()):
            self.algs.handle_logits(scheduled_requests.context_requests,
                                    scheduled_requests.generation_requests,
                                    model_outputs["logits"],
                                    num_context_logits_prefix_sum,
                                    self.max_num_sequences, beam_width)

        # For beam search, cache indirection needs to be updated
        if beam_width > 1:
            self._update_cache_indirection_buffer(scheduled_requests)

        # TODO: Enable this back once nanobind is merged and/or llm request is a pure python object
        # decoding_input = self.algs.make_decoding_batch_input_output(
        #     scheduled_requests, model_outputs["logits"], beam_width,
        #     num_context_logits_prefix_sum)

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
            if request.request_id in finalize_events:
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
