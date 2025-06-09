from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.pyexecutor.handle_context_logits import \
    HandleContextLogits
from tensorrt_llm._torch.pyexecutor.handle_generation_logits import \
    HandleGenerationLogits
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings import (CudaStream, DataType, ModelConfig,
                                   WorldConfig, make_sampling_config)
from tensorrt_llm.bindings.executor import (DecodingConfig, DecodingMode,
                                            ExecutorConfig, FinishReason)
from tensorrt_llm.bindings.internal.algorithms import (
    CreateNewDecoderRequests, MakeDecodingBatchInputOutput)
from tensorrt_llm.bindings.internal.batch_manager import (DecoderBuffers,
                                                          DecoderInputBuffers)
from tensorrt_llm.bindings.internal.runtime import (BufferManager, DecoderState,
                                                    GptDecoderBatched)
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.mapping import Mapping

from .llm_request import LlmRequest, LlmRequestState
from .scheduler import ScheduledRequests


@dataclass(frozen=True, kw_only=True)
class SampleStateTensors:
    new_tokens: torch.Tensor

    def values(self):
        return vars(self).values()


@dataclass(kw_only=True, frozen=True)
class SampleState:
    scheduled_requests: ScheduledRequests

    logits: torch.Tensor = None

    device: SampleStateTensors = None
    host: SampleStateTensors = None

    sampler_event: torch.cuda.Event = None


class Sampler(ABC):

    SampleState = SampleState

    def setup_sampler_step(self, scheduled_requests: ScheduledRequests):
        pass

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
        return SampleState(scheduled_requests=scheduled_requests,
                           logits=model_outputs['logits'])

    def update_requests(self, state: SampleState) -> None:
        scheduled_requests = state.scheduled_requests
        assert (not scheduled_requests.generation_requests)
        for idx, request in enumerate(scheduled_requests.context_requests):
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)
            if request.py_return_context_logits:
                logits = state.logits[idx]
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
    assert logits_dim == 2, "logits should be 2D： [batch_size, vocab_size]"

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


def sample_single_request(request: LlmRequest, logits: torch.Tensor):
    assert logits.dim(
    ) == 2 and logits.shape[0] == 1, "logits should have shape [1, vocab_size]"
    if request.sampling_config.top_p is not None and len(
            request.sampling_config.top_p) > 0:
        return top_p_sampling_batch(logits, request.sampling_config.top_p[0])
    elif request.sampling_config.top_k is not None and len(
            request.sampling_config.top_k) > 0:
        return top_k_sampling_batch(logits, request.sampling_config.top_k[0])
    else:
        return greedy_search_sampling_batch(logits)


def seq_slice(request: LlmRequest, beam: int, *,
              size: int) -> tuple[slice, int, int]:
    return slice(0, size), request.seq_slot, beam


def add_token(request: LlmRequest,
              new_tokens: torch.Tensor,
              beam: int,
              step: int = 0) -> int:
    seq_slot = request.seq_slot
    assert seq_slot is not None
    new_token = int(new_tokens[step, request.seq_slot, beam])
    request.add_new_token(new_token, beam)
    return new_token


@dataclass(frozen=True, kw_only=True)
class SampleStateTorch(SampleState):
    log_probs: torch.Tensor | None = None


class TorchSampler(Sampler):
    BEAM = 0

    @dataclass(frozen=True, kw_only=True)
    class Store:
        new_tokens_device: torch.Tensor

    @dataclass(frozen=True, kw_only=True)
    class Args:
        max_seq_len: int
        max_draft_tokens: int
        max_batch_size: int
        max_beam_width: int
        vocab_size: int
        mixed_sampler: bool

    def __init__(self, args: Args):
        self.max_seq_len = args.max_seq_len
        self.mixed_sampler = args.mixed_sampler
        self.vocab_size = args.vocab_size

        max_tokens = args.max_draft_tokens + 1
        # See cpp DecoderState.getAllNewTokens()
        self.store = self.Store(
            new_tokens_device=torch.zeros((max_tokens, args.max_batch_size,
                                           args.max_beam_width),
                                          dtype=torch.int,
                                          device='cuda'))

    def _meet_max_token_stop_criteria(self, request: LlmRequest,
                                      num_tokens: int):
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

    def _handle_stop_criteria(self, request: LlmRequest, new_token: int, *,
                              beam: int) -> bool:
        """Handle stop criteria and set appropriate finish reasons and state.
        Returns True if generation should stop."""
        if new_token == request.py_end_id:
            request.finish_by_reason(FinishReason.END_ID)
            return True

        num_tokens = request.get_num_tokens(beam)
        if self._meet_max_token_stop_criteria(request, num_tokens):
            request.finish_by_reason(FinishReason.LENGTH)
            return True

        if self._meet_stop_token_criteria(request):
            request.finish_by_reason(FinishReason.STOP_WORDS)
            return True

        return False

    @staticmethod
    def handle_logits(request: LlmRequest, state: SampleStateTorch, *,
                      beam: int, count: int):
        current_slice = seq_slice(request, beam, size=count)
        if request.py_return_generation_logits:
            assert state.logits is not None
            current_logits = state.logits[current_slice]
            request.py_result.append_generation_logits(current_logits)
        if request.py_return_log_probs:
            assert state.log_probs is not None
            log_probs = state.log_probs[current_slice]
            current_tokens = state.host.new_tokens[current_slice]

            token_log_probs = [{
                int(token): Logprob(logprob=logprob, rank=1)
            } for token, logprob in zip(current_tokens, log_probs.tolist())]
            assert beam == 0, "The following call relies on beam_width to be 1 - hence the list with a single element"
            request.py_result.append_log_probs([token_log_probs])

    def update_requests(self, state: SampleStateTorch) -> None:
        assert isinstance(state, SampleStateTorch)
        if state.sampler_event:
            state.sampler_event.synchronize()
        new_tokens = state.host.new_tokens

        requests_to_process = [
            request for request in state.scheduled_requests.all_requests()
            if request.state != LlmRequestState.GENERATION_COMPLETE
            and request.get_context_remaining_length() == 0
        ]
        for req in requests_to_process:
            req.py_decoding_iter += 1
            new_token = add_token(req, new_tokens, beam=self.BEAM)
            if len(req.py_draft_tokens) == 0:
                self._handle_stop_criteria(req, new_token, beam=self.BEAM)
                self.handle_logits(req, state, beam=self.BEAM, count=1)
                continue
            num_accepted = 0
            for draft_token in req.py_draft_tokens:
                if draft_token != new_token:
                    # Reject.
                    break
                num_accepted += 1
                new_token = add_token(req,
                                      new_token,
                                      step=num_accepted,
                                      beam=self.BEAM)
                if self._handle_stop_criteria(req, new_token, beam=self.BEAM):
                    break
            self.handle_logits(req, state, beam=self.BEAM, count=num_accepted)
            req.py_num_accepted_draft_tokens = num_accepted
            req.py_rewind_len = req.py_draft_pages_allocated - num_accepted

    def sample_async(
            self, scheduled_requests: ScheduledRequests,
            model_outputs: dict[str, torch.Tensor]) -> SampleStateTorch:
        requests = scheduled_requests.all_requests()
        raw_logits = model_outputs["logits"]
        new_tokens_device = self.store.new_tokens_device
        logits_shape = (*new_tokens_device.shape, self.vocab_size)
        gen_logits = None
        log_probs = None
        if any(req.py_return_generation_logits for req in requests):
            gen_logits = torch.empty(logits_shape, device="cpu")
        if any(req.py_return_log_probs for req in requests):
            log_probs = torch.empty(logits_shape, device="cpu")
        self._process_requests(requests,
                               raw_logits,
                               new_tokens=new_tokens_device,
                               gen_logits=gen_logits,
                               log_probs=log_probs)
        new_tokens_host = new_tokens_device.to(device="cpu", non_blocking=True)
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        return SampleStateTorch(
            scheduled_requests=scheduled_requests,
            device=SampleStateTensors(new_tokens=new_tokens_device),
            host=SampleStateTensors(new_tokens=new_tokens_host),
            sampler_event=sampler_event,
            log_probs=log_probs,
            logits=gen_logits)

    def _process_requests(self,
                          requests: list[LlmRequest],
                          raw_logits: torch.Tensor,
                          *,
                          new_tokens: torch.Tensor,
                          gen_logits: torch.Tensor | None = None,
                          log_probs: torch.Tensor | None = None):
        beam = self.BEAM
        offset = 0
        for request in requests:
            steps = 1
            if len(request.py_draft_tokens) > 0:
                assert not self.mixed_sampler, "Speculative decoding not supported in mixed sampler"  # TODO: why?
                steps += len(request.py_draft_tokens)
            logits = raw_logits[offset:offset + steps]

            if self.mixed_sampler:
                next_tokens, softmax = sample_single_request(request, logits)
            else:
                next_tokens, softmax = greedy_search_sampling_batch(logits)

            current_slice = seq_slice(request, beam, size=steps)
            new_tokens[current_slice] = next_tokens
            if gen_logits:
                gen_logits[current_slice] = logits
            if log_probs:
                assert beam == 0, "The following call relies on beam_width to be 1 - hence the unsqueeze"
                token_probs = torch.gather(
                    softmax, dim=1, index=next_tokens.unsqueeze(1)).squeeze(-1)
                probs = torch.log(token_probs)
                log_probs[current_slice] = probs.unsqueeze(0)
            offset += steps


class TorchStarAttentionSampler(TorchSampler):

    def update_one_request(self, request: LlmRequest,
                           new_tokens_list: list[int], logits: torch.Tensor):
        beam_idx = 0

        output_token_idx = request.output_token_idx
        new_token = new_tokens_list[output_token_idx]
        num_tokens = request.add_new_token(new_token, beam_idx)

        current_logits = logits[output_token_idx].unsqueeze(0)
        if request.py_return_generation_logits:
            request.py_result.append_generation_logits(current_logits)
        if request.py_return_log_probs:
            _, log_probs = greedy_search_sampling_batch(current_logits)
            request.py_result.append_log_probs([[{
                new_token:
                Logprob(logprob=log_probs.item(), rank=1)
            }]])

        self._handle_stop_criteria(request, new_token, num_tokens, beam_idx)
        if request.state != LlmRequestState.GENERATION_COMPLETE:
            request.py_decoding_iter += 1

    def update_requests(self, state: SampleState):
        if state.sampler_event:
            state.sampler_event.synchronize()
        new_tokens_list = state.host.new_tokens.tolist()
        logits = state.logits

        for request in state.scheduled_requests.context_requests:
            if request.state == LlmRequestState.GENERATION_IN_PROGRESS:
                self.update_one_request(request, new_tokens_list, logits)

        for request in state.scheduled_requests.generation_requests:
            self.update_one_request(request, new_tokens_list, logits)


class Algorithms:

    def defined_algorithms(self):
        return [attr for attr in dir(self) if not attr.startswith("__")]

    def __repr__(self):
        algs = self.defined_algorithms()
        return f"Algs({', '.join(algs)})"


@dataclass(frozen=True, kw_only=True)
class SampleStateTensorsHostTRTLLM(SampleStateTensors):
    finished_sum: torch.Tensor
    finish_reasons: torch.Tensor
    sequence_lengths: torch.Tensor


@dataclass(kw_only=True, frozen=True)
class SampleStateTRTLLM(SampleState):
    host: SampleStateTensorsHostTRTLLM
    device: SampleStateTensors


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
            "decoder_buffers":
            DecoderBuffers(self.max_num_sequences,
                           self.executor_config.max_beam_width,
                           self.max_attention_window, self.MAX_DECODING_TOKENS,
                           buffer_manager, self.model_config,
                           self.world_config),
            "decoder_input_buffers":
            DecoderInputBuffers(self.executor_config.max_batch_size,
                                self.MAX_DECODING_TOKENS, buffer_manager),
            "sequence_lengths_host":
            torch.empty((
                self.executor_config.max_batch_size,
                self.executor_config.max_beam_width,
            ),
                        dtype=torch.int)
        }

    def _instantiate_algorithms(self):
        self.algs = Algorithms()
        self.algs.decoder = GptDecoderBatched(stream=self.store["torch_stream"])
        self.algs.decoder.setup(
            mode=self.decoding_mode,
            max_batch_size=self.executor_config.max_batch_size,
            max_beam_width=self.executor_config.max_beam_width,
            max_sequence_length=self.executor_config.max_seq_len,
            dtype=self.logits_datatype,
            model_config=self.model_config,
            world_config=self.world_config)
        self.algs.decoder_state = DecoderState(
            dtype=self.logits_datatype,
            buffer_manager=self.store["buffer_manager"])
        self.algs.decoder_state.setup(
            max_batch_size=self.executor_config.max_batch_size,
            max_beam_width=self.executor_config.max_beam_width,
            max_attention_window=self.max_attention_window,
            sink_token_length=0,
            max_sequence_length=self.executor_config.max_seq_len,
            model_config=self.model_config,
            world_config=self.world_config,
            buffer_manager=self.store["buffer_manager"])
        self.algs.create_new_decoder_requests = CreateNewDecoderRequests(
            speculative_decoding_fast_logits=False,
            is_leader_in_orch_mode=False,
            is_normalize_log_probs=False)
        self.algs.handle_context_logits = HandleContextLogits()
        self.algs.handle_generation_logits = HandleGenerationLogits()
        self.algs.make_decoding_batch_input_output = MakeDecodingBatchInputOutput(
        )

    def setup_sampler_step(self, requests):
        batch_slots, decoder_requests, sampling_configs = self.algs.create_new_decoder_requests(
            self.model_config, self.world_config, self.decoding_config,
            requests, self.store["buffer_manager"], self.logits_datatype,
            self.store["decoder_input_buffers"], self.algs.decoder_state,
            self.store["cuda_stream"], self.algs.decoder.decoder_stream,
            self.executor_config.max_seq_len, self.beam_width(requests))

        if len(decoder_requests):
            local_batch_size = len(batch_slots)
            sampling_config = make_sampling_config(sampling_configs)
            self.algs.decoder.underlying_decoder().setup(
                sampling_config, local_batch_size, batch_slots,
                self.algs.decoder_state.joint_decoding_output, decoder_requests)

    @staticmethod
    def beam_width(scheduled_requests: Iterable[LlmRequest]) -> int:
        for req in scheduled_requests:
            return req.sampling_config.beam_width
        return 0

    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs) -> SampleStateTRTLLM:
        batch_size = scheduled_requests.batch_size

        self.setup_sampler_step(scheduled_requests.context_requests)

        num_context_logits = [1] * batch_size
        for batch_index, request in enumerate(
                scheduled_requests.context_requests):
            num_context_logits[
                batch_index] = request.context_chunk_size if request.py_return_context_logits else 1

        logits_index = self.algs.handle_context_logits(
            scheduled_requests.context_requests, num_context_logits,
            model_outputs["logits"], self.store["decoder_buffers"])

        self.algs.handle_generation_logits(
            logits_index, scheduled_requests.generation_requests,
            self.store["decoder_buffers"], model_outputs["logits"])

        decoding_input, self.decoding_output = self.algs.make_decoding_batch_input_output(
            scheduled_requests.context_requests,
            scheduled_requests.generation_requests,
            self.store["decoder_buffers"], self.store["decoder_input_buffers"],
            self.algs.decoder_state, self.model_config, self.max_num_sequences)

        self.algs.decoder.forward_async(self.algs.decoder_state,
                                        self.decoding_output, decoding_input)

        new_output_tokens = self.algs.decoder_state.all_new_tokens.to(
            'cpu', non_blocking=True)
        finished_sum = self.algs.decoder_state.finished_sum.to(
            'cpu', non_blocking=True)
        finish_reasons = self.algs.decoder_state.finish_reasons.to(
            'cpu', non_blocking=True)
        sequence_lengths = self.algs.decoder_state.sequence_lengths.to(
            'cpu', non_blocking=True)

        device = SampleStateTensors(
            new_tokens=self.algs.decoder.decoder_state.all_new_tokens)

        host = SampleStateTensorsHostTRTLLM(new_tokens=new_output_tokens,
                                            finished_sum=finished_sum,
                                            finish_reasons=finish_reasons,
                                            sequence_lengths=sequence_lengths)

        sampler_event = torch.cuda.Event()
        sampler_event.record()

        return SampleStateTRTLLM(scheduled_requests=scheduled_requests,
                                 logits=model_outputs["logits"],
                                 device=device,
                                 host=host,
                                 sampler_event=sampler_event)

    def update_requests(self, state: SampleStateTRTLLM):
        assert isinstance(state, SampleStateTRTLLM)

        scheduled_requests = state.scheduled_requests
        assert scheduled_requests.batch_size > 0
        requests = scheduled_requests.all_requests()
        beam_width = self.beam_width(requests)
        sampler_event = state.sampler_event

        if sampler_event:
            sampler_event.synchronize()

        new_tokens_host = state.host.new_tokens
        finished_sum_host = state.host.finished_sum
        finish_reasons_host = state.host.finish_reasons
        sequence_lengths_host_data = state.host.sequence_lengths

        for request in requests:
            if request.is_context_init_state:
                continue

            seq_slot = request.seq_slot
            num_generated_tokens = request.num_draft_tokens + 1
            current_num_of_tokens = request.max_beam_num_tokens
            num_new_tokens = [0] * beam_width

            for beam in range(beam_width):
                seq_len = sequence_lengths_host_data[seq_slot * beam_width +
                                                     beam].item()
                num_new_tokens[beam] = min(
                    num_generated_tokens,
                    seq_len - request.get_num_tokens(beam))

                for step in range(num_new_tokens[beam]):
                    add_token(request, new_tokens_host, beam, step=step)

                finish_reason = finish_reasons_host[seq_slot * beam_width +
                                                    beam].item()
                request.set_finished_reason(FinishReason(finish_reason), beam)

            # Set number of tokens predicted per runtime iteration. Will be > 1 for speculative decoding.
            request.update_num_tokens_per_iteration(
                request.max_beam_num_tokens - current_num_of_tokens,
                self.model_config)

            # Increment the decoding iteration counter
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                request.py_decoding_iter += 1

            if finished_sum_host[seq_slot] == beam_width:
                request.state = LlmRequestState.GENERATION_COMPLETE
