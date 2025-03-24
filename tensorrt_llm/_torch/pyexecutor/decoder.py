import itertools
from abc import ABC, abstractmethod
from typing import Dict

import torch

import tensorrt_llm.bindings as tllm
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.mapping import Mapping

from .llm_request import *
from .scheduler import ScheduledRequests


class Decoder(ABC):

    @abstractmethod
    def setup_decoder(self, scheduled_requests: ScheduledRequests,
                      model_outputs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, scheduled_requests: ScheduledRequests, model_outputs):
        raise NotImplementedError

    @abstractmethod
    def decode_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs):
        raise NotImplementedError

    @abstractmethod
    def update_requests(self, scheduled_requests: ScheduledRequests,
                        new_tensors_host: Dict[str, torch.tensor],
                        decoder_event: torch.cuda.Event):
        raise NotImplementedError


class DummyDecoder(Decoder):

    def setup_decoder(self, scheduled_requests: ScheduledRequests,
                      model_outputs):
        pass

    def decode(self, scheduled_requests: ScheduledRequests, model_outputs):
        for request in scheduled_requests.context_requests:
            request.add_new_token(500, 0)
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
        for request in scheduled_requests.generation_requests:
            if request.get_num_tokens(0) < 10:
                request.add_new_token(request.get_num_tokens(0) + 1000, 0)
            else:
                request.state = LlmRequestState.GENERATION_COMPLETE


class EarlyStopDecoder(Decoder):
    """
    Use for skipping decoding step for non generation model,
    such as encoder-only model (e.g., BERT) or reward models that only need context phase.
    """

    def setup_decoder(self, scheduled_requests: ScheduledRequests,
                      model_outputs):
        pass

    def decode_async(self, model_outputs):
        pass

    def update_requests(self, scheduled_requests: ScheduledRequests,
                        new_tokens_host: torch.tensor,
                        decoder_event: torch.cuda.Event):
        pass

    def decode(self, scheduled_requests: ScheduledRequests, model_outputs):
        assert (not scheduled_requests.generation_requests)
        for idx, request in enumerate(scheduled_requests.context_requests):
            request.state = LlmRequestState.GENERATION_COMPLETE
            #NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(tllm_executor.FinishReason.LENGTH, 0)
            request.context_logits = model_outputs['logits'][idx]


def top_k_sampling_batch(logits, top_k=50):
    logits_dim = logits.dim()
    if logits_dim == 1:
        logits = logits.unsqueeze(0)
    # logits should be 2D ：[batch_size, vocab_size]
    batch_size, vocab_size = logits.size()

    raw_probs = torch.softmax(logits, dim=-1)

    # get first top_k logits of each sample and their indices
    values, indices = torch.topk(logits, top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1).expand(batch_size, vocab_size)

    # set the logits who is less than first top_k logits to -inf
    logits = torch.where(logits < min_values,
                         torch.full_like(logits, float('-inf')), logits)

    # compute probability distribution
    probs = torch.softmax(logits, dim=-1)

    # sample from the distribution and generate result of [batch_size, 1]
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token_probs = torch.gather(raw_probs, dim=1,
                               index=next_tokens.unsqueeze(1)).squeeze(-1)
    log_probs = torch.log(token_probs)
    return next_tokens, log_probs


def top_p_sampling_batch(logits, top_p=0.9):
    logits_dim = logits.dim()
    if logits_dim == 1:
        logits = logits.unsqueeze(0)
    # logits should be 2D ：[batch_size, vocab_size]
    batch_size, vocab_size = logits.size()

    raw_probs = torch.softmax(logits, dim=-1)

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
    probs = torch.softmax(logits, dim=-1)

    # sample from the distribution and generate result of [batch_size, 1]
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token_probs = torch.gather(raw_probs, dim=1,
                               index=next_tokens.unsqueeze(1)).squeeze(-1)
    log_probs = torch.log(token_probs)
    return next_tokens, log_probs


def greedy_search_sampling_batch(logits):
    raw_probs = torch.softmax(logits, dim=-1)
    next_tokens = torch.argmax(logits, dim=-1)
    token_probs = torch.gather(raw_probs, dim=1,
                               index=next_tokens.unsqueeze(1)).squeeze(-1)
    log_probs = torch.log(token_probs)
    return next_tokens, log_probs


def decode_single_request(request, logits):
    assert logits.dim(
    ) == 2 and logits.shape[0] == 1, "logits should have shape [1, vocab_size]"
    if request.sampling_config.top_p is not None and len(
            request.sampling_config.top_p) > 0:
        next_tokens, log_probs = top_p_sampling_batch(
            logits, request.sampling_config.top_p[0])
    elif request.sampling_config.top_k is not None and len(
            request.sampling_config.top_k) > 0:
        next_tokens, log_probs = top_k_sampling_batch(
            logits, request.sampling_config.top_k[0])
    else:
        next_tokens, log_probs = greedy_search_sampling_batch(logits)
    # TODO: enable these lines when log_probs is needed
    # request.log_probs_async = log_probs
    # request.set_cum_log_prob(request.cum_log_probs[0] + log_probs[0].item(), 0)
    return next_tokens


class TorchDecoder(Decoder):

    def __init__(self, max_seq_len: int, mixed_decoder: bool = False):
        self.max_seq_len = max_seq_len
        self.mixed_decoder = mixed_decoder

    def setup_decoder(self, scheduled_requests: ScheduledRequests,
                      model_outputs):
        pass

    def _meet_max_token_stop_criteria(self, request: LlmRequest,
                                      num_tokens: int):

        return (num_tokens - request.py_orig_prompt_len
                >= request.py_max_new_tokens) or (num_tokens
                                                  >= self.max_seq_len)

    def _meet_stop_token_criteria(self, request: LlmRequest):
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

    def _handle_stop_criteria(self, request: LlmRequest, new_token: int,
                              num_tokens: int, beam_idx: int) -> bool:
        """Handle stop criteria and set appropriate finish reasons and state.
        Returns True if generation should stop."""
        if new_token == request.py_end_id:
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.set_finished_reason(tllm.executor.FinishReason.END_ID,
                                        beam_idx)
            return True

        if self._meet_max_token_stop_criteria(request, num_tokens):
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.set_finished_reason(tllm.executor.FinishReason.LENGTH,
                                        beam_idx)
            return True

        if self._meet_stop_token_criteria(request):
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.set_finished_reason(tllm.executor.FinishReason.STOP_WORDS,
                                        beam_idx)
            return True

        return False

    def update_requests(self, scheduled_requests: ScheduledRequests,
                        new_tensors_host: Dict[str, torch.tensor],
                        decoder_event: torch.cuda.Event):
        if decoder_event:
            decoder_event.synchronize()
        new_tokens_list = new_tensors_host["new_tokens_host"].tolist()

        idx = 0
        beam_idx = 0
        for request in scheduled_requests.context_requests:
            if request.get_context_remaining_length() != 0:
                idx += 1
                continue

            if request.state != LlmRequestState.GENERATION_COMPLETE:
                new_token = new_tokens_list[idx]
                num_tokens = request.add_new_token(new_token, beam_idx)
                self._handle_stop_criteria(request, new_token, num_tokens,
                                           beam_idx)
            idx += 1

        if hasattr(scheduled_requests, 'chunked_requests'):
            for request in scheduled_requests.chunked_requests:
                idx += 1

        extend_requests = []
        generation_requests = []
        for request in scheduled_requests.generation_requests:
            if hasattr(
                    request,
                    'py_draft_tokens') and request.py_draft_tokens is not None:
                extend_requests.append(request)
            else:
                generation_requests.append(request)

        for request in extend_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                new_token = new_tokens_list[idx]
                num_tokens = request.add_new_token(new_token, beam_idx)
                self._handle_stop_criteria(request, new_token, num_tokens,
                                           beam_idx)

                # Accept draft tokens (if we have any) if and only if they match the new
                # token exactly.
                for i in range(len(request.py_draft_tokens)):
                    draft_token = request.py_draft_tokens[i]
                    if draft_token != new_token:
                        # Reject.
                        break

                    new_token = new_tokens_list[idx + i + 1]
                    num_tokens = request.add_new_token(new_token, beam_idx)

                    if self._handle_stop_criteria(request, new_token,
                                                  num_tokens, beam_idx):
                        break
            idx += len(request.py_draft_tokens) + 1

        for request in generation_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                new_token = new_tokens_list[idx]
                num_tokens = request.add_new_token(new_token, beam_idx)
                self._handle_stop_criteria(request, new_token, num_tokens,
                                           beam_idx)
            idx += 1

    def _mixed_decode(self, scheduled_requests: ScheduledRequests,
                      model_outputs):
        logits = model_outputs["logits"]
        new_tokens_device_array = []

        idx = 0

        for request in scheduled_requests.context_requests:
            token_logits = logits[idx:idx + 1, :]
            new_token = decode_single_request(request, token_logits)
            new_tokens_device_array += [new_token]
            idx += 1

        for request in scheduled_requests.generation_requests:
            assert request.py_draft_tokens is None, "Speculative decoding not supported in SeparateDecoder."
            token_logits = logits[idx:idx + 1, :]
            new_token = decode_single_request(request, token_logits)
            new_tokens_device_array += [new_token]
            idx += 1

        new_tokens_device = torch.cat(new_tokens_device_array)
        new_tokens_host = new_tokens_device.to('cpu', non_blocking=True)
        new_tensors_device = {"new_tokens_device": new_tokens_device}
        new_tensors_host = {"new_tokens_host": new_tokens_host}
        decoder_event = torch.cuda.Event()
        decoder_event.record()
        return new_tensors_device, new_tensors_host, decoder_event

    def _batch_decode(self, scheduled_requests: ScheduledRequests,
                      model_outputs):
        logits = model_outputs["logits"]
        new_tokens_device = torch.argmax(logits, dim=-1)
        new_tokens_host = new_tokens_device.to('cpu', non_blocking=True)
        new_tensors_device = {"new_tokens_device": new_tokens_device}
        new_tensors_host = {"new_tokens_host": new_tokens_host}
        decoder_event = torch.cuda.Event()
        decoder_event.record()
        return new_tensors_device, new_tensors_host, decoder_event

    def decode_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs):
        if self.mixed_decoder:
            return self._mixed_decode(scheduled_requests, model_outputs)
        else:
            return self._batch_decode(scheduled_requests, model_outputs)

    def decode(self, scheduled_requests: ScheduledRequests, model_outputs):
        _, new_tensors_host, decoder_event = self.decode_async(
            scheduled_requests, model_outputs)
        self.update_requests(scheduled_requests, new_tensors_host,
                             decoder_event)


class TorchStarAttentionDecoder(TorchDecoder):

    def update_requests(self, scheduled_requests: ScheduledRequests,
                        new_tensors_host: Dict[str, torch.tensor],
                        decoder_event: torch.cuda.Event):
        if decoder_event:
            decoder_event.synchronize()
        new_tokens_list = new_tensors_host["new_tokens_host"].tolist()

        beam_idx = 0
        for request in scheduled_requests.context_requests:
            if request.state == LlmRequestState.GENERATION_IN_PROGRESS:
                new_token = new_tokens_list[request.output_token_idx]
                num_tokens = request.add_new_token(new_token, beam_idx)
                self._handle_stop_criteria(request, new_token, num_tokens,
                                           beam_idx)

        for request in scheduled_requests.generation_requests:
            new_token = new_tokens_list[request.output_token_idx]
            num_tokens = request.add_new_token(new_token, beam_idx)
            self._handle_stop_criteria(request, new_token, num_tokens, beam_idx)


class Algorithms():

    def defined_algorithms(self):
        return [attr for attr in dir(self) if not attr.startswith("__")]

    def __repr__(self):
        algs = self.defined_algorithms()
        return f"Algs({', '.join(algs)})"


class TRTLLMDecoder(Decoder):

    def __init__(
        self,
        executor_config: tllm.executor.ExecutorConfig,
        model,
        model_dtype,
        mapping: Mapping,
        decoding_mode: tllm.executor.DecodingMode,
    ):

        vocab_size = model.config.vocab_size
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads

        self.model_datatype = torch_dtype_to_binding(model_dtype)
        self.logits_datatype = tllm.DataType.FLOAT
        self.decoding_mode = decoding_mode
        self.executor_config = executor_config
        self.decoding_config = self.executor_config.decoding_config if self.executor_config.decoding_config else tllm.executor.DecodingConfig(
            decoding_mode)
        max_attn_window = self.executor_config.kv_cache_config.max_attention_window
        self.max_attention_window = max_attn_window if max_attn_window is not None else executor_config.max_seq_len
        self.max_num_sequences = mapping.pp_size * self.executor_config.max_batch_size
        self.max_seq_idle_microseconds = 180 * 1000 * 1000
        self.max_decoding_tokens = 1  # It must be 1 when not in speculative decoding

        self.world_config = tllm.WorldConfig.mpi(mapping.gpus_per_node,
                                                 mapping.tp_size,
                                                 mapping.pp_size)
        self.model_config = tllm.ModelConfig(vocab_size, num_hidden_layers,
                                             num_hidden_layers, 0, num_heads,
                                             hidden_size, self.model_datatype)

        self._initialize_store()
        self._instantiate_algorithms()

    def _initialize_store(self):
        self.store = {}
        self.store["torch_stream"] = torch.cuda.Stream()
        self.store["cuda_stream"] = tllm.internal.runtime.CudaStream(
            self.store["torch_stream"].cuda_stream)
        self.store["buffer_manager"] = tllm.internal.runtime.BufferManager(
            stream=self.store["cuda_stream"])
        self.store[
            "seq_slot_manager"] = tllm.internal.batch_manager.SequenceSlotManager(
                self.max_num_sequences, self.max_seq_idle_microseconds)
        self.store[
            "decoder_buffers"] = tllm.internal.batch_manager.DecoderBuffers(
                self.max_num_sequences, self.executor_config.max_beam_width,
                self.max_attention_window, self.executor_config.max_seq_len,
                self.max_decoding_tokens, self.store["buffer_manager"],
                self.model_config, self.world_config)
        self.store[
            "decoder_input_buffers"] = tllm.internal.batch_manager.DecoderInputBuffers(
                self.executor_config.max_batch_size, self.max_decoding_tokens,
                self.store["buffer_manager"])

    def _instantiate_algorithms(self):
        self.algs = Algorithms()
        self.algs.decoder = tllm.internal.runtime.GptDecoderBatched(
            stream=self.store["cuda_stream"],
            speculative_decoding_mode=tllm.internal.runtime.
            SpeculativeDecodingMode.NoneType(),
            dtype=self.logits_datatype)
        self.algs.decoder.setup(
            mode=self.decoding_mode,
            max_batch_size=self.executor_config.max_batch_size,
            max_beam_width=self.executor_config.max_beam_width,
            max_attention_window=self.max_attention_window,
            sink_token_length=0,
            max_sequence_length=self.executor_config.max_seq_len,
            max_tokens_per_step=self.max_decoding_tokens,
            dtype=self.logits_datatype,
            model_config=self.model_config,
            world_config=self.world_config)
        self.algs.assign_req_seq_slots = tllm.internal.algorithms.AssignReqSeqSlots(
        )
        self.algs.generate_request_options = tllm.internal.algorithms.GenerateRequestOptions(
            speculative_decoding_fast_logits=False,
            is_leader_in_orch_mode=False,
            is_normalize_log_probs=False)
        self.algs.create_new_decoder_requests = tllm.internal.algorithms.CreateNewDecoderRequests(
        )
        self.algs.handle_context_logits = tllm.internal.algorithms.HandleContextLogits(
        )
        self.algs.handle_generation_logits = tllm.internal.algorithms.HandleGenerationLogits(
        )
        self.algs.make_decoding_batch_input_output = tllm.internal.algorithms.MakeDecodingBatchInputOutput(
        )

    def setup_decoder(self, scheduled_requests: ScheduledRequests,
                      model_outputs):
        self.batch_size = scheduled_requests.batch_size

        for req in itertools.chain(scheduled_requests.context_requests,
                                   scheduled_requests.generation_requests):
            self.beam_width = req.sampling_config.beam_width
            break

        logits = model_outputs["logits"].reshape(
            (self.batch_size, self.beam_width, -1))

        with torch.inference_mode():
            self.algs.assign_req_seq_slots(
                self.store["seq_slot_manager"],
                scheduled_requests.context_requests,
                scheduled_requests.generation_requests)

            batch_slots, decoder_requests, sampling_configs = self.algs.generate_request_options(
                self.model_config, self.world_config, self.decoding_config,
                scheduled_requests.context_requests,
                self.store["buffer_manager"], self.logits_datatype,
                self.store["decoder_input_buffers"])

            if len(decoder_requests):
                self.algs.create_new_decoder_requests(
                    batch_slots, decoder_requests, sampling_configs,
                    self.model_config, self.algs.decoder,
                    self.store["cuda_stream"], self.executor_config.max_seq_len)

                local_batch_size = len(batch_slots)
                sampling_config = tllm.make_sampling_config(sampling_configs)
                self.algs.decoder.underlying_decoder().setup(
                    sampling_config, local_batch_size, batch_slots,
                    self.algs.decoder.joint_decoding_output, decoder_requests)

            # Note: In runtimeBuffers.cpp, num_context_logits is set to:
            #       numContextLogits.at(batchIdx) = modelConfig.computeContextLogits() ? contextChunkSize : 1;
            # Revisit this when we support chunked context.
            num_context_logits = [1] * self.batch_size
            logits_index = self.algs.handle_context_logits(
                scheduled_requests.context_requests, num_context_logits, logits,
                self.store["decoder_buffers"], self.model_config,
                self.store["buffer_manager"], self.store["cuda_stream"])
            self.algs.handle_generation_logits(
                logits_index, scheduled_requests.generation_requests,
                self.store["decoder_buffers"], self.model_config,
                self.store["buffer_manager"], logits)
            decoding_input, self.decoding_output = self.algs.make_decoding_batch_input_output(
                scheduled_requests.context_requests,
                scheduled_requests.generation_requests,
                self.store["decoder_buffers"],
                self.store["decoder_input_buffers"], self.model_config,
                self.max_num_sequences, self.beam_width,
                self.store["buffer_manager"], self.store["cuda_stream"])
            self.algs.decoder.forward_async(self.decoding_output,
                                            decoding_input)

            self.decoder_event = torch.cuda.Event()
            self.decoder_event.record()

    def update_requests(self, scheduled_requests: ScheduledRequests):
        self.decoder_event.synchronize()

        # Note: self.algs.decoder.all_new_tokens will be populated after the synchronize
        new_tokens_host = self.algs.decoder.all_new_tokens.to('cpu',
                                                              non_blocking=True)
        finished_sum_host = self.algs.decoder.finished_sum.to('cpu',
                                                              non_blocking=True)
        finish_reasons_host = self.algs.decoder.finish_reasons.to(
            'cpu', non_blocking=True)
        sequence_lengths_host_data = self.store[
            "decoder_buffers"].sequence_lengths.to('cpu', non_blocking=True)

        self.decoder_event.record()
        self.decoder_event.synchronize()

        for request in itertools.chain(scheduled_requests.context_requests,
                                       scheduled_requests.generation_requests):
            if request.is_context_init_state:
                continue

            seq_slot = request.seq_slot
            num_generated_tokens = request.num_draft_tokens + 1
            current_num_of_tokens = request.max_beam_num_tokens

            num_new_tokens = [0] * self.beam_width
            num_dropped_tokens = [0] * self.beam_width

            for beam in range(self.beam_width):
                seq_len = sequence_lengths_host_data[seq_slot * self.beam_width
                                                     + beam].item()
                num_new_tokens[beam] = min(
                    num_generated_tokens,
                    seq_len - request.get_num_tokens(beam))
                num_dropped_tokens[
                    beam] = num_generated_tokens - num_new_tokens[beam]

                for step in range(num_new_tokens[beam]):
                    new_token = new_tokens_host[step][seq_slot][beam]
                    request.add_new_token(new_token, beam)

                finish_reason = finish_reasons_host[seq_slot * self.beam_width +
                                                    beam].item()
                request.set_finished_reason(
                    tllm.executor.FinishReason(finish_reason), beam)

            # Set number of tokens predicted per runtime iteration. Will be > 1 for speculative decoding.
            request.update_num_tokens_per_iteration(
                request.max_beam_num_tokens - current_num_of_tokens,
                self.model_config)

            if finished_sum_host[seq_slot] == self.beam_width:
                request.state = LlmRequestState.GENERATION_COMPLETE

    def decode_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs):
        pass

    def decode(self, scheduled_requests: ScheduledRequests, model_outputs):
        self.update_requests(scheduled_requests)
