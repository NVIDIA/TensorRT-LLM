from abc import ABC, abstractmethod

import torch

from tensorrt_llm.bindings import executor as tllm_executor

from .llm_request import *
from .scheduler import ScheduledRequests


class Decoder(ABC):

    @abstractmethod
    def decode(self, scheduled_requests: ScheduledRequests, model_outputs):
        raise NotImplementedError

    @abstractmethod
    def decode_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs):
        raise NotImplementedError

    @abstractmethod
    def update_requests(self, scheduled_requests: ScheduledRequests,
                        new_tokens_host: torch.tensor,
                        decoder_event: torch.cuda.Event):
        raise NotImplementedError


class DummyDecoder(Decoder):

    def decode(self, scheduled_requests: ScheduledRequests, model_outputs):
        for request in scheduled_requests.context_requests:
            request.add_new_token(500, 0)
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
        for request in scheduled_requests.generation_requests:
            if request.get_num_tokens(0) < 10:
                request.add_new_token(request.get_num_tokens(0) + 1000, 0)
            else:
                request.state = LlmRequestState.GENERATION_COMPLETE


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

    def __init__(self, mixed_decoder: bool = False):
        self.mixed_decoder = mixed_decoder

    def _meet_max_token_stop_criteria(self, request: LlmRequest,
                                      num_tokens: int):

        return num_tokens - request.py_orig_prompt_len >= request.py_max_new_tokens

    def _meet_stop_token_criteria(self, request: LlmRequest):
        if not hasattr(request, 'py_stop_words_list'):
            return False
        assert isinstance(request.py_stop_words_list,
                          list), "request.py_stop_words_list should be a list"
        stop_words_list = request.py_stop_words_list
        tokens = request.get_tokens(0)
        for stop_word in stop_words_list:
            if isinstance(stop_word, int):
                stop_word = [stop_word]
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
            request.set_finished_reason(tllm_executor.FinishReason.END_ID,
                                        beam_idx)
            return True

        if self._meet_max_token_stop_criteria(request, num_tokens):
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.set_finished_reason(tllm_executor.FinishReason.LENGTH,
                                        beam_idx)
            return True

        if self._meet_stop_token_criteria(request):
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.set_finished_reason(tllm_executor.FinishReason.STOP_WORDS,
                                        beam_idx)
            return True

        return False

    def update_requests(self, scheduled_requests: ScheduledRequests,
                        new_tokens_host: torch.tensor,
                        decoder_event: torch.cuda.Event):
        decoder_event.synchronize()
        new_tokens_list = new_tokens_host.tolist()

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
            if request.has_draft_tokens():
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
                for i in range(len(request.draft_tokens)):
                    draft_token = request.draft_tokens[i]
                    if draft_token != new_token:
                        # Reject.
                        break

                    new_token = new_tokens_list[idx + i + 1]
                    num_tokens = request.add_new_token(new_token, beam_idx)

                    if self._handle_stop_criteria(request, new_token,
                                                  num_tokens, beam_idx):
                        break
            idx += len(request.draft_tokens) + 1

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
            assert not request.has_draft_tokens(
            ), "Speculative decoding not supported in SeparateDecoder."
            token_logits = logits[idx:idx + 1, :]
            new_token = decode_single_request(request, token_logits)
            new_tokens_device_array += [new_token]
            idx += 1

        new_tokens_device = torch.cat(new_tokens_device_array)
        new_tokens_host = new_tokens_device.to('cpu', non_blocking=True)
        decoder_event = torch.cuda.Event()
        decoder_event.record()
        return new_tokens_device, new_tokens_host, decoder_event

    def _batch_decode(self, scheduled_requests: ScheduledRequests,
                      model_outputs):
        logits = model_outputs["logits"]
        new_tokens_device = torch.argmax(logits, dim=-1)
        new_tokens_host = new_tokens_device.to('cpu', non_blocking=True)
        decoder_event = torch.cuda.Event()
        decoder_event.record()
        return new_tokens_device, new_tokens_host, decoder_event

    def decode_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs):
        if self.mixed_decoder:
            return self._mixed_decode(scheduled_requests, model_outputs)
        else:
            return self._batch_decode(scheduled_requests, model_outputs)

    def decode(self, scheduled_requests: ScheduledRequests, model_outputs):
        _, new_tokens_host, decoder_event = self.decode_async(
            scheduled_requests, model_outputs)
        self.update_requests(scheduled_requests, new_tokens_host, decoder_event)


class TorchStarAttentionDecoder(TorchDecoder):

    def update_requests(self, scheduled_requests: ScheduledRequests,
                        new_tokens_host: torch.tensor,
                        decoder_event: torch.cuda.Event):
        decoder_event.synchronize()
        new_tokens_list = new_tokens_host.tolist()

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
