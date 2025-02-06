from abc import ABC, abstractmethod

import torch

from tensorrt_llm.bindings import executor as tllm_executor
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .llm_request import *
from .scheduler import ScheduledRequests


class Decoder(ABC):

    @abstractmethod
    def decode(self, scheduled_requests: ScheduledRequests, model_outputs):
        raise NotImplementedError

    @abstractmethod
    def decode_async(self, model_outputs):
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


class TorchGreedySearchDecoder(Decoder):

    def __init__(self, mapping: Mapping = None):
        self.mapping = mapping

    def _meet_max_token_stop_criteria(self, request: LlmRequest,
                                      num_tokens: int):

        return num_tokens - request.py_orig_prompt_len >= request.py_max_new_tokens

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

        return False

    def _decode_single_ctx_block(self, scheduled_requests: ScheduledRequests,
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

    def _decode_star_attention(self, scheduled_requests: ScheduledRequests,
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

    def decode_async(self, model_outputs):
        logits = model_outputs["logits"]
        new_tokens_device = torch.argmax(logits, dim=-1)
        new_tokens_host = new_tokens_device.to('cpu', non_blocking=True)
        decoder_event = torch.cuda.Event()
        decoder_event.record()
        return new_tokens_device, new_tokens_host, decoder_event

    def update_requests(self, scheduled_requests: ScheduledRequests,
                        new_tokens_host: torch.tensor,
                        decoder_event: torch.cuda.Event):
        if self.mapping and self.mapping.cp_config:
            cp_config = self.mapping.cp_config
            if 'cp_type' in cp_config.keys():
                cp_type = cp_config['cp_type']
                if cp_type == 'star_attention':
                    self._decode_star_attention(scheduled_requests,
                                                new_tokens_host, decoder_event)
                    return
                else:
                    assert False, f"Unsupported cp_type {cp_type}"
            else:
                logger.info(
                    f"can't find 'cp_type' in cp_config, will fall back to default decoding mode"
                )

        self._decode_single_ctx_block(scheduled_requests, new_tokens_host,
                                      decoder_event)

    def decode(self, scheduled_requests: ScheduledRequests, model_outputs):
        _, new_tokens_host, decoder_event = self.decode_async(model_outputs)
        self.update_requests(scheduled_requests, new_tokens_host, decoder_event)
