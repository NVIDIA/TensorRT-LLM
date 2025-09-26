# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


class RequestValidationError(Exception):
    pass


def _validate_that(condition: bool, msg: str):
    if not condition:
        raise RequestValidationError(msg)


def _validate_non_empty(data, msg: str):
    if isinstance(data, torch.Tensor):
        _validate_that(data is not None and data.numel() > 0, msg)
    else:
        _validate_that(data is not None and data.size > 0, msg)


def _validate_single_gt_0(data, msg: str):
    _validate_non_empty(data, msg)
    _validate_that(data.flatten()[0] > 0, msg)


def _single_value(data: Optional[np.ndarray]):
    if data is None:
        return None
    return data.flatten()[0]


@dataclass
class Request:
    text_input: np.ndarray = field(default_factory=lambda: np.array([]))
    decoder_text_input: np.ndarray = None
    image_input: Optional[np.ndarray] = None
    image_bytes_input: Optional[np.ndarray] = None
    image_url_input: Optional[np.ndarray] = None
    video_bytes_input: Optional[np.ndarray] = None
    max_tokens: Optional[np.ndarray] = None
    bad_words: Optional[np.ndarray] = None
    stop_words: Optional[np.ndarray] = None
    end_id: Optional[np.ndarray] = None
    pad_id: Optional[np.ndarray] = None
    top_k: Optional[np.ndarray] = None
    top_p: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    length_penalty: Optional[np.ndarray] = None
    repetition_penalty: Optional[np.ndarray] = None
    min_tokens: Optional[np.ndarray] = None
    return_log_probs: Optional[np.ndarray] = None
    prompt_embedding_table: Optional[np.ndarray] = None
    prompt_vocab_size: Optional[np.ndarray] = None
    prompt_table_extra_id: Optional[np.ndarray] = None
    embedding_bias_words: Optional[np.ndarray] = None
    embedding_bias_weights: Optional[np.ndarray] = None
    num_draft_tokens: Optional[np.ndarray] = None
    use_draft_logits: Optional[np.ndarray] = None
    stream: Optional[np.ndarray] = None
    beam_width: Optional[np.ndarray] = None
    return_context_logits: Optional[np.ndarray] = None
    return_generation_logits: Optional[np.ndarray] = None
    seed: Optional[np.ndarray] = None
    presence_penalty: Optional[np.ndarray] = None
    frequency_penalty: Optional[np.ndarray] = None
    lora_task_id: Optional[np.ndarray] = None
    lora_weights: Optional[np.ndarray] = None
    lora_config: Optional[np.ndarray] = None
    exclude_input_in_output: Optional[np.ndarray] = None
    return_perf_metrics: Optional[np.ndarray] = None
    guided_decoding_guide_type: Optional[np.ndarray] = None
    guided_decoding_guide: Optional[np.ndarray] = None
    request_id: Optional[str] = None
    mrope_rotary_cos_sin: Optional[np.ndarray] = None
    mrope_position_deltas: Optional[np.ndarray] = None
    image_sizes_input: Optional[np.ndarray] = None

    def validate(self):
        _validate_non_empty(self.text_input, "text_input is required")
        _validate_single_gt_0(self.max_tokens,
                              "max_tokens must be a single value > 0")

        num_draft_tokens = _single_value(self.num_draft_tokens)
        _single_value(self.return_generation_logits)
        context_logits = _single_value(self.return_context_logits)

        if num_draft_tokens:
            _validate_that(
                not self.stream.any(),
                "streaming is not supported with speculative decoding")
            _validate_that(
                not context_logits,
                "context logits are not supported with speculative decoding")


@dataclass
class DraftRequest:
    draft_input_ids: Optional[np.ndarray] = None
    draft_logits: Optional[np.ndarray] = None


@dataclass
class PreprocResponse:
    input_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    decoder_input_ids: np.ndarray = None
    input_lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    decoder_input_lengths: np.ndarray = None
    bad_words_list: Optional[np.ndarray] = None
    stop_words_list: Optional[np.ndarray] = None
    embedding_bias: Optional[np.ndarray] = None
    end_id: Optional[np.ndarray] = None
    pad_id: Optional[np.ndarray] = None
    prompt_table_extra_ids: Optional[np.ndarray] = None
    pixel_values: Optional[np.ndarray] = None
    image_sizes: Optional[np.ndarray] = None
    is_video_input: Optional[np.ndarray] = None
    attention_mask: Optional[np.ndarray] = None
    image_grid_thw: Optional[np.ndarray] = None
    vision_input_id: Optional[np.ndarray] = None

    @classmethod
    def with_new_inputs(cls,
                        other,
                        input_ids: Optional[np.ndarray] = None,
                        input_lengths: Optional[np.ndarray] = None):
        return cls(
            input_ids=(input_ids if input_ids is not None else other.input_ids),
            input_lengths=(input_lengths if input_lengths is not None else
                           other.input_lengths),
            decoder_input_ids=other.decoder_input_ids,
            decoder_input_lengths=other.decoder_input_lengths,
            bad_words_list=other.bad_words_list,
            stop_words_list=other.stop_words_list,
            end_id=other.end_id,
            pad_id=other.pad_id,
            prompt_table_extra_ids=other.prompt_table_extra_ids)


@dataclass
class MultimodalEncResponse:
    prompt_embedding_table: Optional[torch.Tensor] = None
    prompt_vocab_size: Optional[np.ndarray] = None
    mrope_rotary_cos_sin: Optional[np.ndarray] = None
    mrope_position_deltas: Optional[np.ndarray] = None


@dataclass
class GenerationResponse:
    output_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    sequence_length: np.ndarray = field(default_factory=lambda: np.array([]))
    cum_log_probs: Optional[np.ndarray] = None
    output_log_probs: Optional[np.ndarray] = None
    context_logits: Optional[np.ndarray] = None
    generation_logits: Optional[np.ndarray] = None
    batch_index: Optional[np.ndarray] = None
    sequence_index: Optional[np.ndarray] = None
    kv_cache_alloc_new_blocks: Optional[np.ndarray] = None
    kv_cache_reused_blocks: Optional[np.ndarray] = None
    kv_cache_alloc_total_blocks: Optional[np.ndarray] = None
    arrival_time_ns: Optional[np.ndarray] = None
    first_scheduled_time_ns: Optional[np.ndarray] = None
    first_token_time_ns: Optional[np.ndarray] = None
    last_token_time_ns: Optional[np.ndarray] = None
    acceptance_rate: Optional[np.ndarray] = None
    total_accepted_draft_tokens: Optional[np.ndarray] = None
    total_draft_tokens: Optional[np.ndarray] = None


@dataclass
class Response:
    text_output: np.ndarray = field(default_factory=lambda: np.array([]))
    cum_log_probs: Optional[np.ndarray] = None
    output_log_probs: Optional[np.ndarray] = None
    context_logits: Optional[np.ndarray] = None
    generation_logits: Optional[np.ndarray] = None
    batch_index: Optional[np.ndarray] = None
    sequence_index: Optional[np.ndarray] = None
    kv_cache_alloc_new_blocks: Optional[np.ndarray] = None
    kv_cache_reused_blocks: Optional[np.ndarray] = None
    kv_cache_alloc_total_blocks: Optional[np.ndarray] = None
    arrival_time_ns: Optional[np.ndarray] = None
    first_scheduled_time_ns: Optional[np.ndarray] = None
    first_token_time_ns: Optional[np.ndarray] = None
    last_token_time_ns: Optional[np.ndarray] = None
    acceptance_rate: Optional[np.ndarray] = None
    total_accepted_draft_tokens: Optional[np.ndarray] = None
    total_draft_tokens: Optional[np.ndarray] = None

    def __eq__(self, o) -> bool:
        """Just for testing"""
        if not isinstance(o, Response):
            return False
        return (np.array_equal(self.text_output, o.text_output)
                and np.array_equal(self.cum_log_probs, o.cum_log_probs)
                and np.array_equal(self.output_log_probs, o.output_log_probs)
                and np.array_equal(self.context_logits, o.context_logits)
                and np.array_equal(self.generation_logits, o.generation_logits)
                and np.array_equal(self.batch_index, o.batch_index)
                and np.array_equal(self.sequence_index, o.sequence_index)
                and np.array_equal(self.kv_cache_alloc_new_blocks,
                                   o.kv_cache_alloc_new_blocks)
                and np.array_equal(self.kv_cache_reused_blocks,
                                   o.kv_cache_reused_blocks)
                and np.array_equal(self.kv_cache_alloc_total_blocks,
                                   o.kv_cache_alloc_total_blocks)
                and np.array_equal(self.arrival_time_ns, o.arrival_time_ns)
                and np.array_equal(self.first_scheduled_time_ns,
                                   o.first_scheduled_time_ns) and
                np.array_equal(self.first_token_time_ns, o.first_token_time_ns)
                and np.array_equal(self.last_token_time_ns,
                                   o.last_token_time_ns)
                and np.array_equal(self.acceptance_rate, o.acceptance_rate)
                and np.array_equal(self.total_accepted_draft_tokens,
                                   o.total_accepted_draft_tokens) and
                np.array_equal(self.total_draft_tokens, o.total_draft_tokens))


class Decoder:

    def __init__(self, streaming=False, accumulate=False):
        self._streaming = streaming
        self._accumulate = accumulate

        self._accumulated_tokens = []

    def decode(self,
               request: Request,
               speculative_decoding=False,
               is_multimodal=False) -> Generator[Response, None, None]:

        batch_size = request.text_input.shape[0]
        self._accumulated_tokens = [None] * batch_size
        preproc_response = self.preprocess(request)

        multimodal_enc_response = None
        if is_multimodal:
            multimodal_enc_response = self._multimodal_enc_generate(
                request, preproc_response)

        if speculative_decoding:
            if batch_size > 1:
                raise Exception(
                    "speculative decoding is not supported with batch size > 1")
            for gen_response in self._spec_generate(preproc_response, request):
                yield self.postprocess(gen_response, batch_size)
        else:
            if not self._streaming and batch_size == 1:
                gen_response = self._generate_non_streaming(
                    preproc_response,
                    request,
                    multimodal_enc_response=multimodal_enc_response)
                yield self.postprocess(gen_response, batch_size)
            else:
                for gen_response in self._generate(
                        preproc_response,
                        request,
                        multimodal_enc_response=multimodal_enc_response):
                    yield self.postprocess(gen_response, batch_size)

    def encountered_stop_words(self, input_ids, stop_words_ids):
        for stop_word_ids in stop_words_ids:
            if np.array_equal(input_ids[-len(stop_word_ids):], stop_word_ids):
                return True
        return False

    def _spec_generate(
            self, preproc: PreprocResponse,
            request: Request) -> Generator[GenerationResponse, None, None]:

        if preproc.input_ids.shape[0] > 1:
            raise Exception(
                "Speculative decoding does not support batch size > 1.")

        prompt_input_ids: np.ndarray = preproc.input_ids[0]
        input_ids: np.ndarray = prompt_input_ids
        output_len: int = request.max_tokens[0][0]
        last_input_ids: np.ndarray = None
        draft_output_ids: np.ndarray = None
        draft_logits: np.ndarray = None

        target_response: GenerationResponse = None

        cur_preproc = preproc

        counter = 0
        while True:
            counter += 1
            num_draft_tokens = min(
                request.num_draft_tokens[0][0],
                len(prompt_input_ids) + output_len - len(input_ids) - 1)

            draft_request = None
            if num_draft_tokens > 0:
                request.min_tokens = np.array([num_draft_tokens],
                                              dtype=np.int32)
                draft_response: GenerationResponse = self._draft_generate_non_streaming(
                    cur_preproc, request, num_draft_tokens)
                seq_len: int = draft_response.sequence_length[0][0]
                # [1, beamWidth, outputLength] -> [outputLen]
                draft_output_ids = draft_response.output_ids[0][0]
                # [1, beamWidth, outputLength, vocabSizePadded] -> [outputLength, vocabSizePadded]
                if request.use_draft_logits is not None and request.use_draft_logits[
                        0]:
                    if draft_response.generation_logits is not None:
                        draft_logits = draft_response.generation_logits[0][0]

                input_draft_tokens = draft_output_ids[len(input_ids):seq_len]
                if len(input_draft_tokens) > 0:
                    draft_request = DraftRequest(
                        draft_input_ids=np.expand_dims(input_draft_tokens, 0))
                    if request.use_draft_logits is not None and request.use_draft_logits[
                            0]:
                        draft_request.draft_logits = np.expand_dims(
                            draft_logits[-len(input_draft_tokens):], 0)
                else:
                    draft_request = DraftRequest()
                request.min_tokens = None
            else:
                draft_request = DraftRequest()
            target_response = self._generate_non_streaming(
                cur_preproc, request, draft_request)
            last_input_ids = input_ids
            input_ids = target_response.output_ids[0][0]
            cur_preproc = PreprocResponse.with_new_inputs(
                cur_preproc, np.expand_dims(input_ids, 0),
                np.array([[len(input_ids)]], dtype=np.int32))

            # Evaluate criteria to stop generation loop.
            # If we've hit or exceeded the max output length, should stop
            length_stop = (len(input_ids) >= len(prompt_input_ids) + output_len)
            if length_stop:
                break
            # If draft and target have same outputs, should stop. Normally target should return 1 more token.
            # If they are the same length, they should differ at the last token
            target_draft_equal = draft_output_ids is not None and np.array_equal(
                draft_output_ids, input_ids)
            if target_draft_equal:
                break
            # If tokens no longer change, should stop, means we have hit early stopping
            last_current_equal = np.array_equal(last_input_ids, input_ids)
            if last_current_equal:
                break
            # Need to check if stop words was encountered
            hit_stop_words = self.encountered_stop_words(
                input_ids, preproc.stop_words_list[0])
            if hit_stop_words:
                break

        yield target_response

    def _draft_generate_non_streaming(
            self, preproc: PreprocResponse, request: Request,
            num_draft_tokens: int) -> GenerationResponse:
        raise NotImplementedError()

    def _multimodal_enc_generate(
        self,
        request: Request,
    ) -> MultimodalEncResponse:
        raise NotImplementedError()

    def _generate(
        self,
        preproc: PreprocResponse,
        request: Request,
        draft_request: Optional[DraftRequest] = None,
        multimodal_enc_response: Optional[MultimodalEncResponse] = None,
    ) -> Generator[GenerationResponse, None, None]:
        raise NotImplementedError()

    def _generate_non_streaming(
        self,
        preproc: PreprocResponse,
        request: Request,
        draft_request: Optional[DraftRequest] = None,
        multimodal_enc_response: Optional[MultimodalEncResponse] = None,
    ) -> GenerationResponse:
        raise NotImplementedError()

    def send_cancellation_request(self, request_id, decoupled):
        raise NotImplementedError()

    def postprocess(self, gen_response: GenerationResponse,
                    batch_size) -> Response:
        if self._accumulate and self._streaming:
            new_tokens: np.ndarray = gen_response.output_ids
            if new_tokens.ndim != 3:
                raise Exception("Expected output_ids tensor to have 3 dims.")
            if new_tokens.shape[0] != 1:
                raise Exception("Expected batch size of 1")
            if new_tokens.shape[1] != 1:
                raise Exception(
                    "Accumulation of tokens is only implemented for beam width = 1"
                )

            batch_index = gen_response.batch_index
            if batch_index is not None:
                if batch_index.ndim != 2:
                    raise Exception(
                        "Expected batch_index tensor to have 2 dims.")
                if batch_index.shape[0] != 1:
                    raise Exception("Expected batch size of 1")
                if batch_index.shape[1] != 1:
                    raise Exception("Expected only one batch_index")

            batch_index = batch_index[0][0] if batch_index is not None else 0

            self._accumulated_tokens[batch_index] = new_tokens if (
                self._accumulated_tokens[batch_index]
                is None) else np.concatenate(
                    (self._accumulated_tokens[batch_index], new_tokens), axis=2)
            sequence_lengths = np.array(
                [[self._accumulated_tokens[batch_index].shape[2]]],
                dtype=np.int32)
            return self._postprocess(self._accumulated_tokens[batch_index],
                                     sequence_lengths, gen_response)
        else:
            return self._postprocess(gen_response.output_ids, None,
                                     gen_response)

    def _postprocess(self, tokens: np.ndarray,
                     sequence_lengths: Optional[np.ndarray],
                     gen_response: GenerationResponse) -> Response:
        raise NotImplementedError()

    def preprocess(self, request: Request) -> PreprocResponse:
        raise NotImplementedError()

    def reset_decoder(self):
        self._accumulated_tokens = []
