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

from collections.abc import Callable
from typing import Dict, Optional

import numpy as np
import triton_python_backend_utils as pb_utils
from lib.decode import *
from torch.utils.dlpack import from_dlpack, to_dlpack
from typing_extensions import override


class TritonDecoder(Decoder):

    def __init__(self,
                 streaming=False,
                 accumulate=False,
                 preproc_model_name="preprocessing",
                 postproc_model_name="postprocessing",
                 llm_model_name="tensorrt_llm",
                 draft_llm_model_name: Optional[str] = None,
                 multimodal_encoders_name: Optional[str] = None):
        super().__init__(streaming=streaming, accumulate=accumulate)
        self.preproc_model_name = preproc_model_name
        self.postproc_model_name = postproc_model_name
        self.llm_model_name = llm_model_name
        self.draft_llm_model_name = draft_llm_model_name
        self.multimodal_encoders_name = multimodal_encoders_name

        self._preproc_outputs = [
            "INPUT_ID", "DECODER_INPUT_ID", "REQUEST_INPUT_LEN",
            "REQUEST_DECODER_INPUT_LEN", "BAD_WORDS_IDS", "STOP_WORDS_IDS",
            "EMBEDDING_BIAS", "OUT_PAD_ID", "OUT_END_ID",
            "OUT_PROMPT_TABLE_EXTRA_IDS", "PIXEL_VALUES", "IMAGE_SIZES",
            "IS_VIDEO_INPUT", "VISION_INPUT_ID", "ATTENTION_MASK",
            "IMAGE_GRID_THW"
        ]

        self._multimodal_enc_outputs = [
            "OUT_PROMPT_EMBEDDING_TABLE", "OUT_PROMPT_VOCAB_SIZE",
            "MROPE_ROTARY_COS_SIN", "MROPE_POSITION_DELTAS"
        ]

        self._llm_outputs = [
            "output_ids", "sequence_length", "cum_log_probs",
            "output_log_probs", "context_logits", "generation_logits",
            "batch_index", "sequence_index", "kv_cache_alloc_new_blocks",
            "kv_cache_reused_blocks", "kv_cache_alloc_total_blocks",
            "arrival_time_ns", "first_scheduled_time_ns", "first_token_time_ns",
            "last_token_time_ns", "acceptance_rate",
            "total_accepted_draft_tokens", "total_draft_tokens"
        ]

        self._postproc_outputs = [
            "OUTPUT",
        ]

        self.input_names = [
            "text_input", "decoder_text_input", "image_input",
            "image_bytes_input", "image_url_input", "video_bytes_input",
            "max_tokens", "bad_words", "stop_words", "end_id", "pad_id",
            "top_k", "top_p", "temperature", "length_penalty",
            "repetition_penalty", "min_tokens", "presence_penalty",
            "frequency_penalty", "seed", "return_log_probs",
            "return_context_logits", "return_generation_logits", "beam_width",
            "stream", "prompt_embedding_table", "prompt_vocab_size",
            "prompt_table_extra_id", "embedding_bias_words",
            "embedding_bias_weights", "num_draft_tokens", "use_draft_logits",
            "lora_task_id", "lora_weights", "lora_config",
            "exclude_input_in_output", "return_perf_metrics",
            "guided_decoding_guide_type", "guided_decoding_guide"
        ]

        self.__undo_reshape_whitelist = {
            "max_tokens", "end_id", "pad_id", "top_k", "top_p", "temperature",
            "length_penalty", "repetition_penalty", "min_tokens",
            "presence_penalty", "frequency_penalty", "seed", "return_log_probs",
            "return_context_logits", "return_generation_logits", "beam_width",
            "stream", "prompt_vocab_size", "num_draft_tokens",
            "use_draft_logits", "exclude_input_in_output",
            "return_perf_metrics", "lora_weights", "lora_config", "lora_task_id"
        }

    def _exec_triton_request(self, request):
        responses = request.exec(decoupled=True)
        for r in responses:
            if r.has_error():
                raise pb_utils.TritonModelException(r.error().message())
            yield r

    def _exec_triton_request_single(self, request):
        responses = request.exec(decoupled=False)
        if responses.has_error():
            raise pb_utils.TritonModelException(responses.error().message())
        return responses

    def create_triton_response(self, response: Response):
        name_map = {
            "text_output": "text_output",
            "cum_log_probs": "cum_log_probs",
            "output_log_probs": "output_log_probs",
            "context_logits": "context_logits",
            "generation_logits": "generation_logits",
            "batch_index": "batch_index",
            "sequence_index": "sequence_index",
            "kv_cache_alloc_new_blocks": "kv_cache_alloc_new_blocks",
            "kv_cache_reused_blocks": "kv_cache_reused_blocks",
            "kv_cache_alloc_total_blocks": "kv_cache_alloc_total_blocks",
            "arrival_time_ns": "arrival_time_ns",
            "first_scheduled_time_ns": "first_scheduled_time_ns",
            "first_token_time_ns": "first_token_time_ns",
            "last_token_time_ns": "last_token_time_ns",
            "acceptance_rate": "acceptance_rate",
            "total_accepted_draft_tokens": "total_accepted_draft_tokens",
            "total_draft_tokens": "total_draft_tokens"
        }
        tensors = self.create_triton_tensors(response, name_map)
        return pb_utils.InferenceResponse(output_tensors=tensors)

    def convert_triton_request(self, triton_request) -> Request:
        request = Request()
        for triton_name in self.input_names:
            tensor = pb_utils.get_input_tensor_by_name(triton_request,
                                                       triton_name)
            target_name = triton_name
            if tensor is None:
                continue
            if not hasattr(request, target_name):
                raise AttributeError(
                    f"Request has no attribute '{target_name}'")
            setattr(request, target_name, tensor.as_numpy())
        return request

    def convert_triton_response(self,
                                triton_response,
                                response_factory: Callable,
                                name_map=None):
        response = response_factory()
        for tensor in triton_response.output_tensors():
            if tensor is None:
                continue
            triton_name = tensor.name()
            if tensor.is_cpu():
                try:
                    value = tensor.as_numpy()
                except pb_utils.TritonModelException as e:
                    # Use to_dlpack()/from_dlpack() if as_numpy() fails,
                    # e.g. in case of BF16 tensors
                    value = from_dlpack(tensor.to_dlpack())
            else:
                # If the tensor is in GPU memory make it torch.Tensor type
                value = from_dlpack(tensor.to_dlpack())
            target_name = triton_name
            if name_map and triton_name in name_map:
                target_name = name_map[triton_name]
            if name_map and not triton_name in name_map:
                continue
            if target_name is None:
                # explicitly ignore this triton input
                continue
            if not hasattr(response, target_name):
                raise AttributeError(
                    f"response object has not attribute '{target_name}'")
            setattr(response, target_name, value)
        return response

    def __undo_reshape(self, x, name):
        if name in self.__undo_reshape_whitelist and len(x.shape) == 1:
            # handle reshapes
            return np.expand_dims(x, 0)
        else:
            return x

    def send_cancellation_request(self, request_id, decoupled):
        tensors = []
        tensors.append(
            pb_utils.Tensor("input_ids", np.empty([1, 1], dtype=np.int32)))
        tensors.append(
            pb_utils.Tensor("input_lengths", np.zeros([1, 1], dtype=np.int32)))
        tensors.append(
            pb_utils.Tensor("request_output_len", np.array([[0]],
                                                           dtype=np.int32)))
        tensors.append(pb_utils.Tensor("stop", np.array([[True]],
                                                        dtype='bool')))

        inference_request = pb_utils.InferenceRequest(
            model_name=self.llm_model_name,
            requested_output_names=[],
            inputs=tensors,
            request_id=request_id)
        inference_response = inference_request.exec(decoupled=decoupled)
        if decoupled:
            inference_response = list(inference_response)[0]
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message())

    def create_triton_tensors(self, obj, name_map: dict):
        tensors = []
        for name, triton_name in name_map.items():
            if triton_name is None:
                continue
            value = getattr(obj, name)
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                t = pb_utils.Tensor(triton_name,
                                    self.__undo_reshape(value, name))
            elif isinstance(value, torch.Tensor):
                t = pb_utils.Tensor.from_dlpack(
                    triton_name, to_dlpack(self.__undo_reshape(value, name)))
            tensors.append(t)
        return tensors

    @override
    def preprocess(self, request: Request) -> PreprocResponse:
        input_tensors = self._get_preproc_tensors(request)
        triton_req = pb_utils.InferenceRequest(
            model_name=self.preproc_model_name,
            inputs=input_tensors,
            request_id=request.request_id,
            requested_output_names=self._preproc_outputs)
        triton_output = self._exec_triton_request_single(triton_req)
        return self._get_preproc_response(triton_output)

    def _get_preproc_tensors(self, request: Request):
        name_map = {
            "text_input": "QUERY",
            "image_bytes_input": "IMAGE_BYTES",
            "image_url_input": "IMAGE_URL",
            "image_sizes_input": "IMAGE_SIZES",
            "video_bytes_input": "VIDEO_BYTES",
            "decoder_text_input": "DECODER_QUERY",
            "max_tokens": "REQUEST_OUTPUT_LEN",
            "bad_words": "BAD_WORDS_DICT",
            "stop_words": "STOP_WORDS_DICT",
            "embedding_bias_words": "EMBEDDING_BIAS_WORDS",
            "embedding_bias_weights": "EMBEDDING_BIAS_WEIGHTS",
            "pad_id": "PAD_ID",
            "end_id": "END_ID",
            "prompt_table_extra_id": "PROMPT_TABLE_EXTRA_ID",
        }
        return self.create_triton_tensors(request, name_map)

    def _get_preproc_response(self, triton_output):
        name_map = {
            "INPUT_ID": "input_ids",
            "DECODER_INPUT_ID": "decoder_input_ids",
            "REQUEST_INPUT_LEN": "input_lengths",
            "REQUEST_DECODER_INPUT_LEN": "decoder_input_lengths",
            "BAD_WORDS_IDS": "bad_words_list",
            "STOP_WORDS_IDS": "stop_words_list",
            "EMBEDDING_BIAS": "embedding_bias",
            "OUT_PAD_ID": "pad_id",
            "OUT_END_ID": "end_id",
            "OUT_PROMPT_TABLE_EXTRA_IDS": "prompt_table_extra_ids",
            "PIXEL_VALUES": "pixel_values",
            "IMAGE_SIZES": "image_sizes",
            "IS_VIDEO_INPUT": "is_video_input",
            "ATTENTION_MASK": "attention_mask",
            "IMAGE_GRID_THW": "image_grid_thw",
            "VISION_INPUT_ID": "vision_input_id"
        }
        return self.convert_triton_response(triton_output, PreprocResponse,
                                            name_map)

    @override
    def _multimodal_enc_generate(
        self,
        request: Request,
        preproc: PreprocResponse,
    ) -> MultimodalEncResponse:
        input_tensors = self._get_multimodal_enc_tensors(request, preproc)
        triton_req = pb_utils.InferenceRequest(
            model_name=self.multimodal_encoders_name,
            inputs=input_tensors,
            request_id=request.request_id,
            requested_output_names=self._multimodal_enc_outputs)
        triton_output = self._exec_triton_request_single(triton_req)
        return self._get_multimodal_enc_response(triton_output)

    def _get_multimodal_enc_tensors(self, request: Request,
                                    preproc: PreprocResponse):
        name_map_request = {
            "image_input": "IMAGE",
        }
        name_map_preproc = {
            "pixel_values": "pixel_values",
            "image_sizes": "image_sizes",
            "is_video_input": "is_video_input",
            "attention_mask": "attention_mask",
            "image_grid_thw": "image_grid_thw",
            "vision_input_id": "vision_input_id"
        }
        tensors = []
        tensors.extend(self.create_triton_tensors(request, name_map_request))
        tensors.extend(self.create_triton_tensors(preproc, name_map_preproc))
        return tensors

    def _get_multimodal_enc_response(self, triton_output):
        name_map = {
            "OUT_PROMPT_EMBEDDING_TABLE": "prompt_embedding_table",
            "OUT_PROMPT_VOCAB_SIZE": "prompt_vocab_size",
            "MROPE_ROTARY_COS_SIN": "mrope_rotary_cos_sin",
            "MROPE_POSITION_DELTAS": "mrope_position_deltas"
        }
        return self.convert_triton_response(triton_output,
                                            MultimodalEncResponse, name_map)

    @override
    def _draft_generate_non_streaming(
            self, preproc: PreprocResponse, request: Request,
            num_draft_tokens: int) -> GenerationResponse:
        input_tensors = self._get_llm_tensors(preproc, request,
                                              num_draft_tokens, None, True)
        triton_req = pb_utils.InferenceRequest(
            model_name=self.draft_llm_model_name,
            inputs=input_tensors,
            request_id=request.request_id,
            requested_output_names=self._llm_outputs)
        triton_response = self._exec_triton_request_single(triton_req)
        llm_response = self._get_llm_response(triton_response)
        return llm_response

    @override
    def _generate(
        self,
        preproc: PreprocResponse,
        request: Request,
        draft_request: Optional[DraftRequest] = None,
        multimodal_enc_response: Optional[MultimodalEncResponse] = None
    ) -> Generator[GenerationResponse, None, None]:
        input_tensors = self._get_llm_tensors(
            preproc,
            request,
            None,
            draft_request,
            multimodal_enc_response=multimodal_enc_response)
        triton_req = pb_utils.InferenceRequest(
            model_name=self.llm_model_name,
            inputs=input_tensors,
            request_id=request.request_id,
            requested_output_names=self._llm_outputs)
        for r in self._exec_triton_request(triton_req):
            yield self._get_llm_response(r)

    @override
    def _generate_non_streaming(
        self,
        preproc: PreprocResponse,
        request: Request,
        draft_request: Optional[DraftRequest] = None,
        multimodal_enc_response: Optional[MultimodalEncResponse] = None
    ) -> GenerationResponse:
        input_tensors = self._get_llm_tensors(
            preproc,
            request,
            None,
            draft_request,
            multimodal_enc_response=multimodal_enc_response)
        triton_req = pb_utils.InferenceRequest(
            model_name=self.llm_model_name,
            inputs=input_tensors,
            request_id=request.request_id,
            requested_output_names=self._llm_outputs)
        r = self._exec_triton_request_single(triton_req)
        return self._get_llm_response(r)

    def _get_llm_tensors(self,
                         preproc: PreprocResponse,
                         request: Request,
                         num_output_tokens: Optional[int] = None,
                         draft_request: Optional[DraftRequest] = None,
                         is_draft_model_request: bool = False,
                         multimodal_enc_response: MultimodalEncResponse = None):
        tensors = []
        tensors.extend(self._get_tensors_from_preproc(preproc))
        if multimodal_enc_response is not None:
            tensors.extend(
                self._get_tensors_from_multimodal_enc(multimodal_enc_response))
        tensors.extend(
            self._get_llm_tensors_from_request(request, num_output_tokens,
                                               draft_request,
                                               is_draft_model_request))
        return tensors

    def _get_tensors_from_preproc(self, preproc: PreprocResponse):
        name_map = {
            "input_ids": "input_ids",
            "decoder_input_ids": "decoder_input_ids",
            "input_lengths": "input_lengths",
            "bad_words_list": "bad_words_list",
            "stop_words_list": "stop_words_list",
            "embedding_bias": "embedding_bias",
            "pad_id": "pad_id",
            "end_id": "end_id",
            "prompt_table_extra_ids": "prompt_table_extra_ids",
        }
        return self.create_triton_tensors(preproc, name_map)

    def _get_tensors_from_multimodal_enc(
            self, multimodal_enc_response: MultimodalEncResponse):
        name_map = {
            "prompt_embedding_table": "prompt_embedding_table",
            "prompt_vocab_size": "prompt_vocab_size",
            "mrope_rotary_cos_sin": "mrope_rotary_cos_sin",
            "mrope_position_deltas": "mrope_position_deltas"
        }
        return self.create_triton_tensors(multimodal_enc_response, name_map)

    def _get_llm_tensors_from_request(
            self,
            request: Request,
            num_output_tokens: Optional[int] = None,
            draft_request: Optional[DraftRequest] = None,
            is_draft_model_request: bool = False):
        name_map: Dict[str, Optional[str]] = {
            "beam_width": "beam_width",
            "top_k": "runtime_top_k",
            "top_p": "runtime_top_p",
            "temperature": "temperature",
            "length_penalty": "len_penalty",
            "repetition_penalty": "repetition_penalty",
            "min_tokens": "min_tokens",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
            "seed": "seed",
            "return_log_probs": "return_log_probs",
            "stream": "streaming",
            "prompt_embedding_table": "prompt_embedding_table",
            "prompt_vocab_size": "prompt_vocab_size",
            "lora_task_id": "lora_task_id",
            "lora_weights": "lora_weights",
            "lora_config": "lora_config",
            "exclude_input_in_output": "exclude_input_in_output",
            "return_perf_metrics": "return_perf_metrics",
            "guided_decoding_guide_type": "guided_decoding_guide_type",
            "guided_decoding_guide": "guided_decoding_guide"
        }
        batch_size = request.text_input.shape[0]
        tensors = self.create_triton_tensors(request, name_map)
        out_len_tensor = None
        if request.max_tokens is not None:
            out_len_tensor = request.max_tokens

        out_len = None
        if num_output_tokens is not None:
            out_len = num_output_tokens
        elif draft_request:
            out_len = len(
                draft_request.draft_input_ids[0]
            ) + 1 if draft_request.draft_input_ids is not None else 1

        if out_len is not None:
            out_len_tensor = [[out_len]] * batch_size

        if out_len_tensor is None:
            raise Exception("Could not determine request_output_len")
        else:
            tensors.append(
                pb_utils.Tensor("request_output_len",
                                np.array(out_len_tensor, dtype=np.int32)))

        if draft_request:
            if draft_request.draft_input_ids is not None:
                tensors.append(
                    pb_utils.Tensor("draft_input_ids",
                                    draft_request.draft_input_ids))
                if draft_request.draft_logits is not None and request.use_draft_logits is not None and request.use_draft_logits[
                        0]:
                    tensors.append(
                        pb_utils.Tensor("draft_logits",
                                        draft_request.draft_logits))

        return_context_logits_data = [False]
        return_generation_logits_data = [False]
        if draft_request is None:
            if is_draft_model_request:
                return_generation_logits_data = request.use_draft_logits if request.use_draft_logits is not None else [
                    False
                ]
            else:
                return_context_logits_data = request.return_context_logits if request.return_context_logits is not None else [
                    False
                ]
                return_generation_logits_data = request.return_generation_logits if request.return_generation_logits is not None else [
                    False
                ]
        return_context_logits = np.array([return_context_logits_data] *
                                         batch_size,
                                         dtype=bool)
        return_generation_logits = np.array([return_generation_logits_data] *
                                            batch_size,
                                            dtype=bool)

        assert len(return_context_logits.shape) == 2
        assert len(return_generation_logits.shape) == 2

        tensors.append(
            pb_utils.Tensor("return_context_logits", return_context_logits))
        tensors.append(
            pb_utils.Tensor("return_generation_logits",
                            return_generation_logits))
        return tensors

    def _get_llm_response(self, triton_output):
        name_map = {
            "output_ids": "output_ids",
            "sequence_length": "sequence_length",
            "cum_log_probs": "cum_log_probs",
            "output_log_probs": "output_log_probs",
            "context_logits": "context_logits",
            "generation_logits": "generation_logits",
            "batch_index": "batch_index",
            "sequence_index": "sequence_index",
            "kv_cache_alloc_new_blocks": "kv_cache_alloc_new_blocks",
            "kv_cache_reused_blocks": "kv_cache_reused_blocks",
            "kv_cache_alloc_total_blocks": "kv_cache_alloc_total_blocks",
            "arrival_time_ns": "arrival_time_ns",
            "first_scheduled_time_ns": "first_scheduled_time_ns",
            "first_token_time_ns": "first_token_time_ns",
            "last_token_time_ns": "last_token_time_ns",
            "acceptance_rate": "acceptance_rate",
            "total_accepted_draft_tokens": "total_accepted_draft_tokens",
            "total_draft_tokens": "total_draft_tokens"
        }
        return self.convert_triton_response(triton_output, GenerationResponse,
                                            name_map)

    def _postprocess(self, tokens: np.ndarray,
                     sequence_lengths: Optional[np.ndarray],
                     gen_response: GenerationResponse) -> Response:
        input_tensors = self._get_postproc_tensors(tokens, sequence_lengths,
                                                   gen_response)
        triton_req = pb_utils.InferenceRequest(
            model_name=self.postproc_model_name,
            inputs=input_tensors,
            requested_output_names=self._postproc_outputs)
        r = self._exec_triton_request_single(triton_req)
        response = self._get_response(r, gen_response)
        return response

    def _get_postproc_tensors(self, tokens: np.ndarray,
                              sequence_lengths: Optional[np.ndarray],
                              gen_response: GenerationResponse):
        tensors = [
            pb_utils.Tensor("TOKENS_BATCH", tokens),
            pb_utils.Tensor(
                "SEQUENCE_LENGTH", sequence_lengths
                if sequence_lengths else gen_response.sequence_length)
        ]
        return tensors

    def _get_response(self, triton_output, gen_res: GenerationResponse):
        tensors = triton_output.output_tensors()
        t_map = {}
        for named_t in tensors:
            name = named_t.name()
            t = named_t.as_numpy()
            t_map[name] = t
        response = Response(
            text_output=t_map["OUTPUT"],
            cum_log_probs=gen_res.cum_log_probs,
            output_log_probs=gen_res.output_log_probs,
            context_logits=gen_res.context_logits,
            generation_logits=gen_res.generation_logits,
            batch_index=gen_res.batch_index,
            sequence_index=gen_res.sequence_index,
            kv_cache_alloc_new_blocks=gen_res.kv_cache_alloc_new_blocks,
            kv_cache_reused_blocks=gen_res.kv_cache_reused_blocks,
            kv_cache_alloc_total_blocks=gen_res.kv_cache_alloc_total_blocks,
            arrival_time_ns=gen_res.arrival_time_ns,
            first_scheduled_time_ns=gen_res.first_scheduled_time_ns,
            first_token_time_ns=gen_res.first_token_time_ns,
            last_token_time_ns=gen_res.last_token_time_ns,
            acceptance_rate=gen_res.acceptance_rate,
            total_accepted_draft_tokens=gen_res.total_accepted_draft_tokens,
            total_draft_tokens=gen_res.total_draft_tokens)
        return response
