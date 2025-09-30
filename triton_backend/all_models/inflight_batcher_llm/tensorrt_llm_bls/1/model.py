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

import json
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

import triton_python_backend_utils as pb_utils
from lib.triton_decoder import TritonDecoder


def get_valid_param_value(param, default_value=''):
    value = param.get('string_value', '')
    return default_value if value.startswith('${') or value == '' else value


@dataclass
class StopWordsState:
    beam_indices: set[int] = field(default_factory=set)
    prefix: str = ""


class TritonPythonModel:

    def initialize(self, args):

        # Parse model configs
        model_config = json.loads(args['model_config'])

        params = model_config['parameters']

        accumulate_tokens_str = get_valid_param_value(
            params.get('accumulate_tokens', {}))
        self.accumulate_tokens = accumulate_tokens_str.lower() in [
            'true', 'yes', '1', 't'
        ]

        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config)

        self.logger = pb_utils.Logger

        default_tensorrt_llm_model_name = 'tensorrt_llm'
        self.llm_model_name = get_valid_param_value(
            params.get('tensorrt_llm_model_name', {}),
            default_tensorrt_llm_model_name)

        self.draft_llm_model_name = get_valid_param_value(
            params.get('tensorrt_llm_draft_model_name', {}), None)

        self.multimodal_encoders_name = get_valid_param_value(
            params.get('multimodal_encoders_name', {}), None)

        self.decoder = TritonDecoder(
            streaming=self.decoupled,
            accumulate=self.accumulate_tokens,
            preproc_model_name="preprocessing",
            postproc_model_name="postprocessing",
            llm_model_name=self.llm_model_name,
            draft_llm_model_name=self.draft_llm_model_name,
            multimodal_encoders_name=self.multimodal_encoders_name)

    def get_batch_index(self, response):
        if hasattr(
                response, 'batch_index'
        ) and response.batch_index is not None and response.batch_index.shape == (
                1, 1):
            return response.batch_index[0][0]
        else:
            return 0

    def get_sequence_index(self, response):
        if hasattr(
                response, 'sequence_index'
        ) and response.sequence_index is not None and response.sequence_index.shape == (
                1, 1):
            return response.sequence_index[0][0]
        else:
            return 0

    def check_stop_words(self, request, response, state):
        batch_index = self.get_batch_index(response)
        seq_index = self.get_sequence_index(response)
        if not (hasattr(request, 'stop_words')
                and request.stop_words is not None):
            return False

        text_input = str(request.text_input[batch_index][0], 'utf-8')
        is_streaming = hasattr(
            request,
            'stream') and request.stream and request.stream[batch_index][0]
        exclude_input_in_output = hasattr(
            request, 'exclude_input_in_output'
        ) and request.exclude_input_in_output and request.exclude_input_in_output[
            batch_index][0]
        if is_streaming:
            # For every beam in the response, check if the stop word is detected
            for j, text_output in enumerate(response.text_output):
                text_output = str(text_output, "utf-8")
                if (batch_index, seq_index, j) not in state:
                    state[(batch_index, seq_index, j)] = StopWordsState()

                response_state = state[(batch_index, seq_index, j)]
                # If stop word is already detected for a beam, skip it
                if j in response_state.beam_indices:
                    response.text_output[j] = b""
                    continue

                for i, stop_word in enumerate(request.stop_words[batch_index]):
                    stop_word = str(stop_word, encoding='utf-8')
                    if stop_word == "":
                        continue

                    if stop_word.startswith(response_state.prefix +
                                            text_output):
                        response_state.prefix += text_output
                        if stop_word == response_state.prefix:
                            response_state.beam_indices.add(j)
                            break
                    else:
                        response_state.prefix = ""
        else:
            for j, text_output in enumerate(response.text_output):
                if (batch_index, seq_index, j) not in state:
                    state[(batch_index, seq_index, j)] = StopWordsState()

                response_state = state[(batch_index, seq_index, j)]
                generation_start = 0
                text_output = str(text_output, "utf-8")
                if not exclude_input_in_output:
                    generation_start = len(text_input)
                for i, stop_word in enumerate(request.stop_words[batch_index]):
                    stop_word = str(stop_word, encoding='utf-8')
                    if stop_word == "":
                        continue
                    stop_word_index = text_output.find(stop_word,
                                                       generation_start)
                    if stop_word_index != -1:
                        response.text_output[
                            j] = text_output[:stop_word_index +
                                             len(stop_word)].encode('utf-8')
                        response_state.beam_indices.add(j)
                        break
        return len(response_state.beam_indices) == len(response.text_output)

    def execute(self, requests):

        responses = []

        for request in requests:
            if self.decoupled:
                response_sender = request.get_response_sender()
            try:

                req = self.decoder.convert_triton_request(request)
                req.validate()
                speculative_decode = (req.num_draft_tokens is not None
                                      and req.num_draft_tokens[0][0] > 0)
                if speculative_decode and (self.draft_llm_model_name is None
                                           or self.draft_llm_model_name == ""):
                    raise Exception(
                        "cannot perform speculative decoding without draft model"
                    )
                is_multimodal = req.image_input is not None or req.image_bytes_input is not None or req.image_url_input is not None or req.video_bytes_input is not None

                if speculative_decode and is_multimodal:
                    raise Exception(
                        "Multimodal and speculative decoding is not currently supported"
                    )
                req.request_id = str(uuid.uuid4())
                res_gen = self.decoder.decode(
                    req,
                    speculative_decoding=speculative_decode,
                    is_multimodal=is_multimodal)

                stopped_batch_seq_indices = defaultdict(set)
                stopped_word_status = defaultdict(StopWordsState)
                for res in res_gen:
                    batch_index = self.get_batch_index(res)
                    if batch_index in stopped_batch_seq_indices and self.get_sequence_index(
                            res) in stopped_batch_seq_indices[batch_index]:
                        continue

                    if self.check_stop_words(req, res, stopped_word_status):
                        stopped_batch_seq_indices[batch_index].add(
                            self.get_sequence_index(res))
                        if len(stopped_batch_seq_indices
                               ) == req.text_input.shape[0]:
                            self.decoder.send_cancellation_request(
                                req.request_id, self.decoupled)

                    triton_response = self.decoder.create_triton_response(res)
                    if self.decoupled:
                        response_sender.send(triton_response)
                    else:
                        responses.append(triton_response)

                if self.decoupled:
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

            except Exception:
                self.logger.log_error(traceback.format_exc())
                # If encountering an error, send a response with err msg
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(traceback.format_exc()))

                if self.decoupled:
                    response_sender.send(error_response)
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                else:
                    responses.append(error_response)

            self.decoder.reset_decoder()
        if self.decoupled:
            return None
        else:
            assert len(responses) == len(requests)
            return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
