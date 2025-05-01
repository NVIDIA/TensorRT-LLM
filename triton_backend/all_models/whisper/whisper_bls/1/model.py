# -*- coding: utf-8 -*-
import json
import re
import traceback

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack

from .fbank import FeatureExtractor
from .tokenizer import get_tokenizer


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = json.loads(args['model_config'])

        self.tokenizer = get_tokenizer(num_languages=100)
        self.eos = self.tokenizer.encode(
            "<|endoftext|>",
            allowed_special=self.tokenizer.special_tokens_set)[0]
        self.device = torch.device("cuda")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config)
        self.logger = pb_utils.Logger
        self.init_model(self.model_config['parameters'])

    def init_model(self, parameters):
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        n_mels = int(parameters["n_mels"])
        self.zero_pad = True if parameters["zero_pad"] == "true" else False
        self.feature_extractor = FeatureExtractor(n_mels=n_mels)

    def _prepare_inputs(self,
                        request,
                        mel_feature,
                        mel_len,
                        prompt,
                        max_tokens=50):
        """
        Prepares inputs for the language model based on the parameters in the
        request, image features, and prompt. It tokenizes prompt,
        extracts and processes additional parameters from the request:
            - max_tokens: Maximum number of tokens to generate (default: 50)
            - temperature: Controls randomness in generation (default: 0.5)
            - top_k: Top K sampling parameter (default: 1)
            - frequency_penalty: Penalizes frequent tokens (default: 0.7)
            - seed: Random seed for generation (default: 10)

        Final llm input dictionary is combined out of all processed parameters,
        prompt's tokens and image features. The latter will be passed to llm
        through `prompt_embedding_table`.

        Parameters
        ----------
        - request: The original request object containing additional parameters.
        - image_features (list): A list containing image feature tensors.
        - prompt (str): The text prompt to be processed.

        Returns
        -------
        - dict: A dictionary containing all the prepared inputs for the language model.
        """
        input_dict = {
            "request_output_len": np.array([[max_tokens]], dtype=np.int32),
            "end_id": np.array([[self.eos]], dtype=np.int32),
            "pad_id": np.array([[self.eos]], dtype=np.int32),
            "encoder_output_lengths": mel_len // 2,
            "input_lengths": mel_len,
            "decoder_input_ids": prompt,
            "streaming": np.array([[self.decoupled]], dtype=np.bool_),
        }
        input_tensor_list = [
            pb_utils.Tensor(k, v) for k, v in input_dict.items()
        ]
        input_tensor_list.append(
            pb_utils.Tensor.from_dlpack("encoder_input_features",
                                        to_dlpack(mel_feature.contiguous())))

        return input_tensor_list

    def _prepare_llm_response(self, llm_request_inputs):
        """
        Prepares the response from the language model based on the provided
        inputs. Creates a `pb_utils.InferenceRequest` object with passed
        `llm_request_inputs` to send to a decoupled TensorRTLLM model.
        For each response from the language model:
            - Checks for errors and raise an exception if any are found.
            - Extracts the "output_ids" tensor from the response.
            - Determines the finish reason based on the presence of the
              end-of-sequence token or reaching the maximum length.
            - Appends the generated token IDs to `output_ids`.
            - If the finish reason is determined, decodes the output IDs to text
              and prepares the final response.

        The final response includes the generated text, finish reason,
        completion tokens, prompt tokens, and total tokens.

        Parameters
        ----------
        - llm_request_inputs (dict): A dictionary containing the inputs for the language model.

        Returns
        -------
        - pb_utils.InferenceResponse: The response object containing the generated text and additional metadata.
        """
        llm_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length"],
            inputs=llm_request_inputs,
        )
        responses = llm_request.exec(decoupled=self.decoupled)
        if not self.decoupled:
            llm_response = responses
            if llm_response.has_error():
                raise pb_utils.TritonModelException(
                    llm_response.error().message())
            output_ids = (pb_utils.get_output_tensor_by_name(
                llm_response, "output_ids").as_numpy().flatten().tolist())

            output_text = self.tokenizer.decode(output_ids).strip()
            output_text = re.sub(r'<\|.*?\|>', '', output_text)
            response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("TRANSCRIPTS",
                                np.array([output_text], np.object_)),
            ])
            yield response
        else:
            output_ids = []
            for llm_response in responses:
                if llm_response.has_error():
                    raise pb_utils.TritonModelException(
                        llm_response.error().message())
                stream_output_ids = (pb_utils.get_output_tensor_by_name(
                    llm_response, "output_ids").as_numpy().flatten().tolist())
                if len(stream_output_ids) == 0:
                    continue
                output_ids.extend(stream_output_ids)
                output_text = self.tokenizer.decode(output_ids).strip()
                output_text = re.sub(r'<\|.*?\|>', '', output_text)
                response = pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("TRANSCRIPTS",
                                    np.array([output_text], np.object_)),
                ])
                yield response

    def execute(self, requests):

        responses = []

        for request in requests:
            # Perform inference on the request and append it to responses list...
            decoder_text_prompt = pb_utils.get_input_tensor_by_name(
                request, "TEXT_PREFIX").as_numpy().tolist()
            text_prefix = decoder_text_prompt[0][0].decode('utf-8')
            if text_prefix == "":
                text_prefix = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
            prompt_id = self.tokenizer.encode(
                text_prefix, allowed_special=self.tokenizer.special_tokens_set)
            decoder_input_ids = np.array([prompt_id], dtype=np.int32)

            wav = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
            assert wav.shape[0] == 1, "Only support batch size 1"
            # To support batch > 1
            # cat mel,text_prompt, also, need to increase decoder_input_len as a triton input
            wav = torch.from_numpy(wav[0]).to(self.device)
            wav_len = pb_utils.get_input_tensor_by_name(
                request, "WAV_LENS").as_numpy().item()
            if self.zero_pad:
                wav = wav[:wav_len]
                target = 0
            else:
                target = 3000
            mel = self.feature_extractor.compute_feature(wav, target).transpose(
                1, 2)
            mel_len = np.array([[mel.shape[1]]], dtype=np.int32)
            if self.decoupled:
                response_sender = request.get_response_sender()
            try:

                llm_request_inputs = self._prepare_inputs(
                    request, mel, mel_len, decoder_input_ids)
                if isinstance(llm_request_inputs, pb_utils.TritonError):
                    error = pb_utils.InferenceResponse(error=llm_request_inputs)
                    if self.decoupled:
                        response_sender.send(
                            error,
                            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                    else:
                        responses.append(error)
                llm_responses = self._prepare_llm_response(llm_request_inputs)

                for triton_response in llm_responses:
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

        if self.decoupled:
            return None
        else:
            assert len(responses) == len(requests)
            return responses
