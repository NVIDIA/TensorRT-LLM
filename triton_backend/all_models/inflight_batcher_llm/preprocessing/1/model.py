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

import base64
import io
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import requests
import triton_python_backend_utils as pb_utils
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, T5Tokenizer


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
        # Parse model configs
        model_config = json.loads(args['model_config'])
        tokenizer_dir = model_config['parameters']['tokenizer_dir'][
            'string_value']

        add_special_tokens = model_config['parameters'].get(
            'add_special_tokens')
        multimodal_model_path = model_config['parameters'][
            'multimodal_model_path']['string_value']
        max_num_images = model_config['parameters'].get('max_num_images')

        if max_num_images is not None:
            max_num_images_str = max_num_images['string_value']
            if max_num_images_str.isdigit():
                self.max_num_images = int(max_num_images_str)
            else:
                print(
                    f"[TensorRT-LLM][WARNING] 'max_num_images' parameter is not set correctly (value is {max_num_images_str}). Will be set to None"
                )
                self.max_num_images = None
        else:
            print(
                f"[TensorRT-LLM][WARNING] Don't setup 'max_num_images'. Set it as None by default."
            )
            self.max_num_images = None
        if multimodal_model_path == "${multimodal_model_path}" or multimodal_model_path == "":
            multimodal_model_path = None

        if add_special_tokens is not None:
            add_special_tokens_str = add_special_tokens['string_value'].lower()
            if add_special_tokens_str in [
                    'true', 'false', '1', '0', 't', 'f', 'y', 'n', 'yes', 'no'
            ]:
                self.add_special_tokens = add_special_tokens_str in [
                    'true', '1', 't', 'y', 'yes'
                ]
            else:
                print(
                    f"[TensorRT-LLM][WARNING] Don't setup 'add_special_tokens' correctly (set value is {add_special_tokens['string_value']}). Set it as True by default."
                )
                self.add_special_tokens = True
        else:
            print(
                f"[TensorRT-LLM][WARNING] Don't setup 'add_special_tokens'. Set it as True by default."
            )
            self.add_special_tokens = True

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                       legacy=False,
                                                       padding_side='left',
                                                       trust_remote_code=True)

        if isinstance(self.tokenizer, T5Tokenizer):
            self.tokenizer_bos_id = self.tokenizer.sp_model.bos_id()

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer_end_id = self.tokenizer.encode(
            self.tokenizer.eos_token, add_special_tokens=False)[0]
        self.tokenizer_pad_id = self.tokenizer.encode(
            self.tokenizer.pad_token, add_special_tokens=False)[0]
        self.vocab_size = self.tokenizer.vocab_size

        self.is_multimodal = False
        self.model_type = None
        self.vision_preprocessor = None

        if multimodal_model_path is not None:
            self.is_multimodal = True
            multimodal_model_path = os.path.join(multimodal_model_path,
                                                 'config.json')
            with open(multimodal_model_path, 'r') as f:
                visual_model_config = json.load(f)
            self.model_type = visual_model_config['builder_config'][
                'model_type']

            assert self.model_type in [
                'llava', 'blip2-opt', 'pixtral', 'vila', 'mllama',
                'llava_onevision', 'qwen2_vl'
            ], f"[TensorRT-LLM][ERROR] Currently supported multi-modal models are llava, blip2-opt, pixtral, vila, mllama, llava_onevision and qwen2_vl. Got {self.model_type}."

            assert self.model_type != 'llava_onevison' or self.max_num_images is None or self.max_num_images <= 1, f"LLaVA-OneVsion is not support multi image inference currently."

            llm_model_path = model_config['parameters']['gpt_model_path'][
                'string_value']
            llm_model_path = os.path.join(llm_model_path, 'config.json')
            with open(llm_model_path, 'r') as f:
                llm_model_config = json.load(f)
            self.vocab_size = int(
                llm_model_config["pretrained_config"]["vocab_size"])
            self._setup_ptable_shape(llm_model_config)

            if self.model_type in [
                    'mllama', 'llava_onevision', 'qwen2_vl', 'pixtral'
            ]:
                full_processor = AutoProcessor.from_pretrained(
                    tokenizer_dir, trust_remote_code=True)
                self.hf_config = AutoConfig.from_pretrained(tokenizer_dir)
                self.vision_preprocessor = VisionPreProcessor(
                    self.model_type,
                    full_processor,
                    model_config,
                    self.hf_config,
                )

        # Parse model output configs and convert Triton types to numpy types
        output_names = [
            "INPUT_ID", "DECODER_INPUT_ID", "REQUEST_INPUT_LEN",
            "REQUEST_DECODER_INPUT_LEN", "BAD_WORDS_IDS", "STOP_WORDS_IDS",
            "OUT_END_ID", "OUT_PAD_ID", "OUT_PROMPT_TABLE_EXTRA_IDS",
            "PIXEL_VALUES", "IMAGE_SIZES"
        ]
        input_names = ["EMBEDDING_BIAS_WORDS", "EMBEDDING_BIAS_WEIGHTS"]
        for input_name in input_names:
            setattr(
                self,
                input_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_input_config_by_name(model_config,
                                                      input_name)['data_type']))

        for output_name in output_names:
            setattr(
                self,
                output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(
                        model_config, output_name)['data_type']))

    def _setup_ptable_shape(self, llm_model_config):
        max_prompt_embedding_table_size = llm_model_config['build_config'][
            'max_prompt_embedding_table_size']
        max_batch_size = llm_model_config['build_config']['max_batch_size']

        num_multimodal_features = max_prompt_embedding_table_size // max_batch_size
        hidden_size = llm_model_config['pretrained_config']['hidden_size']
        if self.max_num_images is not None:
            num_multimodal_features = num_multimodal_features // self.max_num_images

        self.ptable_shape = (-1, num_multimodal_features, hidden_size)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors
            query = pb_utils.get_input_tensor_by_name(request,
                                                      'QUERY').as_numpy()
            batch_size = query.shape[0]

            decoder_query = pb_utils.get_input_tensor_by_name(
                request, 'DECODER_QUERY')
            if decoder_query is not None:
                decoder_query = decoder_query.as_numpy()

            request_output_len = pb_utils.get_input_tensor_by_name(
                request, 'REQUEST_OUTPUT_LEN').as_numpy()

            bad_words_dict = pb_utils.get_input_tensor_by_name(
                request, 'BAD_WORDS_DICT')
            if bad_words_dict is not None:
                bad_words_dict = bad_words_dict.as_numpy()

            stop_words_dict = pb_utils.get_input_tensor_by_name(
                request, 'STOP_WORDS_DICT')
            if stop_words_dict is not None:
                stop_words_dict = stop_words_dict.as_numpy()

            embedding_bias_words = pb_utils.get_input_tensor_by_name(
                request, 'EMBEDDING_BIAS_WORDS')
            if embedding_bias_words is not None:
                embedding_bias_words = embedding_bias_words.as_numpy()

            embedding_bias_weights = pb_utils.get_input_tensor_by_name(
                request, 'EMBEDDING_BIAS_WEIGHTS')
            if embedding_bias_weights is not None:
                embedding_bias_weights = embedding_bias_weights.as_numpy()

            # Take the end_id from the input tensors
            # If not specified, use tokenizer to get end_id
            end_id = pb_utils.get_input_tensor_by_name(request, 'END_ID')
            if end_id is not None:
                end_id = end_id.as_numpy()
            else:
                end_id = [[self.tokenizer_end_id]] * batch_size

            # Take the pad_id from the input tensors
            # If not specified, use tokenizer to get pad_id
            pad_id = pb_utils.get_input_tensor_by_name(request, 'PAD_ID')
            if pad_id is not None:
                pad_id = pad_id.as_numpy()
            else:
                pad_id = [[self.tokenizer_pad_id]] * batch_size

            # Take the extra_id from the input tensors
            # Extra id is used in kv cache reuse for p-tuning
            prompt_table_extra_id = pb_utils.get_input_tensor_by_name(
                request, 'PROMPT_TABLE_EXTRA_ID')
            if prompt_table_extra_id is not None:
                prompt_table_extra_id = prompt_table_extra_id.as_numpy()
                assert prompt_table_extra_id.shape[
                    0] == batch_size, "Prompt table extra id must have the same batch size as Query"
                assert prompt_table_extra_id.shape[
                    1] == 1, "Multiple IDs cannot be provided for a single image"

            # Preprocessing vision input passed as a url or bytes tensor
            img_urls = pb_utils.get_input_tensor_by_name(request, 'IMAGE_URL')
            image_bytes = pb_utils.get_input_tensor_by_name(
                request, 'IMAGE_BYTES')
            video_bytes = pb_utils.get_input_tensor_by_name(
                request, 'VIDEO_BYTES')
            vision_processed_tensors = []
            visual_tokens = []
            # Pixtral supports text-only input
            if self.is_multimodal and (img_urls or image_bytes or video_bytes
                                       or self.model_type == 'pixtral'):
                assert self.vision_preprocessor != None, "Vision preprocessor for preparing images before encoding is None"
                processed_tensors = {}
                if self.model_type == 'mllama':
                    processed_tensors = self.vision_preprocessor.mllama_process(
                        queries=query.astype(str).tolist(),
                        img_urls=img_urls,
                        image_bytes=image_bytes,
                    )
                elif self.model_type == 'llava_onevision':
                    if video_bytes is None:
                        processed_tensors, visual_tokens = self.vision_preprocessor.llava_onevision_process_image(
                            queries=query.astype(str).tolist(),
                            img_urls=img_urls,
                            image_bytes=image_bytes,
                        )
                    else:
                        processed_tensors, visual_tokens = self.vision_preprocessor.llava_onevision_process_video(
                            queries=query.astype(str).tolist(),
                            video_bytes=video_bytes,
                        )
                elif self.model_type == 'qwen2_vl':
                    processed_tensors = self.vision_preprocessor.qwen2_vl_process_image(
                        queries=query.astype(str).tolist(),
                        img_urls=img_urls,
                        image_bytes=image_bytes,
                    )
                    qwen2vl_input_id_tensor = processed_tensors.get("INPUT_IDS")
                    processed_tensors.pop("INPUT_IDS")
                    qwen2vl_input_length_tensor = processed_tensors.get(
                        "REQUEST_INPUT_LEN")
                    processed_tensors.pop("REQUEST_INPUT_LEN")
                elif self.model_type == 'pixtral':
                    image_sizes = pb_utils.get_input_tensor_by_name(
                        request, 'IMAGE_SIZES')
                    processed_tensors, visual_tokens = self.vision_preprocessor.pixtral_process(
                        queries=query.astype(str).tolist(),
                        img_urls=img_urls,
                        image_bytes=image_bytes,
                        image_sizes=image_sizes,
                    )
                    pixtral_input_id_tensor = processed_tensors.pop("INPUT_IDS")
                    request_input_len = np.array(
                        [[len(input_ids_for_batch)]
                         for input_ids_for_batch in pixtral_input_id_tensor])
                else:
                    raise ValueError(
                        "Unsupported model type for IMAGE_BYTES or IMAGE_URL inputs"
                    )
                vision_processed_tensors = [
                    pb_utils.Tensor.from_dlpack(k, v)
                    for k, v in processed_tensors.items()
                ]
            else:
                assert self.model_type != "llava_onevision", "Image processing requires IMAGE_BYTES or IMAGE_URL to be provided"

            # Preprocessing input data.
            # For the LLaVA_OneVision model, num_multimodal_features is not a fixed value
            if self.model_type != 'pixtral':
                input_id, request_input_len = self._create_request(
                    query, visual_tokens)
            if decoder_query is not None:
                decoder_input_id, request_decoder_input_len = self._create_request(
                    decoder_query)
            else:
                decoder_input_id = pad_id * np.ones((batch_size, 1), np.int32)
                request_decoder_input_len = 1 * np.ones(
                    (batch_size, 1), np.int32)

            bad_words = self._to_word_list_format(bad_words_dict, batch_size)
            stop_words = self._to_word_list_format(stop_words_dict, batch_size)

            embedding_bias = self._get_embedding_bias(
                embedding_bias_words, embedding_bias_weights,
                self.embedding_bias_weights_dtype, batch_size)

            if prompt_table_extra_id is not None and self.model_type != 'qwen2_vl':
                prompt_table_extra_ids = np.zeros_like(input_id)
                for i in range(batch_size):
                    prompt_table_extra_ids[i] = np.where(
                        input_id[i] >= self.vocab_size,
                        prompt_table_extra_id[i], 0)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            # Qwen2-VL model has special logic to process input ids
            if self.model_type == 'qwen2_vl':
                input_id_tensor = pb_utils.Tensor.from_dlpack(
                    'INPUT_ID', qwen2vl_input_id_tensor)
                request_input_len_tensor = pb_utils.Tensor.from_dlpack(
                    'REQUEST_INPUT_LEN', qwen2vl_input_length_tensor)
            elif self.model_type == 'pixtral':
                input_id_tensor = pb_utils.Tensor(
                    'INPUT_ID',
                    pixtral_input_id_tensor.numpy().astype(self.input_id_dtype))
                request_input_len_tensor = pb_utils.Tensor(
                    'REQUEST_INPUT_LEN',
                    request_input_len.astype(self.request_input_len_dtype))
            else:
                input_id_tensor = pb_utils.Tensor(
                    'INPUT_ID', input_id.astype(self.input_id_dtype))
                request_input_len_tensor = pb_utils.Tensor(
                    'REQUEST_INPUT_LEN',
                    request_input_len.astype(self.request_input_len_dtype))
            decoder_input_id_tensor = pb_utils.Tensor(
                'DECODER_INPUT_ID',
                decoder_input_id.astype(self.decoder_input_id_dtype))
            request_decoder_input_len_tensor = pb_utils.Tensor(
                'REQUEST_DECODER_INPUT_LEN',
                request_decoder_input_len.astype(
                    self.request_decoder_input_len_dtype))
            request_output_len_tensor = pb_utils.Tensor('REQUEST_OUTPUT_LEN',
                                                        request_output_len)
            bad_words_ids_tensor = pb_utils.Tensor('BAD_WORDS_IDS', bad_words)
            stop_words_ids_tensor = pb_utils.Tensor('STOP_WORDS_IDS',
                                                    stop_words)
            embedding_bias_tensor = pb_utils.Tensor('EMBEDDING_BIAS',
                                                    embedding_bias)
            end_id_tensor = pb_utils.Tensor('OUT_END_ID',
                                            np.array(end_id, dtype=np.int32))
            pad_id_tensor = pb_utils.Tensor('OUT_PAD_ID',
                                            np.array(pad_id, dtype=np.int32))
            if prompt_table_extra_id is not None:
                prompt_table_extra_ids_tensor = pb_utils.Tensor(
                    'OUT_PROMPT_TABLE_EXTRA_IDS',
                    np.array(prompt_table_extra_ids,
                             dtype=self.out_prompt_table_extra_ids_dtype))
                inference_response = pb_utils.InferenceResponse(output_tensors=[
                    input_id_tensor, decoder_input_id_tensor,
                    bad_words_ids_tensor, stop_words_ids_tensor,
                    request_input_len_tensor, request_decoder_input_len_tensor,
                    request_output_len_tensor, embedding_bias_tensor,
                    end_id_tensor, pad_id_tensor, prompt_table_extra_ids_tensor
                ] + vision_processed_tensors)
            else:
                inference_response = pb_utils.InferenceResponse(output_tensors=[
                    input_id_tensor, decoder_input_id_tensor,
                    bad_words_ids_tensor, stop_words_ids_tensor,
                    request_input_len_tensor, request_decoder_input_len_tensor,
                    request_output_len_tensor, embedding_bias_tensor,
                    end_id_tensor, pad_id_tensor
                ] + vision_processed_tensors)
            responses.append(inference_response)
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    def _split_prompt_by_images(self, concatenated_ids, image_token_index=-200):
        """
        Splits tokenized prompts by image placeholders for each sample in the batch.

        Args:
            concatenated_ids (np.ndarray): A batch of concatenated token IDs, where image placeholders are indicated by `image_token_index`.

        Returns:
            List[List[np.ndarray]]: A list containing lists of token ID arrays for each prompt segment, per batch sample.
        """
        batch_splits = []
        for batch in concatenated_ids:
            zero_indices = np.where(batch == image_token_index)[0]
            start_idx = 0
            splits = []
            for idx in zero_indices:
                if start_idx != idx:
                    splits.append(batch[start_idx:idx].reshape(1, -1))
                start_idx = idx + 1
            if start_idx < len(batch):
                splits.append(batch[start_idx:].reshape(1, -1))

            splits = [split for split in splits if split.size > 0]
            batch_splits.append(splits)

        return batch_splits

    def _setup_fake_prompts(self, batch_size, batch_split_prompts):
        """
        Replaces image placeholders with unique fake prompt IDs for multi-image inputs.

        Args:
            batch_size (int): The number of samples in the batch.
            batch_split_prompts (List[List[np.ndarray]]): Tokenized prompt segments for each batch sample.

        Returns:
            np.ndarray: An array of input IDs with image placeholders replaced by fake prompt IDs.
        """

        num_multimodal_features = self.ptable_shape[1]
        input_ids_list = []

        for batch_idx in range(batch_size):
            splits = batch_split_prompts[batch_idx]
            sample_input_ids = [splits[0]]
            sample_fake_prompt_counter = self.vocab_size

            for split_idx in range(len(splits) - 1):
                fake_prompt_id = np.arange(
                    sample_fake_prompt_counter,
                    sample_fake_prompt_counter + num_multimodal_features)
                sample_fake_prompt_counter += num_multimodal_features
                fake_prompt_id = np.expand_dims(fake_prompt_id, axis=0)
                sample_input_ids.append(fake_prompt_id)
                sample_input_ids.append(splits[split_idx + 1])

            sample_input_ids = np.concatenate(sample_input_ids, axis=1)
            input_ids_list.append(sample_input_ids)

        # Pad the input_ids to the same length for bs > 1
        max_seq_len = max(
            [sample_input_ids.shape[1] for sample_input_ids in input_ids_list])
        input_ids_padded = []
        for sample_input_ids in input_ids_list:
            seq_len = sample_input_ids.shape[1]
            pad_width = max_seq_len - seq_len
            if pad_width > 0:
                sample_input_ids_padded = np.pad(
                    sample_input_ids, ((0, 0), (0, pad_width)),
                    'constant',
                    constant_values=self.tokenizer_pad_id)
            else:
                sample_input_ids_padded = sample_input_ids
            input_ids_padded.append(sample_input_ids_padded)

        input_ids = np.stack(input_ids_padded)
        input_ids = input_ids.reshape(batch_size, -1).astype(np.int32)

        return input_ids

    def _process_multi_image_inputs(self, query, image_token_index=-200):
        """
        Processes input queries that contain multiple images by tokenizing the input strings and inserting image_token_index between the parts.

        Args:
            query (np.ndarray): Batch of input strings.

        Returns:
            List[np.ndarray]: List of tokenized input IDs for each sample.
        """
        start_ids = []
        for s in query:
            parts = s[0].decode().split('<image>')
            num_images = len(parts) - 1
            if num_images > self.max_num_images:
                raise ValueError(
                    f"The number of images in the request ({num_images}) exceeds the maximum allowed ({self.max_num_images})."
                )
            tokenized_parts = [
                self.tokenizer.encode(part, add_special_tokens=False)
                for part in parts
            ]

            # Insert `image_token_index` between the parts to represent <image>
            final_ids = []
            for i, part in enumerate(tokenized_parts):
                final_ids.extend(part)
                if i < len(tokenized_parts) - 1:
                    final_ids.append(image_token_index)

            start_ids.append(np.array(final_ids).astype(int))

        return start_ids

    def _create_request(self, query, visual_tokens=None):
        """
            query : batch string (2D numpy array)
        """
        if isinstance(self.tokenizer, T5Tokenizer):
            start_ids = [
                np.array([self.tokenizer_bos_id] + self.tokenizer.encode(
                    s[0].decode(), add_special_tokens=self.add_special_tokens)).
                astype(int) for s in query
            ]
        else:
            # Qwen2-VL input id is calculated when processing image
            if 'qwen2_vl' == self.model_type:
                return None, None
            if self.is_multimodal and self.max_num_images and self.max_num_images > 1:
                start_ids = self._process_multi_image_inputs(query)

            else:
                start_ids = [
                    np.array(
                        self.tokenizer.encode(
                            s[0].decode(),
                            add_special_tokens=self.add_special_tokens)).astype(
                                int) for s in query
                ]

        if self.is_multimodal:
            if 'blip2' in self.model_type or 'mllama' == self.model_type:
                pre_prompt = None
                post_prompt = None
            elif 'llava' == self.model_type:
                pre_prompt = "USER:\n"
                post_prompt = " ASSISTANT:"
            elif 'vila' == self.model_type:
                pre_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
                post_prompt = " ASSISTANT:"
            elif 'llava_onevision' == self.model_type:
                pre_prompt = "<|im_start|>user "
                post_prompt = "<|im_end|><|im_start|>assistant\n"
            pre_prompt_id = np.array(
                self.tokenizer.encode(
                    pre_prompt,
                    add_special_tokens=self.add_special_tokens,
                    padding=True)) if pre_prompt is not None else np.array(
                        [], dtype=int)

            post_prompt_id = np.array(
                self.tokenizer.encode(
                    post_prompt,
                    add_special_tokens=self.add_special_tokens,
                    padding=True)) if post_prompt is not None else np.array(
                        [], dtype=int)

            if self.max_num_images and self.max_num_images > 1:
                concatenated_ids = [
                    np.concatenate((pre_prompt_id, ids, post_prompt_id), axis=0)
                    for ids in start_ids
                ]
                batch_split_prompts = self._split_prompt_by_images(
                    concatenated_ids)
                start_ids = self._setup_fake_prompts(query.shape[0],
                                                     batch_split_prompts)
            elif self.model_type == 'llava_onevision':
                fake_prompt_ids = []
                extra_id = np.array(
                    self.tokenizer.encode(
                        '\n',
                        add_special_tokens=self.add_special_tokens,
                        padding=True))
                for tokens in visual_tokens:
                    prompt_id = np.arange(self.vocab_size,
                                          self.vocab_size + tokens)
                    fake_prompt_ids.append(prompt_id)
                start_ids = [
                    np.concatenate((pre_prompt_id, prompt_id, extra_id, ids,
                                    post_prompt_id),
                                   axis=0)
                    for prompt_id, ids in zip(fake_prompt_ids, start_ids)
                ]
            else:
                fake_prompt_id = np.arange(
                    self.vocab_size, self.vocab_size + self.ptable_shape[1])
                start_ids = [
                    np.concatenate(
                        (pre_prompt_id, fake_prompt_id, ids, post_prompt_id),
                        axis=0) for ids in start_ids
                ]

        start_lengths = np.array([[len(ids)] for ids in start_ids]).astype(int)

        max_len = 0
        for seq in start_ids:
            max_len = max(max_len, seq.shape[0])
        start_ids = np.stack([
            np.pad(seq, (0, max_len - seq.shape[0]),
                   'constant',
                   constant_values=(0, self.tokenizer_pad_id))
            for seq in start_ids
        ])

        return start_ids, start_lengths

    def _to_word_list_format(self, word_lists: List[List[str | bytes]],
                             batch_size):
        '''
        word_lists format:
            len(word_lists) == batch_size
            word_lists[i] means the words associated to batch item i. A "word" may actually be any string. Like "lorem" or "lorem ipsum".
        '''
        assert self.tokenizer != None, "need to set tokenizer"

        if word_lists is None:
            # Return an empty array of shape (1,2,0)
            return np.empty([batch_size, 2, 0], dtype="int32")

        flat_ids = []
        offsets = []
        for word_list in word_lists:
            item_flat_ids = []
            item_offsets = []

            for word in word_list:
                if isinstance(word, bytes):
                    word = word.decode()

                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if len(ids) == 0:
                    continue

                item_flat_ids += ids
                item_offsets.append(len(ids))

            flat_ids.append(np.array(item_flat_ids))
            offsets.append(np.cumsum(np.array(item_offsets)))

        pad_to = max(1, max(len(ids) for ids in flat_ids))

        for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
            flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
            offsets[i] = np.pad(offs, (0, pad_to - len(offs)),
                                constant_values=-1)

        return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))

    def _get_embedding_bias(self, embedding_bias_words, embedding_bias_weights,
                            bias_dtype, batch_size):

        assert self.tokenizer != None, "need to set tokenizer"

        if embedding_bias_words is None or embedding_bias_weights is None:
            return np.empty([batch_size, 0],
                            dtype=self.embedding_bias_weights_dtype)

        batch_embedding_bias = []
        for words, weights in zip(embedding_bias_words, embedding_bias_weights):

            vocab_size = len(self.tokenizer.vocab)
            embedding_bias = [0.] * vocab_size

            assert len(words) == len(
                weights
            ), "Embedding bias words must have same dimension as embedding bias weights"

            for word, weight in zip(words, weights):
                if isinstance(word, bytes):
                    word = word.decode()
                ids = self.tokenizer.encode(word)

                if len(ids) == 0:
                    continue

                for id in ids:
                    embedding_bias[id] += weight

            batch_embedding_bias.append(np.array(embedding_bias))

        return np.array(batch_embedding_bias, dtype=bias_dtype)


class VisionPreProcessor:
    """ A class that can load images from url requests, and process them via a vision model processor,
    in preparation for the vision encoder.
    """

    def __init__(self,
                 vision_model_type,
                 vision_model_processor,
                 preprocessor_model_config=None,
                 hf_config=None):
        preprocessor_model_config = preprocessor_model_config or {}

        # import libraries that are only relevant for multimodal models
        import torch
        from torch.utils.dlpack import from_dlpack

        # NOTE: Due to the behavior of MPI initialization, it is recommended to avoid using import tensorrt_llm
        #       except for the specific modules tensorrt_llm and multimodal_encoders.
        #       As a result, the function str_dtype_to_torch has been copied directly from tensorrt_llm._utils.
        _str_to_torch_dtype_dict = dict(
            bfloat16=torch.bfloat16,
            float16=torch.float16,
            float32=torch.float32,
            int64=torch.int64,
            int32=torch.int32,
            int8=torch.int8,
            bool=torch.bool,
            fp8=torch.float8_e4m3fn,
        )

        def str_dtype_to_torch(dtype):
            ret = _str_to_torch_dtype_dict.get(dtype)
            assert ret is not None, f'Unsupported dtype: {dtype}'
            return ret

        self.load_images_tensor = lambda tensor: tensor if not hasattr(
            tensor, 'to_dlpack') else from_dlpack(tensor.to_dlpack())

        # extract expected output tensor dtype
        self.output_str_dtypes = {}
        for properties in preprocessor_model_config.get('output', []):
            dtype = properties['data_type']
            self.output_str_dtypes[properties['name']] = np.dtype(
                pb_utils.triton_string_to_numpy(dtype)).name

        # create method for converting output tensors batch to the expected type
        self.convert_tensor_list_to_tensor = lambda tensor_list: torch.concat(
            [
                torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                for x in tensor_list
            ],
            dim=0)
        self.convert_tensor_to_str_dtype = lambda tensor, dtype: tensor.to(
            str_dtype_to_torch(dtype))

        # create model-specific processor
        self.vision_model_processor = vision_model_processor
        self.vision_model_type = vision_model_type

        if vision_model_type == 'pixtral':
            assert hf_config is not None, "Pixtral model requires hf_config to be set"
            self.vocab_size = hf_config.text_config.vocab_size
            self.image_size = hf_config.vision_config.image_size
            self.image_token_index = hf_config.image_token_index

    def load_images_from_urls(self, img_urls):
        images = []
        for img_url in img_urls:
            img_url = img_url.decode()
            if img_url.startswith("data:image/jpeg;base64,"):
                image_base64 = img_url.split(",")[1]
                # Decode the base64 string
                image_data = base64.b64decode(image_base64)
                # Create a BytesIO object from the decoded data
                image_buffer = io.BytesIO(image_data)
                images.append(Image.open(image_buffer).convert("RGB"))
            else:
                images.append(
                    Image.open(requests.get(img_url,
                                            stream=True).raw).convert("RGB"))
        return images

    def mllama_process(self, queries, img_urls=None, image_bytes=None):
        vision_processed_tensors = {}
        if img_urls is not None or image_bytes is not None:
            if img_urls is not None:
                # download and read images
                images = [
                    self.load_images_from_urls(urls)
                    for urls in img_urls.as_numpy()
                ]
            else:
                images = [
                    img for img_list in self.load_images_tensor(image_bytes)
                    for img in img_list
                ]

            batch_size = len(images)

            preprocessor_outputs = {}
            possible_output_names = [
                'PIXEL_VALUES', 'ASPECT_RATIO_IDS', 'ASPECT_RATIO_MASK',
                'CROSS_ATTENTION_MASK'
            ]
            for batch_id in range(batch_size):
                # Preprocess images and query
                processed_vision_data = self.vision_model_processor(
                    images=images[batch_id],
                    text=queries[batch_id],
                    return_tensors="pt")
                # Reshape pixel_values to [num_images, *HWC/CHW]
                val = processed_vision_data["pixel_values"]
                val = val.reshape(1, -1, *(val.shape[-3:]))
                processed_vision_data["pixel_values"] = val

                # Create vision output tensors
                for key in possible_output_names:
                    val = processed_vision_data.get(key.lower())
                    if val is not None:
                        if key not in preprocessor_outputs:
                            preprocessor_outputs[key] = []
                        preprocessor_outputs[key].append(val)

            for key, tensor_list in preprocessor_outputs.items():
                val = self.convert_tensor_list_to_tensor(tensor_list)
                if key in self.output_str_dtypes:
                    val = self.convert_tensor_to_str_dtype(
                        val, self.output_str_dtypes[key])
                vision_processed_tensors[key] = val
        return vision_processed_tensors

    def llava_onevision_process_image(self,
                                      queries,
                                      img_urls=None,
                                      image_bytes=None):

        import torch
        vision_processed_tensors = {}
        if img_urls is not None:
            # download and read images
            images = [
                self.load_images_from_urls(urls)
                for urls in img_urls.as_numpy()
            ]
        else:
            images = [
                img for img_list in self.load_images_tensor(image_bytes)
                for img in img_list
            ]

        batch_size = len(images)
        assert len(
            queries
        ) == batch_size, f"Image must have the same batch size as Query."
        preprocessor_outputs = {}
        possible_output_names = ['PIXEL_VALUES', 'IMAGE_SIZES']
        visual_tokens = []
        for batch_id in range(batch_size):
            # Preprocess images and query
            processed_vision_data = self.vision_model_processor(
                images=images[batch_id], text='<image>', return_tensors="pt")
            visual_tokens.append(processed_vision_data['input_ids'].shape[1])
            # Create vision output tensors
            for key in possible_output_names:
                val = processed_vision_data.get(key.lower())
                if val is not None:
                    if key not in preprocessor_outputs:
                        preprocessor_outputs[key] = []
                    preprocessor_outputs[key].append(val)

        max_patch = max(x.shape[1]
                        for x in preprocessor_outputs['PIXEL_VALUES'])
        preprocessor_outputs['PIXEL_VALUES'] = [
            torch.nn.functional.pad(
                image, (0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[1], 0, 0),
                mode='constant')
            for image in preprocessor_outputs['PIXEL_VALUES']
        ]
        # Add a dimension image_sizes to match the dimensions defined in config.pbtxt
        for elem in preprocessor_outputs['IMAGE_SIZES']:
            elem.unsqueeze_(1)
        for key, tensor_list in preprocessor_outputs.items():
            val = self.convert_tensor_list_to_tensor(tensor_list)
            if key in self.output_str_dtypes:
                val = self.convert_tensor_to_str_dtype(
                    val, self.output_str_dtypes[key])
            vision_processed_tensors[key] = val
        return vision_processed_tensors, visual_tokens

    def llava_onevision_process_video(self, queries, video_bytes=None):
        import torch
        vision_processed_tensors = {}
        videos = [video for video in self.load_images_tensor(video_bytes)]

        batch_size = len(videos)
        assert len(
            queries
        ) == batch_size, f"Video must have the same batch size as Query."
        preprocessor_outputs = {}
        preprocessor_outputs['PIXEL_VALUES'] = []
        preprocessor_outputs['IS_VIDEO_INPUT'] = []
        visual_tokens = []
        for batch_id in range(len(queries)):
            processed_vision_data = self.vision_model_processor(
                videos=list(videos[batch_id]),
                text='<video>',
                return_tensors="pt")
            visual_tokens.append(processed_vision_data['input_ids'].shape[1])
            preprocessor_outputs['PIXEL_VALUES'].append(
                processed_vision_data['pixel_values_videos'])
            preprocessor_outputs['IS_VIDEO_INPUT'].append(
                torch.ones((1, 1), dtype=torch.bool))

        for key, tensor_list in preprocessor_outputs.items():
            val = self.convert_tensor_list_to_tensor(tensor_list)
            if key in self.output_str_dtypes:
                val = self.convert_tensor_to_str_dtype(
                    val, self.output_str_dtypes[key])
            vision_processed_tensors[key] = val
        return vision_processed_tensors, visual_tokens

    def qwen2_vl_process_image(self, queries, img_urls=None, image_bytes=None):
        import torch
        vision_processed_tensors = {}
        # Retrieved from https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/config.json
        vision_token_id = 151654
        image_token_id = 151655
        video_token_id = 151656
        vocab_size = 152064

        if img_urls is not None:
            # download and read images
            images = [
                self.load_images_from_urls(urls)
                for urls in img_urls.as_numpy()
            ]
        else:
            images = [
                img for img_list in self.load_images_tensor(image_bytes)
                for img in img_list
            ]
        batch_size = len(images)
        preprocessor_outputs = defaultdict(list)
        possible_output_names = [
            'PIXEL_VALUES', 'IMAGE_GRID_THW', 'ATTENTION_MASK', 'INPUT_IDS'
        ]
        for batch_id in range(batch_size):
            messages = [{
                "role":
                "user",
                "content": [
                    {
                        "type": "image",
                        "image": images[batch_id],
                    },
                    {
                        "type":
                        "text",
                        "text":
                        queries[batch_id][0] if isinstance(
                            queries[batch_id], list) else queries[batch_id],
                    },
                ],
            }]
            text_inputs = self.vision_model_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            # Preprocess images and query
            processed_vision_data = self.vision_model_processor(
                images=images[batch_id],
                text=text_inputs,
                padding=True,
                return_tensors="pt")

            # Create vision output tensors
            for key in possible_output_names:
                val = processed_vision_data.get(key.lower())
                if val is not None:
                    # Add two dummy dim to reshape pixel value tensor to 5 dim
                    if key == 'PIXEL_VALUES':
                        val = val.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    elif key == 'INPUT_IDS':
                        val = val.to(torch.int32)
                        pre_process_val = val.clone()
                        mask = (val == image_token_id) | (
                            val == vision_token_id) | (val == video_token_id)
                        cumulative_counts = mask.cumsum(dim=1,
                                                        dtype=torch.int32)
                        values = (vocab_size - 1) + cumulative_counts
                        val[mask] = values[mask]
                        preprocessor_outputs["VISION_INPUT_ID"].append(
                            pre_process_val)
                        preprocessor_outputs["REQUEST_INPUT_LEN"].append(
                            torch.tensor([val.shape[1]],
                                         dtype=torch.int32).unsqueeze(0))
                    preprocessor_outputs[key].append(val)

        for key, tensor_list in preprocessor_outputs.items():
            val = self.convert_tensor_list_to_tensor(tensor_list)
            if key in self.output_str_dtypes:
                val = self.convert_tensor_to_str_dtype(
                    val, self.output_str_dtypes[key])
            vision_processed_tensors[key] = val
        return vision_processed_tensors

    def pixtral_process(self,
                        queries,
                        img_urls=None,
                        image_bytes=None,
                        image_sizes=None
                        ) -> Tuple[Dict[str, "torch.Tensor"], List[int]]:
        import torch
        vision_processed_tensors = {}
        if img_urls is not None:
            assert image_sizes is None, "IMAGE_SIZES should not be supplied together with IMAGE_URL"
            # download and read images
            images = [
                self.load_images_from_urls(urls)
                for urls in img_urls.as_numpy()
            ]
            images = [[np.array(img) for img in batch] for batch in images]

            # pad to the max_h, max_w dimensions to create one tensor for all images
            shapes = [img.shape for batch in images for img in batch]
            assert all(
                len(s) == 3
                for s in shapes), "All input images must have three dimensions"
            assert all(
                s[-1] == shapes[0][-1] for s in shapes
            ), "All input images must have the same number of channels"
            max_h, max_w = max(s[0] for s in shapes), max(s[1] for s in shapes)
            for batch_idx in range(len(images)):
                for image_idx in range(len(images[batch_idx])):
                    images[batch_idx][image_idx] = np.pad(
                        images[batch_idx][image_idx],
                        ((0, max_h - images[batch_idx][image_idx].shape[0]),
                         (0, max_w - images[batch_idx][image_idx].shape[1]),
                         (0, 0)),
                        mode='constant',
                    )
            images = np.array(images)
        elif image_bytes is not None:
            images = self.load_images_tensor(image_bytes)
        else:
            images = np.empty((len(queries), 0, 0, 0, 0), dtype=np.uint8)

        batch_size = len(images)
        assert len(
            queries
        ) == batch_size, f"Image must have the same batch size as Query."

        if image_sizes is not None:
            image_sizes = self.load_images_tensor(image_sizes)
        else:
            s = images.shape
            image_sizes = np.array([[[s[2], s[3]]] * s[1]] * s[0])

        preprocessor_outputs = {}
        possible_output_names = ['PIXEL_VALUES', 'IMAGE_SIZES', 'INPUT_IDS']
        visual_tokens = []
        for batch_id in range(batch_size):
            # Preprocess images and query
            query = queries[batch_id]
            if not isinstance(query, (str, bytes)):
                query = query[0]
            if isinstance(query, bytes):
                query = query.decode("utf-8")
            if "[IMG]" not in query:
                query = "[IMG]" * len(images[batch_id]) + query
            assert query.count("[IMG]") == len(
                images[batch_id]
            ), "Number of [IMG] tags must match number of images"

            if not query.startswith("[INST]"):
                query = "[INST]" + query
            if not query.endswith("[/INST]"):
                query = query + "[/INST]"

            sizes = image_sizes[batch_id]
            curr_images = [
                img[:sizes[idx][0], :sizes[idx][1], :]
                for idx, img in enumerate(images[batch_id])
            ]
            if not curr_images:
                curr_images = None

            processed_vision_data = self.vision_model_processor(
                images=curr_images, text=query, return_tensors="pt")
            visual_tokens.append(processed_vision_data['input_ids'].shape[1])
            if "pixel_values" in processed_vision_data:
                # Pad to self.image_size x self.image_size
                processed_vision_data['pixel_values'] = torch.nn.functional.pad(
                    processed_vision_data['pixel_values'], (
                        0,
                        self.image_size -
                        processed_vision_data['pixel_values'].shape[-1],
                        0,
                        self.image_size -
                        processed_vision_data['pixel_values'].shape[-2],
                    ),
                    mode='constant')
            # Create vision output tensors
            for key in possible_output_names:
                val = processed_vision_data.get(key.lower())
                if val is not None:
                    if key not in preprocessor_outputs:
                        preprocessor_outputs[key] = []
                    if key != 'INPUT_IDS':
                        val.unsqueeze_(0)  # unsqueeze to add batch dimension
                    preprocessor_outputs[key].append(val)

        for key, tensor_list in preprocessor_outputs.items():
            val = self.convert_tensor_list_to_tensor(tensor_list)
            if key in self.output_str_dtypes:
                val = self.convert_tensor_to_str_dtype(
                    val, self.output_str_dtypes[key])
            vision_processed_tensors[key] = val

        # Replace all image tokens with a unique token_id > vocab_size.
        # This shall be used to lookup the prompt table.
        for batch_id in range(batch_size):
            # Note: We reset replacer to vocab_size for each sample. This is as opposed to doing `replacer = vocab_size + img_idx * tokens_per_task`.
            # That part of the look-up manipulation is done by the `task_ids` input to PromptEmbedding forward.
            replacer = self.vocab_size
            input_ids = vision_processed_tensors['INPUT_IDS'][batch_id]
            for token_idx in range(len(input_ids)):
                if input_ids[token_idx] == self.image_token_index:
                    input_ids[token_idx] = replacer
                    replacer += 1

        return vision_processed_tensors, visual_tokens
