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
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from safetensors.torch import load_file
from torch.utils.dlpack import from_dlpack, to_dlpack

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, torch_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.runtime import Session, TensorInfo

logger.set_level('info')


def triton_string_to_torch(dtype):
    type_map = {
        "TYPE_BOOL": torch.bool,
        "TYPE_UINT8": torch.uint8,
        "TYPE_INT8": torch.int8,
        "TYPE_INT16": torch.int16,
        "TYPE_INT32": torch.int32,
        "TYPE_INT64": torch.int64,
        "TYPE_FP16": torch.float16,
        "TYPE_FP32": torch.float32,
        "TYPE_FP64": torch.float64,
        "TYPE_BF16": torch.bfloat16
    }
    return type_map[dtype]


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

        # Will load non-llm engiens only to GPU0 since the requests are coming only to GPU0

        self.rank = tensorrt_llm.mpi_rank()
        if self.rank != 0:
            return

        # Parse model configs
        model_config = json.loads(args['model_config'])

        self.max_batch_size = model_config['max_batch_size']

        # First vision engine
        multimodal_model_path = model_config['parameters'].get(
            'multimodal_model_path', None)
        if multimodal_model_path:
            multimodal_model_path = multimodal_model_path['string_value']
            self.vision_stream = torch.cuda.current_stream()

            visual_config_path = os.path.join(multimodal_model_path,
                                              'config.json')
            with open(visual_config_path, 'r') as f:
                visual_config = json.load(f)
            self.model_type = visual_config['builder_config']['model_type']

            multimodal_encoder_path = os.path.join(multimodal_model_path,
                                                   'model.engine')
            with open(multimodal_encoder_path, 'rb') as f:
                engine_buffer = f.read()
            self.image_session = Session.from_serialized_engine(engine_buffer)

            self.vision_dtype_str = visual_config['builder_config']['precision']
            self.vision_max_batch_size = visual_config['builder_config'][
                'max_batch_size']
            features_output_name = "OUT_PROMPT_EMBEDDING_TABLE"
            if self.model_type == "mllama":
                features_output_name = "ENCODER_INPUT_FEATURES"
            self.vision_output_dtype = triton_string_to_torch(
                pb_utils.get_output_config_by_name(
                    model_config, features_output_name)['data_type'])

            if self.model_type == 'llava_onevision':
                from multimodal_utils import LlavaOnevisionUtils
                from transformers import AutoConfig
                hf_model_path = model_config['parameters'].get(
                    'hf_model_path', None)
                assert hf_model_path is not None and hf_model_path[
                    'string_value'] != "${hf_model_path}", "Need to provide hf_model_path for the LLaVA OneVision model"

                newline_buffer_path = os.path.join(
                    multimodal_model_path, 'image_newlines.safetensors')
                image_newline = load_file(
                    newline_buffer_path)['image_newline'].cuda()
                model_config = AutoConfig.from_pretrained(
                    hf_model_path['string_value'])
                self.llava_onevision_utils = LlavaOnevisionUtils(
                    model_config, image_newline)

            if self.model_type == 'mllama':
                from transformers import AutoConfig
                hf_model_path = model_config['parameters'].get(
                    'hf_model_path', None)
                assert hf_model_path is not None and hf_model_path[
                    'string_value'] != "${hf_model_path}", "Need to provide hf_model_path for the MLLaMA model"
                model_config = AutoConfig.from_pretrained(
                    hf_model_path['string_value'])
                self.text_hidden_size = model_config.text_config.hidden_size

            if self.model_type == "qwen2_vl":
                from multimodal_utils import Qwen2VLUtils
                from transformers import AutoConfig
                hf_model_path = model_config['parameters'].get(
                    'hf_model_path', None)
                assert hf_model_path is not None and hf_model_path[
                    'string_value'] != "${hf_model_path}", "Need to provide hf_model_path for the Qwen2-VL model"
                hf_config = AutoConfig.from_pretrained(
                    hf_model_path['string_value'])
                self.config = hf_config
                self.vision_token_id = hf_config.vision_token_id
                self.image_token_id = hf_config.image_token_id
                self.video_token_id = hf_config.video_token_id
                self.vocab_size = hf_config.vocab_size
                self.qwen2vl_utils = Qwen2VLUtils(hf_config)

            if self.model_type == 'pixtral':
                from transformers import AutoConfig
                hf_model_path = model_config['parameters'].get(
                    'hf_model_path', None)
                assert hf_model_path is not None and hf_model_path[
                    'string_value'] != "${hf_model_path}", "Need to provide hf_model_path for the Pixtral model"
                hf_config = AutoConfig.from_pretrained(
                    hf_model_path['string_value'])
                self.image_size = hf_config.vision_config.image_size
                self.patch_size = hf_config.vision_config.patch_size
                self.vocab_size = hf_config.text_config.vocab_size
                self.spatial_merge_size = hf_config.spatial_merge_size
                self.relevant_patch_size = self.patch_size * self.spatial_merge_size

    def get_requests(self, request) -> Dict[str, torch.Tensor]:
        """
        Processes the incoming request to extract and organize input tensors
        for different model types.

        This function retrieves image tensors from the request, reshapes them
        as needed, and organizes them into a dictionary based on the model type
        being used. It supports handling both 'mllama' and 'llava_onevision' models,
        as well as other types.

        Args:
            request: The incoming request containing input tensors. The request
                    should contain tensors named either 'pixel_values' or 'IMAGE',
                    and optionally 'aspect_ratio_ids', 'aspect_ratio_mask',
                    'is_video_input' and 'image_sizes'.

        Returns:
            A tuple containing:
                - input_tensors (Dict[str, List[torch.Tensor]]): A dictionary
                of processed input tensors organized by their type.
                - batch_size (int): The size of the batch processed.
                - num_image (int): The number of images in the batch.

        Raises:
            AssertionError: If no valid image tensor is found in the request.
        """

        input_tensors: Dict[str, List(torch.Tensor)] = defaultdict(list)

        img_tensor = (pb_utils.get_input_tensor_by_name(request, 'pixel_values')
                      or pb_utils.get_input_tensor_by_name(request, 'IMAGE'))
        # mllama and pixtral support img_tensor is None case
        assert img_tensor != None or self.model_type in [
            'mllama', 'pixtral'
        ], "There is no preprocessed image tensor to encode"
        if img_tensor is not None:
            img_tensor = from_dlpack(img_tensor.to_dlpack())

            batch_size = img_tensor.shape[0]
            num_image = img_tensor.shape[1]
            img_tensor = img_tensor.to(str_dtype_to_torch(
                self.vision_dtype_str)).pin_memory()
        else:
            batch_size = 1
            num_image = 0
        # TODO these should be refactored into a factory
        if self.model_type == 'mllama':
            if img_tensor is not None:
                aspect_ratio_ids = from_dlpack(
                    pb_utils.get_input_tensor_by_name(
                        request, "aspect_ratio_ids").to_dlpack()).to(
                            torch.int64).pin_memory()
                aspect_ratio_mask = from_dlpack(
                    pb_utils.get_input_tensor_by_name(
                        request, "aspect_ratio_mask").to_dlpack()).to(
                            torch.int64).pin_memory()
                num_tiles = aspect_ratio_mask.shape[-1]
                # Reshape img_tensor to [bs, num_image, num_tiles, ...]
                if img_tensor is not None:
                    pixel_values = img_tensor.view(img_tensor.shape[0], -1,
                                                   num_tiles,
                                                   *(img_tensor.shape[2:]))
            else:
                pixel_values = None
                aspect_ratio_ids = None
                aspect_ratio_mask = None
            input_tensors['pixel_values'].append(pixel_values)
            input_tensors['aspect_ratio_ids'].append(aspect_ratio_ids)
            input_tensors['aspect_ratio_mask'].append(aspect_ratio_mask)
        elif self.model_type == 'llava_onevision':
            is_video_input = pb_utils.get_input_tensor_by_name(
                request, 'is_video_input')
            is_video_input = is_video_input.as_numpy(
            )[0] if is_video_input is not None else False
            if is_video_input:
                input_tensors['input'].append(
                    img_tensor.view(-1, img_tensor.shape[2],
                                    img_tensor.shape[3], img_tensor.shape[4]))
            else:
                image_sizes = from_dlpack(
                    pb_utils.get_input_tensor_by_name(
                        request, 'image_sizes').to_dlpack())
                # Remove dimension 1, which was added to match the dimensions defined in config.pbtxt
                assert image_sizes.shape[1] == 1
                image_sizes.squeeze_(1)
                from transformers.models.llava_onevision.modeling_llava_onevision import \
                    image_size_to_num_patches
                image_num_patches = [
                    image_size_to_num_patches(
                        image_size=imsize,
                        grid_pinpoints=self.llava_onevision_utils.config.
                        image_grid_pinpoints,
                        patch_size=self.llava_onevision_utils.config.
                        vision_config.image_size) for imsize in image_sizes
                ]
                img_tensor = img_tensor.to('cuda')
                img_list = [
                    img[:num_patch]
                    for img, num_patch in zip(img_tensor, image_num_patches)
                ]
                img_tensor = torch.cat(img_list, dim=0).contiguous()
                input_tensors['input'].append(img_tensor)

        elif self.model_type == 'qwen2_vl':
            image_grid_thw = from_dlpack(
                pb_utils.get_input_tensor_by_name(
                    request,
                    "image_grid_thw").to_dlpack()).to(torch.int64).pin_memory()
            attention_mask = from_dlpack(
                pb_utils.get_input_tensor_by_name(
                    request,
                    "attention_mask").to_dlpack()).to(torch.int64).pin_memory()
            #remove dummy dim and reshape to 2D dim
            img_tensor = img_tensor.squeeze(1).squeeze(1)
            img_tensor = img_tensor.view(-1, img_tensor.shape[-1])
            input_tensors['input'].append(img_tensor)
            input_tensors['attention_mask_llm'].append(attention_mask)
            input_tensors['image_grid_thw'].append(image_grid_thw)

        elif self.model_type == 'pixtral':
            if img_tensor is None:
                input_tensors['pixel_values'].append(None)
            else:
                assert batch_size == 1, "Only support batch size 1 for Pixtral, because each batch can contain a different number of images"
                d_min = torch.finfo(self.vision_output_dtype).min
                total_images = img_tensor.shape[0] * img_tensor.shape[1]
                num_patches = self.image_size // self.patch_size
                input_tensors['input'].append(
                    img_tensor.view(-1, img_tensor.shape[2],
                                    img_tensor.shape[3], img_tensor.shape[4]))
                attention_mask_shape = (total_images, num_patches, num_patches)
                attention_mask = torch.full(attention_mask_shape,
                                            fill_value=d_min,
                                            dtype=self.vision_output_dtype,
                                            device="cuda")
                image_sizes = from_dlpack(
                    pb_utils.get_input_tensor_by_name(
                        request,
                        'image_sizes').to_dlpack()).reshape(total_images, 2)
                for image_idx in range(total_images):
                    image_h, image_w = image_sizes[image_idx][0], image_sizes[
                        image_idx][1]
                    attention_mask[image_idx, :image_h //
                                   self.patch_size, :image_w //
                                   self.patch_size] = 0
                input_tensors['attention_mask'].append(attention_mask)
        else:
            input_tensors['input'].append(
                img_tensor.view(-1, img_tensor.shape[2], img_tensor.shape[3],
                                img_tensor.shape[4]))

        return input_tensors, batch_size, num_image

    def postprocess_output_tensors(
            self, output_tensor: torch.Tensor, requests: List,
            batch_sizes: List[int], num_images: List[int],
            is_skip_encoders: List[bool],
            other_vision_input_tensors: Dict[str, torch.Tensor]):
        """
        Processes the batched output tensor and generates inference responses
        for each request.

        This function splits the output tensor into individual embeddings for
        each request, reshapes the embeddings as needed based on the model type,
        and prepares the output tensors for returning. It supports handling
        multiple model types, including 'mllama' and 'llava_onevision'.

        Args:
            output_tensor (torch.Tensor): The batched output tensor containing
                                        the embeddings. Note that the num_images of
                                        all requests are fused together in dimension 0.
            requests (List): A list of requests containing input
                                        tensors for each model.
            batch_sizes (List[int]): A list of batch sizes corresponding to each
                                    request.
            num_images (List[int]): A list of image counts corresponding to each
                                    request.
            is_skip_encoders (List[bool]): A list of skipping the computing of each
                                    request since image_url is None.
            other_vision_input_tensors (Dict[str, torch.Tensor]): A dict of tensor containing tensor needed for some models postprocessing

        Returns:
            List[pb_utils.InferenceResponse]: A list of inference responses,
                                                each containing the output tensors
                                                for the corresponding request.

        Raises:
            Any exceptions raised by tensor operations (e.g., shape mismatches,
            invalid tensor types) during processing.

        Notes:
            - The function assumes that the input tensors are structured correctly
            and that the provided model types are supported.
            - The output tensors are created in a format suitable for the specific
            model being processed.
        """

        responses = []
        # split the batched output back to no batching
        if self.model_type == 'mllama':
            # Here, output_tensor is not the full tensor because we skip the encoder
            # engine computation for the requests which don't have img_url
            if output_tensor is not None:
                splitted_output_tensor = torch.tensor_split(
                    output_tensor, output_tensor.shape[0], dim=0)
            output_tensor_idx = 0
            for req_idx, request in enumerate(requests):
                batch_size = batch_sizes[req_idx]
                max_tokens = pb_utils.get_input_tensor_by_name(
                    request, 'max_tokens')
                # max_tokens is needed to prepare the cross_attention_mask
                max_tokens = 0 if max_tokens is None else max_tokens.as_numpy()[
                    0, 0]

                if is_skip_encoders[req_idx]:
                    # For the case that img_url is None, creating a dummy output tensor with short length
                    # and set skip_cross_attn_blocks as True. Also, creating corresponding dummy attention mask.
                    embeddings = torch.zeros(
                        [batch_size, 1, 4, 1,
                         self.text_hidden_size]).to(self.vision_output_dtype)
                    encoder_input_features = embeddings
                    output_shape = encoder_input_features.shape
                    skip_cross_attn_blocks = torch.ones([output_shape[0], 1],
                                                        dtype=torch.bool,
                                                        device='cpu')
                    cross_attention_mask = torch.zeros(
                        [batch_size, max_tokens, 1, 4],
                        dtype=torch.bool,
                        device='cuda')
                else:
                    embeddings = splitted_output_tensor[output_tensor_idx]
                    encoder_input_features = embeddings
                    output_shape = encoder_input_features.shape
                    skip_cross_attn_blocks = torch.zeros([batch_size, 1],
                                                         dtype=torch.bool,
                                                         device='cpu')
                    # prepare cross_attention_mask
                    # [bs, seq_len, num_tiles] to [bs, seq_len+max_tokens, encoder_length]
                    cross_attention_mask = pb_utils.get_input_tensor_by_name(
                        request, "cross_attention_mask")
                    if cross_attention_mask != None:
                        cross_attention_mask = from_dlpack(
                            pb_utils.get_input_tensor_by_name(
                                request, "cross_attention_mask").to_dlpack())
                    output_tensor_idx += 1

                cross_attention_mask = cross_attention_mask.repeat_interleave(
                    output_shape[3], dim=3)
                cross_attention_mask = cross_attention_mask.to(
                    encoder_input_features.device).to(torch.bool).view(
                        [output_shape[0], -1, encoder_input_features.shape[1]])
                tmp_mask = [cross_attention_mask] + [
                    cross_attention_mask[:, -1:, :] for _ in range(max_tokens)
                ]
                cross_attention_mask = torch.concat(tmp_mask, dim=1)
                logger.debug(
                    f"cross attention mask shape: {cross_attention_mask.shape}")
                logger.debug(
                    f"skip_cross_attn_blocks: {skip_cross_attn_blocks}")

                # reshape encoder output
                # [bs, num_image, num_tiles, num_patches, hidden_size] to [bs, encoder_length, hidden_size]
                encoder_input_features = encoder_input_features.view(
                    output_shape[0],
                    output_shape[1] * output_shape[2] * output_shape[3],
                    output_shape[4])
                logger.debug(
                    f"encoder_input_features shape: {encoder_input_features.shape}"
                )
                # prepare encoder output lengths
                # shape [bs], value [encoder_length]
                encoder_output_lengths = torch.tensor(
                    [[output_shape[1] * output_shape[2] * output_shape[3]]],
                    dtype=torch.int32)
                logger.debug(
                    f"encoder_output_lengths: {encoder_output_lengths}")
                # True when the request does not have image input

                response_tensors = [
                    pb_utils.Tensor.from_dlpack(
                        'ENCODER_INPUT_FEATURES',
                        to_dlpack(encoder_input_features)),
                    pb_utils.Tensor.from_dlpack(
                        'ENCODER_OUTPUT_LENGTHS',
                        to_dlpack(encoder_output_lengths))
                ]
                if cross_attention_mask is not None:
                    response_tensors.append(
                        pb_utils.Tensor.from_dlpack(
                            'CROSS_ATTENTION_MASK',
                            to_dlpack(cross_attention_mask)))
                response_tensors.append(
                    pb_utils.Tensor.from_dlpack(
                        'SKIP_CROSS_ATTN_BLOCKS',
                        to_dlpack(skip_cross_attn_blocks)))
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=response_tensors)
                responses.append(inference_response)
        elif self.model_type == 'llava_onevision':
            for req_idx, embeddings in enumerate(
                    torch.split(output_tensor, num_images, dim=0)):
                request = requests[req_idx]
                batch_size = batch_sizes[req_idx]
                num_image = num_images[req_idx]
                is_video_input = pb_utils.get_input_tensor_by_name(
                    request, 'is_video_input')
                if is_video_input:
                    prompt_table = self.llava_onevision_utils.postprocess_video(
                        embeddings, batch_size, num_image)
                else:
                    image_sizes = from_dlpack(
                        pb_utils.get_input_tensor_by_name(
                            request, 'image_sizes').to_dlpack())
                    # Remove dimension 1, which was added to match the dimensions defined in config.pbtxt
                    assert image_sizes.shape[1] == 1
                    image_sizes.squeeze_(1)
                    from transformers.models.llava_onevision.modeling_llava_onevision import \
                        image_size_to_num_patches
                    image_num_patches = [
                        image_size_to_num_patches(
                            image_size=imsize,
                            grid_pinpoints=self.llava_onevision_utils.config.
                            image_grid_pinpoints,
                            patch_size=self.llava_onevision_utils.config.
                            vision_config.image_size) for imsize in image_sizes
                    ]

                    prompt_table = self.llava_onevision_utils.postprocess_image(
                        embeddings, image_sizes, image_num_patches)
                prompt_embedding_table_tensor = pb_utils.Tensor.from_dlpack(
                    'OUT_PROMPT_EMBEDDING_TABLE', to_dlpack(prompt_table))
                response_tensors = [prompt_embedding_table_tensor]

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=response_tensors)
                responses.append(inference_response)
        elif self.model_type == 'qwen2_vl':
            image_grid_thw = other_vision_input_tensors.get('image_grid_thw')
            attention_mask = other_vision_input_tensors.get('attention_mask')
            total_num_image = [i * j for i, j in zip(batch_sizes, num_images)]
            image_grid_thw_list = list(
                torch.split(image_grid_thw, total_num_image, dim=0))
            attention_mask_list = list(
                torch.split(attention_mask, total_num_image, dim=0))
            single_image_prompt_table_size = int(output_tensor.shape[0] /
                                                 len(requests))
            prompt_embedding_table_tensor_tuple = torch.split(
                output_tensor, single_image_prompt_table_size)

            for req_idx in range(len(requests)):
                input_ids = from_dlpack(
                    pb_utils.get_input_tensor_by_name(
                        requests[req_idx], 'vision_input_id').to_dlpack())
                image_grid_thw = image_grid_thw_list[req_idx]
                attention_mask = attention_mask_list[req_idx]
                mrope_rotary_cos_sin, mrope_position_deltas = self.qwen2vl_utils.compute_mrope(
                    input_ids, image_grid_thw, attention_mask)
                prompt_embedding_table_tensor = pb_utils.Tensor.from_dlpack(
                    'OUT_PROMPT_EMBEDDING_TABLE',
                    to_dlpack(
                        prompt_embedding_table_tensor_tuple[req_idx].unsqueeze(
                            0)))
                mrope_rotary_cos_sin_tensor = pb_utils.Tensor.from_dlpack(
                    'MROPE_ROTARY_COS_SIN', to_dlpack(mrope_rotary_cos_sin))
                mrope_position_deltas_tensor = pb_utils.Tensor.from_dlpack(
                    'MROPE_POSITION_DELTAS', to_dlpack(mrope_position_deltas))
                response_tensors = [
                    prompt_embedding_table_tensor, mrope_rotary_cos_sin_tensor,
                    mrope_position_deltas_tensor
                ]
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=response_tensors)
                responses.append(inference_response)
        elif self.model_type == 'pixtral':
            assert len(num_images) == len(batch_sizes) == len(
                is_skip_encoders) == len(requests)
            images_per_batch = [i * b for i, b in zip(num_images, batch_sizes)]
            split_along = np.cumsum(images_per_batch).tolist()
            if output_tensor is not None:
                splitted_output_tensor = torch.tensor_split(output_tensor,
                                                            split_along,
                                                            dim=0)
                visual_embed_dim = output_tensor.shape[-1]
                output_img_size = self.image_size // self.relevant_patch_size

            for req_idx, request in enumerate(requests):
                if is_skip_encoders[req_idx]:
                    responses.append(
                        pb_utils.InferenceResponse(output_tensors=[]))
                    continue

                response_tensors = []
                assert splitted_output_tensor[req_idx].ndim == 3
                current_output_tensor = splitted_output_tensor[req_idx].reshape(
                    batch_sizes[req_idx], num_images[req_idx],
                    splitted_output_tensor[req_idx].shape[-2],
                    splitted_output_tensor[req_idx].shape[-1])
                image_sizes = from_dlpack(
                    pb_utils.get_input_tensor_by_name(
                        request, 'image_sizes').to_dlpack())
                complete_visual_features = []
                vocab_size = []
                for batch_idx in range(batch_sizes[req_idx]):
                    batch_visual_features = []
                    for image_idx in range(num_images[req_idx]):
                        image_h = image_sizes[batch_idx][image_idx][0]
                        image_w = image_sizes[batch_idx][image_idx][1]
                        h_patches = image_h // self.relevant_patch_size
                        w_patches = image_w // self.relevant_patch_size
                        relevant_visual_features = torch.zeros(
                            1, h_patches * w_patches, visual_embed_dim)
                        visual_features = current_output_tensor[batch_idx][
                            image_idx].reshape(output_img_size, output_img_size,
                                               visual_embed_dim)
                        flattened_features = visual_features[:h_patches, :
                                                             w_patches, :].flatten(
                                                                 0, 1)
                        relevant_visual_features[
                            0, :h_patches * w_patches, :] = flattened_features
                        batch_visual_features.append(relevant_visual_features)
                    batch_visual_features = torch.cat(batch_visual_features,
                                                      dim=1)
                    vocab_size.append(batch_visual_features.shape[1])
                    complete_visual_features.append(batch_visual_features)

                # Pad elements of complete_visual_features to have the same shape[1],
                # to allow concatenation over batch dimension
                max_vocab_size = max(vocab_size)
                for batch_idx in range(batch_sizes[req_idx]):
                    complete_visual_features[
                        batch_idx] = torch.nn.functional.pad(
                            complete_visual_features[batch_idx],
                            (0, 0, 0, max_vocab_size -
                             complete_visual_features[batch_idx].shape[1]),
                            mode='constant')
                complete_visual_features = torch.cat(complete_visual_features,
                                                     dim=0)

                prompt_embedding_table_tensor = pb_utils.Tensor.from_dlpack(
                    'OUT_PROMPT_EMBEDDING_TABLE',
                    to_dlpack(
                        complete_visual_features.type(
                            self.vision_output_dtype)))
                prompt_vocab_size_tensor = pb_utils.Tensor(
                    'OUT_PROMPT_VOCAB_SIZE',
                    np.array(vocab_size,
                             dtype=np.int32).reshape(batch_sizes[req_idx], 1))

                response_tensors.extend(
                    [prompt_embedding_table_tensor, prompt_vocab_size_tensor])
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=response_tensors)
                responses.append(inference_response)
        else:
            for req_idx, embeddings in enumerate(
                    torch.tensor_split(output_tensor,
                                       output_tensor.shape[0],
                                       dim=0)):
                batch_size = batch_sizes[req_idx]
                num_image = num_images[req_idx]
                vision_prompt_table = embeddings
                vision_prompt_vocab_size = np.array(
                    [[vision_prompt_table.shape[1]]])
                # Concatenate the prompt tables if there are multiple images in single request
                if num_image > 1:
                    prompt_table = vision_prompt_table.view(
                        batch_size, -1, vision_prompt_table.shape[-1])
                    prompt_vocab_size = np.repeat(vision_prompt_vocab_size,
                                                  batch_size,
                                                  axis=0)
                else:
                    # Use the single prompt table directly
                    vision_prompt_vocab_size = np.repeat(
                        vision_prompt_vocab_size, batch_size, axis=0)
                    prompt_table = vision_prompt_table
                    prompt_vocab_size = vision_prompt_vocab_size

                prompt_embedding_table_tensor = pb_utils.Tensor.from_dlpack(
                    'OUT_PROMPT_EMBEDDING_TABLE', to_dlpack(prompt_table))

                prompt_vocab_size_tensor = pb_utils.Tensor(
                    'OUT_PROMPT_VOCAB_SIZE', prompt_vocab_size.astype(np.int32))

                response_tensors = [
                    prompt_embedding_table_tensor, prompt_vocab_size_tensor
                ]

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=response_tensors)
                responses.append(inference_response)
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def run_vision_encoder(self, vit_input: Dict[str,
                                                 torch.Tensor]) -> torch.Tensor:
        batch_size = [v.shape[0] for v in vit_input.values()]
        assert all(
            b == batch_size[0]
            for b in batch_size), "Batch sizes of encoder inputs must match"
        batch_size = batch_size[0]

        embeddings = []
        for start_idx in range(0, batch_size, self.vision_max_batch_size):
            end_idx = min(start_idx + self.vision_max_batch_size, batch_size)
            logger.debug(
                f"Running encoder (max_batch_size={self.vision_max_batch_size}) "
                + f"with batch indices {start_idx}:{end_idx} of {batch_size}.")

            # Slice the input tensors along the batch dimension
            vit_input_batch = {
                k: v[start_idx:end_idx]
                for k, v in vit_input.items()
            }

            # Set up output tensors
            vit_input_info = [
                TensorInfo(key, torch_dtype_to_trt(val.dtype), val.shape)
                for key, val in vit_input_batch.items()
            ]
            vit_output_info = self.image_session.infer_shapes(vit_input_info)

            vit_output_batch = {
                t.name:
                torch.empty(tuple(t.shape),
                            dtype=trt_dtype_to_torch(t.dtype),
                            device='cuda')
                for t in vit_output_info
            }

            # Run the vision encoder
            with torch.cuda.stream(self.vision_stream):
                ok = self.image_session.run(vit_input_batch, vit_output_batch,
                                            self.vision_stream.cuda_stream)
                assert ok, "Runtime execution failed for vision encoder session"
            embeddings.append(vit_output_batch['encoder_output'].to(
                self.vision_output_dtype))

        with torch.cuda.stream(self.vision_stream):
            embeddings = torch.cat(embeddings, dim=0)

        self.vision_stream.synchronize()
        return embeddings

    def execute(self, requests: List):
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
        start_idx = 0
        responses = []
        next_input_tensors: Dict[str, List(torch.Tensor)] = defaultdict(list)
        next_batch_sizes = []
        next_num_images = []
        next_micro_batch_size = 0
        next_is_skip_encoders = []
        other_vision_input_tensors: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.tensor([]))
        while start_idx < len(requests):
            # Split the full requests into several micro batches
            # the batch size of each micro batch is smaller than self.max_batch_size
            input_tensors = next_input_tensors
            batch_sizes = next_batch_sizes
            num_images = next_num_images
            micro_batch_size = next_micro_batch_size
            is_skip_encoders = next_is_skip_encoders
            end_idx = start_idx
            # Continue adding requests to the current batch until we reach the maximum batch size or process all requests
            while micro_batch_size <= self.max_batch_size and end_idx < len(
                    requests):
                input_tensor, bs, num_image = self.get_requests(
                    requests[end_idx])
                # Check if adding this request to the current batch would exceed the maximum batch size
                if micro_batch_size + bs <= self.max_batch_size:
                    is_skip_encoder = False
                    for tensor_name, tensor in input_tensor.items():
                        if tensor[0] is None:
                            is_skip_encoder = True
                            break
                        if tensor_name in input_tensors:
                            input_tensors[tensor_name].extend(tensor)
                        else:
                            input_tensors[tensor_name] = tensor
                    # Update this batch information
                    batch_sizes.append(bs)
                    num_images.append(num_image)
                    is_skip_encoders.append(is_skip_encoder)
                    micro_batch_size += bs
                    end_idx += 1
                else:
                    # If adding this request would exceed the max batch size, prepare it for the next batch
                    is_skip_encoder = False
                    for tensor_name, tensor in input_tensor.items():
                        if tensor[0] is None:
                            is_skip_encoder = True
                            break
                        if tensor_name in next_input_tensors:
                            next_input_tensors[tensor_name].extend(tensor)
                        else:
                            next_input_tensors[tensor_name] = tensor
                    # Update next batch information
                    next_batch_sizes.append(bs)
                    next_num_images.append(num_image)
                    next_is_skip_encoders.append(is_skip_encoder)
                    next_micro_batch_size += bs
                    end_idx += 1

            logger.info(
                f"batch {end_idx - start_idx} requests (batch size {micro_batch_size}) together to run encoder model."
            )
            embeddings = None
            if not all(is_skip_encoders):
                vit_input = {}
                with torch.cuda.stream(self.vision_stream):
                    for tensor_name, tensors in input_tensors.items():
                        vit_input[tensor_name] = torch.cat(
                            [t.to('cuda', non_blocking=True) for t in tensors])

                if self.model_type == 'qwen2_vl':
                    import torch.nn.functional as F
                    from transformers.models.qwen2_vl.modeling_qwen2_vl import \
                        VisionRotaryEmbedding

                    from tensorrt_llm.tools.multimodal_builder import \
                        compute_rotary_pos_emb
                    img_tensor = vit_input.get('input')
                    image_grid_thw = vit_input.get('image_grid_thw')
                    vit_input.pop('image_grid_thw')
                    other_vision_input_tensors[
                        'image_grid_thw'] = image_grid_thw
                    attention_mask = vit_input.get('attention_mask_llm')
                    other_vision_input_tensors[
                        'attention_mask'] = attention_mask
                    vit_input.pop('attention_mask_llm')
                    cu_seqlens = torch.repeat_interleave(
                        image_grid_thw[:, 1] * image_grid_thw[:, 2],
                        image_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
                    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                    seq_length = img_tensor.shape[0]
                    attention_mask_vit = torch.full([1, seq_length, seq_length],
                                                    torch.finfo(
                                                        torch.float16).min,
                                                    dtype=img_tensor.dtype)
                    for i in range(1, len(cu_seqlens)):
                        attention_mask_vit[..., cu_seqlens[i - 1]:cu_seqlens[i],
                                           cu_seqlens[i - 1]:cu_seqlens[i]] = 0
                    rotary_pos_emb = compute_rotary_pos_emb(
                        image_grid_thw, self.config,
                        VisionRotaryEmbedding).to("cuda")
                    vit_input['rotary_pos_emb'] = rotary_pos_emb.to('cuda')
                    vit_input['attention_mask'] = attention_mask_vit.to(
                        str_dtype_to_torch(self.vision_dtype_str)).to('cuda')

                embeddings = self.run_vision_encoder(vit_input)

            # Post process output and save in responses
            responses.extend(
                self.postprocess_output_tensors(embeddings,
                                                requests[start_idx:end_idx],
                                                batch_sizes, num_images,
                                                is_skip_encoders,
                                                other_vision_input_tensors))
            start_idx = end_idx

        assert len(responses) == len(requests), \
            f"Number of responses ({len(responses)}) from the vision model does not match number of requests ({len(requests)})"
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        logger.info('Cleaning up...')
