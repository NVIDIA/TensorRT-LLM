import copy
import math
import os
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from PIL import Image
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)

from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import register_auto_model

DISAGG = os.getenv('TLLM_MULTIMODAL_DISAGGREGATED', '0') == '1'


# Copied from HyperCLOVAX-SEED-Vision-Instruct-3B/modeling_hyperclovax.py
def select_best_resolution(original_size: tuple,
                           possible_resolutions: list) -> tuple:
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height,
                                   original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution
                and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


# Copied from HyperCLOVAX-SEED-Vision-Instruct-3B/modeling_hyperclovax.py
def unpad_image(tensor: torch.Tensor,
                original_size: Tuple[int, int]) -> torch.Tensor:
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


# Copied from HyperCLOVAX-SEED-Vision-Instruct-3B/modeling_hyperclovax.py
def get_anyres_image_grid_shape(
    image_size: Tuple[int, int],
    grid_pinpoints: Union[str, List[Tuple[int, int]]],
    patch_size: int,
) -> Tuple[int, int]:
    import ast
    possible_resolutions = grid_pinpoints if isinstance(
        grid_pinpoints, list) else ast.literal_eval(grid_pinpoints)

    original_width, original_height = image_size
    height, width = select_best_resolution((original_height, original_width),
                                           possible_resolutions)
    return width // patch_size, height // patch_size


# Copied from HyperCLOVAX-SEED-Vision-Instruct-3B/modeling_hyperclovax.py
def reshape_and_unpad_image_features(
    image_feature: torch.Tensor,
    height: int,
    width: int,
    image_size: Tuple[int, int],
    possible_resolutions: List[Tuple[int, int]],
    grid_size: int,
    unpad: bool,
    image_newline: torch.Tensor,
) -> torch.Tensor:
    base_image_feature = image_feature[0]
    image_feature = image_feature[1:]

    assert (
        height * width == base_image_feature.shape[0]
    ), f"height: {height}, width: {width}, base_image_feature.shape[0]: {base_image_feature.shape[0]}"

    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
        image_size, possible_resolutions, grid_size)
    image_feature = image_feature.view(num_patch_height, num_patch_width,
                                       height, width, -1)

    if unpad:
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_size)
        image_feature = torch.cat(
            (
                image_feature,
                image_newline[:, None, None].expand(*image_feature.shape[:-1],
                                                    1).to(image_feature.device),
            ),
            dim=-1,
        )
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    else:
        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
        image_feature = image_feature.flatten(0, 3)
    image_feature = torch.cat((base_image_feature, image_feature), dim=0)

    return image_feature


# Copied from HyperCLOVAX-SEED-Vision-Instruct-3B/modeling_hyperclovax.py
def anyres_postprocessing(
    image_forward_outs: torch.FloatTensor,
    split_sizes: List[int],
    image_sizes: List[List[int]],
    possible_resolutions: List[Tuple[int, int]],
    is_videos: List[bool],
    patch_size: int,
    grid_size: int,
    image_newline: torch.FloatTensor,
    num_queries_vis_abstractor: int = -1,
    unpad: bool = False,
) -> List[torch.FloatTensor]:
    height = width = grid_size // patch_size

    if num_queries_vis_abstractor > 0:
        assert (num_queries_vis_abstractor**0.5
                ).is_integer(), "n_queries must be square number"
        height = width = int(num_queries_vis_abstractor**0.5)

    image_features = torch.split(image_forward_outs, split_sizes, dim=0)

    new_image_features = []
    for image_idx, (image_feature,
                    is_video) in enumerate(zip(image_features, is_videos)):
        if image_feature.shape[0] > 1:
            if not is_video:
                image_feature = reshape_and_unpad_image_features(
                    image_feature=image_feature,
                    height=height,
                    width=width,
                    image_size=image_sizes[image_idx],
                    possible_resolutions=possible_resolutions,
                    grid_size=grid_size,
                    unpad=unpad,
                    image_newline=image_newline,
                )
            else:
                image_feature = image_feature.flatten(0, 1)
        else:
            image_feature = image_feature[0]
            if unpad and not is_video:
                image_feature = torch.cat(
                    (image_feature, image_newline[None].to(
                        image_feature.device)),
                    dim=0)
        new_image_features.append(image_feature)
    image_features = new_image_features
    return image_features


# Copied from HyperCLOVAX-SEED-Vision-Instruct-3B/modeling_hyperclovax.py
def adaptive_anyres_postprocessing(
    image_forward_outs: torch.FloatTensor,
    image_sizes: List[List[int]],
    possible_resolutions: List[Tuple[int, int]],
    is_videos: List[bool],
    group_ids: List[List[int]],
    num_queries_vis_abstractors: List[List[int]],
    grid_size: int,
    image_newline: torch.FloatTensor,
    unpad: bool = False,
) -> List[torch.FloatTensor]:
    new_image_features = []
    for image_idx, (image_feature,
                    is_video) in enumerate(zip(image_forward_outs, is_videos)):
        num_queries_vis_abstractor = num_queries_vis_abstractors[image_idx]
        assert (num_queries_vis_abstractor**0.5
                ).is_integer(), "n_queries must be square number"
        height = width = int(num_queries_vis_abstractor**0.5)

        if image_feature.shape[0] > 1:
            if not is_video:
                image_feature = reshape_and_unpad_image_features(
                    image_feature=image_feature,
                    height=height,
                    width=width,
                    image_size=image_sizes[image_idx],
                    possible_resolutions=possible_resolutions,
                    grid_size=grid_size,
                    unpad=unpad,
                    image_newline=image_newline,
                )
            else:
                image_feature = image_feature.flatten(0, 1)
        else:
            image_feature = image_feature[0]
            if unpad and not is_video:
                image_feature = torch.cat(
                    (image_feature, image_newline[None].to(
                        image_feature.device)),
                    dim=0)
        new_image_features.append(image_feature)

    image_features = [
        torch.cat([new_image_features[group_id] for group_id in group_ids_list],
                  dim=0) for group_ids_list in group_ids
    ]
    return image_features


# Copied from HyperCLOVAX-SEED-Vision-Instruct-3B/modeling_hyperclovax.py
def compute_adaptive_params(
    pixel_values: Optional[List[List[torch.FloatTensor]]] = None,
    num_queries_vis_abstractors: Optional[List[List[int]]] = None,
    num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
    image_sizes: Optional[List[List[List[int]]]] = None,
    is_videos: Optional[List[bool]] = None,
    first_last_frames_slows: Optional[List[bool]] = None,
) -> Tuple[List[int], List[int], List[List[int]], List[bool], List[List[int]]]:
    assert all(
        all(isinstance(value, int) and value >= 0 for value in sublist)
        for sublist in num_queries_vis_abstractors
    ), "All values in num_queries_vis_abstractors must be integers >= 0."

    assert all(
        all(isinstance(value, int) and value >= 0 for value in sublist)
        for sublist in num_queries_vis_abstractors_slow
    ), "All values in num_queries_vis_abstractors_slow must be integers >= 0."

    assert is_videos is not None

    is_first_images = []
    is_last_images = []
    for is_video in is_videos:
        for idx, is_video_item in enumerate(is_video):
            if idx == 0:
                is_first_images.append(True)
            else:
                is_first_images.append(False)
            if idx == len(is_video) - 1:
                is_last_images.append(True)
            else:
                is_last_images.append(False)

    num_queries_vis_abstractors = list(chain(*num_queries_vis_abstractors))
    num_queries_vis_abstractors_slow = list(
        chain(*num_queries_vis_abstractors_slow))
    image_sizes = list(chain(*image_sizes))
    is_videos = list(chain(*is_videos))
    first_last_frames_slows = list(chain(*first_last_frames_slows))

    use_slowfast = any(
        [num_query > 0 for num_query in num_queries_vis_abstractors_slow])
    num_grids = [pixel_value.shape[0] for pixel_value in chain(*pixel_values)]
    num_grids = [0] + num_grids
    group_ids = []

    if use_slowfast:
        new_num_grids = [num_grids[0]]
        new_num_queries = []
        new_image_sizes = []
        new_is_videos = []

        for (
                num_query,
                num_query_slow,
                num_grid,
                image_size,
                is_video,
                first_last_frames_slow,
                is_first_image,
                is_last_image,
        ) in zip(
                num_queries_vis_abstractors,
                num_queries_vis_abstractors_slow,
                num_grids[1:],
                image_sizes,
                is_videos,
                first_last_frames_slows,
                is_first_images,
                is_last_images,
        ):

            if not first_last_frames_slow and num_query_slow > 0:
                assert is_video

                this_group_ids = [group_ids[-1][-1] + 1 if group_ids else 0]

                new_num_grids.append(new_num_grids[-1] + 1)
                new_num_queries.append(num_query_slow)
                new_image_sizes.append(image_size)
                new_is_videos.append(is_video)

                if num_grid >= 2:
                    new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                    new_num_queries.append(num_query)
                    new_image_sizes.append(image_size)
                    new_is_videos.append(is_video)
                    this_group_ids.append(this_group_ids[-1] + 1)

                group_ids.append(this_group_ids)
            elif (first_last_frames_slow and num_query_slow > 0
                  and (is_first_image or is_last_image)):
                assert is_video

                this_group_ids = [group_ids[-1][-1] + 1 if group_ids else 0]

                if num_grid == 1:
                    new_num_grids.append(new_num_grids[-1] + 1)
                    new_num_queries.append(num_query_slow)
                    new_image_sizes.append(image_size)
                    new_is_videos.append(is_video)

                if num_grid >= 2:
                    if is_first_image:
                        new_num_grids.append(new_num_grids[-1] + 1)
                        new_num_queries.append(num_query_slow)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                        new_num_queries.append(num_query)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        this_group_ids.append(this_group_ids[-1] + 1)
                    elif is_last_image:
                        new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                        new_num_queries.append(num_query)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        new_num_grids.append(new_num_grids[-1] + 1)
                        new_num_queries.append(num_query_slow)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        this_group_ids.append(this_group_ids[-1] + 1)
                    else:
                        raise Exception("This case should not be reached.")
                group_ids.append(this_group_ids)
            else:
                new_num_grids.append(new_num_grids[-1] + num_grid)
                new_num_queries.append(num_query)
                new_image_sizes.append(image_size)
                new_is_videos.append(is_video)

                start_group_id = group_ids[-1][-1] + 1 if group_ids else 0
                group_ids.append([start_group_id])

        num_grids = new_num_grids
        num_queries_vis_abstractors = new_num_queries
        image_sizes = new_image_sizes
        is_videos = new_is_videos
    else:
        num_grids = [sum(num_grids[:i]) for i in range(1, len(num_grids) + 1)]
        group_ids = [[group_id] for group_id in range(len(is_videos))]

    return num_queries_vis_abstractors, num_grids, image_sizes, is_videos, group_ids


# Copied from HyperCLOVAX-SEED-Vision-Instruct-3B/modeling_hyperclovax.py
def determine_non_vision_query_lengths(input_ids: torch.LongTensor, pad_id: int,
                                       img_start_id: int) -> List[int]:
    non_vision_query_lengths = []
    batch_size, len_seq = input_ids.size(0), input_ids.size(1)

    for i in range(batch_size):
        temp_idx = (input_ids[i] == pad_id).nonzero()
        eos_idx = temp_idx[0, 0].item() if len(temp_idx) > 0 else len_seq
        num_imgs = (input_ids[i] == img_start_id).sum().item()
        non_vision_query_lengths.append(eos_idx - num_imgs)

    if all([pad_id in input_id for input_id in input_ids.tolist()]):
        non_vision_query_lengths = [
            non_vision_query_length + 1
            for non_vision_query_length in non_vision_query_lengths
        ]

    return non_vision_query_lengths


class HCXVisionInputProcessor(InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True):

        self.pretrained_config = model_config
        self.tokenizer = tokenizer
        self.use_fast = True
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast)
        self.tllm_image_token_id = self.pretrained_config.language_config[
            "vocab_size"] + 1
        if DISAGG:
            self.mm_encoder = HCXVisionModel(self.pretrained_config,
                                             skip_processor=True)

    def _post_process(self,
                      input_ids: torch.Tensor,
                      preprocessed_image: dict[str, any] = None):
        if not preprocessed_image:
            return input_ids

        vision_query_lengths = preprocessed_image.get("vision_query_lengths",
                                                      None)
        non_vision_query_lengths = determine_non_vision_query_lengths(
            input_ids, self.tokenizer.pad_token_id,
            self.pretrained_config.img_start_id)
        batch_size = input_ids.size(0)

        len_inputs_embeds = max([
            sum(vision_query_length) + non_vision_query_length
            for non_vision_query_length, vision_query_length in zip(
                non_vision_query_lengths, vision_query_lengths)
        ])

        len_inputs_embeds = min(self.pretrained_config.decoder_max_length,
                                len_inputs_embeds)

        image_cnts = (input_ids == self.pretrained_config.img_start_id).sum(
            dim=1).tolist()

        fused_input_ids = torch.zeros([batch_size, len_inputs_embeds],
                                      dtype=input_ids.dtype)
        for batch_idx, sample in enumerate(input_ids):
            non_vision_query_length = non_vision_query_lengths[batch_idx]
            sample = sample[:non_vision_query_length + image_cnts[batch_idx]]

            mask = (sample == self.pretrained_config.img_start_id)
            img_start_ids = mask.nonzero()
            input_start, temp_start = 0, 0

            for multi_img_idx, img_start_idx in enumerate(img_start_ids):
                token_len = img_start_idx - temp_start

                fused_input_ids[batch_idx, input_start:input_start +
                                token_len] = input_ids[batch_idx,
                                                       temp_start:temp_start +
                                                       token_len]

                fused_input_ids[
                    batch_idx,
                    input_start + token_len:input_start + token_len +
                    vision_query_lengths[batch_idx][multi_img_idx],
                ] = self.tllm_image_token_id

                input_start += token_len + vision_query_lengths[batch_idx][
                    multi_img_idx]
                temp_start += token_len + 1

            token_len = min(sample[temp_start:].size(0),
                            fused_input_ids.size(1) - input_start)
            fused_input_ids[batch_idx, input_start:input_start +
                            token_len] = input_ids[batch_idx,
                                                   temp_start:temp_start +
                                                   token_len]

        return fused_input_ids[0]

    def _preprocess(self, text_prompt: dict[str, any], images: List[Any],
                    mm_processor_kwargs: Dict[str, Any]):

        preprocessed_image = None
        is_video_list = [False] * len(images)
        if images is not None:
            is_video_list = [False] * len(images)
            preprocessed_image = self.processor(
                images=images,
                is_video_list=is_video_list,
                **mm_processor_kwargs,
            )

        input_ids = self.tokenizer.encode(text_prompt,
                                          add_special_tokens=False,
                                          return_tensors="pt")
        return input_ids, preprocessed_image

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = inputs.get("prompt"), \
                        inputs.get("multi_modal_data", {}), inputs.get("mm_processor_kwargs", {})

        images = mm_data.get("image", None)
        if images is not None:
            if isinstance(images[0], torch.Tensor):
                # NOTE: HyperCLOVA-SEED-Vision-Instruct-3B uses the image data in the format of (H, W, C) and not in the range of [0, 1], otherwise gives an error.
                images = [(image.permute(1, 2, 0) * 255).to(torch.uint8)
                          for image in images]

        input_ids, preprocessed_image = self._preprocess(
            text_prompt, images, mm_processor_kwargs)

        fused_input_ids = self._post_process(input_ids, preprocessed_image)

        if not preprocessed_image:
            return fused_input_ids.to(torch.int32).tolist(), {}

        if DISAGG:
            mm_embeds = self.mm_encoder.forward(preprocessed_image)
            mm_embeds = torch.cat(mm_embeds, dim=0)
        else:
            # NOTE: For now, I am using "mm_embeding" in tensor format to send the image data to the model.
            # CASE 1: Sending raw image data
            if isinstance(images[0], Image.Image):
                images = [torch.from_numpy(np.array(image)) for image in images]
            mm_embeds = torch.stack(images, dim=0)

            # NOTE: After refactoring the llmRequest, we can use preprocessed_image['pixel_values'] to send the image data to the model.
            # CASE 2: Sending preprocessed image data
            # mm_embeds = torch.cat(preprocessed_image['pixel_values'][0],
            #                                dim=0)

        return fused_input_ids.to(torch.int32).tolist(), {
            "mm_embedding": mm_embeds,
        }


class HCXVisionModel:

    def __init__(self,
                 pretrained_config: PretrainedConfig,
                 skip_processor: bool = False):

        self.pretrained_config = pretrained_config
        self.vision_config = self.pretrained_config.vision_config

        model_path = self.pretrained_config._name_or_path

        # TODO: Remove this when we refactor LlmRequest
        # NOTE: trust_remote_code can be removed once we refactor LlmRequest
        self.skip_processor = skip_processor
        if not self.skip_processor:
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, use_fast=True)

        # NOTE: There is no way of importing mm_projector, HCXVisionCAbstractor from HF. So, can not do the sharded_loading.
        # NOTE: trust_rmemote_code can be removed once we change the model into TRT-LLM's format
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True)
        model.eval()
        self.device = 'cuda'

        # TODO: Convert to TRT-LLM's SIGLIP
        self.vision_model = model.vision_model.to(self.device)
        self.mm_projector = model.mm_projector.to(self.device)
        self.image_newline = model.image_newline.to(self.device)

        self.unpad = self.pretrained_config.unpad
        self.use_nth_layer = self.pretrained_config.use_nth_layer
        self.anyres = self.pretrained_config.anyres
        self.possible_resolutions = self._init_possible_resolutions(
            self.pretrained_config)

    def _init_possible_resolutions(self, config: PretrainedConfig):
        possible_resolutions = []
        if config.anyres:
            assert config.max_num_grids > 0
            for i in range(1, config.max_num_grids + 1):
                for j in range(1, config.max_num_grids + 1):
                    if i == 1 and j == 1 and not config.use_1x1_grid:
                        continue
                    if i * j <= config.max_num_grids:
                        possible_resolutions.append([i, j])

            possible_resolutions = [[
                ys * config.vision_config["image_size"],
                xs * config.vision_config["image_size"]
            ] for ys, xs in possible_resolutions]
        return possible_resolutions

    def _to_device(
            self, input_tensor: Union[torch.Tensor,
                                      List]) -> Union[torch.Tensor, List]:
        if isinstance(input_tensor, list):
            return [self._to_device(item) for item in input_tensor]
        elif isinstance(input_tensor, torch.Tensor):
            return input_tensor.to(self.device)

    # TODO: Remove this when we refactor LlmRequuest
    def _preprocess(self, mm_data: List[Any]) -> Dict[str, List[Any]]:
        preprocessed_image_list = []

        for images in mm_data:
            images = torch.unbind(images, dim=0)
            preprocessed_image = self.processor(
                images=images,
                is_video_list=[False] * len(images),
            )

            # NOTE: The HCXVisionInputProcessor makes pixel_vlues to CPU values even though use_fast = True.
            # So, we need to transfer them to GPU.
            preprocessed_image["pixel_values"] = self._to_device(
                preprocessed_image["pixel_values"])

            preprocessed_image_list.append(preprocessed_image)

        return {
            key: [d[key][0] for d in preprocessed_image_list]
            for key in preprocessed_image_list[0].keys()
        }

    def forward(self, mm_data: Union[List[Any], Dict[str, Any]]):
        if not self.skip_processor:
            # NOTE: This should be done in the input processor and got the preprocessed_image metadata from request level.
            # But before refactoring the llmRequest, we are re-doing inputprocessor here.
            preprocessed_image = self._preprocess(mm_data)
        else:
            # NOTE: When we refactor the llmRequest, we will get the extra_mm_data from mm_data, and need to make it as preprocessed_image.
            preprocessed_image = mm_data
            preprocessed_image["pixel_values"] = self._to_device(
                preprocessed_image["pixel_values"])

        pixel_values = preprocessed_image.get("pixel_values", None)
        image_sizes = preprocessed_image.get("image_sizes", None)
        is_videos = preprocessed_image.get("is_videos", None)
        num_queries_vis_abstractors = preprocessed_image.get(
            "num_queries_vis_abstractors", None)
        num_queries_vis_abstractors_slow = preprocessed_image.get(
            "num_queries_vis_abstractors_slow", None)
        first_last_frames_slows = preprocessed_image.get(
            "first_last_frames_slows", None)

        len_pixel_values = [len(pixel_value) for pixel_value in pixel_values]
        concat_pixel_values = torch.cat(list(chain(*pixel_values)),
                                        dim=0)  # list of list of 4D Tensor
        visual_token_idx = 0 if "siglip" in self.vision_config[
            "model_type"] else 1

        n_chunks = 1
        total_len = concat_pixel_values.size(0)
        chunk_size = math.ceil(total_len / n_chunks) if total_len > 0 else 1
        image_forward_outs_chunks = []

        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = concat_pixel_values[start:end].to(self.vision_model.dtype)
            if chunk.size(0) < chunk_size:
                pad_size = chunk_size - chunk.size(0)
                dummy_shape = (pad_size, ) + tuple(
                    concat_pixel_values.shape[1:])
                dummy = torch.zeros(
                    dummy_shape,
                    dtype=concat_pixel_values.dtype,
                    device=concat_pixel_values.device,
                )
                chunk = torch.cat([chunk, dummy], dim=0)

            if self.use_nth_layer == -1:
                self.vision_model.vision_model.post_layernorm = nn.Identity()
                outs = self.vision_model(chunk)
                outs = outs.last_hidden_state[:, visual_token_idx:]
            else:
                outs = self.vision_model(chunk, output_hidden_states=True)
                outs = outs.hidden_states[self.use_nth_layer][:,
                                                              visual_token_idx:]
            image_forward_outs_chunks.append(outs)

        image_forward_outs = torch.cat(image_forward_outs_chunks, dim=0).to(
            image_forward_outs_chunks[0].dtype)

        if num_queries_vis_abstractors is None:
            assert num_queries_vis_abstractors_slow is None
            image_sizes = list(chain(*image_sizes))
            if is_videos is not None:
                is_videos = list(chain(*is_videos))
            group_ids = None
            image_forward_outs = image_forward_outs.to(
                dtype=self.mm_projector.dtype)
            image_forward_outs = self.mm_projector(image_forward_outs)
        else:
            (
                num_queries_vis_abstractors,
                num_grids,
                image_sizes,
                is_videos,
                group_ids,
            ) = compute_adaptive_params(
                pixel_values,
                num_queries_vis_abstractors,
                num_queries_vis_abstractors_slow,
                image_sizes,
                is_videos,
                first_last_frames_slows,
            )

            image_forward_outs = image_forward_outs.to(
                dtype=self.mm_projector.dtype)
            image_forward_outs = self.mm_projector(
                image_forward_outs,
                num_queries_vis_abstractors=num_queries_vis_abstractors,
                num_grids=num_grids,
            )
        if self.anyres:
            split_sizes = [
                pixel_value.shape[0] for pixel_value in chain(*pixel_values)
            ]
            if num_queries_vis_abstractors is None:
                image_features = anyres_postprocessing(
                    image_forward_outs=image_forward_outs,
                    split_sizes=split_sizes,
                    image_sizes=image_sizes,
                    num_queries_vis_abstractor=self.num_queries_vis_abstractor,
                    unpad=self.unpad,
                    is_videos=is_videos,
                    patch_size=self.vision_model.config.patch_size,
                    grid_size=self.vision_model.config.image_size,
                    image_newline=self.image_newline,
                    possible_resolutions=self.possible_resolutions,
                )
            else:
                image_features = adaptive_anyres_postprocessing(
                    image_forward_outs=image_forward_outs,
                    image_sizes=image_sizes,
                    num_queries_vis_abstractors=num_queries_vis_abstractors,
                    unpad=self.unpad,
                    is_videos=is_videos,
                    grid_size=self.vision_model.config.image_size,
                    image_newline=self.image_newline,
                    possible_resolutions=self.possible_resolutions,
                    group_ids=group_ids,
                )
        else:
            if num_queries_vis_abstractors is None:
                image_features = [
                    image_forward_out
                    for image_forward_out in image_forward_outs
                ]
            else:
                image_features = [
                    image_forward_out.unsqueeze(0)
                    for image_forward_out in image_forward_outs
                ]
        image_features = [
            image_features[sum(len_pixel_values[:i]):sum(len_pixel_values[:i +
                                                                          1])]
            for i in range(len(len_pixel_values))
        ]
        mm_embeds = [
            torch.cat(list(chain(image_feature)), dim=0)
            for image_feature in image_features
        ]
        return mm_embeds


@register_auto_model("HCXVisionForCausalLM")
@register_input_processor(HCXVisionInputProcessor, model_type="hyperclovax_vlm")
class HCXVisionForCausalLM(PreTrainedModel):

    def __init__(self, model_config: ModelConfig):
        config = model_config.pretrained_config
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        if not DISAGG:
            self.mm_encoder = HCXVisionModel(model_config.pretrained_config)

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = PretrainedConfig.from_dict(
            llm_model_config.pretrained_config.language_config)
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.model_dtype = getattr(config, "torch_dtype", torch.float16)
        logger.info(f"{self.dtype=} {self.model_dtype=}")
        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v.to(self.dtype)
                    assert result[new_k] is not None
            return result

        weights = filter_weights("language_model", weights)
        self.llm.load_weights(weights)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        VLM forward logic with inflight batching support.
        """
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(
            f"num_context_requests: {num_context_requests}, num_generation_requests: {num_generation_requests}"
        )

        mm_data = kwargs.get("multi_modal_data", [])
        mm_embeds = []
        if len(mm_data) > 0:
            assert len(
                mm_data
            ) == num_context_requests, f"Number of multimodal tensors ({len(mm_data)}) should be equal to number of context requests ({num_context_requests}) in the batch."
            if DISAGG:
                # NOTE: In the DISAGG, we are assuming we get the mm_embeds from the llmRequest.
                mm_embeds = mm_data
            else:
                mm_embeds = self.mm_encoder.forward(mm_data)

        input_ids, input_embeds = fuse_input_embeds(self.llm.model.embed_tokens,
                                                    input_ids, mm_embeds)
        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits)

        logger.debug(f'output shape: {output_prob.shape}')
        return output_prob
