# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is based on official VILA: https://github.com/NVlabs/VILA/

import copy
import math
import os
import re
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import repo_exists, snapshot_download
from huggingface_hub.utils import HFValidationError
from PIL import Image
from transformers import AutoConfig, AutoModel
from transformers import AutoModelForCausalLM as HFAutoModelForCausalLM
from transformers import (AutoTokenizer, LlavaConfig, PretrainedConfig,
                          PreTrainedModel)

from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_encoder import (CLIPVisionTower, CLIPVisionTowerS2,
                                          SiglipVisionTower,
                                          SiglipVisionTowerDynamicS2,
                                          SiglipVisionTowerS2)
from .modeling_multimodal_utils import (merge_chessboard,
                                        merge_features_for_dynamic_s2,
                                        split_chessboard)
from .modeling_utils import ModelConfig, register_auto_model

SENTINEL_TOKEN = "<vila/sentinel>"  # nosec B105
MEDIA_TOKENS = {
    "image": "<image>",
    "video": "<vila/video>",
}


def _convert_dtype(dtype):
    # If encountered tiny accuracy difference, note that VILA sometimes forced torch_dtype / model_dtype to be float16, while TRT-LLM follows the checkpoint dtype.
    # ref: https://github.com/NVlabs/VILA/blob/86e009759a14eee045c669421128d703227da362/llava/model/builder.py#L53
    if (isinstance(dtype, str)
            and dtype == "torch.float16") or (isinstance(dtype, torch.dtype)
                                              and dtype == torch.float16):
        return torch.float16
    elif (isinstance(dtype, str)
          and dtype == "torch.bfloat16") or (isinstance(dtype, torch.dtype)
                                             and dtype == torch.bfloat16):
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupportede dtype for VILA: {dtype}")


def get_model_paths(config):
    default_keys = ["llm_cfg", "vision_tower_cfg", "mm_projector_cfg"]

    root_path = None
    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path

    # download from huggingface
    if root_path is not None and not os.path.exists(root_path):
        try:
            valid_hf_repo = repo_exists(root_path)
        except HFValidationError:
            valid_hf_repo = False
        if valid_hf_repo:
            root_path = snapshot_download(root_path)

    paths = []
    for key in default_keys:
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            component_path = cfg.get("_name_or_path", None)
        elif isinstance(cfg, PretrainedConfig):
            component_path = getattr(cfg, "_name_or_path", None)
        elif isinstance(cfg, str):
            component_path = cfg
        else:
            raise RuntimeError(f"Invalid config type: {type(cfg)}")

        assert component_path is not None, f"Cannot find _name_or_path in config.json for component {key}!"

        # parse ckpt structure. NVILA is /llm, /vision_tower, /mm_projector, but variants could have arbitrary structure that should be parsed rather than hard-coded
        if not isinstance(cfg, str):
            component_name = os.path.basename(component_path.strip("/"))
            component_path = os.path.join(root_path, component_name)
        paths.append(component_path)

    if len(paths) != 3:
        raise ValueError(
            "one of `llm_cfg`, `mm_projector_cfg`, `vision_tower_cfg` not found in the config.json"
        )

    return paths


def _ptuning_setup(mm_feature, input_ids):
    assert mm_feature is not None, "Multimodal feature is empty!"

    task_vocab_size = torch.tensor(
        [mm_feature.shape[1]],
        dtype=torch.int32,
    ).cuda()

    prompt_table = mm_feature.view(
        (mm_feature.shape[0] * mm_feature.shape[1], mm_feature.shape[2]))

    tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

    return [prompt_table, tasks, task_vocab_size]


"""
VILA vision tower processing utilities
based on: https://github.com/NVlabs/VILA/llava/mm_utils.py
"""


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image,
                       min_num=1,
                       max_num=12,
                       image_size=384,
                       use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios,
                                                    orig_width, orig_height,
                                                    image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dynamic_s2_preprocess(image,
                          s2_scales=[384, 768, 1152],
                          max_num=12,
                          image_size=384):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    min_num = (
        s2_scales[-1] //
        s2_scales[0])**2  # at least use number of tiles as the largest scale

    processed_images = []

    ##########################################################################################
    ############# Add tiles for all but the last scale using fixed squre ratio ###############
    ##########################################################################################

    for scale in s2_scales[:-1]:
        target_width = image_size * (scale // s2_scales[0])
        target_height = image_size * (scale // s2_scales[0])
        blocks = (scale // s2_scales[0])**2

        # resize the image
        resized_img = image.resize((target_width, target_height))
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

    ##########################################################################################
    ################ Add tiles for the last scale using dynamic aspect ratio #################
    ##########################################################################################

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios,
                                                    orig_width, orig_height,
                                                    image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    return processed_images, (target_aspect_ratio[1], target_aspect_ratio[0])


def process_image(image_file, image_processor, model_config):
    image = image_file.convert("RGB")
    image_aspect_ratio = model_config.image_aspect_ratio

    if hasattr(image_processor, "crop_size"):
        crop_size = image_processor.crop_size  # CLIP vision tower
    elif hasattr(image_processor, "size"):
        crop_size = image_processor.size  # SIGLIP vision tower

    if image_aspect_ratio == "resize":
        # VILA 1.0
        image = image.resize((crop_size["height"], crop_size["width"]))

    if image_aspect_ratio == "pad":
        # VILA 1.0
        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width),
                                   background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height),
                                   background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean))
        image = image_processor.preprocess(
            image, return_tensors="pt")["pixel_values"][0]

    if image_aspect_ratio == "dynamic":
        # VILA 2.0
        assert crop_size["height"] == crop_size["width"]
        images = dynamic_preprocess(image,
                                    min_num=model_config.min_tiles,
                                    max_num=model_config.max_tiles,
                                    image_size=crop_size["height"])
        images = [
            image_processor.preprocess(image,
                                       return_tensors="pt")["pixel_values"][0]
            for image in images
        ]
        return torch.stack(images)

    if image_aspect_ratio == "dynamic_s2":
        # VILA 2.0
        assert crop_size["height"] == crop_size["width"]
        if type(model_config.s2_scales) is str:
            s2_scales = list(map(int, model_config.s2_scales.split(",")))
        else:
            s2_scales = model_config.s2_scales
        images, block_size = dynamic_s2_preprocess(
            image,
            s2_scales=s2_scales,
            max_num=model_config.max_tiles,
            image_size=crop_size["height"])
        images = [
            image_processor.preprocess(image,
                                       return_tensors="pt")["pixel_values"][0]
            for image in images
        ]
        return torch.stack(images), block_size

    else:
        # Using default behavior of the vision encoder
        image = image_processor.preprocess(
            image, return_tensors="pt")["pixel_values"][0]
    return image


def process_images(image_files, image_processor, model_config):
    images = [
        process_image(image, image_processor, model_config)
        for image in image_files
    ]

    block_sizes = None
    if isinstance(images[0], tuple):
        # VILA 2.0 dynamic S2 has block_sizes parameter
        images, block_sizes = zip(*images)
        block_sizes = list(block_sizes)

    if all(x.shape == images[0].shape for x in images):
        if len(images[0].shape) == 4:
            images = torch.cat(images, dim=0)
        elif len(images[0].shape) == 3:
            images = torch.stack(images, dim=0)
        else:
            raise ValueError(
                f"images rank does not equal to 4, rank: {len(images[0].shape)}"
            )
    else:
        raise ValueError("The shapes of images in the list are different!")
    return images, block_sizes


"""End of vision processing utilities"""
"""
VILA multimodal projector module
"""


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(),
                                  nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class DownSampleBlock(nn.Module):

    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat(
                [x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)],
                dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat(
                [x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)],
                dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.contiguous()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class DownSample2x2BlockFix(nn.Module):

    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = flat_square_2x2(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        return vit_embeds


def flat_square_2x2(x):
    n, w, h, c = x.size()
    if w % 2 == 1:
        x = torch.concat(
            [x, torch.zeros(
                (n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
        n, w, h, c = x.size()
    x = x.contiguous()
    if h % 2 == 1:
        x = torch.concat(
            [x, torch.zeros(
                (n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
        n, w, h, c = x.size()
    x = x.view(n, w, int(h / 2), int(c * 2))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class DownSample3x3BlockFix(nn.Module):

    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = flat_square_3x3(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        return vit_embeds


def flat_square_3x3(x):
    n, w, h, c = x.size()
    if w % 3 != 0:
        x = torch.concat([
            x,
            torch.zeros((n, 3 - (w % 3), h, c), dtype=x.dtype).to(x.device)
        ],
                         dim=1).contiguous()
        n, w, h, c = x.size()
    x = x.contiguous()
    if h % 3 != 0:
        x = torch.concat([
            x,
            torch.zeros((n, w, 3 - (h % 3), c), dtype=x.dtype).to(x.device)
        ],
                         dim=2).contiguous()
        n, w, h, c = x.size()
    x = x.view(n, w, int(h / 3), int(c * 3))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h / 3), int(w / 3), int(c * 9))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class VilaMultimodalProjectorConfig(PretrainedConfig):
    model_type = "v2l_projector"

    def __init__(self, mm_projector_type: str = None, **kwargs):
        super().__init__()
        self.mm_projector_type = mm_projector_type


class VilaMultimodalProjector(PreTrainedModel):
    config_class = VilaMultimodalProjectorConfig

    def __init__(self, mm_projector_cfg: VilaMultimodalProjectorConfig,
                 config: PretrainedConfig):
        super().__init__(mm_projector_cfg)
        mm_projector_type = mm_projector_cfg.mm_projector_type
        self.downsample_rate = 1
        if mm_projector_type == "identity":
            self.layers = IdentityMap()
        elif mm_projector_type == "linear":
            self.layers = nn.Linear(config.mm_hidden_size, config.hidden_size)
        elif mm_projector_type == "mlp_downsample":
            self.layers = nn.Sequential(
                DownSampleBlock(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        elif mm_projector_type == "mlp_downsample_2x2_fix":
            self.layers = nn.Sequential(
                DownSample2x2BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 2
        elif mm_projector_type == "mlp_downsample_3x3_fix":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 3),
                nn.Linear(config.mm_hidden_size * 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 3
        elif mm_projector_type == "mlp_downsample_3x3_s2":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 3),
                nn.Linear(config.mm_hidden_size * 3, config.mm_hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size),
                nn.Linear(config.mm_hidden_size, config.mm_hidden_size // 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size // 3),
                nn.Linear(config.mm_hidden_size // 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        elif mm_projector_type == "mlp_downsample_3x3_s2_new":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 4),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.mm_hidden_size * 2),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 2),
                nn.Linear(config.mm_hidden_size * 2, config.mm_hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size),
                nn.Linear(config.mm_hidden_size, config.mm_hidden_size // 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size // 3),
                nn.Linear(config.mm_hidden_size // 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mm_projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(
                        nn.Linear(config.hidden_size, config.hidden_size))
                self.layers = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type: {mm_projector_type}")

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


"""End of multimodal projector module"""


def infer_stop_tokens(tokenizer) -> List[str]:

    def _maybe_add_sentinel_token(tokenizer) -> None:
        if not hasattr(tokenizer, "sentinel_token"):
            tokenizer.add_tokens([SENTINEL_TOKEN], special_tokens=True)
            tokenizer.sentinel_token = SENTINEL_TOKEN
            tokenizer.sentinel_token_id = tokenizer.convert_tokens_to_ids(
                SENTINEL_TOKEN)

    _maybe_add_sentinel_token(tokenizer)

    # Note: shortcut stop_token_ids with eos_token_id to avoid overcomplicated logic
    return [tokenizer.decode(tokenizer.eos_token_id)]


def init_tokenizer(llm_path: str):
    llm_cfg = AutoConfig.from_pretrained(llm_path)
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        model_max_length=llm_cfg.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=False,
    )

    # Set stop tokens for the tokenizer
    tokenizer.stop_tokens = infer_stop_tokens(tokenizer)
    tokenizer.stop_token_ids = tokenizer.convert_tokens_to_ids(
        tokenizer.stop_tokens)

    # Add media tokens to the tokenizer
    tokenizer.media_tokens = MEDIA_TOKENS
    tokenizer.media_token_ids = {}
    for name, token in MEDIA_TOKENS.items():
        tokenizer.add_tokens([token], special_tokens=True)
        tokenizer.media_token_ids[name] = tokenizer.convert_tokens_to_ids(token)

    return tokenizer


def init_vision_tower(model_name_or_path: str,
                      config: PretrainedConfig) -> PreTrainedModel:
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    if config.resume_path and "radio" not in model_name_or_path:
        assert os.path.exists(
            model_name_or_path
        ), f"Pretrained vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path,
                                                      trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = vision_tower_arch if vision_tower_arch is not None else model_name_or_path

    use_s2 = getattr(config, "s2", False)
    use_dynamic_s2 = getattr(config, "dynamic_s2", False)

    if "intern" in vision_tower_name.lower():
        raise NotImplementedError("Intern vision tower is not Implemented yet")
    elif "radio" in vision_tower_name:
        raise NotImplementedError("Radio vision tower is not Implemented yet")
    elif "clip" in vision_tower_name:
        if use_s2:
            vision_tower = CLIPVisionTowerS2(model_name_or_path, config)
        else:
            vision_tower = CLIPVisionTower(model_name_or_path, config)
    elif "siglip" in vision_tower_name:
        if use_dynamic_s2:
            vision_tower = SiglipVisionTowerDynamicS2(model_name_or_path,
                                                      config)
        elif use_s2:
            vision_tower = SiglipVisionTowerS2(model_name_or_path, config)
        else:
            vision_tower = SiglipVisionTower(model_name_or_path, config)
    else:
        raise ValueError(f"Unknown vision tower: {model_name_or_path}")

    config.mm_hidden_size = (vision_tower.config.hidden_size
                             if not (use_s2 or use_dynamic_s2) else
                             vision_tower.hidden_size)

    return vision_tower


def init_mm_projector(model_type_or_path: str,
                      config: PretrainedConfig) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    if config.resume_path:
        # load from pretrained model
        assert os.path.exists(
            model_type_or_path
        ), f"Pretrained mm projector path {model_type_or_path} does not exist!"
        return VilaMultimodalProjector.from_pretrained(model_type_or_path,
                                                       config)
    else:
        # build from scratch
        mm_projector_cfg = VilaMultimodalProjectorConfig(model_type_or_path)
        mm_projector = VilaMultimodalProjector(mm_projector_cfg, config)
        return mm_projector


def init_llm(
    llm_path: str,
    model_config: ModelConfig[PretrainedConfig],
    attn_implementation=None,
    model_max_length=None,
    *args,
    **kwargs,
) -> PreTrainedModel:
    # Pre-run to resize vocab embedding (see: https://github.com/NVlabs/VILA/blob/86e009759a14eee045c669421128d703227da362/llava/model/builder.py#L137)
    llm_cfg = AutoConfig.from_pretrained(llm_path)
    tokenizer = init_tokenizer(llm_path)
    if llm_cfg.vocab_size != len(tokenizer):
        warnings.warn(
            "LLM's vocab size does not match tokenizer's vocab size! It is likely this multimodal model has extended the vocabulary with extra special tokens, and have used resize_token_embeddings() (https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings) in the PyTorch implementation. Here, the only way is to refresh the word embedding weight and re-save the checkpoint and update vocab size. This is a one-off operation for a given model. Tokenzier is not re-saved, just the LLM checkpoint."
        )
        warnings.warn(
            "Please be patient when the checkpoint is being updated...")
        model_hf = HFAutoModelForCausalLM.from_pretrained(llm_path,
                                                          torch_dtype="auto")
        model_hf.resize_token_embeddings(len(tokenizer))
        warnings.warn(f"Saving to {llm_path} by overwriting...")
        try:
            model_hf.save_pretrained(llm_path)
        except (OSError, PermissionError):
            import tempfile
            llm_path = os.path.join(tempfile.gettempdir(), "vila")
            warnings.warn(
                f"Current checkpoint directory is read-only. Saving to {llm_path} instead."
            )
            model_hf.save_pretrained(llm_path)

    # Real run
    llm_cfg = AutoConfig.from_pretrained(llm_path)
    llm_cfg._attn_implementation = attn_implementation
    llm_cfg.model_max_length = model_max_length
    if model_max_length is not None:
        orig_ctx_len = getattr(llm_cfg, "max_position_embeddings", None)
        model_max_length = getattr(llm_cfg, "model_max_length", None)
        if orig_ctx_len and model_max_length > orig_ctx_len:
            logger.info(
                f"Scaling RoPE from {orig_ctx_len} to {model_max_length}")
            scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
            llm_cfg.rope_scaling = {"type": "linear", "factor": scaling_factor}

    llm_model_config = copy.deepcopy(model_config)
    llm_model_config.pretrained_config = llm_cfg
    llm = AutoModelForCausalLM.from_config(llm_model_config)

    model_config.pretrained_config.hidden_size = llm.config.hidden_size
    return tokenizer, llm, llm_path, llm_cfg.vocab_size


class VilaConfig(LlavaConfig):
    model_type = "llava_llama"
    model_architecture = "LlavaLlamaModel"

    # VILA 2.0 extra configs
    min_tiles: Optional[int] = 1
    max_tiles: Optional[int] = 12


class VilaInputProcessor(InputProcessor):

    def __init__(self, model_config, tokenizer):
        self.model_config = model_config
        llm_path, vision_tower_path, mm_projector_path = get_model_paths(
            self.model_config)
        device = 'cuda'
        self.model_dtype = _convert_dtype(self.model_config.model_dtype)

        self.tokenizer = init_tokenizer(
            llm_path) if tokenizer is None else tokenizer
        self.vision_tower = init_vision_tower(
            vision_tower_path,
            self.model_config).to(device=device,
                                  dtype=torch.float16)  # must be fp16
        self.mm_projector = init_mm_projector(
            mm_projector_path,
            self.model_config).to(device=device,
                                  dtype=torch.float16)  # must be fp16

    def _preprocess(self, input: list[any], processor, processor_kwargs,
                    model_config: PretrainedConfig):
        """Pre-process logic on multimodal data, e.g. resize, patchify, etc."""

        assert isinstance(
            input, list
        ), "Multimodal input data must be a list of Image/Video/Audio, etc."
        image_tensor, block_sizes = process_images(input, processor,
                                                   model_config)
        return image_tensor, block_sizes

    @torch.inference_mode()  # critical to minimize memory footprint
    def _forward(self, model, input, model_config):
        """Model forward logic, usually a nn.Module model."""

        vision_tower, mm_projector = model
        image_tensor, block_sizes = input
        image_tensor = image_tensor.to(vision_tower.dtype)
        if getattr(model_config, "dynamic_s2", False):
            # dynamic S2 logic in https://github.com/NVlabs/VILA/blob/main/llava/model/llava_arch.py::encoder_images()
            if block_sizes is None:
                block_sizes = [None] * len(image_tensor)
            visual_features = vision_tower(image_tensor)
            visual_features, new_block_sizes = merge_features_for_dynamic_s2(
                vision_tower, visual_features, block_sizes)

            visual_features = [
                split_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(visual_features, new_block_sizes)
            ]  # list of B * C * H * W tensors
            visual_features = torch.cat(
                [rearrange(x, "b c h w -> b (h w) c") for x in visual_features],
                dim=0)  # B * N * C
            visual_features = mm_projector(visual_features)
            visual_features = list(
                visual_features.split([
                    block_size[0] * block_size[1]
                    for block_size in new_block_sizes
                ],
                                      dim=0))
            visual_features = [
                merge_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(visual_features, new_block_sizes)
            ]  # list of 1 * C * H * W tensors
            visual_features = [
                rearrange(x, "1 c h w -> (h w) c") for x in visual_features
            ]  # list of N * C tensors
            if all([
                    feature.shape[0] == visual_features[0].shape[0]
                    for feature in visual_features
            ]):
                visual_features = torch.stack(visual_features, dim=0)
        else:
            visual_features = vision_tower(image_tensor)
            visual_features = mm_projector(visual_features)

        return visual_features  # [M, feature_length, hidden_dim], where M is number of multimodal inputs (e.g. images) in the current request

    def _postprocess(self, text_input, mm_input, tokenizer, model_config):
        """Post-process logic on model output, e.g. reorder, merge, etc."""

        num_mm_features, mm_feature_length, mm_hidden_dim = mm_input.shape
        mm_total_length = num_mm_features * mm_feature_length
        assert mm_hidden_dim == model_config.hidden_size, "Multimodal embedding_dim must match model hidden_size"

        input_ids = self.tokenizer(
            text_input, return_tensors="pt").input_ids[0].to(mm_input.device)
        vocab_size = len(self.tokenizer)  # vocab including special tokens

        # find mm token positions in input_ids & split input_ids into segments
        mm_tokens = torch.tensor([*tokenizer.media_token_ids.values()
                                  ]).to(input_ids.device)
        mm_token_positions = torch.where(torch.isin(input_ids, mm_tokens))[0]
        mm_split_positions = torch.cat(
            [mm_token_positions,
             mm_token_positions + 1]).sort().values  # isolate mm token ids

        # prepend & append start/end tokens around multimodal features
        # default tokens, see https://github.com/NVlabs/VILA/blob/main/llava/model/encoders/image/basic.py#L15-L16
        start_tokens, start_ids, start_len = None, None, 0
        end_tokens, end_ids, end_len = "\n", None, 0
        if start_tokens is not None:
            start_ids = torch.tensor(tokenizer(start_tokens).input_ids,
                                     device=mm_input.device)
            start_len = len(start_ids)
        if end_tokens is not None:
            end_ids = torch.tensor(tokenizer(end_tokens).input_ids,
                                   device=mm_input.device)
            end_len = len(end_ids)

        # replace mm token ids with expanded out-of-vocab ids
        input_ids_splits = list(input_ids.tensor_split(
            mm_split_positions.cpu()))
        mm_ids_splits = list(
            torch.arange(vocab_size,
                         vocab_size + mm_total_length,
                         device=input_ids.device).split(mm_feature_length))
        start_idx = 0 if mm_token_positions[
            0] == 0 else 1  # whether mm token is the start token
        for mm_ids in mm_ids_splits:
            if start_ids is not None:
                mm_ids = torch.cat([start_ids, mm_ids])
            if end_ids is not None:
                mm_ids = torch.cat([mm_ids, end_ids])
            input_ids_splits[start_idx] = mm_ids
            start_idx += 2  # jump 1 text segment & 1 mm segment

        # concat text & mm input_ids, wrap mm feature in prompt tuning config
        fused_input_ids = torch.cat(input_ids_splits).to(
            device=input_ids.device)
        fused_length = len(input_ids) + mm_total_length + num_mm_features * (
            start_len + end_len - 1
        )  # -1 because special token ID itself is replaced
        assert len(
            fused_input_ids
        ) == fused_length, "Fused input_ids length should match the sum of text and multimodal embedding lengths"
        ptuning_config = {
            "prompt_tuning_config": _ptuning_setup(mm_input, fused_input_ids)
        }

        fused_input_ids = fused_input_ids.to(
            torch.int32).tolist()  # must be List[int] for LlmRequest
        return fused_input_ids, ptuning_config

    @torch.inference_mode()  # critical to minimize memory footprint
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """
        Input processor will only process 1 prompt at a time (Note: "prompt" here is a general class that contains text prompt (1 text) and multimodal prompt (1 or N images). See TextPrompt class in inputs/data.py). The mechanism between InputProcessor and LLM is:

        (1) InputProcessor run vision model. TODO: allow batch processing for vision model, otherwise it can only batch process N images in a single prompt, but cannot batch process N images across multiple prompts

        (2) Processed multimodal features is saved in prompt tuning config as a single torch.Tensor, i.e. M features will be stacked. Note that text prompt is "<text1><image><text2><video><image>...", tokenized ID is [token_1, ..., token_x, image_token, token_y, ..., token_z, video_token, image_token, ...], where <image> and <video> are multimodal special tokens that tokenized as a single token ID like image_token and video_token. Each text [token_i] maps to a length-1 embedding vector, but each multimodal token [image_token or video_token] maps to a length-N embedding tensor. Therefore the input embedding is a interleaved/fused tensor of text embed & multimodal features. The way we handle this is by providing a expanded input_ids and a packed feature tensor to the LLM:

        Assume the total length of multimodal features is L = sum_{i in 1 to M}(feature_len_i).

        (i) expand the input_ids to match the fused embedding shape, e.g. [token_1, ..., token_x, (image_token), token_y, ..., token_z, (video_token), (image_token), ...] expands to [token_1, ..., token_x, (image1_token_1, ..., image1_token_i), token_y, ..., token_z, (video1_token_1, ..., video1_token_i), (image2_token_1, ..., image2_token_i), ...], where the expanded tokens start from out-of-vocabulary IDs, i.e. (image1_token_1, ..., image1_token_i, video1_token_1, ..., video1_token_i, image2_token_1, ..., image2_token_i) is arange(vocab_size, vocab_size + L). See _postprocess().

        (ii) packed multimodal feature, concatenate M features into a single tensor mm_embed.

        (iii) later in LLM, we can slice-and-assign based on condition fused_embed[input_ids < vocab_size] = text_embed, fused_embed[inputI > vocab_size] = mm_embed.

        (3) passed input_ids and mm_embed via LlmRequest's prompt_token_ids and prompt_embedding_table fields respectively. LlmRequests can be inflight batched, and the mm_embed is passed to LLM model as `multi_modal_data` which is List[torch.Tensor] for batched requests.
        """

        text_prompt, mm_data, mm_processor_kwargs = inputs.get(
            "prompt"), inputs.get("multi_modal_data"), inputs.get(
                "mm_processor_kwargs")

        mm_in = self._preprocess(mm_data,
                                 self.get_vision_tower().image_processor,
                                 mm_processor_kwargs, self.model_config)

        mm_out = self._forward(
            (self.get_vision_tower(), self.get_mm_projector()), mm_in,
            self.model_config)

        input_ids, ptuning_config = self._postprocess(text_prompt, mm_out,
                                                      self.tokenizer,
                                                      self.model_config)

        return input_ids, ptuning_config

    def get_vision_tower(self):
        return self.vision_tower

    def get_mm_projector(self):
        return self.mm_projector


@register_auto_model(VilaConfig.model_architecture)
@register_input_processor(VilaInputProcessor)
class VilaModel(PreTrainedModel):
    config_class = VilaConfig

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs) -> None:
        config = model_config.pretrained_config
        super().__init__(config)

        # ModelConfig is TRT-LLM config class w/ pretrained_cfg and quant_cfg
        # pretrained_cfg is HuggingFace config class
        # To compat with TRT-LLM AutoModelForCausalLM class, we need ModelConfig class
        self.model_config = model_config

        if hasattr(self, "llm"):
            return

        self.model_dtype = _convert_dtype(
            getattr(config, "model_dtype", "torch.float16"))
        if not hasattr(config, "model_dtype"):
            warnings.warn(
                "model_dtype not found in config, defaulting to torch.float16.")
        config.model_dtype = self.model_dtype

        self.llm_path, _, _ = get_model_paths(config)
        self.tokenizer, self.llm, self.llm_path, self.vocab_size = init_llm(
            self.llm_path, model_config, *args, **kwargs
        )  # self.llm_path may be updated if ckpt re-saving is needed & existing path is read-only
        device = kwargs.get("device", "cuda")
        self.llm.to(device=device, dtype=self.model_dtype)

        self.post_config()
        self.is_loaded = True

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        VLM forward logic with inflight batching support.
        """

        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        mm_embed = kwargs.get("multi_modal_data", [])

        logger.debug(
            f"num_context_requests: {num_context_requests}, num_generation_requests: {num_generation_requests}"
        )
        assert mm_embed == [] or len(
            mm_embed
        ) == num_context_requests, "Number of multimodal features (if provided) should be equal to number of context requests"

        input_ids, inputs_embeds = self._fuse_input_embeds(input_ids, mm_embed)
        output_prob = self.llm.forward(attn_metadata, input_ids, position_ids,
                                       inputs_embeds, return_context_logits)

        logger.debug(
            f"output_ids: {(output_prob if output_prob.dim() == 2 else output_prob.unsqueeze(0)).argmax(dim=1).tolist()}"
        )
        return output_prob

    def _fuse_input_embeds(
        self,
        input_ids: torch.LongTensor,
        mm_embeds: List[torch.Tensor],
    ) -> Tuple[Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """
        Fuse text and multimodal embeddings. input_ids is [text_total_length + mm_total_length] and mm_embed is [mm_total_length, hidden_dim]. We just need to fuse them into [text_total_length + mm_total_length, hidden_dim] by slice-and-assign to the corresponding entries.

        Args:
            input_ids: shape [text_total_length + mm_total_length], flattened from List[(text_length1 + mm_total_length1), ..., (text_lengthi + mm_total_lengthi)]. For LLM model, the requests are inflight batched together, but the input_ids are flattened with padding removed. By the slice condition < vocab_size, we can easily separate text / multimodal tokens and naturally batched the LLM embedding lookup
            mm_embed: List[(mm_total_length1, hidden_dim), ..., (mm_total_lengthi, hidden_dim)].
        Returns:
            - If (1) JIT test run, (2) non-multimodal run, i.e. all text-only requests, either context or generation phase (3) multimodal run, all requests in generation phase --> there is no multimodal data, return only the input_ids
            - If (4) multimodal run, mixed batch of context and generation requests, each context request has a multimodal feature --> return only the fused input_embeds of shape [total length, hidden_dim]. For text tokens, LLM embedding layer has already run.
        """
        if len(mm_embeds) == 0:
            return input_ids, None

        mm_embed = torch.cat(mm_embeds, dim=0)
        input_embeds = torch.empty(input_ids.shape[0],
                                   mm_embed.shape[-1],
                                   device=input_ids.device,
                                   dtype=self.model_dtype)

        text_token_indices = torch.where(
            input_ids < self.vocab_size
        )[0]  # these indices are text tokens in context requests (discreate, interleaved by multimodal tokens) & text tokens in generation requests (continuous, each request has a single token)
        mm_token_indices = torch.where(input_ids >= self.vocab_size)[0]

        text_embed = self.llm.model.embed_tokens(input_ids[text_token_indices])
        input_embeds[text_token_indices, :] = text_embed.to(self.model_dtype)
        input_embeds[mm_token_indices, :] = mm_embed.to(self.model_dtype)

        return None, input_embeds.to(self.dtype)

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    @property
    def llm_checkpoint_dir(self):
        """Return the directory of the LLM checkpoint. Workaround for multimodal models that config is in root directory but llm checkpoint is in subdirectory. Without this, PyTorchModelEngine will try to load LLM checkpoint from root."""
        return self.llm_path

    def load_weights(self, weights):
        self.llm.load_weights(weights)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config


AutoConfig.register(VilaConfig.model_type, VilaConfig)
AutoModel.register(VilaConfig, VilaModel)
