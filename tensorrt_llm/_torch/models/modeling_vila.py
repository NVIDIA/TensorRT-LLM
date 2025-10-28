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
import dataclasses
import math
import os
import re
import warnings
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import repo_exists, snapshot_download
from huggingface_hub.utils import HFValidationError
from PIL import Image
from transformers import (AutoConfig, AutoImageProcessor, AutoModel,
                          AutoTokenizer, LlavaConfig, PretrainedConfig,
                          PreTrainedModel)

from ..._utils import nvtx_range
from ...inputs import (ExtraProcessedInputs, InputProcessor,
                       MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..modules.embedding import Embedding, LMHead
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_encoder import (VisionTower, VisionTowerDynamicS2,
                                          VisionTowerS2)
from .modeling_multimodal_utils import (dynamic_preprocess_dispatch,
                                        dynamic_s2_preprocess_dispatch,
                                        fuse_input_embeds, merge_chessboard,
                                        merge_features_for_dynamic_s2,
                                        preprocess_dispatch, split_chessboard)
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


def _get_model_paths(config):
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


def process_image(image: Union[Image.Image, torch.Tensor],
                  image_processor,
                  model_config,
                  enable_dynamic_res=False,
                  enable_dynamic_s2=False,
                  use_fast: bool = True,
                  device=None,
                  dtype=None):

    image_aspect_ratio = model_config.image_aspect_ratio
    if hasattr(image_processor, "size"):
        crop_size = image_processor.size  # SIGLIP vision tower
    elif hasattr(image_processor, "crop_size"):
        crop_size = image_processor.crop_size  # CLIP vision tower

    if image_aspect_ratio == "dynamic" and enable_dynamic_res:
        # VILA 2.0
        assert crop_size["height"] == crop_size["width"]
        return dynamic_preprocess_dispatch(image,
                                           image_processor,
                                           min_num=model_config.min_tiles,
                                           max_num=model_config.max_tiles,
                                           image_size=crop_size["height"],
                                           use_fast=use_fast,
                                           device=device,
                                           dtype=dtype)

    if image_aspect_ratio == "dynamic_s2" and enable_dynamic_s2:
        # VILA 2.0
        assert crop_size["height"] == crop_size["width"]
        if type(model_config.s2_scales) is str:
            s2_scales = list(map(int, model_config.s2_scales.split(",")))
        else:
            s2_scales = model_config.s2_scales
        return dynamic_s2_preprocess_dispatch(image,
                                              image_processor,
                                              s2_scales=s2_scales,
                                              max_num=model_config.max_tiles,
                                              image_size=crop_size["height"],
                                              use_fast=use_fast,
                                              device=device,
                                              dtype=dtype)

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

    image = preprocess_dispatch(image,
                                image_processor,
                                use_fast=use_fast,
                                device=device,
                                dtype=dtype)

    return image


def process_images(images: Union[List[Image.Image], List[torch.Tensor]],
                   image_processor,
                   model_config,
                   enable_dynamic_res=False,
                   enable_dynamic_s2=False,
                   use_fast: bool = True,
                   device=None,
                   dtype=None):
    if isinstance(images[0], torch.Tensor) and not use_fast:
        use_fast = True
        logger.warning(
            "Images are given as Pytorch tensors. Automatically turn on use_fast=True because PIL operations don't apply to Pytorch format."
        )

    images = [
        process_image(image, image_processor, model_config, enable_dynamic_res,
                      enable_dynamic_s2, use_fast, device, dtype)
        for image in images
    ]

    block_sizes = None
    if isinstance(images[0], tuple):
        # VILA 2.0 dynamic S2 has block_sizes parameter
        images, block_sizes = zip(*images)
        block_sizes = list(block_sizes)

    if all(x.shape[1:] == images[0].shape[1:] for x in images):
        if len(images[0].shape) == 4:  # [num_tiles, C, H, W]
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

    use_s2 = getattr(config, "s2", False)
    use_dynamic_s2 = getattr(config, "dynamic_s2", False)

    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path,
                                                         use_fast=True)
    if use_dynamic_s2:
        vision_tower = VisionTowerDynamicS2(model_name_or_path, config)
        image_processor.size["height"] = image_processor.size[
            "width"] = vision_tower.scales[0]
    elif use_s2:
        vision_tower = VisionTowerS2(model_name_or_path, config)
        if "clip" in vision_tower.name:
            image_processor.size["shortest_edge"] = vision_tower.scales[-1]
            image_processor.crop_size["height"] = image_processor.crop_size[
                "width"] = vision_tower.scales[-1]
        elif "siglip" in vision_tower.name:
            image_processor.size["height"] = image_processor.size[
                "width"] = vision_tower.scales[-1]
    else:
        vision_tower = VisionTower(model_name_or_path, config)

    config.mm_hidden_size = (vision_tower.config.hidden_size
                             if not (use_s2 or use_dynamic_s2) else
                             vision_tower.hidden_size)

    return vision_tower, image_processor


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


def _resize_embeds(old_embeddings: Embedding, new_num_tokens: int):
    # build new embeddings
    new_embeddings = Embedding(
        num_embeddings=new_num_tokens,
        embedding_dim=old_embeddings.embedding_dim,
        dtype=old_embeddings.weight.dtype,
        mapping=old_embeddings.mapping,
        tensor_parallel_mode=old_embeddings.tp_mode,
        gather_output=old_embeddings.gather_output,
    ).to("cuda")

    # copy weights
    num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
    new_embeddings.weight.data[:
                               num_tokens_to_copy, :] = old_embeddings.weight.data[:
                                                                                   num_tokens_to_copy, :]
    old_embeddings.weight.data = new_embeddings.weight.data
    old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]
    if hasattr(old_embeddings,
               "padding_idx") and old_embeddings.padding_idx is not None:
        if (new_num_tokens - 1) < old_embeddings.padding_idx:
            old_embeddings.padding_idx = None
    return old_embeddings


def _resize_lm_head(old_lm_head: LMHead, new_num_tokens: int):
    # build new lm head
    new_lm_head = LMHead(
        num_embeddings=new_num_tokens,
        embedding_dim=old_lm_head.embedding_dim,
        dtype=old_lm_head.weight.dtype,
        mapping=old_lm_head.mapping,
        tensor_parallel_mode=old_lm_head.tp_mode,
        gather_output=old_lm_head.gather_output,
    ).to("cuda")

    # copy weights
    num_tokens_to_copy = min(old_lm_head.num_embeddings, new_num_tokens)
    new_lm_head.weight.data[:
                            num_tokens_to_copy, :] = old_lm_head.weight.data[:
                                                                             num_tokens_to_copy, :]
    old_lm_head.weight.data = new_lm_head.weight.data

    return old_lm_head


def _resize_token_embeddings(llm, new_num_tokens: int):
    _resize_embeds(llm.model.embed_tokens, new_num_tokens)
    if hasattr(llm, "lm_head"):
        _resize_lm_head(llm.lm_head, new_num_tokens)


def init_llm(
    llm_path: str,
    model_config: ModelConfig[PretrainedConfig],
    attn_implementation=None,
    model_max_length=None,
    *args,
    **kwargs,
) -> PreTrainedModel:
    llm_cfg = AutoConfig.from_pretrained(llm_path)
    tokenizer = init_tokenizer(llm_path)

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
    if llm_cfg.vocab_size != len(tokenizer):
        warnings.warn(
            "LLM have a different vocab size than tokenizer. Consider update the LLM checkpoint with the tokenizer's vocab size with _resize_token_embeddings()."
        )

    model_config.pretrained_config.hidden_size = llm.config.hidden_size
    return tokenizer, llm, llm_path, llm_cfg.vocab_size


class VilaConfig(LlavaConfig):
    model_type = "llava_llama"
    model_architecture = "LlavaLlamaModel"

    # VILA 2.0 extra configs
    min_tiles: Optional[int] = 1
    max_tiles: Optional[int] = 12


class SeparatorStyle(Enum):
    """Different separator style."""

    AUTO = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_3 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    sep_style: Enum
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.LLAMA_3:
            ret = self.system + self.sep
            for rid, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    sep = self.sep if rid < len(messages) - 1 else self.sep2
                    ret += role + message + sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )


conv_auto = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    sep_style=SeparatorStyle.AUTO,
    sep="\n",
)

conv_vicuna_v1 = Conversation(
    system=
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

hermes_2 = Conversation(
    system="<|im_start|>system\nAnswer the questions.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
    messages=(),
    version="hermes-2",
)

# Template added by Yukang. Note (kentang-mit@): sep is <|eot_id|> for official template.
llama_3_chat = Conversation(
    system=
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n",
           "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama_v3",
    messages=(),
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<|eot_id|>",
    sep2="<|end_of_text|>",
)

conv_templates = {
    "auto": conv_auto,
    "hermes-2": hermes_2,
    "llama_3": llama_3_chat,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "plain": conv_llava_plain,
}

CONVERSATION_MODE_MAPPING = {
    "vila1.5-3b": "vicuna_v1",
    "vila1.5-8b": "llama_3",
    "vila1.5-13b": "vicuna_v1",
    "vila1.5-40b": "hermes-2",
    "llama-3": "llama_3",
    "llama3": "llama_3",
}


def _get_conversation_mode(model_name_or_path: str) -> Conversation:
    # CAVEAT: VILA uses pathname-based check which is error prone, consider register with model
    default_conversation = conv_auto
    for k, v in CONVERSATION_MODE_MAPPING.items():
        if k in model_name_or_path.lower():
            logger.info(
                f"Setting conversation mode to `{v}` based on model name/path `{model_name_or_path}`."
            )
            default_conversation = conv_templates[v]
            break
    return default_conversation


def _apply_chat_template(text, conv, tokenizer):
    """Apply VILA-style conversational template to text prompt."""
    text = text.strip()
    if conv.sep_style == SeparatorStyle.AUTO:
        # VILA 2.0
        message = {}
        message["role"] = "user"
        message["content"] = text
        text = tokenizer.apply_chat_template(
            [message],
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        # VILA 1.0
        messages = [{"from": "human", "value": text}]
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Skip the first message if it is not from human
        if messages[0]["from"] != "human":
            messages = messages[1:]

        # Add a generation prompt if needed
        messages.append({"from": "gpt", "value": None})

        conv.messages = []
        for turn, message in enumerate(messages):
            role = roles[message["from"]]
            assert role == conv.roles[turn % 2]
            conv.append_message(role, message["value"])
        text = conv.get_prompt()
    return text


class VilaInputProcessor(InputProcessor):

    def __init__(self,
                 model_path,
                 model_config,
                 tokenizer,
                 trust_remote_code: bool = True):
        self.model_config = model_config
        llm_path, vision_tower_path, mm_projector_path = _get_model_paths(
            self.model_config)
        self.device = 'cuda'
        self.model_dtype = _convert_dtype(self.model_config.model_dtype)
        self.conv_mode = _get_conversation_mode(llm_path)

        self.tokenizer = init_tokenizer(
            llm_path) if tokenizer is None else tokenizer
        self.vision_tower, self.image_processor = init_vision_tower(
            vision_tower_path, self.model_config)
        self.mm_projector = init_mm_projector(mm_projector_path,
                                              self.model_config)

        # must be fp16
        self.vision_tower.to(device=self.device, dtype=torch.float16)
        self.mm_projector.to(device=self.device, dtype=torch.float16)

    @nvtx_range("[Vision] preprocess")
    def _preprocess(self,
                    mm_data: dict[str, any],
                    processor_kwargs,
                    use_fast: bool = True) -> Tuple[torch.Tensor, List[int]]:
        """Preprocess multimodal data, e.g. resize, patchify, etc., into torch.Tensor"""

        assert isinstance(
            mm_data, dict
        ), "Multimodal input data must be a dict of Image/Video/Audio, etc."
        images = []
        if "image" in mm_data and len(mm_data["image"]) > 0:
            images = mm_data["image"]
            return process_images(images,
                                  self.image_processor,
                                  self.model_config,
                                  enable_dynamic_res=True,
                                  enable_dynamic_s2=True,
                                  use_fast=use_fast,
                                  device='cuda',
                                  dtype=self.vision_tower.dtype)
        elif "video" in mm_data and len(mm_data["video"]) > 0:
            video_datas = mm_data["video"]
            videos = [video_data.frames for video_data in video_datas]
            mm_tensors = []
            block_sizes = []
            for video in videos:
                mm_tensor, block_sizes = process_images(
                    video,
                    self.image_processor,
                    self.model_config,
                    enable_dynamic_res=False,
                    enable_dynamic_s2=False,
                    use_fast=use_fast,
                    device='cuda',
                    dtype=self.vision_tower.dtype)
                mm_tensors.append(mm_tensor)
            return torch.cat(mm_tensors, dim=0), block_sizes
        else:
            raise RuntimeError(f"invalid mm_data: {mm_data}")

    @nvtx_range("[Vision] process")
    def _process(self, mm_tensor, block_sizes):
        """Extract multimodal features from multimodal input"""

        mm_tensor = mm_tensor.to(self.vision_tower.dtype)  # must be fp16
        if getattr(self.model_config, "dynamic_s2", False):
            # dynamic S2 logic in https://github.com/NVlabs/VILA/blob/main/llava/model/llava_arch.py::encoder_images()
            if block_sizes is None:
                block_sizes = [None] * len(mm_tensor)
            visual_features = self.vision_tower(mm_tensor)
            visual_features, new_block_sizes = merge_features_for_dynamic_s2(
                self.vision_tower, visual_features, block_sizes)

            visual_features = [
                split_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(visual_features, new_block_sizes)
            ]  # list of B * C * H * W tensors
            visual_features = torch.cat(
                [rearrange(x, "b c h w -> b (h w) c") for x in visual_features],
                dim=0)  # B * N * C
            visual_features = self.mm_projector(visual_features)
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
            visual_features = self.vision_tower(mm_tensor)
            visual_features = self.mm_projector(visual_features)
        return visual_features  # [M, feature_length, hidden_dim], where M is number of multimodal inputs (e.g. images or frames) in the current request; or list of [feature_length_i, hidden_dim] tensors if images have different lengths

    @nvtx_range("[Vision] postprocess")
    def _postprocess(self, input_ids, mm_features):
        """Postprocess multimodal features by fusing text + multimodal information"""

        ## find mm token positions in input_ids
        mm_tokens = torch.tensor([*self.tokenizer.media_token_ids.values()
                                  ]).to(input_ids.device)
        mm_token_positions = torch.where(torch.isin(input_ids, mm_tokens))[0]
        num_medias = num_mm_tokens = len(mm_token_positions)
        if num_medias > 1 and isinstance(mm_features, torch.Tensor):
            mm_features = list(
                mm_features.split(mm_features.shape[0] // num_medias))

        if isinstance(mm_features, torch.Tensor):
            # 1 prompt + 1 media
            # "split" means what a single mm_token in the input_ids should represent
            # image: one split --> one frame
            # video: one split --> N frames
            num_frames, mm_feature_length, mm_hidden_dim = mm_features.shape
            mm_lengths_per_split = [mm_feature_length * num_frames]
            mm_lengths_per_frame = [mm_feature_length]
        elif isinstance(mm_features, list):
            # 1 prompt + N media
            num_frames = len(mm_features) if mm_features[0].dim() == 2 else sum(
                [f.shape[0] for f in mm_features])
            mm_lengths_per_split = [
                f.shape[0] if f.dim() == 2 else f.shape[0] * f.shape[1]
                for f in mm_features
            ]
            mm_lengths_per_frame = [
                f.shape[0] if f.dim() == 2 else f.shape[1] for f in mm_features
            ]
            mm_hidden_dim = mm_features[0].shape[-1]
            mm_features = torch.cat(mm_features, dim=0)
        else:
            raise ValueError(
                f"Invalid multimodal features type: {type(mm_features)}")
        mm_total_length = sum(mm_lengths_per_split)
        assert mm_hidden_dim == self.model_config.hidden_size, "Multimodal embedding_dim must match model hidden_size"

        ## split input_ids into segments by isolating mm tokens
        vocab_size = len(self.tokenizer)  # vocab including special tokens
        mm_split_positions = torch.cat(
            [mm_token_positions, mm_token_positions + 1]).unique()
        input_ids_splits = list(input_ids.tensor_split(mm_split_positions.cpu(
        )))  # len(input_ids_splits) = num_segments after mm tokens are isolated
        mm_ids_splits = list(
            torch.arange(vocab_size,
                         vocab_size + mm_total_length,
                         device=input_ids.device).split(mm_lengths_per_split)
        )  # len(mm_ids_splits) = num_mm_segments

        # prepend & append start/end tokens around mm ids
        # VILA needs to prepend/append default start/end tokens for EACH image/frame, see https://github.com/NVlabs/VILA/blob/main/llava/model/encoders/image/basic.py#L15-L16
        start_tokens, start_ids, start_len = None, None, 0
        end_tokens, end_ids, end_len = "\n", None, 0
        if start_tokens is not None:
            start_ids = torch.tensor(self.tokenizer(start_tokens).input_ids,
                                     device=input_ids.device)
            start_len = len(start_ids)
        if end_tokens is not None:
            end_ids = torch.tensor(self.tokenizer(end_tokens).input_ids,
                                   device=input_ids.device)
            end_len = len(end_ids)

        for i, mm_ids in enumerate(mm_ids_splits):
            mm_ids = mm_ids.reshape(-1, mm_lengths_per_frame[i])
            if start_ids is not None:
                mm_ids = torch.cat(
                    [start_ids.unsqueeze(0).repeat(mm_ids.shape[0], 1), mm_ids],
                    dim=1)
            if end_ids is not None:
                mm_ids = torch.cat(
                    [mm_ids,
                     end_ids.unsqueeze(0).repeat(mm_ids.shape[0], 1)],
                    dim=1)
            mm_ids_splits[i] = mm_ids.flatten()

        ## replace mm token ids with the expanded out-of-vocab ids
        mm_split_idx = 0
        for i, split in enumerate(input_ids_splits):
            if torch.isin(split, mm_tokens).any().item():
                input_ids_splits[i] = mm_ids_splits[mm_split_idx]
                mm_split_idx += 1
        assert mm_split_idx == len(
            mm_ids_splits), "All mm_ids_splits should be consumed"

        ## concat text & mm input_ids, wrap mm feature in prompt tuning config
        fused_input_ids = torch.cat(input_ids_splits).to(
            device=input_ids.device)
        fused_length = len(input_ids) + mm_total_length + num_frames * (
            start_len + end_len) - num_medias
        assert len(
            fused_input_ids
        ) == fused_length, f"Fused input_ids length {len(fused_input_ids)} should match the sum of text and multimodal embedding lengths {fused_length}"

        # [num_frames, feature_length, hidden_dim] -> [num_frames * feature_length, hidden_dim]
        mm_features = mm_features.view(-1, mm_features.shape[-1])
        return fused_input_ids, mm_features

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

        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
        mm_processor_kwargs = inputs.get("mm_processor_kwargs", {})

        text_prompt = _apply_chat_template(text_prompt, self.conv_mode,
                                           self.tokenizer)
        input_ids = self.tokenizer(
            text_prompt, return_tensors="pt").input_ids[0].to(self.device)

        if not mm_data:
            return input_ids.to(torch.int32).tolist(), {}

        mm_tensor, block_sizes = self._preprocess(
            mm_data, mm_processor_kwargs, use_fast=True
        )  # use_fast uses Pytorch GPU preprocessing, otherwise uses PIL CPU preprocessing
        mm_features = self._process(mm_tensor, block_sizes)
        fused_input_ids, mm_features = self._postprocess(input_ids, mm_features)
        multimodal_data = {}
        multimodal_data["multimodal_embedding"] = mm_features
        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data
        }


@register_auto_model(VilaConfig.model_architecture)
@register_input_processor(
    VilaInputProcessor,
    model_type="llava_llama",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<image>",
            "video": "<vila/video>"
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ))
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

        self.llm_path, _, _ = _get_model_paths(config)
        self.tokenizer, self.llm, self.llm_path, self.vocab_size = init_llm(
            self.llm_path, model_config, *args, **kwargs
        )  # self.llm_path may be updated if ckpt re-saving is needed & existing path is read-only
        device = kwargs.get("device", "cuda")
        self.llm.to(device=device, dtype=self.model_dtype)

        self.post_config()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        VLM forward logic with inflight batching support.
        """

        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds = []
        if len(multimodal_params) > 0:
            mm_embeds = [
                multimodal_param.multimodal_data["multimodal_embedding"]
                for multimodal_param in multimodal_params
            ]

        input_ids, inputs_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens, input_ids, mm_embeds, **kwargs)
        logits = self.llm.forward(attn_metadata=attn_metadata,
                                  input_ids=input_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=inputs_embeds,
                                  return_context_logits=return_context_logits)
        return logits

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
        # resize token embeddings if vocab size mismatch after loading weights
        if self.vocab_size != len(self.tokenizer):
            _resize_token_embeddings(self.llm, len(self.tokenizer))
            self.vocab_size = len(self.tokenizer)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config


AutoConfig.register(VilaConfig.model_type, VilaConfig)
AutoModel.register(VilaConfig, VilaModel)
