# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Native preprocessing for DeepSeek-V4.1 formatted text and image prompts."""

import math
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from tensorrt_llm.inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ExtraProcessedInputs,
    TextPrompt,
)
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.tokenizer import TransformersTokenizer

__all__ = ["DeepseekV41InputProcessor", "IMAGE_PLACEHOLDER"]

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
_IMAGE_START, _IMAGE, _IMAGE_NEW_LINE, _IMAGE_END = range(4)


def _num_image_tokens(height: int, width: int) -> int:
    return height * (width + 1) + 2


def _plan_image_grid(
    width: int,
    height: int,
    patch_size: int,
    downsample_ratio: int,
    max_image_tokens: int,
    min_pixels: int,
    max_wh_ratio: float | None,
) -> tuple[int, int, int, int]:
    """Return (LLM rows, LLM columns, pixel height, pixel width)."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if width * height < min_pixels:
        ratio = math.sqrt(min_pixels / (width * height))
        width, height = int(width * ratio), int(height * ratio)

    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    llm_height = math.ceil((best_height // patch_size) / downsample_ratio)
    llm_width = math.ceil((best_width // patch_size) / downsample_ratio)
    if _num_image_tokens(llm_height, llm_width) > max_image_tokens:
        aspect_ratio = height / width
        max_width = math.sqrt((max_image_tokens - 2) / aspect_ratio + 0.25) - 0.5
        max_height = max_width * aspect_ratio
        cell = patch_size * downsample_ratio
        if max_width < 1:
            best_height, best_width = (max_image_tokens - 2) // 2 * cell, cell
        elif max_height < 1:
            best_height, best_width = cell, (max_image_tokens - 3) * cell
        else:
            ratio = min(
                math.floor(max_width) * cell / width,
                math.floor(max_height) * cell / height,
            )
            best_height = math.floor(height * ratio / patch_size) * patch_size
            best_width = math.floor(width * ratio / patch_size) * patch_size
        llm_height = math.ceil((best_height // patch_size) / downsample_ratio)
        llm_width = math.ceil((best_width // patch_size) / downsample_ratio)
    return llm_height, llm_width, best_height, best_width


def _image_token_types(height: int, width: int) -> torch.Tensor:
    return torch.tensor(
        [_IMAGE_START] + ([_IMAGE] * width + [_IMAGE_NEW_LINE]) * height + [_IMAGE_END],
        dtype=torch.int64,
    )


class DeepseekV41InputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    """Expand image placeholders in already formatted prompts.

    Accepts decoded PIL, HWC NumPy, or CHW Torch images, paired with one
    ``<｜deepseek_image｜>`` placeholder per image in prompt order.

    Args:
        model_path: Checkpoint directory, used to load a tokenizer if necessary.
        config: Checkpoint configuration with nested text and vision configs.
        tokenizer: Tokenizer to use, or None to load it from the checkpoint.
        trust_remote_code: Whether tokenizer loading may execute checkpoint code.
    """

    supports_token_id_mm_expansion = True

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizerBase | None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path, config, tokenizer, trust_remote_code, **kwargs)
        if tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, use_fast=self.use_fast
            )
        self._image_token_id = getattr(config, "image_token_id", 129264)
        vision_config = getattr(config, "vision_config", None)
        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig(**vision_config)
        self._vision_enabled = getattr(vision_config, "num_hidden_layers", 0) > 0
        self._patch_size = getattr(vision_config, "patch_size", 14)
        self._downsample_ratio = getattr(vision_config, "downsample_ratio", 3)
        self._max_image_tokens = getattr(vision_config, "max_image_tokens", 1024)
        self._min_pixels = getattr(vision_config, "min_pixels", 295936)
        self._max_wh_ratio = getattr(vision_config, "max_wh_ratio", None)
        if self._patch_size <= 0 or self._downsample_ratio <= 0:
            raise ValueError("Vision patch_size and downsample_ratio must be positive")
        if self._max_image_tokens < 4 or self._min_pixels < 0:
            raise ValueError(
                "Vision max_image_tokens must be at least 4 and min_pixels nonnegative"
            )
        if self._max_wh_ratio is not None and self._max_wh_ratio <= 0:
            raise ValueError("Vision max_wh_ratio must be positive")

        hf_tokenizer = (
            self.tokenizer.tokenizer
            if isinstance(self.tokenizer, TransformersTokenizer)
            else self.tokenizer
        )
        placeholder_id = hf_tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if placeholder_id not in (None, hf_tokenizer.unk_token_id, self._image_token_id):
            raise ValueError(
                f"Tokenizer image placeholder ID {placeholder_id} does not match "
                f"config.image_token_id {self._image_token_id}"
            )

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def processor(self) -> None:
        """The native image path does not use AutoProcessor."""
        return None

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def get_vocab_size(self) -> int:
        text_config = getattr(self.config, "text_config", self.config)
        if isinstance(text_config, dict):
            return int(text_config["vocab_size"])
        return int(text_config.vocab_size)

    def get_mm_token_ids(self) -> torch.Tensor:
        """All image positions, including learned delimiters, receive embeddings."""
        return torch.tensor([self._image_token_id], dtype=torch.int32)

    def _dummy_grid(self, budget: int | None) -> tuple[int, int]:
        """Maximize encoder patches within the patch budget and LLM span limit."""
        ratio = self._downsample_ratio
        if budget is None:
            budget = ratio**2 * (self._max_image_tokens - 3)
        if budget <= 0:
            raise ValueError("max_num_encoder_tokens must be positive")
        best = (1, 1)
        max_height = min(budget, ratio * ((self._max_image_tokens - 2) // 2))
        for height in range(1, max_height + 1):
            rows = math.ceil(height / ratio)
            width = min(budget // height, ratio * ((self._max_image_tokens - 2) // rows - 1))
            if height * width > best[0] * best[1]:
                best = (height, width)
        return best

    def get_mm_max_tokens_per_item(
        self, max_num_encoder_tokens: int | None = None
    ) -> dict[str, int]:
        if not self._vision_enabled:
            return {}
        height, width = self._dummy_grid(max_num_encoder_tokens)
        return {"image": height * width}

    def get_dummy_mm_data(
        self,
        *,
        max_num_encoder_tokens: int,
        mm_counts: Mapping[str, int],
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        if set(mm_counts) - {"image"}:
            raise ValueError("DeepSeek-V4.1 supports only the image modality")
        count = mm_counts.get("image", 0)
        if count < 0:
            raise ValueError("Image count must be nonnegative")
        if count == 0:
            return {}
        height, width = self._dummy_grid(max_num_encoder_tokens)
        if count * height * width > max_num_encoder_tokens:
            raise ValueError("Dummy images exceed the encoder token budget")
        rows, columns = (
            math.ceil(height / self._downsample_ratio),
            math.ceil(width / self._downsample_ratio),
        )
        length = _num_image_tokens(rows, columns)
        return {
            "image": {
                "pixel_values": [
                    torch.zeros(
                        height * width,
                        3,
                        self._patch_size,
                        self._patch_size,
                        dtype=dtype or self.dtype,
                    )
                    for _ in range(count)
                ],
                "image_grid_hw": [(height, width)] * count,
                "llm_grid_hw": [(rows, columns)] * count,
                "num_tokens": [length] * count,
                "token_types": [_image_token_types(rows, columns) for _ in range(count)],
                "positions": [index * length for index in range(count)],
            }
        }

    def _plan_grid(self, width: int, height: int) -> tuple[int, int, int, int]:
        return _plan_image_grid(
            width,
            height,
            self._patch_size,
            self._downsample_ratio,
            self._max_image_tokens,
            self._min_pixels,
            self._max_wh_ratio,
        )

    def get_num_tokens_per_image(
        self, *, image: Image.Image | torch.Tensor | np.ndarray, **kwargs: Any
    ) -> int:
        """Return the complete span length, including start, row breaks, and end."""
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            width, height = image.size
        llm_height, llm_width, _, _ = self._plan_grid(width, height)
        return _num_image_tokens(llm_height, llm_width)

    def _process_image(
        self, image: Image.Image | torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        if isinstance(image, (torch.Tensor, np.ndarray)):
            image = to_pil_image(image)
        if not isinstance(image, Image.Image):
            raise TypeError("DeepSeek-V4.1 images must be decoded PIL images, arrays, or tensors")
        # Match the reference: no EXIF transpose or alpha compositing.
        image = image.convert("RGB")
        llm_height, llm_width, height, width = self._plan_grid(image.width, image.height)
        if self._max_wh_ratio is not None and image.width >= self._max_wh_ratio * image.height:
            image = image.resize((width, height))
        else:
            image = ImageOps.pad(image, (width, height), color=(127, 127, 127))
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
        pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
        patch = self._patch_size
        vit_height, vit_width = height // patch, width // patch
        patches = (
            pixels.reshape(3, vit_height, patch, vit_width, patch)
            .permute(1, 3, 0, 2, 4)
            .reshape(vit_height * vit_width, 3, patch, patch)
        )
        return patches, (vit_height, vit_width), (llm_height, llm_width)

    def get_text_with_mm_placeholders(self, mm_counts: dict[str, int]) -> str:
        """Return one placeholder per image."""
        if any(count for modality, count in mm_counts.items() if modality != "image"):
            raise ValueError("DeepSeek-V4.1 supports only the image modality")
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def expand_prompt_token_ids_for_mm(
        self,
        prompt_token_ids: list[int],
        num_mm_tokens_per_placeholder: list[int],
        *,
        hf_processor_mm_kwargs: dict[str, Any] | None = None,
        mm_data: dict[str, Any] | None = None,
    ) -> tuple[list[int], dict[str, dict[str, Any]]]:
        """Expand image spans without decoding or retokenizing the text."""
        num_placeholders = prompt_token_ids.count(self._image_token_id)
        if num_placeholders != len(num_mm_tokens_per_placeholder):
            raise ValueError(
                f"Found {num_placeholders} image tokens but got "
                f"{len(num_mm_tokens_per_placeholder)} images"
            )
        expanded_ids, positions = [], []
        image_lengths = iter(num_mm_tokens_per_placeholder)
        for token in prompt_token_ids:
            if token == self._image_token_id:
                positions.append(len(expanded_ids))
                expanded_ids.extend([token] * next(image_lengths))
            else:
                expanded_ids.append(token)
        return expanded_ids, {"image": {"positions": positions}}

    def call_with_token_ids(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> tuple[list[int], ExtraProcessedInputs | None]:
        """Expand image spans without retokenizing text IDs."""
        return self._process_prompt(list(inputs["prompt_token_ids"]), inputs)

    def call_with_text_prompt(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> tuple[list[int], ExtraProcessedInputs | None]:
        """Tokenize a formatted prompt and replace each image placeholder."""
        if inputs.get("prompt") is None:
            return self.call_with_token_ids(inputs, sampling_params)
        add_special_tokens = sampling_params.add_special_tokens if sampling_params else True
        token_ids = self.tokenizer.encode(inputs["prompt"], add_special_tokens=add_special_tokens)
        return self._process_prompt(token_ids, inputs)

    def _process_prompt(
        self, token_ids: list[int], inputs: TextPrompt
    ) -> tuple[list[int], ExtraProcessedInputs | None]:
        mm_data = inputs.get("multi_modal_data") or {}
        if any(modality != "image" for modality in mm_data):
            raise ValueError("DeepSeek-V4.1 supports only the image modality")
        if inputs.get("mm_processor_kwargs"):
            raise ValueError("DeepSeek-V4.1 does not support per-request image processor overrides")
        images = mm_data.get("image", [])
        if not isinstance(images, list):
            images = [images]
        num_placeholders = token_ids.count(self._image_token_id)
        if num_placeholders != len(images):
            raise ValueError(f"Found {num_placeholders} image tokens but got {len(images)} images")
        if not images:
            return token_ids, {}
        if not self._vision_enabled:
            raise ValueError("The model config has no vision tower but the prompt contains images")

        pixels, image_grids, llm_grids, token_types, num_tokens = [], [], [], [], []
        for image in images:
            patches, image_grid, llm_grid = self._process_image(image)
            types = _image_token_types(*llm_grid)
            pixels.append(patches)
            image_grids.append(image_grid)
            llm_grids.append(llm_grid)
            token_types.append(types)
            num_tokens.append(types.numel())
        expanded_ids, updates = self.expand_prompt_token_ids_for_mm(token_ids, num_tokens)
        return expanded_ids, {
            "multimodal_data": {
                "image": {
                    "pixel_values": pixels,
                    "image_grid_hw": image_grids,
                    "llm_grid_hw": llm_grids,
                    "token_types": token_types,
                    "num_tokens": num_tokens,
                    "positions": updates["image"]["positions"],
                }
            }
        }
