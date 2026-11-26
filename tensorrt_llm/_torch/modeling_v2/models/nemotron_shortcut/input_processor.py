# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare merge-ordered image patches and placeholder blocks for the Shortcut vision encoder."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.data import TextPrompt, TokensPrompt
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalItemPartition,
)
from tensorrt_llm.inputs.registry import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    DefaultInputProcessor,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
)
from tensorrt_llm.sampling_params import SamplingParams

PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={"image": "<nemo:img>"},
    content_format=ContentFormat.OPENAI,
)
ImageInput = Image.Image | np.ndarray | torch.Tensor


def _size(image: ImageInput) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.height, image.width
    if isinstance(image, torch.Tensor):
        return int(image.shape[-2]), int(image.shape[-1])
    return int(image.shape[0]), int(image.shape[1])


def _unit_rgb(image: ImageInput) -> np.ndarray:
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            rgba = image.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, "white")
            bg.paste(rgba, mask=rgba)
            image = bg.convert("RGB")
        image = np.asarray(image)
    if isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[0] not in (1, 3):
            raise ValueError("An image tensor must have one or three channels in CHW order.")
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)
        hwc = image.detach().cpu().permute(1, 2, 0)
        if image.is_floating_point():
            return hwc.float().numpy()
        image = hwc.numpy()
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("An image array must be uint8 RGB in HWC order.")
    return image.astype(np.float32) / np.float32(255)


class ShortcutTextInputProcessor(DefaultInputProcessor):
    """Tokenize text when the Shortcut vision encoder is disabled."""

    supports_tokenization_cache = True

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizerBase | None,
        *,
        enable_tokenization_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path, config, tokenizer, enable_tokenization_cache=enable_tokenization_cache
        )
        self.multimodal_hashing_supported = False

    def __call__(
        self, inputs: TextPrompt | TokensPrompt, sampling_params: SamplingParams
    ) -> tuple[list[int], ExtraProcessedInputs | None]:
        for items in (inputs.get("multi_modal_data") or {}).values():
            if items is not None and (not isinstance(items, (list, tuple)) or items):
                raise ValueError("Raw media cannot be processed with disable_mm_encoder=True.")
        if "prompt" in inputs:
            return super().__call__(inputs, sampling_params)
        return inputs["prompt_token_ids"], None


class ShortcutInputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    """Prepare native image features and the matching model-visible placeholder blocks."""

    runs_without_gpu = True
    supports_tokenization_cache = True

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizerBase | None,
        trust_remote_code: bool = True,
        enable_tokenization_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path, config, tokenizer, trust_remote_code, **kwargs)
        spec = json.loads((Path(model_path) / "processor_config.json").read_text())
        img = spec["image_processor"]
        self.patch = int(spec["patch_size"])
        self.merge = int(spec["spatial_merge_size"])
        self.unit = self.patch * self.merge
        self.longest = int(img["size"]["longest_edge"])
        if img["patch_size"] != {"height": self.unit, "width": self.unit}:
            raise ValueError("Image processor patch size must match the vision tower and merger.")
        if not (
            img["do_resize"]
            and img["do_rescale"]
            and img["do_normalize"]
            and math.isclose(img["rescale_factor"], 1 / 255)
        ):
            raise ValueError("Unsupported Shortcut image normalization configuration.")
        self.mean = np.asarray(img["image_mean"], dtype=np.float32)
        self.std = np.asarray(img["image_std"], dtype=np.float32)
        self.image_token, self.image_break, self.image_end = (
            spec[k] for k in ("image_token", "image_break_token", "image_end_token")
        )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
        self._tokenizer = tokenizer
        self._init_tokenization_cache(enable_tokenization_cache, self._tokenizer)
        self._token_ids = self._tokenizer.convert_tokens_to_ids(
            [self.image_token, self.image_break, self.image_end]
        )
        if len(set(self._token_ids)) != 3 or self._token_ids[0] != config.image_token_id:
            raise ValueError("The tokenizer image tokens do not match the composite checkpoint.")

    @property
    def processor(self) -> "ShortcutInputProcessor":
        return self

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def dtype(self) -> torch.dtype:
        return self.config.dtype or torch.bfloat16

    @property
    def supported_modalities(self) -> frozenset[str]:
        return frozenset(("image",))

    def get_preferred_media_io_kwargs(self) -> dict[str, dict[str, Any]]:
        return {"image": {"format": "pil"}}

    def _resized(self, image: ImageInput) -> tuple[int, int]:
        height, width = _size(image)
        if min(height, width) <= 0:
            raise ValueError("Image dimensions must be positive.")
        ratio = max(height / self.longest, width / self.longest, 1.0)
        return tuple(
            max(1, math.ceil(round(x / ratio) / self.unit)) * self.unit for x in (height, width)
        )

    def preprocess_image(self, image: ImageInput) -> tuple[torch.Tensor, tuple[int, int]]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "Shortcut media preprocessing requires OpenCV. Install opencv-python-headless."
            ) from exc

        height, width = self._resized(image)
        resized = cv2.resize(_unit_rgb(image), (width, height), interpolation=cv2.INTER_AREA)
        hwc = (resized - self.mean) / self.std
        p, m = self.patch, self.merge
        # Tiles precede patches and each patch uses the flattened convolution's C,H,W order.
        patches = hwc.reshape(height // self.unit, m, p, width // self.unit, m, p, 3)
        patches = patches.transpose(0, 3, 1, 4, 6, 2, 5)
        packed = torch.from_numpy(np.ascontiguousarray(patches)).reshape(-1, 3 * p * p)
        return packed.to(torch.bfloat16), (height, width)

    def _image_block(self, height: int, width: int) -> str:
        row = self.image_token * (width // self.unit)
        return (row + self.image_break) * (height // self.unit - 1) + row + self.image_end

    def get_vocab_size(self) -> int:
        return self.config.get_text_config().vocab_size

    def get_mm_token_ids(self) -> torch.Tensor:
        return torch.tensor(self._token_ids)

    def get_mm_special_token_ids(self) -> torch.Tensor:
        return torch.tensor(self._token_ids[1:])

    def get_num_tokens_per_image(self, *, image: ImageInput, **kwargs: Any) -> int:
        h, w = self._resized(image)
        return h // self.unit * (w // self.unit + 1)

    def _payload(self, patches: torch.Tensor, sizes: list[tuple[int, int]]) -> dict:
        lengths = tuple(h // self.patch * (w // self.patch) for h, w in sizes)
        metadata = MultimodalItemMetadata(
            [("image", i) for i in range(len(sizes))],
            encoder_token_lengths=lengths,
            output_embedding_lengths=[length // self.merge**2 for length in lengths],
            input_partitions={
                "image": {
                    "patches": MultimodalItemPartition(item_lengths=lengths),
                    "image_sizes": MultimodalItemPartition(),
                }
            },
        )
        return {
            "image": {"patches": patches, "image_sizes": sizes},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
        }

    def call_with_text_prompt(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> tuple[list[int], ExtraProcessedInputs | None]:
        prompt = inputs["prompt"]
        media = inputs.get("multi_modal_data") or {}
        if unknown := media.keys() - self.supported_modalities:
            raise ValueError(f"Unsupported Shortcut media: {sorted(unknown)}.")
        images = media.get("image", [])
        if not isinstance(images, (tuple, list)):
            images = [images]
        if not images and self.image_token not in prompt:
            token_ids = self._encode_with_tokenization_cache(prompt, sampling_params)
            if token_ids is not None:
                return token_ids, None
        pieces = prompt.split(self.image_token)
        if len(pieces) != len(images) + 1:
            raise ValueError("Each image requires one corresponding prompt placeholder.")
        expanded, patches, sizes = [pieces[0]], [], []
        for image, suffix in zip(images, pieces[1:], strict=True):
            rows, size = self.preprocess_image(image)
            patches.append(rows)
            sizes.append(size)
            expanded.extend((self._image_block(*size), suffix))
        options = {}
        if sampling_params.truncate_prompt_tokens is not None:
            options.update(truncation=True, max_length=sampling_params.truncate_prompt_tokens)
        ids = self.tokenizer.encode(
            "".join(expanded), add_special_tokens=sampling_params.add_special_tokens, **options
        )
        if not images:
            return ids, None
        payload = self._payload(torch.cat(patches), sizes)
        expected = sum(payload[MULTIMODAL_ENCODER_ITEM_METADATA_KEY].output_embedding_lengths)
        if ids.count(self._token_ids[0]) != expected:
            raise ValueError("The tokenized prompt must retain all image embedding positions.")
        return ids, {"multimodal_data": payload}

    def get_mm_max_tokens_per_item(
        self, max_num_encoder_tokens: int | None = None
    ) -> dict[str, int]:
        maximum = (math.ceil(self.longest / self.unit) * self.merge) ** 2
        if max_num_encoder_tokens is not None:
            maximum = min(maximum, max_num_encoder_tokens // self.merge**2 * self.merge**2)
        return {"image": maximum} if maximum else {}

    def get_max_mm_encoder_output_embeddings(self, max_num_encoder_tokens: int) -> int:
        return max_num_encoder_tokens // self.merge**2

    def get_mm_encoder_attention_metadata_capacity(self, max_num_tokens: int) -> dict[str, int]:
        return {"attention": max(1, max_num_tokens // self.merge**2)}

    def get_dummy_mm_data(
        self,
        *,
        max_num_encoder_tokens: int,
        mm_counts: Mapping[str, int],
        dtype: torch.dtype | None = None,
    ) -> dict:
        if mm_counts.keys() - self.supported_modalities or mm_counts.get("image", 0) < 0:
            raise ValueError("Unsupported modalities or negative counts for encoder profiling.")
        count = mm_counts.get("image", 0)
        if not count:
            return {}
        tile = self.merge**2
        edge = min(
            math.ceil(self.longest / self.unit),
            math.isqrt(max_num_encoder_tokens // (count * tile)),
        )
        if edge == 0:
            raise ValueError("Encoder budget is too small to profile the requested images.")
        size = edge * self.unit
        rows = torch.zeros(
            (count * edge**2 * tile, 3 * self.patch**2), dtype=dtype or torch.bfloat16
        )
        return self._payload(rows, [(size, size)] * count)
