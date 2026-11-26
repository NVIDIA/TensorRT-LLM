# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare merge-ordered image patches and placeholder blocks for the Shortcut vision encoder."""

import json
import math
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.data import TextPrompt, TokensPrompt
from tensorrt_llm.inputs.evs import compute_retained_tokens_from_tubelet_budget
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalItemPartition,
)
from tensorrt_llm.inputs.multimodal_data import VideoData
from tensorrt_llm.inputs.registry import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    DefaultInputProcessor,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
)
from tensorrt_llm.sampling_params import SamplingParams

_VIDEO_PLACEHOLDER = "<nemo:video>"
_MAX_VIDEO_FRAMES = 64
_DEFAULT_VIDEO_FPS = 2

PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={"image": "<nemo:img>", "video": _VIDEO_PLACEHOLDER},
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
        video_pruning_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path, config, tokenizer, trust_remote_code, **kwargs)
        self.video_pruning_rate = video_pruning_rate or 0.0
        # The executor passes a flattened media config. Read the checkpoint's video capability.
        config_path = Path(model_path) / "config.json"
        temporal_patch_size = getattr(config, "video_temporal_patch_size", None)
        if temporal_patch_size is None and config_path.is_file():
            temporal_patch_size = json.loads(config_path.read_text()).get(
                "video_temporal_patch_size"
            )
        self._supports_video = temporal_patch_size == 1
        if self.video_pruning_rate and "video" not in self.supported_modalities:
            raise ValueError("EVS requires a checkpoint with a single-frame video processor.")
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
        if self._supports_video:
            return frozenset(("image", "video"))
        return frozenset(("image",))

    def get_preferred_media_io_kwargs(self) -> dict[str, dict[str, Any]]:
        options = {"image": {"format": "pil"}}
        if "video" in self.supported_modalities:
            options["video"] = {
                "format": "np",
                "num_frames": _MAX_VIDEO_FRAMES,
                "fps": _DEFAULT_VIDEO_FPS,
            }
        return options

    @cached_property
    def _video_prompt_processor(self):
        return AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=self._trust_remote_code
        )

    def _video_layout(self, frames, metadata):
        fps = (metadata or {}).get("fps")
        if fps is None or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video timestamps require a positive source FPS.")
        # Raw clips carry source indices from VideoMediaIO. Direct frame sequences
        # without indices are interpreted as consecutive frames at the supplied FPS.
        indices = metadata.get("frames_indices", range(len(frames)))
        if len(frames) == 0 or len(indices) != len(frames):
            raise ValueError("Video frames and their source indices must be nonempty and aligned.")
        size = self._resized(frames[0])
        text = self._video_prompt_processor.get_video_prompt(
            [float(i) / fps for i in indices], temporal_patch_size=1, frame_size=size
        )
        height, width = (edge // self.unit for edge in size)
        if not self.video_pruning_rate:
            return frames, size, text, [], len(frames) * height * (width + 1)
        # The text between the first and last patch includes row ends and timestamps.
        # Its actual positions depend on the encoder's retention mask. Reserve the
        # exact final length and fuse these text embeddings with surviving patches.
        left = text.index(self.image_token)
        right = text.rindex(self.image_end) + len(self.image_end)
        template = self.tokenizer.encode(text[left:right], add_special_tokens=False)
        num_patches = len(frames) * height * width
        retained = compute_retained_tokens_from_tubelet_budget(
            len(frames), height * width, self.video_pruning_rate
        )
        count = len(template) - num_patches + retained
        return (
            frames,
            size,
            text[:left] + self.image_token * count + text[right:],
            template,
            count,
        )

    def get_num_tokens_per_video(self, *, video, video_metadata=None, **kwargs) -> int:
        return self._video_layout(video, video_metadata)[-1]

    def _prepare_videos(self, prompt, videos, sampling_params):
        pieces = prompt.split(_VIDEO_PLACEHOLDER)
        if len(pieces) != len(videos) + 1 or self.image_token in prompt:
            raise ValueError("Each video requires one video placeholder and no image placeholders.")
        expanded, patches, sizes, templates, costs, outputs = [pieces[0]], [], [], [], [], []
        for video, suffix in zip(videos, pieces[1:], strict=True):
            if not isinstance(video, VideoData):
                raise ValueError(
                    "Video inputs require VideoData with decoded frames and source FPS."
                )
            frames, size, text, template, count = self._video_layout(video.frames, video.metadata)
            for frame in frames:
                patch_rows, frame_size = self.preprocess_image(frame)
                if frame_size != size:
                    raise ValueError("All frames of a video must have the same processed size.")
                patches.append(patch_rows)
            cost = len(frames) * (size[0] // self.patch) * (size[1] // self.patch)
            costs.append(cost)
            outputs.append(count if template else cost // self.merge**2)
            expanded.extend((text, suffix))
            sizes.append(size)
            templates.append(template)
        ids = self._encode_media_prompt(expanded, sampling_params)
        if ids.count(self._token_ids[0]) != sum(outputs):
            raise ValueError("The tokenized prompt must retain all video embedding positions.")
        return ids, {
            "multimodal_data": self._video_payload(
                torch.cat(patches), sizes, templates, costs, outputs
            )
        }

    def _video_payload(self, patches, sizes, templates, costs, outputs):
        metadata = MultimodalItemMetadata(
            [("video", i) for i in range(len(sizes))],
            encoder_token_lengths=costs,
            output_embedding_lengths=outputs,
            input_partitions={
                "video": {
                    "patches": MultimodalItemPartition(item_lengths=tuple(costs)),
                    "frame_sizes": MultimodalItemPartition(),
                    "templates": MultimodalItemPartition(),
                    "pruning_rate": MultimodalItemPartition(item_axis=None),
                }
            },
        )
        return {
            "video": {
                "patches": patches,
                "frame_sizes": sizes,
                "templates": templates,
                "pruning_rate": self.video_pruning_rate,
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
        }

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
        images = media.get("image", ())
        if not isinstance(images, (tuple, list)):
            images = [images]
        videos = media.get("video", ())
        if not isinstance(videos, (tuple, list)):
            videos = [videos]
        if videos:
            if images:
                raise ValueError("The checkpoint accepts either images or videos per request.")
            return self._prepare_videos(prompt, videos, sampling_params)
        if _VIDEO_PLACEHOLDER in prompt:
            raise ValueError("A video placeholder requires a video input.")
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
        ids = self._encode_media_prompt(expanded, sampling_params)
        if not images:
            return ids, None
        payload = self._payload(torch.cat(patches), sizes)
        expected = sum(payload[MULTIMODAL_ENCODER_ITEM_METADATA_KEY].output_embedding_lengths)
        if ids.count(self._token_ids[0]) != expected:
            raise ValueError("The tokenized prompt must retain all image embedding positions.")
        return ids, {"multimodal_data": payload}

    def _encode_media_prompt(self, pieces, sampling_params):
        options = {}
        if sampling_params.truncate_prompt_tokens is not None:
            options.update(truncation=True, max_length=sampling_params.truncate_prompt_tokens)
        return self.tokenizer.encode(
            "".join(pieces), add_special_tokens=sampling_params.add_special_tokens, **options
        )

    def get_mm_max_tokens_per_item(
        self, max_num_encoder_tokens: int | None = None
    ) -> dict[str, int]:
        maximum = (math.ceil(self.longest / self.unit) * self.merge) ** 2
        caps = {"image": maximum}
        if "video" in self.supported_modalities:
            caps["video"] = _MAX_VIDEO_FRAMES * maximum
        if max_num_encoder_tokens is None:
            return caps
        budget = max_num_encoder_tokens // self.merge**2 * self.merge**2
        return {key: min(cap, budget) for key, cap in caps.items()} if budget else {}

    def get_max_mm_encoder_output_embeddings(self, max_num_encoder_tokens: int) -> int | None:
        if self.video_pruning_rate:
            # The retained sequence includes variable-length timestamp labels. The checkpoint's
            # inline encoder uses exact item lengths and does not use item scheduling.
            return None
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
        if mm_counts.keys() - self.supported_modalities or any(n < 0 for n in mm_counts.values()):
            raise ValueError("Unsupported modalities or negative counts for encoder profiling.")
        if mm_counts.get("image") and mm_counts.get("video"):
            raise ValueError("The checkpoint accepts either images or videos per request.")
        count = sum(mm_counts.values())
        if not count:
            return {}
        tile = self.merge**2
        edge = min(
            math.ceil(self.longest / self.unit),
            math.isqrt(max_num_encoder_tokens // (count * tile)),
        )
        if edge == 0:
            raise ValueError("Encoder budget is too small to profile the requested media.")
        size = edge * self.unit
        cost = edge**2 * tile
        frames = (
            min(_MAX_VIDEO_FRAMES, max_num_encoder_tokens // (count * cost))
            if mm_counts.get("video")
            else 1
        )
        rows = torch.zeros(
            (count * frames * cost, 3 * self.patch**2), dtype=dtype or torch.bfloat16
        )
        sizes = [(size, size)] * count
        if not mm_counts.get("video"):
            return self._payload(rows, sizes)
        template, output = [], frames * edge**2
        if self.video_pruning_rate:
            frame = Image.new("RGB", (size, size))
            _, _, _, template, output = self._video_layout(
                [frame] * frames,
                {"fps": _DEFAULT_VIDEO_FPS},
            )
        return self._video_payload(
            rows, sizes, [template] * count, [frames * cost] * count, [output] * count
        )
