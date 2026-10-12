# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Image and video preprocessing follows the Cosmos3-Edge processor: SigLIP2
# patches for images, per-frame patches for video, and the Qwen3-VL chat
# template. Adapted from the Apache-2.0 vLLM Cosmos3-Edge processor.

import math

import torch
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
    smart_resize,
)
from transformers.models.siglip2.image_processing_siglip2 import (
    Siglip2ImageProcessor,
    convert_image_to_patches,
)
from transformers.utils import TensorType
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import group_videos_by_shape, reorder_videos


def round_by_factor(number: int, factor: int) -> int:
    """Return the integer closest to ``number`` that is divisible by ``factor``."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Return the smallest multiple of ``factor`` that is at least ``number``."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Return the largest multiple of ``factor`` that is at most ``number``."""
    return math.floor(number / factor) * factor


def _cacheable_stats(value):
    """Turn mean/std lists into tuples so rescale helpers can cache them."""
    if isinstance(value, list):
        return tuple(value)
    return value


class Cosmos3EdgeImagesKwargs(Siglip2ImageProcessor.valid_kwargs, total=False):
    max_pixels: int | None
    min_pixels: int | None
    per_image_kwargs: list[dict] | None
    merge_size: int | None


class Cosmos3EdgeImageProcessor(Siglip2ImageProcessor):
    resample = PILImageResampling.BICUBIC
    valid_kwargs = Cosmos3EdgeImagesKwargs

    def __init__(self, **kwargs):
        merge_size = kwargs.get("merge_size", 2)
        super().__init__(**kwargs)
        self.merge_size = merge_size or 2

    def _resize_image(
        self,
        image: torch.Tensor,
        max_ratio=200,
        resample: PILImageResampling | InterpolationMode | int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Resize so both sides are divisible by ``merge_size * patch_size``.

        The pixel count stays inside ``[min_pixels, max_pixels]`` and the aspect
        ratio is preserved as closely as possible.
        """
        image_min_pixels = kwargs.get("min_pixels") or self.size.get("shortest_edge", None)
        image_max_pixels = kwargs.get("max_pixels") or self.size.get("longest_edge", None)
        if image_min_pixels is None or image_max_pixels is None:
            raise ValueError("When do_resize is True, min_pixels and max_pixels must be provided.")
        if image_max_pixels < image_min_pixels:
            raise ValueError("The max_pixels of image must be greater than or equal to min_pixels.")

        _, height, width = image.shape
        aspect_ratio = max(height, width) / min(height, width)
        if aspect_ratio > max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {max_ratio}, got {aspect_ratio}"
            )
        factor = self.merge_size * self.patch_size
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > image_max_pixels:
            beta = math.sqrt((height * width) / image_max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < image_min_pixels:
            beta = math.sqrt(image_min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)

        return self.resize(
            image,
            size=SizeDict(height=h_bar, width=w_bar),
            resample=resample,
        )

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        patch_size: int,
        max_num_patches: int,
        resample: PILImageResampling | InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        del max_num_patches  # Edge sizes images by min/max pixels, not a patch cap.
        per_image_kwargs = kwargs.pop("per_image_kwargs", None)
        pixel_values = []
        spatial_shapes = []
        for idx, image in enumerate(images):
            image_kwargs = kwargs.copy()
            if per_image_kwargs is not None and idx < len(per_image_kwargs) and per_image_kwargs[idx]:
                image_kwargs.update(per_image_kwargs[idx])
            if do_resize:
                image = self._resize_image(
                    image,
                    max_ratio=200,
                    resample=resample,
                    **image_kwargs,
                )

            image = self.rescale_and_normalize(
                image,
                do_rescale,
                rescale_factor,
                do_normalize,
                _cacheable_stats(image_mean),
                _cacheable_stats(image_std),
            )
            patches = convert_image_to_patches(image, patch_size)
            num_patches_height = image.shape[1] // patch_size
            num_patches_width = image.shape[2] // patch_size
            spatial_shapes.append((num_patches_height, num_patches_width))
            pixel_values.append(patches)

        pixel_values = torch.cat(pixel_values, dim=0)
        spatial_shapes = torch.tensor(spatial_shapes)
        t_dim = torch.ones(
            (spatial_shapes.shape[0], 1),
            dtype=spatial_shapes.dtype,
            device=spatial_shapes.device,
        )
        image_grid_thw = torch.cat([t_dim, spatial_shapes], dim=1)
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )


class Cosmos3EdgeVideoProcessor(Qwen3VLVideoProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: PILImageResampling | InterpolationMode | int | None = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        del do_convert_rgb, kwargs
        if patch_size is None:
            raise ValueError("Cosmos3-Edge video preprocessing requires patch_size.")

        merge_size = self.merge_size
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            batch, frames, channels, height, width = stacked_videos.shape
            if do_resize:
                if size is None:
                    raise ValueError("Cosmos3-Edge video resize requires a size dict.")
                resized_height, resized_width = smart_resize(
                    num_frames=frames,
                    height=height,
                    width=width,
                    temporal_factor=1,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(batch * frames, channels, height, width)
                stacked_videos = BaseVideoProcessor.resize(
                    self,
                    image=stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.view(
                    batch, frames, channels, resized_height, resized_width
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(
                stacked_videos[0], channel_dim=ChannelDimension.FIRST
            )
            stacked_videos = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                _cacheable_stats(image_mean),
                _cacheable_stats(image_std),
            )
            batch_size, grid_t, channel = stacked_videos.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            patches = stacked_videos.view(
                batch_size,
                grid_t,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            patches = patches.permute(0, 1, 3, 5, 4, 6, 2)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                patch_size * patch_size * channel,
            )
            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        return BatchFeature(
            data={
                "pixel_values_videos": torch.cat(processed_videos, dim=0),
                "video_grid_thw": torch.tensor(processed_grids),
            },
            tensor_type=return_tensors,
        )


class Cosmos3EdgeProcessor(Qwen3VLProcessor):
    """Qwen3-VL chat template with Cosmos3-Edge image and video processors."""

    attributes = ["image_processor", "tokenizer", "video_processor"]

    @classmethod
    def _get_arguments_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        processor_dict=None,
        **kwargs,
    ):
        del processor_dict
        component_kwargs = dict(kwargs)
        component_kwargs.pop("trust_remote_code", None)
        image_processor = Cosmos3EdgeImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=False,
            **component_kwargs,
        )
        video_processor = Cosmos3EdgeVideoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        return [image_processor, tokenizer, video_processor]


def register_cosmos3_edge_processor() -> None:
    """Register the Edge processor so ``AutoProcessor.from_pretrained`` finds it."""
    from tensorrt_llm._torch.configs.cosmos3_edge import Cosmos3EdgeConfig

    try:
        AutoProcessor.register(Cosmos3EdgeConfig, Cosmos3EdgeProcessor, exist_ok=True)
    except TypeError:
        try:
            AutoProcessor.register(Cosmos3EdgeConfig, Cosmos3EdgeProcessor)
        except ValueError:
            return
    except ValueError:
        return
