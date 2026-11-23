# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash (``hy_v4_flash``) processor stack and multimodal placeholder metadata.

The bundled transformers has no ``hy_v4_flash`` model type, so this ports the checkpoint's processors.

Ported from HuggingFace Transformers at commit
``b865850d05326cf20e306b31f14b0027de802da3`` (PR #50903):

- ``src/transformers/models/hy_v4_flash/processing_hy_v4_flash.py``
- ``src/transformers/models/hy_v4_vl/image_processing_hy_v4_vl.py``
- ``src/transformers/models/hy_v4_vl/video_processing_hy_v4_vl.py``

Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import warnings
from collections.abc import Iterable

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF
from transformers import AutoProcessor, AutoTokenizer
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from transformers.processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.utils import TensorType, auto_docstring, logging
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoMetadata, group_videos_by_shape, reorder_videos

from tensorrt_llm.inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
)
from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Image processor (hy_v4_vl)
# ---------------------------------------------------------------------------


# Adapted from transformers.models.hy_v4_vl.image_processing_hy_v4_vl.HYV4VLImageProcessorKwargs
class HYV4VLImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    min_pixels (`int`, *optional*, defaults to `512 * 512`):
        The min pixels of the image to resize the image.
    max_pixels (`int`, *optional*, defaults to `2048 * 2048`):
        The max pixels of the image to resize the image.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """

    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@auto_docstring
class HYV4VLImageProcessor(TorchvisionBackend):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 512 * 512, "longest_edge": 2048 * 2048}
    default_to_square = False
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = HYV4VLImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]
    spatial_patch_size = 1

    def __init__(self, **kwargs: Unpack[HYV4VLImageProcessorKwargs]):
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        size = kwargs.pop("size", None)
        size = dict(self.size) if size is None else size
        if (min_pixels := kwargs.pop("min_pixels", None)) is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if (max_pixels := kwargs.pop("max_pixels", None)) is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None or max_pixels is not None:
            warnings.warn(
                "Passing `min_pixels` and `max_pixels` to a processor call is deprecated and will be removed in v5.23. "
                "Pass in `size={'longest_edge': xxx, 'shortest_edge': xxx} to override the target size.`",
                FutureWarning,
            )

            size_dict = dict(size) if isinstance(size, (dict, SizeDict)) else {}
            if min_pixels is not None:
                size_dict["shortest_edge"] = min_pixels
            if max_pixels is not None:
                size_dict["longest_edge"] = max_pixels
            size = SizeDict(**size_dict)
        return super()._standardize_kwargs(size=size, **kwargs)

    # Not `@auto_docstring` as upstream: on a method it raises outside an `image_processing_*.py` module.
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[HYV4VLImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        images: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        factor: int,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize dynamically based on input image aspect ratio."""
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError(
                f"`size` dict must contain 'shortest_edge' and 'longest_edge' keys but got {size}."
            )

        height, width = images.shape[-2:]
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        return super().resize(
            image=images,
            size=SizeDict(height=resized_height, width=resized_width),
            # The reference HunyuanOCR processor calls `PIL.Image.resize` without a
            # resampling argument, which uses BICUBIC for RGB images. Its config has
            # `resample=1` (LANCZOS), but the original implementation never uses it.
            resample=PILImageResampling.BICUBIC,
        )

    def patchify(
        self,
        images: "torch.Tensor",
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple["torch.Tensor", int, int]:
        """Patchifies each image into flat layout of shape (`seq_len`, `patch_dim`).

        The flat layout lets dynamically shaped pixels be concatenated.
        """
        batch_size, channel, resized_height, resized_width = images.shape
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = images.reshape(
            batch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Flatten patches in row-major order in comparison to Qwen
        patches = patches.permute(0, 2, 3, 5, 6, 1, 4, 7)

        flatten_patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
            .reshape(
                batch_size,
                grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
        )

        return flatten_patches, grid_h, grid_w

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    images=stacked_images,
                    size=size,
                    resample=resample,
                    factor=patch_size * merge_size,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_h, grid_w = self.patchify(
                stacked_images,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )

            processed_images_grouped[shape] = patches
            processed_grids[shape] = [[1, grid_h, grid_w]] * len(stacked_images)

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids_ordered = reorder_images(processed_grids, grouped_images_index)
        pixel_values = (
            processed_images[0]
            if len(processed_images) == 1
            else torch.cat(processed_images, dim=0)
        )
        image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict | None = None
    ) -> tuple[int, int]:
        """
        A utility that returns number of image patches for a given image size.

        Note: Do not remove this method! It is used by vLLM to infer the number of patches and placeholders
        without an image input.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `tuple[int, int]`: Number of image patches per image, as a `(height, width)` grid.
        """
        images_kwargs = images_kwargs or {}
        min_pixels = (
            images_kwargs["min_pixels"]
            if "min_pixels" in images_kwargs
            else self.size["shortest_edge"]
        )
        max_pixels = (
            images_kwargs["max_pixels"]
            if "max_pixels" in images_kwargs
            else self.size["longest_edge"]
        )
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return resized_height // patch_size, resized_width // patch_size


# ---------------------------------------------------------------------------
# Video processor (hy_v4_vl)
# ---------------------------------------------------------------------------


class HYV4VLVideoProcessorInitKwargs(VideosKwargs, total=False):
    r"""
    size (`dict[str, int]`, *optional*):
        Total clip pixel budget, given by the required `longest_edge` key. A `shortest_edge` entry, which existing
        configs carry, is accepted but not used: frames are never upscaled.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    min_frames (`int`, *optional*, defaults to 2):
        The minimum number of frames to sample from a video.
    max_frames (`int`, *optional*, defaults to 1024):
        The maximum number of frames to sample from a video.
    max_image_size (`dict[str, int]`, *optional*):
        Per-frame pixel-area limit, specified by the `longest_edge` key.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int
    max_image_size: dict[str, int]


@auto_docstring
class HYV4VLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 4096, "longest_edge": 33554432}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    min_frames = 2
    max_frames = 1024
    do_sample_frames = False
    valid_kwargs = HYV4VLVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]
    max_image_size = {"longest_edge": 1024 * 1024}
    max_duration = None
    num_frames = None
    fps = 2

    def __init__(self, **kwargs: Unpack[HYV4VLVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and
        total number of frames. If `fps` is passed along with metadata, `fps` frames per second are
        sampled uniformly. Arguments `num_frames` and `fps` are mutually exclusive.

        Args:
            video (`torch.Tensor`):
                Video that need to be sampled.
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.
        Returns:
            np.ndarray: Sampled video frames.
        """
        if fps is not None and num_frames is not None:
            raise ValueError(
                "`num_frames` and `fps` are mutually exclusive arguments, please use only one!"
            )

        total_num_frames = metadata.total_num_frames
        fps = fps if fps is not None else self.fps

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is None and fps is not None:
            if metadata.fps is None:
                metadata.fps = 24
                logger.warning_once(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is "
                    "required when sampling with `fps`. "
                    "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                )
            num_frames = int(total_num_frames / metadata.fps * fps)
            num_frames = min(max(num_frames, self.min_frames), self.max_frames, total_num_frames)

        if num_frames is None:
            num_frames = min(max(total_num_frames, self.min_frames), self.max_frames)

        indices = np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)

        return indices

    def resize(
        self,
        videos: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        factor: int,
        max_image_size: dict[str, int],
        **kwargs,
    ) -> "torch.Tensor":
        """Resize dynamically based on input video aspect ratio."""
        # Only the clip budget is needed: `shortest_edge` is not a lower bound, frames are never upscaled.
        if not size.longest_edge:
            raise ValueError(f"`size` dict must contain a 'longest_edge' key but got {size}.")
        height, width = self.smart_resize(
            frame_height=videos.shape[-2],
            frame_width=videos.shape[-1],
            num_frames=videos.shape[1],
            factor=factor,
            max_pixels=size.longest_edge,
            max_image_size=max_image_size["longest_edge"],
        )
        return super().resize(
            image=videos,
            size=SizeDict(height=height, width=width),
            resample=resample,
        )

    def patchify(
        self,
        videos: "torch.Tensor",
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple["torch.Tensor", int, int, int]:
        batch_size, num_frames, channel, resized_height, resized_width = videos.shape
        if pad := -num_frames % temporal_patch_size:
            videos = torch.cat((videos, videos[:, -1:].expand(-1, pad, -1, -1, -1)), dim=1)
            num_frames += pad

        grid_t = num_frames // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = videos.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Key difference: row-major order within each merged block, as in the image processor
        patches = patches.permute(0, 1, 4, 5, 7, 8, 3, 2, 6, 9)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        return flatten_patches, grid_t, grid_h, grid_w

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_image_size: dict[str, int],
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_resize:
                stacked_videos = self.resize(
                    videos=stacked_videos,
                    size=size,
                    resample=resample,
                    factor=patch_size * merge_size,
                    max_image_size=max_image_size,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_t, grid_h, grid_w = self.patchify(
                stacked_videos,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )
            processed_videos_grouped[shape] = patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * len(stacked_videos)

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        return BatchFeature(
            data={
                "pixel_values_videos": torch.cat(processed_videos, dim=0),
                "video_grid_thw": torch.tensor(processed_grids),
            },
            tensor_type=return_tensors,
        )

    def get_num_of_video_patches(
        self, num_frames: int, height: int, width: int, videos_kwargs=None
    ) -> int | tuple[int, int, int, int]:
        """
        A utility that returns number of video patches a given video size.

        Args:
            num_frames (`int`):
                Number of frames in the input video.
            height (`int`):
                Height of the input video.
            width (`int`):
                Width of the input video.
            videos_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the video processor.
        Returns:
            `int`: Number of video patches per video, or `tuple[int, int, int, int]` with the number of patches and
            the `(grid_t, grid_h, grid_w)` grids.
        """
        videos_kwargs = videos_kwargs if videos_kwargs is not None else {}
        do_resize = (
            videos_kwargs.get("do_resize")
            if videos_kwargs.get("do_resize") is not None
            else self.do_resize
        )
        size = videos_kwargs.get("size", None) or self.size
        max_image_size = videos_kwargs.get("max_image_size", None) or self.max_image_size
        patch_size = videos_kwargs.get("patch_size", None) or self.patch_size
        merge_size = videos_kwargs.get("merge_size", None) or self.merge_size
        temporal_patch_size = (
            videos_kwargs.get("temporal_patch_size", None) or self.temporal_patch_size
        )

        if do_resize:
            height, width = self.smart_resize(
                frame_height=height,
                frame_width=width,
                num_frames=num_frames,
                factor=patch_size * merge_size,
                max_pixels=size["longest_edge"],
                max_image_size=max_image_size["longest_edge"],
            )

        grid_t = (num_frames + -num_frames % temporal_patch_size) // temporal_patch_size
        grid_h, grid_w = height // patch_size, width // patch_size
        num_patches = grid_t * grid_h * grid_w

        return num_patches, grid_t, grid_h, grid_w

    @staticmethod
    def smart_resize(
        frame_height: int,
        frame_width: int,
        num_frames: int,
        *,
        factor: int,
        max_pixels: int = 33554432,
        max_image_size: int = 1024 * 1024,
    ) -> tuple[int, int]:
        """Return the largest aligned ``(height, width)`` for a frame that fits every budget.

        ``factor`` is the merged patch edge in pixels (``patch_size * merge_size``). A frame of ``rows x cols``
        merged patches costs ``rows * (cols + 1)`` tokens, one newline token per row, which is the same accounting the
        processor uses. The result never upscales, keeps each frame within ``max_image_size`` pixels, and keeps
        ``num_frames * rows * (cols + 1)`` inside the visual token budget of ``max_pixels``. Calling it again on its
        own output returns the same size, so a frame that was already resized is left untouched.
        """
        if min(num_frames, frame_height, frame_width, factor, max_pixels, max_image_size) <= 0:
            raise ValueError("Video dimensions, resize factor, and pixel budgets must be positive.")
        if max(frame_height, frame_width) / min(frame_height, frame_width) > 200:
            raise ValueError(
                "absolute aspect ratio must be smaller than 200, got "
                f"{max(frame_height, frame_width) / min(frame_height, frame_width)}"
            )
        max_rows = max(1, frame_height // factor)
        max_cols = max(1, frame_width // factor)
        visual_budget = max_pixels // (factor * factor)
        for rows in range(max_rows, 0, -1):
            # On the ray scale * (height, width), rows == k spans cols floor(W*k/H) .. ceil(W*(k+1)/H) - 1;
            # rows clamp at 1, so the last step admits any narrower width.
            low_cols = 1 if rows == 1 else max(1, frame_width * rows // frame_height)
            high_cols = min(max_cols, max(1, -(-frame_width * (rows + 1) // frame_height) - 1))
            budget_cols = visual_budget // (num_frames * rows) - 1 if visual_budget > 0 else 0
            cap_cols = max_image_size // (factor * factor * rows)
            cols = min(high_cols, budget_cols, cap_cols)
            if cols >= low_cols:
                return rows * factor, cols * factor
        return factor, factor


# ---------------------------------------------------------------------------
# Processor (hy_v4_flash)
# ---------------------------------------------------------------------------


@auto_docstring
class HYV4FlashProcessor(ProcessorMixin):
    return_mm_token_type_ids = True

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = (
            "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    @classmethod
    def _get_arguments_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        processor_dict=None,
        **kwargs,
    ):
        """Build the sub-processors from the checkpoint (no ``hy_v4_vl`` Auto* classes here)."""
        del processor_dict
        component_kwargs = dict(kwargs)
        component_kwargs.pop("trust_remote_code", None)
        image_processor = HYV4VLImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=False,
            **component_kwargs,
        )
        video_processor = HYV4VLVideoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        return [image_processor, tokenizer, video_processor]

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        # The grid row may be a tensor or a list; string repetition needs Python ints.
        _, grid_h, grid_w = image_inputs["image_grid_thw"][image_idx]
        merge_size = kwargs.get("merge_size") or self.image_processor.merge_size
        return self.image_token * self._num_frame_tokens(int(grid_h), int(grid_w), int(merge_size))

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        grid_t, grid_h, grid_w = video_inputs["video_grid_thw"][video_idx]
        merge_size = kwargs.get("merge_size") or self.video_processor.merge_size
        return self.video_token * (
            int(grid_t) * self._num_frame_tokens(int(grid_h), int(grid_w), int(merge_size))
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = {**kwargs, **kwargs.get("images_kwargs", {})}
            merge_size = images_kwargs.get("merge_size") or self.image_processor.merge_size
            grids = [
                self.image_processor.get_number_of_image_patches(height, width, images_kwargs)
                for height, width in image_sizes
            ]
            vision_data["num_image_patches"] = [grid_h * grid_w for grid_h, grid_w in grids]
            vision_data["num_image_tokens"] = [
                self._num_frame_tokens(*grid, merge_size) for grid in grids
            ]
        if video_sizes is not None:
            videos_kwargs = {**kwargs, **kwargs.get("videos_kwargs", {})}
            merge_size = videos_kwargs.get("merge_size") or self.video_processor.merge_size
            num_video_tokens = []
            for num_frames, height, width in video_sizes:
                _, grid_t, grid_h, grid_w = self.video_processor.get_num_of_video_patches(
                    num_frames, height, width, videos_kwargs
                )
                num_video_tokens.append(grid_t * self._num_frame_tokens(grid_h, grid_w, merge_size))
            vision_data["num_video_tokens"] = num_video_tokens
        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape
                `(batch_size, sequence_length)` or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's
                `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's
                `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]

    @staticmethod
    def _num_frame_tokens(grid_h: int, grid_w: int, merge_size: int) -> int:
        # One extra newline token per merged row
        return (grid_h // merge_size) * (grid_w // merge_size + 1)


def register_hyv4_flash_processor() -> None:
    """Register the processor so ``AutoProcessor.from_pretrained`` finds it."""
    from tensorrt_llm._torch.configs.hyv4_flash import HYV4FlashConfig

    try:
        AutoProcessor.register(HYV4FlashConfig, HYV4FlashProcessor, exist_ok=True)
    except TypeError:
        try:
            AutoProcessor.register(HYV4FlashConfig, HYV4FlashProcessor)
        except ValueError:
            return
    except ValueError:
        return


register_hyv4_flash_processor()

# One placeholder per media item; the model expands the pad run from the grid.
PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={
        "image": "<｜hy_image▁start:opensource｜><｜hy_image▁pad:opensource｜><｜hy_image▁end:opensource｜>",
        "video": "<｜hy_video▁start:opensource｜><｜hy_video▁pad:opensource｜><｜hy_video▁end:opensource｜>",
    },
    placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    placeholders_separator="",
    content_format=ContentFormat.STRING,
)

MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
    "hy_v4_flash", PLACEHOLDER_METADATA, registrant_module=__name__
)


__all__ = [
    "HYV4FlashProcessor",
    "HYV4VLImageProcessor",
    "HYV4VLVideoProcessor",
    "PLACEHOLDER_METADATA",
    "register_hyv4_flash_processor",
]
