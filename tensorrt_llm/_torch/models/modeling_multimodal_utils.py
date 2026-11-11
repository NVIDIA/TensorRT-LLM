# Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES
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
# and s2wrapper: https://github.com/bfshi/scaling_on_scales

import functools
import math
import os
from typing import Any, NamedTuple, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from PIL import Image
from torchvision.transforms import Normalize, Resize, ToTensor
from transformers.image_processing_backends import TorchvisionBackend

from tensorrt_llm.inputs.multimodal import MultimodalParams

from ..multimodal.embeddings import _as_copy_index
from ..multimodal.embeddings import \
    _get_uncached_multimodal_params as _get_uncached_multimodal_params
from ..multimodal.embeddings import _join_embeddings
from ..multimodal.embeddings import \
    _normalize_encoder_embeddings as _normalize_encoder_embeddings
from ..multimodal.embeddings import \
    _store_chunked_prefill_embeddings as _store_chunked_prefill_embeddings
from ..multimodal.embeddings import \
    filter_mm_token_from_input_ids as filter_mm_token_from_input_ids
from ..multimodal.embeddings import find_input_mm_embeds, fuse_input_embeds
from ..multimodal.embeddings import \
    get_attached_multimodal_embeddings as get_attached_multimodal_embeddings
from ..multimodal.embeddings import \
    get_multimodal_embeddings as get_multimodal_embeddings

__all__ = [
    "VisionNormalization",
    "vision_normalization_from_processor",
    "_vision_pixel_lookup_table",
    "_lookup_vision_pixels_kernel",
    "normalize_vision_pixels",
    "_MULTIMODAL_ENV_NAME",
    "_is_mm_disagg",
    "has_raw_multimodal_payload",
    "_PROCESSOR_OUTPUT_KEYS",
    "_get_cached_merged_typed_dict",
    "_install_processor_output_validation_filter",
    "preprocess_dispatch",
    "find_closest_aspect_ratio",
    "dynamic_preprocess",
    "dynamic_preprocess_torch",
    "dynamic_preprocess_dispatch",
    "dynamic_s2_preprocess",
    "dynamic_s2_preprocess_torch",
    "dynamic_s2_preprocess_dispatch",
    "s2_split_chessboard",
    "s2_merge_chessboard",
    "s2_batched_forward",
    "multiscale_forward",
    "merge_chessboard",
    "split_chessboard",
    "merge_features_for_dynamic_s2",
    "_get_uncached_multimodal_params",
    "_store_chunked_prefill_embeddings",
    "_normalize_encoder_embeddings",
    "_join_embeddings",
    "get_multimodal_embeddings",
    "get_attached_multimodal_embeddings",
    "find_input_mm_embeds",
    "filter_mm_token_from_input_ids",
    "_as_copy_index",
    "fuse_input_embeds",
]


class VisionNormalization(NamedTuple):
    """Processor statistics shared with vision encoder input preparation."""

    rescale_factor: float | None
    image_mean: tuple[float, ...] | None
    image_std: tuple[float, ...] | None
    out_dtype: torch.dtype
    compute_dtype: torch.dtype | None = None


def vision_normalization_from_processor(
        frames: list[Any], processor: Any,
        out_dtype: torch.dtype) -> VisionNormalization | None:
    """Describe elementwise normalization for RGB PIL or uint8 HWC frames."""
    # Workers reconstruct the table, so the processor must use the shared backend.
    uses_shared_backend = all(
        getattr(getattr(processor, name, None), "__func__", None) is getattr(
            TorchvisionBackend, name)
        for name in ("rescale_and_normalize", "rescale", "normalize"))
    if not uses_shared_backend:
        return None
    if not (processor.do_rescale or processor.do_normalize):
        return None
    rgb_frames = frames and all(
        (isinstance(frame, Image.Image) and frame.mode == "RGB") or
        (isinstance(frame, np.ndarray) and frame.dtype == np.uint8
         and frame.ndim == 3 and frame.shape[-1] == 3) for frame in frames)
    if not rgb_frames:
        return None
    return VisionNormalization(
        rescale_factor=processor.rescale_factor
        if processor.do_rescale else None,
        image_mean=((processor.image_mean, ) * 3 if np.isscalar(
            processor.image_mean) else tuple(processor.image_mean))
        if processor.do_normalize else None,
        image_std=((processor.image_std, ) * 3 if np.isscalar(
            processor.image_std) else tuple(processor.image_std))
        if processor.do_normalize else None,
        out_dtype=out_dtype,
        compute_dtype=torch.get_default_dtype(),
    )


@functools.lru_cache(maxsize=32)
def _vision_pixel_lookup_table(
    normalization: VisionNormalization,
    device: torch.device,
    channels: int,
    default_dtype: torch.dtype,
) -> torch.Tensor:
    """Precompute processor outputs for every uint8 value of each channel."""
    values = torch.arange(256, dtype=default_dtype,
                          device="cpu").expand(channels, 1, -1)
    image_mean, image_std = normalization.image_mean, normalization.image_std
    if normalization.rescale_factor is not None and image_mean is not None:
        image_mean = torch.tensor(image_mean, dtype=default_dtype,
                                  device="cpu").unbind()
        image_std = torch.tensor(image_std, dtype=default_dtype,
                                 device="cpu").unbind()
    values = TorchvisionBackend().rescale_and_normalize(
        values,
        do_rescale=normalization.rescale_factor is not None,
        rescale_factor=normalization.rescale_factor,
        do_normalize=normalization.image_mean is not None,
        image_mean=image_mean,
        image_std=image_std,
    )
    return values.to(dtype=normalization.out_dtype,
                     device=device).reshape(channels, 256).contiguous()


@triton.jit(do_not_specialize=["count", "channel_stride", "channels"])
def _lookup_vision_pixels_kernel(source, table, target, count, channel_stride,
                                 channels, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pixels = tl.load(source + offsets, offsets < count, other=0).to(tl.int32)
    channel = (offsets // channel_stride) % channels
    values = tl.load(table + channel * 256 + pixels)
    tl.store(target + offsets, values, offsets < count)


def normalize_vision_pixels(
    pixel_values: torch.Tensor,
    normalization: VisionNormalization,
) -> torch.Tensor:
    """Apply processor normalization in the tensor's device.

    Models reshape uint8 pixels to [..., channels, values_per_channel]. The
    processor's shared elementwise step evaluates all 256 values on the CPU,
    then casts once to the vision dtype. Models restore any padding inserted
    after CPU normalization.
    """
    if pixel_values.dtype != torch.uint8:
        raise ValueError("Device normalization requires uint8 pixels")
    if pixel_values.ndim < 2:
        raise ValueError("Vision pixels must include channel and value axes")
    channels, channel_stride = pixel_values.shape[-2:]
    if not channels or not channel_stride:
        return pixel_values.to(normalization.out_dtype)
    # Match the producer's intermediate precision when rebuilding the table.
    table = _vision_pixel_lookup_table(
        normalization, pixel_values.device, channels,
        normalization.compute_dtype or torch.get_default_dtype())
    if not pixel_values.is_cuda:
        offsets = torch.arange(pixel_values.numel(), device=pixel_values.device)
        channel = (offsets // channel_stride) % channels
        return table[channel, pixel_values.reshape(-1).long()].reshape(
            pixel_values.shape)
    output = torch.empty_like(pixel_values,
                              dtype=normalization.out_dtype,
                              memory_format=torch.contiguous_format)
    if pixel_values.numel():
        grid = (triton.cdiv(pixel_values.numel(), 1024), )
        _lookup_vision_pixels_kernel[grid](
            pixel_values.contiguous(),
            table,
            output,
            pixel_values.numel(),
            channel_stride,
            channels,
            BLOCK=1024,
            num_warps=4,
        )
    return output


_MULTIMODAL_ENV_NAME = "TLLM_MULTIMODAL_DISAGGREGATED"


# Make this a runtime lookup rather than a module-wide constant for easier unit testing.
# MM E/P split flag. Not generic disaggregated serving.
def _is_mm_disagg() -> bool:
    return os.getenv(_MULTIMODAL_ENV_NAME, "0") == "1"


def has_raw_multimodal_payload(param: MultimodalParams) -> bool:
    multimodal_data = param.multimodal_data or {}
    return any(
        multimodal_data.get(m) is not None for m in ("image", "video", "audio"))


# Processor *output* keys that transformers 5.x's
# ``ProcessorMixin._merge_kwargs`` strictly rejects when they leak into
# ``output_kwargs[<modality>]`` and reach ``validate_typed_dict``. They
# round-trip into call kwargs via saved tokenizer ``init_kwargs`` /
# ``model_input_names``, sub-processor metadata, or per-item processed
# fields, even when no caller passed them as inputs.
_PROCESSOR_OUTPUT_KEYS = frozenset({
    "image_grid_thw",
    "video_grid_thw",
    "pixel_values",
    "pixel_values_videos",
    "second_per_grid_ts",
    "mm_token_type_ids",
})


def _get_cached_merged_typed_dict(schema, cache):
    """Return a stable copy of ProcessorMixin's ephemeral merged TypedDict.

    `ProcessorMixin._merge_kwargs` creates a fresh
    `TypedDict("merged_typed_dict", ...)` for image/video kwargs on every
    processor call. `huggingface_hub` caches strict dataclass validators by
    schema-object identity, so a fresh type defeats that cache. Reuse an
    equivalent TypedDict class keyed by (totality, annotation items) so the
    upstream validator hits cache and skips the recursive type validation.
    """
    if getattr(schema, "__name__", None) != "merged_typed_dict":
        return schema
    try:
        cache_key = (getattr(schema, "__total__",
                             True), tuple(schema.__annotations__.items()))
        cached_schema = cache.get(cache_key)
    except TypeError:
        return schema
    if cached_schema is None:
        cached_schema = TypedDict(
            "merged_typed_dict",
            dict(schema.__annotations__),
            total=getattr(schema, "__total__", True),
        )
        cache[cache_key] = cached_schema
    return cached_schema


@functools.lru_cache(maxsize=None)
def _install_processor_output_validation_filter():
    """Install a process-wide filter over transformers' ``validate_typed_dict``.

    transformers 5.x added strict per-modality TypedDict validation in
    ``ProcessorMixin._merge_kwargs``. The keys in ``_PROCESSOR_OUTPUT_KEYS``
    are processor *outputs* that leak into ``output_kwargs[<modality>]`` via
    upstream bugs — e.g. ``Qwen2_5_VLProcessor._get_num_multimodal_tokens``
    mutates the class-level default dict instead of a copy, so once any
    caller passes ``video_grid_thw`` to ``get_num_multimodal_tokens`` it gets
    baked into the per-modality default and trips ``validate_typed_dict`` on
    every subsequent processor call. We filter those keys out before calling
    the genuine huggingface_hub implementation.

    Called from each Qwen VL input processor's ``__init__``. ``@lru_cache``
    guarantees the patch runs at most once per process: ``base_orig`` is
    captured exactly once from the genuine HF function, so concurrent
    ``trtllm-serve`` workers dispatched via ``asyncio.to_thread`` cannot
    observe a partially-patched state or chain filters recursively.

    Patches every transformers module that binds ``validate_typed_dict`` —
    each does its own ``from huggingface_hub.dataclasses import …``, so
    patching only one is insufficient. The set of binders differs across
    transformers versions (5.3.x rebinds it on ``image_processing_utils_fast``;
    5.5.x dropped that module and rebinds it on ``image_processing_utils``
    instead), so we discover binders by ``hasattr`` rather than hard-coding.
    """
    import transformers.processing_utils as _pu
    import transformers.video_processing_utils as _vpu

    _candidate_binders = [_pu, _vpu]
    for _name in ("transformers.image_processing_utils",
                  "transformers.image_processing_utils_fast"):
        try:
            _candidate_binders.append(__import__(_name, fromlist=[""]))
        except ImportError:
            pass
    binders = tuple(b for b in _candidate_binders
                    if hasattr(b, "validate_typed_dict"))
    if not binders:
        raise RuntimeError(
            "No transformers module exposes validate_typed_dict; "
            "cannot patch processor output validation.")
    base_orig = binders[0].validate_typed_dict
    merged_schema_cache: dict = {}

    def _filtered_validate(schema, data):
        if isinstance(data, dict):
            data = {
                k: v
                for k, v in data.items() if k not in _PROCESSOR_OUTPUT_KEYS
            }
        schema = _get_cached_merged_typed_dict(schema, merged_schema_cache)
        return base_orig(schema, data)

    for b in binders:
        b.validate_typed_dict = _filtered_validate


#region VILA utils
#  ------------------------------------------------------------------------------------------
#  VILA image preprocessing utils
#  based on: https://github.com/NVlabs/VILA/llava/mm_utils.py
#  ------------------------------------------------------------------------------------------


def preprocess_dispatch(image,
                        image_processor,
                        device=None,
                        dtype=None,
                        use_fast: bool = True):

    if use_fast:
        image = ToTensor()(image) if isinstance(image, Image.Image) else image
        if device is not None or dtype is not None:
            image = image.to(device=device, dtype=dtype)
        if image.shape[1] != image_processor.size["height"] or image.shape[
                2] != image_processor.size["width"]:
            image = Resize((image_processor.size["height"],
                            image_processor.size["width"]))(image)
        image = Normalize(image_processor.image_mean,
                          image_processor.image_std,
                          inplace=True)(image)
    else:
        image = image_processor.preprocess(
            image, return_tensors="pt", device=device)["pixel_values"][0].to(
                device)  # resize and normalize the images
    return image


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
                       image_processor,
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

    images = [
        image_processor.preprocess(image, return_tensors="pt",
                                   device="cuda")["pixel_values"][0].to("cuda")
        for image in processed_images
    ]  # Basically only normalize the images

    return torch.stack(images)


def dynamic_preprocess_torch(image,
                             image_processor,
                             min_num=1,
                             max_num=12,
                             image_size=384,
                             use_thumbnail=True,
                             interpolation_mode="bicubic",
                             device=None,
                             dtype=None):
    if device is not None or dtype is not None:
        image = image.to(device=device, dtype=dtype)

    if image.ndim == 3:
        image = image.unsqueeze(0)

    orig_height, orig_width = image.shape[-2:]
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
    resized_img = F.interpolate(image, (target_height, target_width),
                                mode=interpolation_mode)
    processed_images = []
    for i in range(blocks):
        # left, top, right, bottom
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img[:, :, box[1]:box[3], box[0]:box[2]]
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = F.interpolate(image, (image_size, image_size),
                                      mode=interpolation_mode)
        processed_images.append(thumbnail_img)

    images = torch.cat(processed_images, dim=0)
    images = Normalize(image_processor.image_mean,
                       image_processor.image_std,
                       inplace=True)(images)

    return images


def dynamic_preprocess_dispatch(image,
                                image_processor,
                                min_num=1,
                                max_num=12,
                                image_size=384,
                                use_thumbnail=True,
                                interpolation_mode="bicubic",
                                device=None,
                                dtype=None,
                                use_fast: bool = True):

    if use_fast:
        return dynamic_preprocess_torch(
            ToTensor()(image) if isinstance(image, Image.Image) else image,
            image_processor,
            min_num=min_num,
            max_num=max_num,
            image_size=image_size,
            use_thumbnail=use_thumbnail,
            interpolation_mode=interpolation_mode,
            device=device,
            dtype=dtype)
    else:
        return dynamic_preprocess(image,
                                  image_processor,
                                  min_num=min_num,
                                  max_num=max_num,
                                  image_size=image_size,
                                  use_thumbnail=use_thumbnail)


def dynamic_s2_preprocess(image,
                          image_processor,
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
    ############# Add tiles for all but the last scale using fixed square ratio ##############
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

    images = [
        image_processor.preprocess(image, return_tensors="pt",
                                   device="cuda")["pixel_values"][0].to("cuda")
        for image in processed_images
    ]  # Basically only normalize the images

    return torch.stack(images), (target_aspect_ratio[1], target_aspect_ratio[0])


def dynamic_s2_preprocess_torch(image,
                                image_processor,
                                s2_scales=[384, 768, 1152],
                                max_num=12,
                                image_size=384,
                                interpolation_mode="bicubic",
                                device=None,
                                dtype=None):
    if device is not None or dtype is not None:
        image = image.to(device=device, dtype=dtype)

    if image.ndim == 3:
        image = image.unsqueeze(0)

    orig_height, orig_width = image.shape[-2:]
    aspect_ratio = orig_width / orig_height
    min_num = (
        s2_scales[-1] //
        s2_scales[0])**2  # at least use number of tiles as the largest scale
    processed_images = []
    ##########################################################################################
    ############# Add tiles for all but the last scale using fixed square ratio ##############
    ##########################################################################################
    for scale in s2_scales[:-1]:
        target_width = image_size * (scale // s2_scales[0])
        target_height = image_size * (scale // s2_scales[0])
        blocks = (scale // s2_scales[0])**2
        # resize the image
        resized_img = F.interpolate(image, (target_height, target_width),
                                    mode=interpolation_mode)
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img[:, :, box[1]:box[3], box[0]:box[2]]
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
    resized_img = F.interpolate(image, (target_height, target_width),
                                mode=interpolation_mode)
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img[:, :, box[1]:box[3], box[0]:box[2]]
        processed_images.append(split_img)

    images = torch.cat(processed_images, dim=0)
    images = Normalize(image_processor.image_mean,
                       image_processor.image_std,
                       inplace=True)(images)

    return images, (target_aspect_ratio[1], target_aspect_ratio[0])


def dynamic_s2_preprocess_dispatch(image,
                                   image_processor,
                                   s2_scales=[384, 768, 1152],
                                   max_num=12,
                                   image_size=384,
                                   interpolation_mode="bicubic",
                                   device=None,
                                   dtype=None,
                                   use_fast: bool = True):

    if use_fast:
        return dynamic_s2_preprocess_torch(
            ToTensor()(image) if isinstance(image, Image.Image) else image,
            image_processor,
            s2_scales=s2_scales,
            max_num=max_num,
            image_size=image_size,
            interpolation_mode=interpolation_mode,
            device=device,
            dtype=dtype)
    else:
        return dynamic_s2_preprocess(image,
                                     image_processor,
                                     s2_scales=s2_scales,
                                     max_num=max_num,
                                     image_size=image_size)


#  ------------------------------------------------------------------------------------------
#  VILA ViT utils
#  Original code by Baifeng Shi, licensed under the MIT License:
#  https://github.com/bfshi/scaling_on_scales/blob/master/LICENSE.md
#  ------------------------------------------------------------------------------------------


def s2_split_chessboard(x, num_split):
    """
        x: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    x_split = rearrange(x,
                        'b c (nh h) (nw w) -> (nh nw b) c h w',
                        nh=num_split,
                        nw=num_split)
    return x_split


def s2_merge_chessboard(x, num_split):
    """
        x: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        (inverse of split_chessboard)
    """
    B, C, H, W = x.shape
    assert B % (num_split**2) == 0
    x_merge = rearrange(x,
                        '(nh nw b) c h w -> b c (nh h) (nw w)',
                        nh=num_split,
                        nw=num_split)

    return x_merge


def s2_batched_forward(model, x, batch_size=-1):
    if batch_size == -1:
        return model(x)
    else:
        x_batched = x.split(batch_size)
        outs = [model(x) for x in x_batched]
        return torch.cat(outs, dim=0)


def multiscale_forward(model,
                       input,
                       scales=None,
                       img_sizes=None,
                       max_split_size=None,
                       resize_output_to_idx=0,
                       num_prefix_token=0,
                       output_shape='bnc',
                       split_forward=False):

    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert input.shape[2] == input.shape[
        3], "Currently only square images are supported."
    assert output_shape in [
        'bnc', 'bchw'
    ], "Output shape should be either BxNxC (e.g., ViT) or BxCxHxW (e.g., ConvNet)."
    assert output_shape == 'bnc' or num_prefix_token == 0, "For ConvNet there shouldn't be any prefix token."

    b, c, input_size, _ = input.shape

    # image size for each scale
    assert scales is not None or img_sizes is not None, "Please assign either scales or img_sizes."
    img_sizes = img_sizes or [int(input_size * scale) for scale in scales]

    # prepare multiscale inputs
    max_split_size = max_split_size or input_size  # The maximum size of each split of image. Set as the input size by default
    num_splits = [math.ceil(size / max_split_size)
                  for size in img_sizes]  # number of splits each scale
    input_multiscale = []
    for size, num_split in zip(img_sizes, num_splits, strict=True):
        x = F.interpolate(input.to(torch.float32), size=size,
                          mode='bicubic').to(input.dtype)
        x = s2_split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    # run feedforward on each scale
    outs_multiscale = [
        s2_batched_forward(model, x, b) if split_forward else model(x)
        for x in input_multiscale
    ]
    if num_prefix_token > 0:
        outs_prefix_multiscale = [
            out[:, :num_prefix_token] for out in outs_multiscale
        ]
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
    if output_shape == 'bnc':
        outs_multiscale = [
            rearrange(out,
                      'b (h w) c -> b c h w',
                      h=int(out.shape[1]**0.5),
                      w=int(out.shape[1]**0.5)) for out in outs_multiscale
        ]

    # merge outputs of different splits for each scale separately
    outs_multiscale = [
        s2_merge_chessboard(out, num_split=num_split)
        for num_split, out in zip(num_splits, outs_multiscale, strict=True)
    ]

    # interpolate outputs from different scales and concat together
    output_size = outs_multiscale[resize_output_to_idx].shape[-2]
    out = torch.cat([
        F.interpolate(outs_multiscale[i].to(torch.float32),
                      size=output_size,
                      mode='area').to(outs_multiscale[i].dtype)
        for i in range(len(outs_multiscale))
    ],
                    dim=1)
    if output_shape == 'bnc':
        out = rearrange(out, 'b c h w -> b (h w) c')
    if num_prefix_token > 0:
        # take the mean of prefix tokens from different splits for each scale
        outs_prefix_multiscale = [
            torch.stack(out.split(b, dim=0), dim=0).mean(dim=0)
            for out in outs_prefix_multiscale
        ]
        out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
        out = torch.cat([out_prefix_multiscale, out], dim=1)

    return out


#  ------------------------------------------------------------------------------------------
#  VILA ViT utils (continued)
#  ------------------------------------------------------------------------------------------


def merge_chessboard(x, num_split_h, num_split_w):
    """
    x: b * n * c or b * h * w * c
    out: b * c * h * w
    Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
    """
    B = x.shape[0]
    if x.dim() == 3:
        N = x.shape[1]
        x = rearrange(x, "b (h w) c -> b c h w", h=int(N**0.5), w=int(N**0.5))

    assert B % (num_split_h * num_split_w) == 0
    b = B // (num_split_h * num_split_w)

    x_merge = torch.cat(
        [
            torch.cat([
                x[(i * num_split_w + j) * b:(i * num_split_w + j + 1) * b]
                for j in range(num_split_w)
            ],
                      dim=-1) for i in range(num_split_h)
        ],
        dim=-2,
    )

    return x_merge


def split_chessboard(x, num_split_h, num_split_w):
    """
    x: b * c * h * w
    out: b * c * h * w
    Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split_h == 0 and W % num_split_w == 0
    h, w = H // num_split_h, W // num_split_w
    x_split = torch.cat(
        [
            x[:, :, i * h:(i + 1) * h, j * w:(j + 1) * w]
            for i in range(num_split_h) for j in range(num_split_w)
        ],
        dim=0,
    )
    return x_split


def merge_features_for_dynamic_s2(vision_tower, image_features, block_sizes):
    scales = vision_tower.scales
    resize_output_to_scale_idx = vision_tower.resize_output_to_scale_idx

    image_features_each_image = []
    new_block_sizes = []
    block_cnt = 0
    for block_size_each_image in block_sizes:
        if block_size_each_image is None:
            cur_features = image_features[block_cnt:block_cnt + 1]
            cur_features = rearrange(cur_features,
                                     "1 (h w) c -> 1 c h w",
                                     h=int(cur_features.shape[1]**0.5))
            cur_features = cur_features.repeat(1, len(scales), 1, 1)
            image_features_each_image.append(cur_features)
            new_block_sizes.append((1, 1))
            block_cnt += 1
        else:
            cur_features_each_scale = []
            for scale in scales[:-1]:
                num_blocks_this_scale = (scale // scales[0])**2
                cur_features_each_scale.append(
                    merge_chessboard(
                        image_features[block_cnt:block_cnt +
                                       num_blocks_this_scale],
                        num_split_h=scale // scales[0],
                        num_split_w=scale // scales[0],
                    ))  # 1 * C * H * W
                block_cnt += num_blocks_this_scale
            num_blocks_last_scale = block_size_each_image[
                0] * block_size_each_image[1]
            cur_features_each_scale.append(
                merge_chessboard(
                    image_features[block_cnt:block_cnt + num_blocks_last_scale],
                    num_split_h=block_size_each_image[0],
                    num_split_w=block_size_each_image[1],
                ))  # 1 * C * H * W
            block_cnt += num_blocks_last_scale

            # resize and concat features from different scales
            output_size = cur_features_each_scale[
                resize_output_to_scale_idx].shape[-2:]
            cur_features = torch.cat(
                [
                    F.interpolate(cur_features_each_scale[i].to(torch.float32),
                                  size=output_size,
                                  mode="area").to(
                                      cur_features_each_scale[i].dtype)
                    for i in range(len(cur_features_each_scale))
                ],
                dim=1,
            )

            image_features_each_image.append(cur_features)

            if resize_output_to_scale_idx == len(
                    scales) - 1 or resize_output_to_scale_idx == -1:
                new_block_sizes.append(block_size_each_image)
            else:
                new_block_sizes.append((
                    scales[resize_output_to_scale_idx] // scales[0],
                    scales[resize_output_to_scale_idx] // scales[0],
                ))

    assert block_cnt == len(image_features)

    return image_features_each_image, new_block_sizes


#endregion
