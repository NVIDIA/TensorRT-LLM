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
# and s2wrapper: https://github.com/bfshi/scaling_on_scales

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision.transforms import Normalize, Resize, ToTensor


def fuse_input_embeds(
    model,
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
                               dtype=model.model_dtype)

    text_token_indices = torch.where(input_ids < model.vocab_size)[0]
    mm_token_indices = torch.where(input_ids >= model.vocab_size)[0]

    text_embed = model.llm.model.embed_tokens(input_ids[text_token_indices])
    input_embeds[text_token_indices, :] = text_embed.to(model.model_dtype)
    input_embeds[mm_token_indices, :] = mm_embed.to(model.model_dtype)

    return None, input_embeds.to(model.dtype)


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
    for size, num_split in zip(img_sizes, num_splits):
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
        for num_split, out in zip(num_splits, outs_multiscale)
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
