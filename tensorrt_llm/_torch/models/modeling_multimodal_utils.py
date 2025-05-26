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

from tensorrt_llm._torch.modules.embedding import Embedding

from ..._utils import nvtx_range


@nvtx_range("prepare_multimodal_ifb")
def prepare_multimodal_ifb(embedding_layer,
                           attn_metadata,
                           input_ids,
                           position_ids,
                           mm_tokens=None,
                           mm_embeds=None,
                           mm_embed_lengths=None,
                           prepare_position_ids=False):
    """
    Prepare for inflight batching LLM forward in multimodal mode. When Encoder + LLM is in the same forward, this function should be called between the encoder forward and the LLM forward.

    Challenges:
    in multimodal, the actual seq_lens (text + media) is only known AFTER the encoder forward (there are VLM models that vision seq_len is deterministic and independent of the media size; but a generic solution shouldn't base on such assumption). We need a way to update from the text-only seq_lens to the actual (text + media) seq_lens, which is not easy.
    - approach 1: update the input_ids field in llm request, such as using a dummy input_ids that matches the real seq_lens.
        Pros: worry-free about setting the correct lengths during scheduling and metadata init.
        Cons: (i) inside forward, we have no handle to the request object (ii) input_ids field in LlmRequest class is immutable (iii) wasted space to store the dummy input_ids
    - approach 2: update the actual lengths in model forward.
        Pros: (i) model-specific handling, doesn't affect other models (ii) no wasted space.
        Cons: (i) a bit hacky, modifying the metadata is error-prone (ii) in theory, the IFB scheduling may over-shoot because it's based on the text-only seq_lens
    The current implementation is approach 2.
    The ideal approach is to add a new seq_len field in LlmRequest. Let this field be mutable & all seq_len getter routes to this field rather than calculating len(input_ids).
    TODO: handle the hybrid case of both text-only & multimodal requests
    """

    num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations

    input_embeds = None

    # multimodal context requests
    if num_context_requests > 0 and mm_embeds is not None:  # skip dummy requests
        ## Fuse input embeddings
        # 1. remove mm tokens from input_ids
        raw_ctx_tokens, raw_gen_tokens = input_ids[:attn_metadata.
                                                   num_ctx_tokens], input_ids[
                                                       attn_metadata.
                                                       num_ctx_tokens:]
        raw_text_mask = ~torch.isin(raw_ctx_tokens, mm_tokens)
        input_ids = torch.cat([raw_ctx_tokens[raw_text_mask], raw_gen_tokens])
        input_embeds = torch.empty(input_ids.shape[0] + mm_embeds.shape[0],
                                   mm_embeds.shape[-1],
                                   device=mm_embeds.device,
                                   dtype=mm_embeds.dtype)
        fused_text_mask = torch.full((input_embeds.shape[0], ), False)
        if raw_gen_tokens.shape[0] > 0:
            fused_text_mask[-raw_gen_tokens.shape[0]:] = True

        # 2. calculate the text token indices in the fused input_embeds
        raw_text_masks = list(
            raw_text_mask.split(attn_metadata.context_lens.tolist()))
        mm_embed_splits = list(mm_embeds.split(mm_embed_lengths))
        start_idx, last_start_idx = 0, 0
        fused_lengths = []
        for text_mask, mm_embed in zip(raw_text_masks,
                                       mm_embed_splits):  # per request
            mm_positions = torch.where(text_mask == False)[0].tolist()
            num_medias = len(mm_positions)
            media_length = len(mm_embed) // num_medias

            # Diagram for 2 media, length 3 & length 4 each. After processing the 1st media:
            # index:           0  1  2  3  4  5  6  7  8  9  10 11 12 13
            # raw tokens:      T  T  T  M  T  T  M  T  T (T - text, M - media)
            # mm_positions:             ^        ^
            #                                    ^pos
            #                              ^last_pos
            # fused_text mask: T  T  T  F  F  F  T  T  F  F  F  F  T  T (T-True, F-False)
            #                                    ^start_idx
            last_pos = 0
            for pos in mm_positions:  # per media in each request
                text_length = pos - last_pos
                fused_text_mask[start_idx:start_idx + text_length] = True
                start_idx += text_length + media_length
                last_pos = pos + 1

            if last_pos < len(text_mask):
                # text between last media token & the end
                text_length = len(text_mask) - last_pos
                fused_text_mask[start_idx:start_idx + text_length] = True
                start_idx += text_length

            fused_lengths.append(start_idx - last_start_idx)
            last_start_idx = start_idx

        # 3. fuse embeddings
        input_embeds[fused_text_mask] = embedding_layer(input_ids)
        input_embeds[~fused_text_mask] = mm_embeds
        input_ids = None  # use input_embeds mode for multimodal

        ## Update metadata
        # 1. Update attn_metadata for the following LLM forward. This can be done in-place in this forward(). NOTE: we cannot simply do attn_metadata.seq_lens[:num_context_requests] = xx, because this won't trigger the setter calls (see interface.py) thus the underlying seq_lens_cuda buffer won't get updated. We MUST do an explicit setter.
        attn_metadata.prompt_lens[:num_context_requests] = fused_lengths
        attn_metadata.seq_lens[:num_context_requests] = torch.tensor(
            fused_lengths, dtype=torch.int32)
        attn_metadata.seq_lens_kv[:num_context_requests] = torch.tensor(
            fused_lengths, dtype=torch.int32)
        attn_metadata.on_update_gpu()
        # 2. Update request data for the generation phase. It's not straightforward to propagate the updated info back to request fields. A viable solution is to temporarily store the info in KVCacheManager, and leverage its update_resources() -- which is invoked after the forward() step -- to update the request data. see resource_manager.py.
        attn_metadata.kv_cache_manager.extra_info_for_update_resources[
            'updated_seq_lens_ctx'] = attn_metadata.seq_lens[:
                                                             num_context_requests].tolist(
                                                             )
        # 3. Update KV cache size allocation
        # TODO: is there a better way to add multiple tokens or extend the sequence in one go?
        for i, req in enumerate(
                attn_metadata.request_ids[:num_context_requests]):
            for _ in range(fused_lengths[i] -
                           attn_metadata.orig_prompt_lens[i]):
                attn_metadata.kv_cache_manager.impl.add_token(req)

    # multimodal generation requests
    if num_generation_requests > 0:
        # 4. Update KV cache length for generation requests
        # num_cached_tokens_per_seq is counting based on the original prompt length (because llmRequest class only stores the original text input ids). Number of generated tokens is the delta between the two.
        attn_metadata.kv_cache_params.num_cached_tokens_per_seq[
            num_context_requests:] = list(
                map(
                    lambda x, y, z: x + y - z,
                    attn_metadata.prompt_lens[num_context_requests:],
                    attn_metadata.kv_cache_params.
                    num_cached_tokens_per_seq[num_context_requests:],
                    attn_metadata.orig_prompt_lens[num_context_requests:]))
        # TODO: (1) could just save the delta between original & current prompt lens to save a subtract op exposed in step latency (2) more performant impl than list map

    attn_metadata.prepare()  # must update internal buffers

    if prepare_position_ids:
        position_ids_list = []
        for i in range(num_context_requests + num_generation_requests):
            if i < num_context_requests:
                position_ids_list.append(
                    torch.arange(start=0,
                                 end=attn_metadata.seq_lens[i],
                                 dtype=torch.int,
                                 device='cuda'))
            else:
                position_ids_list.append(
                    torch.tensor([
                        attn_metadata.kv_cache_params.
                        num_cached_tokens_per_seq[i]
                    ],
                                 dtype=torch.int,
                                 device='cuda'))
        if len(position_ids_list) > 0:
            position_ids = torch.cat(position_ids_list).unsqueeze(0)

    return attn_metadata, input_ids, position_ids, input_embeds


@nvtx_range("fuse_input_embeds")
def fuse_input_embeds(
    embedding_layer: Embedding,
    input_ids: torch.LongTensor,
    mm_embeds: List[torch.Tensor],
    mm_token_ids: Optional[torch.LongTensor] = None,
) -> Tuple[Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
    """
    Fuse text and multimodal embeddings. input_ids is [text_total_length + mm_total_length] and mm_embed is [mm_total_length, hidden_dim]. We just need to fuse them into [text_total_length + mm_total_length, hidden_dim] by slice-and-assign to the corresponding entries.

    Args:
        input_ids: shape [text_total_length + mm_total_length], flattened from List[(text_length1 + mm_total_length1), ..., (text_lengthi + mm_total_lengthi)]. For LLM model, the requests are inflight batched together, but the input_ids are flattened with padding removed. By the slice condition < vocab_size, we can easily separate text / multimodal tokens and naturally batched the LLM embedding lookup
        mm_embed: List[(mm_total_length1, hidden_dim), ..., (mm_total_lengthi, hidden_dim)].
        mm_token_ids: possible token ids for multimodal tokens, if known. If not known and set to None, it is assumed that the multimodal tokens are out-of-vocabulary tokens i.e. the `input_ids` contains tokens >= vocab_size that represent the multimodal tokens.
    Returns:
        - If (1) JIT test run, (2) non-multimodal run, i.e. all text-only requests, either context or generation phase (3) multimodal run, all requests in generation phase --> there is no multimodal data, return only the input_ids
        - If (4) multimodal run, mixed batch of context and generation requests, each context request has a multimodal feature --> return only the fused input_embeds of shape [total length, hidden_dim]. For text tokens, LLM embedding layer has already run.
    """
    if len(mm_embeds) == 0:
        return input_ids, None

    mm_embed = torch.cat(mm_embeds, dim=0)

    if mm_token_ids is None:
        # NOTE:
        # If mm_token_ids is None, it is assumed that the multimodal
        # tokens are out-of-vocab tokens i.e. the `input_ids` contains
        # tokens >= vocab_size that represent the multimodal tokens.
        # Since mm_token_ids is be unbounded in this case,
        # using torch.isin() may not be performant.
        # This provides a more performant alternative while keeping
        # the flexibility of still specifying all possible mm_token_ids,
        # if the user wants to.
        vocab_size = embedding_layer.num_embeddings
        mm_token_mask = input_ids >= vocab_size
        text_token_mask = input_ids < vocab_size
    else:
        mm_token_mask = torch.isin(input_ids, mm_token_ids)
        text_token_mask = ~mm_token_mask
    text_token_indices = torch.where(text_token_mask)[0]
    mm_token_indices = torch.where(mm_token_mask)[0]

    text_embed = embedding_layer(input_ids[text_token_indices])
    input_embeds = torch.empty(input_ids.shape[0],
                               mm_embed.shape[-1],
                               device=text_embed.device,
                               dtype=text_embed.dtype)

    input_embeds[text_token_indices, :] = text_embed.to(
        dtype=input_embeds.dtype, device=input_embeds.device)
    input_embeds[mm_token_indices, :] = mm_embed.to(dtype=input_embeds.dtype,
                                                    device=input_embeds.device)

    return None, input_embeds


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
