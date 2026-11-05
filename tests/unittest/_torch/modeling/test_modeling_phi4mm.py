# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for Phi-4 multimodal grouped encoder inputs.

The input processor packs each request's image and audio tensors per
modality, and the encoder runs one image call and one audio call per batch
through EncoderGroups. The oracle is the previous payload and token-level
encode path, copied verbatim below, with the checkpoint's audio embedding
forward it called. The towers are deterministic stubs; the image
post-processing is the real vision forward.
"""

import math
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from tensorrt_llm._torch.models.modeling_phi4mm import (
    _AUDIO_SPECIAL_TOKEN_ID,
    _IMAGE_SPECIAL_TOKEN_ID,
    HFPhi4MultimodalEncoder,
    Phi4MMForCausalLM,
    Phi4MMInputProcessor,
    _reshape_attention_masks,
    vision_encoder_forward,
)
from tensorrt_llm.inputs.multimodal import MULTIMODAL_ENCODER_ITEM_METADATA_KEY, MultimodalParams

pytestmark = pytest.mark.cpu_only

_PAD_TOKEN_ID = 199999  # '<|endoftext|>', which the previous encoder padded prompts with


# ---------------------------------------------------------------------------
# Oracle: the previous payload and encode path, copied verbatim.
# ---------------------------------------------------------------------------


def _reference_audio_embedding_forward(
    self,
    input_ids,
    input_embeds,
    audio_embed_sizes=None,
    audio_attention_mask=None,
    audio_projection_mode="speech",
    **kwargs,
):
    """The checkpoint's audio embedding forward that the previous encoder called.

    Inference path only: features for every clip, then the first
    `audio_embed_sizes[i]` rows of clip `i` scattered into the embedded
    prompt at the audio token positions.
    """
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    positions = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=False)
    positions_tuple = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=True)
    target_device = self.audio_projection.bias.device
    target_dtype = self.audio_projection.bias.dtype
    if input_embeds is not None:
        input_embeds = input_embeds.to(target_device).to(target_dtype)
    if len(positions.tolist()) > 0:
        audio_set_tensor = self.get_audio_features(
            input_embeds, audio_attention_mask, audio_projection_mode
        )
    hidden_states = kwargs["wte"](input_ids)
    if len(positions.tolist()) > 0:
        merged_audio_set_tensor = torch.cat(
            [audio_set_tensor[i, : audio_embed_sizes[i], :] for i in range(len(audio_embed_sizes))],
            dim=0,
        )
        merged_audio_set_tensor = merged_audio_set_tensor.to(hidden_states.dtype).to(
            hidden_states.device
        )
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            hidden_states = hidden_states.index_put(
                indices=positions_tuple, values=merged_audio_set_tensor, accumulate=False
            )
    if self.drop is not None:
        hidden_states = self.drop(hidden_states)
    return hidden_states


def _main_vision_encoder_forward(
    self, input_ids: torch.LongTensor, input_embeds: torch.FloatTensor, image_sizes=None, **kwargs
) -> torch.FloatTensor:
    """Optimize vision encoder forward.

    * Remove many unnecessary if-else conditions and assertions.
    * Optimize positions/positions_tuple calculation.
    * Optimize the input parameters for get_img_features.

    Ref code: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/modeling_phi4mm.py#L268
    """
    if isinstance(image_sizes, torch.Tensor):
        image_sizes = image_sizes.view(-1, 2)
    image_attention_mask = kwargs["image_attention_mask"]
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    positions = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=False)
    positions_tuple = torch.unbind(positions, dim=1)
    if isinstance(self.img_projection, torch.nn.Sequential):
        target_device = self.img_projection[0].bias.device
        target_dtype = self.img_projection[0].bias.dtype
    else:
        target_device = self.img_projection.bias.device
        target_dtype = self.img_projection.bias.dtype

    # Inference with SigLip Vision Encoder.
    batch_size = input_embeds.shape[0]
    input_embeds = input_embeds.flatten(0, 1)
    flatten_img_attn_mask = image_attention_mask.to(torch.bool).flatten(0, 1)
    img_features = self.get_img_features(input_embeds, attention_mask=flatten_img_attn_mask)

    # Reshape and combine global/sub image features.
    base_resolution = self.crop_size
    base_feat_height_reduction = self.base_feat_height_reduction
    base_feat_height = self.base_feat_height_target
    img_features = img_features.view(
        batch_size, -1, base_feat_height * base_feat_height, self.image_dim_out
    )
    C = self.image_dim_out
    H = base_feat_height
    rh = base_feat_height_reduction
    _h1 = H // rh
    _fused_dim = rh * rh * C
    output_imgs, output_len = [], []
    for batch_idx in range(batch_size):
        h, w = image_sizes[batch_idx]
        h = h // base_resolution
        w = w // base_resolution
        B_ = h * w

        # Global image features.
        global_image_feature = img_features[batch_idx, :1]
        global_image_feature = global_image_feature.reshape(1, H, H, C)
        global_image_feature = global_image_feature.reshape(1, _h1, rh, _h1, rh, C)
        global_image_feature = global_image_feature.permute(0, 1, 3, 2, 4, 5)
        global_image_feature = global_image_feature.reshape(1, _h1, _h1, rh * rh * C)
        global_GN = self.sub_GN.repeat(1, _h1, 1, 1)
        global_image_feature = torch.cat([global_image_feature, global_GN], dim=2).reshape(
            1, -1, _fused_dim
        )

        # Sub image features.
        sub_image_feature = img_features[batch_idx, 1 : (1 + B_)]
        sub_image_feature = sub_image_feature.reshape(B_, H, H, C)
        sub_image_feature = sub_image_feature.reshape(B_, _h1, rh, _h1, rh, C)
        sub_image_feature = sub_image_feature.permute(0, 1, 3, 2, 4, 5)
        sub_image_feature = sub_image_feature.reshape(B_, -1, rh * rh * C)
        sub_image_feature = sub_image_feature.reshape(1, h, w, _h1, _h1, -1)
        sub_image_feature = sub_image_feature.permute(0, 1, 3, 2, 4, 5)
        sub_image_feature = sub_image_feature.reshape(1, h * _h1, w * _h1, rh * rh * C)

        # Fetch useful content.
        downsample_attn_mask = image_attention_mask[batch_idx, 1 : B_ + 1, 0::2, 0::2]
        downsample_attn_mask = downsample_attn_mask.reshape(1, h, w, _h1, _h1)
        downsample_attn_mask = downsample_attn_mask.permute(0, 1, 3, 2, 4)
        downsample_attn_mask = downsample_attn_mask.reshape(1, h * _h1, w * _h1)
        useful_height = int(downsample_attn_mask[0, :, 0].sum())  # Not optimized for D2H memcpy.
        useful_width = int(downsample_attn_mask[0, 0, :].sum())  # Not optimized for D2H memcpy.
        sub_image_feature = sub_image_feature[:, :useful_height, :useful_width]
        sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
        num_image_tokens = (
            image_attention_mask[batch_idx, : B_ + 1, 0::2, 0::2].sum().to(torch.int32)
            + (useful_height + 1)
            + _h1
        )  # Not optimized.
        sub_image_feature = torch.cat([sub_image_feature, sub_GN], dim=2).reshape(1, -1, _fused_dim)

        # Concat global/sub image features.
        if self.hd_transform_order == "glb_sub":
            output_imgs.append(
                torch.cat([global_image_feature, self.glb_GN, sub_image_feature], dim=1)
            )
        elif self.hd_transform_order == "sub_glb":
            output_imgs.append(
                torch.cat([sub_image_feature, self.glb_GN, global_image_feature], dim=1)
            )
        else:
            raise NotImplementedError(
                f"hd_transform_order = {self.hd_transform_order}, not implemented"
            )
        output_len.append(num_image_tokens)

    # Project image features.
    img_set_tensor = []
    for _output_img in output_imgs:
        img_feature_proj = self.img_projection(_output_img.to(target_device).to(target_dtype))
        img_set_tensor.append(img_feature_proj)

    # Combine image embeddings with text embeddings.
    hidden_states = kwargs["wte"](input_ids)
    merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        new_hidden_states = hidden_states.index_put(
            indices=positions_tuple,
            values=merged_img_set_tensor,
            accumulate=False,
        )  # Not optimized for D2D memcpy.
    hidden_states = new_hidden_states

    if self.drop is not None:
        hidden_states = self.drop(hidden_states)

    return hidden_states


def _main_batch_infer_image_embeds(
    self, batched_input_ids: torch.Tensor, multimodal_params: List[MultimodalParams]
) -> torch.Tensor:
    # Batch image inputs and attention mask with padding along dim=1 (patch num).
    input_image_embeds_list, input_image_attn_mask_list, input_image_sizes_list = [], [], []
    for mm_param in multimodal_params:
        mm_data = mm_param.multimodal_data
        input_image_embeds = mm_data["input_image_embeds"]
        if input_image_embeds is not None and input_image_embeds.numel() > 0:
            input_image_embeds_list.append(input_image_embeds)
            input_image_attn_mask_list.append(mm_data["image_attention_mask"])
            input_image_sizes_list.append(mm_data["image_sizes"])
    batched_image_hidden_states = None
    if len(input_image_embeds_list) > 0:
        # Padding image embeds/attn_masks along dim=1 (patch dimension).
        b_list = [x.shape[0] for x in input_image_embeds_list]
        p_list = [x.shape[1] for x in input_image_embeds_list]
        c_i, h_i, w_i = input_image_embeds_list[0].shape[2:5]
        h_i_attn, w_i_attn = input_image_attn_mask_list[0].shape[2:4]
        total_b = sum(b_list)
        max_p = max(p_list)
        batched_image_embeds = torch.zeros(
            (total_b, max_p, c_i, h_i, w_i),
            dtype=input_image_embeds_list[0].dtype,
            device=input_image_embeds_list[0].device,
        )
        batched_image_attn_mask = torch.zeros(
            (total_b, max_p, h_i_attn, w_i_attn),
            dtype=input_image_embeds_list[0].dtype,
            device=input_image_embeds_list[0].device,
        )
        b_offset = 0
        for i, tensor in enumerate(input_image_embeds_list):
            b, p = tensor.shape[:2]
            batched_image_embeds[b_offset : b_offset + b, :p] = tensor
            if input_image_attn_mask_list[i] is not None:
                batched_image_attn_mask[b_offset : b_offset + b, :p] = input_image_attn_mask_list[i]
            else:
                batched_image_attn_mask[b_offset : b_offset + b, :p] = 1
            b_offset += b
        batched_image_sizes = torch.cat(input_image_sizes_list, dim=0)
        # Forward image encoder with batched image embeds.
        batched_image_hidden_states = _main_vision_encoder_forward(
            self.image_embed,
            input_ids=batched_input_ids,
            input_embeds=batched_image_embeds,
            image_sizes=batched_image_sizes,
            image_attention_mask=batched_image_attn_mask,
            wte=self.embed_tokens,
        )
    return batched_image_hidden_states


def _main_batch_infer_audio_embeds(
    self, batched_input_ids: torch.Tensor, multimodal_params: List[MultimodalParams]
) -> torch.Tensor:
    # Batch audio inputs and attention mask with padding along dim=1 (patch num).
    input_audio_embeds_list, input_audio_attn_mask_list, input_audio_sizes_list = [], [], []
    for mm_param in multimodal_params:
        mm_data = mm_param.multimodal_data
        input_audio_embeds = mm_data["input_audio_embeds"]
        if input_audio_embeds is not None and input_audio_embeds.numel() > 0:
            input_audio_embeds_list.append(input_audio_embeds)
            input_audio_attn_mask_list.append(mm_data["audio_attention_mask"])
            input_audio_sizes_list.append(mm_data["audio_embed_sizes"])
    batched_audio_hidden_states = None
    if len(input_audio_embeds_list) > 0:
        b_list = [x.shape[0] for x in input_audio_embeds_list]
        p_list = [x.shape[1] for x in input_audio_embeds_list]
        d_a = input_audio_embeds_list[0].shape[2]
        total_b = sum(b_list)
        max_p = max(p_list)
        batched_audio_embeds = torch.zeros(
            (total_b, max_p, d_a),
            dtype=input_audio_embeds_list[0].dtype,
            device=input_audio_embeds_list[0].device,
        )
        batched_audio_attn_mask = torch.zeros(
            (total_b, max_p),
            dtype=input_audio_embeds_list[0].dtype,
            device=input_audio_embeds_list[0].device,
        )
        b_offset = 0
        for i, tensor in enumerate(input_audio_embeds_list):
            b, p = tensor.shape[:2]
            batched_audio_embeds[b_offset : b_offset + b, :p] = tensor
            if input_audio_attn_mask_list[i] is not None:
                batched_audio_attn_mask[b_offset : b_offset + b, :p] = input_audio_attn_mask_list[i]
            else:
                batched_audio_attn_mask[b_offset : b_offset + b, :p] = 1
            b_offset += b
        batched_audio_sizes = torch.cat(input_audio_sizes_list, dim=0)
        # Forward audio encoder with batched audio embeds.
        batched_audio_hidden_states = _reference_audio_embedding_forward(
            self.audio_embed,
            input_ids=batched_input_ids,
            input_embeds=batched_audio_embeds,
            audio_embed_sizes=batched_audio_sizes,
            audio_attention_mask=batched_audio_attn_mask,
            wte=self.embed_tokens,
        )
    return batched_audio_hidden_states


def _main_encoding_batch_request(
    self, multimodal_params: List[MultimodalParams], mm_token_ids: torch.Tensor
) -> List[torch.FloatTensor]:
    # Batch input_ids.
    input_ids_list = [
        multimodal_params[i].multimodal_data["input_ids"] for i in range(len(multimodal_params))
    ]
    max_input_ids_len = max([input_ids.shape[1] for input_ids in input_ids_list])
    batched_input_ids = torch.full(
        (len(multimodal_params), max_input_ids_len), _PAD_TOKEN_ID, device=input_ids_list[0].device
    )
    for i, input_ids in enumerate(input_ids_list):
        batched_input_ids[i, : input_ids.shape[1]] = input_ids
    batched_input_ids = batched_input_ids.view(-1, max_input_ids_len)
    image_position_mask = batched_input_ids == _IMAGE_SPECIAL_TOKEN_ID
    non_image_position_mask = ~image_position_mask

    # Batch inference for image and audio embeds.
    batched_image_hidden_states = _main_batch_infer_image_embeds(
        self, batched_input_ids, multimodal_params
    )
    batched_audio_hidden_states = _main_batch_infer_audio_embeds(
        self, batched_input_ids, multimodal_params
    )

    # Combine different modalities into one.
    if batched_image_hidden_states is not None and batched_audio_hidden_states is not None:
        batched_hidden_states = batched_image_hidden_states * image_position_mask.unsqueeze(
            -1
        ) + batched_audio_hidden_states * non_image_position_mask.unsqueeze(-1)
    elif batched_image_hidden_states is not None:
        batched_hidden_states = batched_image_hidden_states
    elif batched_audio_hidden_states is not None:
        batched_hidden_states = batched_audio_hidden_states
    else:
        batched_hidden_states = self.embed_tokens(batched_input_ids)

    # Postprocessing to get multimodal-only embeddings.
    mm_token_mask = torch.isin(batched_input_ids, mm_token_ids)
    batched_hidden_states = batched_hidden_states[mm_token_mask]
    batched_hidden_states = [batched_hidden_states]
    return batched_hidden_states


def _main_payload(inputs, dtype):
    self = SimpleNamespace(dtype=dtype)
    # Package inputs for language model forward in AGGREGATE mode.
    multimodal_data = {}
    multimodal_data["input_ids"] = inputs["input_ids"]
    multimodal_data["input_image_embeds"] = inputs["input_image_embeds"].to(self.dtype)
    multimodal_data["image_sizes"] = inputs["image_sizes"]
    multimodal_data["image_attention_mask"] = inputs["image_attention_mask"]
    multimodal_data["input_audio_embeds"] = inputs["input_audio_embeds"]
    multimodal_data["audio_embed_sizes"] = inputs["audio_embed_sizes"]
    multimodal_data["audio_attention_mask"] = inputs["audio_attention_mask"]
    multimodal_data["audio_projection_mode"] = inputs["audio_projection_mode"]
    return multimodal_data


# ---------------------------------------------------------------------------
# A small CPU encoder
# ---------------------------------------------------------------------------

HIDDEN = 8  # decoder width
IMAGE_DIM = 6  # vision tower width
CROP_PIXELS = 32  # stub crop side; the vision forward never reads it
MASK_RESOLUTION = 32  # processor mask cells per crop side (448 // 14)
CROP_SIZE = 448  # pixels per crop side in image_sizes
AUDIO_FEATURES = 80
AUDIO_STRIDE = 8  # input frames per audio row


class _StubImageEmbedding(nn.Module):
    """What the vision forward reads, with a per-crop pooling tower."""

    def __init__(self):
        super().__init__()
        self.crop_size = CROP_SIZE
        self.base_feat_height_reduction = 1
        self.base_feat_height_target = MASK_RESOLUTION // 2
        self.image_dim_out = IMAGE_DIM
        self.hd_transform_order = "sub_glb"
        self.sub_GN = nn.Parameter(torch.randn(1, 1, 1, IMAGE_DIM))
        self.glb_GN = nn.Parameter(torch.randn(1, 1, IMAGE_DIM))
        self.img_projection = nn.Sequential(
            nn.Linear(IMAGE_DIM, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN)
        )
        self.tower = nn.Linear(3, IMAGE_DIM)
        self.drop = None

    def get_img_features(self, img_embeds, attention_mask):
        pooled = F.adaptive_avg_pool2d(img_embeds.float(), self.base_feat_height_target)
        tokens = self.tower(pooled.flatten(2).transpose(1, 2))
        return tokens * attention_mask[:, ::2, ::2].reshape(tokens.shape[0], -1, 1)


class _StubAudioEmbedding(nn.Module):
    """A causal per-clip audio tower: one row per AUDIO_STRIDE frames."""

    def __init__(self):
        super().__init__()
        self.audio_projection = nn.Linear(AUDIO_FEATURES, HIDDEN)
        self.drop = None

    def get_audio_features(
        self, input_embeds, audio_attention_mask, audio_projection_mode="speech"
    ):
        frames = input_embeds[:, ::AUDIO_STRIDE] * audio_attention_mask[:, ::AUDIO_STRIDE, None].to(
            input_embeds.dtype
        )
        return self.audio_projection(torch.cumsum(frames, dim=1))


class _CallCounter:
    """Counts calls through a callable while keeping its results."""

    def __init__(self, function):
        self.calls = 0
        self._function = function

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._function(*args, **kwargs)


@pytest.fixture
def encoder() -> HFPhi4MultimodalEncoder:
    torch.manual_seed(0)
    encoder = HFPhi4MultimodalEncoder.__new__(HFPhi4MultimodalEncoder)
    nn.Module.__init__(encoder)
    encoder.embed_tokens = nn.Embedding(_AUDIO_SPECIAL_TOKEN_ID + 1, HIDDEN)
    encoder.image_embed = _StubImageEmbedding()
    encoder.image_embed.forward = MethodType(vision_encoder_forward, encoder.image_embed)
    encoder.audio_embed = _StubAudioEmbedding()
    return encoder


# ---------------------------------------------------------------------------
# Requests: the processor outputs one prompt sees
# ---------------------------------------------------------------------------


def _image(seed: int, crops_h: int, crops_w: int, pad_rows: int, pad_cols: int):
    """One image as the image processor makes it: crops (global first), masks, size, rows."""
    mask = torch.ones(MASK_RESOLUTION * crops_h, MASK_RESOLUTION * crops_w)
    if pad_rows:
        mask[-pad_rows:, :] = 0
    if pad_cols:
        mask[:, -pad_cols:] = 0
    (sub_masks,), (rows,) = _reshape_attention_masks([mask], MASK_RESOLUTION)
    masks = torch.cat([torch.ones(1, MASK_RESOLUTION, MASK_RESOLUTION), sub_masks])
    torch.manual_seed(seed)
    crops = torch.randn(masks.shape[0], 3, CROP_PIXELS, CROP_PIXELS)
    return crops, masks, [CROP_SIZE * crops_h, CROP_SIZE * crops_w], rows


def _request(items: List[Tuple[str, Any]], seed: int) -> Dict[str, Any]:
    """One prompt's processor outputs; `items` in prompt order.

    Images are `("image", (crops_h, crops_w, pad_rows, pad_cols))` and audio
    clips `("audio", num_frames)`. Images are padded to the most crops of the
    prompt and clips to its longest clip, as the processors do.
    """
    images = [
        _image(seed * 10 + position, *spec)
        for position, (modality, spec) in enumerate(items)
        if modality == "image"
    ]
    torch.manual_seed(seed)
    clips = [
        torch.randn(frames, AUDIO_FEATURES) for modality, frames in items if modality == "audio"
    ]
    audio_rows = [math.ceil(clip.shape[0] / AUDIO_STRIDE) for clip in clips]

    prompt, image_index, audio_index = [1, 2], 0, 0
    for modality, _ in items:
        if modality == "image":
            prompt += [_IMAGE_SPECIAL_TOKEN_ID] * images[image_index][3]
            image_index += 1
        else:
            # Text after each clip; images stay adjacent.
            prompt += [_AUDIO_SPECIAL_TOKEN_ID] * audio_rows[audio_index] + [3]
            audio_index += 1
    prompt.append(4)

    inputs = {
        "input_ids": torch.tensor([prompt], dtype=torch.long),
        "input_image_embeds": torch.tensor([]),
        "image_sizes": torch.tensor([]),
        "image_attention_mask": torch.tensor([]),
        "input_audio_embeds": torch.tensor([]),
        "audio_embed_sizes": torch.tensor([]),
        "audio_attention_mask": None,
        "audio_projection_mode": "vision" if images else "speech",
    }
    image_inputs = {}
    if images:
        most_crops = max(crops.shape[0] for crops, *_ in images)
        inputs["input_image_embeds"] = torch.stack(
            [
                F.pad(crops, (0, 0, 0, 0, 0, 0, 0, most_crops - crops.shape[0]))
                for crops, *_ in images
            ]
        )
        inputs["image_attention_mask"] = torch.stack(
            [F.pad(masks, (0, 0, 0, 0, 0, most_crops - masks.shape[0])) for _, masks, *_ in images]
        )
        inputs["image_sizes"] = torch.tensor([size for _, _, size, _ in images], dtype=torch.long)
        image_inputs["num_img_tokens"] = torch.tensor([rows for *_, rows in images])
    if clips:
        frames = torch.tensor([clip.shape[0] for clip in clips])
        inputs["input_audio_embeds"] = pad_sequence(clips, batch_first=True)
        inputs["audio_embed_sizes"] = torch.tensor(audio_rows, dtype=torch.long)
        if len(clips) > 1:
            inputs["audio_attention_mask"] = torch.arange(int(frames.max())).unsqueeze(
                0
            ) < frames.unsqueeze(1)
    mm_data = {}
    if images:
        mm_data["image"] = [object()] * len(images)
    if clips:
        mm_data["audio"] = [object()] * len(clips)
    return {"inputs": inputs, "image_inputs": image_inputs, "mm_data": mm_data}


def _processed(request: Dict[str, Any]) -> MultimodalParams:
    """Run the real input processor over stubbed HF processors."""
    processor = Phi4MMInputProcessor.__new__(Phi4MMInputProcessor)
    processor._dtype = torch.float32
    processor._processor = SimpleNamespace(
        image_processor=lambda images, return_tensors: request["image_inputs"],
        audio_processor=lambda audios, return_tensors: {"processed": True},
    )
    processor._post_process = lambda image_inputs, audio_inputs, text: request["inputs"]
    prompt_ids, extra = processor.call_with_text_prompt(
        {"prompt": "", "multi_modal_data": request["mm_data"]}, None
    )
    assert prompt_ids == request["inputs"]["input_ids"][0].tolist()
    return MultimodalParams(multimodal_data=extra["multimodal_data"])


def _main_params(request: Dict[str, Any]) -> MultimodalParams:
    return MultimodalParams(multimodal_data=_main_payload(request["inputs"], torch.float32))


BATCHES = {
    "one_image": [[("image", (1, 2, 0, 6))]],
    "images_with_different_crops": [
        [("image", (1, 1, 4, 0))],
        [("image", (2, 2, 0, 10)), ("image", (1, 2, 0, 0))],
        [("image", (1, 3, 0, 0))],
    ],
    "audio_one_clip_and_several": [[("audio", 30)], [("audio", 17), ("audio", 64)]],
    "mixed_in_and_across_requests": [
        [("audio", 40), ("image", (1, 2, 0, 4)), ("image", (1, 1, 0, 0))],
        [("image", (2, 1, 6, 0))],
        [("audio", 9), ("audio", 25)],
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_processor_emits_modality_payloads_and_item_record():
    request = _request(BATCHES["mixed_in_and_across_requests"][0], seed=0)
    data = _processed(request).multimodal_data
    inputs = request["inputs"]

    image = data["image"]
    assert torch.equal(image["input_image_embeds"], inputs["input_image_embeds"])
    assert torch.equal(image["image_attention_mask"], inputs["image_attention_mask"])
    assert image["image_attention_mask"].dtype == image["input_image_embeds"].dtype
    assert torch.equal(image["image_sizes"], inputs["image_sizes"])
    audio = data["audio"]
    assert torch.equal(audio["input_audio_embeds"], inputs["input_audio_embeds"])
    # A single clip has no processor mask; the payload always carries one.
    assert inputs["audio_attention_mask"] is None
    assert torch.equal(audio["audio_attention_mask"], torch.ones(1, 40))
    assert torch.equal(audio["audio_embed_sizes"], inputs["audio_embed_sizes"])
    assert "input_ids" not in data

    metadata = data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    # Prompt order, with two adjacent image runs.
    assert metadata.item_refs == (("audio", 0), ("image", 0), ("image", 1))
    assert metadata.output_embedding_lengths == (
        5,
        *request["image_inputs"]["num_img_tokens"].tolist(),
    )
    assert Phi4MMForCausalLM.multimodal_data_device_paths.fget(None) == [
        "image.input_image_embeds",
        "image.image_attention_mask",
        "audio.input_audio_embeds",
        "audio.audio_attention_mask",
    ]


@pytest.mark.parametrize("batch", list(BATCHES))
def test_groups_match_the_previous_encode_path(encoder, batch):
    requests = [_request(items, seed) for seed, items in enumerate(BATCHES[batch])]
    mm_token_ids = torch.tensor([_IMAGE_SPECIAL_TOKEN_ID, _AUDIO_SPECIAL_TOKEN_ID])
    (expected,) = _main_encoding_batch_request(
        encoder, [_main_params(request) for request in requests], mm_token_ids
    )
    params = [_processed(request) for request in requests]
    image_calls = _CallCounter(encoder.image_embed.forward)
    audio_calls = _CallCounter(encoder.audio_embed.get_audio_features)
    encoder.image_embed.forward = image_calls
    encoder.audio_embed.get_audio_features = audio_calls

    (encoded,) = encoder.forward(params)

    present = {modality for items in BATCHES[batch] for modality, _ in items}
    # One image encoder call and one audio encoder call per batch, as before.
    assert image_calls.calls == int("image" in present)
    assert audio_calls.calls == int("audio" in present)
    assert encoded.shape[0] == sum(
        sum(param.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY].output_embedding_lengths)
        for param in params
    )
    assert torch.equal(encoded, expected)


def test_batched_rows_equal_per_request_rows(encoder):
    """Each request's rows match encoding it alone, up to padding and GEMM shape."""
    params = [
        _processed(_request(items, seed))
        for seed, items in enumerate(BATCHES["mixed_in_and_across_requests"])
    ]
    (batched,) = encoder.forward(params)
    alone = torch.cat([encoder.forward([param])[0] for param in params])
    torch.testing.assert_close(batched, alone, rtol=1e-5, atol=1e-5)
