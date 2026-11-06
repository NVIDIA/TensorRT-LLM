# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import List

import pytest
import torch
from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches

from tensorrt_llm._torch.models.modeling_llava_next import (
    LlavaNextInputProcessor,
    LlavaNextVisionModel,
)
from tensorrt_llm.inputs.multimodal import MULTIMODAL_ENCODER_ITEM_METADATA_KEY, MultimodalParams

pytestmark = pytest.mark.cpu_only


class _MainLlavaNextEncoder:
    """The upstream request packer and encoder entry, copied verbatim."""

    def _pad_for_batching(
        self,
        pixel_values: list[torch.Tensor],
    ):
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`list[torch.Tensor]`):
                A list of pixel value tensors, each of shape (`batch`, `num_patches`, `channels`, `height`, `width`)

        Returns:
            list[`torch.Tensor`]: The padded image tensors.
        """
        if not pixel_values:
            return pixel_values

        # Find max patches across all images
        max_patches = max(tensor.shape[1] for tensor in pixel_values)

        # Pad each tensor by concatenating zeros on dim=1 (num_patches)
        padded_values = []

        for pixel_value in pixel_values:
            current_patches = pixel_value.shape[1]
            if current_patches < max_patches:
                pad_len = max_patches - current_patches
                zeros = pixel_value.new_zeros(
                    (pixel_value.shape[0], pad_len, *pixel_value.shape[2:])
                )
                padded_pixel_value = torch.cat([pixel_value, zeros], dim=1)
            else:
                padded_pixel_value = pixel_value

            padded_values.append(padded_pixel_value)

        return padded_values

    @torch.inference_mode()
    def encode_like_main(self, multimodal_params: List[MultimodalParams]):
        pixel_values = [
            multimodal_param.multimodal_data["image"]["pixel_values"]
            for multimodal_param in multimodal_params
        ]
        image_sizes = [
            multimodal_param.multimodal_data["image"]["image_sizes"]
            for multimodal_param in multimodal_params
        ]
        pixel_values = self._pad_for_batching(pixel_values)
        pixel_values = torch.cat(pixel_values, dim=0)
        image_sizes = torch.cat(image_sizes, dim=0)

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.pretrained_config.image_grid_pinpoints,
                patch_size=self.config.image_size,
            )
            for imsize in image_sizes
        ]

        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [
                pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            pixel_values = torch.cat(_pixel_values_list, dim=0)

        attn_metadata = self.vision_model.prepare_attn_metadata(pixel_values.shape[0])
        image_features = self.vision_model(
            pixel_values,
            attn_metadata=attn_metadata,
        )
        selected_image_feature = image_features[-2][:, 1:]
        image_features = self.mm_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)

        # NOTE: 'pack_image_features' is from the HF's code
        image_features, _ = self.pack_image_features(
            image_features,
            image_sizes,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            image_newline=self.image_newline,
        )
        image_features = torch.cat(image_features, dim=0)
        return [image_features]


class _StubVisionTower:
    def __init__(self):
        self.calls = []

    def prepare_attn_metadata(self, num_patches):
        return num_patches

    def __call__(self, pixel_values, attn_metadata):
        assert attn_metadata == pixel_values.shape[0]
        self.calls.append(pixel_values.clone())
        channels = pixel_values.mean(dim=(-2, -1))
        features = channels[:, None, :] + torch.arange(5, dtype=pixel_values.dtype)[None, :, None]
        return [None, features, None]


class _CpuLlavaNextEncoder(_MainLlavaNextEncoder):
    mm_encoder_groups = LlavaNextVisionModel.mm_encoder_groups
    forward = LlavaNextVisionModel.forward
    pack_image_features = LlavaNextVisionModel.pack_image_features

    def __init__(self):
        self.config = SimpleNamespace(image_size=4, patch_size=2)
        self.pretrained_config = SimpleNamespace(
            vision_config=self.config,
            image_grid_pinpoints=[[4, 4], [4, 8], [8, 4], [8, 8]],
        )
        self.vision_feature_select_strategy = "default"
        self.vision_model = _StubVisionTower()
        self.mm_projector = torch.nn.Identity()
        self.image_newline = torch.tensor([-1.0, -2.0, -3.0])
        self.input_calls = []

    def encode_batched(self, pixel_values, image_sizes):
        self.input_calls.append((pixel_values.clone(), image_sizes.clone()))
        return LlavaNextVisionModel.encode_batched(self, pixel_values, image_sizes)


def _processor(config):
    processor = LlavaNextInputProcessor.__new__(LlavaNextInputProcessor)
    processor._config = config
    return processor


def _request(encoder, sizes, seed, dtype=torch.float32):
    num_patches = [
        image_size_to_num_patches(size, encoder.pretrained_config.image_grid_pinpoints, 4)
        for size in sizes
    ]
    pixels = torch.zeros(len(sizes), max(num_patches), 3, 4, 4, dtype=dtype)
    for index, count in enumerate(num_patches):
        pixels[index, :count] = (
            torch.arange(count * 3 * 4 * 4, dtype=dtype).reshape(count, 3, 4, 4) / 64 + seed
        )
    data = {"image": {"pixel_values": pixels, "image_sizes": torch.tensor(sizes)}}
    metadata = _processor(encoder.pretrained_config).get_mm_item_metadata([], data)
    data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = metadata
    return MultimodalParams(multimodal_data=data)


@pytest.mark.parametrize(
    "request_sizes",
    [
        [[(4, 4)]],
        [[(4, 4)], [(8, 8)], [(3, 8)]],
        [[(4, 8), (3, 8)], [(8, 4)], [(4, 4), (8, 8)]],
    ],
)
def test_llava_next_groups_match_main(request_sizes):
    encoder = _CpuLlavaNextEncoder()
    params = [_request(encoder, sizes, index) for index, sizes in enumerate(request_sizes)]
    expected = torch.cat([encoder.encode_like_main([param])[0] for param in params])
    oracle_pixels = encoder._pad_for_batching(
        [param.multimodal_data["image"]["pixel_values"] for param in params]
    )
    oracle_sizes = torch.cat([param.multimodal_data["image"]["image_sizes"] for param in params])
    encoder.vision_model.calls.clear()

    output = encoder.forward(params)

    assert len(output) == 1
    assert torch.equal(output[0], expected)
    assert len(encoder.vision_model.calls) == len(encoder.input_calls) == 1
    pixels, sizes = encoder.input_calls[0]
    assert torch.equal(pixels, torch.cat(oracle_pixels))
    assert torch.equal(sizes, oracle_sizes)
    assert pixels.dtype == oracle_pixels[0].dtype
    assert pixels.device == oracle_pixels[0].device
    assert sizes.dtype == oracle_sizes.dtype
    assert sizes.device == oracle_sizes.device
    assert output[0].shape[0] == sum(
        sum(param.item_metadata.output_embedding_lengths) for param in params
    )


def test_llava_next_metadata_without_hashing_counts_unpadding_and_newlines():
    encoder = _CpuLlavaNextEncoder()
    param = _request(encoder, [(4, 4), (3, 8), (8, 3)], seed=3)
    data = param.multimodal_data
    metadata = data.pop(MULTIMODAL_ENCODER_ITEM_METADATA_KEY)
    expected_rows = []
    for index in range(3):
        item = MultimodalParams(
            multimodal_data={
                "image": {name: tensor[index : index + 1] for name, tensor in data["image"].items()}
            }
        )
        expected_rows.append(encoder.encode_like_main([item])[0].shape[0])

    processor = _processor(encoder.pretrained_config)
    rebuilt = processor.get_mm_item_metadata([], data)
    assert rebuilt.item_refs == (("image", 0), ("image", 1), ("image", 2))
    assert rebuilt.output_embedding_lengths == tuple(expected_rows)
    assert rebuilt == metadata
    assert processor.get_mm_item_metadata([], {}) is None
    data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = metadata
    assert processor.get_mm_item_metadata([], data) is metadata


def test_llava_next_dtype_groups_restore_request_order():
    encoder = _CpuLlavaNextEncoder()
    params = [
        _request(encoder, [(4, 4)], 0),
        _request(encoder, [(3, 8)], 1, dtype=torch.float64),
        _request(encoder, [(8, 8)], 2),
    ]
    expected = torch.cat([encoder.encode_like_main([param])[0] for param in params])
    encoder.vision_model.calls.clear()
    actual = encoder.forward(params)[0]
    assert torch.equal(actual, expected)
    assert len(encoder.vision_model.calls) == len(encoder.input_calls) == 2
