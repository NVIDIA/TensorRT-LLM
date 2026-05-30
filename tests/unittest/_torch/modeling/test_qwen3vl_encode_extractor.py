# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3VL multimodal item extractor and bucket adapters."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VisionModelBase, _qwen3vl_extract_items
from tensorrt_llm._torch.models.multimodal_encoding import MultimodalItem
from tensorrt_llm.inputs.multimodal import MultimodalParams


def _make_param(multimodal_data: dict) -> MultimodalParams:
    return MultimodalParams(
        multimodal_input=None,
        multimodal_data=multimodal_data,
        multimodal_runtime=None,
    )


class TestQwen3VLExtractItems:
    def test_pure_image(self):
        payload = {
            "pixel_values": torch.randn(20, 1176),  # 20 patches -> 5 tokens at merge=4
            "image_grid_thw": torch.tensor([[1, 16, 16]]),
            "num_tokens": 5,  # explicit test convention, like Nano Task 7
        }
        param = _make_param({"image": payload})
        items = list(_qwen3vl_extract_items(0, param))
        assert len(items) == 1
        assert items[0].modality == "image"
        assert items[0].item_idx_in_param == 0
        assert items[0].token_count == 5
        assert items[0].src_param_idx == 0

    def test_pure_video(self):
        payload = {
            "pixel_values_videos": torch.randn(32, 1176),
            "video_grid_thw": torch.tensor([[2, 16, 16]]),
            "num_tokens": 8,
        }
        param = _make_param({"video": payload})
        items = list(_qwen3vl_extract_items(0, param))
        assert len(items) == 1
        assert items[0].modality == "video"
        assert items[0].token_count == 8

    def test_mixed_image_video(self):
        payload_image = {
            "pixel_values": torch.randn(20, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16]]),
            "num_tokens": 5,
        }
        payload_video = {
            "pixel_values_videos": torch.randn(32, 1176),
            "video_grid_thw": torch.tensor([[2, 16, 16]]),
            "num_tokens": 8,
        }
        param = _make_param(
            {
                "image": payload_image,
                "video": payload_video,
                "multimodal_item_order": [("video", 0), ("image", 0)],
                "multimodal_embedding_lengths": [8, 5],
            }
        )
        items = list(_qwen3vl_extract_items(0, param))
        assert len(items) == 2
        positions = {it.modality: it.item_idx_in_param for it in items}
        assert positions == {"video": 0, "image": 1}

    def test_no_items_yields_empty(self):
        param = _make_param({})
        assert list(_qwen3vl_extract_items(0, param)) == []


class TestQwen3VLExtractItemsDictFormOrder:
    def test_mixed_image_video_dict_form_item_order(self):
        # Runtime registry emits dict-form item order; extractor must
        # normalize it (regression guard for the tuple(pair) bug).
        payload_image = {
            "pixel_values": torch.randn(20, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16]]),
            "num_tokens": 5,
        }
        payload_video = {
            "pixel_values_videos": torch.randn(32, 1176),
            "video_grid_thw": torch.tensor([[2, 16, 16]]),
            "num_tokens": 8,
        }
        param = _make_param(
            {
                "image": payload_image,
                "video": payload_video,
                # DICT-form order entries (as the registry emits them):
                "multimodal_item_order": [
                    {"modality": "video", "index": 0},
                    {"modality": "image", "index": 0},
                ],
                "multimodal_embedding_lengths": [8, 5],
            }
        )
        items = list(_qwen3vl_extract_items(0, param))
        assert len(items) == 2
        positions = {it.modality: it.item_idx_in_param for it in items}
        # video at prompt slot 0, image at slot 1 — distinct, no collapse
        assert positions == {"video": 0, "image": 1}


class TestQwen3VLBucketAdapters:
    def _make_encoder_stub(self, encode_visual_inputs_return):
        enc = MagicMock(spec=Qwen3VisionModelBase)
        enc._encode_visual_inputs = MagicMock(return_value=encode_visual_inputs_return)
        enc._adapter_image_bucket = Qwen3VisionModelBase._adapter_image_bucket.__get__(enc)
        enc._adapter_video_bucket = Qwen3VisionModelBase._adapter_video_bucket.__get__(enc)
        return enc

    def test_image_adapter_stacks_pixel_values_and_grids(self):
        H = 1024
        item0 = MultimodalItem(
            0,
            0,
            "image",
            5,
            {
                "pixel_values": torch.randn(20, 1176),
                "image_grid_thw": torch.tensor([[1, 16, 16]]),
                "num_tokens": 5,
            },
        )
        item1 = MultimodalItem(
            1,
            0,
            "image",
            3,
            {
                "pixel_values": torch.randn(12, 1176),
                "image_grid_thw": torch.tensor([[1, 12, 12]]),
                "num_tokens": 3,
            },
        )
        out_tensor = torch.randn(8, H)  # 5 + 3 tokens
        enc = self._make_encoder_stub(out_tensor)
        result = enc._adapter_image_bucket([item0, item1], [object(), object()])
        # Verify _encode_visual_inputs was called once with concatenated tensors
        call_args = enc._encode_visual_inputs.call_args[0]
        pixel_values_arg, grid_arg = call_args[0], call_args[1]
        assert pixel_values_arg.shape == (32, 1176)  # 20 + 12
        assert grid_arg.shape == (2, 3)
        torch.testing.assert_close(result, out_tensor)

    def test_video_adapter_stacks_pixel_values_and_grids(self):
        H = 1024
        item = MultimodalItem(
            0,
            0,
            "video",
            8,
            {
                "pixel_values_videos": torch.randn(32, 1176),
                "video_grid_thw": torch.tensor([[2, 16, 16]]),
                "num_tokens": 8,
            },
        )
        out_tensor = torch.randn(8, H)
        enc = self._make_encoder_stub(out_tensor)
        result = enc._adapter_video_bucket([item], [object()])
        call_args = enc._encode_visual_inputs.call_args[0]
        assert call_args[0].shape == (32, 1176)
        assert call_args[1].shape == (1, 3)
        torch.testing.assert_close(result, out_tensor)
