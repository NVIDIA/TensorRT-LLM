# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3VL multimodal item extractor and bucket adapters."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VisionModelBase, _qwen3vl_extract_items
from tensorrt_llm._torch.models.multimodal_encoding import ModalityItem
from tensorrt_llm.inputs.multimodal import MultimodalParams


def _make_param(multimodal_data: dict) -> MultimodalParams:
    return MultimodalParams(
        multimodal_input=None,
        multimodal_data=multimodal_data,
        multimodal_runtime=None,
    )


class TestQwen3VLExtractItems:
    @pytest.mark.parametrize(
        "modality, payload, expected_token_count",
        [
            (
                "image",
                {
                    # 20 patches -> 5 tokens at merge=4
                    "pixel_values": torch.randn(20, 1176),
                    "image_grid_thw": torch.tensor([[1, 16, 16]]),
                    "num_tokens": 5,  # explicit test convention, like Nano Task 7
                },
                5,
            ),
            (
                "video",
                {
                    "pixel_values_videos": torch.randn(32, 1176),
                    "video_grid_thw": torch.tensor([[2, 16, 16]]),
                    "num_tokens": 8,
                },
                8,
            ),
        ],
        ids=["image", "video"],
    )
    def test_pure_single_modality(self, modality, payload, expected_token_count):
        param = _make_param({modality: payload})
        items = list(_qwen3vl_extract_items(0, param))
        assert len(items) == 1
        assert items[0].modality == modality
        assert items[0].item_idx_in_param == 0
        assert items[0].token_count == expected_token_count
        assert items[0].src_param_idx == 0

    @pytest.mark.parametrize(
        "item_order",
        [
            # Tuple(pair)-form order entries.
            [("video", 0), ("image", 0)],
            # Dict-form order entries (as the runtime registry emits them).
            # Regression guard for the tuple(pair) bug: the extractor must
            # normalize dict-form order identically to tuple-form.
            [{"modality": "video", "index": 0}, {"modality": "image", "index": 0}],
        ],
        ids=["tuple_form", "dict_form_regression"],
    )
    def test_mixed_image_video(self, item_order):
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
                "multimodal_item_order": item_order,
                "multimodal_embedding_lengths": [8, 5],
            }
        )
        items = list(_qwen3vl_extract_items(0, param))
        assert len(items) == 2
        positions = {it.modality: it.item_idx_in_param for it in items}
        # video at prompt slot 0, image at slot 1 — distinct, no collapse
        assert positions == {"video": 0, "image": 1}

    def test_no_items_yields_empty(self):
        param = _make_param({})
        assert list(_qwen3vl_extract_items(0, param)) == []

    def test_mixed_interleaved_repeated_modality_raises(self):
        # An interleaved repeated modality (image -> video -> image) cannot be
        # scattered correctly by the one-aggregate-item-per-modality extractor:
        # the trailing image would fold into the leading image block. The
        # extractor must reject it fail-loud rather than silently mis-scatter.
        payload_image = {
            "pixel_values": torch.randn(40, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 16, 16]]),
            "num_tokens": 10,
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
                "multimodal_item_order": [
                    {"modality": "image", "index": 0},
                    {"modality": "video", "index": 0},
                    {"modality": "image", "index": 1},
                ],
                "multimodal_embedding_lengths": [5, 8, 5],
            }
        )
        with pytest.raises(ValueError, match="interleaved repeated modality"):
            list(_qwen3vl_extract_items(0, param))


class TestQwen3VLBucketAdapters:
    def _make_encoder_stub(self, encode_visual_inputs_return):
        enc = MagicMock(spec=Qwen3VisionModelBase)
        enc._encode_visual_inputs = MagicMock(return_value=encode_visual_inputs_return)
        enc._adapter_image_bucket = Qwen3VisionModelBase._adapter_image_bucket.__get__(enc)
        enc._adapter_video_bucket = Qwen3VisionModelBase._adapter_video_bucket.__get__(enc)
        return enc

    @pytest.mark.parametrize(
        "adapter_attr, items, expected_grid_shape",
        [
            (
                # Image bucket: two items stack into one (32, 1176) call with a
                # (2, 3) grid (20 + 12 patches across two grid rows).
                "_adapter_image_bucket",
                [
                    ModalityItem(
                        0,
                        0,
                        "image",
                        5,
                        {
                            "pixel_values": torch.randn(20, 1176),
                            "image_grid_thw": torch.tensor([[1, 16, 16]]),
                            "num_tokens": 5,
                        },
                    ),
                    ModalityItem(
                        1,
                        0,
                        "image",
                        3,
                        {
                            "pixel_values": torch.randn(12, 1176),
                            "image_grid_thw": torch.tensor([[1, 12, 12]]),
                            "num_tokens": 3,
                        },
                    ),
                ],
                (2, 3),
            ),
            (
                # Video bucket: one item -> (32, 1176) call with a (1, 3) grid.
                "_adapter_video_bucket",
                [
                    ModalityItem(
                        0,
                        0,
                        "video",
                        8,
                        {
                            "pixel_values_videos": torch.randn(32, 1176),
                            "video_grid_thw": torch.tensor([[2, 16, 16]]),
                            "num_tokens": 8,
                        },
                    ),
                ],
                (1, 3),
            ),
        ],
        ids=["image", "video"],
    )
    def test_adapter_stacks_pixel_values_and_grids(self, adapter_attr, items, expected_grid_shape):
        H = 1024
        out_tensor = torch.randn(8, H)  # total tokens across the bucket
        enc = self._make_encoder_stub(out_tensor)
        result = getattr(enc, adapter_attr)(items, [object()] * len(items))
        # _encode_visual_inputs is called once with the concatenated tensors.
        call_args = enc._encode_visual_inputs.call_args[0]
        pixel_values_arg, grid_arg = call_args[0], call_args[1]
        assert pixel_values_arg.shape == (32, 1176)
        assert grid_arg.shape == expected_grid_shape
        torch.testing.assert_close(result, out_tensor)
