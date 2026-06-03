# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3VL multimodal item extractor and bucket adapters.

The extractor parses the two wire keys (`multimodal_item_order` +
`multimodal_embedding_lengths`) into one transient `MixedModalEncodeContext` via
`from_metadata`, walks `ctx.order`, and yields one canonical six-field
`ModalityItem` per prompt slot. A `ModalityItem` owns exactly one slot
(`prompt_pos`); its `rows` is both the encoder-output row count and its scatter
footprint. Per-item payload slicing (one image / one video out of an aggregate
`pixel_values` + `*_grid_thw` blob) goes through `_qwen3vl_slice_payload`; the
per-grid post-merge token count is `_qwen3vl_grid_rows`. When the extractor has a
`MixedModalEncodeContext`, rows come from `ctx.embedding_lengths`; the grid-row
helper is the fallback for pure single-modality requests that carry no
`multimodal_item_order` key (so `from_metadata` returns None and the extractor
synthesizes the modality-major default order).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from _mm_encode_helpers import _make_param

from tensorrt_llm._torch.models.mixed_modal_encode import ModalityItem
from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VisionModelBase,
    _qwen3vl_extract_items,
    _qwen3vl_grid_rows,
    _qwen3vl_slice_payload,
)


class TestQwen3VLPayloadSlicer:
    """The grid-row + slice helpers are the per-model source of truth for
    Qwen3VL: `_qwen3vl_slice_payload` slices the aggregate `pixel_values` (+
    `*_grid_thw`) blob into per-item encoder inputs by the raw-patch prefix sum
    `Sigma prod(grid_thw[:i])`, and `_qwen3vl_grid_rows` reports the per-grid
    post-merge token count `t * (h // merge) * (w // merge)`.
    """

    def test_rows_for_is_per_grid_post_merge_count(self):
        # Two image sub-items: grids [1,16,16] and [1,8,8] at merge=4 ->
        # 1*(16//4)*(16//4)=16 and 1*(8//4)*(8//4)=4 post-merge rows.
        payload = {
            "pixel_values": torch.randn(1 * 16 * 16 + 1 * 8 * 8, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 8, 8]]),
        }
        param = _make_param({"image": payload})
        assert _qwen3vl_grid_rows(param, "image", 0, spatial_merge_size=4) == 16
        assert _qwen3vl_grid_rows(param, "image", 1, spatial_merge_size=4) == 4

    def test_slice_payload_slices_pixels_by_raw_patch_prefix_sum(self):
        # Raw-patch rows per grid = t*h*w: 256 then 64. The first item's pixel
        # slice is rows [0:256), the second is [256:320); each carries its own
        # single-row `*_grid_thw`.
        pv = torch.arange(320 * 2, dtype=torch.float32).reshape(320, 2)
        payload = {
            "pixel_values": pv,
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 8, 8]]),
        }
        param = _make_param({"image": payload})

        s0 = _qwen3vl_slice_payload(param, "image", 0, spatial_merge_size=4)
        s1 = _qwen3vl_slice_payload(param, "image", 1, spatial_merge_size=4)
        torch.testing.assert_close(s0["pixel_values"], pv[0:256])
        torch.testing.assert_close(s1["pixel_values"], pv[256:320])
        assert s0["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert s1["image_grid_thw"].tolist() == [[1, 8, 8]]

    def test_slice_payload_video_uses_video_keys(self):
        pv = torch.arange(64 * 2, dtype=torch.float32).reshape(64, 2)
        payload = {
            "pixel_values_videos": pv,
            "video_grid_thw": torch.tensor([[1, 8, 8]]),
        }
        param = _make_param({"video": payload})

        s0 = _qwen3vl_slice_payload(param, "video", 0, spatial_merge_size=4)
        torch.testing.assert_close(s0["pixel_values_videos"], pv[0:64])
        assert s0["video_grid_thw"].tolist() == [[1, 8, 8]]
        # merge=4 -> 1*(8//4)*(8//4) = 4 post-merge rows.
        assert _qwen3vl_grid_rows(param, "video", 0, spatial_merge_size=4) == 4


class TestQwen3VLExtractItems:
    @pytest.mark.parametrize(
        "modality, payload, expected_rows",
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
        ],
        ids=["image"],
    )
    def test_pure_single_modality(self, modality, payload, expected_rows):
        param = _make_param({modality: payload})
        items = list(_qwen3vl_extract_items(0, param))
        assert len(items) == 1
        assert items[0].modality == modality
        assert items[0].mm_idx_per_modality == 0
        assert items[0].prompt_pos == 0
        assert items[0].rows == expected_rows
        assert items[0].src_param_idx == 0

    def test_pure_single_modality_multi_item_enumerates_all(self):
        # A SINGLE modality (image) carrying TWO sub-items, with NO
        # `multimodal_item_order` key (a plain 2-image prompt on the
        # direct/non-hashing path). `from_metadata` returns None, so the
        # extractor synthesizes the modality-major default order. That default
        # must enumerate EVERY sub-item (one per `*_grid_thw` row), not collapse
        # to a single `(image, 0)` entry — otherwise the 2nd image is never
        # sliced or encoded.
        merge = 4
        # grids [1,16,16] -> 16 post-merge rows (256 raw patches) and
        #       [1, 8, 8] ->  4 post-merge rows ( 64 raw patches).
        payload = {
            "pixel_values": torch.randn(256 + 64, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 8, 8]]),
        }
        param = _make_param({"image": payload})
        items = list(_qwen3vl_extract_items(0, param, spatial_merge_size=merge))
        assert len(items) == 2
        assert [(it.modality, it.mm_idx_per_modality, it.prompt_pos) for it in items] == [
            ("image", 0, 0),
            ("image", 1, 1),
        ]
        # Per-grid post-merge row counts (grid-driven fallback, no context).
        assert [it.rows for it in items] == [16, 4]
        # Each item carries its own sliced single-grid payload, so the bucket
        # adapter re-concatenation reproduces the original aggregate.
        assert items[0].payload["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert items[0].payload["pixel_values"].shape[0] == 256
        assert items[1].payload["image_grid_thw"].tolist() == [[1, 8, 8]]
        assert items[1].payload["pixel_values"].shape[0] == 64

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
        # video at prompt slot 0, image at slot 1 — distinct, no collapse.
        by_modality = {it.modality: it for it in items}
        assert by_modality["video"].prompt_pos == 0
        assert by_modality["image"].prompt_pos == 1
        assert by_modality["video"].mm_idx_per_modality == 0
        assert by_modality["image"].mm_idx_per_modality == 0
        assert by_modality["video"].rows == 8
        assert by_modality["image"].rows == 5

    def test_image_video_image_row_order(self):
        # Interleaved repeated modality: image -> video -> image. The trailing
        # image (prompt slot 2) must land AFTER the video (slot 1), not folded
        # into the leading image block. Each image sub-item is sliced out of the
        # aggregate `pixel_values`/`image_grid_thw` by grid; `rows` is the
        # per-grid post-merge count (grid-driven, the Qwen3VL source of truth).
        merge = 4
        # image grids: [1,16,16] -> 16 post-merge rows (256 raw patches);
        #              [1, 8, 8] ->  4 post-merge rows ( 64 raw patches).
        image_payload = {
            "pixel_values": torch.randn(256 + 64, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 8, 8]]),
        }
        # video grid: [2,16,16] -> 2*(16//4)*(16//4) = 32 post-merge rows
        #             (2*16*16 = 512 raw patches).
        video_payload = {
            "pixel_values_videos": torch.randn(512, 1176),
            "video_grid_thw": torch.tensor([[2, 16, 16]]),
        }
        param = _make_param(
            {
                "image": image_payload,
                "video": video_payload,
                "multimodal_item_order": [
                    {"modality": "image", "index": 0},
                    {"modality": "video", "index": 0},
                    {"modality": "image", "index": 1},
                ],
                # The transient MixedModalEncodeContext requires the per-slot row
                # counts alongside the order (length-agreement is validated in
                # `__post_init__`). Rows still come from the grid via `rows_for`
                # until Task 5b sources them from the context; these match.
                "multimodal_embedding_lengths": [16, 32, 4],
            }
        )
        items = list(_qwen3vl_extract_items(0, param, spatial_merge_size=merge))
        assert len(items) == 3
        # Items are emitted in prompt order, one per slot.
        assert [(it.modality, it.mm_idx_per_modality, it.prompt_pos) for it in items] == [
            ("image", 0, 0),
            ("video", 0, 1),
            ("image", 1, 2),
        ]
        # Per-grid post-merge row counts.
        assert [it.rows for it in items] == [16, 32, 4]
        # The trailing image carries its own sliced single-grid payload (not the
        # leading image's), so the bucket adapter re-concatenation is faithful.
        assert items[2].payload["image_grid_thw"].tolist() == [[1, 8, 8]]
        assert items[2].payload["pixel_values"].shape[0] == 64
        assert items[0].payload["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert items[0].payload["pixel_values"].shape[0] == 256

    def test_mixed_order_length_mismatch_raises(self):
        # The extractor now builds a transient MixedModalEncodeContext from the
        # two wire keys, so a `multimodal_item_order` whose length disagrees with
        # `multimodal_embedding_lengths` is rejected at construction time (the
        # context's `__post_init__` length-agreement check). The old
        # `MultimodalPromptOrder.from_metadata` path ignored the lengths entirely
        # and would have yielded items, so this guards the substitution.
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
                # 2-entry order, 1-entry lengths -> the typed view rejects it.
                "multimodal_item_order": [("video", 0), ("image", 0)],
                "multimodal_embedding_lengths": [8],
            }
        )
        with pytest.raises(ValueError, match="embedding_lengths"):
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
                        "image",
                        0,
                        0,
                        5,
                        {
                            "pixel_values": torch.randn(20, 1176),
                            "image_grid_thw": torch.tensor([[1, 16, 16]]),
                            "num_tokens": 5,
                        },
                    ),
                    ModalityItem(
                        1,
                        "image",
                        0,
                        1,
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
                        "video",
                        0,
                        0,
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
