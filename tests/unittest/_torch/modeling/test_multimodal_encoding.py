# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tensorrt_llm._torch.models.multimodal_encoding."""

from __future__ import annotations

import pytest

from tensorrt_llm._torch.models.multimodal_encoding import EncodingPlan, MultimodalItem


class TestMultimodalItem:
    def test_basic_fields(self):
        item = MultimodalItem(
            src_param_idx=2,
            item_idx_in_param=1,
            modality="image",
            token_count=5,
            payload={"data": "x"},
        )
        assert item.src_param_idx == 2
        assert item.item_idx_in_param == 1
        assert item.modality == "image"
        assert item.token_count == 5
        assert item.payload == {"data": "x"}

    def test_ghost_sentinel(self):
        item = MultimodalItem(
            src_param_idx=0,
            item_idx_in_param=-1,
            modality="audio",
            token_count=4,
            payload={"data": "y"},
        )
        assert item.item_idx_in_param == -1

    def test_is_frozen(self):
        item = MultimodalItem(
            src_param_idx=0,
            item_idx_in_param=0,
            modality="image",
            token_count=1,
            payload={},
        )
        with pytest.raises(Exception):  # FrozenInstanceError subclass of AttributeError
            item.src_param_idx = 99


def _identity_extractor(items_by_param):
    """Test helper: yield pre-built MultimodalItems passed in by-param."""

    def extract(param_idx, _param):
        yield from items_by_param[param_idx]

    return extract


class TestEncodingPlanPartition:
    def test_empty_batch(self):
        plan = EncodingPlan.from_params(
            multimodal_params=[],
            extract=lambda i, p: iter([]),
        )
        assert plan.total_tokens == 0
        assert plan.active_modalities == []
        assert len(plan.items) == 0

    def test_single_pure_image(self):
        items_by_param = {
            0: [MultimodalItem(0, 0, "image", 5, {"id": "img_A"})],
        }
        plan = EncodingPlan.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        assert plan.total_tokens == 5
        assert plan.active_modalities == ["image"]
        assert plan._param_lengths.tolist() == [5]
        assert plan._param_offsets.tolist() == [0]
        assert plan._modality_slots["image"].tolist() == [0]
        assert plan._bucket_offsets["image"].tolist() == [0, 5]

    def test_mixed_image_audio(self):
        items_by_param = {
            0: [
                MultimodalItem(0, 0, "image", 5, {"id": "img_A"}),
                MultimodalItem(0, 1, "audio", 4, {"id": "aud_A"}),
                MultimodalItem(0, 2, "image", 5, {"id": "img_B"}),
            ],
            1: [
                MultimodalItem(1, 0, "image", 5, {"id": "img_C"}),
            ],
        }
        plan = EncodingPlan.from_params(
            multimodal_params=[object(), object()],
            extract=_identity_extractor(items_by_param),
        )
        assert plan.total_tokens == 19
        assert set(plan.active_modalities) == {"image", "audio"}
        assert plan._param_lengths.tolist() == [14, 5]
        assert plan._param_offsets.tolist() == [0, 14]
        assert plan._modality_slots["image"].tolist() == [0, 2, 3]
        assert plan._modality_slots["audio"].tolist() == [1]
        assert plan._bucket_offsets["image"].tolist() == [0, 5, 10, 15]
        assert plan._bucket_offsets["audio"].tolist() == [0, 4]

    def test_ghost_item_excluded_from_param_length(self):
        items_by_param = {
            0: [
                MultimodalItem(0, 0, "video", 5, {"id": "vid_A"}),
                MultimodalItem(0, -1, "audio", 4, {"id": "vid_A.audio"}),
            ],
        }
        plan = EncodingPlan.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        assert plan._param_lengths.tolist() == [5]
        assert plan._modality_slots["audio"].tolist() == [1]
        assert plan._bucket_offsets["audio"].tolist() == [0, 4]


class TestEncodingPlanDstIndices:
    def test_pure_image_single_param(self):
        items_by_param = {
            0: [MultimodalItem(0, 0, "image", 5, {"id": "img_A"})],
        }
        plan = EncodingPlan.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        assert plan._dst_indices["image"].tolist() == [0, 1, 2, 3, 4]

    def test_mixed_image_audio(self):
        # param 0: <image><audio><image>  -> img_A(5)@pos0, aud_A(4)@pos1, img_B(5)@pos2
        # param 1: <image>                -> img_C(5)@pos0
        # Final: [param 0 in MMItemOrder | param 1] = img_A | aud_A | img_B | img_C
        items_by_param = {
            0: [
                MultimodalItem(0, 0, "image", 5, {"id": "img_A"}),
                MultimodalItem(0, 1, "audio", 4, {"id": "aud_A"}),
                MultimodalItem(0, 2, "image", 5, {"id": "img_B"}),
            ],
            1: [MultimodalItem(1, 0, "image", 5, {"id": "img_C"})],
        }
        plan = EncodingPlan.from_params(
            multimodal_params=[object(), object()],
            extract=_identity_extractor(items_by_param),
        )
        # Image bucket in append order: img_A, img_B, img_C
        # img_A -> final[0:5], img_B -> final[9:14] (param0 start 0, after img_A(5)+aud_A(4))
        # img_C -> final[14:19] (param1 start 14)
        assert plan._dst_indices["image"].tolist() == [
            0,
            1,
            2,
            3,
            4,  # img_A
            9,
            10,
            11,
            12,
            13,  # img_B
            14,
            15,
            16,
            17,
            18,  # img_C
        ]
        assert plan._dst_indices["audio"].tolist() == [5, 6, 7, 8]  # aud_A

    def test_ghost_audio_excluded(self):
        items_by_param = {
            0: [
                MultimodalItem(0, 0, "video", 5, {"id": "vid_A"}),
                MultimodalItem(0, -1, "audio", 4, {"id": "vid_A.audio"}),
            ],
        }
        plan = EncodingPlan.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        assert plan._dst_indices["video"].tolist() == [0, 1, 2, 3, 4]
        assert plan._dst_indices["audio"].numel() == 0

    def test_duplicate_item_idx_raises(self):
        items_by_param = {
            0: [
                MultimodalItem(0, 0, "image", 5, {"id": "x"}),
                MultimodalItem(0, 0, "image", 5, {"id": "y"}),
            ],
        }
        with pytest.raises(ValueError, match="duplicate item_idx_in_param"):
            EncodingPlan.from_params(
                multimodal_params=[object()],
                extract=_identity_extractor(items_by_param),
            )
