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
