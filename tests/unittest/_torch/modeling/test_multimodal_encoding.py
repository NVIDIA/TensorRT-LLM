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


import torch  # noqa: E402

from tensorrt_llm._torch.models.multimodal_encoding import encode_with_plan  # noqa: E402


class TestEncodeWithPlan:
    def _make_fake_encoder(self, hidden_dim, call_log):
        """Returns a fake encoder: row i in bucket = [i, i, ..., i] (hidden_dim copies)."""

        def encoder(items, multimodal_params):
            call_log.append(len(items))
            total_rows = sum(item.token_count for item in items)
            return (
                torch.arange(total_rows, dtype=torch.float32)
                .view(-1, 1)
                .expand(-1, hidden_dim)
                .contiguous()
            )

        return encoder

    def test_pure_single_modality_single_call(self):
        call_log_image: list = []
        H = 4
        items_by_param = {
            0: [MultimodalItem(0, 0, "image", 5, {})],
            1: [MultimodalItem(1, 0, "image", 5, {})],
        }
        plan = EncodingPlan.from_params(
            multimodal_params=[object(), object()],
            extract=_identity_extractor(items_by_param),
        )
        encoders = {"image": self._make_fake_encoder(H, call_log_image)}
        final = encode_with_plan(
            plan,
            encoders,
            multimodal_params=[object(), object()],
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_dim=H,
        )
        assert call_log_image == [2]  # ONE call, two items
        assert final.shape == (10, H)
        # Bucket rows 0..4 == item_0 (param 0), 5..9 == item_1 (param 1)
        # Final: param 0 (rows 0..4) | param 1 (rows 5..9), each is item's MMItemOrder slot 0
        assert final[0, 0].item() == 0
        assert final[4, 0].item() == 4
        assert final[5, 0].item() == 5
        assert final[9, 0].item() == 9

    def test_mixed_modalities_single_call_each(self):
        call_log_image: list = []
        call_log_audio: list = []
        H = 4
        items_by_param = {
            0: [
                MultimodalItem(0, 0, "image", 5, {}),
                MultimodalItem(0, 1, "audio", 4, {}),
                MultimodalItem(0, 2, "image", 5, {}),
            ],
            1: [MultimodalItem(1, 0, "image", 5, {})],
        }
        plan = EncodingPlan.from_params(
            multimodal_params=[object(), object()],
            extract=_identity_extractor(items_by_param),
        )
        encoders = {
            "image": self._make_fake_encoder(H, call_log_image),
            "audio": self._make_fake_encoder(H, call_log_audio),
        }
        final = encode_with_plan(
            plan,
            encoders,
            multimodal_params=[object(), object()],
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_dim=H,
        )
        assert call_log_image == [3]  # img_A, img_B, img_C in one call
        assert call_log_audio == [1]  # aud_A
        assert final.shape == (19, H)
        # Image bucket: img_A(rows 0..4), img_B(rows 5..9), img_C(rows 10..14)
        # img_A -> final[0:5]: values 0..4
        assert final[0, 0].item() == 0
        assert final[4, 0].item() == 4
        # aud_A -> final[5:9]: audio bucket rows 0..3
        assert final[5, 0].item() == 0
        assert final[8, 0].item() == 3
        # img_B -> final[9:14]: image bucket rows 5..9
        assert final[9, 0].item() == 5
        assert final[13, 0].item() == 9
        # img_C -> final[14:19]: image bucket rows 10..14
        assert final[14, 0].item() == 10
        assert final[18, 0].item() == 14

    def test_post_process_callback_receives_buckets(self):
        H = 4
        post_seen: dict = {}

        def post_process(bucket_outputs, plan, multimodal_params):
            post_seen["modalities"] = sorted(bucket_outputs.keys())
            post_seen["shapes"] = {m: tuple(t.shape) for m, t in bucket_outputs.items()}

        items_by_param = {0: [MultimodalItem(0, 0, "image", 3, {})]}
        plan = EncodingPlan.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        call_log: list = []
        encoders = {"image": self._make_fake_encoder(H, call_log)}
        encode_with_plan(
            plan,
            encoders,
            multimodal_params=[object()],
            post_process=post_process,
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_dim=H,
        )
        assert post_seen["modalities"] == ["image"]
        assert post_seen["shapes"]["image"] == (3, H)

    def test_empty_plan_returns_zero_size(self):
        plan = EncodingPlan.from_params(
            multimodal_params=[],
            extract=lambda i, p: iter([]),
        )
        final = encode_with_plan(
            plan,
            encoders={},
            multimodal_params=[],
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_dim=4,
        )
        assert final.shape == (0, 4)

    def test_bucket_shape_assert_fires_on_encoder_regression(self):
        H = 4

        def broken_encoder(items, multimodal_params):
            return torch.zeros((1, H), dtype=torch.float32)  # wrong row count

        items_by_param = {0: [MultimodalItem(0, 0, "image", 5, {})]}
        plan = EncodingPlan.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        with pytest.raises(AssertionError, match="image"):
            encode_with_plan(
                plan,
                encoders={"image": broken_encoder},
                multimodal_params=[object()],
                device=torch.device("cpu"),
                dtype=torch.float32,
                hidden_dim=H,
            )


class TestEncodeWithPlanHighBatch:
    def test_32_params_three_modalities_three_calls(self):
        """High-throughput guard: <=3 encoder calls regardless of pure/mixed mix.

        Direct regression guard against the per-mixed-param launch
        anti-pattern. The plan must collapse 32 source params into one
        encoder call per active modality.
        """
        H = 4
        call_log_image: list = []
        call_log_video: list = []
        call_log_audio: list = []

        def fake(call_log):
            def enc(items, multimodal_params):
                call_log.append(len(items))
                total = sum(it.token_count for it in items)
                return torch.zeros((total, H), dtype=torch.float32)

            return enc

        items_by_param: dict = {}
        # 16 pure-image params (4 tokens each)
        for i in range(16):
            items_by_param[i] = [MultimodalItem(i, 0, "image", 4, {})]
        # 16 mixed image+audio params (image @ pos 0, audio @ pos 1)
        for i in range(16, 32):
            items_by_param[i] = [
                MultimodalItem(i, 0, "image", 4, {}),
                MultimodalItem(i, 1, "audio", 3, {}),
            ]
        plan = EncodingPlan.from_params(
            multimodal_params=[object()] * 32,
            extract=_identity_extractor(items_by_param),
        )
        encoders = {
            "image": fake(call_log_image),
            "audio": fake(call_log_audio),
            "video": fake(call_log_video),  # registered but no items -> no call
        }
        encode_with_plan(
            plan,
            encoders,
            multimodal_params=[object()] * 32,
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_dim=H,
        )
        # Exactly one image call (32 items) + one audio call (16 items) + zero video
        assert call_log_image == [32], f"expected 1 image call with 32 items, got {call_log_image}"
        assert call_log_audio == [16], f"expected 1 audio call with 16 items, got {call_log_audio}"
        assert call_log_video == [], f"expected zero video calls, got {call_log_video}"
        total_calls = len(call_log_image) + len(call_log_audio) + len(call_log_video)
        assert total_calls <= 3
