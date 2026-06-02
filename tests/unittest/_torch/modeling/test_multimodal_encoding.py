# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tensorrt_llm._torch.models.multimodal_encoding.

Covers the per-item-scatter assembly: every `ModalityItem` owns exactly one
prompt slot (`prompt_pos`), its `rows` is both the encoder-output footprint and
the scatter-destination footprint, and there are no ghosts. The assembly batches
items per modality (one encoder launch per active modality) and scatters each
item's `rows` rows into its source-param destination range at its prompt-order
rank.
"""

from __future__ import annotations

import pytest
from _mm_encode_helpers import _identity_extractor

from tensorrt_llm._torch.models.multimodal_encoding import MixedModalityAssembly, ModalityItem


class _Runtime:
    """Minimal stand-in for `MultimodalParams.multimodal_runtime` exposing the
    single `total_embeds_in_request` field the assembly cross-check reads."""

    def __init__(self, total_embeds_in_request: int) -> None:
        self.total_embeds_in_request = total_embeds_in_request


class _ParamWithRuntime:
    """Param stub carrying a `multimodal_runtime` so the `sum(rows) ==
    total_embeds_in_request` cross-check has metadata to validate against."""

    def __init__(self, total_embeds_in_request: int) -> None:
        self.multimodal_runtime = _Runtime(total_embeds_in_request)


class TestMultimodalItem:
    def test_is_frozen(self):
        item = ModalityItem(
            src_param_idx=0,
            modality="image",
            mm_idx_per_modality=0,
            prompt_pos=0,
            rows=1,
            payload={},
        )
        with pytest.raises(AttributeError):  # FrozenInstanceError from `@dataclass`(frozen=True)
            item.src_param_idx = 99

    def test_six_canonical_fields_only(self):
        """The canonical item is exactly six fields: no `item_idx_in_param`,
        no `token_count`/`encoder_token_count` split, no `metadata`, no flags."""
        item = ModalityItem(
            src_param_idx=1,
            modality="video",
            mm_idx_per_modality=2,
            prompt_pos=3,
            rows=7,
            payload={"k": "v"},
        )
        assert item.src_param_idx == 1
        assert item.modality == "video"
        assert item.mm_idx_per_modality == 2
        assert item.prompt_pos == 3
        assert item.rows == 7
        assert item.payload == {"k": "v"}
        assert set(item.__slots__) == {
            "src_param_idx",
            "modality",
            "mm_idx_per_modality",
            "prompt_pos",
            "rows",
            "payload",
        }


def _interleaved_image_video_image_assembly():
    """image -> video -> image interleaving within one param (repeated modality).

    param0 prompt order: img_A(pos0, 5) | vid_X(pos1, 8) | img_B(pos2, 5).
    Two image items at non-adjacent prompt positions exercises the repeated-
    modality scatter: the image bucket holds img_A then img_B, but img_B's
    destination is AFTER the video, not contiguous with img_A.
    """
    items_by_param = {
        0: [
            ModalityItem(0, "image", 0, 0, 5, {"id": "img_A"}),
            ModalityItem(0, "video", 0, 1, 8, {"id": "vid_X"}),
            ModalityItem(0, "image", 1, 2, 5, {"id": "img_B"}),
        ],
    }
    return MixedModalityAssembly.from_params(
        multimodal_params=[_ParamWithRuntime(18)],
        extract=_identity_extractor(items_by_param),
    )


def _multi_video_audio_assembly():
    """PRIMARY new case: image -> video(+audio) -> image -> video(+audio).

    With the audio hoist, the embedded audio is a first-class `(audio, k)` item
    at the next prompt rank, so a single param's prompt order is the plain
    sequence:

        (image,0) (video,0) (audio,0) (image,1) (video,1) (audio,1)

    No ghost, no shared slot, no post-process. Every item owns its prompt slot.
    """
    items_by_param = {
        0: [
            ModalityItem(0, "image", 0, 0, 5, {"id": "img0"}),
            ModalityItem(0, "video", 0, 1, 8, {"id": "vid0"}),
            ModalityItem(0, "audio", 0, 2, 3, {"id": "aud0"}),
            ModalityItem(0, "image", 1, 3, 5, {"id": "img1"}),
            ModalityItem(0, "video", 1, 4, 8, {"id": "vid1"}),
            ModalityItem(0, "audio", 1, 5, 3, {"id": "aud1"}),
        ],
    }
    return MixedModalityAssembly.from_params(
        multimodal_params=[_ParamWithRuntime(32)],
        extract=_identity_extractor(items_by_param),
    )


class TestEncodingPlanPartition:
    def test_empty_batch(self):
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[],
            extract=lambda i, p: iter([]),
        )
        assert assembly.total_tokens == 0
        assert assembly.active_modalities == []
        assert len(assembly.items) == 0

    def test_interleaved_image_video_image(self):
        assembly = _interleaved_image_video_image_assembly()
        assert assembly.total_tokens == 18
        assert set(assembly.active_modalities) == {"image", "video"}
        assert assembly._param_lengths.tolist() == [18]
        # Image bucket holds img_A (flat 0) then img_B (flat 2); video holds vid_X.
        assert assembly._modality_slots["image"].tolist() == [0, 2]
        assert assembly._modality_slots["video"].tolist() == [1]
        # Bucket offsets are the prefix sum of `rows` (single source of truth).
        assert assembly._bucket_offsets["image"].tolist() == [0, 5, 10]
        assert assembly._bucket_offsets["video"].tolist() == [0, 8]

    def test_multi_video_audio_partition(self):
        assembly = _multi_video_audio_assembly()
        assert assembly.total_tokens == 32
        assert set(assembly.active_modalities) == {"image", "video", "audio"}
        assert assembly._param_lengths.tolist() == [32]
        # Each modality bucket holds its two items in append (prompt) order.
        assert assembly._modality_slots["image"].tolist() == [0, 3]
        assert assembly._modality_slots["video"].tolist() == [1, 4]
        assert assembly._modality_slots["audio"].tolist() == [2, 5]
        assert assembly._bucket_offsets["image"].tolist() == [0, 5, 10]
        assert assembly._bucket_offsets["video"].tolist() == [0, 8, 16]
        assert assembly._bucket_offsets["audio"].tolist() == [0, 3, 6]


class TestRowsCrossCheck:
    """`sum(rows)` for a param must equal its `total_embeds_in_request`."""

    def test_rows_sum_matches_total_embeds_in_request(self):
        # 5 + 8 + 5 == 18, matching _ParamWithRuntime(18); must not raise.
        assembly = _interleaved_image_video_image_assembly()
        assert assembly._param_lengths.tolist() == [18]

    def test_rows_sum_mismatch_raises(self):
        items_by_param = {
            0: [
                ModalityItem(0, "image", 0, 0, 5, {"id": "img_A"}),
                ModalityItem(0, "image", 1, 1, 5, {"id": "img_B"}),
            ],
        }
        # Declared total (11) != sum(rows) (10) -> fail-loud cross-check.
        with pytest.raises(ValueError, match="total_embeds_in_request"):
            MixedModalityAssembly.from_params(
                multimodal_params=[_ParamWithRuntime(11)],
                extract=_identity_extractor(items_by_param),
            )

    def test_cross_check_skipped_without_runtime(self):
        # Params lacking `multimodal_runtime` (plain stubs) must not trip the
        # cross-check; the assembly still builds.
        items_by_param = {0: [ModalityItem(0, "image", 0, 0, 5, {})]}
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        assert assembly.total_tokens == 5


class TestEncodingPlanDstIndices:
    def test_pure_image_single_param(self):
        items_by_param = {
            0: [ModalityItem(0, "image", 0, 0, 5, {"id": "img_A"})],
        }
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[_ParamWithRuntime(5)],
            extract=_identity_extractor(items_by_param),
        )
        assert assembly._dst_indices["image"].tolist() == [0, 1, 2, 3, 4]

    def test_interleaved_image_video_image_scatters_by_prompt_pos(self):
        assembly = _interleaved_image_video_image_assembly()
        # Destinations follow prompt_pos within the single param:
        #   img_A pos0 -> final[0:5]
        #   vid_X pos1 -> final[5:13]
        #   img_B pos2 -> final[13:18]   (AFTER the video, not contiguous w/ img_A)
        # Image bucket order is [img_A, img_B]; dst concatenates their ranges.
        assert assembly._dst_indices["image"].tolist() == [
            0,
            1,
            2,
            3,
            4,  # img_A pos0
            13,
            14,
            15,
            16,
            17,  # img_B pos2
        ]
        assert assembly._dst_indices["video"].tolist() == [5, 6, 7, 8, 9, 10, 11, 12]

    def test_multi_video_audio_scatters_by_prompt_pos(self):
        assembly = _multi_video_audio_assembly()
        # prompt order: img0(0:5) vid0(5:13) aud0(13:16) img1(16:21) vid1(21:29) aud1(29:32)
        assert assembly._dst_indices["image"].tolist() == [
            0,
            1,
            2,
            3,
            4,  # img0 pos0
            16,
            17,
            18,
            19,
            20,  # img1 pos3
        ]
        assert assembly._dst_indices["video"].tolist() == [
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,  # vid0 pos1
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,  # vid1 pos4
        ]
        assert assembly._dst_indices["audio"].tolist() == [
            13,
            14,
            15,  # aud0 pos2
            29,
            30,
            31,  # aud1 pos5
        ]

    def test_duplicate_prompt_pos_raises(self):
        items_by_param = {
            0: [
                ModalityItem(0, "image", 0, 0, 5, {"id": "x"}),
                ModalityItem(0, "image", 1, 0, 5, {"id": "y"}),  # same prompt_pos
            ],
        }
        with pytest.raises(ValueError, match="duplicate prompt_pos"):
            MixedModalityAssembly.from_params(
                multimodal_params=[object()],
                extract=_identity_extractor(items_by_param),
            )

    def test_duplicate_modality_group_raises(self):
        items_by_param = {
            0: [
                ModalityItem(0, "image", 0, 0, 5, {"id": "x"}),
                ModalityItem(0, "image", 0, 1, 5, {"id": "y"}),  # same (modality, idx)
            ],
        }
        with pytest.raises(ValueError, match="duplicate.*mm_idx_per_modality"):
            MixedModalityAssembly.from_params(
                multimodal_params=[object()],
                extract=_identity_extractor(items_by_param),
            )


import torch  # noqa: E402

from tensorrt_llm._torch.models.multimodal_encoding import (  # noqa: E402
    _expand_ranges,
    assemble_embeddings,
)


def _loop_expand_reference(starts, lengths):
    """Reference: the original per-token ``rows.extend(range(...))`` loop."""
    rows = []
    for s, length in zip(starts, lengths, strict=True):
        rows.extend(range(s, s + length))
    return torch.tensor(rows, dtype=torch.int64)


class TestExpandRangesVectorization:
    """`_expand_ranges` must be byte-identical to the per-token loop it replaces."""

    @pytest.mark.parametrize(
        "starts,lengths",
        [
            ([], []),
            ([7], [0]),  # zero-length segment
            ([0], [5]),
            ([0, 13], [5, 5]),  # interleaved image bucket shape (img_A, img_B)
            ([5], [4]),
            ([3, 100, 0, 50], [2, 0, 3, 1]),  # interleaved zero-length
            ([0, 682], [682, 357]),  # the dynamic-res multi-image case (sums to 1039)
        ],
    )
    def test_matches_loop_reference(self, starts, lengths):
        vec = _expand_ranges(
            torch.tensor(starts, dtype=torch.int64),
            torch.tensor(lengths, dtype=torch.int64),
        )
        ref = _loop_expand_reference(starts, lengths)
        assert vec.dtype == torch.int64
        assert torch.equal(vec, ref), f"vec={vec.tolist()} ref={ref.tolist()}"


class TestEncodeWithPlan:
    def _make_fake_encoder(self, hidden_dim, call_log):
        """Returns a fake encoder: row i in bucket = [i, i, ..., i] (hidden_dim copies)."""

        def encoder(items, multimodal_params):
            call_log.append(len(items))
            total_rows = sum(item.rows for item in items)
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
            0: [ModalityItem(0, "image", 0, 0, 5, {})],
            1: [ModalityItem(1, "image", 0, 0, 5, {})],
        }
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[_ParamWithRuntime(5), _ParamWithRuntime(5)],
            extract=_identity_extractor(items_by_param),
        )
        encoders = {"image": self._make_fake_encoder(H, call_log_image)}
        final = assemble_embeddings(
            assembly,
            encoders,
            multimodal_params=[object(), object()],
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_dim=H,
        )
        assert call_log_image == [2]  # ONE call, two items
        assert final.shape == (10, H)
        # Bucket rows 0..4 == item_0 (param 0), 5..9 == item_1 (param 1)
        # Final: param 0 (rows 0..4) | param 1 (rows 5..9), each is item's prompt slot 0
        assert final[0, 0].item() == 0
        assert final[4, 0].item() == 4
        assert final[5, 0].item() == 5
        assert final[9, 0].item() == 9

    def test_interleaved_repeated_image_single_call(self):
        """image -> video -> image: ONE image encoder call (2 items), ONE video
        call; img_B's rows scatter AFTER the video despite sharing the image
        bucket with img_A."""
        call_log_image: list = []
        call_log_video: list = []
        H = 4
        assembly = _interleaved_image_video_image_assembly()
        encoders = {
            "image": self._make_fake_encoder(H, call_log_image),
            "video": self._make_fake_encoder(H, call_log_video),
        }
        final = assemble_embeddings(
            assembly,
            encoders,
            multimodal_params=[object()],
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_dim=H,
        )
        assert call_log_image == [2]  # img_A, img_B in ONE call
        assert call_log_video == [1]  # vid_X
        assert final.shape == (18, H)
        # img_A -> final[0:5]: image bucket rows 0..4
        assert final[0, 0].item() == 0
        assert final[4, 0].item() == 4
        # vid_X -> final[5:13]: video bucket rows 0..7
        assert final[5, 0].item() == 0
        assert final[12, 0].item() == 7
        # img_B -> final[13:18]: image bucket rows 5..9 (AFTER the video)
        assert final[13, 0].item() == 5
        assert final[17, 0].item() == 9

    def test_multi_video_audio_three_calls_no_post_process(self):
        """PRIMARY: image->video(+audio)->image->video(+audio) assembles via
        plain per-item scatter, one call per modality, NO post_process."""
        call_log_image: list = []
        call_log_video: list = []
        call_log_audio: list = []
        H = 4
        assembly = _multi_video_audio_assembly()
        encoders = {
            "image": self._make_fake_encoder(H, call_log_image),
            "video": self._make_fake_encoder(H, call_log_video),
            "audio": self._make_fake_encoder(H, call_log_audio),
        }
        final = assemble_embeddings(
            assembly,
            encoders,
            multimodal_params=[object()],
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_dim=H,
        )
        assert call_log_image == [2]
        assert call_log_video == [2]
        assert call_log_audio == [2]
        assert final.shape == (32, H)
        # aud0 -> final[13:16]: audio bucket rows 0..2
        assert final[13, 0].item() == 0
        assert final[15, 0].item() == 2
        # vid1 -> final[21:29]: video bucket rows 8..15
        assert final[21, 0].item() == 8
        assert final[28, 0].item() == 15
        # aud1 -> final[29:32]: audio bucket rows 3..5
        assert final[29, 0].item() == 3
        assert final[31, 0].item() == 5

    def test_empty_plan_returns_zero_size(self):
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[],
            extract=lambda i, p: iter([]),
        )
        final = assemble_embeddings(
            assembly,
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

        items_by_param = {0: [ModalityItem(0, "image", 0, 0, 5, {})]}
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        with pytest.raises(AssertionError, match="image"):
            assemble_embeddings(
                assembly,
                encoders={"image": broken_encoder},
                multimodal_params=[object()],
                device=torch.device("cpu"),
                dtype=torch.float32,
                hidden_dim=H,
            )

    def _make_sentinel_encoder(self, hidden_dim):
        """Fake encoder: every row of item ``k`` (in bucket order) is filled with
        that item's ``payload["sentinel"]`` value (broadcast over hidden_dim).

        Rows are emitted in ``_modality_slots[modality]`` order (the adapter
        contract), so the per-item sentinel block lets a test distinguish which
        item a scattered row originated from — unlike an ``arange`` encoder whose
        values track row POSITION rather than item identity.
        """

        def encoder(items, multimodal_params):
            blocks = [
                torch.full((item.rows, hidden_dim), float(item.payload["sentinel"]))
                for item in items
            ]
            return torch.cat(blocks, dim=0) if blocks else torch.empty((0, hidden_dim))

        return encoder

    def test_repeated_modality_scatters_correct_item_rows(self):
        """A repeated-modality bucket must scatter each item's OWN rows to its
        own prompt slot.

        param0: image @pos0 (5 rows) | audio @pos1 (4 rows) | image @pos2 (5 rows).
        The image bucket order is [img_A, img_B]. img_A must land in final[0:5]
        and img_B in final[9:14] (after the audio), each carrying its own
        sentinel — not the other's.
        """
        H = 4
        IMG_A, AUD, IMG_B = 10.0, 50.0, 90.0
        items_by_param = {
            0: [
                ModalityItem(0, "image", 0, 0, 5, {"sentinel": IMG_A}),
                ModalityItem(0, "audio", 0, 1, 4, {"sentinel": AUD}),
                ModalityItem(0, "image", 1, 2, 5, {"sentinel": IMG_B}),
            ]
        }
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[_ParamWithRuntime(14)],
            extract=_identity_extractor(items_by_param),
        )
        encoders = {
            "image": self._make_sentinel_encoder(H),
            "audio": self._make_sentinel_encoder(H),
        }
        final = assemble_embeddings(
            assembly,
            encoders,
            multimodal_params=[object()],
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_dim=H,
        )
        assert final.shape == (14, H)
        assert torch.all(final[0:5] == IMG_A), f"final[0:5] should be img_A ({IMG_A})"
        assert torch.all(final[5:9] == AUD), f"final[5:9] should be audio ({AUD})"
        assert torch.all(final[9:14] == IMG_B), f"final[9:14] should be img_B ({IMG_B})"


class TestEncodeWithPlanHighBatch:
    def test_32_params_three_modalities_three_calls(self):
        """High-throughput guard: <=3 encoder calls regardless of pure/mixed mix.

        Direct regression guard against the per-mixed-param launch
        anti-pattern. The assembly must collapse 32 source params into one
        encoder call per active modality.
        """
        H = 4
        call_log_image: list = []
        call_log_video: list = []
        call_log_audio: list = []

        def fake(call_log):
            def enc(items, multimodal_params):
                call_log.append(len(items))
                total = sum(it.rows for it in items)
                return torch.zeros((total, H), dtype=torch.float32)

            return enc

        items_by_param: dict = {}
        # 16 pure-image params (4 tokens each)
        for i in range(16):
            items_by_param[i] = [ModalityItem(i, "image", 0, 0, 4, {})]
        # 16 mixed image+audio params (image @ pos 0, audio @ pos 1)
        for i in range(16, 32):
            items_by_param[i] = [
                ModalityItem(i, "image", 0, 0, 4, {}),
                ModalityItem(i, "audio", 0, 1, 3, {}),
            ]
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[object()] * 32,
            extract=_identity_extractor(items_by_param),
        )
        encoders = {
            "image": fake(call_log_image),
            "audio": fake(call_log_audio),
            "video": fake(call_log_video),  # registered but no items -> no call
        }
        assemble_embeddings(
            assembly,
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
