# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tensorrt_llm._torch.models.multimodal_encoding."""

from __future__ import annotations

import pytest
from _mm_encode_helpers import _identity_extractor

from tensorrt_llm._torch.models.multimodal_encoding import MixedModalityAssembly, ModalityItem


class TestMultimodalItem:
    def test_is_frozen(self):
        item = ModalityItem(
            src_param_idx=0,
            item_idx_in_param=0,
            modality="image",
            token_count=1,
            payload={},
        )
        with pytest.raises(AttributeError):  # FrozenInstanceError from `@dataclass`(frozen=True)
            item.src_param_idx = 99


def _mixed_image_audio_assembly():
    """Shared fixture: param0 = img_A(5)|aud_A(4)|img_B(5), param1 = img_C(5).

    Final prompt order = img_A | aud_A | img_B | img_C.
    """
    items_by_param = {
        0: [
            ModalityItem(0, 0, "image", 5, {"id": "img_A"}),
            ModalityItem(0, 1, "audio", 4, {"id": "aud_A"}),
            ModalityItem(0, 2, "image", 5, {"id": "img_B"}),
        ],
        1: [
            ModalityItem(1, 0, "image", 5, {"id": "img_C"}),
        ],
    }
    return MixedModalityAssembly.from_params(
        multimodal_params=[object(), object()],
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

    def test_mixed_image_audio(self):
        assembly = _mixed_image_audio_assembly()
        assert assembly.total_tokens == 19
        assert set(assembly.active_modalities) == {"image", "audio"}
        assert assembly._param_lengths.tolist() == [14, 5]
        assert assembly._param_offsets.tolist() == [0, 14]
        assert assembly._modality_slots["image"].tolist() == [0, 2, 3]
        assert assembly._modality_slots["audio"].tolist() == [1]
        assert assembly._bucket_offsets["image"].tolist() == [0, 5, 10, 15]
        assert assembly._bucket_offsets["audio"].tolist() == [0, 4]

    def test_ghost_item_excluded_from_param_length(self):
        items_by_param = {
            0: [
                ModalityItem(0, 0, "video", 5, {"id": "vid_A"}),
                ModalityItem(0, -1, "audio", 4, {"id": "vid_A.audio"}),
            ],
        }
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        assert assembly._param_lengths.tolist() == [5]
        assert assembly._modality_slots["audio"].tolist() == [1]
        assert assembly._bucket_offsets["audio"].tolist() == [0, 4]

    def test_bucket_offsets_follow_encoder_rows_not_token_count(self):
        # A Nano video item whose `token_count` (post-interleave scatter
        # destination) exceeds its `encoder_rows` (vision-only encoder output):
        # the video bucket holds only `encoder_rows` rows, so `_bucket_offsets`
        # — which indexes the raw encoder bucket — must follow `encoder_rows`,
        # not `token_count`. `_param_lengths` (destination space) still uses
        # `token_count`.
        items_by_param = {
            0: [
                # video: encoder emits 6 vision rows; destination spans 10
                # (6 vision + 4 interleaved audio).
                ModalityItem(0, 0, "video", 10, {"id": "vid_A"}, encoder_token_count=6),
                ModalityItem(0, -1, "audio", 4, {"id": "vid_A.audio"}),
            ],
        }
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        # Encoder bucket has 6 video rows -> offsets [0, 6], NOT [0, 10].
        assert assembly._bucket_offsets["video"].tolist() == [0, 6]
        # Destination length still reflects the post-interleave token_count.
        assert assembly._param_lengths.tolist() == [10]


class TestEncodingPlanDstIndices:
    def test_pure_image_single_param(self):
        items_by_param = {
            0: [ModalityItem(0, 0, "image", 5, {"id": "img_A"})],
        }
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        assert assembly._dst_indices["image"].tolist() == [0, 1, 2, 3, 4]

    def test_mixed_image_audio(self):
        assembly = _mixed_image_audio_assembly()
        # Image bucket in append order: img_A, img_B, img_C
        # img_A -> final[0:5], img_B -> final[9:14] (param0 start 0, after img_A(5)+aud_A(4))
        # img_C -> final[14:19] (param1 start 14)
        assert assembly._dst_indices["image"].tolist() == [
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
        assert assembly._dst_indices["audio"].tolist() == [5, 6, 7, 8]  # aud_A

    def test_ghost_audio_excluded(self):
        items_by_param = {
            0: [
                ModalityItem(0, 0, "video", 5, {"id": "vid_A"}),
                ModalityItem(0, -1, "audio", 4, {"id": "vid_A.audio"}),
            ],
        }
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )
        assert assembly._dst_indices["video"].tolist() == [0, 1, 2, 3, 4]
        assert assembly._dst_indices["audio"].numel() == 0

    def test_duplicate_item_idx_raises(self):
        items_by_param = {
            0: [
                ModalityItem(0, 0, "image", 5, {"id": "x"}),
                ModalityItem(0, 0, "image", 5, {"id": "y"}),
            ],
        }
        with pytest.raises(ValueError, match="duplicate item_idx_in_param"):
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
            ([0, 9, 14], [5, 5, 5]),  # mixed_image_audio image bucket shape
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
            0: [ModalityItem(0, 0, "image", 5, {})],
            1: [ModalityItem(1, 0, "image", 5, {})],
        }
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[object(), object()],
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
        # Final: param 0 (rows 0..4) | param 1 (rows 5..9), each is item's MultimodalPromptOrder slot 0
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
                ModalityItem(0, 0, "image", 5, {}),
                ModalityItem(0, 1, "audio", 4, {}),
                ModalityItem(0, 2, "image", 5, {}),
            ],
            1: [ModalityItem(1, 0, "image", 5, {})],
        }
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[object(), object()],
            extract=_identity_extractor(items_by_param),
        )
        encoders = {
            "image": self._make_fake_encoder(H, call_log_image),
            "audio": self._make_fake_encoder(H, call_log_audio),
        }
        final = assemble_embeddings(
            assembly,
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

        items_by_param = {0: [ModalityItem(0, 0, "image", 5, {})]}
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
                torch.full((item.encoder_rows, hidden_dim), float(item.payload["sentinel"]))
                for item in items
            ]
            return torch.cat(blocks, dim=0) if blocks else torch.empty((0, hidden_dim))

        return encoder

    def test_ghost_first_audio_bucket_scatters_standalone_rows(self):
        """Ghost-first audio bucket must NOT mis-scatter ghost rows into the
        standalone-audio destination.

        Reproduces the Nano contract for a single param carrying BOTH an
        audio-bearing video and a standalone audio (which the extractor would
        yield as audio bucket order ``[ghost, standalone]`` — ghost FIRST):

            image @pos0      (5 rows)               -> final[0:5]
            video @pos1      (encoder_rows=6, tc=10) -> final[5:15] (6 vision + 4 audio)
            GHOST audio @-1  (4 rows)                -> no prompt-order slot
            standalone audio @pos2 (3 rows)          -> final[15:18]

        A ``post_process`` mirrors ``_nano_post_encode``: it interleaves the
        ghost-audio rows into the paired video chunk (6 -> 10 rows) and truncates
        the audio bucket to its leading ``n_non_ghost_rows``. The standalone-audio
        destination ``final[15:18]`` must receive the STANDALONE sentinel rows,
        not the GHOST sentinel rows.

        Fails on the pre-fix code: with ghost first, both the post-process
        leading-``n_non_ghost`` truncation and the scatter ``bucket[:dst.numel()]``
        slice keep the GHOST rows, so final[15:18] holds the ghost sentinel.
        """
        H = 4
        IMG, VID, GHOST, STANDALONE = 10.0, 20.0, 100.0, 200.0
        items_by_param = {
            0: [
                ModalityItem(0, 0, "image", 5, {"sentinel": IMG}),
                ModalityItem(0, 1, "video", 10, {"sentinel": VID}, encoder_token_count=6),
                ModalityItem(0, -1, "audio", 4, {"sentinel": GHOST}),  # ghost FIRST
                ModalityItem(0, 2, "audio", 3, {"sentinel": STANDALONE}),
            ]
        }
        assembly = MixedModalityAssembly.from_params(
            multimodal_params=[object()],
            extract=_identity_extractor(items_by_param),
        )

        def post_process(bucket_outputs, asm, multimodal_params):
            # Mirror `_nano_post_encode`: find the ghost audio's rows via the
            # audio bucket offsets, append them to the paired video chunk so the
            # video bucket grows from encoder_rows(6) to token_count(10), then
            # drop the trailing ghost rows from the audio bucket.
            audio_slots = asm._modality_slots["audio"].tolist()
            audio_offsets = asm._bucket_offsets["audio"]
            ghost_slot_pos = next(
                pos for pos, fi in enumerate(audio_slots) if asm.items[fi].item_idx_in_param == -1
            )
            g_start = int(audio_offsets[ghost_slot_pos].item())
            g_end = int(audio_offsets[ghost_slot_pos + 1].item())
            ghost_rows = bucket_outputs["audio"][g_start:g_end]
            bucket_outputs["video"] = torch.cat([bucket_outputs["video"], ghost_rows], dim=0)
            n_non_ghost_rows = sum(
                asm.items[fi].token_count
                for fi in audio_slots
                if asm.items[fi].item_idx_in_param != -1
            )
            bucket_outputs["audio"] = bucket_outputs["audio"][:n_non_ghost_rows]

        encoders = {
            "image": self._make_sentinel_encoder(H),
            "video": self._make_sentinel_encoder(H),
            "audio": self._make_sentinel_encoder(H),
        }
        final = assemble_embeddings(
            assembly,
            encoders,
            multimodal_params=[object()],
            post_process=post_process,
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_dim=H,
        )
        assert final.shape == (18, H)
        # The standalone-audio destination is final[15:18]; it must hold the
        # STANDALONE sentinel rows, never the GHOST sentinel rows.
        standalone_dst = final[15:18]
        assert torch.all(standalone_dst == STANDALONE), (
            f"standalone-audio dst final[15:18] should be all {STANDALONE} (standalone "
            f"rows) but got {standalone_dst[:, 0].tolist()} "
            f"(ghost sentinel is {GHOST}: ghost rows mis-scattered into the standalone slot)"
        )


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
                total = sum(it.token_count for it in items)
                return torch.zeros((total, H), dtype=torch.float32)

            return enc

        items_by_param: dict = {}
        # 16 pure-image params (4 tokens each)
        for i in range(16):
            items_by_param[i] = [ModalityItem(i, 0, "image", 4, {})]
        # 16 mixed image+audio params (image @ pos 0, audio @ pos 1)
        for i in range(16, 32):
            items_by_param[i] = [
                ModalityItem(i, 0, "image", 4, {}),
                ModalityItem(i, 1, "audio", 3, {}),
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
