# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Nano-side multimodal item extractor.

The extractor walks one ``MultimodalParams`` and yields
:class:`MultimodalItem` instances, one per modality item the encoder will
process. Per-item ``token_count`` is sourced from the per-payload
``num_tokens`` field that the Nano preprocessing pipeline populates (this
matches what ``multimodal_embedding_lengths`` would carry for mixed
requests, but is also valid for the pure-modality single-item path).

For a video payload that carries an embedded audio track, the extractor
yields the video item first (non-ghost, ``item_idx_in_param`` = MMItemOrder
position, ``token_count`` = video tokens + interleaved audio tokens) and a
ghost audio item second (``item_idx_in_param == -1``,
``token_count`` = raw audio rows the audio encoder returns).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.models.modeling_nemotron_nano import (
    NemotronH_Nano_VL_V2,
    _nano_extract_items,
)
from tensorrt_llm._torch.models.multimodal_encoding import EncodingPlan, MultimodalItem
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData


def _identity_extractor(items_by_param):
    """Test helper: yield pre-built MultimodalItems passed in by-param."""

    def extract(param_idx, _param):
        yield from items_by_param.get(param_idx, [])

    return extract


def _make_param(multimodal_data: dict) -> MultimodalParams:
    """Build a stub MultimodalParams for extractor unit tests."""
    return MultimodalParams(
        multimodal_input=None,
        multimodal_data=multimodal_data,
        multimodal_runtime=None,
    )


class TestNanoExtractItems:
    """Schema: per-modality payloads carry ``num_tokens`` (int).

    For mixed-modality requests, the Nano preprocessor also populates
    ``multimodal_item_order`` + ``multimodal_embedding_lengths`` in
    prompt order; the extractor prefers those when present so that
    ``item_idx_in_param`` and ``token_count`` match the canonical
    prompt-order projection used by ``MMItemOrder.split_embeddings``.
    """

    def test_pure_image(self):
        payload = {
            "image": {"pixel_values": "fake", "num_tokens": 5},
            "modality_type": "image",
        }
        param = _make_param(payload)
        items = list(_nano_extract_items(0, param))
        assert len(items) == 1
        assert items[0].modality == "image"
        assert items[0].token_count == 5
        assert items[0].item_idx_in_param == 0
        assert items[0].src_param_idx == 0

    def test_pure_audio(self):
        payload = {
            "audio": {"input_features": "fake", "num_tokens": 4},
            "modality_type": "audio",
        }
        items = list(_nano_extract_items(0, _make_param(payload)))
        assert len(items) == 1
        assert items[0].modality == "audio"
        assert items[0].token_count == 4
        assert items[0].item_idx_in_param == 0
        assert items[0].src_param_idx == 0

    def test_pure_video_no_audio(self):
        payload = {
            "video": {
                "pixel_values": "fake",
                "num_tokens": 7,
                "video_size": [],
            },
            "modality_type": "video",
        }
        items = list(_nano_extract_items(0, _make_param(payload)))
        assert len(items) == 1
        assert items[0].modality == "video"
        assert items[0].token_count == 7
        assert items[0].item_idx_in_param == 0

    def test_video_with_embedded_audio_emits_ghost(self):
        payload = {
            "video": {
                "pixel_values": "fake",
                "num_tokens": 5,
                "video_size": [],
                "audio": {
                    "input_features": "fake",
                    "num_tokens": 4,
                    "has_audio": [True],
                    "audio_num_clips": 1,
                },
            },
            "modality_type": "video",
        }
        items = list(_nano_extract_items(0, _make_param(payload)))
        assert len(items) == 2
        video, ghost_audio = items
        assert video.modality == "video"
        assert video.item_idx_in_param == 0
        assert video.token_count == 9  # 5 video + 4 audio = post-interleave
        assert video.src_param_idx == 0
        assert ghost_audio.modality == "audio"
        assert ghost_audio.item_idx_in_param == -1
        assert ghost_audio.token_count == 4
        assert ghost_audio.src_param_idx == 0
        # Ghost carries back-reference to the paired video item position.
        assert ghost_audio.metadata.get("paired_video_item_idx") == 0

    def test_mixed_image_audio(self):
        payload = {
            "image": {"pixel_values": "fake", "num_tokens": 5},
            "audio": {"input_features": "fake", "num_tokens": 4},
            "modality_type": ["image", "audio"],
            "multimodal_item_order": [
                {"modality": "image", "index": 0},
                {"modality": "audio", "index": 0},
            ],
            "multimodal_embedding_lengths": [5, 4],
        }
        items = list(_nano_extract_items(0, _make_param(payload)))
        assert len(items) == 2
        positions = {it.modality: it.item_idx_in_param for it in items}
        assert positions == {"image": 0, "audio": 1}
        token_counts = {it.modality: it.token_count for it in items}
        assert token_counts == {"image": 5, "audio": 4}

    def test_mixed_audio_first_then_image(self):
        # Prompt order is audio, then image -> item_idx_in_param should reflect that.
        payload = {
            "image": {"pixel_values": "fake", "num_tokens": 3},
            "audio": {"input_features": "fake", "num_tokens": 7},
            "modality_type": ["audio", "image"],
            "multimodal_item_order": [
                {"modality": "audio", "index": 0},
                {"modality": "image", "index": 0},
            ],
            "multimodal_embedding_lengths": [7, 3],
        }
        items = list(_nano_extract_items(0, _make_param(payload)))
        assert len(items) == 2
        positions = {it.modality: it.item_idx_in_param for it in items}
        assert positions == {"audio": 0, "image": 1}

    def test_unknown_modality_raises(self):
        payload = {
            "weird": {"foo": "bar"},
            "modality_type": "weird",
        }
        with pytest.raises(ValueError, match="Unknown modality"):
            list(_nano_extract_items(0, _make_param(payload)))


def _make_runtime(total_embeds: int) -> MultimodalRuntimeData:
    """Build a real ``MultimodalRuntimeData`` with the given total token count."""
    cumsum = torch.arange(1, total_embeds + 1, dtype=torch.int64)
    return MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=total_embeds,
        embed_mask_cumsum=cumsum,
    )


def _make_param_with_runtime(multimodal_data: dict, total_embeds: int) -> MultimodalParams:
    return MultimodalParams(
        multimodal_input=None,
        multimodal_data=multimodal_data,
        multimodal_runtime=_make_runtime(total_embeds),
    )


class TestNanoExtractorProductionSchema:
    """Exercise the production-fields fallback for per-item token counts.

    When the synthetic ``num_tokens`` field is absent from a payload, the
    extractor resolves token counts via the production preprocessing
    fields: ``multimodal_embedding_lengths`` indexed by
    ``multimodal_item_order`` for mixed params, and
    ``multimodal_runtime.total_embeds_in_request`` for pure single-modality
    params.
    """

    def test_pure_image_uses_multimodal_runtime(self):
        # No num_tokens on payload; total_embeds_in_request drives the count.
        payload = {"pixel_values": "fake"}
        param = _make_param_with_runtime(
            {"image": payload, "modality_type": "image"},
            total_embeds=7,
        )
        items = list(_nano_extract_items(0, param))
        assert len(items) == 1
        assert items[0].modality == "image"
        assert items[0].token_count == 7
        assert items[0].item_idx_in_param == 0

    def test_pure_audio_uses_multimodal_runtime(self):
        payload = {"input_features": "fake"}
        param = _make_param_with_runtime(
            {"audio": payload, "modality_type": "audio"},
            total_embeds=4,
        )
        items = list(_nano_extract_items(0, param))
        assert len(items) == 1
        assert items[0].modality == "audio"
        assert items[0].token_count == 4

    def test_mixed_uses_multimodal_embedding_lengths(self):
        # No num_tokens on either payload; mixed params get counts from
        # multimodal_embedding_lengths indexed by multimodal_item_order.
        payload_image = {"pixel_values": "fake"}
        payload_audio = {"input_features": "fake"}
        param = _make_param(
            {
                "image": payload_image,
                "audio": payload_audio,
                "modality_type": ["image", "audio"],
                "multimodal_item_order": [
                    {"modality": "image", "index": 0},
                    {"modality": "audio", "index": 0},
                ],
                "multimodal_embedding_lengths": [5, 4],
            }
        )
        items = list(_nano_extract_items(0, param))
        by_modality = {it.modality: it.token_count for it in items}
        assert by_modality == {"image": 5, "audio": 4}
        # item_idx_in_param tracks MMItemOrder positions.
        positions = {it.modality: it.item_idx_in_param for it in items}
        assert positions == {"image": 0, "audio": 1}

    def test_num_tokens_overrides_production_fields(self):
        # When both are present, ``num_tokens`` wins (test convention,
        # cheapest lookup).
        payload = {"pixel_values": "fake", "num_tokens": 3}
        param = _make_param_with_runtime(
            {"image": payload, "modality_type": "image"},
            total_embeds=99,
        )
        items = list(_nano_extract_items(0, param))
        assert items[0].token_count == 3

    def test_missing_sources_raises(self):
        # No num_tokens on the payload AND no production fields populated.
        # The extractor must surface a clear error rather than silently
        # producing a wrong count.
        payload = {"pixel_values": "fake"}
        param = _make_param({"image": payload, "modality_type": "image"})
        with pytest.raises(KeyError, match="Cannot resolve per-item token count"):
            list(_nano_extract_items(0, param))


class TestNanoVisionBucketAdapter:
    """Tests for the vision-bucket encoder adapter on ``NemotronH_Nano_VL_V2``.

    Bridges ``List[MultimodalItem]`` (image OR video bucket) to the existing
    ``self.vision_encoder(params)`` interface by building per-item
    ``MultimodalParams`` views, cats per-item outputs into one bucket tensor
    (rows in bucket order), and stashes per-item ``num_tokens_in_video`` for
    EVS on the source params (side-channel, matches legacy behavior).
    """

    def _make_model_stub(self, vision_encoder_return):
        model = MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = MagicMock(return_value=vision_encoder_return)
        # Bind the real method to the stub instance
        model._adapter_vision_bucket = NemotronH_Nano_VL_V2._adapter_vision_bucket.__get__(model)
        model._build_single_modality_param = NemotronH_Nano_VL_V2._build_single_modality_param
        return model

    def test_single_pure_image_param(self):
        item = MultimodalItem(0, 0, "image", 5, {"num_tokens": 5})
        params = [_make_param({"image": item.payload, "modality_type": "image"})]
        emb = torch.randn(5, 4)
        model = self._make_model_stub(([emb], None))
        out = model._adapter_vision_bucket([item], params)
        model.vision_encoder.assert_called_once()
        assert out.shape == (5, 4)
        torch.testing.assert_close(out, emb)

    def test_evs_num_tokens_stashed_on_video_params(self):
        item = MultimodalItem(0, 0, "video", 9, {"num_tokens": 9})
        params = [_make_param({"video": item.payload, "modality_type": "video"})]
        emb = torch.randn(9, 4)
        model = self._make_model_stub(([emb], [[3, 3, 3]]))
        out = model._adapter_vision_bucket([item], params)
        assert out.shape == (9, 4)
        assert params[0].multimodal_data["num_tokens_in_video"] == [3, 3, 3]
        # Also stash the by-modality view (matches legacy 3463-3466 behavior)
        assert params[0].multimodal_data["num_tokens_in_video_by_modality"]["video"] == [3, 3, 3]

    def test_mixed_param_synthesizes_single_modality_view(self):
        # Image item from a mixed image+audio param: adapter should
        # synthesize a single-modality view so vision_encoder sees image-only
        item = MultimodalItem(0, 0, "image", 5, {"num_tokens": 5})
        params = [
            _make_param(
                {
                    "image": item.payload,
                    "audio": {"num_tokens": 4},
                    "modality_type": ["image", "audio"],
                }
            )
        ]
        emb = torch.randn(5, 4)
        model = self._make_model_stub(([emb], None))
        out = model._adapter_vision_bucket([item], params)
        assert out.shape == (5, 4)
        passed_params = model.vision_encoder.call_args[0][0]
        assert len(passed_params) == 1
        assert passed_params[0].multimodal_data["modality_type"] == "image"
        assert "audio" not in passed_params[0].multimodal_data

    def test_multi_item_bucket_concatenates_in_order(self):
        # Two image items from two pure-image params
        items = [
            MultimodalItem(0, 0, "image", 5, {"num_tokens": 5}),
            MultimodalItem(1, 0, "image", 3, {"num_tokens": 3}),
        ]
        params = [
            _make_param({"image": items[0].payload, "modality_type": "image"}),
            _make_param({"image": items[1].payload, "modality_type": "image"}),
        ]
        emb0 = torch.ones(5, 4)
        emb1 = torch.ones(3, 4) * 2.0
        model = self._make_model_stub(([emb0, emb1], None))
        out = model._adapter_vision_bucket(items, params)
        assert out.shape == (8, 4)
        torch.testing.assert_close(out[:5], emb0)
        torch.testing.assert_close(out[5:], emb1)


class TestNanoAudioBucketAdapter:
    def _make_model_stub(self, encode_audio_return):
        model = MagicMock(spec=NemotronH_Nano_VL_V2)
        model._encode_audio = MagicMock(return_value=encode_audio_return)
        model._adapter_audio_bucket = NemotronH_Nano_VL_V2._adapter_audio_bucket.__get__(model)
        return model

    def test_list_return_normalization(self):
        emb0 = torch.randn(4, 4)
        emb1 = torch.randn(3, 4)
        model = self._make_model_stub([(emb0, [4]), (emb1, [3])])
        items = [
            MultimodalItem(0, 0, "audio", 4, {"id": "a0"}),
            MultimodalItem(1, 0, "audio", 3, {"id": "a1"}),
        ]
        out = model._adapter_audio_bucket(items, [object(), object()])
        assert out.shape == (7, 4)
        torch.testing.assert_close(out[:4], emb0)
        torch.testing.assert_close(out[4:], emb1)

    def test_scalar_tensor_return_normalization(self):
        emb = torch.randn(4, 4)
        model = self._make_model_stub(emb)
        items = [MultimodalItem(0, 0, "audio", 4, {"id": "a0"})]
        out = model._adapter_audio_bucket(items, [object()])
        torch.testing.assert_close(out, emb)

    def test_unexpected_shape_raises_typeerror(self):
        model = self._make_model_stub("garbage")
        items = [
            MultimodalItem(0, 0, "audio", 4, {"id": "a0"}),
            MultimodalItem(1, 0, "audio", 3, {"id": "a1"}),
        ]
        with pytest.raises(TypeError, match="must return a list"):
            model._adapter_audio_bucket(items, [object(), object()])


class TestNanoPostEncode:
    """Tests for ``_nano_post_encode``: video-audio interleave + ghost-audio bucket truncation.

    After per-modality encoders run, this hook reassembles the video
    bucket to its POST-interleave row count and trims the ghost audio
    rows from the audio bucket so the scatter assembly sees only the
    non-ghost portion.
    """

    def _make_model_stub(self, interleave_fn):
        model = MagicMock(spec=NemotronH_Nano_VL_V2)
        model._interleave_video_audio_embeddings = interleave_fn
        model._nano_post_encode = NemotronH_Nano_VL_V2._nano_post_encode.__get__(model)
        return model

    def test_no_video_bucket_noop(self):
        # If no video modality, post_encode is a no-op
        items_by_param = {0: [MultimodalItem(0, 0, "image", 5, {"num_tokens": 5})]}
        params = [_make_param({"image": {"num_tokens": 5}, "modality_type": "image"})]
        plan = EncodingPlan.from_params(
            multimodal_params=params,
            extract=_identity_extractor(items_by_param),
        )
        bucket_outputs = {"image": torch.ones((5, 4))}
        model = self._make_model_stub(lambda *a, **kw: None)
        model._nano_post_encode(bucket_outputs, plan, params)
        assert bucket_outputs["image"].shape == (5, 4)

    def test_video_with_audio_interleaved(self):
        # One param: video with embedded audio
        # video_item: post-interleave 9 tokens; ghost audio: 4 tokens
        # Pre-interleave video rows = post-interleave (9) - audio (4) = 5
        audio_payload = {"num_tokens": 4, "has_audio": [True], "audio_num_clips": torch.tensor([1])}
        video_payload = {"num_tokens": 5, "audio": audio_payload, "video_size": []}
        video_item = MultimodalItem(0, 0, "video", 9, video_payload)
        audio_item = MultimodalItem(
            0,
            -1,
            "audio",
            4,
            audio_payload,
            metadata={"paired_video_item_idx": 0},
        )
        items_by_param = {0: [video_item, audio_item]}
        params = [_make_param({"video": video_payload, "modality_type": "video"})]
        plan = EncodingPlan.from_params(
            multimodal_params=params,
            extract=_identity_extractor(items_by_param),
        )

        # Pre-interleave video bucket has 5 rows; audio bucket has 4 rows (all ghost)
        video_bucket = torch.ones((5, 4))
        audio_bucket = torch.ones((4, 4)) * 2.0

        # Fake interleave: concatenate v|a -> 9 rows
        def fake_interleave(video_emb, audio_emb, *args, **kwargs):
            return torch.cat([video_emb, audio_emb], dim=0)

        model = self._make_model_stub(fake_interleave)

        bucket_outputs = {"video": video_bucket, "audio": audio_bucket}
        model._nano_post_encode(bucket_outputs, plan, params)

        # Video bucket grew to post-interleave shape (9)
        assert bucket_outputs["video"].shape == (9, 4)
        # Audio bucket truncated to 0 rows (only ghost present)
        assert bucket_outputs["audio"].shape == (0, 4)

    def test_standalone_audio_preserved(self):
        # One param with a standalone audio item (no video, no pairing)
        items_by_param = {
            0: [MultimodalItem(0, 0, "audio", 4, {"num_tokens": 4})],
        }
        params = [_make_param({"audio": {"num_tokens": 4}, "modality_type": "audio"})]
        plan = EncodingPlan.from_params(
            multimodal_params=params,
            extract=_identity_extractor(items_by_param),
        )
        bucket_outputs = {"audio": torch.ones((4, 4))}
        model = self._make_model_stub(lambda *a, **kw: None)
        model._nano_post_encode(bucket_outputs, plan, params)
        assert bucket_outputs["audio"].shape == (4, 4)
