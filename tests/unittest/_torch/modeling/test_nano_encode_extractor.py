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
from tensorrt_llm._torch.models.multimodal_encoding import MultimodalItem
from tensorrt_llm.inputs.multimodal import MultimodalParams


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
