# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Nano-side multimodal item extractor (canonical 6-field item).

The extractor walks one `MultimodalParams` and yields one `ModalityItem` per
prompt-order slot the encoder will process. There is no ghost audio and no
post-encode interleave: video-embedded audio is hoisted to a first-class
`(audio, k)` item by the input processor, so the extractor treats it like any
other audio item.

Two extraction regimes:

  * single-modality (one distinct modality present): the AGGREGATE carve-out —
    exactly one item whose `rows` is `total_embeds_in_request` and whose payload
    is the whole per-modality blob (the Nano vision encoder emits one
    concatenated tensor per request).
  * multi-modality (more than one distinct modality): one item per
    `MultimodalPromptOrder` entry; `prompt_pos` is the rank, `rows` is the
    per-slot `multimodal_embedding_lengths` entry, and the payload is sliced to
    that single sub-item by the model's `_slice_payload` method.
"""

from __future__ import annotations

from types import MethodType
from unittest.mock import MagicMock

import pytest
import torch
from _mm_encode_helpers import _make_param

from tensorrt_llm._torch.models.mixed_modal_encode import ModalityItem
from tensorrt_llm._torch.models.modeling_nemotron_nano import (
    NemotronH_Nano_VL_V2,
    _nano_extract_items,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData


def _make_runtime(total_embeds: int) -> MultimodalRuntimeData:
    """Build a real `MultimodalRuntimeData` with the given total token count."""
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


class _StubSlicer:
    """Test slice-payload callable that returns a sentinel-tagged per-item payload.

    The extractor's payload-slicing is delegated to a `slice_payload` callable
    (the bound `NemotronH_Nano_VL_V2._slice_payload` in production); in
    extractor-only tests we substitute this stub so we can assert WHICH
    `(modality, mm_idx_per_modality)` the extractor asked to slice, independent
    of the real per-modality tensor math (covered separately in
    `TestNanoPayloadSlicer`). It is itself the callable passed as
    `slice_payload=`.
    """

    def __init__(self):
        self.calls: list = []

    def slice_payload(self, param, modality, mm_idx_per_modality, *, rows=None):
        self.calls.append((modality, mm_idx_per_modality))
        return {"sliced": (modality, mm_idx_per_modality)}

    def __call__(self, param, modality, mm_idx_per_modality, *, rows=None):
        return self.slice_payload(param, modality, mm_idx_per_modality, rows=rows)


# ---------------------------------------------------------------------------
# Single-modality AGGREGATE carve-out (KEPT — not the per-item path).
# ---------------------------------------------------------------------------


class TestNanoExtractSingleModalityAggregate:
    """Single-modality requests stay on the aggregate path: one item per param,
    `rows == total_embeds_in_request`, whole-blob payload, `prompt_pos == 0`.
    """

    def test_pure_video_no_audio_num_tokens_fast_path(self):
        payload = {
            "video": {"pixel_values": "fake", "num_tokens": 7, "video_size": []},
            "modality_type": "video",
        }
        items = list(_nano_extract_items(0, _make_param(payload)))
        assert len(items) == 1
        item = items[0]
        assert item.modality == "video"
        assert item.rows == 7
        assert item.prompt_pos == 0
        assert item.mm_idx_per_modality == 0
        assert item.src_param_idx == 0
        # Aggregate payload is the whole per-modality blob (not sliced).
        assert item.payload is payload["video"]

    @pytest.mark.parametrize(
        "modality, src, expected_rows",
        [
            pytest.param("image", "payload_num_tokens", 5, id="image-payload_num_tokens"),
            pytest.param("audio", "payload_num_tokens", 4, id="audio-payload_num_tokens"),
            pytest.param("image", "runtime_total_embeds", 7, id="image-runtime_total_embeds"),
            pytest.param("audio", "runtime_total_embeds", 4, id="audio-runtime_total_embeds"),
        ],
    )
    def test_single_modality_row_source(self, modality, src, expected_rows):
        feature_key = "pixel_values" if modality == "image" else "input_features"
        if src == "payload_num_tokens":
            payload = {feature_key: "fake", "num_tokens": expected_rows}
            param = _make_param({modality: payload, "modality_type": modality})
        else:
            payload = {feature_key: "fake"}
            param = _make_param_with_runtime(
                {modality: payload, "modality_type": modality},
                total_embeds=expected_rows,
            )
        items = list(_nano_extract_items(0, param))
        assert len(items) == 1
        assert items[0].modality == modality
        assert items[0].rows == expected_rows
        assert items[0].prompt_pos == 0
        assert items[0].mm_idx_per_modality == 0

    def test_pure_multi_image_uses_per_param_total_not_first_slot(self):
        # Regression for the dynamic-resolution MMMU failure: a SINGLE pure-image
        # param holding TWO images. Production preprocessing populates per-slot
        # `multimodal_embedding_lengths` ([682, 357]) AND `multimodal_item_order`
        # even for a pure-image request. The Nano vision encoder, however, emits
        # ONE per-request tensor concatenating BOTH images = 1039 rows
        # (== total_embeds_in_request). The single-modality carve-out must size
        # the aggregate item by the per-param total (1039), NOT the first slot
        # (682), and emit ONE item (not per-image), so the bucket assertion holds.
        payload = {"pixel_values": "fake"}
        param = _make_param_with_runtime(
            {
                "image": payload,
                "modality_type": "image",
                "multimodal_item_order": [
                    {"modality": "image", "index": 0},
                    {"modality": "image", "index": 1},
                ],
                "multimodal_embedding_lengths": [682, 357],
            },
            total_embeds=1039,
        )
        items = list(_nano_extract_items(0, param))
        assert len(items) == 1
        assert items[0].rows == 1039
        assert items[0].prompt_pos == 0
        assert items[0].payload is payload

    def test_unknown_modality_raises(self):
        payload = {"weird": {"foo": "bar"}, "modality_type": "weird"}
        with pytest.raises(ValueError, match="Unknown modality"):
            list(_nano_extract_items(0, _make_param(payload)))

    def test_single_modality_missing_sources_raises(self):
        # No num_tokens on the payload AND no runtime populated -> clear error.
        payload = {"pixel_values": "fake"}
        param = _make_param({"image": payload, "modality_type": "image"})
        with pytest.raises(KeyError, match="Cannot resolve"):
            list(_nano_extract_items(0, param))


# ---------------------------------------------------------------------------
# Multi-modality PER-ITEM plan (one ModalityItem per prompt-order slot).
# ---------------------------------------------------------------------------


def _multi_modality_param(item_order, embedding_lengths, payloads) -> MultimodalParams:
    """Build a multi-modality param with prompt-order metadata.

    `item_order` is a list of `(modality, idx)` pairs; `embedding_lengths` is the
    matching per-slot row list; `payloads` maps each present modality to its
    aggregate payload dict.
    """
    multimodal_data = {
        "modality_type": list(dict.fromkeys(m for m, _ in item_order)),
        "multimodal_item_order": [{"modality": m, "index": i} for m, i in item_order],
        "multimodal_embedding_lengths": list(embedding_lengths),
    }
    multimodal_data.update(payloads)
    return _make_param(multimodal_data)


class TestNanoExtractMultiModalityPerItem:
    """Multi-modality params yield one item per `MultimodalPromptOrder` entry."""

    def test_image_video_image_row_order(self):
        # image(0) -> video(0) -> image(1): the repeated image must appear at
        # prompt_pos 2 (after the video), each image as its own item with its
        # own per-slot rows. mm_idx_per_modality tracks the per-modality index.
        order = [("image", 0), ("video", 0), ("image", 1)]
        param = _multi_modality_param(
            order,
            embedding_lengths=[5, 8, 6],
            payloads={"image": {"pixel_values": "imgs"}, "video": {"pixel_values": "vid"}},
        )
        slicer = _StubSlicer()
        items = list(_nano_extract_items(0, param, slice_payload=slicer))
        assert [(it.modality, it.mm_idx_per_modality, it.prompt_pos, it.rows) for it in items] == [
            ("image", 0, 0, 5),
            ("video", 0, 1, 8),
            ("image", 1, 2, 6),
        ]
        # The slicer was asked for each (modality, idx) and the payloads attached.
        assert slicer.calls == [("image", 0), ("video", 0), ("image", 1)]
        assert items[0].payload == {"sliced": ("image", 0)}
        assert items[2].payload == {"sliced": ("image", 1)}

    def test_primary_image_video_audio_interleave_plain_order_no_ghost(self):
        # PRIMARY new case: image -> video(+audio) -> image -> video(+audio).
        # With the audio hoist, the order is a PLAIN flat sequence; the embedded
        # audio is a first-class (audio, k) item at its own prompt rank. No ghost
        # item, no shared slot, no -1 sentinel.
        order = [
            ("image", 0),
            ("video", 0),
            ("audio", 0),
            ("image", 1),
            ("video", 1),
            ("audio", 1),
        ]
        param = _multi_modality_param(
            order,
            embedding_lengths=[5, 8, 3, 5, 8, 3],
            payloads={
                "image": {"pixel_values": "imgs"},
                "video": {"pixel_values": "vids"},
                "audio": {"input_audio_features": "auds"},
            },
        )
        items = list(_nano_extract_items(0, param, slice_payload=_StubSlicer()))
        assert [(it.modality, it.mm_idx_per_modality, it.prompt_pos) for it in items] == [
            ("image", 0, 0),
            ("video", 0, 1),
            ("audio", 0, 2),
            ("image", 1, 3),
            ("video", 1, 4),
            ("audio", 1, 5),
        ]
        assert [it.rows for it in items] == [5, 8, 3, 5, 8, 3]
        # No item carries the legacy -1 sentinel or any embedded-audio flag.
        assert all(it.mm_idx_per_modality >= 0 for it in items)

    def test_mixed_image_audio_standalone(self):
        # image -> audio -> image with a STANDALONE audio (not video-embedded):
        # repeated image plus a first-class audio, all per-item.
        order = [("image", 0), ("audio", 0), ("image", 1)]
        param = _multi_modality_param(
            order,
            embedding_lengths=[5, 4, 5],
            payloads={"image": {"pixel_values": "imgs"}, "audio": {"input_audio_features": "a"}},
        )
        items = list(_nano_extract_items(0, param, slice_payload=_StubSlicer()))
        assert [(it.modality, it.prompt_pos, it.rows) for it in items] == [
            ("image", 0, 5),
            ("audio", 1, 4),
            ("image", 2, 5),
        ]

    def test_multi_modality_unknown_modality_raises(self):
        order = [("image", 0), ("weird", 0)]
        param = _multi_modality_param(
            order,
            embedding_lengths=[5, 4],
            payloads={"image": {"pixel_values": "i"}, "weird": {"x": "y"}},
        )
        with pytest.raises(ValueError, match="Unknown modality"):
            list(_nano_extract_items(0, param, slice_payload=_StubSlicer()))

    def test_multi_modality_embedding_lengths_mismatch_raises(self):
        # item_order length must match multimodal_embedding_lengths length. The
        # invariant is owned by MixedModalEncodeContext.__post_init__, which
        # raises with an "embedding_lengths length" message; the extractor no
        # longer pre-guards it.
        order = [("image", 0), ("video", 0), ("image", 1)]
        param = _multi_modality_param(
            order,
            embedding_lengths=[5, 8],  # one short
            payloads={"image": {"pixel_values": "i"}, "video": {"pixel_values": "v"}},
        )
        with pytest.raises(ValueError, match="embedding_lengths length"):
            list(_nano_extract_items(0, param, slice_payload=_StubSlicer()))

    def test_multi_modality_requires_encode_context(self):
        # A multi-modality param with NO multimodal_item_order key cannot build a
        # transient MixedModalEncodeContext (from_metadata returns None), so the
        # per-item extractor raises a clear error naming both the typed view it
        # could not build and the missing wire key it needs.
        param = _make_param(
            {
                "modality_type": ["image", "audio"],
                "image": {"pixel_values": "imgs"},
                "audio": {"input_audio_features": "auds"},
            }
        )
        with pytest.raises(KeyError, match="MixedModalEncodeContext.*multimodal_item_order"):
            list(_nano_extract_items(0, param, slice_payload=_StubSlicer()))


# ---------------------------------------------------------------------------
# _slice_payload: per-modality single-item payload slicing (sec 3.4).
# ---------------------------------------------------------------------------


class TestNanoPayloadSlicer:
    """`NemotronH_Nano_VL_V2._slice_payload` carves one sub-item out of an
    aggregate per-modality payload. Zero-copy where the source already stores
    per-item lists (dynamic-image, temporal-video); tile-range slice for
    fixed-tile; clip-range slice for audio.
    """

    def _make_slicer(self, *, num_image_token=4):
        model = MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = MagicMock()
        model.vision_encoder.num_image_token = num_image_token
        model._nano_slice_payload = MethodType(NemotronH_Nano_VL_V2._nano_slice_payload, model)
        # `_nano_slice_payload` delegates the fixed-tile offset to
        # `_nano_fixed_tile_offset`; bind the real method too, else the spec'd
        # MagicMock returns a MagicMock offset and the tile-range slice is empty.
        model._nano_fixed_tile_offset = MethodType(
            NemotronH_Nano_VL_V2._nano_fixed_tile_offset, model
        )
        # `_slice_payload` is the per-model method the extractor now calls
        # directly (the `NanoPayloadSlicer` wrapper was dissolved); return the
        # bound method so it is itself the `slice_payload=` callable.
        model._slice_payload = MethodType(NemotronH_Nano_VL_V2._slice_payload, model)
        return model._slice_payload

    def test_dynamic_image_indexes_per_item_lists(self):
        # Dynamic-resolution image payload stores per-image lists; slicing item k
        # indexes element k of each list (zero-copy) and sets num_patches=[1].
        pv = ["pix0", "pix1"]
        payload = {
            "pixel_values": pv,
            "image_sizes": [(56, 56), (84, 84)],
            "num_tokens_per_image": [10, 20],
            "num_patches": torch.tensor([2]),
        }
        param = _make_param({"image": payload, "modality_type": ["image", "audio"]})
        slice_payload = self._make_slicer()
        sliced = slice_payload(param, "image", 1)
        assert sliced["pixel_values"] == ["pix1"]
        assert sliced["image_sizes"] == [(84, 84)]
        assert sliced["num_tokens_per_image"] == [20]
        assert sliced["num_patches"].tolist() == [1]

    def test_fixed_tile_slices_tile_range(self):
        # Fixed-tile image payload stores a flat [total_tiles, 3, H, W] tensor.
        # Image k owns `embedding_lengths[pos] // num_image_token` tiles, and its
        # tile offset is the prefix sum of prior images' tile counts (derived
        # from the per-slot embedding lengths). Here image0 has 4 rows (1 tile)
        # and image1 has 8 rows (2 tiles), so image1 starts at tile 1.
        pixel_values = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2)
        payload = {
            "pixel_values": pixel_values,
            "num_patches": torch.tensor([5]),
            "video_size": None,
        }
        param = _make_param(
            {
                "image": payload,
                "modality_type": ["image", "video"],
                "multimodal_item_order": [
                    {"modality": "image", "index": 0},
                    {"modality": "image", "index": 1},
                    {"modality": "video", "index": 0},
                ],
                "multimodal_embedding_lengths": [4, 8, 7],
            }
        )
        slice_payload = self._make_slicer(num_image_token=4)
        sliced0 = slice_payload(param, "image", 0, rows=4)
        assert sliced0["pixel_values"].shape[0] == 1
        torch.testing.assert_close(sliced0["pixel_values"], pixel_values[0:1])
        assert sliced0["num_patches"].tolist() == [1]
        sliced1 = slice_payload(param, "image", 1, rows=8)
        assert sliced1["pixel_values"].shape[0] == 2
        torch.testing.assert_close(sliced1["pixel_values"], pixel_values[1:3])
        assert sliced1["num_patches"].tolist() == [2]

    def test_fixed_tile_indivisible_rows_raises(self):
        pixel_values = torch.zeros(3, 3, 2, 2)
        payload = {
            "pixel_values": pixel_values,
            "num_patches": torch.tensor([3]),
            "video_size": None,
        }
        param = _make_param({"image": payload, "modality_type": ["image", "audio"]})
        slice_payload = self._make_slicer(num_image_token=4)
        with pytest.raises(ValueError, match="num_image_token"):
            slice_payload(param, "image", 0, rows=6)  # 6 % 4 != 0

    def test_temporal_video_list_indexes_per_video(self):
        # Per-video list layout (aspect-ratio-preserving): index video k.
        pv = [torch.ones(2, 3, 2, 2), torch.ones(2, 3, 4, 4) * 2.0]
        payload = {
            "pixel_values": pv,
            "video_size": [[2, 1, 28, 28], [2, 1, 56, 56]],
        }
        param = _make_param({"video": payload, "modality_type": ["image", "video"]})
        slice_payload = self._make_slicer()
        sliced = slice_payload(param, "video", 1)
        torch.testing.assert_close(sliced["pixel_values"], pv[1])
        assert sliced["video_size"] == [[2, 1, 56, 56]]

    def test_temporal_video_flat_tensor_slices_tile_range(self):
        # Flat tensor layout (all videos same shape): slice [t*p] tiles for video k.
        pixel_values = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2)
        payload = {
            "pixel_values": pixel_values,
            "video_size": [[2, 1, 28, 28], [2, 1, 28, 28]],  # each video: t*p = 2 tiles
        }
        param = _make_param({"video": payload, "modality_type": ["image", "video"]})
        slice_payload = self._make_slicer()
        sliced0 = slice_payload(param, "video", 0)
        torch.testing.assert_close(sliced0["pixel_values"], pixel_values[0:2])
        assert sliced0["video_size"] == [[2, 1, 28, 28]]
        sliced1 = slice_payload(param, "video", 1)
        torch.testing.assert_close(sliced1["pixel_values"], pixel_values[2:4])

    def test_audio_slices_clip_range(self):
        # Aggregate audio payload covering 2 streams with audio_num_clips [2, 1].
        # Slicing stream k carves out its clip-range of features/mask and sets
        # audio_num_clips=[n_clips_k].
        feats = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
        mask = torch.ones(3, 4)
        payload = {
            "input_audio_features": feats,
            "feature_attention_mask": mask,
            "audio_num_clips": torch.tensor([2, 1]),
        }
        param = _make_param({"audio": payload, "modality_type": ["video", "audio"]})
        slice_payload = self._make_slicer()
        sliced0 = slice_payload(param, "audio", 0)
        torch.testing.assert_close(sliced0["input_audio_features"], feats[0:2])
        torch.testing.assert_close(sliced0["feature_attention_mask"], mask[0:2])
        assert sliced0["audio_num_clips"].tolist() == [2]
        sliced1 = slice_payload(param, "audio", 1)
        torch.testing.assert_close(sliced1["input_audio_features"], feats[2:3])
        assert sliced1["audio_num_clips"].tolist() == [1]


# ---------------------------------------------------------------------------
# End-to-end extractor + slicer: real slicing through the per-item plan.
# ---------------------------------------------------------------------------


class TestNanoExtractorWithRealSlicer:
    """The extractor wired to the real bound `_slice_payload` method yields
    per-item payloads that are correctly sliced sub-items (not the aggregate
    blob)."""

    def _make_slicer(self, *, num_image_token=4):
        model = MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = MagicMock()
        model.vision_encoder.num_image_token = num_image_token
        model._nano_slice_payload = MethodType(NemotronH_Nano_VL_V2._nano_slice_payload, model)
        # `_nano_slice_payload` delegates the fixed-tile offset to
        # `_nano_fixed_tile_offset`; bind the real method too, else the spec'd
        # MagicMock returns a MagicMock offset and the tile-range slice is empty.
        model._nano_fixed_tile_offset = MethodType(
            NemotronH_Nano_VL_V2._nano_fixed_tile_offset, model
        )
        # Return the bound per-model `_slice_payload`; it is the `slice_payload=`
        # callable the extractor now invokes directly.
        model._slice_payload = MethodType(NemotronH_Nano_VL_V2._slice_payload, model)
        return model._slice_payload

    def test_fixed_tile_repeated_image_slices_each_image(self):
        # image(0) -> video(0) -> image(1), fixed-tile images. Each image item
        # gets its OWN tile-slice from the shared aggregate pixel_values, sized by
        # its per-slot embedding length // num_image_token.
        img_pixels = torch.arange(3 * 3 * 2 * 2, dtype=torch.float32).reshape(3, 3, 2, 2)
        order = [("image", 0), ("video", 0), ("image", 1)]
        # image0: 4 rows -> 1 tile; image1: 8 rows -> 2 tiles; video: 8 rows.
        param = _multi_modality_param(
            order,
            embedding_lengths=[4, 8, 8],
            payloads={
                "image": {
                    "pixel_values": img_pixels,
                    "num_patches": torch.tensor([3]),
                    "video_size": None,
                },
                "video": {"pixel_values": torch.zeros(2, 3, 2, 2), "video_size": [[2, 1, 28, 28]]},
            },
        )
        items = list(
            _nano_extract_items(0, param, slice_payload=self._make_slicer(num_image_token=4))
        )
        img_items = [it for it in items if it.modality == "image"]
        # image0 owns tile 0; image1 owns tiles 1..2 (its own rows, not folded).
        torch.testing.assert_close(img_items[0].payload["pixel_values"], img_pixels[0:1])
        torch.testing.assert_close(img_items[1].payload["pixel_values"], img_pixels[1:3])


# ---------------------------------------------------------------------------
# Adapters (6-field item; audio adapter simplified — list-only).
# ---------------------------------------------------------------------------


class TestNanoVisionBucketAdapter:
    """Bridges `List[ModalityItem]` (image or video bucket) to the existing
    `self.vision_encoder(params)` interface: one per-item single-modality view,
    per-item outputs cat'd in bucket order, EVS `num_tokens_in_video` stashed.
    """

    def _make_model_stub(self, vision_encoder_return):
        model = MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = MagicMock(return_value=vision_encoder_return)
        model._vision_encoder_adapter = NemotronH_Nano_VL_V2._vision_encoder_adapter.__get__(model)
        model._build_single_modality_param = NemotronH_Nano_VL_V2._build_single_modality_param
        return model

    def test_evs_num_tokens_stashed_on_video_params(self):
        item = ModalityItem(0, "video", 0, 0, 9, {"num_tokens": 9})
        params = [_make_param({"video": item.payload, "modality_type": "video"})]
        emb = torch.randn(9, 4)
        model = self._make_model_stub(([emb], [[3, 3, 3]]))
        out = model._vision_encoder_adapter([item], params)
        assert out.shape == (9, 4)
        assert params[0].multimodal_data["num_tokens_in_video"] == [3, 3, 3]

    def test_mixed_param_synthesizes_single_modality_view(self):
        # Image item from a mixed image+audio param: adapter synthesizes a
        # single-modality view so vision_encoder sees image-only.
        item = ModalityItem(0, "image", 0, 0, 5, {"num_tokens": 5})
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
        out = model._vision_encoder_adapter([item], params)
        assert out.shape == (5, 4)
        passed_params = model.vision_encoder.call_args[0][0]
        assert len(passed_params) == 1
        assert passed_params[0].multimodal_data["modality_type"] == "image"
        assert "audio" not in passed_params[0].multimodal_data

    def test_multi_item_bucket_concatenates_in_order(self):
        items = [
            ModalityItem(0, "image", 0, 0, 5, {"num_tokens": 5}),
            ModalityItem(1, "image", 0, 0, 3, {"num_tokens": 3}),
        ]
        params = [
            _make_param({"image": items[0].payload, "modality_type": "image"}),
            _make_param({"image": items[1].payload, "modality_type": "image"}),
        ]
        emb0 = torch.ones(5, 4)
        emb1 = torch.ones(3, 4) * 2.0
        model = self._make_model_stub(([emb0, emb1], None))
        out = model._vision_encoder_adapter(items, params)
        model.vision_encoder.assert_called_once()
        assert out.shape == (8, 4)
        torch.testing.assert_close(out[:5], emb0)
        torch.testing.assert_close(out[5:], emb1)

    def test_evs_counts_grouped_per_param_and_concatenated_in_order(self):
        # One video bucket spanning TWO requests: param 0 has two videos
        # (interleaved video->video), param 1 has one. Each param must get its OWN
        # per-tubelet counts concatenated in prompt order -- no overwrite within a
        # param, and no bleed across params (the per-src_param_idx grouping). Tubelet
        # counts have distinct lengths AND values so a wrong grouping/order/truncation
        # cannot coincidentally pass.
        items = [
            ModalityItem(0, "video", 0, 0, 6, {"num_tokens": 6}),  # param 0, video 0
            ModalityItem(0, "video", 1, 1, 2, {"num_tokens": 2}),  # param 0, video 1
            ModalityItem(1, "video", 0, 0, 5, {"num_tokens": 5}),  # param 1, video 0
        ]
        params = [
            _make_param({"video": items[0].payload, "modality_type": "video"}),
            _make_param({"video": items[2].payload, "modality_type": "video"}),
        ]
        embs = [torch.randn(6, 4), torch.randn(2, 4), torch.randn(5, 4)]
        num_tokens = [
            torch.tensor([4, 2]),  # param 0, video 0: 2 tubelets
            torch.tensor([2]),  # param 0, video 1: 1 tubelet
            torch.tensor([3, 1, 1]),  # param 1, video 0: 3 tubelets
        ]
        model = self._make_model_stub((embs, num_tokens))
        out = model._vision_encoder_adapter(items, params)
        assert out.shape == (13, 4)
        # param 0: video 0 ++ video 1, in prompt order (NOT [2] from an overwrite).
        torch.testing.assert_close(
            params[0].multimodal_data["num_tokens_in_video"],
            torch.tensor([4, 2, 2]),
        )
        # param 1: its single video only -- untouched by param 0's videos.
        torch.testing.assert_close(
            params[1].multimodal_data["num_tokens_in_video"],
            torch.tensor([3, 1, 1]),
        )


class TestNanoAudioBucketAdapter:
    """`_encode_audio` always returns a list of (embeddings, counts); the adapter
    cats the per-item tensors in bucket order. No scalar-tensor branch, no
    per-clip sidecar (the interleave that needed it is deleted)."""

    def _make_model_stub(self, encode_audio_return):
        model = MagicMock(spec=NemotronH_Nano_VL_V2)
        model._encode_audio = MagicMock(return_value=encode_audio_return)
        model._audio_encoder_adapter = NemotronH_Nano_VL_V2._audio_encoder_adapter.__get__(model)
        return model

    def test_list_return_concatenated_in_order(self):
        emb0 = torch.randn(4, 4)
        emb1 = torch.randn(3, 4)
        model = self._make_model_stub([(emb0, [4]), (emb1, [3])])
        items = [
            ModalityItem(0, "audio", 0, 0, 4, {"id": "a0"}),
            ModalityItem(1, "audio", 0, 0, 3, {"id": "a1"}),
        ]
        out = model._audio_encoder_adapter(items, [object(), object()])
        assert out.shape == (7, 4)
        torch.testing.assert_close(out[:4], emb0)
        torch.testing.assert_close(out[4:], emb1)
        model._encode_audio.assert_called_once_with([items[0].payload, items[1].payload])

    def test_single_item_concatenated(self):
        emb = torch.randn(4, 4)
        model = self._make_model_stub([(emb, [4])])
        items = [ModalityItem(0, "audio", 0, 0, 4, {"id": "a0"})]
        out = model._audio_encoder_adapter(items, [object()])
        torch.testing.assert_close(out, emb)
