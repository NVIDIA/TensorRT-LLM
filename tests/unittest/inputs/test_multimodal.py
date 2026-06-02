# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for `tensorrt_llm.inputs.multimodal`.

Covers the `MultimodalPromptOrder` prompt-order type, `MultimodalRuntimeData`
cumsum math, and the flat-mask producers (`maybe_compute_mm_embed_cumsum` and
the `_compute_mm_masks` helpers).
"""

import pytest
import torch

from tensorrt_llm.inputs.multimodal import (
    MultimodalPromptOrder,
    MultimodalRuntimeData,
    _compute_mm_masks,
    _find_mm_embedding_lengths_from_masks,
)
from tensorrt_llm.inputs.registry import maybe_compute_mm_embed_cumsum


def test_maybe_compute_mm_embed_cumsum_populates_py_multimodal_data():
    """Producer writes a flat int64 cumsum tensor at py_multimodal_data[multimodal_embed_mask_cumsum]."""

    class FakeProcessor:
        def get_vocab_size(self):
            return 1000

        def get_mm_token_ids(self):
            return None

        def get_mm_special_token_ids(self):
            return torch.tensor([2000])

    # [text, img, img, special, img, img, img, text]
    prompt_token_ids = [10, 1001, 1002, 2000, 1003, 1004, 1005, 20]
    extra = {"multimodal_data": {}}
    maybe_compute_mm_embed_cumsum(prompt_token_ids, extra, FakeProcessor())

    cumsum = extra["multimodal_data"]["multimodal_embed_mask_cumsum"]
    torch.testing.assert_close(
        cumsum,
        torch.tensor([0, 1, 2, 2, 3, 4, 5, 5], dtype=torch.int64),
        rtol=0,
        atol=0,
    )


class TestMultimodalPromptOrder:
    """Tests for the MultimodalPromptOrder class."""

    # ---- Constructors ----

    def test_default_single_modality(self):
        sentinel_a = object()
        sentinel_b = object()
        sentinel_c = object()
        result = MultimodalPromptOrder.default({"image": [sentinel_a, sentinel_b, sentinel_c]})
        assert list(result) == [("image", 0), ("image", 1), ("image", 2)]
        assert isinstance(result, MultimodalPromptOrder)

    def test_default_empty_mm_items_returns_empty(self):
        result = MultimodalPromptOrder.default({})
        assert list(result) == []
        assert isinstance(result, MultimodalPromptOrder)

    def test_from_raw_entries_dict_form(self):
        result = MultimodalPromptOrder.from_raw_entries(
            [{"modality": "image", "index": 1}, {"type": "video"}],
            source="x",
        )
        assert list(result) == [("image", 1), ("video", 0)]

    def test_from_raw_entries_tuple_form(self):
        result = MultimodalPromptOrder.from_raw_entries([("image", 0), ("video", 2)], source="x")
        assert list(result) == [("image", 0), ("video", 2)]

    def test_from_raw_entries_rejects_unsupported_types(self):
        # float is none of dict/str/2-tuple
        with pytest.raises(ValueError):
            MultimodalPromptOrder.from_raw_entries([3.5], source="x")
        # bare int: falls to tuple unpack → ValueError
        with pytest.raises(ValueError):
            MultimodalPromptOrder.from_raw_entries([7], source="x")

    def test_from_metadata_prefers_multimodal_item_order(self):
        # Mixed dict + tuple entry shapes resolve to the same (modality, index) pairs.
        result = MultimodalPromptOrder.from_metadata(
            {
                "multimodal_item_order": [
                    {"modality": "image", "index": 1},
                    {"type": "video"},
                    ("image", 0),
                ]
            }
        )
        assert list(result) == [("image", 1), ("video", 0), ("image", 0)]
        assert isinstance(result, MultimodalPromptOrder)

    def test_from_metadata_returns_none_when_absent(self):
        assert MultimodalPromptOrder.from_metadata(None) is None
        assert MultimodalPromptOrder.from_metadata({}) is None
        assert MultimodalPromptOrder.from_metadata({"other": 1}) is None

    def test_resolve_uses_metadata_when_present(self):
        # processor hook must NOT be called when metadata is present
        class PoisonProcessor:
            def get_mm_item_order(self, *args, **kwargs):
                raise AssertionError("get_mm_item_order must not be called")

        a, b, c = object(), object(), object()
        mm_data = {"image": [a, b], "video": [c]}
        multimodal_data = {
            "multimodal_item_order": [
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 0},
                {"modality": "image", "index": 1},
            ]
        }
        result = MultimodalPromptOrder.resolve(
            mm_data,
            PoisonProcessor(),
            prompt_token_ids=[1, 2, 3],
            multimodal_data=multimodal_data,
        )
        assert list(result) == [("video", 0), ("image", 0), ("image", 1)]

    def test_resolve_uses_default_for_single_modality(self):
        class NoHookProcessor:
            pass

        a, b = object(), object()
        result = MultimodalPromptOrder.resolve(
            {"image": [a, b]},
            NoHookProcessor(),
        )
        assert list(result) == [("image", 0), ("image", 1)]

    def test_resolve_uses_input_processor_protocol_for_multi_modality(self):
        a, b = object(), object()

        class OrderingProcessor:
            def get_mm_item_order(self, prompt_token_ids, mm_data):
                return [("video", 0), ("image", 0)]

        result = MultimodalPromptOrder.resolve(
            {"image": [a], "video": [b]},
            OrderingProcessor(),
            prompt_token_ids=[1, 2],
        )
        assert list(result) == [("video", 0), ("image", 0)]

    def test_resolve_falls_back_to_default_when_processor_lacks_hook(self):
        class NoHookProcessor:
            pass

        a, b = object(), object()
        result = MultimodalPromptOrder.resolve(
            {"image": [a], "video": [b]},
            NoHookProcessor(),
            prompt_token_ids=[1, 2],
        )
        # default order is modality-major: image first, then video
        assert list(result) == [("image", 0), ("video", 0)]

    # ---- Validation ----

    def test_validate_rejects_unknown_modality(self):
        a = object()
        with pytest.raises(ValueError, match="modality 'audio'"):
            MultimodalPromptOrder([("audio", 0)]).validate({"image": [a]})

    def test_validate_rejects_out_of_bounds_index(self):
        a = object()
        with pytest.raises(ValueError, match=r"image\[5\]"):
            MultimodalPromptOrder([("image", 5)]).validate({"image": [a]})

    def test_validate_rejects_coverage_mismatch(self):
        a, b = object(), object()
        # order covers only 1 image item but mm_items has 2
        with pytest.raises(ValueError, match="expected 2"):
            MultimodalPromptOrder([("image", 0)]).validate({"image": [a, b]})

    def test_validate_rejects_duplicate_index(self):
        a, b = object(), object()
        # order references image[0] twice and never references image[1];
        # the per-modality count (2) matches len(items) (2), so a count-only
        # check would wrongly pass. validate must reject the duplicate reference
        # (and thereby the uncovered image[1]).
        with pytest.raises(ValueError, match=r"references image\[0\] more than once"):
            MultimodalPromptOrder([("image", 0), ("image", 0)]).validate({"image": [a, b]})

    # ---- Projections ----

    def test_flatten_reorders_by_key(self):
        result = MultimodalPromptOrder([("image", 0), ("video", 0), ("image", 1)]).flatten(
            {"image": [10, 11], "video": [20]}
        )
        assert result == [10, 20, 11]

    def test_flatten_uuids_passes_through_none(self):
        result = MultimodalPromptOrder([("image", 0)]).flatten_uuids(None)
        assert result is None

    def test_flatten_uuids_handles_missing_modality(self):
        result = MultimodalPromptOrder([("image", 0), ("video", 0)]).flatten_uuids({"image": ["a"]})
        assert result == ["a", None]


def test_mixed_image_video_audio_masks_runs_embedding_lengths():
    image_token = 100
    video_token = 200
    audio_token = 300
    video_start = 201
    video_end = 202
    audio_start = 301
    audio_end = 302
    # 10 img img 11 | vid_start vid vid vid_end 12 | aud_start aud aud_end 13
    input_ids = torch.tensor(
        [
            10,
            image_token,
            image_token,
            11,
            video_start,
            video_token,
            video_token,
            video_end,
            12,
            audio_start,
            audio_token,
            audio_end,
            13,
        ]
    )
    num_mm_tokens = [2, 4, 3]

    mm_mask, embed_mask, _ = _compute_mm_masks(
        input_ids,
        vocab_size=None,
        mm_token_ids=torch.tensor([image_token, video_token, audio_token]),
        mm_special_token_ids=torch.tensor([video_start, video_end, audio_start, audio_end]),
    )

    # Sole coverage of `_find_mm_embedding_lengths_from_masks` on a mixed
    # request: embed-only tokens per item (specials excluded) are [2, 2, 1].
    # The start-position / run helpers are covered in test_multimodal_runtime.py
    # and test_llm_kv_cache_events.py.
    embedding_lengths = _find_mm_embedding_lengths_from_masks(
        mm_mask,
        embed_mask,
        num_mm_tokens,
    )

    assert embedding_lengths == [2, 2, 1]


def test_runtime_data_cumsum_math_simplest():
    """All-True mask, full request, no cache."""
    is_embed = torch.ones(5, dtype=torch.bool)
    rt = MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=5,
        embed_mask_cumsum=is_embed.cumsum(0, dtype=torch.int64),
    )
    assert rt.num_cached_mm_tokens == 0
    assert rt.num_mm_tokens_in_chunk == 5
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_partial_chunk():
    """Chunk ends before end of mask."""
    is_embed = torch.tensor([True, True, False, True, True, False, True])
    rt = MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=4,
        embed_mask_cumsum=is_embed.cumsum(0, dtype=torch.int64),
    )
    assert rt.num_cached_mm_tokens == 0
    assert rt.num_mm_tokens_in_chunk == 3
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_partial_cache():
    """past_seen_token_num > 0: cached counts embeds before watermark."""
    is_embed = torch.tensor([True, True, False, True, True, False, True])
    rt = MultimodalRuntimeData(
        past_seen_token_num=3,
        chunk_end_pos=7,
        embed_mask_cumsum=is_embed.cumsum(0, dtype=torch.int64),
    )
    assert rt.num_cached_mm_tokens == 2
    assert rt.num_mm_tokens_in_chunk == 3
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_with_specials_mistral_shape():
    """Chunk boundary inside a unit with inline special (Mistral-shape)."""
    # [text, img, img, special, img, img, img, text]
    is_embed = torch.tensor([False, True, True, False, True, True, True, False])
    cumsum = is_embed.cumsum(0, dtype=torch.int64)

    rt0 = MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5, embed_mask_cumsum=cumsum)
    assert rt0.num_cached_mm_tokens == 0
    assert rt0.num_mm_tokens_in_chunk == 3
    assert rt0.total_embeds_in_request == 5

    rt1 = MultimodalRuntimeData(past_seen_token_num=5, chunk_end_pos=8, embed_mask_cumsum=cumsum)
    assert rt1.num_cached_mm_tokens == 3
    assert rt1.num_mm_tokens_in_chunk == 2
    assert rt1.total_embeds_in_request == 5


def test_runtime_data_requires_cumsum():
    """embed_mask_cumsum is required."""
    with pytest.raises(TypeError):
        MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5)
