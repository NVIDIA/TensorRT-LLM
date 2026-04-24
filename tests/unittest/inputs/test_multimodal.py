# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for MultimodalRuntimeData cumsum math and the flat-mask producer."""

from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm.inputs.multimodal import MultimodalRuntimeData, find_mm_token_lengths
from tensorrt_llm.inputs.registry import compute_mm_embed_cumsum_if_absent


def test_compute_mm_embed_cumsum_if_absent_populates_py_multimodal_data():
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
    compute_mm_embed_cumsum_if_absent(prompt_token_ids, extra, FakeProcessor())

    cumsum = extra["multimodal_data"]["multimodal_embed_mask_cumsum"]
    assert torch.equal(
        cumsum,
        torch.tensor([0, 1, 2, 2, 3, 4, 5, 5], dtype=torch.int64),
    )


def test_runtime_data_cumsum_math_simplest():
    """All-True mask, full request, no cache."""
    is_embed = torch.ones(5, dtype=torch.bool)
    rt = MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=5,
        embed_mask_cumsum=is_embed.to(torch.int64).cumsum(0),
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
        embed_mask_cumsum=is_embed.to(torch.int64).cumsum(0),
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
        embed_mask_cumsum=is_embed.to(torch.int64).cumsum(0),
    )
    assert rt.num_cached_mm_tokens == 2
    assert rt.num_mm_tokens_in_chunk == 3
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_with_specials_mistral_shape():
    """Chunk boundary inside a unit with inline special (Mistral-shape)."""
    # [text, img, img, special, img, img, img, text]
    is_embed = torch.tensor([False, True, True, False, True, True, True, False])
    cumsum = is_embed.to(torch.int64).cumsum(0)

    rt0 = MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5, embed_mask_cumsum=cumsum)
    assert rt0.num_cached_mm_tokens == 0
    assert rt0.num_mm_tokens_in_chunk == 3
    assert rt0.total_embeds_in_request == 5

    rt1 = MultimodalRuntimeData(past_seen_token_num=5, chunk_end_pos=8, embed_mask_cumsum=cumsum)
    assert rt1.num_cached_mm_tokens == 3
    assert rt1.num_mm_tokens_in_chunk == 2
    assert rt1.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_negative_past_seen_rejected():
    """past_seen_token_num must be non-negative."""
    cumsum = torch.arange(1, 6, dtype=torch.int64)
    with pytest.raises(ValueError, match="past_seen_token_num must be non-negative"):
        MultimodalRuntimeData(past_seen_token_num=-1, chunk_end_pos=5, embed_mask_cumsum=cumsum)


def test_runtime_data_requires_cumsum():
    """embed_mask_cumsum is required."""
    with pytest.raises(TypeError):
        MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5)


def _fake_video(num_frames: int = 4):
    """Video must be a list of frames per find_mm_token_lengths contract.

    Non-Tensor placeholder skips the torch.Tensor -> PIL conversion branch.
    """
    return [object() for _ in range(num_frames)]


def test_find_mm_token_lengths_video_fast_path_uses_video_grid_thw():
    """Video fast path uses video_grid_thw.

    When multimodal_data provides video_grid_thw, the per-video count is
    derived from it rather than the slow-path processor call.
    """
    processor = Mock()
    processor.get_num_tokens_per_image = Mock(return_value=100)

    def _count_video(*, video, video_grid_thw=None, **kwargs):
        if video_grid_thw is not None:
            t, h, w = (int(x) for x in video_grid_thw)
            return t * h * w
        return 999  # slow-path sentinel — must not be hit.

    processor.get_num_tokens_per_video = Mock(side_effect=_count_video)

    mm_data = {"video": [_fake_video(), _fake_video(), _fake_video()]}
    vgt = torch.tensor([[2, 14, 14], [1, 7, 7], [3, 28, 28]])
    multimodal_data = {"video": {"video_grid_thw": vgt}}

    result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)

    assert result == {"video": torch.prod(vgt, dim=1).tolist()}
    assert processor.get_num_tokens_per_video.call_count == 3


def test_find_mm_token_lengths_video_grid_thw_shape_mismatch_falls_back():
    """Shape-mismatched video_grid_thw warns and falls back to slow path.

    When video_grid_thw row count does not match the number of videos,
    the fast path is skipped: the processor is called without the
    video_grid_thw kwarg and a single warning is emitted.
    """
    processor = Mock()
    processor.get_num_tokens_per_image = Mock(return_value=100)

    def _count_video(*, video, video_grid_thw=None, **kwargs):
        if video_grid_thw is not None:
            t, h, w = (int(x) for x in video_grid_thw)
            return t * h * w
        return 99  # slow-path sentinel.

    processor.get_num_tokens_per_video = Mock(side_effect=_count_video)

    mm_data = {"video": [_fake_video(), _fake_video()]}
    # 3 rows, 2 videos -> mismatch triggers fallback + warning.
    vgt = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    multimodal_data = {"video": {"video_grid_thw": vgt}}

    with patch("tensorrt_llm.inputs.multimodal.logger.warning") as warn_mock:
        result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)

    assert result == {"video": [99, 99]}
    assert processor.get_num_tokens_per_video.call_count == 2
    for call in processor.get_num_tokens_per_video.call_args_list:
        assert call.kwargs.get("video_grid_thw") is None
    warn_mock.assert_called_once()
    assert "video_grid_thw" in warn_mock.call_args.args[0]


def test_find_mm_token_lengths_image_only_request_unaffected():
    """Image-only requests never invoke the video counter."""
    processor = Mock()
    processor.get_num_tokens_per_image = Mock(return_value=128)
    processor.get_num_tokens_per_video = Mock()

    mm_data = {"image": [object(), object()]}
    result = find_mm_token_lengths(mm_data, processor)

    assert result == {"image": [128, 128]}
    processor.get_num_tokens_per_video.assert_not_called()
