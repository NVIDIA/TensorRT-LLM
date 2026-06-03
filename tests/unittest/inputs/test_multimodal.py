# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for `tensorrt_llm.inputs.multimodal`.

Covers `MultimodalRuntimeData` cumsum math and the flat-mask producers
(`maybe_compute_mm_embed_cumsum` and the `_compute_mm_masks` helpers). The
prompt-order type now lives on `MixedModalEncodeContext`; its tests are in
`test_mixed_modal_encode_context.py`.
"""

import pytest
import torch

from tensorrt_llm.inputs.multimodal import (
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
