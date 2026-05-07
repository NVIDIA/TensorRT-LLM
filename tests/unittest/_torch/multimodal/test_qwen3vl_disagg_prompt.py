# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3vl import _expand_prompt_token_ids_for_mm_handoff


def test_qwen3vl_disagg_video_prompt_layout_includes_non_embedding_start():
    vision_start_token_id = 151652
    vision_end_token_id = 151653
    video_token_id = 151656
    image_token_id = 151655
    placeholder_id = 200000
    input_ids = torch.tensor([11, vision_start_token_id, video_token_id, vision_end_token_id, 12])

    expanded, lengths, positions, layout = _expand_prompt_token_ids_for_mm_handoff(
        input_ids,
        [{"tensor_size": (3, 16)}],
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        placeholder_id=placeholder_id,
    )

    assert expanded == [
        11,
        vision_start_token_id,
        placeholder_id,
        placeholder_id,
        placeholder_id,
        vision_end_token_id,
        12,
    ]
    assert lengths == [4]
    assert positions == [1]
    assert layout["multimodal_item_run_cu_offsets"] == [0, 1]
    assert layout["multimodal_run_positions"] == [1]
    assert layout["multimodal_run_lengths"] == [4]
    assert layout["multimodal_embedding_lengths"] == [3]
    assert layout["special_token_offsets"] == [0]
    assert layout["item_types"] == [1]


def test_qwen3vl_disagg_mixed_prompt_layout_preserves_item_order():
    vision_start_token_id = 151652
    vision_end_token_id = 151653
    image_token_id = 151655
    video_token_id = 151656
    placeholder_id = 200000
    input_ids = torch.tensor(
        [
            vision_start_token_id,
            image_token_id,
            vision_end_token_id,
            42,
            vision_start_token_id,
            video_token_id,
            vision_end_token_id,
        ]
    )

    expanded, lengths, positions, layout = _expand_prompt_token_ids_for_mm_handoff(
        input_ids,
        [{"tensor_size": (2, 16)}, {"tensor_size": (3, 16)}],
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        placeholder_id=placeholder_id,
    )

    assert expanded == [
        vision_start_token_id,
        placeholder_id,
        placeholder_id,
        vision_end_token_id,
        42,
        vision_start_token_id,
        placeholder_id,
        placeholder_id,
        placeholder_id,
        vision_end_token_id,
    ]
    assert lengths == [3, 4]
    assert positions == [0, 5]
    assert layout["multimodal_item_run_cu_offsets"] == [0, 1, 2]
    assert layout["multimodal_run_positions"] == [0, 5]
    assert layout["multimodal_run_lengths"] == [3, 4]
    assert layout["multimodal_embedding_lengths"] == [2, 3]
    assert layout["special_token_offsets"] == [0, 3]
    assert layout["item_types"] == [0, 1]


def test_qwen3vl_disagg_prompt_layout_requires_handle_per_placeholder():
    with pytest.raises(ValueError, match="placeholders=1, mm_handles=0"):
        _expand_prompt_token_ids_for_mm_handoff(
            torch.tensor([1, 2, 3]),
            [],
            image_token_id=2,
            video_token_id=4,
            vision_start_token_id=1,
            placeholder_id=99,
        )
