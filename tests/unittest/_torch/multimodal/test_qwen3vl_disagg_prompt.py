# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VLInputProcessorBase,
    _expand_prompt_token_ids_for_mm_handoff,
)


class _FakeTokenizer:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __call__(self, prompt, return_tensors=None):
        assert prompt == "video prompt"
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([self.input_ids]))


def test_qwen3vl_disagg_video_prompt_expands_video_placeholder():
    """Video placeholder expands to one token per encoder embedding."""
    vision_start_token_id = 151652
    vision_end_token_id = 151653
    image_token_id = 151655
    video_token_id = 151656
    placeholder_id = 200000
    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        text_config=SimpleNamespace(hidden_size=16),
        vision_config=SimpleNamespace(deepstack_visual_indexes=[]),
    )
    processor._tokenizer = _FakeTokenizer(
        [11, vision_start_token_id, video_token_id, vision_end_token_id, 12]
    )
    processor.tllm_multimodal_token_id = placeholder_id

    handoff = processor.build_disagg_prefill_multimodal_inputs(
        {"prompt": "video prompt"},
        [{"tensor_size": (3, 16)}],
    )

    assert handoff.prompt_token_ids == [
        11,
        vision_start_token_id,
        placeholder_id,
        placeholder_id,
        placeholder_id,
        vision_end_token_id,
        12,
    ]
    assert handoff.multimodal_lengths == [4]
    assert handoff.multimodal_positions == [1]
    assert handoff.multimodal_embedding_lengths == [3]
    assert handoff.multimodal_item_run_cu_offsets == [0, 1]
    assert handoff.multimodal_run_positions == [1]
    assert handoff.multimodal_run_lengths == [4]
    assert handoff.special_token_offsets == [0]
    assert handoff.item_types == [1]


def test_qwen3vl_disagg_mixed_prompt_layout_preserves_item_order():
    """Image and video handoff metadata follows prompt order."""
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

    handoff = _expand_prompt_token_ids_for_mm_handoff(
        input_ids,
        [{"tensor_size": (2, 16)}, {"tensor_size": (3, 16)}],
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        placeholder_id=placeholder_id,
    )

    assert handoff.prompt_token_ids == [
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
    assert handoff.multimodal_lengths == [3, 4]
    assert handoff.multimodal_positions == [0, 5]
    assert handoff.multimodal_item_run_cu_offsets == [0, 1, 2]
    assert handoff.multimodal_run_positions == [0, 5]
    assert handoff.multimodal_run_lengths == [3, 4]
    assert handoff.multimodal_embedding_lengths == [2, 3]
    assert handoff.special_token_offsets == [0, 3]
    assert handoff.item_types == [0, 1]


def test_qwen3vl_disagg_prompt_layout_requires_handle_per_placeholder():
    """Each prompt placeholder must have a matching MM handle."""
    with pytest.raises(ValueError, match="placeholders=1, mm_handles=0"):
        _expand_prompt_token_ids_for_mm_handoff(
            torch.tensor([1, 2, 3]),
            [],
            image_token_id=2,
            video_token_id=4,
            vision_start_token_id=1,
            placeholder_id=99,
        )
