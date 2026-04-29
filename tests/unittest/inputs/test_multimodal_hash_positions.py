# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from tensorrt_llm.inputs.multimodal import (
    MultimodalInput,
    find_mm_hash_token_positions,
    find_mm_token_positions,
)


def test_find_mm_hash_token_positions_preserves_sparse_prompt_positions():
    prompt_token_ids = [11, 999, 77, 999, 12]

    hash_positions = find_mm_hash_token_positions(
        input_ids=prompt_token_ids, num_mm_tokens=[2], mm_token_ids=torch.tensor([999])
    )
    start_positions, special_token_positions = find_mm_token_positions(
        input_ids=prompt_token_ids, num_mm_tokens=[2], mm_token_ids=torch.tensor([999])
    )

    assert hash_positions == [[1, 3]]
    assert start_positions == [1]
    assert special_token_positions == []


def test_multimodal_input_from_components_preserves_legacy_uuid_position():
    mm_hashes = [[1, 2, 3, 4, 5, 6, 7, 8]]
    mm_positions = [1]
    mm_lengths = [2]
    mm_uuids = ["uuid-a"]

    mm_input = MultimodalInput.from_components(mm_hashes, mm_positions, mm_lengths, mm_uuids)

    assert mm_input.multimodal_uuids == mm_uuids
    assert mm_input.multimodal_hash_positions is None


def test_multimodal_input_accepts_hash_positions():
    mm_input = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], mm_uuids=["uuid-a"], mm_hash_positions=[[1, 3]]
    )

    assert mm_input.multimodal_hash_positions == [[1, 3]]


def test_multimodal_input_rejects_hash_position_length_mismatch():
    with pytest.raises(ValueError, match="multimodal_hash_positions\\[0\\] length"):
        MultimodalInput.from_components(
            [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], mm_hash_positions=[[1]]
        )


def test_multimodal_input_rejects_unordered_hash_positions():
    with pytest.raises(ValueError, match="strictly increasing"):
        MultimodalInput.from_components(
            [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], mm_hash_positions=[[3, 1]]
        )
