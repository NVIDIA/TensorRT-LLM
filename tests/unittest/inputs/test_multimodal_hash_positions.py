# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from tensorrt_llm.inputs.multimodal import (
    MultimodalInput,
    MultimodalRuntimeData,
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


def test_multimodal_runtime_uses_sparse_hash_positions_for_chunk_bounds():
    run_length = 254
    run_starts = [37, 319, 601, 883, 1165]
    hash_positions = [
        pos for run_start in run_starts for pos in range(run_start, run_start + run_length)
    ]
    special_token_offsets = [0, 253, 254, 507, 508, 761, 762, 1015, 1016, 1269]

    legacy_runtime = MultimodalRuntimeData(
        past_seen_token_num=256,
        mm_token_positions=[37],
        mm_token_lengths=[1270],
        chunk_end_pos=512,
        special_token_offsets=special_token_offsets,
    )

    sparse_runtime = MultimodalRuntimeData(
        past_seen_token_num=256,
        mm_token_positions=[37],
        mm_token_lengths=[1270],
        chunk_end_pos=512,
        special_token_offsets=special_token_offsets,
        mm_token_hash_positions=[hash_positions],
    )

    assert legacy_runtime.num_mm_tokens_in_chunk - legacy_runtime.num_special_tokens_in_chunk == 254
    assert sparse_runtime.num_unseen_mm_tokens == 219
    assert sparse_runtime.num_mm_tokens_in_chunk == 228
    assert sparse_runtime.num_unseen_special_tokens == 1
    assert sparse_runtime.num_special_tokens_in_chunk == 2
    assert sparse_runtime.num_mm_tokens_in_chunk - sparse_runtime.num_special_tokens_in_chunk == 226
