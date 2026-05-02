# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from tensorrt_llm.inputs.multimodal import (
    MultimodalInput,
    MultimodalRuntimeData,
    _find_mm_token_start_pos_from_masks,
    find_mm_token_item_runs,
    get_multimodal_embedding_lengths,
)


def test_find_mm_token_start_pos_from_masks_returns_two_lists_for_all_text():
    # Regression: an all-False `mm_mask` (text-only prompt) must return the
    # declared `(start_positions, start_special_token_positions)` 2-tuple so
    # the caller's tuple unpack stays valid.
    mm_mask = torch.zeros(5, dtype=torch.bool)
    special_mask = torch.zeros(5, dtype=torch.bool)

    start_positions, start_special_token_positions = _find_mm_token_start_pos_from_masks(
        mm_mask, special_mask, num_mm_tokens=[]
    )

    assert start_positions == []
    assert start_special_token_positions == []


def test_find_mm_token_item_runs_preserves_sparse_prompt_coverage():
    prompt_token_ids = [11, 999, 77, 999, 12]

    item_runs = find_mm_token_item_runs(
        input_ids=prompt_token_ids, num_mm_tokens=[2], mm_token_ids=torch.tensor([999])
    )

    assert item_runs == [[(1, 1, []), (3, 1, [])]]
    assert [runs[0][0] for runs in item_runs] == [1]


def test_find_mm_token_item_runs_preserves_non_embed_offsets():
    prompt_token_ids = [11, 999, 999, 777, 999, 777, 12]

    item_runs = find_mm_token_item_runs(
        input_ids=prompt_token_ids,
        num_mm_tokens=[5],
        mm_token_ids=torch.tensor([999]),
        mm_special_token_ids=torch.tensor([777]),
    )

    assert item_runs == [[(1, 5, [2, 4])]]
    assert get_multimodal_embedding_lengths(item_runs) == [3]


def test_multimodal_input_from_components_preserves_legacy_uuid_position():
    mm_hashes = [[1, 2, 3, 4, 5, 6, 7, 8]]
    mm_positions = [1]
    mm_lengths = [2]
    mm_uuids = ["uuid-a"]

    mm_input = MultimodalInput.from_components(mm_hashes, mm_positions, mm_lengths, mm_uuids)

    assert mm_input.multimodal_uuids == mm_uuids
    assert mm_input.multimodal_item_runs == [[(1, 2, [])]]


def test_multimodal_input_accepts_item_runs():
    mm_input = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]],
        [1],
        [2],
        mm_uuids=["uuid-a"],
        mm_item_runs=[[(1, 1, []), (3, 1, [])]],
    )

    assert mm_input.multimodal_item_runs == [[(1, 1, []), (3, 1, [])]]


def test_multimodal_input_derives_position_and_length_from_item_runs():
    mm_input = MultimodalInput(
        multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
        multimodal_item_runs=[[(1, 1, []), (3, 1, [])]],
    )

    assert mm_input.multimodal_positions == [1]
    assert mm_input.multimodal_lengths == [2]
    assert mm_input.multimodal_embedding_lengths == [2]


def test_multimodal_input_derives_embedding_lengths_from_non_embed_offsets():
    mm_input = MultimodalInput(
        multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
        multimodal_item_runs=[[(1, 5, [2, 4])]],
    )

    assert mm_input.multimodal_positions == [1]
    assert mm_input.multimodal_lengths == [5]
    assert mm_input.multimodal_embedding_lengths == [3]
    assert mm_input.multimodal_item_runs == [[(1, 5, [2, 4])]]


def test_multimodal_input_rejects_item_run_length_mismatch():
    with pytest.raises(ValueError, match="multimodal_item_runs\\[0\\] total length"):
        MultimodalInput.from_components(
            [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], mm_item_runs=[[(1, 1, [])]]
        )


def test_multimodal_input_rejects_overlapping_item_runs():
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        MultimodalInput.from_components(
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            [1],
            [2],
            mm_item_runs=[[(1, 2, []), (2, 1, [])]],
        )


@pytest.mark.parametrize(
    "item_runs",
    [
        [[(1, 1), (3, 1)]],
        [[((1, 1), []), ((3, 1), [])]],
    ],
)
def test_multimodal_input_rejects_non_canonical_item_run_shapes(item_runs):
    with pytest.raises(TypeError, match="prompt_start, run_length, non_embed_offsets"):
        MultimodalInput.from_components(
            [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], mm_item_runs=item_runs
        )


def _embed_cumsum_from_runs(item_runs, prompt_len):
    embed_mask = torch.zeros(prompt_len, dtype=torch.bool)
    for item_run in item_runs:
        for run in item_run:
            run_start, run_length, non_embed_offsets = run
            embed_mask[run_start : run_start + run_length] = True
            for offset in non_embed_offsets:
                embed_mask[run_start + offset] = False
    return embed_mask.cumsum(0, dtype=torch.int64)


def test_multimodal_runtime_uses_sparse_embed_cumsum_for_chunk_bounds():
    run_length = 254
    run_starts = [37, 319, 601, 883, 1165]
    item_runs = [[(run_start, run_length, []) for run_start in run_starts]]
    prompt_len = 1420
    legacy_contiguous_runs = [[(37, 1270, [])]]

    legacy_runtime = MultimodalRuntimeData(
        past_seen_token_num=256,
        chunk_end_pos=512,
        embed_mask_cumsum=_embed_cumsum_from_runs(legacy_contiguous_runs, prompt_len),
    )

    sparse_runtime = MultimodalRuntimeData(
        past_seen_token_num=256,
        chunk_end_pos=512,
        embed_mask_cumsum=_embed_cumsum_from_runs(item_runs, prompt_len),
    )

    assert legacy_runtime.num_mm_tokens_in_chunk == 256
    assert sparse_runtime.num_cached_mm_tokens == 219
    assert sparse_runtime.num_mm_tokens_in_chunk == 228
    assert sparse_runtime.total_embeds_in_request == 1270
