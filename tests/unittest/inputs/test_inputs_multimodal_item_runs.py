# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

import tensorrt_llm.inputs.multimodal as multimodal_module
from tensorrt_llm.inputs.multimodal import (
    MultimodalInput,
    MultimodalRuntimeData,
    _find_mm_token_start_pos_from_masks,
    find_mm_token_item_runs,
    get_multimodal_embedding_lengths,
    validate_mm_item_runs,
)
from tensorrt_llm.inputs.registry import create_input_processor_with_hash


class _HashingTestInputProcessor:
    def __init__(self):
        self.multimodal_hashing_supported = None
        self.call_count = 0

    def __call__(self, inputs, sampling_params):
        self.call_count += 1
        return [11, 999, 999, 12], {"multimodal_data": {}}

    def get_num_tokens_per_image(self, *, image):
        return 2

    def get_vocab_size(self):
        return None

    def get_mm_token_ids(self):
        return torch.tensor([999])

    def get_mm_special_token_ids(self):
        return None


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


def test_validate_mm_item_runs_rejects_prompt_overflow():
    with pytest.raises(AssertionError, match="exceeding input sequence length"):
        validate_mm_item_runs(
            prompt_token_ids=[11, 999],
            mm_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
            mm_item_runs=[[(1, 2, [])]],
        )


def test_validate_mm_item_runs_rejects_global_overlap():
    with pytest.raises(AssertionError, match="globally non-overlapping"):
        validate_mm_item_runs(
            prompt_token_ids=[999, 999, 999],
            mm_hashes=[
                [1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1],
            ],
            mm_item_runs=[[(1, 2, [])], [(2, 1, [])]],
        )


def test_validate_mm_item_runs_allows_interleaved_non_overlapping_items():
    validate_mm_item_runs(
        prompt_token_ids=[999, 999, 999, 999, 999],
        mm_hashes=[
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        mm_item_runs=[[(1, 1, []), (4, 1, [])], [(3, 1, [])]],
    )


def test_multimodal_input_from_components_preserves_uuid():
    mm_hashes = [[1, 2, 3, 4, 5, 6, 7, 8]]
    mm_item_runs = [[(1, 2, [])]]
    mm_uuids = ["uuid-a"]

    mm_input = MultimodalInput.from_components(mm_hashes, mm_item_runs, mm_uuids)

    assert mm_input.multimodal_uuids == mm_uuids
    assert mm_input.multimodal_item_runs == mm_item_runs
    assert mm_input.multimodal_embedding_lengths == [2]
    assert mm_input.multimodal_prompt_lengths == [2]


def test_multimodal_input_accepts_item_runs():
    mm_input = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]],
        [[(1, 1, []), (3, 1, [])]],
        mm_uuids=["uuid-a"],
    )

    assert mm_input.multimodal_item_runs == [[(1, 1, []), (3, 1, [])]]
    assert mm_input.multimodal_embedding_lengths == [2]
    assert mm_input.multimodal_prompt_lengths == [2]


def test_create_input_processor_with_hash_preserves_item_runs_and_support():
    input_processor = _HashingTestInputProcessor()
    wrapped_processor = create_input_processor_with_hash(input_processor)

    prompt_token_ids, extra_processed_inputs = wrapped_processor(
        {
            "prompt": "text <image> text",
            "multi_modal_data": {"image": torch.tensor([1, 2, 3])},
        },
        sampling_params=object(),
    )

    multimodal_input = extra_processed_inputs["multimodal_input"]
    assert prompt_token_ids == [11, 999, 999, 12]
    assert input_processor.call_count == 1
    assert input_processor.multimodal_hashing_supported is True
    assert isinstance(multimodal_input, MultimodalInput)
    assert multimodal_input.multimodal_item_runs == [[(1, 2, [])]]
    assert multimodal_input.multimodal_embedding_lengths == [2]
    assert multimodal_input.multimodal_prompt_lengths == [2]
    assert len(multimodal_input.multimodal_hashes) == 1
    assert len(multimodal_input.multimodal_hashes[0]) == 8


def test_multimodal_input_rejects_legacy_layout_fields():
    with pytest.raises(TypeError, match="multimodal_positions"):
        MultimodalInput(
            multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
            multimodal_positions=[1],
            multimodal_lengths=[2],
        )


def test_multimodal_input_requires_item_runs():
    with pytest.raises(TypeError, match="multimodal_item_runs"):
        MultimodalInput(multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]])


def test_multimodal_input_stores_lengths_from_item_runs_once(monkeypatch):
    mm_input = MultimodalInput(
        multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
        multimodal_item_runs=[[(1, 1, []), (3, 1, [])]],
    )

    def fail_on_recompute(item_runs):
        raise AssertionError("multimodal lengths should be cached")

    monkeypatch.setattr(multimodal_module, "get_multimodal_embedding_lengths", fail_on_recompute)

    assert mm_input.multimodal_embedding_lengths == [2]
    assert mm_input.multimodal_prompt_lengths == [2]


def test_multimodal_input_derives_embedding_lengths_from_non_embed_offsets():
    mm_input = MultimodalInput(
        multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
        multimodal_item_runs=[[(1, 5, [2, 4])]],
    )

    assert mm_input.multimodal_embedding_lengths == [3]
    assert mm_input.multimodal_prompt_lengths == [5]
    assert mm_input.multimodal_item_runs == [[(1, 5, [2, 4])]]


def test_multimodal_input_rejects_item_run_length_mismatch():
    with pytest.raises(ValueError, match="multimodal_item_runs length"):
        MultimodalInput.from_components(
            [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
            [[(1, 1, [])]],
        )


def test_multimodal_input_rejects_overlapping_item_runs():
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        MultimodalInput.from_components(
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            [[(1, 2, []), (2, 1, [])]],
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
        MultimodalInput.from_components([[1, 2, 3, 4, 5, 6, 7, 8]], item_runs)


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
