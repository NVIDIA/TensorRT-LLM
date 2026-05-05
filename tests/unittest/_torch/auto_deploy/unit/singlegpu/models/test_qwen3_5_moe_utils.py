# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen3_5_moe_utils import (
    build_request_item_runs,
    compute_request_chunk_mrope_positions,
    select_request_chunk_multimodal_embeds,
)


def test_build_request_item_runs_legacy_contiguous_layout():
    item_runs = build_request_item_runs([10, 30], [4, 6])

    assert item_runs == [[(10, 4, [])], [(30, 6, [])]]


def test_build_request_item_runs_sparse_layout_returns_canonical_runs():
    item_runs = build_request_item_runs(
        req_mm_positions=[10, 30],
        req_mm_lengths=[5, 3],
        req_mm_item_run_cu_seqlen=[0, 2, 3],
        req_mm_run_positions=[10, 20, 30],
        req_mm_run_lengths=[2, 3, 3],
    )

    assert item_runs == [[(10, 2, []), (20, 3, [])], [(30, 3, [])]]


def test_build_request_item_runs_projects_special_offsets_to_runs():
    item_runs = build_request_item_runs(
        req_mm_positions=[10, 30],
        req_mm_lengths=[5, 3],
        req_mm_item_run_cu_seqlen=[0, 2, 3],
        req_mm_run_positions=[10, 20, 30],
        req_mm_run_lengths=[2, 3, 3],
        req_special_offsets=[1, 3, 7],
    )

    assert item_runs == [[(10, 2, [1]), (20, 3, [1])], [(30, 3, [2])]]


def test_build_request_item_runs_rejects_sparse_length_mismatch():
    with pytest.raises(ValueError, match="do not match item length"):
        build_request_item_runs(
            req_mm_positions=[10],
            req_mm_lengths=[5],
            req_mm_item_run_cu_seqlen=[0, 1],
            req_mm_run_positions=[10],
            req_mm_run_lengths=[4],
        )


def test_select_request_chunk_multimodal_embeds_uses_canonical_non_embed_offsets():
    image_embeds = [torch.arange(6, dtype=torch.float32).reshape(3, 2)]

    chunk_embeds = select_request_chunk_multimodal_embeds(
        req_input_pos=12,
        req_seq_len=2,
        req_mm_item_types=[0],
        req_mm_positions=[10],
        req_mm_lengths=[4],
        req_special_offsets=[1],
        image_embeds_list=image_embeds,
        video_embeds_list=None,
        hidden_size=2,
    )

    assert torch.equal(chunk_embeds, image_embeds[0][1:])


def test_compute_request_chunk_mrope_positions_uses_canonical_non_embed_offsets():
    pos = compute_request_chunk_mrope_positions(
        req_input_pos=10,
        req_seq_len=4,
        req_mm_item_types=[0],
        req_mm_positions=[10],
        req_mm_lengths=[4],
        req_special_offsets=[1],
        req_mm_item_run_cu_seqlen=None,
        req_mm_run_positions=None,
        req_mm_run_lengths=None,
        image_grid_thw=torch.tensor([[1, 2, 6]]),
        video_grid_thw=None,
        spatial_merge_size=2,
        dtype=torch.long,
        device=torch.device("cpu"),
    )

    assert torch.equal(
        pos[:, 0, :],
        torch.tensor(
            [
                [10, 13, 10, 10],
                [10, 13, 10, 10],
                [10, 13, 11, 12],
            ]
        ),
    )
