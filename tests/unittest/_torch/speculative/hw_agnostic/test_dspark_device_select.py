# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for fresh-confidence DSpark device-window selection."""

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_device_select import (
    gather_packed_draft_tokens,
    select_windows_device,
)
from tensorrt_llm._torch.speculative.dspark_schedule import DSparkScheduleConfig


def test_fresh_confidence_spends_budget_on_the_best_real_row():
    result = select_windows_device(
        confidence_logits=torch.tensor(
            [[10.0, 10.0, 10.0, 10.0, 10.0], [-10.0, -10.0, -10.0, -10.0, -10.0]]
        ),
        slot_idx=torch.tensor([0, 1, 0, 0]),
        num_real=torch.tensor(2),
        budget=torch.tensor(2),
        graph_num_tokens=8,
        cfg=DSparkScheduleConfig(block_size=5, min_verify_len=1),
        pad_len=1,
    )

    assert result.verify_lens.tolist() == [4, 2, 1, 1]
    assert result.qo_indptr.tolist() == [0, 4, 6, 7, 8]
    assert result.req_idx.tolist() == [0, 0, 0, 0, 1, 1, 2, 3]
    assert result.kv_correction.tolist() == [-3, -2, -1, 0, -1, 0, 0, 0]


def test_stale_row_fails_open_to_neutral_confidence():
    result = select_windows_device(
        confidence_logits=torch.full((2, 5), -10.0),
        slot_idx=torch.tensor([0, 1]),
        num_real=torch.tensor(2),
        budget=torch.tensor(1),
        graph_num_tokens=5,
        cfg=DSparkScheduleConfig(block_size=5, min_verify_len=1),
        stamp=torch.tensor([7, 6]),
        expected_stamp=torch.tensor(7),
    )

    assert result.verify_lens.tolist() == [2, 3]


def test_gather_packed_draft_tokens_excludes_each_bonus():
    packed = gather_packed_draft_tokens(
        next_draft_tokens=torch.tensor([[10, 11, 12], [20, 21, 22]]),
        batch_slots=torch.tensor([1, 0]),
        verify_lens=torch.tensor([3, 2], dtype=torch.int32),
        qo_indptr=torch.tensor([0, 3, 5], dtype=torch.int32),
        num_real=2,
        total_draft_tokens=3,
    )

    assert packed.tolist() == [20, 21, 10]


def test_gather_packed_draft_tokens_rejects_negative_size():
    with pytest.raises(ValueError, match="non-negative"):
        gather_packed_draft_tokens(
            next_draft_tokens=torch.empty((1, 1)),
            batch_slots=torch.tensor([0]),
            verify_lens=torch.tensor([1], dtype=torch.int32),
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
            num_real=1,
            total_draft_tokens=-1,
        )
