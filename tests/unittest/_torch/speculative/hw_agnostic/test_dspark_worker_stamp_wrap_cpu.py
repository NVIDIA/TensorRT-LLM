# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draft-stamp scalar bounds and neutral reset without device allocation."""

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker
from tensorrt_llm._torch.speculative.dspark_schedule import NEUTRAL_CONFIDENCE_LOGIT

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("previous", [0, (1 << 31) - 2, (1 << 31) - 1])
def test_stamp_sequence_stays_representable_and_wraps_neutrally(previous):
    worker = object.__new__(DSv4DSparkWorker)
    worker._draft_seq_host = previous
    worker._draft_seq_cuda = torch.tensor(previous, dtype=torch.int32)
    worker._confidence_stamp = torch.tensor([previous, 0], dtype=torch.int32)
    worker._confidence_logits = torch.full((2, 3), -3.0)
    pointers = [
        worker._draft_seq_cuda.data_ptr(),
        worker._confidence_stamp.data_ptr(),
        worker._confidence_logits.data_ptr(),
    ]

    assert worker.bump_draft_seq() == previous
    current = (previous + 1) & ((1 << 31) - 1)
    assert worker._draft_seq_host == worker._draft_seq_cuda.item() == current
    assert worker.verified_draft_seq_cuda().item() == current - 1
    if current == 0:
        assert worker._confidence_stamp.tolist() == [0, 0]
        assert torch.all(worker._confidence_logits == NEUTRAL_CONFIDENCE_LOGIT)
        # A real draft emitted at sequence0 is fresh on the next step. Rows
        # without that draft remain neutral even if their reset stamp matches.
        worker._confidence_logits[0].fill_(-2.0)
        assert worker.bump_draft_seq() == 0
        assert worker.verified_draft_seq_cuda().item() == 0
        assert torch.all(worker._confidence_logits[0] == -2.0)
        assert torch.all(worker._confidence_logits[1] == NEUTRAL_CONFIDENCE_LOGIT)
    else:
        assert worker._confidence_stamp.tolist() == [previous, 0]
        assert torch.all(worker._confidence_logits == -3.0)
    assert pointers == [
        worker._draft_seq_cuda.data_ptr(),
        worker._confidence_stamp.data_ptr(),
        worker._confidence_logits.data_ptr(),
    ]


def test_stamp_wrap_without_allocated_confidence_buffers():
    worker = object.__new__(DSv4DSparkWorker)
    worker._draft_seq_host = (1 << 31) - 1
    worker._draft_seq_cuda = None
    worker._confidence_stamp = None
    worker._confidence_logits = None
    assert worker.bump_draft_seq() == (1 << 31) - 1
    assert worker._draft_seq_host == 0
    assert worker.verified_draft_seq_cuda() is None
