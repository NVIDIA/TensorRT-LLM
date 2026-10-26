# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter boundary push bookkeeping (``M3Eagle3Draft.arm_push`` / ``take_pushed``, ``DRAFT_AR_PUSH``): the
boundary all-reduce runs prepushed only when the armed call pushed, and only on that call's own output; any other
input raises instead of reducing rows the exchange does not hold. Host state only (no GPU)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling,
)


@pytest.fixture
def draft():
    """A drafter with only its push bookkeeping (the module and its weights are not built)."""
    d = object.__new__(modeling.M3Eagle3Draft)
    d._push_arm, d._pushed = None, 0
    return d


def test_not_pushed_is_false_and_disarms(draft) -> None:
    draft._push_arm = (1, 2, 3, 0, 8, 8)
    assert draft.take_pushed(torch.zeros(4)) is False
    assert draft._push_arm is None


def test_pushed_output_is_true_once(draft) -> None:
    out = torch.zeros(4, 8)
    draft._pushed = out.data_ptr()
    assert draft.take_pushed(out) is True
    assert draft._pushed == 0
    assert draft.take_pushed(out) is False


def test_other_input_raises(draft) -> None:
    out = torch.zeros(4, 8)
    other = out.clone()
    draft._pushed = out.data_ptr()
    with pytest.raises(RuntimeError, match="not the boundary all-reduce's input"):
        draft.take_pushed(other)
    assert draft._pushed == 0
