# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter boundary push bookkeeping (kernels/m3_draft_lowm arm_push / take_pushed, dispatch.DRAFT_AR_PUSH): the
boundary all-reduce runs prepushed only when the armed call pushed, and only on that call's own output; any other
input raises instead of reducing rows the exchange does not hold. Host state only (no GPU)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.kernels import (  # noqa: E501
    m3_draft_lowm,
)


@pytest.fixture(autouse=True)
def _clean():
    m3_draft_lowm._PUSH_ARM, m3_draft_lowm._PUSHED = None, 0
    yield
    m3_draft_lowm._PUSH_ARM, m3_draft_lowm._PUSHED = None, 0


def test_not_pushed_is_false_and_disarms() -> None:
    m3_draft_lowm._PUSH_ARM = (1, 2, 3, 0, 8, 8)
    assert m3_draft_lowm.take_pushed(torch.zeros(4)) is False
    assert m3_draft_lowm._PUSH_ARM is None


def test_pushed_output_is_true_once() -> None:
    out = torch.zeros(4, 8)
    m3_draft_lowm._PUSHED = out.data_ptr()
    assert m3_draft_lowm.take_pushed(out) is True
    assert m3_draft_lowm._PUSHED == 0
    assert m3_draft_lowm.take_pushed(out) is False


def test_other_input_raises() -> None:
    out = torch.zeros(4, 8)
    other = out.clone()
    m3_draft_lowm._PUSHED = out.data_ptr()
    with pytest.raises(RuntimeError, match="not the boundary all-reduce's input"):
        m3_draft_lowm.take_pushed(other)
    assert m3_draft_lowm._PUSHED == 0
