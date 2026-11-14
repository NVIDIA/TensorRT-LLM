# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SM107 S5 scope (tp4): the window-capped drafter decode only up to the measured batch per draft length."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
    dspark,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("k, expected", [(1, 4), (3, 4), (5, 1), (7, 1), (15, 1)])
def test_s5_max_batch_by_draft_length(k: int, expected: int) -> None:
    assert dspark.S5_MAX_BATCH_BY_K == {3: 4, 7: 1}
    assert dspark.s5_max_batch_for(k) == expected


def test_s5_on_s9b_t6_opt_in() -> None:
    """S5 is on by default at this target (its own natural-AL gate passed, in-model win at batch 1); S9b
    (drafter-only precision, not bitwise, no in-model win here) stay opt-in; S2 stays off with the draft length 3
    set."""
    assert dspark.S5_DEFAULT_ON is True
    assert dspark.S9B_DEFAULT_ON is False
