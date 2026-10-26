# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VR S5 scope (tp4): the window-capped drafter decode only up to the measured batch per draft length."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4 import (  # noqa: E501
    dspark,
)


@pytest.mark.parametrize("k, expected", [(1, 4), (3, 4), (5, 4), (7, 4), (15, 4)])
def test_s5_max_batch_by_draft_length(k: int, expected: int) -> None:
    assert dspark.S5_MAX_BATCH_BY_K == {3: 4, 7: 4}
    assert dspark.s5_max_batch_for(k) == expected
