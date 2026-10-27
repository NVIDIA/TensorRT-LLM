# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VR S5 scope (tp2): the TP8 table of the window-capped drafter decode (OFF by default, K3-bundle gate c; a TP2
table needs its own microbench), only up to the measured batch per draft length."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    dspark,
)


@pytest.mark.parametrize("k, expected", [(1, 0), (3, 0), (5, 0), (7, 2), (15, 2)])
def test_s5_max_batch_by_draft_length(k: int, expected: int) -> None:
    assert dspark.S5_MAX_BATCH_BY_K == {3: 0, 7: 2}
    assert dspark.s5_max_batch_for(k) == expected
