# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_WINDOW_CAPPED_DECODE on the sm_107 TP2 target: the window-capped drafter decode applies only up to the
table's batch per draft length (draft length 3: never; 7: up to batch 2), and only with
``TRTLLM_M31_ENABLE_DRAFT_WINDOW_CAPPED_DECODE=1`` (off by default). CPU only."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        dspark,
        m31_switches,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("k, expected", [(1, 0), (3, 0), (5, 0), (7, 2), (15, 2)])
def test_max_batch_by_draft_length(k: int, expected: int) -> None:
    assert dspark.S5_MAX_BATCH_BY_K == {3: 0, 7: 2}
    assert dspark.s5_max_batch_for(k) == expected


def test_it_is_an_opt_in() -> None:
    assert "DRAFT_WINDOW_CAPPED_DECODE" in m31_switches.OPT_INS
