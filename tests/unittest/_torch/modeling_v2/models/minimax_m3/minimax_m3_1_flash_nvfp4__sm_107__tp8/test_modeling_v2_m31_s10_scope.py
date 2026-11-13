# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lean drafter glue scope (DRAFTER_LEAN_GLUE, tp8): it applies at every draft length up to 63."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        dspark,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("k", [1, 3, 4, 7, 15])
def test_s10_applies_at_every_draft_length(k: int) -> None:
    assert dspark.S10_MAX_DRAFT_LEN == 63
    assert dspark.s10_applies(k) is True
