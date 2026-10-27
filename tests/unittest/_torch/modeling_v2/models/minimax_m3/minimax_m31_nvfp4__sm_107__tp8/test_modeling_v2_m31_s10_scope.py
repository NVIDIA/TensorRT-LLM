# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VR S10 scope (tp8): the lean drafter glue at every draft length (K3 and K7 gated token-identical)."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    dspark,
)


@pytest.mark.parametrize("k", [1, 3, 4, 7, 15])
def test_s10_applies_at_every_draft_length(k: int) -> None:
    assert dspark.S10_MAX_DRAFT_LEN == 63
    assert dspark.s10_applies(k) is True
