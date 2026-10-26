# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VR S10 scope (tp4): the lean drafter glue only up to the measured draft length (K3)."""

import subprocess
import sys

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4 import (  # noqa: E501
    dspark,
)


@pytest.mark.parametrize("k, on", [(1, True), (3, True), (4, False), (7, False), (15, False)])
def test_s10_applies_up_to_k3(k: int, on: bool) -> None:
    assert dspark.S10_MAX_DRAFT_LEN == 3
    assert dspark.s10_applies(k) is on


def test_s10_k7_gate_opt_in() -> None:
    """TRTLLM_M31_ENABLE_S10_K7=1 (gate-only, read at import) lifts the bound to every draft length (default K3)."""
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m31_nvfp4__sm_107__tp4 import dspark\n"
        "print(dspark.S10_MAX_DRAFT_LEN, dspark.s10_applies(7))"
    )
    env = {**__import__("os").environ, "TRTLLM_M31_ENABLE_S10_K7": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-2:] == ["63", "True"]
