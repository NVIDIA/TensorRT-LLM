# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VR TP4 MXDRAFT scope: default off on this target; TRTLLM_M31_ENABLE_MXDRAFT=1 opts in and
TRTLLM_M31_DISABLE_MXDRAFT=1 still wins (both read at import, so each case runs in a fresh process)."""

import os
import subprocess
import sys

import pytest

TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4"
)


@pytest.mark.parametrize(
    "enable, disable, on", [("", "", False), ("1", "", True), ("1", "1", False), ("", "1", False)]
)
def test_mxdraft_tp4_default_off_and_switches(enable: str, disable: str, on: bool) -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_"))
    }
    env.update(TRTLLM_M31_ENABLE_MXDRAFT=enable, TRTLLM_M31_DISABLE_MXDRAFT=disable)
    out = subprocess.run(
        [sys.executable, "-c", f"from {TARGET} import dspark\nprint(dspark.MXDRAFT)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[-1] == str(on)
