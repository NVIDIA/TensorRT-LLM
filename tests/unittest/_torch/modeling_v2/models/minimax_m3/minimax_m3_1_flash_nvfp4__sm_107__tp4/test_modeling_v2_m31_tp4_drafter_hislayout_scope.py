# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SM107 TP4 HISLAYOUT scope: default ON; TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT=1 turns it off; MXDRAFT
stays default OFF (read at import, so each case runs in a fresh process)."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_only

TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4"


@pytest.mark.parametrize("disable, layout_on, mx", [("", True, False), ("1", False, False)])
def test_hislayout_default_on_and_kill_switch(disable: str, layout_on: bool, mx: bool) -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_"))
    }
    env.update(TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT=disable)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from {TARGET} import dspark\nprint(dspark.HISLAYOUT, dspark.MXDRAFT)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[-2:] == [str(layout_on), str(mx)]
