# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VR TP2 MXDRAFT scope (the TP8 default): ON by default on this target;
TRTLLM_M31_DISABLE_MXDRAFT=1 turns it off (an ENABLE variable changes nothing); a value outside the m31_switches
contract fails at import. Read at import, so each case runs in a fresh process."""

import os
import subprocess
import sys

import pytest

TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2"
)


def _env(**switches: str) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_"))
    }
    env.update(switches)
    return env


@pytest.mark.parametrize(
    "enable, disable, on",
    [("", "", True), ("1", "", True), ("1", "1", False), ("", "1", False), ("", "0", True)],
)
def test_mxdraft_tp2_default_on_and_kill_switch(enable: str, disable: str, on: bool) -> None:
    out = subprocess.run(
        [sys.executable, "-c", f"from {TARGET} import dspark\nprint(dspark.MXDRAFT)"],
        env=_env(TRTLLM_M31_ENABLE_MXDRAFT=enable, TRTLLM_M31_DISABLE_MXDRAFT=disable),
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[-1] == str(on)


def test_mxdraft_kill_switch_typo_fails_at_import() -> None:
    out = subprocess.run(
        [sys.executable, "-c", f"from {TARGET} import dspark"],
        env=_env(TRTLLM_M31_DISABLE_MXDRAFT="nope"),
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0 and "TRTLLM_M31_DISABLE_MXDRAFT" in out.stderr
