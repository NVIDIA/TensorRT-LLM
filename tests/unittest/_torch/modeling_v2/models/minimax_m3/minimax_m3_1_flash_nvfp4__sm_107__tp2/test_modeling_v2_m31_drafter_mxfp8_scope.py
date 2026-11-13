# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 TP2 drafter MXFP8 scope (DRAFTER_MXFP8): ON by default on this target; TRTLLM_M31_DISABLE_DRAFTER_MXFP8=1
turns it off; a value outside the m31_switches contract fails at import. Read at import, so each case runs in a fresh
process."""

import os
import subprocess
import sys

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import (  # noqa: F401
        minimax_m3_1_flash_nvfp4__sm_107__tp2,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2"


def _env(**switches: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRTLLM_M31_DISABLE_")}
    env.update(switches)
    return env


@pytest.mark.parametrize("disable, on", [("", True), ("1", False), ("0", True)])
def test_drafter_mxfp8_default_on_and_kill_switch(disable: str, on: bool) -> None:
    out = subprocess.run(
        [sys.executable, "-c", f"from {TARGET} import dspark\nprint(dspark.MXDRAFT)"],
        env=_env(TRTLLM_M31_DISABLE_DRAFTER_MXFP8=disable),
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[-1] == str(on)


def test_drafter_mxfp8_kill_switch_typo_fails_at_import() -> None:
    out = subprocess.run(
        [sys.executable, "-c", f"from {TARGET} import dspark"],
        env=_env(TRTLLM_M31_DISABLE_DRAFTER_MXFP8="nope"),
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0 and "TRTLLM_M31_DISABLE_DRAFTER_MXFP8" in out.stderr
