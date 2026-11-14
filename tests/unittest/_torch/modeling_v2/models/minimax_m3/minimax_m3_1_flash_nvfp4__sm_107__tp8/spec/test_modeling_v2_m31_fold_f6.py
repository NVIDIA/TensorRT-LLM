# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F6 (int32 accept counts to the fused context store and advance) is on by default with an import-time kill
switch. CPU only."""

import os
import subprocess
import sys

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        dspark,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def _fold_f6_in_subprocess(value):
    name = "TRTLLM_M31_DISABLE_FOLD_ACCEPT_COUNTS_INT32"
    env = {k: v for k, v in os.environ.items() if k != name}
    if value is not None:
        env[name] = value
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m3_1_flash_nvfp4__sm_107__tp8 import dspark as m; print('FOLD_F6=' + str(m.FOLD_F6))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("FOLD_F6=")][-1]


def test_fold_f6_kill_switch_subprocess():
    """FOLD_F6 is on by default; TRTLLM_M31_DISABLE_FOLD_ACCEPT_COUNTS_INT32=1 (read at import) turns it off."""
    assert isinstance(dspark.FOLD_F6, bool)
    assert _fold_f6_in_subprocess(None) == "FOLD_F6=True"
    assert _fold_f6_in_subprocess("0") == "FOLD_F6=True"
    assert _fold_f6_in_subprocess("1") == "FOLD_F6=False"
