# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REJ_SLOTS on the tp2 target: ON by default (exact: the rejection step reads the same draft floats by slot), and the
kill switch TRTLLM_M31_DISABLE_REJ_SLOTS=1, read at import, turns it off."""

import os
import subprocess
import sys

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling as dispatch,
)


def test_rej_slots_is_on_by_default():
    assert dispatch.REJ_SLOTS


def test_kill_switch():
    code = f"from {dispatch.__name__} import REJ_SLOTS\nprint(REJ_SLOTS)"
    env = {**os.environ, "TRTLLM_M31_DISABLE_REJ_SLOTS": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-1] == "False"
