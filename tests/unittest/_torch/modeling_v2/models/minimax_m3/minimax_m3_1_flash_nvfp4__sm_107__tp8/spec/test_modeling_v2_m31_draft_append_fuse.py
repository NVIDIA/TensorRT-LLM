# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_APPEND_FUSE on the sm_107 tp8 target: default ON, kill switch
TRTLLM_M31_DISABLE_DRAFT_APPEND_FUSE, and the target hands the family's qk_norm_rope_append (through its counting
wrapper) to the DSpark worker only with EX11 (the fused drafter q / k norm + RoPE). The fused append's bits are
covered by the family's ops/dspark qk_norm_rope tests."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        dspark,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp8"
_DSPARK = Path(_family.__file__).resolve().parent / TARGET / "dspark.py"


def test_on_by_default_and_off_by_its_kill_switch() -> None:
    assert isinstance(dspark.DRAFT_APPEND_FUSE, bool)
    probe = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        f"{TARGET} import dspark as d; print(d.DRAFT_APPEND_FUSE)"
    )
    for value, expected in ((None, "True"), ("1", "False")):
        env = {k: v for k, v in os.environ.items() if "DRAFT_APPEND_FUSE" not in k}
        if value is not None:
            env["TRTLLM_M31_DISABLE_DRAFT_APPEND_FUSE"] = value
        out = subprocess.run(
            [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
        )
        assert out.stdout.strip().splitlines()[-1] == expected, value


def test_wired_after_ex11_with_the_counting_wrapper() -> None:
    tree = ast.parse(_DSPARK.read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "enable_fused_qk_norm_rope_append"
    ]
    assert len(calls) == 1
    src = ast.unparse(calls[0].args[1])
    assert src == "_qk_norm_rope_append_logged if ex11 and DRAFT_APPEND_FUSE else None", src
