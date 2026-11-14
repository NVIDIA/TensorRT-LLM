# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp2ctx target's model, derived from the built-in MiniMax-M3 model, stays a target: it registers no architecture
and no op under the built-in model's names, and its relative imports reach only its own modules and the family's,
never another target. Reads source text only. CPU only.
"""

import re
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

pytestmark = pytest.mark.cpu_only

_FORK = (
    Path(_family.__file__).resolve().parent
    / "minimax_m3_1_flash_nvfp4__sm_107__tp2ctx"
    / "modeling.py"
)


def test_the_fork_registers_no_architecture_and_no_builtin_op_name():
    fork = _FORK.read_text()
    assert "@register_auto_model(" not in fork
    assert "trtllm::minimax_m3_" not in fork


def test_the_forks_relative_imports_stay_in_the_target_and_the_family():
    relative = re.findall(r"^[ \t]*from (\.+)(\S*) import", _FORK.read_text(), flags=re.M)
    assert relative
    for dots, module in relative:
        assert len(dots) <= 2, (dots, module)
        if len(dots) == 2:
            assert not module.startswith("minimax_m3"), f"imports another target: ..{module}"
