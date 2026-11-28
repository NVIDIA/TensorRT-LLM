# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the MegaMoE entries' 4-GPU certification matrix.

The matrix itself is `_cute_dsl_megamoe_nvfp4_op_matrix.py` beside this file, which is its own
4-rank launcher (the kernel dispatches tokens over NVLink, so it cannot run in one process). It
covers `cute_dsl_megamoe_nvfp4` (every cell of its `CELLS`) and `megamoe_workspace`, and runs the other `megamoe_*`
entries as the composition a target uses.
"""

import os
from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../.."]  # the family test directory (_v41_launch)
from _v41_launch import run_launcher  # noqa: E402

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 4
    or any(torch.cuda.get_device_capability(i) != (10, 3) for i in range(4)),
    reason="the MegaMoE matrix needs 4 SM103 GPUs (EP4 over NVLink; the entry's ARCHS is SM103 only)",
)

_LAUNCHER = Path(__file__).resolve().with_name("_cute_dsl_megamoe_nvfp4_op_matrix.py")
_WORLD = 4


def _devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    assert len(devices) >= _WORLD, f"the matrix needs {_WORLD} devices; {len(devices)} visible"
    return ",".join(devices[:_WORLD])


# Starts its own 4-rank mpirun over every visible device: it must run alone, not under xdist.
@pytest.mark.no_xdist
def test_cute_dsl_megamoe_nvfp4_op_matrix() -> None:
    run_launcher(_LAUNCHER, _devices())
