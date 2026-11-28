# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the generation targets' MegaMoE layer matrix (4 GPUs, EP4).

`_megamoe_target_layer_matrix.py` beside this file is its own 4-rank launcher; it drives the dep target's
`megamoe.py` and `weights.py` end to end for one layer against a plain-torch reference (see its docstring).
"""

import os
from pathlib import Path

import pytest
import torch

__extra_import_path__ = [".."]  # the family test directory (_v41_launch)
from _v41_launch import run_launcher  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 4
    or torch.cuda.get_device_capability() != (10, 3),
    reason="the MegaMoE layer matrix runs the sm_103 targets: 4 GB300 GPUs (EP4 over NVLink)",
)

_LAUNCHER = Path(__file__).resolve().parents[1] / "_megamoe_target_layer_matrix.py"


def _devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    assert len(devices) >= 4, f"the matrix needs 4 devices; {len(devices)} visible"
    return ",".join(devices[:4])


# Starts its own 4-rank mpirun over four devices: it must run alone, not under xdist.
@pytest.mark.no_xdist
@pytest.mark.parametrize("target", ["dep"])
def test_megamoe_target_layer_matrix(target) -> None:
    run_launcher(_LAUNCHER, _devices(), env={"TARGET": target})
