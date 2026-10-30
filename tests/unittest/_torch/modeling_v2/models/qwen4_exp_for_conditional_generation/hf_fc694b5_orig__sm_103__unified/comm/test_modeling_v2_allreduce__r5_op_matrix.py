# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the qwen4 unified target's allreduce variant
(`ops/comm/allreduce__r5`), a copy of `catalog/comm/allreduce` that adds the
plain sum (`op=NONE`) at hidden 2560.

The matrix is `_allreduce__r5_op_matrix.py` beside this file, its own launcher;
`_rank_job` (in the shared comm test directory) starts it in a fresh
interpreter and turns its exit code into an assertion. Its cells are keyed by
world size, so it runs once per size: world 2 drives the inherited
RESIDUAL_RMS_NORM cells, world 4 those and the qwen4 `NONE` cells, and world 8
in one node needs a B300. The variant claims sm_103 only, so a size runs only
on that many sm_103 devices.
"""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "allreduce requires CUDA devices"


def _sm103_devices(world_size: int) -> int:
    """How many of the devices a `world_size` job would claim are sm_103."""
    claimed = range(min(torch.cuda.device_count(), world_size))
    return sum(torch.cuda.get_device_capability(i) == (10, 3) for i in claimed)


# Each case spawns its own rank processes over the visible devices; under
# xdist several workers would fight for the same GPUs.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_allreduce__r5_op_matrix(world_size: int) -> None:
    if _sm103_devices(world_size) < world_size:
        # World 8 in one node exists on B300 only; a GB300 tray holds four GPUs.
        pytest.skip(f"{world_size} ranks need {world_size} visible sm_103 GPUs")
    _rank_job.run("allreduce__r5", world_size, directory=Path(__file__).resolve().parent)
