# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the qwen4_exp unified target's `allgather__r6` matrix.

The matrix is `_allgather__r6_op_matrix.py` beside this file, its own launcher:
one job per world size, which runs its checks first on spawned ranks through
`allgather_pg` and then on mpirun ranks through the MPI-session `allgather`.
`_rank_job` (in the shared comm test directory) starts it in a fresh
interpreter and turns its exit code into an assertion.

World 4 is the target's dep4 member, world 2 the dep2 generation engine of its
disaggregated deployment.
"""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "allgather__r6 requires CUDA devices"


# Each case spawns its own rank processes over the visible devices. Under
# xdist several workers would fight for the same GPUs, so this must run alone,
# like the shared collective entries.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", [2, 4])
def test_allgather__r6_op_matrix(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip(f"{world_size} ranks need {world_size} visible GPUs")
    _rank_job.run("allgather__r6", world_size, directory=Path(__file__).resolve().parent)
