# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the ``ops/comm/autotuned_allreduce`` certification matrix.

The matrix is ``_autotuned_allreduce_op_matrix.py`` beside this file, its own mpirun launcher (the autotuner runs
only with MPI on); its cells are keyed by world size, so it runs once per size this node holds.
"""

from pathlib import Path

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402
import pytest
import torch

assert torch.cuda.is_available(), "autotuned_allreduce requires CUDA devices"


# Each case starts its own mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_autotuned_allreduce_op_matrix(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip(f"{world_size} ranks in one node need {world_size} visible GPUs")
    _rank_job.run("autotuned_allreduce", world_size, directory=Path(__file__).resolve().parent)
