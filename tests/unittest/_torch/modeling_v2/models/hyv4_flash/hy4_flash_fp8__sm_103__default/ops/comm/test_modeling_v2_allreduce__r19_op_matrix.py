# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runs ``_allreduce__r19_op_matrix.py`` at each world size the cells name and asserts its exit code.

World 8 spans two GB300 trays; it runs as one plain copy of the matrix per node, not here.
"""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2/comm"]
import _rank_job  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "allreduce__r19 requires CUDA devices"


# Spawns its own ranks over the visible devices; xdist workers would fight for them.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_allreduce__r19_op_matrix(world_size: int) -> None:
    visible = torch.cuda.device_count()
    if world_size > visible:
        pytest.skip(f"world {world_size} needs {world_size} visible devices; {visible} visible")
    _rank_job.run("allreduce__r19", world_size, directory=Path(__file__).resolve().parent)
