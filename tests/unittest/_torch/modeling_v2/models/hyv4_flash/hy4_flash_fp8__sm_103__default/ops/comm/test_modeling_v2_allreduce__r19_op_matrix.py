# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the allreduce__r19 certification matrix.

The matrix is ``_allreduce__r19_op_matrix.py`` beside this file. It certifies
the Hy4-Flash target's cells for ``op=NONE`` (the plain sum) at hidden 3072:
world 2 -- the disaggregated context engine, both ranks in one node -- at 1,
64, 5120 and 8192 rows; world 4 and world 8 at 1, 64 and 5120 rows.

World 2 and world 4 fit one GB300 node and run here. World 8 spans two trays,
which a single node cannot form; the matrix also starts from the spool's
multi-node environment (``MV2FLOW_JOB_NODES`` and friends, one process per
node), and the world-8 cells are driven by that two-node run rather than
emulated here.
"""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2/comm"]
import _rank_job  # noqa: E402

assert torch.cuda.is_available(), "allreduce__r19 requires CUDA devices"

WORLD_SIZES = (2, 4, 8)


# Each case spawns its own rank processes over the visible devices; under
# xdist several workers would fight for the same GPUs.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_allreduce__r19_op_matrix(world_size: int) -> None:
    visible = torch.cuda.device_count()
    if world_size > visible:
        pytest.skip(
            f"world {world_size} needs {world_size} visible devices; this job has "
            f"{visible}, and a world that spans nodes is driven by the target's "
            f"two-node run of the same matrix"
        )
    _rank_job.run("allreduce__r19", world_size, directory=Path(__file__).parent)
