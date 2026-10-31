# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the allgather op's certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py`` and not a
duplicate of it: that file covers the *fusion patterns* through the
``AllReduce`` module, while this covers the op itself cell by cell -- strategy
x operation, which strategies are bitwise identical, and which combinations
are silently wrong rather than loud. The names are kept apart so review does
not read one as a copy of the other.

The matrix itself is ``_allgather_op_matrix.py`` beside this file, its own
launcher: one job per world size, which runs its checks first on spawned ranks
through ``allgather_pg`` and then on mpirun ranks through the MPI-session
``allgather``; see ``_rank_job`` for why that is left intact. World 4 is the
dep4 layout every caller runs; world 2 is the qwen4 unified target's dep2
generation engine of its disaggregated deployment.
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "allgather requires CUDA devices"


# Each case spawns its own rank processes over the visible devices. Under
# xdist several workers would fight for the same GPUs, so this must run
# alone -- the same reason the other collective entry in this repo
# (_torch/moe/kernels/serial/test_moe_alltoall.py) carries the marker.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", [2, 4])
def test_allgather_op_matrix(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip(f"{world_size} ranks in one node need {world_size} visible GPUs")
    _rank_job.run("allgather", world_size)
