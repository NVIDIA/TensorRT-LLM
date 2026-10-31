# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the reducescatter op's certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py`` and not a
duplicate of it: that file covers the *fusion patterns* through the
``AllReduce`` module, while this covers the op itself cell by cell. The names
are kept apart so review does not read one as a copy of the other.

The matrix is ``_reducescatter_op_matrix.py`` beside this file, which is its
own launcher and runs two jobs: the ordered check sequence, then a separately
capped job that certifies the one call-order divergence that wedges instead of
lying (it cannot be a normal check, because the job that runs it never
reports). It runs once per world size the entry's cells name: 4 (dep4, every
caller) and 2 (dep2, the qwen4 unified target's disaggregated engines).
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "reducescatter requires CUDA devices"


# Each case spawns its own rank processes over the visible devices. Under
# xdist several workers would fight for the same GPUs, so this must run
# alone -- the same reason the other collective entry in this repo
# (_torch/moe/kernels/serial/test_moe_alltoall.py) carries the marker.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", [2, 4])
def test_reducescatter_op_matrix(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip(f"{world_size} ranks in one node need {world_size} visible GPUs")
    _rank_job.run("reducescatter", world_size)
