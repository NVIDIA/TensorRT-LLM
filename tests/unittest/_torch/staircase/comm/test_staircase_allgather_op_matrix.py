# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the allgather op's certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py`` and not a
duplicate of it: that file covers the *fusion patterns* through the
``AllReduce`` module, while this covers the op itself cell by cell -- strategy
x operation, which strategies are bitwise identical, and which combinations
are silently wrong rather than loud. The names are kept apart so review does
not read one as a copy of the other.

The matrix itself lives in ``allgather_test.py``, which is its own 4-rank
launcher; see ``_rank_job`` for why that is left intact.
"""

import pytest
import torch

from tensorrt_llm._torch.staircase.catalog.comm import _rank_job

assert torch.cuda.is_available(), "allgather requires CUDA devices"


# Each case starts its own 4-rank mpirun over every visible device. Under
# xdist several workers would fight for the same GPUs, so this must run
# alone -- the same reason the other collective entry in this repo
# (_torch/thop/serial/test_moe_alltoall.py) carries the marker.
@pytest.mark.no_xdist
def test_allgather_op_matrix() -> None:
    _rank_job.run("allgather")
