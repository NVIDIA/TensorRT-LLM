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

import torch

from . import _rank_job

assert torch.cuda.is_available(), "allgather requires CUDA devices"


def test_allgather_op_matrix() -> None:
    _rank_job.run("allgather")
