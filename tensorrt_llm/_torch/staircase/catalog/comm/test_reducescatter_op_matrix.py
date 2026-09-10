# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the reducescatter op's certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py`` and not a
duplicate of it: that file covers the *fusion patterns* through the
``AllReduce`` module, while this covers the op itself cell by cell. The names
are kept apart so review does not read one as a copy of the other.

The matrix lives in ``reducescatter_test.py``, which is its own launcher and
runs two jobs: the ordered test sequence, then a separately capped job that
certifies the one call-order divergence that wedges instead of lying (it
cannot be a normal test, because the job that runs it never reports).
"""

import torch

from . import _rank_job

assert torch.cuda.is_available(), "reducescatter requires CUDA devices"


def test_reducescatter_op_matrix() -> None:
    _rank_job.run("reducescatter")
