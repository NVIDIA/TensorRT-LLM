# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the k3_latent_reduce op's certification matrix.

The matrix is ``_k3_latent_reduce_op_matrix.py`` beside this file, its own W-rank launcher (call sequences over
caller-owned exchanges, so one job, not independent cases); see ``_rank_job`` for why that is left intact.
"""

from pathlib import Path

__extra_import_path__ = ["../../comm"]
import _rank_job  # noqa: E402
import pytest
import torch

assert torch.cuda.is_available(), "k3_latent_reduce requires CUDA devices"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("k3_latent_reduce is certified on sm_100 and sm_103", allow_module_level=True)


# Each case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_k3_latent_reduce_op_matrix() -> None:
    _rank_job.run("k3_latent_reduce", directory=Path(__file__).resolve().parent)
