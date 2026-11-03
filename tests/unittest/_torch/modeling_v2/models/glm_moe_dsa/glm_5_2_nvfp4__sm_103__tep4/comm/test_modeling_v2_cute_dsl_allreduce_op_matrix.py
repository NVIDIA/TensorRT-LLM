# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the GLM-5.2 target's Lamport all-reduce entries.

One job certifies all three -- `cute_dsl_allreduce`,
`cute_dsl_allreduce_add_rmsnorm` and `cute_dsl_allreduce_add_rmsnorm_quant_nvfp4`
-- because they share one workspace and its device slot counter. The matrix is
`_cute_dsl_allreduce_op_matrix.py` beside this file, its own 4-rank launcher;
`_rank_job` (in the shared comm test directory) starts it in a fresh
interpreter and turns its exit code into an assertion.
"""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402 — needs the path declared above


def _sm103_devices() -> int:
    """How many of the devices the job would claim are sm_103."""
    if not torch.cuda.is_available():
        return 0
    claimed = range(min(torch.cuda.device_count(), _rank_job.WORLD_SIZE))
    return sum(torch.cuda.get_device_capability(i) == (10, 3) for i in claimed)


# Spawns 4 rank processes over the visible devices. Under xdist several
# workers would fight for the same GPUs, so this must run alone, like the
# shared collective entries.
@pytest.mark.no_xdist
def test_cute_dsl_allreduce_op_matrix() -> None:
    if _sm103_devices() < _rank_job.WORLD_SIZE:
        pytest.skip(f"the Lamport all-reduce entries run on {_rank_job.WORLD_SIZE} sm_103 GPUs")
    _rank_job.run("cute_dsl_allreduce", directory=Path(__file__).resolve().parent)
