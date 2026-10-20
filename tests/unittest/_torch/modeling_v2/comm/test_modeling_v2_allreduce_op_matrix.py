# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the allreduce op's certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py``: that file
covers the ``AllReduce`` module, including the MNNVL and symmetric-memory
paths it tries first. This covers the raw op the catalog entry wraps, cell by
cell.

The matrix is ``_allreduce_op_matrix.py`` beside this file. Its cells are
keyed by world size, so it runs once per size the gpt-oss and deepseek default
targets are certified at; world 8 in one node needs a B200/B300. The world-8
MNNVL entry is not collected here: it runs as one srun task per rank (see
``_mnnvl_allreduce_op_matrix.py``).
"""

import _rank_job
import pytest
import torch

assert torch.cuda.is_available(), "allreduce requires CUDA devices"


# Each case spawns its own rank processes over the visible devices; under
# xdist several workers would fight for the same GPUs.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_allreduce_op_matrix(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        # World 8 in one node exists on B200/B300 only; a GB200/GB300 tray
        # holds four GPUs.
        pytest.skip(f"{world_size} ranks in one node need {world_size} visible GPUs")
    _rank_job.run("allreduce", world_size)
