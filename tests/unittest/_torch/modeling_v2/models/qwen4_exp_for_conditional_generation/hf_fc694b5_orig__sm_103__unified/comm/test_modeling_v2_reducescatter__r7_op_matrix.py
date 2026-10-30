# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the reducescatter__r7 variant's certification matrix.

The matrix is ``_reducescatter__r7_op_matrix.py`` beside this file, a copy of
the shared ``comm/_reducescatter_op_matrix.py`` driving this target's variant
``ops/comm/reducescatter__r7``. It is its own launcher; ``_rank_job`` (in the
shared comm test directory) starts it in a fresh interpreter and turns its exit
code into an assertion. It runs once per world size the variant's cells name:
4 (dep4) and 2 (dep2, the disaggregated engines).
"""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "reducescatter requires CUDA devices"


# Each case spawns its own rank processes over the visible devices. Under
# xdist several workers would fight for the same GPUs, so this must run
# alone, like the shared collective entries.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", [2, 4])
def test_reducescatter_op_matrix(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        # The single-GPU modeling_v2 stage collects this directory too.
        pytest.skip(f"{world_size} ranks in one node need {world_size} visible GPUs")
    _rank_job.run("reducescatter__r7", world_size, directory=Path(__file__).resolve().parent)
