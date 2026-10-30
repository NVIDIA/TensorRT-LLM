# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the reducescatter__r7 variant's certification matrix.

The matrix is ``_reducescatter__r7_op_matrix.py`` beside this file, a copy of
the shared ``comm/_reducescatter_op_matrix.py`` driving this target's variant
``ops/comm/reducescatter__r7``. It is its own launcher; ``_rank_job`` (in the
shared comm test directory) starts it in a fresh interpreter and turns its exit
code into an assertion.
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
def test_reducescatter_op_matrix() -> None:
    _rank_job.run("reducescatter__r7", directory=Path(__file__).resolve().parent)
