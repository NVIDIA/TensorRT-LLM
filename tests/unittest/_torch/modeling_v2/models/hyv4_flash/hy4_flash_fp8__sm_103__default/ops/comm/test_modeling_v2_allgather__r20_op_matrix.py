# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the allgather op's certification matrix, as this
target's ``__r20`` variant certifies it.

The variant's cells are the Hy4-Flash surface -- trailing 3072 at the 1-row,
64-row and 5120-row ends -- and are driven inside the matrix, which is
``_allgather__r20_op_matrix.py`` beside this file: its own 4-rank launcher. See
``_rank_job`` for why that structure is left intact.

The rest of the matrix is the shared entry's, unchanged: uniform and ragged
gathers, dtypes, CUDA graphs at every engine batch size, the guards, and what
call-order disagreement does. Those are the op's behaviour, not a cell of this
target.
"""

from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2/comm"]
import _rank_job  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "allgather requires CUDA devices"


# Each case spawns its own 4 rank processes over every visible device. Under
# xdist several workers would fight for the same GPUs, so this must run
# alone -- the same reason the other collective entry in this repo
# (_torch/moe/kernels/serial/test_moe_alltoall.py) carries the marker.
@pytest.mark.no_xdist
def test_allgather__r20_op_matrix() -> None:
    _rank_job.run("allgather__r20", directory=Path(__file__).resolve().parent)
