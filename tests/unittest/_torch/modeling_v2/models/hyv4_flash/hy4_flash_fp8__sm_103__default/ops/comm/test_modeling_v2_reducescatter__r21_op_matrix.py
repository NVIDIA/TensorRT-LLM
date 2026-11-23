# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for this target's reducescatter variant (`__r21`).

Local variant of
`tests/unittest/_torch/modeling_v2/comm/test_modeling_v2_reducescatter_op_matrix.py`,
owned by the Hy4-Flash unified target: same launcher contract, with the matrix
beside this file (`_reducescatter__r21_op_matrix.py`) driving
`ops/comm/reducescatter__r21.py`'s cells at this checkpoint's hidden width and
CUDA-graph batch list. `_rank_job` (in the shared comm test directory) starts it
in a fresh interpreter and turns its exit code into an assertion.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

__extra_import_path__ = ["../../../../../comm"]
if TYPE_CHECKING:
    from _torch.modeling_v2.comm import _rank_job
else:
    import _rank_job  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "reducescatter requires CUDA devices"


# Spawns one rank process per visible device. Under xdist several workers
# would fight for the same GPUs, so this must run alone, like the shared
# collective entries.
@pytest.mark.no_xdist
def test_reducescatter__r21_op_matrix() -> None:
    _rank_job.run("reducescatter__r21", directory=Path(__file__).resolve().parent)
