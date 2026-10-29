# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the k3_moe_front op's certification matrix.

The matrix beside this file also certifies moe/k3_moe's cells behind the front
(trtllm::k3_moe_front, then the k3_moe entry on a plain or head_flags K3MoeState).
The shared comm/_rank_job helper starts its W-rank launcher as one ordered job.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "comm"))
import _rank_job  # noqa: E402


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="k3_moe_front needs sm_100 or sm_103")


# Each case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_k3_moe_front_op_matrix() -> None:
    _rank_job.run("k3_moe_front", directory=Path(__file__).resolve().parent)
