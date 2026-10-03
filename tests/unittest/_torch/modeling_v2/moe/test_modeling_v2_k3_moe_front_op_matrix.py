# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the k3_moe_front op's certification matrix.

The matrix also certifies moe/k3_moe's fused-front cells (k3_moe_fused_front on a plain and a head_flags
K3MoeState). It is ``comm/_k3_moe_front_op_matrix.py``: rank bodies live beside ``_lockstep`` and ``_rank_job`` in
``comm/``, and it is its own W-rank launcher (call sequences over caller-owned workspaces and states, so one job, not
independent cases); see ``_rank_job`` for why that is left intact. This file puts ``comm/`` on the import path to
reach ``_rank_job``.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comm"))
import _rank_job  # noqa: E402


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="k3_moe_front needs sm_100")


# Each case starts its own W-rank mpirun over the visible devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_k3_moe_front_op_matrix() -> None:
    _rank_job.run("k3_moe_front")
