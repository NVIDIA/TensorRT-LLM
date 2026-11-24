# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the OPROJ_PUSH_GEMM boundary certification (``minimax_m31_ar_prepushed``).

The matrix is ``_ar_prepushed_op_matrix.py`` beside this file, its own 2-rank launcher; see ``_rank_job``. It requires
the prepushed one-shot all-reduce + RMSNorm + MXFP8 to equal the stock AllReduce(ONESHOT) + mxfp8_quantize bit for bit,
fed both by a producer doing the stock push and by the o_proj GEMM with the Lamport-push epilogue.
"""

import _rank_job
import pytest
import torch

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
        reason="the sm_107 TP2 target runs on SM107 only",
    ),
    pytest.mark.skipif(
        torch.cuda.device_count() < _rank_job.WORLD_SIZE,
        reason=f"certified at world size {_rank_job.WORLD_SIZE}",
    ),
]


# Spawns 2 rank processes over the visible devices; under xdist workers would fight for them.
@pytest.mark.no_xdist
def test_ar_prepushed_op_matrix() -> None:
    out = _rank_job.run("ar_prepushed")
    assert "[ar_prepushed]" in out, f"the launcher reported no result:\n{out}"
