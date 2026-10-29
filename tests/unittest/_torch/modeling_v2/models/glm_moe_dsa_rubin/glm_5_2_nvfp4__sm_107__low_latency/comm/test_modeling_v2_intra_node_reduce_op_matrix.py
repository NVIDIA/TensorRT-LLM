# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real four-rank IPC comparison with shared kernels, dirty ranges and graph replay."""

import _rank_job
import pytest

from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="owned GLM IPC kernels require SM107"
)


@pytest.mark.no_xdist
@pytest.mark.parametrize("cta1", ["0", "1"])
@pytest.mark.parametrize("fp32", ["0", "1"])
def test_intra_node_reduce_op_matrix(monkeypatch, cta1, fp32):
    monkeypatch.setenv("TRTLLM_MOE_FINALIZE_CTA1", cta1)
    monkeypatch.setenv("ALL_REDUCE_FUSION_KERNEL_ACC_FP32", fp32)
    _rank_job.run("intra_node_reduce")
