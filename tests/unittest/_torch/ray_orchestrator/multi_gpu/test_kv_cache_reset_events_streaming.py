# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from _torch.ray_orchestrator.cache_reset_utils import check_cache_reset_events_through_control_rpc


@pytest.mark.ray
@pytest.mark.gpu2
@pytest.mark.timeout(240)
def test_each_attention_dp_rank_publishes_reset_before_control_rpc_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("TLLM_CUDA_OBJECT_COLLECTIVES", "0")
    check_cache_reset_events_through_control_rpc(tmp_path, ranks=2)
