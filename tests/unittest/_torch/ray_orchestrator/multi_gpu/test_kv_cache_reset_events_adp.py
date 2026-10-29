# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from _torch.ray_orchestrator.cache_reset_utils import check_cache_reset_events_through_control_rpc


@pytest.mark.ray
@pytest.mark.gpu2
@pytest.mark.timeout(240)
def test_buffered_attention_dp_reset_with_an_empty_rank(tmp_path, monkeypatch):
    """Exercise the event gather through Gloo's CPU object transport."""
    monkeypatch.setenv("TLLM_CUDA_OBJECT_COLLECTIVES", "0")
    check_cache_reset_events_through_control_rpc(False, tmp_path)
