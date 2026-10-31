# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from _torch.ray_orchestrator.cache_reset_utils import check_cache_reset_events_through_control_rpc


@pytest.mark.ray
@pytest.mark.timeout(240)
def test_streaming_reset_is_published_before_control_rpc_resumes(tmp_path):
    check_cache_reset_events_through_control_rpc(tmp_path)
