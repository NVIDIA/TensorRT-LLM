# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3 all-reduce workspace sizing at the one-shot boundary."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.mnnvl_allreduce import (
        K3MnnvlAllReduce,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)


@pytest.mark.cpu_only
@pytest.mark.parametrize("world", (4, 16))
@pytest.mark.parametrize("rows", (1, 3, 8, 16, 33, 64))
def test_k3_workspace_threshold_is_inclusive_and_two_shot_pads(rows, world):
    hidden = 7168
    one_shot = rows * hidden * world * 2
    assert (
        K3MnnvlAllReduce.get_required_workspace_size(rows, hidden, world, torch.bfloat16, one_shot)
        == one_shot
    )
    assert (
        K3MnnvlAllReduce.get_required_workspace_size(
            rows, hidden, world, torch.bfloat16, one_shot - 1
        )
        == 2 * ((rows + world - 1) // world) * world * hidden * 2
    )
