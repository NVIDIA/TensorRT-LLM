# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MNNVL workspace a default target builds holds every call it can make.

A target sizes the workspace once, for `max_num_tokens` rows, and the kernel
never checks the size. The one-shot / two-shot switch makes the need
non-monotonic in rows, so sizing for `max_rows` alone undersizes some smaller
calls. Plain arithmetic: no GPU, no workspace allocated.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allreduce import (
    _ONE_SHOT_BYTES,
    workspace_bytes,
    workspace_bytes_up_to,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.mnnvl_allreduce import (
    K3MnnvlAllReduce,
)
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

# (hidden, world): the two default targets at their MNNVL sizes.
_SHAPES = [(2880, 8), (2880, 16), (7168, 8), (7168, 16)]


@pytest.mark.parametrize("hidden, world", _SHAPES)
def test_workspace_bytes_is_the_builtin_arithmetic(hidden, world):
    for rows in range(1, 513):
        assert workspace_bytes(rows, hidden, world, torch.bfloat16) == (
            MNNVLAllReduce.get_required_workspace_size(rows, hidden, world, torch.bfloat16)
        ), rows


@pytest.mark.parametrize("hidden, world", _SHAPES)
def test_every_call_up_to_max_rows_fits(hidden, world):
    last_one_shot = _ONE_SHOT_BYTES // (hidden * world * 2)
    # Both sides of the switch, the band where two-shot needs less than the
    # largest one-shot call, and a full prefill chunk.
    for max_rows in sorted({1, last_one_shot, last_one_shot + 1, 64, 80, 81, 128, 8192}):
        have = workspace_bytes_up_to(max_rows, hidden, world, torch.bfloat16)
        need = max(
            workspace_bytes(r, hidden, world, torch.bfloat16) for r in range(1, max_rows + 1)
        )
        assert have == need, (max_rows, have, need)


def test_the_largest_one_shot_call_sets_the_size_below_the_band_edge():
    # gpt-oss at TP8 with max_num_tokens=64: 64 rows run two-shot over 737,280
    # bytes, 22 rows run one-shot over 1,013,760.
    assert workspace_bytes(64, 2880, 8, torch.bfloat16) == 737_280
    assert workspace_bytes(22, 2880, 8, torch.bfloat16) == 1_013_760
    assert workspace_bytes_up_to(64, 2880, 8, torch.bfloat16) == 1_013_760


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
