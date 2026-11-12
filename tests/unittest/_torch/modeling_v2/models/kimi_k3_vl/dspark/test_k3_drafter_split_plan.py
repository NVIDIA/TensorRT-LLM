# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The DSpark drafter's split-KV plan (``_launch_mla_split_plan``) is main's planner kernel on the
CuTe DSL MLA decode's geometry at the drafter's piece cost. At main's piece cost it must write
the plan main's own launcher (``cute_dsl_mla.launch_mla_split_kv_plan``) writes."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import cute_dsl_mla


def _runner():
    try:
        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
            CuteDSLNVMlaDecodeBlackwellRunner,
        )
    except ImportError:
        return None
    return CuteDSLNVMlaDecodeBlackwellRunner


requires_blackwell_mla_decode = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or _runner() is None,
    reason="needs an SM100-family GPU and the CuTe DSL MLA decode runner",
)


@requires_blackwell_mla_decode
@pytest.mark.parametrize("num_heads, seq_len_q", [(64, 8), (64, 6), (32, 8), (128, 1)])
@pytest.mark.parametrize("num_requests", [1, 3, 17, 64, 200])
def test_drafter_plan_matches_mains_launcher_at_mains_cost(num_heads, seq_len_q, num_requests):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark import drafter

    runner = _runner()
    # The CuTe DSL occupancy query runs on the current CUDA context, which the first
    # allocation creates (a serving drafter's weights are on the device by then).
    torch.empty(1, device="cuda")
    rows, waves = drafter._mla_split_plan_geometry(num_heads, seq_len_q)
    assert rows == runner.get_num_q_tiles(num_heads, seq_len_q)
    assert waves == max(runner._get_max_active_blocks() // runner._CLUSTER_SHAPE_MNK[0], 1)

    gen = torch.Generator(device="cuda").manual_seed(num_requests * 13 + num_heads + seq_len_q)
    seq_lens = torch.randint(
        1, 200_000, (num_requests,), device="cuda", generator=gen, dtype=torch.int32
    )
    expected = torch.full((num_requests + 5,), -1, dtype=torch.int32, device="cuda")
    cute_dsl_mla.launch_mla_split_kv_plan(
        seq_lens, expected, num_requests=num_requests, rows_per_request=rows, wave_clusters=waves
    )
    got = torch.full_like(expected, -1)
    drafter._launch_mla_split_plan(
        seq_lens,
        got,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        piece_cost=cute_dsl_mla._SPLIT_PLAN_PIECE_COST_TILES,
    )
    torch.cuda.synchronize()
    assert torch.equal(got, expected)
    written = got[:num_requests]
    assert ((written >= 1) & (written <= cute_dsl_mla._SPLIT_PLAN_MAX_SPLIT)).all()
