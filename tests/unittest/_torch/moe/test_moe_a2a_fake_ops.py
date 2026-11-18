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
"""Check that the fake implementations of the MoE all-to-all ops accept every schema argument."""

import pytest
import torch
from torch._subclasses import FakeTensorMode

import tensorrt_llm  # noqa: F401

pytestmark = pytest.mark.cpu_only


def test_moe_a2a_fake_ops_accept_every_schema_argument():
    """Pass the rank-mask arguments as keywords, as a bound NVLinkOneSided layer does.

    They are not their defaults, so the dispatcher hands all of them to the fake implementations.
    """
    ep_rank, ep_size, max_tokens, top_k, num_experts, hidden = 0, 2, 4, 2, 8, 16
    with FakeTensorMode():
        experts = torch.empty(max_tokens, top_k, dtype=torch.int32)
        payload = torch.empty(max_tokens, hidden)
        workspace = torch.empty(ep_size, 1024, dtype=torch.uint8)
        metainfo = torch.empty(16, dtype=torch.int64)
        rank_mask = {
            "enable_rank_mask": True,
            "active_rank_mask": torch.empty(1, dtype=torch.uint64),
            "abort_state": torch.empty(16, dtype=torch.int32),
        }
        torch.ops.trtllm.moe_a2a_initialize(workspace, metainfo, ep_rank, ep_size, barrier=False)
        layout = (workspace, metainfo, max_tokens, ep_rank, ep_size, top_k)
        # EPLB stats, expert-id payload index and invalid token expert id.
        dispatch_options = (None, 0, num_experts)
        recv, offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            experts, [payload], *layout, num_experts, *dispatch_options, **rank_mask
        )
        # payload_in_workspace and use_low_precision.
        combine_options = (False, False)
        output = torch.ops.trtllm.moe_a2a_combine(
            recv[0], max_tokens, *layout, offset, *combine_options, **rank_mask
        )
        torch.ops.trtllm.moe_a2a_check_incomplete_round(
            workspace, metainfo, ep_rank, rank_mask["abort_state"]
        )
    assert recv[0].shape == (ep_size, max_tokens, hidden)
    assert output.shape == (max_tokens, hidden)
