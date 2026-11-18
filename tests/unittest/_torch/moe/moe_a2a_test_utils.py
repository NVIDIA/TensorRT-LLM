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
"""Helpers of the tests of the MoE All-to-All ops, with every logical peer on one GPU."""

import torch

from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm.bindings import internal

CAPACITY, HIDDEN = 8, 128


def a2a_workspace(ep_size: int, tokens: int = CAPACITY) -> tuple[torch.Tensor, ...]:
    meta = NVLinkOneSided._make_workspace_layout(
        ep_size, 1, tokens, HIDDEN, torch.bfloat16, None, 32, rank_mask=True
    )
    size = int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    workspace = torch.zeros((ep_size, size), dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, ep_size, barrier=False)
    experts = torch.zeros((tokens, 1), dtype=torch.int32, device="cuda")
    payload = torch.arange(tokens * HIDDEN, dtype=torch.bfloat16, device="cuda")
    return workspace, meta, experts, payload.reshape(tokens, HIDDEN)


def a2a_dispatch(workspace, meta, experts, controls, payload, tokens=CAPACITY, rank=0):
    """Dispatch [payload, experts] like the MoE layer as ``rank``."""
    ep_size = workspace.size(0)
    args = (experts, [payload, experts], workspace, meta, tokens, rank, ep_size, 1, ep_size, None)
    return torch.ops.trtllm.moe_a2a_dispatch(*args, **controls)


def a2a_combine(workspace, meta, payload, offset, controls, tokens=CAPACITY, rank=0):
    """Combine like the MoE layer as ``rank``."""
    args = (payload, tokens, workspace, meta, tokens, rank, workspace.size(0), 1, offset, False)
    return torch.ops.trtllm.moe_a2a_combine(*args, **controls)
