# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the MoE router graph transform (softmax + topk fusion)."""

from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.auto_deploy.custom_ops.moe_router import *  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transformations.library.moe_router import fuse_moe_router
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class SimpleMoERouter(nn.Module):
    """A minimal MoE router module for testing the graph transform."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, normalize: bool = True):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device="cuda", dtype=torch.float16)
        self.top_k = top_k
        self.normalize = normalize

    def forward(self, x: torch.Tensor):
        # This pattern should be matched by the graph transform
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        if self.normalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)
        return topk_weights, topk_indices


@pytest.mark.parametrize(
    "variant, op",
    [
        ("triton", torch.ops.auto_deploy.triton_moe_router),
        ("torch", torch.ops.auto_deploy.torch_moe_router),
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
def test_moe_router_fusion(variant, op, normalize):
    """Test that the MoE router pattern is matched and replaced with the fused op."""
    hidden_size = 64
    num_experts = 8
    top_k = 2
    model = SimpleMoERouter(hidden_size, num_experts, top_k, normalize=normalize)

    x = torch.randn(4, hidden_size, device="cuda", dtype=torch.float16)

    # Get reference output
    with torch.no_grad():
        ref_weights, ref_indices = model(x)

    # Export and transform
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    fuse_moe_router(gm, backend=variant)

    # Verify the fused op is present in the graph
    found_fused_op = any(is_op(n, op) for n in gm.graph.nodes)
    assert found_fused_op, f"Expected {op} in transformed graph, but not found"

    # Verify the original softmax + topk pattern is gone
    has_softmax = any(is_op(n, torch.ops.aten._softmax) for n in gm.graph.nodes)
    has_topk = any(is_op(n, torch.ops.aten.topk) for n in gm.graph.nodes)
    assert not has_softmax, "Softmax should have been replaced"
    assert not has_topk, "Topk should have been replaced"

    # Verify numerical correctness
    gm = gm.to("cuda")
    with torch.no_grad():
        fused_weights, fused_indices = gm(x)

    torch.testing.assert_close(fused_indices, ref_indices)
    torch.testing.assert_close(fused_weights, ref_weights, rtol=1e-3, atol=1e-3)
