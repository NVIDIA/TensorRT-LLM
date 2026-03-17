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

"""Tests for the FuseGdnGating graph transform."""

import torch
from _graph_test_helpers import run_test_transformed_gm
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.custom_ops.fla import (
    gdn_gating as _gdn_gating_ops,  # noqa: F401
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class GdnGatingModel(torch.nn.Module):
    """Minimal model that uses torch_fused_gdn_gating followed by a linear."""

    def __init__(self, num_heads: int = 16, hidden: int = 64):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, num_heads, device="cuda", dtype=torch.float16)
        self.A_log = torch.nn.Parameter(torch.randn(num_heads, device="cuda", dtype=torch.float16))
        self.dt_bias = torch.nn.Parameter(
            torch.randn(num_heads, device="cuda", dtype=torch.float16)
        )
        self.out = torch.nn.Linear(num_heads, hidden, device="cuda", dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.proj(x)  # [B, S, H]
        g = torch.ops.auto_deploy.torch_fused_gdn_gating(self.A_log, a, self.dt_bias)
        return self.out(g.to(x.dtype))


def test_fuse_gdn_gating():
    """Verify FuseGdnGating replaces torch source op with triton op."""
    model = GdnGatingModel()

    def checker(gm):
        return any(is_op(n, torch.ops.auto_deploy.triton_fused_gdn_gating) for n in gm.graph.nodes)

    x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16)
    dynamic_shapes = {0: Dim.DYNAMIC}
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dynamic_shapes,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_gdn_gating": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        checker,
        lambda num_p_og: num_p_og,
        dynamic_shapes=dynamic_shapes,
    )

    # Also verify with different batch size (dynamic shapes)
    new_input = torch.randn(4, 8, 64, device="cuda", dtype=torch.float16)
    y_transformed = gm_transformed(new_input)
    y_model = model(new_input)
    torch.testing.assert_close(y_transformed, y_model, atol=1e-3, rtol=1e-3)


def test_no_match_without_source_op():
    """FuseGdnGating should be a no-op when no torch_fused_gdn_gating ops exist."""

    class PlainModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(64, 64, device="cuda", dtype=torch.float16)

        def forward(self, x):
            return self.linear(x)

    model = PlainModel()
    x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16)
    dynamic_shapes = {0: Dim.DYNAMIC}
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dynamic_shapes,), clone=True)

    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_gdn_gating": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    # No triton ops should appear
    has_triton_op = any(
        is_op(n, torch.ops.auto_deploy.triton_fused_gdn_gating) for n in gm_transformed.graph.nodes
    )
    assert not has_triton_op, "triton_fused_gdn_gating should not appear when there is no source op"
