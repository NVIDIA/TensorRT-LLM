# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import torch.nn as nn
from _graph_test_helpers import run_test_transformed_gm
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer


class DummyMambaALogModule(nn.Module):
    def __init__(self, num_features=16, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.register_parameter(
            "A_log",
            nn.Parameter(torch.randn(num_features, device=device, dtype=dtype)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fused_a = -torch.exp(self.A_log.float())
        return inputs + fused_a

    def get_input(self, device="cuda", dtype=torch.float32) -> torch.Tensor:
        return torch.randn(self.A_log.shape[0], device=device, dtype=dtype)


def _apply_fuse_mamba_a_log(gm: GraphModule) -> GraphModule:
    return InferenceOptimizer(
        None,
        {
            "fuse_mamba_a_log": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)


def test_fuse_mamba_a_log_creates_fused_param():
    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(42)

    model = DummyMambaALogModule(num_features=8, dtype=dtype, device=device).to(
        device=device, dtype=dtype
    )
    x = model.get_input(device=device, dtype=dtype)

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = _apply_fuse_mamba_a_log(gm)

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        lambda gm_out: any(
            node.op == "get_attr" and str(node.target).endswith("A_fused")
            for node in gm_out.graph.nodes
        ),
        lambda num: num,
        atol=1e-5,
        rtol=1e-5,
        test_load_hook=False,
        strict_loading=True,
    )

    fused_params = [
        name for name, _ in gm_transformed.named_parameters() if name.endswith("A_fused")
    ]
    assert fused_params, "Expected fused A parameter to be registered."
    assert not any(name.endswith("A_log") for name, _ in gm_transformed.named_parameters()), (
        "A_log parameter should be removed after fusion."
    )
    assert not any(
        node.target in {torch.exp, torch.ops.aten.exp.default}
        for node in gm_transformed.graph.nodes
    ), "exp node should be removed after fusion."
