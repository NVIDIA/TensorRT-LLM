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
"""Unit tests for causal conv pattern matcher."""

import pytest
import torch
from _graph_test_helpers import run_test
from torch.export import Dim
from torch.fx import GraphModule

import tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_causal_conv  # noqa: F401
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_DYNAMIC_SHAPES = {0: Dim.DYNAMIC, 1: Dim.DYNAMIC}


class CausalConv1dModel(torch.nn.Module):
    """Model that uses the causal conv pattern: transpose -> conv1d -> slice -> transpose."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return self.conv1d(x.transpose(1, 2))[..., :seq_len].transpose(1, 2)


class StackedCausalConvModel(torch.nn.Module):
    """Model with multiple causal conv layers to test multiple pattern matches."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size, padding=kernel_size - 1)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size, padding=kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self.conv1(x.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x = self.conv2(x.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        return x


def _apply_causal_conv_transform(gm: GraphModule) -> GraphModule:
    return InferenceOptimizer(
        None,
        {"match_causal_conv": {"stage": "pattern_matcher"}},
    )(None, gm)


def _check_graph(gm: GraphModule) -> bool:
    has_custom_op = any(is_op(n, torch.ops.auto_deploy.torch_causal_conv1d) for n in gm.graph.nodes)
    no_aten_conv1d = not any(is_op(n, torch.ops.aten.conv1d.default) for n in gm.graph.nodes)
    return has_custom_op and no_aten_conv1d


@pytest.mark.parametrize("kernel_size", [3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_causal_conv_pattern_match(kernel_size, dtype):
    """Test that single causal conv pattern is matched and replaced."""
    channels = 64
    model = CausalConv1dModel(channels, channels, kernel_size).to("cuda", dtype)
    x = torch.randn(2, 16, channels, device="cuda", dtype=dtype)

    run_test(
        model,
        x,
        _apply_causal_conv_transform,
        _check_graph,
        lambda n: n,
        dynamic_shapes=_DYNAMIC_SHAPES,
    )


@torch.inference_mode()
def test_stacked_causal_conv_pattern_match():
    """Test that multiple causal conv patterns are matched and replaced."""
    channels = 64
    model = StackedCausalConvModel(channels).to("cuda", torch.float16)
    x = torch.randn(2, 16, channels, device="cuda", dtype=torch.float16)

    run_test(
        model,
        x,
        _apply_causal_conv_transform,
        _check_graph,
        lambda n: n,
        dynamic_shapes=_DYNAMIC_SHAPES,
    )
