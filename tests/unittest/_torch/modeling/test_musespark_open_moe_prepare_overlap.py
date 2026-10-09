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
"""Exercise cross-stream preparation and CUDA graph input updates."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenMoE,
    MuseSparkOpenRoutingMethod,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 2, 8, 16, 80, 128, 129])
def test_prepare_overlap_matches_serial_and_graph_replay(tokens):
    torch.manual_seed(923)
    with torch.inference_mode():
        mod = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
        torch.nn.Module.__init__(mod)
        mod._prepare_stream = torch.cuda.Stream()
        mod._pool_gather_workspace = None
        mod.nvtx = lambda name: contextlib.nullcontext()
        router_weight = torch.randn(256, 256, device="cuda")
        bias = torch.randn(256, device="cuda")
        routing = MuseSparkOpenRoutingMethod(
            top_k=8, num_experts=256, callable_e_score_correction_bias=lambda: bias, renorm_eps=1e-6
        )
        mod.experts = SimpleNamespace(comm=None, routing_method=routing)
        mod.router_logits = lambda x: x.float() @ router_weight
        mod.pooler = torch.nn.Linear(256, 128, bias=False, dtype=torch.bfloat16, device="cuda")
        mod.prepare_expert_input = lambda x: x * 0.5
        x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
        expected = (mod.router_logits(x), mod.prepare_expert_input(mod.pooler(x)))
        got = mod.prepare_moe_inputs(x)
        torch.cuda.synchronize()
        for a, b in zip(got, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = mod.prepare_moe_inputs(x)
        for _ in range(3):
            x.normal_()
            expected = (mod.router_logits(x), mod.prepare_expert_input(mod.pooler(x)))
            graph.replay()
            torch.cuda.synchronize()
            for a, b in zip(got, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
