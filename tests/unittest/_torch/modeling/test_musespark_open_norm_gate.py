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
"""Validate attention norm/gate rounding and strided gate addressing."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention
from tensorrt_llm._torch.modules.rms_norm import RMSNorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 2, 16, 128])
@pytest.mark.parametrize("strided", [False, True])
def test_norm_gate_and_graph_replay(tokens, strided):
    torch.manual_seed(950)
    with torch.inference_mode():
        mod = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
        torch.nn.Module.__init__(mod)
        mod.head_dim = 64
        mod._fuse_qk_norm_rope_gate = False
        mod.o_head_norm = RMSNorm(
            hidden_size=64, eps=1e-5, dtype=torch.bfloat16, has_weights=False
        ).cuda()
        x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
        gate = (
            torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)[:, :, 64:]
            if strided
            else torch.randn_like(x)
        )
        normed = mod.o_head_norm(x.reshape(-1, 64)).reshape_as(x)
        expected = normed * gate.reshape_as(x).sigmoid()
        got = mod.apply_output_gate(x, gate)
        torch.testing.assert_close(got, expected, rtol=0.016, atol=0.002)
        assert (got.float() - expected.float()).norm() / expected.float().norm() < 0.003
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = mod.apply_output_gate(x, gate)
        x.normal_()
        gate.normal_()
        expected = mod.apply_output_gate(x, gate)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
