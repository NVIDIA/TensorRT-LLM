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
"""Tests for MiniMax-H3's gate/up SwiGLU GEMM configuration."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.minimax_h3.kernels import gated_mlp_quack
from tensorrt_llm._utils import get_sm_version


@pytest.mark.skipif(
    not (torch.cuda.is_available() and get_sm_version() in (100, 103)),
    reason="needs an SM100/SM103 GPU",
)
def test_quack_h3_gate_up_swiglu_matches_reference():
    """``gate_up_swiglu_quack_h3`` computes ``silu(x W_gate^T) * (x W_up^T)`` for ``weight = [W_gate; W_up]``.

    4100 rows are 17 row tiles of 256: a 4-row last tile and an odd tile count for the 2x1 cluster. With
    8192 weight rows there are more output tiles than SMs, so persistent CTAs take several tiles each.
    """
    torch.manual_seed(0)
    tokens, hidden, intermediate = 4100, 512, 4096
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn((2 * intermediate, hidden), device="cuda") * hidden**-0.5).to(
        torch.bfloat16
    )
    actual = gated_mlp_quack.gate_up_swiglu_quack_h3(x, weight)
    gate, up = F.linear(x.double(), weight.double()).chunk(2, dim=-1)
    expected = F.silu(gate) * up
    assert actual.shape == (tokens, intermediate) and actual.dtype == torch.bfloat16
    # One BF16 rounding of the FP32-accumulated SwiGLU: within one BF16 ulp of the output scale.
    assert (actual.double() - expected).abs().max() <= expected.abs().max() * 2**-7
