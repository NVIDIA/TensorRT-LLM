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

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul import gelu_tanh_mul


# 256 is the Gemma4-E4B per-layer-input width and 42 its layer count; up is a row slice of the
# [tokens, layers, width] per-layer-input tensor, as in the model.
@pytest.mark.parametrize("n_tokens", [1, 333, 4096])
@pytest.mark.parametrize("width", [256, 1000])
def test_gelu_tanh_mul_parity(n_tokens, width):
    torch.manual_seed(3)
    gate = torch.randn((n_tokens, width), dtype=torch.bfloat16, device="cuda") * 3
    up = torch.randn((n_tokens, 42, width), dtype=torch.bfloat16, device="cuda")[:, 7, :]
    fused = gelu_tanh_mul(gate, up)
    assert fused.shape == (n_tokens, width) and fused.is_contiguous()
    ref = F.gelu(gate, approximate="tanh") * up
    mismatches = (fused != ref).sum().item()
    # The same fp32 math and roundings as PyTorch; allow a rare one-ulp tanh difference.
    assert mismatches <= max(1, ref.numel() // 10000), mismatches
    torch.testing.assert_close(fused, ref, atol=1e-2, rtol=1e-2)
