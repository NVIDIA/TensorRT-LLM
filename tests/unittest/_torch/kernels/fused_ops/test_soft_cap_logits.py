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

from tensorrt_llm._torch.kernels.fused_ops.soft_cap_logits import soft_cap_logits


def _reference(logits, multiplier, soft_cap):
    x = logits.float()
    if multiplier != 1.0:
        x = x * multiplier
    if soft_cap is not None:
        x = torch.tanh(x / soft_cap) * soft_cap
    return x


# 262144 is the Gemma4 vocabulary and 30.0 its final-logit soft cap; the 1000-wide case is a row
# slice of a wider buffer.
@pytest.mark.parametrize("rows", [1, 32])
@pytest.mark.parametrize("vocab", [262144, 1000])
@pytest.mark.parametrize("multiplier,soft_cap", [(1.0, 30.0), (0.5, None)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_soft_cap_logits_bitwise(rows, vocab, multiplier, soft_cap, dtype):
    torch.manual_seed(0)
    width = vocab if vocab > 1000 else vocab + 548
    logits = (torch.randn((rows, width), device="cuda") * 40).to(dtype)[:, :vocab]
    out = soft_cap_logits(logits, multiplier, soft_cap)
    ref = _reference(logits, multiplier, soft_cap)
    assert out.dtype == torch.float32 and out.shape == (rows, vocab) and out.is_contiguous()
    assert torch.equal(out, ref)
