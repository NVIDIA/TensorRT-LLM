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
"""The Muse Spark Open head returns the torch chain's soft-capped FP32 logits, bit for bit."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenLogitsProcessor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Head(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return x.to(self.weight.dtype) @ self.weight.T


def test_processor_returns_the_torch_chain_bitwise():
    torch.manual_seed(0)
    weight = torch.randn(4099, 16, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(5, 16, dtype=torch.bfloat16, device="cuda") * 20
    multiplier, soft_cap = 0.17677669529663687, 20.0
    processor = MuseSparkOpenLogitsProcessor(logit_multiplier=multiplier, soft_cap=soft_cap)

    out = processor(x, _Head(weight), None, return_context_logits=True)

    want = soft_cap * torch.tanh((x @ weight.T).float() * multiplier / soft_cap)
    assert out.dtype == torch.float32
    assert torch.equal(out.view(torch.int32), want.view(torch.int32))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
