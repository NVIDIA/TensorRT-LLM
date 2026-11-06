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
"""The one-pass logit soft cap is the torch chain, bit for bit.

Its output feeds sampled tokens and rejection sampling's distributions, so the
kernel must not move a single bit. Its inputs are BF16 or FP16: every value of
the input type is checked.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.logit_soft_cap import soft_capped_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Muse Spark Open's output_multiplier and final_logit_softcapping, and a second pair.
CONSTANTS = [(0.17677669529663687, 20.0), (32.0**-0.5, 30.0)]


def torch_chain(logits, multiplier, soft_cap):
    x = logits.float() * multiplier
    return soft_cap * torch.tanh(x / soft_cap)


def assert_bitwise(got, want):
    assert got.dtype == torch.float32 and got.shape == want.shape
    nan = torch.isnan(want)
    assert torch.equal(torch.isnan(got), nan)
    assert torch.equal(got[~nan].view(torch.int32), want[~nan].view(torch.int32))


def every_value(dtype):
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    return bits.view(dtype).reshape(64, 1024).cuda()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("multiplier,soft_cap", CONSTANTS)
def test_every_input_value_matches_the_torch_chain(dtype, multiplier, soft_cap):
    logits = every_value(dtype)
    assert_bitwise(
        soft_capped_logits(logits, multiplier, soft_cap),
        torch_chain(logits, multiplier, soft_cap),
    )


@pytest.mark.parametrize("shape", [(3, 5003), (2, 3, 4097), (1, 202048), (0, 7)])
def test_shapes_and_tails(shape):
    torch.manual_seed(0)
    logits = (torch.randn(shape, device="cuda") * 200).to(torch.bfloat16)
    multiplier, soft_cap = CONSTANTS[0]
    assert_bitwise(
        soft_capped_logits(logits, multiplier, soft_cap),
        torch_chain(logits, multiplier, soft_cap),
    )


def test_fp32_input_in_place():
    torch.manual_seed(0)
    logits = torch.randn(4, 9001, device="cuda") * 200
    want = torch_chain(logits, *CONSTANTS[0])
    out = soft_capped_logits(logits, *CONSTANTS[0], out=logits)
    assert out is logits
    assert_bitwise(logits, want)


def test_non_contiguous_input_takes_the_torch_chain():
    logits = (torch.randn(8, 64, device="cuda") * 50).to(torch.bfloat16)[:, ::2]
    assert_bitwise(soft_capped_logits(logits, *CONSTANTS[0]), torch_chain(logits, *CONSTANTS[0]))


def test_out_must_be_fp32_of_the_logits_shape():
    logits = torch.randn(2, 8, device="cuda").to(torch.bfloat16)
    with pytest.raises(ValueError):
        soft_capped_logits(logits, *CONSTANTS[0], out=logits)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
