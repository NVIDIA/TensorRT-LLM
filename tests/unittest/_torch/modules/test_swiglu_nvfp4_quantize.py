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
"""Tests for trtllm::swiglu_nvfp4_quantize against fp4_quantize(swiglu(x))."""

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.utils import ceil_div, pad_up, unswizzle_sf

SF_VEC = 16
FP8_MAX, E2M1_MAX = 448.0, 6.0
_FP4_MATCH_THRESHOLD = 0.97


def swiglu_nvfp4_quantize_available():
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "swiglu_nvfp4_quantize")


pytestmark = pytest.mark.skipif(
    getSMVersion() < 100 or not swiglu_nvfp4_quantize_available(),
    reason="Requires SM100+ and the trtllm swiglu_nvfp4_quantize op",
)


def _unswizzled_sf(sf: torch.Tensor, m: int, n: int) -> torch.Tensor:
    padded_rows = pad_up(m, 128)
    num_sf_cols = ceil_div(n, SF_VEC)
    padded_cols = pad_up(num_sf_cols, 4) * SF_VEC
    return unswizzle_sf(sf, padded_rows, padded_cols, SF_VEC)[:m, :num_sf_cols]


@pytest.mark.parametrize("m", [1, 10, 50, 128, 4096])
@pytest.mark.parametrize("interm_size", [512, 9216])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_swiglu_nvfp4_quantize_vs_separate(m, interm_size, dtype):
    torch.manual_seed(0)
    x = torch.randn(m, 2 * interm_size, dtype=dtype, device="cuda") * 2
    h = swiglu(x)
    sf_scale = (FP8_MAX * E2M1_MAX / h.float().abs().amax()).view(1)
    fp4_ref, sf_ref = torch.ops.trtllm.fp4_quantize(h, sf_scale, SF_VEC, False)

    fp4, sf = torch.ops.trtllm.swiglu_nvfp4_quantize(x, sf_scale)

    ctx = f"m={m} interm_size={interm_size} {dtype}"
    fp4 = fp4.view(torch.uint8)
    fp4_ref = fp4_ref.view(torch.uint8)
    assert fp4.shape == fp4_ref.shape == (m, interm_size // 2), ctx
    match_rate = (fp4 == fp4_ref).float().mean().item()
    assert match_rate >= _FP4_MATCH_THRESHOLD, f"FP4 match rate {match_rate:.4f} ({ctx})"

    assert sf.numel() == sf_ref.numel(), ctx
    u_sf = _unswizzled_sf(sf, m, interm_size).to(torch.int16)
    u_ref = _unswizzled_sf(sf_ref, m, interm_size).to(torch.int16)
    max_diff = int((u_sf - u_ref).abs().max().item())
    assert max_diff <= 1, f"SF differs by {max_diff} E4M3 steps ({ctx})"


def test_swiglu_nvfp4_quantize_gate_first_layout():
    """Input layout is [gate | up]."""
    torch.manual_seed(0)
    m, interm_size = 16, 512
    gate = torch.randn(m, interm_size, dtype=torch.bfloat16, device="cuda") * 2
    up = torch.randn(m, interm_size, dtype=torch.bfloat16, device="cuda") * 2
    sf_scale = torch.tensor([1.0], device="cuda")

    fp4, _ = torch.ops.trtllm.swiglu_nvfp4_quantize(torch.cat([gate, up], dim=1), sf_scale)
    fp4_swapped, _ = torch.ops.trtllm.swiglu_nvfp4_quantize(torch.cat([up, gate], dim=1), sf_scale)
    fp4_ref, _ = torch.ops.trtllm.fp4_quantize(
        torch.nn.functional.silu(gate.float()).mul(up.float()).to(torch.bfloat16),
        sf_scale,
        SF_VEC,
        False,
    )

    fp4, fp4_swapped = fp4.view(torch.uint8), fp4_swapped.view(torch.uint8)
    assert (fp4 == fp4_ref.view(torch.uint8)).float().mean().item() >= _FP4_MATCH_THRESHOLD
    assert (fp4 == fp4_swapped).float().mean().item() < 0.5


def test_swiglu_nvfp4_quantize_zero_tokens():
    x = torch.empty(0, 1024, dtype=torch.bfloat16, device="cuda")
    fp4, sf = torch.ops.trtllm.swiglu_nvfp4_quantize(x, torch.tensor([1.0], device="cuda"))
    assert fp4.shape == (0, 256)
    assert sf.numel() == 0
