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
"""The Triton MXFP8 epilogue must match torch.ops.trtllm.mxfp8_quantize exactly.

A fused producer and the standalone op are used interchangeably across layers,
so anything short of a bitwise match is a silent accuracy change.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.triton_mxfp8_quantize import (
    MXFP8_BLOCK_SIZE,
    mxfp8_quantize_triton,
    swizzled_sf_numel,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _skip_without_op():
    if not hasattr(torch.ops, "trtllm") or not hasattr(torch.ops.trtllm, "mxfp8_quantize"):
        pytest.skip("build predates the MXFP8 quantization ops")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 quantization requires Blackwell")


def _assert_matches_reference(x, chunk=128):
    ref_fp8, ref_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    got_fp8, got_sf = mxfp8_quantize_triton(x, chunk=chunk)

    assert got_sf.numel() == ref_sf.numel() == swizzled_sf_numel(
        x.shape[0], x.shape[1] // MXFP8_BLOCK_SIZE)
    assert torch.equal(got_sf, ref_sf.view(torch.uint8)), "scale factors differ"
    assert torch.equal(got_fp8.view(torch.uint8),
                       ref_fp8.view(torch.uint8)), "quantized values differ"


@pytest.mark.parametrize("m", [1, 4, 31, 128, 129, 256])
@pytest.mark.parametrize("k", [1024, 6144])
def test_matches_reference_on_random_input(m, k):
    """The two production shapes: o_proj at K=1024, qkv at K=6144."""
    _skip_without_op()
    torch.manual_seed(0)
    x = (torch.randn(m, k, dtype=torch.float32, device="cuda") * 4).to(torch.bfloat16)
    _assert_matches_reference(x)


def test_matches_reference_at_the_reciprocal_boundary():
    """amax = 1.75 * 2^k is where rcp.approx(448) and 1/448 part ways.

    The scale is amax * rcp(448), rounded up onto a power of two. At these
    amax values the product sits exactly on a power of two, so a one-ulp
    difference in the reciprocal moves the whole exponent.
    """
    _skip_without_op()
    k = 1024
    exponents = torch.arange(-40, 40, dtype=torch.float32)
    amax = (1.75 * torch.exp2(exponents)).to(torch.bfloat16)
    x = torch.zeros(amax.numel(), k, dtype=torch.bfloat16, device="cuda")
    # One block per row set to its target amax, the rest left at zero, so every
    # row exercises a different exponent through the round-up.
    x[:, ::MXFP8_BLOCK_SIZE] = amax[:, None].cuda()
    _assert_matches_reference(x)


def test_matches_reference_on_saturating_and_zero_blocks():
    """Blocks that clamp at e4m3 max, and all-zero blocks that take scale 0."""
    _skip_without_op()
    k = 1024
    x = torch.zeros(8, k, dtype=torch.bfloat16, device="cuda")
    x[1] = 448.0
    x[2] = -448.0
    x[3] = torch.finfo(torch.bfloat16).max
    x[4] = torch.finfo(torch.bfloat16).tiny
    x[5, :MXFP8_BLOCK_SIZE] = 1.0  # a live block beside zero blocks in one row
    x[6] = 1.0
    x[7] = -1.0
    _assert_matches_reference(x)


@pytest.mark.parametrize("chunk", [32, 64, 128, 256])
def test_result_does_not_depend_on_the_chunk_width(chunk):
    """Each program owns whole blocks, so the split must not be observable."""
    _skip_without_op()
    torch.manual_seed(0)
    x = (torch.randn(9, 1024, dtype=torch.float32, device="cuda") * 4).to(torch.bfloat16)
    _assert_matches_reference(x, chunk=chunk)
