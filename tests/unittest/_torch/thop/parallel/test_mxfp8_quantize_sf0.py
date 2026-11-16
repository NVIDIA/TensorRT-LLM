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
"""trtllm::mxfp8_quantize on 32-element blocks whose UE8M0 scale byte is 0 (0 < amax <= 448 * 2^-127).

The byte 0 is 2^-127, an fp32 denormal that rcp.approx.ftz returns +inf for: such a block used to quantize its zeros
to NaN and its other values to +-448, and the GEMM row went NaN. It now quantizes to zero, each value's sign kept,
like an all-zero block. A block with a scale byte > 0 quantizes as before: here checked as power-of-two equivariance
over 2^-12..2^12, which pins the scale and the codes together.
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops.torch_custom_ops  # noqa: F401
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100,
    reason="MXFP8 quantization needs an SM100+ GPU",
)

K = 8192
ROWS = 128
BLOCK = 176
SCALE_BYTE_0_AMAX = (1e-40, 1e-38, 2e-36)


def _quantize(x):
    q, sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    return q.view(torch.uint8), sf.view(torch.uint8)


def _scales(sf, rows, k):
    """The swizzled 128x4 scale bytes as a [rows, k / 32] matrix."""
    r = torch.arange(rows, device=sf.device)[:, None]
    b = torch.arange(k // 32, device=sf.device)[None, :]
    tiles_k = (k // 32 + 3) // 4
    offset = (
        (r // 128) * (tiles_k * 512)
        + (b // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + b % 4
    )
    return sf[offset]


def _rows(amax, zeros, seed=0):
    """ROWS rows of normal values whose block BLOCK holds values of magnitude <= amax (one of them exactly +-amax)
    and, with ``zeros``, some exact zeros."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((ROWS, K), generator=gen, device="cuda")
    block = (torch.rand((ROWS, 32), generator=gen, device="cuda") * 2 - 1) * amax
    block[:, 7] = amax * torch.where(
        torch.rand(ROWS, generator=gen, device="cuda") < 0.5, -1.0, 1.0
    )
    if zeros:
        block[:, 0::5] = 0.0
    x[:, BLOCK * 32 : (BLOCK + 1) * 32] = block
    return x.to(torch.bfloat16)


def _block(t):
    return t[:, BLOCK * 32 : (BLOCK + 1) * 32]


def _no_nan_codes(q):
    return not bool(((q & 0x7F) == 0x7F).any())


@pytest.mark.parametrize("zeros", [False, True])
@pytest.mark.parametrize("amax", (0.0,) + SCALE_BYTE_0_AMAX)
def test_a_block_with_scale_byte_0_quantizes_to_zero(amax, zeros):
    x = _rows(amax, zeros)
    q, sf = _quantize(x)
    assert _no_nan_codes(q)
    assert bool((_scales(sf, ROWS, K)[:, BLOCK] == 0).all())
    signed_zero = torch.where(torch.signbit(_block(x)), 0x80, 0x00).to(torch.uint8)
    assert torch.equal(_block(q), signed_zero)
    # Blocks are independent: every other block is what it is with this one all zero.
    clean = x.clone()
    _block(clean).zero_()
    q_clean, sf_clean = _quantize(clean)
    keep = torch.ones(K, dtype=torch.bool, device="cuda")
    keep[BLOCK * 32 : (BLOCK + 1) * 32] = False
    assert torch.equal(q[:, keep], q_clean[:, keep])
    assert torch.equal(sf, sf_clean)


def test_the_smallest_nonzero_scale_byte_still_quantizes_the_block():
    """amax 2.7e-36 > 448 * 2^-127 takes scale byte 1, so its block keeps codes: those of the same block at 2^100,
    with the scale byte 100 higher."""
    x = _rows(2.7e-36, zeros=True)
    q, sf = _quantize(x)
    q_up, sf_up = _quantize(x * 2.0**100)
    assert _no_nan_codes(q)
    assert bool((_scales(sf, ROWS, K)[:, BLOCK] == 1).all())
    assert bool((_block(q) != 0).any())
    assert torch.equal(_block(q), _block(q_up))
    assert torch.equal(
        _scales(sf_up, ROWS, K)[:, BLOCK].int(), _scales(sf, ROWS, K)[:, BLOCK].int() + 100
    )


@pytest.mark.parametrize("shift", [-12, -5, -1, 1, 5, 12])
def test_ordinary_scales_are_power_of_two_equivariant(shift):
    gen = torch.Generator(device="cuda").manual_seed(shift + 100)
    x = (torch.randn((ROWS, K), generator=gen, device="cuda") * 3).to(torch.bfloat16)
    q, sf = _quantize(x)
    q_shift, sf_shift = _quantize(x * 2.0**shift)
    assert torch.equal(q, q_shift)
    assert torch.equal(_scales(sf_shift, ROWS, K).int(), _scales(sf, ROWS, K).int() + shift)


def test_a_gemm_row_with_a_scale_byte_0_block_is_finite():
    """The served case (an o_proj input row with one block of ~3.5e-39 values and a few exact zeros): the row's GEMM
    output is finite, and bitwise that of the row with the block zeroed."""
    x = _rows(3.49e-39, zeros=True)
    clean = x.clone()
    _block(clean).zero_()
    gen = torch.Generator(device="cuda").manual_seed(7)
    w_q, w_sf = torch.ops.trtllm.mxfp8_quantize(
        torch.randn((256, K), generator=gen, device="cuda").to(torch.bfloat16), True
    )
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    outs = []
    for rows in (x, clean):
        a_q, a_sf = torch.ops.trtllm.mxfp8_quantize(rows, True)
        outs.append(torch.ops.trtllm.mxfp8_mxfp8_gemm(a_q, a_sf, w_q, w_sf, alpha, torch.bfloat16))
    assert bool(torch.isfinite(outs[0]).all())
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
