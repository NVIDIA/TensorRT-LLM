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
"""FP8 absorbed query of a NoPE MLA decode step (``mla_fp8_q_absorb_nope``).

The q_pe half must be bitwise the FP8 conversion of q_pe (what the MLA RoPE
kernel writes for an identity rotation); the absorbed half is the FP32 GEMM
rounded once to FP8, so it may differ from the BF16-BMM-then-FP8 path by one
FP8 step where the BF16 rounding crossed an FP8 rounding boundary.
"""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 9,
        reason="FP8 E4M3 conversion needs sm89+",
    ),
]

H, NOPE, ROPE, LORA = 96, 128, 64, 512


def _fp8_step(x: torch.Tensor) -> torch.Tensor:
    """Spacing of FP8 E4M3 values at |x| (3 mantissa bits, min normal 2**-6)."""
    e = torch.floor(torch.log2(x.abs().clamp_min(2.0**-6)))
    return torch.exp2(e - 3)


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 65, 224, 896])
def test_fp8_q_absorb_nope(num_tokens):
    from tensorrt_llm._torch.modules.mla_fp8_q_absorb import mla_fp8_q_absorb_nope

    torch.manual_seed(num_tokens)
    q = torch.randn(num_tokens, H, NOPE + ROPE, device="cuda", dtype=torch.bfloat16) * 4
    w = (torch.randn(H, LORA, NOPE, device="cuda") * 0.3).to(torch.bfloat16)
    out = torch.full((num_tokens, H, LORA + ROPE), 0x7F, dtype=torch.uint8, device="cuda")
    mla_fp8_q_absorb_nope(q, w, out, NOPE, ROPE)
    got = out.view(torch.float8_e4m3fn).float()

    acc = torch.einsum("thk,hnk->thn", q[..., :NOPE].float(), w.float())
    ref = acc.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    # FP32 accumulation order differs from the reference: at most one FP8 step
    diff = (got[..., :LORA] - ref).abs()
    assert (diff <= _fp8_step(ref) * 1.0001).all()
    assert (diff > 0).float().mean().item() < 0.01
    # saturating conversion: no NaN even past the FP8 range
    assert not torch.isnan(got).any()
    pe = q[..., NOPE:].float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
    assert torch.equal(got[..., LORA:], pe)


def test_fp8_q_absorb_nope_strided_q():
    """q as the head-split view of a wider row (the q_b_proj output slice)."""
    from tensorrt_llm._torch.modules.mla_fp8_q_absorb import mla_fp8_q_absorb_nope

    torch.manual_seed(0)
    T = 40
    base = torch.randn(T + 3, H * (NOPE + ROPE), device="cuda", dtype=torch.bfloat16)
    q = base[3:].view(T, H, NOPE + ROPE)
    w = (torch.randn(H, LORA, NOPE, device="cuda") * 0.1).to(torch.bfloat16)
    out = torch.empty(T, H, LORA + ROPE, dtype=torch.float8_e4m3fn, device="cuda")
    mla_fp8_q_absorb_nope(q, w, out, NOPE, ROPE)
    acc = torch.einsum("thk,hnk->thn", q[..., :NOPE].float(), w.float())
    ref = acc.to(torch.float8_e4m3fn).float()
    assert ((out.float()[..., :LORA] - ref).abs() <= _fp8_step(ref) * 1.0001).all()
    assert torch.equal(
        out.float()[..., LORA:], q[..., NOPE:].float().to(torch.float8_e4m3fn).float()
    )


def test_fp8_q_absorb_nope_saturates():
    """Values past the FP8 E4M3 range store +-448, never NaN."""
    from tensorrt_llm._torch.modules.mla_fp8_q_absorb import mla_fp8_q_absorb_nope

    torch.manual_seed(1)
    T = 16
    q = torch.randn(T, H, NOPE + ROPE, device="cuda", dtype=torch.bfloat16) * 200
    w = torch.randn(H, LORA, NOPE, device="cuda").to(torch.bfloat16)
    out = torch.empty(T, H, LORA + ROPE, dtype=torch.float8_e4m3fn, device="cuda")
    mla_fp8_q_absorb_nope(q, w, out, NOPE, ROPE)
    got = out.float()
    assert not torch.isnan(got).any()
    assert got.abs().max().item() == 448.0
    acc = torch.einsum("thk,hnk->thn", q[..., :NOPE].float(), w.float())
    ref = acc.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    big = acc.abs() > 464
    assert torch.equal(got[..., :LORA][big], ref[big])
