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
"""vup_gate (the MLA V up-projection with the K3 sigmoid gate in one kernel) and its K3.1 route."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import KimiK3MLAAttention

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _separate(attn, w, gate):
    """The unfused path: the BF16 BMM of MLA.forward_absorption_generation, then sigmoid_gate_mul."""
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate import sigmoid_gate_mul

    rows, heads = attn.shape[0], w.shape[0]
    out = torch.empty(rows, heads, w.shape[1], dtype=torch.bfloat16, device=attn.device)
    torch.bmm(
        attn.view(rows, heads, -1).transpose(0, 1), w.transpose(1, 2), out=out.transpose(0, 1)
    )
    out = out.view(rows, -1)
    return out if gate is None else sigmoid_gate_mul(out, gate, min_tokens=1)


@skip_no_cuda
@pytest.mark.parametrize("rows,heads", [(1, 6), (6, 6), (24, 6), (24, 12), (32, 16)])
@pytest.mark.parametrize("with_gate", [True])
def test_vup_gate_matches_separate_kernels(rows, heads, with_gate):
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate import sigmoid_gate_mul
    from tensorrt_llm._torch.modules.kimi_k3_mla.vup_gate import vup_gate

    device = torch.device("cuda")
    torch.manual_seed(rows * 7 + heads)
    attn = torch.randn(rows, heads * 512, device=device).to(torch.bfloat16)
    w = (torch.randn(heads, 128, 512, device=device) * 0.05).to(torch.bfloat16)
    # The gate is a column slice of the wider packed projection output.
    wide = torch.empty(rows, 3 * heads * 128, device=device).uniform_(-6, 6).to(torch.bfloat16)
    gate = wide[:, heads * 128 : 2 * heads * 128] if with_gate else None
    out = vup_gate(attn, w, gate)
    expect = _separate(attn, w, gate)
    # FP32 accumulation in a possibly different order: within one BF16 rounding of the result.
    diff = (out.float() - expect.float()).abs()
    assert (diff <= expect.float().abs() * 2**-7 + 1e-6).all(), diff.max()
    if with_gate:
        # The epilogue is sigmoid_gate_mul's arithmetic on the rounded projection.
        torch.testing.assert_close(
            out, sigmoid_gate_mul(vup_gate(attn, w, None), gate, min_tokens=1), rtol=0, atol=0
        )
    # Writes into a given buffer.
    buf = torch.empty_like(out)
    assert vup_gate(attn, w, gate, out=buf) is buf
    torch.testing.assert_close(buf, out, rtol=0, atol=0)


def _attention(rows, heads, *, gate=True, dtype=torch.bfloat16):
    hidden = torch.zeros(rows, 64, dtype=torch.bfloat16, device="cuda")
    packed = torch.empty(rows, 40 + heads * 128, device="cuda").uniform_(-4, 4).to(torch.bfloat16)
    return SimpleNamespace(
        num_heads_tp_cp=heads,
        v_head_dim=128,
        use_output_gate=gate,
        v_b_proj=(torch.randn(heads, 128, 512, device="cuda") * 0.05).to(dtype),
        dsa_packed_gate_out=(hidden, packed[:, 40:]) if gate else None,
        _k31_vup_gate_max_rows=32,
    ), hidden


@skip_no_cuda
def test_k31_route_fuses_the_packed_gate():
    rows, heads = 6, 6
    attn, hidden = _attention(rows, heads)
    gate = attn.dsa_packed_gate_out[1]
    latent = torch.randn(rows, heads * 512, device="cuda").to(torch.bfloat16)
    output = torch.empty(rows, heads * 128, dtype=torch.bfloat16, device="cuda")

    from tensorrt_llm._torch.modules.kimi_k3_mla.vup_gate import vup_gate

    assert KimiK3MLAAttention._absorption_v_up_override(attn, latent, output)
    assert attn.dsa_packed_gate_out is None and attn._output_gate_fused
    torch.testing.assert_close(output, vup_gate(latent, attn.v_b_proj, gate), atol=0, rtol=0)
    # _apply_output_gate passes the gated output through once, then gates again.
    assert KimiK3MLAAttention._apply_output_gate(attn, hidden, output) is output
    assert not attn._output_gate_fused


@pytest.mark.parametrize(
    "case", ["off", "too_many_rows", "no_packed_gate", "fp8_v_b_proj", "no_output_gate"]
)
@skip_no_cuda
def test_k31_route_declines(case):
    rows = 24 if case == "too_many_rows" else 6
    attn, _ = _attention(
        rows,
        6,
        gate=case not in ("no_packed_gate", "no_output_gate"),
        dtype=torch.float8_e4m3fn if case == "fp8_v_b_proj" else torch.bfloat16,
    )
    attn._k31_vup_gate_max_rows = 0 if case == "off" else 16
    if case == "no_output_gate":
        attn.use_output_gate = False
    latent = torch.randn(rows, 6 * 512, device="cuda").to(torch.bfloat16)
    output = torch.zeros(rows, 6 * 128, dtype=torch.bfloat16, device="cuda")
    packed = attn.dsa_packed_gate_out
    assert not KimiK3MLAAttention._absorption_v_up_override(attn, latent, output)
    assert attn.dsa_packed_gate_out is packed
    assert not getattr(attn, "_output_gate_fused", False)
    assert torch.count_nonzero(output) == 0
