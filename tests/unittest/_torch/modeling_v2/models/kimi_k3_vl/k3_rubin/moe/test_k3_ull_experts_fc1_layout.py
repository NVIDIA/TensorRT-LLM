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
"""FC1 row layouts of the Kimi-K3 ULL routed experts kernel (k3_moe_ull).

The same NVFP4 experts are laid out with ``[up64, gate64]`` row groups
(``interleave_linear_and_gate`` group 64) and with ``[gate16, up16]`` row groups
(``interleave_gate_and_linear`` group 16, the order ``NVFP4CuteDslFusedMoEMethod`` gives on SM107),
weights and block scales alike. The kernel told the
matching layout must give the same output bits for both (slice combine); told the other layout it
must not; and both stay close to a dequantized torch reference (unquantized intermediate). The
backend probe (``fc1_gate_up16``) must report the order the quantization method applies.
"""

import math
import os
import types

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4CuteDslFusedMoEMethod,
    interleave_gate_and_linear,
    interleave_linear_and_gate,
)
from tensorrt_llm._torch.utils import swizzle_sf
from tensorrt_llm._utils import get_sm_version

HIDDEN = 3584
TOP_K = 16
SF_VEC = 16
SITU_BETA = 2.5
SITU_LINEAR_BETA = 0.7
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="SM100 / SM103 / SM107 kernel",
)


def _ull():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.moe import (
        k3_ull_experts,
    )

    return k3_ull_experts


def _dequant_fp4(codes: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """[..., cols / 2] e2m1 pairs (low nibble first) x [..., cols / 16] E4M3 -> FP32 [..., cols]."""
    lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), dtype=torch.float32, device=codes.device)
    vals = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], dim=-1)
    vals = vals.flatten(-2)
    scales = sf.view(torch.float8_e4m3fn).float().repeat_interleave(SF_VEC, dim=-1)
    return vals * scales


def _experts(num_local, rows, cols, gen):
    w = torch.randint(
        0, 256, (num_local, rows, cols // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    sf = (
        (torch.rand((num_local, rows, cols // SF_VEC), device="cuda", generator=gen) * 1.5 + 0.5)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return w, sf


def _kernel_operands(w, sf, rows, cols, interleave):
    """Row-interleaved weights and 128x4-swizzled block scales, as the quantization method stores
    them (``interleave`` None: rows as given)."""
    if interleave is not None:
        w = interleave(w, dim=1).contiguous()
        sf = interleave(sf, dim=1).contiguous()
    sf_sw = swizzle_sf(sf, rows, cols).view(sf.shape)
    return w, sf_sw


def _situ(gate, up):
    return (
        SITU_BETA
        * torch.tanh(gate / SITU_BETA)
        * torch.sigmoid(gate)
        * SITU_LINEAR_BETA
        * torch.tanh(up / SITU_LINEAR_BETA)
    )


def _rel_l2(a, b):
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-30)


@pytest.mark.parametrize(
    "inter,num_local,num_experts,num_tokens,moetp",
    [
        (768, 16, 64, 8, "0"),  # MoE-TP slice, permuted layout
        (768, 16, 64, 32, "1"),  # MoE-TP slice, token-indexed layout
        (3072, 8, 32, 16, "0"),  # EP expert
    ],
)
def test_ull_experts_fc1_layouts(
    monkeypatch, k3_features, inter, num_local, num_experts, num_tokens, moetp
):
    monkeypatch.setenv("TRTLLM_K3_ULL_EXPERTS", "1")
    k3_features(ull_moetp=moetp == "1")
    ull = _ull()
    monkeypatch.setattr(ull, "_COMBINE_RED", False)
    ops = torch.ops.trtllm
    gen = torch.Generator(device="cuda").manual_seed(1234 + inter + num_tokens)

    logits = torch.randn(num_tokens, num_experts, device="cuda", generator=gen)
    scales, ids = logits.topk(TOP_K, dim=-1)
    ids = ids.to(torch.int32)
    scales = scales.softmax(dim=-1).float()
    xb = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen)
    x_gsf = xb.abs().max().float() / (448 * 6)
    x, x_sf = ops.fp4_quantize(xb, 1 / x_gsf, SF_VEC, False, False)
    x_sf = x_sf.view(num_tokens, HIDDEN // SF_VEC)

    # logical W1 rows [up | gate] (w3 then w1), W2 [HIDDEN, inter]
    w1, w1_sf = _experts(num_local, 2 * inter, HIDDEN, gen)
    w2, w2_sf = _experts(num_local, HIDDEN, inter, gen)
    alpha1 = (torch.linspace(0.5, 1.5, num_local, device="cuda") * x_gsf * 0.004).float()
    global_sf = torch.tensor([448 * 6 / 64.0], dtype=torch.float32, device="cuda")
    alpha2 = (torch.linspace(0.5, 1.5, num_local, device="cuda") * 0.003 / global_sf[0]).float()
    t2g, t2lim, e2p, p2e, _, nnet = ops.moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=num_experts,
        top_k=TOP_K,
        local_expert_offset=0,
        local_num_experts=num_local,
        tile_tokens_dim=128,
    )
    w2_k, w2_sf_k = _kernel_operands(w2, w2_sf, HIDDEN, inter, None)
    counters = ull.K3UllExpertsRunner.new_counters(torch.device("cuda"))

    def run(interleave, gate_up16):
        w1_k, w1_sf_k = _kernel_operands(w1, w1_sf, 2 * inter, HIDDEN, interleave)
        out = torch.empty(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ops.k3_ull_experts_nvfp4(
            input=x.view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            weight1=w1_k.view(torch.float4_e2m1fn_x2),
            weight1_scale=w1_sf_k,
            alpha1=alpha1,
            weight2=w2_k.view(torch.float4_e2m1fn_x2),
            weight2_scale=w2_sf_k,
            alpha2=alpha2,
            output=out,
            tile_idx_to_group_idx=t2g,
            tile_idx_to_mn_limit=t2lim,
            permuted_idx_to_expanded_idx=p2e,
            num_non_exiting_tiles=nnet,
            global_sf=global_sf,
            token_final_scales=scales,
            counters=counters,
            top_k=TOP_K,
            situ_beta=SITU_BETA,
            situ_linear_beta=SITU_LINEAR_BETA,
            expanded_idx_to_permuted_idx=e2p,
            gate_up16=gate_up16,
        )
        torch.cuda.synchronize()
        assert int(counters.abs().sum().item()) == 0
        return out

    def up64(t, dim):
        return interleave_linear_and_gate(t, group_size=64, dim=dim)

    def gate16(t, dim):
        return interleave_gate_and_linear(t, group_size=16, dim=dim)

    out_up64 = run(up64, False)
    out_gate16 = run(gate16, True)
    out_wrong = run(gate16, False)

    # dequantized reference over this rank's experts (intermediate kept in FP32)
    xd = _dequant_fp4(x.view(torch.uint8), x_sf.view(torch.uint8)) * x_gsf
    ref = torch.zeros(num_tokens, HIDDEN, dtype=torch.float32, device="cuda")
    for e in range(num_local):
        hit = ids == e
        tok = hit.any(dim=-1).nonzero().flatten()
        if tok.numel() == 0:
            continue
        pre = (xd[tok] @ _dequant_fp4(w1[e], w1_sf[e]).t()) * (alpha1[e] / x_gsf)
        h = _situ(pre[:, inter:], pre[:, :inter])
        y = (h @ _dequant_fp4(w2[e], w2_sf[e]).t()) * (alpha2[e] * global_sf[0])
        ref[tok] += y * (scales * hit)[tok].sum(dim=-1, keepdim=True)

    rel = {
        "gate16~up64": _rel_l2(out_gate16, out_up64),
        "wrong~up64": _rel_l2(out_wrong, out_up64),
        "up64~ref": _rel_l2(out_up64, ref),
        "gate16~ref": _rel_l2(out_gate16, ref),
    }
    print(f"inter {inter} tokens {num_tokens} moetp {moetp}: |ref| {ref.norm().item():.4g} {rel}")
    assert math.isfinite(ref.norm().item()) and ref.norm().item() > 0
    assert torch.equal(out_gate16.view(torch.int16), out_up64.view(torch.int16)), rel
    assert rel["wrong~up64"] > 0.5, rel
    assert rel["up64~ref"] < 0.2, rel
    assert rel["gate16~ref"] < 0.2, rel


def test_fc1_gate_up16_probe():
    ull = _ull()
    w1 = torch.empty(4, 1536, 1792, dtype=torch.uint8, device="meta")

    backend = types.SimpleNamespace(quant_method=NVFP4CuteDslFusedMoEMethod(), w3_w1_weight=w1)
    assert ull.fc1_gate_up16(backend) is (get_sm_version() == 107)
    # cached on the backend
    assert backend._ull_fc1_gate_up16 is (get_sm_version() == 107)

    class _RowsAsGiven:
        @staticmethod
        def _interleave_fc1_rows(t):
            return t

    for quant_method in (_RowsAsGiven(), object()):
        stub = types.SimpleNamespace(quant_method=quant_method, w3_w1_weight=w1)
        assert ull.fc1_gate_up16(stub) is None


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v", "-s"])
