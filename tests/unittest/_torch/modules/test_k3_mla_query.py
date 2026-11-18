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
"""The fused MLA decode query (``TRTLLM_K3_MLA_FUSED_Q``) against the stock chain on SM100/103.

Stock chain: q_a RMSNorm (FlashInfer CuTe), ``fp8_swap_ab_gemm`` (1x128 quantize + DeepGEMM) for q_b, the Triton FP8
q_absorb. The fused kernel differs only by the FP32 summation order of the two GEMMs: a handful of FP8 codes may
differ by one step. The kv_a norm + concat kernel is bitwise equal.
"""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear as Fp8Linear
from tensorrt_llm._torch.modules.kimi_k3_mla import k3_mla_query as kq
from tensorrt_llm._torch.modules.mla_fp8_q_absorb import mla_fp8_q_absorb_nope
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="needs SM100/103",
)

H, QL, KVL, NOPE, ROPE = kq.NUM_HEADS, kq.Q_LORA, kq.LATENT, kq.NOPE, kq.ROPE


def _fixture(seed):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    w_qb = Fp8Linear.quantize_weight((torch.randn(H * (NOPE + ROPE), QL, device=dev) * 0.02).bfloat16())
    q_ln = RMSNorm(hidden_size=QL, eps=1e-5, dtype=torch.bfloat16).to(dev)
    kv_ln = RMSNorm(hidden_size=KVL, eps=1e-5, dtype=torch.bfloat16).to(dev)
    q_ln.weight.data.normal_(1.0, 0.1)
    kv_ln.weight.data.normal_(1.0, 0.1)
    w_kc = (torch.randn(H, KVL, NOPE, device=dev) * 0.05).bfloat16()
    fq = kq.K3MlaQuery(w_qb[0], w_qb[1], q_ln.weight, 1e-5, w_kc, kv_ln.weight, 1e-5)
    return dev, w_qb, q_ln, kv_ln, w_kc, fq


@pytest.mark.parametrize("rows", [1, 2, 6, 8, 9, 12, 16, 17, 24, 30, 32])
@pytest.mark.parametrize("seed", [0, 1])
def test_fused_query_matches_stock_chain(rows, seed):
    dev, w_qb, q_ln, kv_ln, w_kc, fq = _fixture(seed)
    # the fused projection's output rows: [g | q_a | kv_a | k_pe], the layer slices the last 2112 columns
    out = (torch.randn(rows, 3648, device=dev) * (1 + 3 * (seed == 1))).bfloat16()
    fused_a = out[:, QL:]
    q_a, kv_a = fused_a[:, :QL], fused_a[:, QL:]
    with torch.no_grad():
        qn = q_ln(q_a)
        q = torch.ops.trtllm.fp8_swap_ab_gemm(qn, w_qb[0], w_qb[1], output_dtype=torch.bfloat16, disable_ue8m0_cast=True)
        ref = torch.empty(rows, H, KVL + ROPE, dtype=torch.uint8, device=dev)
        mla_fp8_q_absorb_nope(q.view(-1, H, NOPE + ROPE), w_kc, ref, NOPE, ROPE)
        got = torch.zeros_like(ref)
        fq(q_a, got)
        latent = torch.empty(rows, KVL + ROPE, dtype=torch.bfloat16, device=dev)
        fq.kv_prep(kv_a, latent)
        torch.cuda.synchronize()
        ref_latent = torch.cat([kv_ln(kv_a[:, :KVL]), kv_a[:, KVL:]], dim=-1)
    assert torch.equal(latent.view(torch.int16), ref_latent.view(torch.int16)), "kv_prep is not bitwise equal"
    assert torch.equal(ref[..., KVL:], got[..., KVL:]), "q_pe columns differ"
    r8, g8 = ref.view(torch.float8_e4m3fn).float(), got.view(torch.float8_e4m3fn).float()
    mismatch = (ref != got).float().mean().item()
    assert mismatch < 2e-4, f"{mismatch:.2e} of the FP8 codes differ"
    rel = ((r8 - g8) ** 2).mean().sqrt() / (r8**2).mean().sqrt()
    assert rel < 1e-3, f"relative RMS {rel.item():.2e}"


def test_fused_query_is_graph_safe():
    dev, w_qb, q_ln, kv_ln, w_kc, fq = _fixture(0)
    out = torch.randn(8, 3648, device=dev).bfloat16()
    q_a = out[:, QL : 2 * QL]
    buf = torch.zeros(8, H, KVL + ROPE, dtype=torch.uint8, device=dev)
    fq(q_a, buf)  # eager warmup
    torch.cuda.synchronize()
    eager = buf.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fq(q_a, buf)
    buf.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(buf, eager)
