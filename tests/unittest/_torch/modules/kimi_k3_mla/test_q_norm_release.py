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
"""Kimi K3.1 q-norm release: the DSA q_a_layernorm released at launch for the planned TGV q_b GEMM."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import KimiK3MLAAttention
from tensorrt_llm._torch.modules.low_m_gemm_plans import KIMI_K31_DECODE_PLAN

K3 = KimiK3MLAAttention


def _stub(n_out=5248, plan=KIMI_K31_DECODE_PLAN, device="cpu"):
    stub = SimpleNamespace(
        dsa_packed_q_weight=torch.empty(n_out, 1536, dtype=torch.bfloat16, device=device),
        dsa_packed_gemm_plan=plan,
        _k31_q_norm_release=True,
    )
    stub._packed_q_b_runs_tgv = lambda rows: K3._packed_q_b_runs_tgv(stub, rows)
    return stub


@pytest.mark.parametrize(
    "rows,expect", [(1, False), (2, True), (6, True), (24, True), (30, True), (31, False)]
)
def test_tgv_rows_of_the_tp16_plan(rows, expect):
    # TP16 [q_b | wq_b] rows: direct below 2 rows, TGV up to 30, the caller's GEMM above.
    assert K3._packed_q_b_runs_tgv(_stub(), rows) is expect


def test_no_plan_or_no_packed_q():
    assert not K3._packed_q_b_runs_tgv(_stub(plan=None), 6)
    assert not K3._packed_q_b_runs_tgv(_stub(n_out=1234), 6)
    stub = _stub()
    stub.dsa_packed_q_weight = None
    assert not K3._packed_q_b_runs_tgv(stub, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [6, 24, 1])
def test_released_norm_is_the_norm(monkeypatch, rows):
    from tensorrt_llm._torch.modules.kimi_k3_mla import q_norm_pdl
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if not q_norm_pdl.IS_AVAILABLE:
        pytest.skip("FlashInfer CuTe DSL RMSNorm unavailable")
    calls = []
    real = q_norm_pdl.rmsnorm_release_at_launch
    monkeypatch.setattr(
        q_norm_pdl, "rmsnorm_release_at_launch", lambda *a: calls.append(1) or real(*a)
    )
    norm = RMSNorm(hidden_size=1536, eps=1e-6, dtype=torch.bfloat16).cuda()
    norm.weight.data.normal_(1.0, 0.2)
    stub = _stub(device="cuda")
    stub.q_a_layernorm = norm
    stub._resolve_qa_fused_scale = lambda: None
    stub._runs_flashinfer_rmsnorm = K3._runs_flashinfer_rmsnorm
    x = (torch.randn(rows, 1536, device="cuda") * 3).to(torch.bfloat16)
    if rows == 1:
        # Not a TGV row count: the base path (not reachable from the stub) would run.
        with pytest.raises(TypeError):
            K3._q_a_layernorm_maybe_fused(stub, x, return_norm_out=True)
        assert not calls
        return
    out, normed = K3._q_a_layernorm_maybe_fused(stub, x, return_norm_out=True)
    assert calls and out is normed
    torch.testing.assert_close(out, norm(x), rtol=0, atol=0)
