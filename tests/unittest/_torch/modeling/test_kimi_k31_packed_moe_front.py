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
"""KimiK3MoERuntime.prepare_packed_front (the packed MoE front): the router, latent_down and shared
gate_up weights become row views of one packed tensor, idempotently; the packed GEMM returns the three
projections."""

import types
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiK3MoERuntime


def _runtime(rows=(896, 3584, 768), k=256, device="cpu"):
    def lin(n):
        return SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.randn(n, k, device=device).to(torch.bfloat16), requires_grad=False
            )
        )

    rt = SimpleNamespace(
        gate=lin(rows[0]),
        routed_expert_down_proj=lin(rows[1]),
        shared_experts=SimpleNamespace(gate_up_proj=lin(rows[2])),
        _packed_front_unsupported=lambda: None,
    )
    for name in (
        "_packed_front_weights",
        "prepare_packed_front",
        "_packed_front_tactic",
        "_packed_front_forward",
    ):
        setattr(rt, name, types.MethodType(getattr(KimiK3MoERuntime, name), rt))
    return rt


def test_off_by_default():
    rt = _runtime()
    assert rt.prepare_packed_front() is False and rt._packed_front is None


def test_weights_become_row_views(monkeypatch):
    monkeypatch.setattr(mkl, "_K31_PACKED_MOE_FRONT", True)
    rt = _runtime()
    before = [w.clone() for w in rt._packed_front_weights()]
    assert rt.prepare_packed_front()
    packed, rows = rt._packed_front
    assert rows == (896, 3584, 768) and packed.shape == (896 + 3584 + 768, 256)
    torch.testing.assert_close(packed, torch.cat(before), rtol=0, atol=0)
    offs = (0, 896, 896 + 3584)
    for w, off, ref in zip(rt._packed_front_weights(), offs, before):
        assert w.data_ptr() == packed[off:].data_ptr()
        torch.testing.assert_close(w, ref, rtol=0, atol=0)
    # A second call (cache_derived_state after post_load_weights) keeps the same storage.
    assert rt.prepare_packed_front()
    assert rt._packed_front[0].data_ptr() == packed.data_ptr()
    torch.testing.assert_close(rt._packed_front[0], packed, rtol=0, atol=0)


def test_tactic_range(monkeypatch):
    monkeypatch.setattr(mkl, "_K31_PACKED_MOE_FRONT", True)
    rt = _runtime()
    rt.prepare_packed_front()
    # Below _K31_PACKED_MOE_FRONT_MIN_M (193) rows the separate projections run.
    for m in (1, 4, 24, 129, 192):
        assert rt._packed_front_tactic(torch.zeros(m, 256, dtype=torch.bfloat16)) is None
    # From 193 rows up to _K31_PACKED_MOE_FRONT_MAX_M (288) the packed GEMM runs tactic 24.
    x = torch.zeros(193, 256, dtype=torch.bfloat16)
    assert rt._packed_front_tactic(x) == 24
    assert rt._packed_front_tactic(torch.zeros(288, 256, dtype=torch.bfloat16)) == 24
    assert rt._packed_front_tactic(torch.zeros(289, 256, dtype=torch.bfloat16)) is None
    assert rt._packed_front_tactic(x.float()) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_packed_forward_matches_the_projections(monkeypatch):
    from tensorrt_llm._torch.kernels.blackwell import tgv_multi_out_gemm
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() not in (100, 103) or not tgv_multi_out_gemm.IS_AVAILABLE:
        pytest.skip("needs an SM100 / SM103 GPU with the TGV CuTe DSL kernel")
    monkeypatch.setattr(mkl, "_K31_PACKED_MOE_FRONT", True)
    rt = _runtime(k=7168, device="cuda")
    rt.prepare_packed_front()
    # A row count the packed front serves (_K31_PACKED_MOE_FRONT_MIN_M .. _K31_PACKED_MOE_FRONT_MAX_M).
    x = torch.randn(224, 7168, device="cuda").to(torch.bfloat16)
    tactic = rt._packed_front_tactic(x)
    assert tactic == 24
    logits, latent, gate_up = rt._packed_front_forward(x, tactic)
    assert logits.dtype == torch.float32 and latent.dtype == gate_up.dtype == torch.bfloat16
    for got, w in zip((logits, latent, gate_up), rt._packed_front_weights()):
        torch.testing.assert_close(got.float(), x.float() @ w.float().t(), rtol=2e-2, atol=2e-2)
