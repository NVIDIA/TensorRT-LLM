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
"""Dispatch of the Kimi K3.1 gen-step collective knobs, on stand-ins (no GPU, no MNNVL):

* the BF16 o_proj's deferred finish (``_K31_BF16_OPROJ_AR_FINISH``): an o_proj without the push epilogue gets a
  push target only when A_mid finishes the all-reduce (a deferred target), and keeps its planned GEMM otherwise;
  with ``_K31_BF16_OPROJ_AR_FINISH_PUSH_ONLY`` only for the shapes the push GEMM serves;
* the BF16 o_proj push GEMM (``_K31_BF16_OPROJ_AR_PUSH``): taken only for a deferred, not yet pushed target;
* the mailbox fold: a fused-tail residual left in the mailbox materializes as the copy kernel's
  output plus the residual (the arithmetic of the path it replaces);
* the PDL attn_res kernel in mode "2" beside the mailbox fold: rows the fused tail does not serve hand
  ``residual + bf16(routed + shared)`` to the next layer (``defer_residual_to``), the caller's two BF16 adds.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn


class _Target(SimpleNamespace):
    pass


class _FakeAllReduce:
    def __init__(self, deferred: bool):
        self.deferred = deferred
        self.calls = []

    def lamport_push_target(self, rows, hidden, in_place=False, deferred=False):
        self.calls.append((rows, hidden, in_place, deferred))
        return _Target(deferred=deferred and self.deferred, pushed=False)


def _kda_stub(o_proj, allreduce, deferred_finish):
    from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

    stub = SimpleNamespace(
        o_proj=o_proj,
        _o_allreduce=allreduce,
        ar_finish_deferred=deferred_finish,
        ar_result_in_place=deferred_finish,
        hidden_size=7168,
    )
    stub._o_proj_pushes = lambda: KimiKDALinearAttention._o_proj_pushes(stub)
    return lambda rows, gen_only: KimiKDALinearAttention._o_proj_ar_push_target(
        stub, rows, gen_only
    )


@pytest.mark.parametrize("flag", [False, True])
def test_bf16_o_proj_gets_deferred_target_only(monkeypatch, flag):
    from tensorrt_llm._torch.distributed import ops

    # The deferred finish comes with the push GEMM (configure_k31_oproj_all_reduce sets both) and
    # keeps the push-only restriction; 56 rows of the TP8 o_proj are a shape the push GEMM serves
    # (its table: test_push_gemm_table_bounds).
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_FINISH", flag)
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_PUSH", flag)
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_FINISH_PUSH_ONLY", True)
    monkeypatch.setattr(
        ops, "_bf16_push_gemm_serves", lambda m, n, k: (m, n, k) == (56, 7168, 1536)
    )
    linear = nn.Linear(1536, 7168, bias=False)
    # A_mid finishes the all-reduce: the BF16 projection gets the deferred target with the flag.
    target_of = _kda_stub(linear, _FakeAllReduce(deferred=True), deferred_finish=True)
    target = target_of(56, True)
    assert (target is not None) == flag
    if flag:
        assert target.deferred
    # Not generation-only, or A_mid does not finish it: no target.
    assert target_of(56, False) is None
    assert _kda_stub(linear, _FakeAllReduce(True), deferred_finish=False)(56, True) is None
    # A transport that is not deferred is refused for a projection without the push epilogue.
    assert _kda_stub(linear, _FakeAllReduce(False), deferred_finish=True)(56, True) is None


def test_bf16_o_proj_finish_push_only(monkeypatch):
    """``_K31_BF16_OPROJ_AR_FINISH_PUSH_ONLY``: a deferred target only for shapes the push GEMM serves
    (and only with the push GEMM on); the C++ all-reduce keeps every other shape."""
    from tensorrt_llm._torch.distributed import ops

    served = {(56, 7168, 1536), (8, 7168, 768)}
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_FINISH", True)
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_FINISH_PUSH_ONLY", True)
    monkeypatch.setattr(ops, "_bf16_push_gemm_serves", lambda m, n, k: (m, n, k) in served)
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_PUSH", True)
    linear = nn.Linear(1536, 7168, bias=False)
    target_of = _kda_stub(linear, _FakeAllReduce(deferred=True), deferred_finish=True)
    target = target_of(56, True)
    assert target is not None and target.deferred
    assert target_of(112, True) is None
    # Without the push GEMM no shape is served.
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_PUSH", False)
    assert target_of(56, True) is None
    # Without the push-only restriction (no shape test) and the shape-less query keep the flag's answer.
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_FINISH_PUSH_ONLY", False)
    assert target_of(112, True) is not None
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_FINISH_PUSH_ONLY", True)
    assert ops.bf16_oproj_ar_finish_enabled()


def test_push_gemm_table_bounds():
    """The shapes the push-only deferred finish takes: TP8 o_proj up to 72 rows, TP16 up to 128, nothing else."""
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.distributed import ops

    assert ops._bf16_push_gemm_serves(72, 7168, 1536)
    assert not ops._bf16_push_gemm_serves(73, 7168, 1536)
    assert ops._bf16_push_gemm_serves(128, 7168, 768)
    assert not ops._bf16_push_gemm_serves(129, 7168, 768)
    assert not ops._bf16_push_gemm_serves(16, 4096, 1536)


def test_push_epilogue_projection_keeps_its_target(monkeypatch):
    from tensorrt_llm._torch.distributed import ops

    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_FINISH", False)
    pushing = SimpleNamespace(next_ar_push=None)
    target = _kda_stub(pushing, _FakeAllReduce(deferred=False), deferred_finish=False)(28, True)
    assert target is not None and not target.deferred


def test_push_bf16_projection_gates(monkeypatch):
    from tensorrt_llm._torch.distributed import ops

    x = torch.zeros(8, 1536, dtype=torch.bfloat16)
    w = torch.zeros(7168, 1536, dtype=torch.bfloat16)
    target = _Target(deferred=True, pushed=False, output=torch.zeros(8, 7168, dtype=torch.bfloat16))
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_PUSH", False)
    assert not ops.push_bf16_projection(x, w, target)
    monkeypatch.setattr(ops, "_K31_BF16_OPROJ_AR_PUSH", True)
    assert not ops.push_bf16_projection(x, w, None)
    assert not ops.push_bf16_projection(
        x, w, _Target(deferred=False, pushed=False, output=target.output)
    )
    assert not ops.push_bf16_projection(
        x, w, _Target(deferred=True, pushed=True, output=target.output)
    )
    # CPU tensors: the kernel does not serve them, nothing is launched or marked.
    assert not ops.push_bf16_projection(x, w, target)
    assert not target.pushed


def test_deferred_tail_residual_materializes_copy_plus_residual():
    from tensorrt_llm._torch.models import modeling_kimi_linear as ml

    residual = torch.randn(5, 16).to(torch.bfloat16)
    moe = torch.randn(5, 16).to(torch.bfloat16)
    copies = []

    class _Tail:
        def copy_out(self, mailbox, m):
            copies.append((mailbox, m))
            return moe

    mailbox = object()
    deferred = ml._DeferredTailResidual(residual, _Tail(), mailbox)
    got = ml._materialize_prefix_sum(deferred)
    assert copies == [(mailbox, 5)]
    assert torch.equal(got, residual + moe)
    # Not a deferred value: unchanged.
    assert ml._materialize_prefix_sum(residual) is residual


def test_moe_runtime_defers_unfused_residual(monkeypatch):
    """``defer_residual_to`` is part of the MoE runtime's forward, the deferral gate follows
    the PDL attn_res kernel's mode "2" and its row cap, and the deferred update materializes as the two
    BF16 adds it replaces."""
    import inspect

    from tensorrt_llm._torch.models import modeling_kimi_linear as mk

    assert "defer_residual_to" in inspect.signature(mk.KimiK3MoERuntime.forward).parameters
    monkeypatch.setattr(mk, "_K3_ATTN_RES_PDL_MODE", "2")
    monkeypatch.setattr(mk, "_K3_ATTN_RES_PDL_MAX_M", 576)
    assert mk._defer_moe_residual(288) and not mk._defer_moe_residual(577)
    monkeypatch.setattr(mk, "_K3_ATTN_RES_PDL_MODE", "1")
    assert not mk._defer_moe_residual(288)
    g = torch.Generator().manual_seed(3)
    residual, routed, shared = (
        torch.randn(288, 7168, generator=g).to(torch.bfloat16) for _ in range(3)
    )
    deferred = mk._DeferredMoEResidual(residual, routed, shared)
    assert torch.equal(deferred.materialize(), residual + (routed + shared))
    assert isinstance(mk._materialize_prefix_sum(deferred), torch.Tensor)
