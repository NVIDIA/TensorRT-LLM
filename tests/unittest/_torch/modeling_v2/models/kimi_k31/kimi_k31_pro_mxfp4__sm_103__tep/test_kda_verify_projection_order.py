# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The perf targets' KDA verify projections on two streams, in either issue order, equal the one-stream result.

``_project_verify_inputs`` of the sm_103 tep / dep targets issues the [q | k | v | g] GEMM and the gate pair
[f_a | b] -> f_b on two streams in an order chosen by the verify's row count (``_k31_verify_projection_pair``).
Every row count gives the same kernels the same inputs, so the outputs are bit-for-bit those of one stream, eager and
replayed from a CUDA graph.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_TARGETS = ("kimi_k31_pro_mxfp4__sm_103__tep", "kimi_k31_pro_mxfp4__sm_103__dep")


def _attention(target: str, heads: int):
    modeling = importlib.import_module(f"{_PKG}.{target}.modeling")
    cfg = SimpleNamespace(
        hidden_size=7168,
        rms_norm_eps=1e-5,
        linear_attn_config=dict(
            num_heads=heads,
            head_dim=128,
            short_conv_kernel_size=4,
            use_full_rank_gate=True,
            gate_lower_bound=-5.0,
        ),
        kda_fp32_gate_params=True,
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(5)
        with torch.device("cuda"):
            attention = modeling.K31KDALinearAttention(
                cfg,
                layer_idx=0,
                extra_attrs={},
                aux_stream=torch.cuda.Stream(),
            )
        with torch.no_grad():
            for p in attention.parameters():
                if p.dtype == torch.bfloat16:
                    p.normal_(0.0, 0.02)
        attention.finalize_decode_weights()
    attention.eval()
    return modeling, attention


def _project(attention, x, rows):
    out = attention._project_verify_inputs(x, rows)
    return [t.clone() for t in out if t is not None]


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize(
    "rows, heads",
    [
        (10, 3),  # gate pair first in tp (TEP32 / TEP16 low rows)
        (40, 6),  # GEMM issued first (TEP16 mid / M2 rows)
        (80, 6),  # gate pair first (TEP16 M2 rows)
        (150, 6),  # above the engine's 128-row two-stream cap
        (80, 96),  # attention DP 96 heads, gate pair first in both targets' ranges
        (160, 96),
    ],
)
def test_two_stream_order_matches_one_stream(target, rows, heads) -> None:
    modeling, attention = _attention(target, heads)
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, 7168, device="cuda", generator=gen)).to(torch.bfloat16)
    with torch.no_grad():
        with with_multi_stream(False):
            ref = _project(attention, x, rows)
        with with_multi_stream(True):
            eager = _project(attention, x, rows)
            static_x = x.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                _project(attention, static_x, rows)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = attention._project_verify_inputs(static_x, rows)
            graph.replay()
            torch.cuda.synchronize()
            replayed = [t.clone() for t in captured if t is not None]
    for got in (eager, replayed):
        assert len(got) == len(ref)
        for a, b in zip(got, ref):
            assert torch.equal(a, b)


@pytest.mark.parametrize("rows", (65, 80, 96, 128))
def test_tep16_planned_gate_pair_matches_cublas(rows) -> None:
    """The tep target's gate pair on the TEP16 low-M plan (rows where it names a kernel for [f_a | b]) gives beta and
    the forget gate of the engine's cuBLAS path bit for bit, on one stream and on two, eager and replayed."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import (
        low_m_choice,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import (
        K31_LOW_M_PLANS,
    )

    _, attention = _attention("kimi_k31_pro_mxfp4__sm_103__tep", 6)
    plan = K31_LOW_M_PLANS["tep16"]
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, 7168, device="cuda", generator=gen)).to(torch.bfloat16)
    assert low_m_choice(plan, x, attention._bfa_proj_weight) is not None

    def gates(out):
        _, _, _, forget_gate, beta, _ = out
        return [forget_gate.clone(), beta.clone()]

    with torch.no_grad():
        attention._k31_low_m_plan = None
        with with_multi_stream(False):
            ref = gates(attention._project_verify_inputs(x, rows))
        attention._k31_low_m_plan = plan
        got = []
        for multi in (False, True):
            with with_multi_stream(multi):
                got.append(gates(attention._project_verify_inputs(x, rows)))
        with with_multi_stream(True):
            static_x = x.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = attention._project_verify_inputs(static_x, rows)
            graph.replay()
            torch.cuda.synchronize()
            got.append(gates(captured))
    for outs in got:
        for a, b in zip(outs, ref):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("rows", (5, 10))
def test_tep8_gate_kernel_without_pdl_matches_pdl(rows) -> None:
    """At attention TP 8 the tep target launches the fused gate kernel without PDL (``_REC_FAB_NO_PDL_TP``): the verify
    projections (12 heads, 1-2 requests of DFlash2 width 5) equal those of the PDL launch bit for bit, on two streams,
    eager and replayed."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.ssm import (  # noqa: E501
        kda_fab,
    )

    modeling, attention = _attention("kimi_k31_pro_mxfp4__sm_103__tep", 12)
    assert 8 in modeling._REC_FAB_NO_PDL_TP
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, 7168, device="cuda", generator=gen)).to(torch.bfloat16)
    if not kda_fab.can_use_k3_kda_fab(x, attention._bfa_proj_weight, attention.f_b_proj.weight, 12):
        pytest.skip("the fused gate kernel does not run on this device")
    results = {}
    with torch.no_grad(), with_multi_stream(True):
        attention._fab_active = True
        for pdl in (True, False):
            attention._fab_pdl = pdl
            eager = _project(attention, x, rows)
            static_x = x.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = attention._project_verify_inputs(static_x, rows)
            graph.replay()
            torch.cuda.synchronize()
            results[pdl] = (eager, [t.clone() for t in captured if t is not None])
        attention._fab_active = False
    ref = results[True][0]
    for got in (*results[True], *results[False]):
        assert len(got) == len(ref)
        for a, b in zip(got, ref):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("target", _TARGETS)
def test_issue_order_by_rows(target) -> None:
    modeling = importlib.import_module(f"{_PKG}.{target}.modeling")
    order = []

    class _Event:
        def record(self, *a):
            pass

        def wait(self, *a):
            pass

    def gemm():
        order.append("gemm")
        return torch.zeros(1, device="cuda")

    def gates():
        order.append("gates")
        return (torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda"), None)

    expected = {
        "kimi_k31_pro_mxfp4__sm_103__tep": {
            8: "gates",
            10: "gates",
            15: "gemm",
            64: "gemm",
            65: "gates",
            200: "gates",
        },
        "kimi_k31_pro_mxfp4__sm_103__dep": {
            8: "gemm",
            64: "gemm",
            65: "gates",
            120: "gates",
            121: "gemm",
            200: "gemm",
        },
    }[target]
    with with_multi_stream(True):
        for rows, first in expected.items():
            order.clear()
            modeling._k31_verify_projection_pair(
                gemm, gates, _Event(), _Event(), torch.cuda.Stream(), rows
            )
            assert order[0] == first, (rows, order)
    # one stream outside multi-stream execution: GEMM, then the gate pair
    order.clear()
    with with_multi_stream(False):
        modeling._k31_verify_projection_pair(
            gemm, gates, _Event(), _Event(), torch.cuda.Stream(), 80
        )
    assert order == ["gemm", "gates"]
