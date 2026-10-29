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
"""Correctness of the Kimi-K3 tcgen05 router GEMM (``k3_router_umma``).

The kernel writes the router logits as FP32 split-K partials ``[S, M, 896]``; the routing tail
(``trtllm::k3_route_tail``) sums the splits in order and routes. The partials are the FP32 product
over each K slice (deterministic, no slot left unwritten), their sum differs from the cuBLAS
router GEMM at FP32 rounding level only, and the tail's routing on the sum is bit-identical to
``noaux_tc_op`` + ``moe_sort`` on it. The wiring tests run ``KimiK3MoERuntime.forward`` with
``TRTLLM_K3_ROUTER_GEMM=umma`` around stand-in routed experts: the UMMA GEMM replaces the cuBLAS
router GEMM from 17 up to ``TRTLLM_K3_ROUTER_GEMM_MAX_M`` tokens and the tail routes on its
partials.
"""

from types import SimpleNamespace

import pytest
import torch
from test_k3_route_tail import (
    HIDDEN,
    NUM_EXPERTS,
    TILE,
    TOP_K,
    _accept_stub_backend,
    _assert_bit_equal,
    _check_outputs,
    _module,
    _noaux,
)
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103/SM107 GPU",
)

# Token counts around the UMMA N padding (multiples of 16), the TP8 graph sizes (18, 36, 84,
# 96) and the kernel limit (256).
UMMA_MS = [1, 7, 16, 17, 18, 24, 30, 32, 33, 36, 48, 64, 65, 84, 96, 128, 256]
K_CHUNK = 64


def _umma_module():
    import tensorrt_llm  # noqa: F401  (registers the reference ops)
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_router_umma

    return k3_router_umma


def _router_inputs(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn((m, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    w = (torch.randn((NUM_EXPERTS, HIDDEN), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    return h, w, bias


def _summed(parts):
    """The FP32 sum of the splits in split order (what the routing tail routes on)."""
    logits = parts[0].clone()
    for s in range(1, parts.shape[0]):
        logits = logits + parts[s]
    return logits


@pytest.mark.parametrize("m", UMMA_MS)
@pytest.mark.parametrize("splits", [5, 7])
def test_umma_partials(m, splits):
    """Every partial is the FP32 product over its K slice; the slices tile K in order, the result
    is deterministic and no slot is left unwritten."""
    ru = _umma_module()
    h, w, _ = _router_inputs(m, seed=m + 100 * splits)
    parts = torch.full((splits, m, NUM_EXPERTS), float("nan"), device="cuda")
    ru.run_k3_router_umma(h, w, parts)
    torch.cuda.synchronize()
    assert not torch.isnan(parts).any()
    chunks = HIDDEN // K_CHUNK
    start = 0
    for s in range(splits):
        count = chunks // splits + (1 if s < chunks % splits else 0)
        lo, hi = start * K_CHUNK, (start + count) * K_CHUNK
        exact = (h[:, lo:hi].double() @ w[:, lo:hi].double().t()).float()
        torch.testing.assert_close(parts[s], exact, rtol=0, atol=2e-5)
        start += count
    assert start == chunks
    exact = (h.double() @ w.double().t()).float()
    torch.testing.assert_close(parts.sum(0), exact, rtol=0, atol=5e-5)
    again = torch.empty_like(parts)
    ru.run_k3_router_umma(h, w, again)
    torch.cuda.synchronize()
    _assert_bit_equal(again, parts, "second launch")


@pytest.mark.parametrize("m", UMMA_MS)
@pytest.mark.parametrize("offset,num_local", [(7 * 56, 56), (224, 224), (0, 896)])
def test_tail_on_umma_partials(m, offset, num_local):
    """The routing tail on the UMMA partials is exact on their sum (``noaux_tc_op`` + ``moe_sort``)
    and selects the cuBLAS router GEMM's experts except at near-ties."""
    rt = _module()
    ru = _umma_module()
    assert rt is not None
    h, w, bias = _router_inputs(m, seed=m + offset)
    assert ru.can_run(h, w)
    parts = torch.empty((ru.DEFAULT_SPLITS, m, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    ru.run_k3_router_umma(h, w, parts)
    outs = torch.ops.trtllm.k3_route_tail(parts, bias, 1.0, offset, num_local, TILE)
    logits = _summed(parts)
    exact = (h.double() @ w.double().t()).float()
    torch.testing.assert_close(logits, exact, rtol=0, atol=5e-5)
    ref_gemm = torch.ops.trtllm.dsv3_router_gemm_op(h, w.t(), bias=None, out_dtype=torch.float32)
    torch.testing.assert_close(logits, ref_gemm, rtol=0, atol=1e-4)
    _check_outputs(outs, logits, bias, 1.0, offset, num_local)
    chain_ids, _ = _noaux(ref_gemm, bias, 1.0)
    differing = sum(
        set(a) != set(b) for a, b in zip(outs[0].cpu().tolist(), chain_ids.cpu().tolist())
    )
    assert differing <= 1, f"{differing} of {m} tokens select other experts than the cuBLAS chain"


def test_umma_can_run_limits():
    ru = _umma_module()
    h, w, _ = _router_inputs(257, seed=1)
    assert ru.can_run(h[:256], w) and not ru.can_run(h, w)
    assert not ru.can_run(h[:0], w)
    assert not ru.can_run(h[:16].float(), w)
    assert not ru.can_run(h[:16], w.float())
    assert not ru.can_run(h[:16, :-64], w[:, :-64])
    assert not ru.can_run(h[:16, ::2], w)


def test_umma_tail_cuda_graph_replay():
    """The GEMM and the tail on its partials capture into a graph and replay on new inputs."""
    rt = _module()
    ru = _umma_module()
    assert rt is not None
    m, offset, num_local = 24, 224, 224
    h, w, bias = _router_inputs(m, seed=7)
    static = h.clone()
    parts = torch.empty((ru.DEFAULT_SPLITS, m, NUM_EXPERTS), dtype=torch.float32, device="cuda")

    def chain():
        ru.run_k3_router_umma(static, w, parts)
        return torch.ops.trtllm.k3_route_tail(parts, bias, 1.0, offset, num_local, TILE)

    chain()  # compile and warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = chain()
    for seed in range(3):
        fresh, _, _ = _router_inputs(m, seed=20 + seed)
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        eager = chain()
        torch.cuda.synchronize()
        for got, exp, what in zip(outs, eager, ("ids", "weights")):
            _assert_bit_equal(got, exp, what)
        _check_outputs(outs, _summed(parts), bias, 1.0, offset, num_local)


# ---------------------------------------------------------------------------
# Wiring in KimiK3MoERuntime.forward (TRTLLM_K3_ROUTER_GEMM=umma)
# ---------------------------------------------------------------------------


class _Identity(nn.Module):
    def forward(self, value, **kwargs):
        return value


def _cublas_logits(h, w):
    return torch.ops.trtllm.dsv3_router_gemm_op(h, w.t(), bias=None, out_dtype=torch.float32)


def _stand_in_runtime(monkeypatch, w, bias, offset, num_local, *, multi_stream, flag="umma"):
    """``KimiK3MoERuntime`` around a real routing method, stand-in routed experts that route when
    they are called (as the CuteDSL backend does after forking its input quantization) and the
    real router GEMMs. ``seen`` collects the logits they were handed and the routing they got;
    ``calls`` counts the cuBLAS router GEMMs."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as mk

    routing = mk._KimiK3RoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    seen, calls = {}, []

    class _Routed:
        routing_method = routing
        use_dp = False
        all_reduce = None
        backend = SimpleNamespace(
            slot_start=offset, expert_size_per_partition=num_local, num_slots=NUM_EXPERTS
        )

        def __call__(self, value, router_logits, input_ready_event=None, **kwargs):
            if input_ready_event is not None:
                input_ready_event.wait()
            seen["logits"] = router_logits
            seen["routing"] = routing.apply(router_logits)
            ids, weights = seen["routing"]
            return value + weights.sum(-1, keepdim=True).to(value.dtype) + ids.sum(-1, keepdim=True)

    def compute_logits(value):
        calls.append(value.shape[0])
        return _cublas_logits(value, w)

    runtime = mk.KimiK3MoERuntime.__new__(mk.KimiK3MoERuntime)
    nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(
        compute_logits=compute_logits,
        weight=w,
        e_score_correction_bias=bias,
        routed_scaling_factor=1.0,
        num_experts=NUM_EXPERTS,
    )
    runtime.routed_experts = _Routed()
    for name in ("routed_expert_down_proj", "routed_expert_norm", "routed_expert_up_proj"):
        setattr(runtime, name, _Identity())
    runtime.shared_experts = _Identity()
    runtime.shared_down_all_reduce = None
    runtime._reduce_routed_output = False
    runtime._defer_shared_allreduce = False
    runtime._merge_moe_allreduce = False
    runtime.moe_routed_event = None
    streams = torch.cuda.Stream if multi_stream else (lambda: None)
    event = torch.cuda.Event if multi_stream else (lambda: None)
    runtime.shared_expert_stream = streams()
    runtime.latent_down_stream = streams()
    runtime.moe_main_event = event()
    runtime.moe_shared_event = event()
    runtime.moe_latent_fork_event = event()
    runtime.moe_latent_ready_event = event()
    monkeypatch.setattr(mk, "_K3_ROUTE_TAIL", True)
    monkeypatch.setattr(mk, "_K3_ROUTER_GEMM", flag)
    _accept_stub_backend(monkeypatch, mk)
    monkeypatch.setattr(runtime, "_can_fuse_router_down", lambda value: False)
    monkeypatch.setattr(runtime, "_routed_projection", lambda value, projection: value)
    return runtime, routing, seen, calls


def _run(runtime, h, multi_stream):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    with with_multi_stream(multi_stream):
        out = runtime(h)
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("multi_stream", [False, True])
@pytest.mark.parametrize("m", [17, 18, 32, 33, 36, 84, 96])
@pytest.mark.parametrize("offset,num_local", [(224, 224), (0, 896)])
def test_moe_runtime_routes_with_umma_gemm(monkeypatch, m, offset, num_local, multi_stream):
    """The runtime computes the UMMA GEMM's split-K partials instead of the cuBLAS GEMM, hands the
    routed experts the first split as the logits and routes with the tail on all splits when the
    experts ask for the routing. The limit is raised to 96 tokens (TRTLLM_K3_ROUTER_GEMM_MAX_M) so
    the wide routing tail summing the partials is covered as well; the default limit of 32 is in
    ``test_umma_gate_limits``."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as mk

    ru = _umma_module()
    h, w, bias = _router_inputs(m, seed=11 + m)
    runtime, routing, seen, calls = _stand_in_runtime(
        monkeypatch, w, bias, offset, num_local, multi_stream=multi_stream
    )
    monkeypatch.setattr(mk, "_K3_ROUTER_GEMM_MAX_M", 96)
    _run(runtime, h, multi_stream)
    assert not calls, "the cuBLAS router GEMM must not run"
    parts = torch.empty((ru.DEFAULT_SPLITS, m, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    ru.run_k3_router_umma(h, w, parts)
    torch.cuda.synchronize()
    _assert_bit_equal(seen["logits"], parts[0], "placeholder logits = first split")
    ids, weights = seen["routing"]
    ref = torch.ops.trtllm.k3_route_tail(parts, bias, 1.0, offset, num_local, TILE)
    _assert_bit_equal(ids, ref[0], "topk_ids")
    _assert_bit_equal(weights, ref[1], "topk_weights")
    key, sort_outputs = ids._k3_presorted
    assert key == (offset, num_local, TILE)
    _check_outputs((ids, weights, *sort_outputs), _summed(parts), bias, 1.0, offset, num_local)
    assert routing.deferred is None


@pytest.mark.parametrize("multi_stream", [False, True])
@pytest.mark.parametrize(
    "m,flag", [(7, "umma"), (16, "umma"), (33, "umma"), (36, "umma"), (96, "umma"), (18, "cublas")]
)
def test_moe_runtime_keeps_cublas_outside_the_umma_gate(monkeypatch, m, flag, multi_stream):
    """Up to 16 tokens (the dsv3 MMA kernel), above TRTLLM_K3_ROUTER_GEMM_MAX_M (default 32) and
    without TRTLLM_K3_ROUTER_GEMM=umma, the dsv3 / cuBLAS router GEMM serves and the tail routes on
    its logits, as before."""
    ru = _umma_module()
    offset, num_local = 224, 224
    h, w, bias = _router_inputs(m, seed=5 + m)
    runtime, routing, seen, calls = _stand_in_runtime(
        monkeypatch, w, bias, offset, num_local, multi_stream=multi_stream, flag=flag
    )

    def _forbidden(*args, **kwargs):
        raise AssertionError("the UMMA router GEMM must not run")

    monkeypatch.setattr(ru, "run_k3_router_umma", _forbidden)
    _run(runtime, h, multi_stream)
    assert calls == [m]
    logits = seen["logits"]
    torch.testing.assert_close(logits, _cublas_logits(h, w), rtol=0, atol=1e-4)
    ids, weights = seen["routing"]
    _check_outputs((ids, weights, *ids._k3_presorted[1]), logits, bias, 1.0, offset, num_local)
    assert routing.deferred is None


def test_umma_gate_limits(monkeypatch):
    """One gate: the flag, ``17 <= M <= TRTLLM_K3_ROUTER_GEMM_MAX_M`` and the kernel's support check."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as mk

    h, w, bias = _router_inputs(40, seed=3)
    runtime, _, _, _ = _stand_in_runtime(monkeypatch, w, bias, 0, 896, multi_stream=False)
    assert mk._K3_ROUTER_GEMM_MAX_M == 32  # the Rubin default of the MoE entry replica
    assert mk._K3_ROUTER_GEMM_MIN_M == 17  # cuBLAS starts above the dsv3 MMA kernel's 16 tokens
    assert runtime._can_umma_route(h[:17]) and runtime._can_umma_route(h[:32])
    assert not runtime._can_umma_route(h[:16]) and not runtime._can_umma_route(h[:0])
    assert not runtime._can_umma_route(h[:33])
    monkeypatch.setattr(mk, "_K3_ROUTER_GEMM_MAX_M", 40)
    assert runtime._can_umma_route(h)
    assert not runtime._can_umma_route(h.float())  # the kernel needs BF16
    monkeypatch.setattr(mk, "_K3_ROUTER_GEMM", "cublas")
    assert not runtime._can_umma_route(h[:18])


def test_moe_runtime_umma_cuda_graph_replay(monkeypatch):
    """The whole forward (UMMA GEMM on main, latent chain and shared expert on side streams, tail on
    the partials) captures into a graph and replays on new inputs like the eager forward."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    m, offset, num_local = 18, 224, 224
    h, w, bias = _router_inputs(m, seed=9)
    runtime, _, seen, _ = _stand_in_runtime(
        monkeypatch, w, bias, offset, num_local, multi_stream=True
    )
    static = h.clone()
    with with_multi_stream(True):
        runtime(static)  # compile and warm up outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runtime(static)
        graph_routing = seen["routing"]
        for seed in range(3):
            fresh, _, _ = _router_inputs(m, seed=30 + seed)
            static.copy_(fresh)
            graph.replay()
            torch.cuda.synchronize()
            captured_ids = graph_routing[0].clone()
            captured = out.clone()
            eager = runtime(static)
            torch.cuda.synchronize()
            _assert_bit_equal(captured_ids, seen["routing"][0], "topk_ids")
            _assert_bit_equal(captured, eager, "forward output")
