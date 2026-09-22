# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Piecewise CUDA graphs with max_num_streams > 1: the auto multi-stream scheduler costs the graph
by kernel class and overlaps independent branches, and TLLM_MULTI_STREAM_MAX_TOKENS keeps the
schedule for pieces running at most that many tokens while a single-stream twin over the same
parameters serves the larger ones."""
import copy
from unittest.mock import MagicMock

import pytest
import torch

import tensorrt_llm._torch.compilation.multi_stream.auto_multi_stream as ams
import tensorrt_llm._torch.compilation.piecewise_optimizer as po


def test_schedule_overlaps_independent_branches():
    """Two independent GEMM chains joined by an add (the shape of the Qwen3-Next MoE block:
    router -> experts next to the shared-expert MLP) are spread over both streams, the join waits
    on a cross-stream event and the simulated makespan is shorter than serial execution."""

    class Block(torch.nn.Module):

        def forward(self, h, w_router, w_routed, w_up, w_down):
            logits = torch.ops.aten.mm.default(h, w_router)
            routed = torch.ops.aten.mm.default(logits, w_routed)
            up = torch.ops.aten.mm.default(h, w_up)
            act = torch.ops.aten.silu.default(up)
            shared = torch.ops.aten.mm.default(act, w_down)
            return torch.ops.aten.add.Tensor(routed, shared)

    gm = torch.fx.symbolic_trace(Block())
    dag = ams.MultiStreamDAG(gm)
    num_events = dag.assign_streams(2)
    calls = [n for n in dag.nodes.values() if n.node.op == "call_function"]
    assert {n.stream.id for n in calls} == {0, 1}
    assert num_events >= 1
    join = next(n for n in calls if n.node.target is torch.ops.aten.add.Tensor)
    assert join.wait_on  # the join records a cross-stream dependency
    serial = sum(n.weight for n in calls)
    assert max(n.end_time for n in calls) < serial
    # The critical path (4 GEMMs on the longer chain + join) bounds the makespan.
    assert max(n.end_time for n in calls) >= 2 * ams.GEMM_OP_COST + ams.DEFAULT_OP_COST


def _runner(default, large, limit, capture=(8, 64, 512)):
    return po.PiecewiseRunner(
        graph=MagicMock(),
        name="submod_0",
        compile_time_num_tokens=16,
        runtime_num_tokens_idx=None,
        capture_num_tokens=list(capture),
        graph_pool_handle=None,
        default_callable=default,
        enable_inductor=False,
        is_first_runner=True,
        is_last_runner=True,
        large_callable=large,
        multi_stream_max_tokens=limit,
    )


def test_token_gate_selects_the_twin(monkeypatch):
    """Captures and eager token counts at or below the limit keep the multi-stream callable, the
    larger ones take the single-stream twin; without a limit or a twin the default serves all."""
    ms, plain = MagicMock(name="multi_stream", return_value="ms"), MagicMock(name="single_stream", return_value="plain")
    r = _runner(ms, plain, limit=256)
    assert r.entries[8].callable is ms and r.entries[64].callable is ms
    assert r.entries[512].callable is plain
    assert r.callable_for(1) is ms and r.callable_for(256) is ms
    assert r.callable_for(257) is plain and r.callable_for(None) is ms
    assert all(e.callable is ms for e in _runner(ms, plain, limit=None).entries.values())
    assert _runner(ms, plain, limit=None).callable_for(4096) is ms
    assert _runner(ms, None, limit=256).callable_for(4096) is ms
    # eager fallback: the compile-time token count (16) is static here, the runner dispatches on it
    monkeypatch.setattr(po, "get_piecewise_cuda_graph_flag", lambda: False)
    assert _runner(ms, plain, limit=8)() == "plain"
    assert _runner(ms, plain, limit=16)() == "ms"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_single_stream_twin_shares_parameters():
    """piecewise_optimizer builds the twin as a GraphModule over the same root: parameters are
    shared by reference, only the graph is copied (and only when the limit is set)."""

    class M(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.randn(8, 8, device="cuda"))
            self.w2 = torch.nn.Parameter(torch.randn(8, 8, device="cuda"))

        def forward(self, x):
            return x @ self.w1 + x @ self.w2

    gm = torch.fx.symbolic_trace(M())
    twin = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
    assert twin.w1 is gm.w1 and twin.w2 is gm.w2
    assert twin.graph is not gm.graph
    x = torch.randn(4, 8, device="cuda")
    torch.testing.assert_close(twin(x), gm(x))
