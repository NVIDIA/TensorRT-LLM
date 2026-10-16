# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Piecewise CUDA graphs with max_num_streams > 1: the auto multi-stream scheduler costs the graph
by kernel class and overlaps independent branches."""
import torch

import tensorrt_llm._torch.compilation.multi_stream.auto_multi_stream as ams


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
