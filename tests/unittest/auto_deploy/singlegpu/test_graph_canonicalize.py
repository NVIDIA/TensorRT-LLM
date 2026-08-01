# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator

import pytest
import torch
from torch import nn
from torch.fx import symbolic_trace

from tensorrt_llm._torch.auto_deploy.utils._graph import canonicalize_graph


class _OutOfOrderCastModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        add = x + 1
        cast = add.to(torch.bfloat16)
        return cast * 2


def test_canonicalize_graph_restores_topological_order():
    gm = symbolic_trace(_OutOfOrderCastModule())

    cast_node = next(
        node for node in gm.graph.nodes if node.op == "call_method" and node.target == "to"
    )
    mul_node = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == operator.mul
    )

    mul_node.append(cast_node)

    with pytest.raises(RuntimeError, match="used before it has been defined"):
        gm.graph.lint()

    canonicalize_graph(gm)
    gm.graph.lint()

    node_order = {node.name: idx for idx, node in enumerate(gm.graph.nodes)}
    assert node_order[cast_node.name] < node_order[mul_node.name]
