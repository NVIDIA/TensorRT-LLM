# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest
import torch
import torch.nn as nn
from utils.cpp_paths import llm_root  # noqa: F401

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
    AllGatherStrategy,
    DistBackend,
    ShardingTransformConfig,
    ShardingTransformContainer,
    WeightShardingInfo,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import SplitDimension
from tensorrt_llm._torch.auto_deploy.utils._graph import recompile
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm.functional import AllReduceStrategy

# needed since LLM API uses MPI executor pool internally for TP>1, which leaks a thread on shutdown
pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.mark.parametrize(
    "strategy",
    [
        "AUTO",
        "NCCL",
        "TWOSHOT",
        "MIN_LATENCY",
        "SYMM_MEM",
    ],
)
def test_allreduce_strategy_propagation(strategy):
    """Test that allreduce_strategy is correctly propagated to graph nodes.

    This test verifies that when we set an allreduce_strategy on the ShardingConfig,
    it gets properly injected into the transforms and passed to the torch_dist_all_reduce
    nodes in the compiled graph.
    """

    # Create a simple MLP model
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(128, 256, bias=False)
            self.linear2 = nn.Linear(256, 128, bias=False)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    model = SimpleMLP()
    dummy_input = torch.randn(2, 128)

    # Export to graph
    gm = torch_export_to_gm(model, (dummy_input,))

    # Find linear nodes in the graph
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op

    linear_nodes = [node for node in gm.graph.nodes if is_linear_op(node)]
    assert len(linear_nodes) == 2, f"Expected 2 linear nodes, found {len(linear_nodes)}"

    linear1_node, linear2_node = linear_nodes[0], linear_nodes[1]

    # Create sharding config with specified strategy
    rank, world_size = 0, 4

    config = ShardingTransformConfig(
        rank=rank,
        world_size=world_size,
        stage="sharding",
        allreduce_strategy=AllReduceStrategy[strategy],
    )
    sharding_container = ShardingTransformContainer(config=config)

    # Add transforms: column shard linear1, row shard linear2 (triggers allreduce)
    sharding_container.add(
        WeightShardingInfo(
            target_node=linear1_node.name,
            config=config,
            split_dim=SplitDimension.COLUMN,
            dist_op=None,
        )
    )
    sharding_container.add(
        WeightShardingInfo(
            target_node=linear2_node.name,
            config=config,
            split_dim=SplitDimension.ROW,
            dist_op="all_reduce",
        )
    )

    # Verify transforms have the strategy injected
    assert len(sharding_container.weight_sharding_transforms) == 2
    for transform in sharding_container.weight_sharding_transforms:
        assert transform.config.allreduce_strategy == AllReduceStrategy[strategy], (
            f"Transform {transform.target_node} should have strategy {strategy}, "
            f"got {transform.config.allreduce_strategy}"
        )

    # Apply transforms
    for transform in sharding_container.weight_sharding_transforms:
        node = next((n for n in gm.graph.nodes if n.name == transform.target_node), None)
        if node:
            transform.check_and_apply(gm, node)

    recompile(gm)

    # Verify the graph contains torch_dist_all_reduce nodes with correct strategy
    allreduce_nodes = [
        node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.torch_dist_all_reduce)
    ]

    # Should have exactly one allreduce node (from linear2 row sharding)
    assert len(allreduce_nodes) == 1, f"Expected 1 allreduce node, found {len(allreduce_nodes)}"

    # Verify the allreduce node has the correct strategy argument
    allreduce_node = allreduce_nodes[0]
    # torch_dist_all_reduce signature: (input, strategy_name)
    assert len(allreduce_node.args) == 2, (
        f"Expected 2 args for allreduce node, got {len(allreduce_node.args)}"
    )

    strategy_arg = allreduce_node.args[1]
    assert strategy_arg == strategy, (
        f"Expected allreduce strategy '{strategy}', got '{strategy_arg}'"
    )

    print(f"✓ Test passed: allreduce_strategy '{strategy}' correctly propagated to graph node")


@pytest.mark.parametrize(
    "strategy",
    [
        "AUTO",
        "SYMM_MEM",
    ],
)
def test_allgather_strategy_propagation(strategy):
    """Test that allgather_strategy is correctly propagated to graph nodes.

    Mirrors test_allreduce_strategy_propagation: when we set an
    allgather_strategy on the ShardingConfig, it must reach the
    trtllm_dist_all_gather node's args at the position the dist op
    expects (2nd positional, immediately after the input tensor —
    strategy is required and intentionally placed early so callers
    can't drop it by accident).

    The test forces dist_backend=TRTLLM because only the TRT-LLM
    allgather op carries a strategy — the torch (demollm) backend op is
    a plain torch.distributed all_gather with signature (tensor, dim=0)
    and intentionally exposes no strategy/symm_mem knobs.
    """

    # Same SimpleMLP as the allreduce variant — keeps the two tests symmetric.
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(128, 256, bias=False)
            self.linear2 = nn.Linear(256, 128, bias=False)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    model = SimpleMLP()
    dummy_input = torch.randn(2, 128)

    gm = torch_export_to_gm(model, (dummy_input,))

    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op

    linear_nodes = [node for node in gm.graph.nodes if is_linear_op(node)]
    assert len(linear_nodes) == 2, f"Expected 2 linear nodes, found {len(linear_nodes)}"
    linear1_node, _ = linear_nodes[0], linear_nodes[1]

    rank, world_size = 0, 4
    # Force the trtllm backend so the emitter takes the strategy-bearing path
    # regardless of whether the test runs under MPI (single-process pytest
    # would otherwise fall back to the torch backend, which is now strategy-free).
    config = ShardingTransformConfig(
        rank=rank,
        world_size=world_size,
        stage="sharding",
        allgather_strategy=AllGatherStrategy[strategy],
        dist_backend=DistBackend.TRTLLM,
    )
    sharding_container = ShardingTransformContainer(config=config)

    # Column shard with all_gather is the only valid emission path for
    # an allgather node (validate() rejects column+all_reduce and
    # row+all_gather).
    sharding_container.add(
        WeightShardingInfo(
            target_node=linear1_node.name,
            config=config,
            split_dim=SplitDimension.COLUMN,
            dist_op="all_gather",
        )
    )

    # Strategy must be visible on the transform itself before it lands in
    # the graph, otherwise propagation can't possibly work below.
    assert len(sharding_container.weight_sharding_transforms) == 1
    for transform in sharding_container.weight_sharding_transforms:
        assert transform.config.allgather_strategy == AllGatherStrategy[strategy], (
            f"Transform {transform.target_node} should have strategy {strategy}, "
            f"got {transform.config.allgather_strategy}"
        )

    for transform in sharding_container.weight_sharding_transforms:
        node = next((n for n in gm.graph.nodes if n.name == transform.target_node), None)
        if node:
            transform.check_and_apply(gm, node)

    recompile(gm)

    # We forced dist_backend=TRTLLM above, so the emitted op must be the
    # TRT-LLM allgather regardless of MPI availability.
    allgather_nodes = [
        node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.trtllm_dist_all_gather)
    ]
    assert len(allgather_nodes) == 1, f"Expected 1 allgather node, found {len(allgather_nodes)}"
    allgather_node = allgather_nodes[0]

    # trtllm_dist_all_gather signature: (tensor, strategy, dim=0, sizes=None, workspace_id=0).
    # The column+all_gather emit path passes (input, strategy_name, -1,
    # None); workspace_id may or may not be explicit, so accept either 4
    # or 5 positional args.
    assert 4 <= len(allgather_node.args) <= 5, (
        f"Expected 4 or 5 args for allgather node, got {len(allgather_node.args)}"
    )
    strategy_arg = allgather_node.args[1]
    assert strategy_arg == strategy, (
        f"Expected allgather strategy '{strategy}', got '{strategy_arg}'"
    )
    assert allgather_node.args[2] == -1, f"Expected gather dim=-1, got {allgather_node.args[2]}"
    assert allgather_node.args[3] is None, f"Expected sizes=None, got {allgather_node.args[3]}"

    print(f"✓ Test passed: allgather_strategy '{strategy}' correctly propagated to graph node")
