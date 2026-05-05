# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for ``apply_sharding_hints`` (hint-driven TP sharding).

``test_sharding`` — multi-GPU end-to-end: exports, transforms, and validates
    output correctness on real GPUs via ``run_test_transformed_gm``.
``test_apply_hints`` — single-process transform check: verifies graph
    rewriting (weight shapes, all_reduce replacement, skip conditions)
    without distributed execution.
"""

import pytest
import torch
import torch.nn as nn
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_test_transformed_gm

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
import tensorrt_llm._torch.auto_deploy.transform.library  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args, is_op

pytestmark = pytest.mark.threadleak(enabled=False)

FEATURES, HIDDEN = 32, 64

DSV4_HIDDEN = 8
DSV4_NUM_HEADS = 4
DSV4_HEAD_DIM = 2
DSV4_GROUPS = 2
DSV4_GROUP_WIDTH = DSV4_NUM_HEADS * DSV4_HEAD_DIM // DSV4_GROUPS
DSV4_O_RANK = 3


class HintedMLP(nn.Module):
    def __init__(self, features=FEATURES, hidden=HIDDEN):
        super().__init__()
        self.up = nn.Linear(features, hidden, bias=False)
        self.down = nn.Linear(hidden, features, bias=False)

    def forward(self, x):
        h = torch.ops.auto_deploy.torch_linear_simple(x, self.up.weight, None, tp_mode="colwise")
        h = torch.relu(h)
        h = torch.ops.auto_deploy.torch_linear_simple(h, self.down.weight, None, tp_mode="rowwise")
        h = torch.ops.auto_deploy.all_reduce(h)
        return h


class DeepSeekV4IRContractBlock(nn.Module):
    """Small DeepSeek V4-shaped graph for sharding contract checks."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(DSV4_HIDDEN, DSV4_NUM_HEADS * DSV4_HEAD_DIM, bias=False)
        self.kv_proj = nn.Linear(DSV4_HIDDEN, DSV4_HEAD_DIM, bias=False)
        self.wo_a = nn.Linear(DSV4_GROUP_WIDTH, DSV4_GROUPS * DSV4_O_RANK, bias=False)
        self.wo_b = nn.Linear(DSV4_GROUPS * DSV4_O_RANK, DSV4_HIDDEN, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(DSV4_NUM_HEADS, dtype=torch.float32))

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        q = torch.ops.auto_deploy.torch_linear_simple(
            x,
            self.q_proj.weight,
            None,
            tp_mode="colwise",
            tp_min_local_shape=DSV4_HEAD_DIM,
            layer_type="mla",
        )
        q_left, q_right = torch.ops.auto_deploy.split_with_sizes(
            q,
            [DSV4_GROUP_WIDTH, DSV4_GROUP_WIDTH],
            dim=-1,
            enable_sharding=True,
            layer_type="mla",
        )
        q = torch.cat((q_left, q_right), dim=-1)
        q = torch.ops.auto_deploy.view(
            q,
            [bsz, seq_len, DSV4_NUM_HEADS, DSV4_HEAD_DIM],
            tp_scaled_dim=2,
            layer_type="mla",
        )

        kv = torch.ops.auto_deploy.torch_linear_simple(x, self.kv_proj.weight, None)
        topk_idxs = torch.zeros((bsz, seq_len, 1), dtype=torch.int64, device=x.device)
        attn_output = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
            q,
            kv,
            self.attn_sink,
            topk_idxs,
            1.0,
            enable_sharding=True,
            layer_type="mla",
        )
        attn_output = torch.ops.auto_deploy.view(
            attn_output,
            [bsz, seq_len, DSV4_GROUPS, DSV4_GROUP_WIDTH],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        wo_a = torch.ops.auto_deploy.view(
            self.wo_a.weight,
            [DSV4_GROUPS, DSV4_O_RANK, DSV4_GROUP_WIDTH],
            tp_scaled_dim=0,
            layer_type="mla",
            tp_min_local_shape=DSV4_O_RANK,
        )
        attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a).flatten(2)
        attn_output = torch.ops.auto_deploy.torch_linear_simple(
            attn_output,
            self.wo_b.weight,
            None,
            tp_mode="rowwise",
            layer_type="mla",
        )
        return torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mla")


# ---------------------------------------------------------------------------
# test_sharding — multi-GPU end-to-end (follows test_tp_sharding.py pattern)
# ---------------------------------------------------------------------------


def _run_sharding_job(rank: int, world_size: int) -> None:
    model = HintedMLP().to(device="cuda", dtype=torch.float16)
    x = torch.randn(4, 8, FEATURES, device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(None, {"apply_sharding_hints": {"stage": "sharding"}})(
        None, gm
    )

    op_ar = torch.ops.auto_deploy.torch_dist_all_reduce

    def check_transformed_graph(gm_mod) -> bool:
        has_dist = any(is_op(n, op_ar) for n in gm_mod.graph.nodes)
        return has_dist == (world_size > 1)

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        check_transformed_graph=check_transformed_graph,
        _get_expected_num_params=lambda n: n // world_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("world_size", get_device_counts([2]))
def test_sharding(world_size: int):
    """Hint-based colwise/rowwise + all_reduce: end-to-end on real GPUs."""
    dist_common.spawn_multiprocess_job(job=_run_sharding_job, size=world_size)


# ---------------------------------------------------------------------------
# test_apply_hints — single-process transform checks (no distributed exec)
# ---------------------------------------------------------------------------


def _make_optimizer(world_size: int, rank: int = 0):
    opt = InferenceOptimizer(
        factory=None,
        config={"apply_sharding_hints": {"stage": "sharding"}},
    )
    opt.shared_config = SharedConfig(
        local_rank=rank,
        world_size=world_size,
        dist_config=DistConfig(
            world_size=world_size, rank=rank, tp_size=world_size, moe_ep_size=world_size
        ),
    )
    return opt


def _export_hinted_mlp():
    model = HintedMLP().cuda()
    x = torch.randn(2, FEATURES, device="cuda")
    return torch_export_to_gm(model, args=(x,), clone=True), model, x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "world_size, expect_skipped, expect_up_shape, expect_down_shape",
    [
        (1, True, (HIDDEN, FEATURES), (FEATURES, HIDDEN)),
        (2, False, (HIDDEN // 2, FEATURES), (FEATURES, HIDDEN // 2)),
    ],
)
def test_apply_hints(world_size, expect_skipped, expect_up_shape, expect_down_shape):
    """Verify graph rewriting without distributed execution."""
    gm, _, _ = _export_hinted_mlp()
    gm_out = _make_optimizer(world_size)(None, gm)

    info = gm_out.meta["_autodeploy"]["transform_history"]["apply_sharding_hints"]
    assert info.skipped is expect_skipped

    assert gm_out.up.weight.shape == expect_up_shape
    assert gm_out.down.weight.shape == expect_down_shape

    has_dist_ar = any(
        is_op(n, torch.ops.auto_deploy.torch_dist_all_reduce.default) for n in gm_out.graph.nodes
    )
    assert has_dist_ar == (not expect_skipped)


def _export_deepseek_v4_contract_block():
    model = DeepSeekV4IRContractBlock()
    x = torch.randn(2, 3, DSV4_HIDDEN)
    return torch_export_to_gm(model, args=(x,), clone=True)


def _call_nodes(gm, op):
    return [node for node in gm.graph.nodes if is_op(node, op)]


def test_deepseek_v4_ir_contract_linear_view_sparse_attention():
    """DeepSeek V4-shaped graph keeps TP, view, split, sink, and collective contracts."""
    gm = _export_deepseek_v4_contract_block()
    gm_out = _make_optimizer(world_size=2)(None, gm)

    assert gm_out.q_proj.weight.shape == (DSV4_NUM_HEADS * DSV4_HEAD_DIM // 2, DSV4_HIDDEN)
    assert gm_out.wo_a.weight.shape == (DSV4_O_RANK, DSV4_GROUP_WIDTH)
    assert gm_out.wo_b.weight.shape == (DSV4_HIDDEN, DSV4_GROUPS * DSV4_O_RANK // 2)
    assert gm_out.attn_sink.shape == (DSV4_NUM_HEADS // 2,)

    split_nodes = _call_nodes(gm_out, torch.ops.auto_deploy.split_with_sizes)
    assert len(split_nodes) == 1
    [split_sizes] = extract_op_args(split_nodes[0], "split_sizes")
    assert split_sizes == [DSV4_GROUP_WIDTH // 2, DSV4_GROUP_WIDTH // 2]

    view_shapes = [
        extract_op_args(node, "shape")[0]
        for node in _call_nodes(gm_out, torch.ops.auto_deploy.view)
    ]
    assert [2, 3, -1, DSV4_HEAD_DIM] in view_shapes
    assert [2, 3, -1, DSV4_GROUP_WIDTH] in view_shapes
    assert [-1, DSV4_O_RANK, DSV4_GROUP_WIDTH] in view_shapes

    sparse_nodes = _call_nodes(gm_out, torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention)
    assert len(sparse_nodes) == 1
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.torch_dist_all_reduce)) == 1


def _annotate(node, value):
    node.meta["val"] = value
    return node


def _make_attr(graph, name, value):
    node = graph.get_attr(name)
    node.meta["val"] = value
    return node


def _make_list_moe_graph():
    hidden_size = 4
    intermediate_size = 6
    num_experts = 4
    root = nn.Module()
    for expert_idx in range(num_experts):
        root.register_parameter(
            f"w1_{expert_idx}",
            nn.Parameter(torch.randn(intermediate_size, hidden_size)),
        )
        root.register_parameter(
            f"w2_{expert_idx}",
            nn.Parameter(torch.randn(hidden_size, intermediate_size)),
        )
        root.register_parameter(
            f"w3_{expert_idx}",
            nn.Parameter(torch.randn(intermediate_size, hidden_size)),
        )

    graph = torch.fx.Graph()
    x = _annotate(graph.placeholder("x"), torch.empty(2, hidden_size))
    selected_experts = _annotate(
        graph.placeholder("selected_experts"), torch.empty(2, 1, dtype=torch.int64)
    )
    routing_weights = _annotate(graph.placeholder("routing_weights"), torch.empty(2, 1))

    w1 = [_make_attr(graph, f"w1_{i}", getattr(root, f"w1_{i}")) for i in range(num_experts)]
    w2 = [_make_attr(graph, f"w2_{i}", getattr(root, f"w2_{i}")) for i in range(num_experts)]
    w3 = [_make_attr(graph, f"w3_{i}", getattr(root, f"w3_{i}")) for i in range(num_experts)]
    moe = graph.call_function(
        torch.ops.auto_deploy.torch_moe.default,
        args=(x, selected_experts, routing_weights, w1, w2, w3),
        kwargs={"layer_type": "moe"},
    )
    moe.meta["val"] = torch.empty(2, hidden_size)
    graph.output(moe)
    gm = torch.fx.GraphModule(root, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def test_list_moe_ir_contract_inserts_all_reduce_for_ep():
    """List-based MoE EP sharding localizes experts and adds a graph collective."""
    gm = _make_list_moe_graph()
    gm_out = _make_optimizer(world_size=2)(None, gm)
    moe_nodes = _call_nodes(gm_out, torch.ops.auto_deploy.torch_moe)
    assert len(moe_nodes) == 1
    [w1_weight, w2_weight, w3_weight] = extract_op_args(
        moe_nodes[0], "w1_weight", "w2_weight", "w3_weight"
    )
    assert len(w1_weight) == 2
    assert len(w2_weight) == 2
    assert len(w3_weight) == 2
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.torch_dist_all_reduce)) == 1


def _optional_auto_deploy_default(name):
    try:
        return getattr(torch.ops.auto_deploy, name).default
    except AttributeError:
        return None


def _make_stacked_mxfp4_graph(base_op, routing_driven=False, include_optionals=False):
    num_experts = 4
    hidden_size = 4
    intermediate_size = 3
    root = nn.Module()
    tensors = {
        "selected_experts": torch.tensor([[0], [3]], dtype=torch.int64),
        "routing_weights": torch.ones(2, 1),
        "router_weight": torch.randn(num_experts, hidden_size),
        "router_bias": torch.randn(num_experts),
        "gate_up_blocks": torch.empty(num_experts, 2 * intermediate_size, 1, 16, dtype=torch.uint8),
        "gate_up_bias": torch.randn(num_experts, 2 * intermediate_size),
        "gate_up_scales": torch.empty(num_experts, 2 * intermediate_size, 1, dtype=torch.uint8),
        "down_blocks": torch.empty(num_experts, hidden_size, 1, 16, dtype=torch.uint8),
        "down_bias": torch.randn(num_experts, hidden_size),
        "down_scales": torch.empty(num_experts, hidden_size, 1, dtype=torch.uint8),
    }
    for name, tensor in tensors.items():
        root.register_buffer(name, tensor)

    graph = torch.fx.Graph()
    x = _annotate(graph.placeholder("x"), torch.empty(2, hidden_size))
    attrs = {name: _make_attr(graph, name, tensor) for name, tensor in tensors.items()}
    if routing_driven:
        args = (
            x,
            attrs["selected_experts"],
            attrs["routing_weights"],
            attrs["gate_up_blocks"],
            attrs["gate_up_bias"],
            attrs["gate_up_scales"],
            1.0,
            10.0,
            attrs["down_blocks"],
            attrs["down_bias"],
            attrs["down_scales"],
        )
        if include_optionals:
            args = args + ("gate_up", "gpt_oss", "moe")
    else:
        args = (
            x,
            attrs["router_weight"],
            attrs["router_bias"],
            1,
            attrs["gate_up_blocks"],
            attrs["gate_up_bias"],
            attrs["gate_up_scales"],
            1.0,
            10.0,
            attrs["down_blocks"],
            attrs["down_bias"],
            attrs["down_scales"],
        )
        if include_optionals:
            args = args + ("moe",)
    moe = graph.call_function(
        base_op,
        args=args,
    )
    moe.meta["val"] = torch.empty(2, hidden_size)
    graph.output(moe)
    gm = torch.fx.GraphModule(root, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def test_stacked_mxfp4_ir_contract_rewrites_to_matching_ep_variant():
    """Stacked MXFP4 MoE sharding is schema-based and rewrites to the matching EP op."""
    base_op = _optional_auto_deploy_default("triton_mxfp4_moe")
    ep_op = _optional_auto_deploy_default("triton_mxfp4_moe_ep")
    if base_op is None or ep_op is None:
        pytest.skip("MXFP4 MoE custom ops are not registered in this environment")

    gm = _make_stacked_mxfp4_graph(base_op)
    gm_out = _make_optimizer(world_size=2)(None, gm)
    ep_nodes = _call_nodes(gm_out, ep_op)
    assert len(ep_nodes) == 1
    [ep_size, ep_rank] = extract_op_args(ep_nodes[0], "ep_size", "ep_rank")
    assert ep_size == 2
    assert ep_rank == 0

    slice_nodes = _call_nodes(gm_out, torch.ops.aten.slice.Tensor)
    expert_slices = [node for node in slice_nodes if node.args[1:5] == (0, 0, 2, 1)]
    assert len(expert_slices) == 6
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.torch_dist_all_reduce)) == 1


def test_stacked_mxfp4_routing_driven_ir_contract_rewrites_to_matching_ep_variant():
    """Routing-driven stacked MXFP4 EP op takes expert_start instead of ep_size/ep_rank."""
    base_op = _optional_auto_deploy_default("torch_mxfp4_moe_from_routing")
    ep_op = _optional_auto_deploy_default("torch_mxfp4_moe_from_routing_ep")
    if base_op is None or ep_op is None:
        pytest.skip("routing-driven MXFP4 MoE custom ops are not registered in this environment")

    gm = _make_stacked_mxfp4_graph(base_op, routing_driven=True, include_optionals=True)
    gm_out = _make_optimizer(world_size=2)(None, gm)
    ep_nodes = _call_nodes(gm_out, ep_op)
    assert len(ep_nodes) == 1
    [
        selected_experts,
        routing_weights,
        expert_start,
        ep_size,
        ep_rank,
        gate_up_order,
        swiglu_mode,
        layer_type,
    ] = extract_op_args(
        ep_nodes[0],
        "selected_experts",
        "routing_weights",
        "expert_start",
        "ep_size",
        "ep_rank",
        "gate_up_order",
        "swiglu_mode",
        "layer_type",
    )
    assert selected_experts is not None
    assert routing_weights is not None
    assert expert_start == 0
    assert ep_size is None
    assert ep_rank is None
    assert gate_up_order == "gate_up"
    assert swiglu_mode == "gpt_oss"
    assert layer_type == "moe"

    slice_nodes = _call_nodes(gm_out, torch.ops.aten.slice.Tensor)
    expert_slices = [node for node in slice_nodes if node.args[1:5] == (0, 0, 2, 1)]
    assert len(expert_slices) == 6
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.torch_dist_all_reduce)) == 1
