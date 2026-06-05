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
from torch.fx import Graph, GraphModule

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
import tensorrt_llm._torch.auto_deploy.transform.library  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.quant_checkpoint_layout import QuantizedCheckpointLayout
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.mxfp4_moe import (
    InsertMXFP4MLP,
    MXFP4MLPConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import _get_dist_ops
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args, is_op, shape

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
        compressor_kv = q.new_empty(bsz, seq_len, 0)
        compressor_gate = q.new_empty(bsz, seq_len, 0)
        compressor_ape = q.new_empty(0, 0)
        compressor_norm_weight = q.new_empty(0)
        cos_table = q.new_empty(0, 0)
        sin_table = q.new_empty(0, 0)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        indexer_q = q.new_empty(bsz, seq_len, 0, 0)
        indexer_weights = q.new_empty(bsz, seq_len, 0)
        indexer_compressor_kv = q.new_empty(bsz, seq_len, 0)
        indexer_compressor_gate = q.new_empty(bsz, seq_len, 0)
        indexer_compressor_ape = q.new_empty(0, 0)
        indexer_compressor_norm_weight = q.new_empty(0)
        attn_output = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
            q,
            kv,
            self.attn_sink,
            topk_idxs,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            cos_table,
            sin_table,
            position_ids,
            indexer_q,
            indexer_weights,
            indexer_compressor_kv,
            indexer_compressor_gate,
            indexer_compressor_ape,
            indexer_compressor_norm_weight,
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
        attn_output = torch.ops.auto_deploy.torch_grouped_linear(
            attn_output,
            wo_a,
            None,
            tp_mode="colwise",
            layer_type="mla",
            tp_min_local_shape=DSV4_O_RANK,
        )
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


def _make_optimizer(
    world_size: int,
    rank: int = 0,
    dist_backend: str | None = None,
    *,
    simple_shard_only: bool = False,
):
    apply_config = {"stage": "sharding"}
    if dist_backend is not None:
        apply_config["dist_backend"] = dist_backend
    if simple_shard_only:
        apply_config["simple_shard_only"] = True

    opt = InferenceOptimizer(
        factory=None,
        config={"apply_sharding_hints": apply_config},
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


def _make_grouped_fp8_quantized_graph(*, num_groups=4, with_group_view=True):
    batch_size, seq_len, rank, group_width = 2, 3, 8, 16

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(num_groups * rank, group_width, dtype=torch.float8_e4m3fn)
            )
            self.register_buffer("weight_scale_inv", torch.ones(num_groups, 1))

    root = Shell()
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(batch_size, seq_len, num_groups, group_width)
    input_node = x
    if with_group_view:
        input_node = graph.call_function(
            torch.ops.auto_deploy.view.default,
            args=(x, [batch_size, seq_len, num_groups, group_width]),
            kwargs={"tp_scaled_dim": 2, "layer_type": "mla"},
        )
        input_node.meta["val"] = torch.empty(batch_size, seq_len, num_groups, group_width)
    weight = graph.get_attr("weight")
    scale = graph.get_attr("weight_scale_inv")
    out = graph.call_function(
        torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear.default,
        args=(input_node, weight, None, [], [scale], [], []),
        kwargs={
            "tp_mode": "colwise",
            "tp_min_local_shape": rank,
            "layer_type": "mla",
            "input_scale_fmt": "ue8m0",
        },
    )
    out.meta["val"] = torch.empty(batch_size, seq_len, num_groups * rank)
    graph.output(out)
    return GraphModule(root, graph), num_groups, rank


def test_apply_hints_grouped_fp8_linear_trusts_group_sharded_view_input():
    gm, num_groups, rank = _make_grouped_fp8_quantized_graph(num_groups=8, with_group_view=True)

    gm_out = _make_optimizer(world_size=8, rank=7)(None, gm)

    grouped_nodes = _call_nodes(
        gm_out,
        torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear.default,
    )
    assert len(grouped_nodes) == 1
    grouped_node = grouped_nodes[0]
    grouped_input = grouped_node.args[0]
    local_groups = num_groups // 8

    assert is_op(grouped_input, torch.ops.auto_deploy.view.default)
    assert not is_op(grouped_input, torch.ops.aten.slice.Tensor)
    assert shape(grouped_input)[2] == local_groups
    assert grouped_input.args[1][2] == -1
    assert gm_out.weight.shape == (local_groups * rank, 16)
    assert gm_out.weight_scale_inv.shape == (local_groups, 1)
    assert grouped_node.meta["val"].shape == (2, 3, local_groups * rank)


def test_apply_hints_grouped_fp8_linear_slices_plain_global_input_groups():
    gm, num_groups, rank = _make_grouped_fp8_quantized_graph(with_group_view=False)

    gm_out = _make_optimizer(world_size=2, rank=1)(None, gm)

    grouped_nodes = _call_nodes(
        gm_out,
        torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear.default,
    )
    assert len(grouped_nodes) == 1
    grouped_node = grouped_nodes[0]
    grouped_input = grouped_node.args[0]
    local_groups = num_groups // 2

    assert is_op(grouped_input, torch.ops.aten.slice.Tensor)
    assert grouped_input.args[1:5] == (2, local_groups, num_groups, 1)
    assert gm_out.weight.shape == (local_groups * rank, 16)
    assert gm_out.weight_scale_inv.shape == (local_groups, 1)
    assert grouped_node.meta["val"].shape == (2, 3, local_groups * rank)


def test_simple_shard_only_does_not_ordinary_shard_grouped_fp8_linear():
    gm, _, _ = _make_grouped_fp8_quantized_graph()

    gm_out = _make_optimizer(world_size=2, rank=1, simple_shard_only=True)(None, gm)

    info = gm_out.meta["_autodeploy"]["transform_history"]["apply_sharding_hints"]
    assert info.num_matches == 0
    assert gm_out.weight.shape == (32, 16)
    assert gm_out.weight_scale_inv.shape == (4, 1)
    assert not _call_nodes(gm_out, torch.ops.auto_deploy.torch_dist_all_gather.default)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_hints_default_dist_backend_uses_auto_selection():
    """Default IR sharding backend remains the existing auto-selected collective."""
    gm, _, _ = _export_hinted_mlp()
    gm_out = _make_optimizer(world_size=2)(None, gm)

    _, expected_all_reduce = _get_dist_ops("auto")
    assert len(_call_nodes(gm_out, expected_all_reduce)) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_hints_torch_dist_backend_forces_torch_all_reduce():
    """Forcing dist_backend='torch' lowers all_reduce placeholders to torch collectives."""
    gm, _, _ = _export_hinted_mlp()
    gm_out = _make_optimizer(world_size=2, dist_backend="torch")(None, gm)

    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.torch_dist_all_reduce.default)) == 1
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.trtllm_dist_all_reduce.default)) == 0


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


_STACKED_MOE_EXPERT_ARG_NAMES = (
    "gate_up_blocks",
    "gate_up_bias",
    "gate_up_scales",
    "down_blocks",
    "down_bias",
    "down_scales",
)


def _make_stacked_test_tensor(shape, dtype=torch.float32):
    numel = 1
    for dim in shape:
        numel *= dim
    values = torch.arange(numel, dtype=torch.float32).reshape(shape)
    if dtype == torch.uint8:
        return values.remainder(251).to(torch.uint8)
    return values.to(dtype)


def _make_stacked_mxfp4_graph(
    base_op, routing_driven=False, include_optionals=False, expert_args_as_attrs=True
):
    num_experts = 4
    hidden_size = 4
    intermediate_size = 3
    root = nn.Module()
    tensors = {
        "selected_experts": torch.tensor([[0], [3]], dtype=torch.int64),
        "routing_weights": torch.ones(2, 1),
        "router_weight": _make_stacked_test_tensor((num_experts, hidden_size)),
        "router_bias": _make_stacked_test_tensor((num_experts,)),
        "gate_up_blocks": _make_stacked_test_tensor(
            (num_experts, 2 * intermediate_size, 1, 16), dtype=torch.uint8
        ),
        "gate_up_bias": _make_stacked_test_tensor((num_experts, 2 * intermediate_size)),
        "gate_up_scales": _make_stacked_test_tensor(
            (num_experts, 2 * intermediate_size, 1), dtype=torch.uint8
        ),
        "down_blocks": _make_stacked_test_tensor(
            (num_experts, hidden_size, 1, 16), dtype=torch.uint8
        ),
        "down_bias": _make_stacked_test_tensor((num_experts, hidden_size)),
        "down_scales": _make_stacked_test_tensor((num_experts, hidden_size, 1), dtype=torch.uint8),
    }
    for name, tensor in tensors.items():
        if name in _STACKED_MOE_EXPERT_ARG_NAMES and not expert_args_as_attrs:
            continue
        root.register_buffer(name, tensor)

    graph = torch.fx.Graph()
    x = _annotate(graph.placeholder("x"), torch.empty(2, hidden_size))
    attrs = {}
    for name, tensor in tensors.items():
        if name in _STACKED_MOE_EXPERT_ARG_NAMES and not expert_args_as_attrs:
            attrs[name] = _annotate(graph.placeholder(name), tensor)
        else:
            attrs[name] = _make_attr(graph, name, tensor)
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


def _ensure_submodule(root, path):
    submod = root
    for name in path.split("."):
        child = getattr(submod, name, None)
        if child is None:
            child = nn.Module()
            submod.add_module(name, child)
        submod = child
    return submod


def _register_nested_buffer(root, target, tensor):
    mod_name, _, attr_name = target.rpartition(".")
    submod = _ensure_submodule(root, mod_name)
    submod.register_buffer(attr_name, tensor)


def _make_deepseek_graph_mxfp4_graph(base_op, layer=3, num_experts=4):
    hidden_size = 32
    intermediate_size = 32
    expert_targets = {
        "gate_up_blocks": f"layers.{layer}.ffn.experts.gate_up_proj_blocks",
        "gate_up_scales": f"layers.{layer}.ffn.experts.gate_up_proj_scales",
        "down_blocks": f"layers.{layer}.ffn.experts.down_proj_blocks",
        "down_scales": f"layers.{layer}.ffn.experts.down_proj_scales",
    }
    tensors = {
        "router_weight": _make_stacked_test_tensor((num_experts, hidden_size)),
        "router_bias": _make_stacked_test_tensor((num_experts,)),
        "gate_up_blocks": _make_stacked_test_tensor(
            (num_experts, 2 * intermediate_size, hidden_size // 32, 16), dtype=torch.uint8
        ),
        "gate_up_bias": _make_stacked_test_tensor((num_experts, 2 * intermediate_size)),
        "gate_up_scales": _make_stacked_test_tensor(
            (num_experts, 2 * intermediate_size, hidden_size // 32), dtype=torch.uint8
        ),
        "down_blocks": _make_stacked_test_tensor(
            (num_experts, hidden_size, intermediate_size // 32, 16), dtype=torch.uint8
        ),
        "down_bias": _make_stacked_test_tensor((num_experts, hidden_size)),
        "down_scales": _make_stacked_test_tensor(
            (num_experts, hidden_size, intermediate_size // 32), dtype=torch.uint8
        ),
    }
    root = nn.Module()
    root.register_buffer("router_weight", tensors["router_weight"])
    root.register_buffer("router_bias", tensors["router_bias"])
    for name, target in expert_targets.items():
        _register_nested_buffer(root, target, tensors[name])
    root.register_buffer("gate_up_bias", tensors["gate_up_bias"])
    root.register_buffer("down_bias", tensors["down_bias"])

    graph = torch.fx.Graph()
    x = _annotate(graph.placeholder("x"), torch.empty(2, hidden_size))
    attrs = {
        "router_weight": _make_attr(graph, "router_weight", tensors["router_weight"]),
        "router_bias": _make_attr(graph, "router_bias", tensors["router_bias"]),
        "gate_up_bias": _make_attr(graph, "gate_up_bias", tensors["gate_up_bias"]),
        "down_bias": _make_attr(graph, "down_bias", tensors["down_bias"]),
    }
    for name, target in expert_targets.items():
        attrs[name] = _make_attr(graph, target, tensors[name])
    moe = graph.call_function(
        base_op,
        args=(
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
        ),
    )
    moe.meta["val"] = torch.empty(2, hidden_size)
    graph.output(moe)
    gm = torch.fx.GraphModule(root, graph)
    gm.graph.lint()
    gm.recompile()
    return gm, expert_targets, hidden_size, intermediate_size


def _make_deepseek_flat_mxfp4_routing_graph(base_op, layer=3, num_experts=4):
    hidden_size = 32
    intermediate_size = 32
    expert_targets = {
        "gate_up_blocks": f"layers_{layer}_ffn_experts_gate_up_proj_blocks",
        "gate_up_scales": f"layers_{layer}_ffn_experts_gate_up_proj_scales",
        "down_blocks": f"layers_{layer}_ffn_experts_down_proj_blocks",
        "down_scales": f"layers_{layer}_ffn_experts_down_proj_scales",
    }
    tensors = {
        "selected_experts": torch.tensor([[0], [3]], dtype=torch.int64),
        "routing_weights": torch.ones(2, 1),
        "gate_up_blocks": _make_stacked_test_tensor(
            (num_experts, 2 * intermediate_size, hidden_size // 32, 16), dtype=torch.uint8
        ),
        "gate_up_bias": _make_stacked_test_tensor((num_experts, 2 * intermediate_size)),
        "gate_up_scales": _make_stacked_test_tensor(
            (num_experts, 2 * intermediate_size, hidden_size // 32), dtype=torch.uint8
        ),
        "down_blocks": _make_stacked_test_tensor(
            (num_experts, hidden_size, intermediate_size // 32, 16), dtype=torch.uint8
        ),
        "down_bias": _make_stacked_test_tensor((num_experts, hidden_size)),
        "down_scales": _make_stacked_test_tensor(
            (num_experts, hidden_size, intermediate_size // 32), dtype=torch.uint8
        ),
    }
    root = nn.Module()
    root.register_buffer("selected_experts", tensors["selected_experts"])
    root.register_buffer("routing_weights", tensors["routing_weights"])
    root.register_buffer("gate_up_bias", tensors["gate_up_bias"])
    root.register_buffer("down_bias", tensors["down_bias"])
    for name, target in expert_targets.items():
        root.register_buffer(target, tensors[name])

    graph = torch.fx.Graph()
    x = _annotate(graph.placeholder("x"), torch.empty(2, hidden_size))
    attrs = {
        "selected_experts": _make_attr(graph, "selected_experts", tensors["selected_experts"]),
        "routing_weights": _make_attr(graph, "routing_weights", tensors["routing_weights"]),
        "gate_up_bias": _make_attr(graph, "gate_up_bias", tensors["gate_up_bias"]),
        "down_bias": _make_attr(graph, "down_bias", tensors["down_bias"]),
    }
    for name, target in expert_targets.items():
        attrs[name] = _make_attr(graph, target, tensors[name])
    moe = graph.call_function(
        base_op,
        args=(
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
            "up_gate",
            "deepseek",
            "moe",
        ),
    )
    moe.meta["val"] = torch.empty(2, hidden_size)
    graph.output(moe)
    gm = torch.fx.GraphModule(root, graph)
    gm.graph.lint()
    gm.recompile()
    return gm, expert_targets, hidden_size, intermediate_size


class _MXFP4CheckpointLayoutFactory:
    def __init__(self, checkpoint_layout):
        self._checkpoint_layout = checkpoint_layout

    def get_quant_config(self):
        return {
            "checkpoint_layout": QuantizedCheckpointLayout(
                checkpoint_consumers=(self._checkpoint_layout,)
            )
        }


def _register_mxfp4_checkpoint_layout_hooks(gm, checkpoint_layout):
    transform = InsertMXFP4MLP(MXFP4MLPConfig(stage=Stages.PATTERN_MATCHER))
    gm, info = transform._apply(
        gm,
        None,
        _MXFP4CheckpointLayoutFactory(checkpoint_layout),
        None,
    )
    assert info.num_matches == 1
    return gm


def _make_deepseek_mxfp4_raw_state(layer, num_experts, hidden_size, intermediate_size):
    def key(expert, projection, kind):
        return f"layers.{layer}.ffn.experts.{expert}.{projection}.{kind}"

    def tensor(shape, offset):
        numel = 1
        for dim in shape:
            numel *= dim
        values = (torch.arange(numel, dtype=torch.int64) + offset) % 251
        return values.to(torch.uint8).reshape(shape)

    state = {}
    for expert in range(num_experts):
        for projection, weight_shape, scale_shape, offset in (
            (
                "w1",
                (intermediate_size, hidden_size // 2),
                (intermediate_size, hidden_size // 32),
                17,
            ),
            (
                "w2",
                (hidden_size, intermediate_size // 2),
                (hidden_size, intermediate_size // 32),
                89,
            ),
            (
                "w3",
                (intermediate_size, hidden_size // 2),
                (intermediate_size, hidden_size // 32),
                151,
            ),
        ):
            state[key(expert, projection, "weight")] = tensor(weight_shape, offset + expert)
            state[key(expert, projection, "scale")] = tensor(scale_shape, offset + 31 + expert)
    return state


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
    assert len(expert_slices) == 0
    expert_args = extract_op_args(ep_nodes[0], *_STACKED_MOE_EXPERT_ARG_NAMES)
    assert all(getattr(arg, "op", None) == "get_attr" for arg in expert_args)
    for name in _STACKED_MOE_EXPERT_ARG_NAMES:
        assert getattr(gm_out, name).shape[0] == 2
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.torch_dist_all_reduce)) == 1


def test_stacked_mxfp4_get_attr_expert_buffers_load_full_checkpoint_slices():
    """Full checkpoint buffers are split by load hooks after physical EP sharding."""
    base_op = _optional_auto_deploy_default("triton_mxfp4_moe")
    if base_op is None or _optional_auto_deploy_default("triton_mxfp4_moe_ep") is None:
        pytest.skip("MXFP4 MoE custom ops are not registered in this environment")

    rank = 1
    world_size = 2
    gm = _make_stacked_mxfp4_graph(base_op)
    full_state = {name: tensor.clone() for name, tensor in gm.state_dict().items()}
    gm_out = _make_optimizer(world_size=world_size, rank=rank)(None, gm)

    for name in _STACKED_MOE_EXPERT_ARG_NAMES:
        getattr(gm_out, name).zero_()

    load_result = gm_out.load_state_dict(
        {name: tensor.clone() for name, tensor in full_state.items()}
    )
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []

    start = 2
    end = 4
    for name in _STACKED_MOE_EXPERT_ARG_NAMES:
        assert torch.equal(getattr(gm_out, name), full_state[name][start:end])


def test_stacked_mxfp4_non_attr_expert_args_keep_runtime_slices():
    """Dynamic expert tensors keep the old aten.slice runtime behavior."""
    base_op = _optional_auto_deploy_default("triton_mxfp4_moe")
    ep_op = _optional_auto_deploy_default("triton_mxfp4_moe_ep")
    if base_op is None or ep_op is None:
        pytest.skip("MXFP4 MoE custom ops are not registered in this environment")

    gm = _make_stacked_mxfp4_graph(base_op, expert_args_as_attrs=False)
    gm_out = _make_optimizer(world_size=2)(None, gm)
    ep_nodes = _call_nodes(gm_out, ep_op)
    assert len(ep_nodes) == 1

    slice_nodes = _call_nodes(gm_out, torch.ops.aten.slice.Tensor)
    expert_slices = [node for node in slice_nodes if node.args[1:5] == (0, 0, 2, 1)]
    assert len(expert_slices) == 6
    expert_args = extract_op_args(ep_nodes[0], *_STACKED_MOE_EXPERT_ARG_NAMES)
    assert all(getattr(arg, "target", None) == torch.ops.aten.slice.Tensor for arg in expert_args)


def test_stacked_mxfp4_rank1_deepseek_graph_pack_loads_high_expert_slice():
    """DeepSeek graph packing emits full buffers before sharding hooks split rank 1."""
    from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
        build_deepseek_v4_packed_mxfp4_experts_layout,
    )

    base_op = _optional_auto_deploy_default("triton_mxfp4_moe")
    if base_op is None or _optional_auto_deploy_default("triton_mxfp4_moe_ep") is None:
        pytest.skip("MXFP4 MoE custom ops are not registered in this environment")

    layer = 3
    rank = 1
    world_size = 2
    num_experts = 4
    gm, expert_targets, hidden_size, intermediate_size = _make_deepseek_graph_mxfp4_graph(
        base_op, layer=layer, num_experts=num_experts
    )
    raw_expert_state = _make_deepseek_mxfp4_raw_state(
        layer, num_experts, hidden_size, intermediate_size
    )
    checkpoint = {
        "router_weight": gm.router_weight.clone(),
        "router_bias": gm.router_bias.clone(),
        "gate_up_bias": gm.gate_up_bias.clone(),
        "down_bias": gm.down_bias.clone(),
        **raw_expert_state,
    }

    layout = build_deepseek_v4_packed_mxfp4_experts_layout()
    expected_full = layout.pack_experts(
        raw_expert_state,
        layer=layer,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )

    gm = _register_mxfp4_checkpoint_layout_hooks(gm, layout)
    gm_out = _make_optimizer(world_size=world_size, rank=rank)(None, gm)
    experts = gm_out.get_submodule(f"layers.{layer}.ffn.experts")
    assert experts.gate_up_proj_blocks.shape[0] == num_experts // world_size

    result = gm_out.load_state_dict(checkpoint)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    lo = num_experts // world_size
    hi = num_experts
    assert torch.equal(
        gm_out.get_buffer(expert_targets["gate_up_blocks"]),
        expected_full.gate_up_blocks[lo:hi],
    )
    assert torch.equal(
        gm_out.get_buffer(expert_targets["gate_up_scales"]),
        expected_full.gate_up_scales[lo:hi],
    )
    assert torch.equal(
        gm_out.get_buffer(expert_targets["down_blocks"]),
        expected_full.down_blocks[lo:hi],
    )
    assert torch.equal(
        gm_out.get_buffer(expert_targets["down_scales"]),
        expected_full.down_scales[lo:hi],
    )


def test_stacked_mxfp4_rank1_deepseek_flat_graph_pack_loads_high_expert_slice():
    """Root FX buffers are packed before sharding hooks split rank 1."""
    from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
        build_deepseek_v4_packed_mxfp4_experts_layout,
    )

    base_op = _optional_auto_deploy_default("torch_mxfp4_moe_from_routing")
    ep_op = _optional_auto_deploy_default("torch_mxfp4_moe_from_routing_ep")
    if base_op is None or ep_op is None:
        pytest.skip("routing-driven MXFP4 MoE custom ops are not registered in this environment")

    layer = 3
    rank = 1
    world_size = 2
    num_experts = 4
    gm, expert_targets, hidden_size, intermediate_size = _make_deepseek_flat_mxfp4_routing_graph(
        base_op, layer=layer, num_experts=num_experts
    )
    raw_expert_state = _make_deepseek_mxfp4_raw_state(
        layer, num_experts, hidden_size, intermediate_size
    )
    layout = build_deepseek_v4_packed_mxfp4_experts_layout()
    expected_full = layout.pack_experts(
        raw_expert_state,
        layer=layer,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )

    gm = _register_mxfp4_checkpoint_layout_hooks(gm, layout)
    checkpoint = {
        "selected_experts": gm.selected_experts.clone(),
        "routing_weights": gm.routing_weights.clone(),
        "gate_up_bias": gm.gate_up_bias.clone(),
        "down_bias": gm.down_bias.clone(),
        **raw_expert_state,
    }

    gm_out = _make_optimizer(world_size=world_size, rank=rank)(None, gm)
    assert gm_out.get_buffer(expert_targets["gate_up_blocks"]).shape[0] == (
        num_experts // world_size
    )

    result = gm_out.load_state_dict(checkpoint)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    lo = num_experts // world_size
    hi = num_experts
    assert torch.equal(
        gm_out.get_buffer(expert_targets["gate_up_blocks"]),
        expected_full.gate_up_blocks[lo:hi],
    )
    assert torch.equal(
        gm_out.get_buffer(expert_targets["gate_up_scales"]),
        expected_full.gate_up_scales[lo:hi],
    )
    assert torch.equal(
        gm_out.get_buffer(expert_targets["down_blocks"]),
        expected_full.down_blocks[lo:hi],
    )
    assert torch.equal(
        gm_out.get_buffer(expert_targets["down_scales"]),
        expected_full.down_scales[lo:hi],
    )


def test_stacked_mxfp4_deepseek_flat_graph_loads_production_shaped_rank7_slice():
    """Root FX buffers load the final EP slice for 256-expert routing-driven graphs."""
    from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
        build_deepseek_v4_packed_mxfp4_experts_layout,
    )

    base_op = _optional_auto_deploy_default("torch_mxfp4_moe_from_routing")
    ep_op = _optional_auto_deploy_default("torch_mxfp4_moe_from_routing_ep")
    if base_op is None or ep_op is None:
        pytest.skip("routing-driven MXFP4 MoE custom ops are not registered in this environment")

    layer = 3
    rank = 7
    world_size = 8
    num_experts = 256
    gm, expert_targets, hidden_size, intermediate_size = _make_deepseek_flat_mxfp4_routing_graph(
        base_op, layer=layer, num_experts=num_experts
    )
    raw_expert_state = _make_deepseek_mxfp4_raw_state(
        layer, num_experts, hidden_size, intermediate_size
    )
    layout = build_deepseek_v4_packed_mxfp4_experts_layout()
    expected_full = layout.pack_experts(
        raw_expert_state,
        layer=layer,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )

    gm = _register_mxfp4_checkpoint_layout_hooks(gm, layout)
    checkpoint = {
        "selected_experts": gm.selected_experts.clone(),
        "routing_weights": gm.routing_weights.clone(),
        "gate_up_bias": gm.gate_up_bias.clone(),
        "down_bias": gm.down_bias.clone(),
        **raw_expert_state,
    }

    gm_out = _make_optimizer(world_size=world_size, rank=rank)(None, gm)
    ep_nodes = _call_nodes(gm_out, ep_op)
    assert len(ep_nodes) == 1
    [expert_start] = extract_op_args(ep_nodes[0], "expert_start")
    assert expert_start == 224

    local_experts = num_experts // world_size
    for target in expert_targets.values():
        local_buffer = gm_out.get_buffer(target)
        assert local_buffer.shape[0] == local_experts
        local_buffer.zero_()

    result = gm_out.load_state_dict(checkpoint)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    lo = expert_start
    hi = num_experts
    assert torch.equal(
        gm_out.get_buffer(expert_targets["gate_up_blocks"]),
        expected_full.gate_up_blocks[lo:hi],
    )
    assert torch.equal(
        gm_out.get_buffer(expert_targets["gate_up_scales"]),
        expected_full.gate_up_scales[lo:hi],
    )
    assert torch.equal(
        gm_out.get_buffer(expert_targets["down_blocks"]),
        expected_full.down_blocks[lo:hi],
    )
    assert torch.equal(
        gm_out.get_buffer(expert_targets["down_scales"]),
        expected_full.down_scales[lo:hi],
    )


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
    assert len(expert_slices) == 0
    expert_args = extract_op_args(ep_nodes[0], *_STACKED_MOE_EXPERT_ARG_NAMES)
    assert all(getattr(arg, "op", None) == "get_attr" for arg in expert_args)
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.torch_dist_all_reduce)) == 1


def test_stacked_mxfp4_routing_driven_rank1_preserves_expert_start_with_torch_backend():
    """Rank 1 routing-driven MXFP4 EP rewrite keeps expert_start and torch all_reduce."""
    base_op = _optional_auto_deploy_default("torch_mxfp4_moe_from_routing")
    ep_op = _optional_auto_deploy_default("torch_mxfp4_moe_from_routing_ep")
    if base_op is None or ep_op is None:
        pytest.skip("routing-driven MXFP4 MoE custom ops are not registered in this environment")

    gm = _make_stacked_mxfp4_graph(base_op, routing_driven=True, include_optionals=True)
    gm_out = _make_optimizer(world_size=2, rank=1, dist_backend="torch")(None, gm)

    ep_nodes = _call_nodes(gm_out, ep_op)
    assert len(ep_nodes) == 1
    [expert_start] = extract_op_args(ep_nodes[0], "expert_start")
    assert expert_start == 2
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.torch_dist_all_reduce.default)) == 1
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.trtllm_dist_all_reduce.default)) == 0
