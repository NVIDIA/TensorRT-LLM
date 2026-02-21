"""Tests for EP sharding transformation."""

from functools import partial

import pytest
import torch
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_sharding_pattern_detection_test, run_test_transformed_gm
from _model_test_utils import MoEOpModel

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
    EPShardingInfo,
    FP8EPShardingInfo,
    NVFP4EPShardingInfo,
    ShardingTransformConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils._graph import lint, recompile
from tensorrt_llm._torch.auto_deploy.utils.mapping_utils import deserialize_mapping
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm.functional import AllReduceStrategy


def _run_ep_shard_job(
    num_experts: int, enable_attention_dp: bool, rank: int, world_size: int
) -> None:
    device = "cuda"
    hidden_size = 32
    intermediate_size = 16
    model = MoEOpModel(
        hidden_size=hidden_size, num_experts=num_experts, intermediate_size=intermediate_size
    ).to(device=device, dtype=torch.bfloat16)
    x = model.get_input(device=device, dtype=torch.bfloat16)

    def _get_expected_num_params(rank: int, world_size: int, num_p_og: int) -> int:
        if world_size <= 1:
            return num_p_og
        # the gate's weight and bias node
        # NOTE:gate layer is also distributed using simple_shard during tp_transform
        n_gate = num_experts * (hidden_size + 1)  # // world_size
        num_experts_per_rank = num_experts // world_size
        if rank == world_size - 1:
            num_experts_per_rank += num_experts % world_size
        expected_expert = num_experts_per_rank * hidden_size * intermediate_size * 3
        return n_gate + expected_expert

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "sharding_dims": ["tp", "ep"],
                "enable_attention_dp": enable_attention_dp,
            },
            "sharding_transform_executor": {
                "stage": "sharding",
            },
        },
    )(None, gm)
    if enable_attention_dp:
        # attention-dp + MoE EP-only all-to-all does not insert explicit
        # communication nodes. Instead, trtllm_moe uses MoEAlltoAll internally.
        # Instead, check if input argument to MoE node, `enable_alltoall` is true
        def transform_check(gm):
            # get MoE node
            moe_nodes = [n for n in gm.graph.nodes if is_op(n, torch.ops.auto_deploy.torch_moe)]
            assert len(moe_nodes) == 1, f"Expected 1 MoE node, got {len(moe_nodes)}"
            moe_node = moe_nodes[0]
            print(f"\nMoE node {moe_node.name}, kwargs: {moe_node.kwargs}\n\n")
            assert "mapping_config" in moe_node.kwargs, (
                f"Mapping config not found in MoE node {moe_node.name}"
            )
            # deserialize the mapping config string to dict
            mapping_config = deserialize_mapping(moe_node.kwargs["mapping_config"])
            return mapping_config.enable_attention_dp
    else:

        def transform_check(gm):
            return any(
                is_op(n, torch.ops.auto_deploy.torch_dist_all_reduce) for n in gm.graph.nodes
            ) == (world_size > 1)

    # NOTE: we need to skip output assert when enable_attention_dp is True because
    # a single rank does not contain the full output (lack of final collective),
    # so the output will be inherently incomplete w.r.t to the single node.
    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        check_transformed_graph=transform_check,
        _get_expected_num_params=partial(_get_expected_num_params, rank, world_size),
        test_load_hook=False,
        skip_output_assert=enable_attention_dp,
    )


def _run_pattern_detection_job(num_experts: int, rank: int, world_size: int) -> None:
    device = "cuda"
    hidden_size = 32
    intermediate_size = 16
    model = MoEOpModel(
        hidden_size=hidden_size, num_experts=num_experts, intermediate_size=intermediate_size
    ).to(device=device, dtype=torch.bfloat16)
    x = model.get_input(device=device, dtype=torch.bfloat16)

    # Test pattern detection - create expected transformations for validation
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    expected_transformations = []
    # if world_size == 1, no sharding transformations should be detected
    if world_size > 1:
        config = ShardingTransformConfig(
            rank=rank,
            world_size=world_size,
            stage="sharding",
            allreduce_strategy=AllReduceStrategy.AUTO,
            dist_backend="auto",
        )
        for node in gm.graph.nodes:
            if is_op(node, torch.ops.auto_deploy.torch_moe):
                expected_transformations.append(
                    EPShardingInfo(
                        target_node=node.name,
                        config=config,
                    )
                )
            elif is_op(node, torch.ops.auto_deploy.torch_quant_fp8_moe):
                expected_transformations.append(
                    FP8EPShardingInfo(
                        target_node=node.name,
                        config=config,
                    )
                )
            elif is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_moe):
                expected_transformations.append(
                    NVFP4EPShardingInfo(
                        target_node=node.name,
                        config=config,
                    )
                )

    # get detected transformations
    optimizer = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "sharding_scope": ["ep"],
            },
        },
    )
    optimizer.shared_config.local_rank = rank
    optimizer.shared_config.world_size = world_size
    _ = optimizer(None, gm)
    detected_transformations = optimizer.shared_config.sharding_transform_container.ep_transforms

    # Run pattern detection test
    run_sharding_pattern_detection_test(detected_transformations, expected_transformations)


@pytest.mark.parametrize("device_count", get_device_counts([2, 8]))
@pytest.mark.parametrize("num_experts", [3, 8])
@pytest.mark.parametrize("enable_attention_dp", [True, False])
def test_ep_shard(device_count: int, num_experts: int, enable_attention_dp: bool):
    if device_count > num_experts:
        pytest.skip(f"world_size {device_count} > num_experts {num_experts}")
    dist_common.spawn_multiprocess_job(
        job=partial(_run_ep_shard_job, num_experts, enable_attention_dp),
        size=device_count,
    )


# Note: EP pattern detection is invariant to enable_attention_dp:
# it is only relevant inside the executor while inserting the right ops.
@pytest.mark.parametrize("world_size", [1, 8])
@pytest.mark.parametrize("num_experts", [3, 8])
def test_sharding_pattern_detection(world_size: int, num_experts: int):
    """Test pattern detection logic without distributed execution.

    This test verifies only the pattern detection logic with provided world_size.
    No need to run distributed job, can be run on single process.
    """
    _run_pattern_detection_job(
        num_experts=num_experts,
        rank=0,
        world_size=world_size,
    )


def test_llama4_stacked_moe_pattern_detection():
    """Minimal test: verify torch_moe with stacked format is detected for EP sharding."""
    # Create a simple graph with torch_moe node using stacked tensor format
    gm = torch.fx.GraphModule({}, torch.fx.Graph())
    graph = gm.graph

    with graph.inserting_after():
        x = graph.placeholder("x")
        selected_experts = graph.placeholder("selected_experts")
        routing_weights = graph.placeholder("routing_weights")
        w3_w1 = graph.placeholder("w3_w1_stacked")
        w2 = graph.placeholder("w2_stacked")

        # Create single-element lists for stacked tensor format
        w1_list = graph.call_function(list, args=([w3_w1],))
        w2_list = graph.call_function(list, args=([w2],))
        w3_list = graph.call_function(list, args=([],))

        moe_node = graph.call_function(
            torch.ops.auto_deploy.torch_moe,
            args=(x, selected_experts, routing_weights, w1_list, w2_list, w3_list),
            kwargs={"is_gated_mlp": True, "apply_routing_on_input": True},
        )
        graph.output(moe_node)

    lint(gm)
    recompile(gm)

    # Run pattern detection for EP
    optimizer = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "sharding_scope": ["ep"],
            },
        },
    )
    optimizer.shared_config.local_rank = 0
    optimizer.shared_config.world_size = 2
    _ = optimizer(None, gm)

    # Verify torch_moe with stacked format is detected for EP sharding
    detected = optimizer.shared_config.sharding_transform_container.ep_transforms
    assert len(detected) == 1, f"Expected 1 EP transform, got {len(detected)}"
    assert detected[0].target_node == moe_node.name


class PreStackedMoEModel(torch.nn.Module):
    """Minimal model using torch_moe_fused with stacked expert tensors."""

    def __init__(self, hidden_size=32, intermediate_size=16, num_experts=4, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.gate = torch.nn.Linear(hidden_size, num_experts, bias=False)
        self.gate_up_proj = torch.nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size)
        )

    def forward(self, x):
        router_logits = self.gate(x)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(x.dtype)
        return torch.ops.auto_deploy.torch_moe_fused(
            x,
            selected_experts,
            routing_weights,
            self.gate_up_proj,
            self.down_proj,
        )

    def get_input(self, device, dtype=torch.bfloat16):
        return torch.randn(2, self.hidden_size, device=device, dtype=dtype)


def test_split_moe_fused_then_ep_sharding_executor():
    """Verify pre-sharding split enables EP path for torch_moe_fused models."""
    device = "cuda"
    model = PreStackedMoEModel().to(device=device, dtype=torch.bfloat16)
    x = model.get_input(device=device, dtype=torch.bfloat16)
    gm = torch_export_to_gm(model, args=(x,), clone=True)

    optimizer = InferenceOptimizer(
        None,
        {
            "split_moe_fused_for_sharding": {
                "stage": "pattern_matcher",
            },
            "detect_sharding": {
                "stage": "sharding",
                "use_sharding_from_factory": False,
                "sharding_dims": ["ep"],
            },
            "sharding_transform_executor": {
                "stage": "sharding",
            },
        },
    )
    optimizer.shared_config.local_rank = 0
    optimizer.shared_config.world_size = 2
    gm_transformed = optimizer(None, gm)

    has_torch_moe = any(
        is_op(n, torch.ops.auto_deploy.torch_moe) for n in gm_transformed.graph.nodes
    )
    has_torch_moe_fused = any(
        is_op(n, torch.ops.auto_deploy.torch_moe_fused) for n in gm_transformed.graph.nodes
    )
    has_allreduce = any(
        is_op(n, torch.ops.auto_deploy.torch_dist_all_reduce) for n in gm_transformed.graph.nodes
    )

    assert has_torch_moe, "Expected converted torch_moe node for EP sharding"
    assert not has_torch_moe_fused, "torch_moe_fused should be converted before EP sharding"
    assert has_allreduce, "Expected EP sharding to insert torch_dist_all_reduce"
