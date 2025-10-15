"""Tests for basic graph sharding."""

from functools import partial
from typing import Type

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_sharding_pattern_detection_test, run_test_transformed_gm
from _model_test_utils import FakeFP8Linear

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
    SplitDimension,
    TPShardingInfo,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op, is_op
from tensorrt_llm._torch.auto_deploy.utils.sharding_utils import FP8TPShardingInfo

base_model_tp_plan = {
    "q_proj": "colwise",
    "k_proj": "colwise",
    "v_proj": "colwise",
    "o_proj": "rowwise",
    "gate_proj": "colwise",
    "up_proj": "colwise",
    "down_proj": "rowwise",
    "linear1": "colwise",
    "linear2": "rowwise",
    "linear": "gather",
    # "input_layernorm.weight": "sequence_parallel",
    # "post_attention_layernorm.weight": "sequence_parallel",
    # "norm.weight": "sequence_parallel",
    # "shared_expert.gate_proj": "local_colwise",
    # "shared_expert.up_proj": "local_colwise",
    # "shared_expert.down_proj": "local_rowwise",
    # "experts.gate_up_proj": "local_packed_rowwise",
    # "experts.down_proj": "local_colwise",
    # "experts": "local",
    "feed_forward": "gather",
    "self": "gather",
    "weight": "gather",
}

predefined_config = {
    "head_dim": 8,
    "tp_plan": base_model_tp_plan,
}


class GQA_Block(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        num_key_value_heads: int,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.is_gqa = num_key_value_heads < num_attention_heads
        assert self.hidden_size == self.num_attention_heads * self.head_dim

        # key, query, value, out projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=False,
        )

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, -1, self.head_dim)
        k = self.k_proj(x).view(b, s, -1, self.head_dim)
        v = self.v_proj(x).view(b, s, -1, self.head_dim)

        y = torch.ops.auto_deploy.torch_attention(q, k, v, is_causal=True, layout="bsnd")
        y = y.contiguous().view(b, s, -1)

        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = nn.Linear(in_features, 4 * in_features, bias=bias)
        self.linear2 = nn.Linear(4 * in_features, out_features, bias=bias)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        return self.linear2(y)


class FP8MLP(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = FakeFP8Linear(in_features, 4 * in_features, bias=bias)
        self.linear2 = FakeFP8Linear(4 * in_features, out_features, bias=bias)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        return self.linear2(y)


def _run_job(
    model_cls: nn.Module,
    dist_op_expected: str,
    bias: bool,
    from_config: bool,
    rank: int,
    world_size: int,
) -> None:
    # init model and input
    batch_size = 4
    sequence_len = 8
    num_features = 32

    # GQA specific parameters
    num_heads = 4
    num_key_value_heads = 1

    if model_cls == GQA_Block:
        model = model_cls(
            num_attention_heads=num_heads,
            hidden_size=num_features,
            num_key_value_heads=num_key_value_heads,
        ).to(device="cuda", dtype=torch.float16)
    elif model_cls == FP8MLP:
        model = model_cls(num_features, num_features, bias=bias).to("cuda")
    else:
        model = model_cls(num_features, num_features, bias=bias).to(
            device="cuda", dtype=torch.float16
        )
    x = torch.randn(batch_size, sequence_len, num_features, device="cuda", dtype=torch.float16)

    if model_cls == GQA_Block:
        head_dim = num_features // num_heads
        min_local_size = head_dim
    else:
        min_local_size = 1

    def _get_expected_num_params(num_p_og: int) -> int:
        num_update = 0
        if bias and dist_op_expected == "torch_dist_all_reduce":
            num_p_og -= num_features
            num_update = num_features * (rank == world_size - 1)

        if min_local_size > 1:
            # it means we are in the GQA. W_kv are partially replicated, we need to count
            # the number of parameters manually.
            W_q_local_size = num_features * num_features // world_size
            W_o_local_size = W_q_local_size
            W_k_local_size = num_features * head_dim * max(num_key_value_heads // world_size, 1)
            W_v_local_size = W_k_local_size
            num_params = W_q_local_size + W_k_local_size + W_v_local_size + W_o_local_size
        else:
            num_params = num_p_og // world_size + num_update
        return num_params

    def verify_local_weight_sizes(gm) -> bool:
        """Verify that all weight tensors have first dimension >= min_local_size after sharding."""
        for name, param in gm.named_parameters():
            # Only check parameters that have at least 1 dimension and are weight matrices
            if param.dim() >= 1 and "weight" in name:
                if param.shape[0] < min_local_size:
                    print(
                        f"Weight {name} has shape {param.shape}, dim {param.shape[0]} < min_local_size {min_local_size}"
                    )
                    return False
        return True

    # now run the test
    op_expected = getattr(torch.ops.auto_deploy, dist_op_expected)

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "use_sharding_from_factory": from_config,
            },
            "sharding_transform_executor": {
                "stage": "sharding",
            },
        },
    )(None, gm)

    def combined_graph_check(gm) -> bool:
        # Check for expected distributed operations
        has_expected_dist_ops = any(is_op(n, op_expected) for n in gm.graph.nodes) == (
            world_size > 1
        )
        # Check weight size constraints
        weight_sizes_valid = verify_local_weight_sizes(gm)
        return has_expected_dist_ops and weight_sizes_valid

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        check_transformed_graph=combined_graph_check,
        _get_expected_num_params=_get_expected_num_params,
    )


def _run_pattern_detection_job(
    model_cls: nn.Module,
    bias: bool,
    rank: int,
    world_size: int,
    from_config: bool,
) -> None:
    # init model and input
    batch_size = 4
    sequence_len = 8
    num_features = 32

    # GQA specific parameters
    num_heads = 4
    num_key_value_heads = 1

    if model_cls == GQA_Block:
        model = model_cls(
            num_attention_heads=num_heads,
            hidden_size=num_features,
            num_key_value_heads=num_key_value_heads,
        ).to(device="cuda", dtype=torch.float16)
    else:
        model = model_cls(num_features, num_features, bias=bias).to(
            device="cuda", dtype=torch.float16
        )
    x = torch.randn(batch_size, sequence_len, num_features, device="cuda", dtype=torch.float16)

    # Test pattern detection - create expected transformations for validation
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    expected_transformations = []
    # if world_size == 1, no sharding transformations should be detected
    if world_size > 1:
        if model_cls == GQA_Block:
            min_local_shape = num_features // num_heads
            for node in gm.graph.nodes:
                if is_linear_op(node):
                    # for Q, K, V layers, we expect:
                    # dim = 0, add_dist = False
                    # for O layer, we expect:
                    # dim = 1, add_dist = True
                    if "o_proj" in node.args[1].name:
                        dim = SplitDimension.ROW
                        dist_op = "all_reduce"
                    else:
                        dim = SplitDimension.COLUMN
                        dist_op = None
                    expected_transformations.append(
                        TPShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            rank=rank,
                            world_size=world_size,
                            dist_op=dist_op,
                            min_local_shape=min_local_shape,
                        )
                    )
        elif model_cls == MLP:
            for node in gm.graph.nodes:
                if is_linear_op(node):
                    # linear1 should be sharded on dim=0, add_dist=False, min_local_shape=1
                    # linear2 should be sharded on dim=1, add_dist=True, min_local_shape=1
                    if "linear1" in node.args[1].name:
                        dim = SplitDimension.COLUMN
                        dist_op = None
                    else:
                        dim = SplitDimension.ROW
                        dist_op = "all_reduce"
                    expected_transformations.append(
                        TPShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            rank=rank,
                            world_size=world_size,
                            dist_op=dist_op,
                            min_local_shape=1,
                        )
                    )
        elif model_cls == nn.Linear:
            # expect simple shard only (dim=0, add_dist=True, min_local_shape=1)
            for node in gm.graph.nodes:
                if is_linear_op(node):
                    expected_transformations.append(
                        TPShardingInfo(
                            target_node=node.name,
                            split_dim=SplitDimension.COLUMN,  # Simple shard uses dim=0
                            rank=rank,
                            world_size=world_size,
                            dist_op="all_gather",
                            min_local_shape=1,
                        )
                    )
        elif model_cls == FP8MLP:
            for node in gm.graph.nodes:
                if is_op(node, torch.ops.auto_deploy.torch_fake_quant_fp8_linear):
                    # linear1 should be sharded on dim=0, add_dist=False, min_local_shape=1
                    # linear2 should be sharded on dim=1, add_dist=True, min_local_shape=1
                    if "linear1" in node.args[1].name:
                        dim = SplitDimension.COLUMN
                        dist_op = None
                    else:
                        dim = SplitDimension.ROW
                        dist_op = "all_reduce"
                    expected_transformations.append(
                        FP8TPShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            rank=rank,
                            world_size=world_size,
                            dist_op=dist_op,
                            min_local_shape=1,
                        )
                    )

    # get detected transformations
    optimizer = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "use_sharding_from_factory": from_config,
            },
        },
    )
    optimizer.shared_config.local_rank = rank
    optimizer.shared_config.world_size = world_size
    _ = optimizer(None, gm)
    detected_transformations = optimizer.shared_config.sharding_config.tp_transforms

    print(f"detected_transformations: {detected_transformations}")
    print(f"expected_transformations: {expected_transformations}")
    # Run pattern detection test
    run_sharding_pattern_detection_test(detected_transformations, expected_transformations)


@pytest.mark.parametrize("device_count", get_device_counts())
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize(
    "model_cls, dist_op_expected",
    (
        (MLP, "torch_dist_all_reduce"),
        (FP8MLP, "torch_dist_all_reduce"),
        (nn.Linear, "torch_dist_all_gather"),
        (GQA_Block, "torch_dist_all_reduce"),
    ),
)
def test_sharding(
    model_cls: Type[nn.Module],
    dist_op_expected: str,
    bias: bool,
    device_count: int,
    from_config: bool,
):
    dist_common.spawn_multiprocess_job(
        job=partial(_run_job, model_cls, dist_op_expected, bias, from_config),
        size=device_count,
    )


@pytest.mark.parametrize("world_size", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize(
    "model_cls, dist_op_expected",
    (
        (MLP, "torch_dist_all_reduce"),
        (FP8MLP, "torch_dist_all_reduce"),
        (nn.Linear, "torch_dist_all_gather"),
        (GQA_Block, "torch_dist_all_reduce"),
    ),
)
def test_sharding_pattern_detection(
    model_cls: Type[nn.Module],
    dist_op_expected: str,
    bias: bool,
    world_size: int,
    from_config: bool,
):
    """Test pattern detection logic without distributed execution.

    This test verifies only the pattern detection logic with provided world_size.
    No need to run distributed job, can be run on single process.
    """
    _run_pattern_detection_job(model_cls, bias, 0, world_size, from_config)


if __name__ == "__main__":
    _run_pattern_detection_job(nn.Linear, False, 0, 8, False)
