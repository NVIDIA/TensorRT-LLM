"""Tests for basic graph sharding."""

from functools import partial
from types import SimpleNamespace
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
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_nemotron_h import NemotronHMamba2Mixer
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
    FP8WeightShardingInfo,
    LayerType,
    ShardingTransformConfig,
    SplitDimension,
    WeightShardingInfo,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op, is_op
from tensorrt_llm.functional import AllReduceStrategy

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
    # Mamba2 specific projections
    "in_proj": "mamba",
    "out_proj": "rowwise",
    # MLA specific projections
    "q_a_proj": "gather",
    "q_b_proj": "colwise",
    "kv_a_proj_with_mqa": "gather",
    "kv_b_proj": "colwise",
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


class MLA_Block(nn.Module):
    """Multi-Latent Attention block - simplified standalone version.

    Based on DeepSeek MLA architecture with KV compression.
    This is a minimal, self-contained implementation for testing sharding patterns.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # KV compression path (not sharded - gather)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=bias)

        # KV decompression (sharded column-wise)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )

        # Query path (sharded column-wise)
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=bias)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=bias)
        self.q_a_layernorm = nn.LayerNorm(q_lora_rank)
        # Output projection (sharded row-wise)
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=bias)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        # Compress KV to latent
        compressed_kv_rope = self.kv_a_proj_with_mqa(x)  # (b, s, kv_lora_rank + rope_dim)
        compressed_kv = compressed_kv_rope[:, :, : self.kv_lora_rank]  # (b, s, kv_lora_rank)

        # Decompress to full K and V
        kv = self.kv_b_proj(compressed_kv)  # (b, s, num_heads * (qk_nope + v))
        k_nope_v = kv.view(b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = k_nope_v[:, :, :, : self.qk_nope_head_dim]
        v = k_nope_v[:, :, :, self.qk_nope_head_dim :]

        # Query projection
        # q = q_b_proj @ (layernorm(q_a_proj @ x))
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))  # (b, s, num_heads * qk_head_dim)
        q = q.view(b, s, self.num_heads, self.qk_head_dim)
        q_nope = q[:, :, :, : self.qk_nope_head_dim]

        attn_out = torch.ops.auto_deploy.torch_attention(q_nope, k_nope, v, is_causal=True)
        attn_out = attn_out.contiguous().view(b, s, -1)
        # Output projection
        output = self.o_proj(attn_out)
        return output


def _run_sharding_execution_job(
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
    skip_output_assert = False

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
    elif model_cls == NemotronHMamba2Mixer:
        # Create config for Mamba2 based on Nemotron models
        # Scaled down from typical values: hidden_size=5120, ssm_state_size=128
        mamba_config = SimpleNamespace(
            hidden_size=num_features,
            ssm_state_size=16,  # Scaled from 128
            mamba_num_heads=num_heads,
            mamba_head_dim=num_features // num_heads,  # 8
            n_groups=num_heads,  # Typical value
            chunk_size=256,
            conv_kernel=4,
            use_conv_bias=bias,
            use_bias=bias,
            mamba_hidden_act="silu",
            layer_norm_epsilon=1e-5,
            time_step_limit=(0.0, float("inf")),
            time_step_min=0.001,
            time_step_max=0.1,
            time_step_floor=1e-4,
            initializer_range=0.02,
            rescale_prenorm_residual=False,
            residual_in_fp32=False,
            num_hidden_layers=1,
        )
        model = model_cls(mamba_config, layer_idx=0).to(device="cuda", dtype=torch.float16)
    elif model_cls == MLA_Block:
        # Use actual DeepSeek-V3/R1 production values
        # From HuggingFace config (HunYuanPretrainedConfig defaults):
        # hidden_size=4096, num_attention_heads=32
        # kv_lora_rank=512, q_lora_rank=1536
        # qk_rope_head_dim=64, v_head_dim=128, qk_nope_head_dim=128
        num_heads_mla = 16
        qk_nope_head_dim = 64
        qk_rope_head_dim = 32
        v_head_dim = 64
        kv_lora_rank = 256
        skip_output_assert = True

        model = model_cls(
            hidden_size=num_features,
            num_heads=num_heads_mla,
            q_lora_rank=kv_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            bias=bias,
        ).to(device="cuda", dtype=torch.float16)
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
        if model_cls == MLA_Block:
            # since q_a_proj is simple-sharded and followed by q_a_layernorm, the layernorm params
            # are NOT sharded - they have to be replicated. To account for this, we need to add the
            # number of parameters of the layernorm (weight and bias)to the number of parameters of the model.
            num_params += 2 * kv_lora_rank * (world_size - 1) // world_size
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
        weight_sizes_valid = verify_local_weight_sizes(gm)
        return has_expected_dist_ops and weight_sizes_valid

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        check_transformed_graph=combined_graph_check,
        _get_expected_num_params=_get_expected_num_params,
        skip_output_assert=skip_output_assert,
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
    elif model_cls == NemotronHMamba2Mixer:
        # Create config for Mamba2
        mamba_config = SimpleNamespace(
            hidden_size=num_features,
            ssm_state_size=16,
            mamba_num_heads=num_heads,
            mamba_head_dim=num_features // num_heads,
            n_groups=num_heads,
            chunk_size=256,
            conv_kernel=4,
            use_conv_bias=bias,
            use_bias=bias,
            mamba_hidden_act="silu",
            layer_norm_epsilon=1e-5,
            time_step_limit=(0.0, float("inf")),
            time_step_min=0.001,
            time_step_max=0.1,
            time_step_floor=1e-4,
            initializer_range=0.02,
            rescale_prenorm_residual=False,
            residual_in_fp32=False,
            num_hidden_layers=1,
        )
        model = model_cls(mamba_config, layer_idx=0).to(device="cuda", dtype=torch.float16)
    elif model_cls == MLA_Block:
        # Create simplified MLA based on DeepSeek-V3 architecture
        qk_nope_head_dim = 2
        qk_rope_head_dim = 1
        v_head_dim = 2
        kv_lora_rank = 8

        model = model_cls(
            hidden_size=num_features,
            num_heads=num_heads,
            q_lora_rank=kv_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            bias=bias,
        ).to(device="cuda", dtype=torch.float16)
    else:
        model = model_cls(num_features, num_features, bias=bias).to(
            device="cuda", dtype=torch.float16
        )
    x = torch.randn(batch_size, sequence_len, num_features, device="cuda", dtype=torch.float16)

    # Test pattern detection - create expected transformations for validation
    config = ShardingTransformConfig(
        rank=rank,
        world_size=world_size,
        stage="sharding",
        allreduce_strategy=AllReduceStrategy.AUTO,
    )
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
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            config=config,
                            dist_op=dist_op,
                            min_local_shape=min_local_shape,
                            layer_type=LayerType.ATTENTION,
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
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            config=config,
                            dist_op=dist_op,
                            min_local_shape=1,
                            layer_type=LayerType.MLP,
                        )
                    )
        elif model_cls == nn.Linear:
            # expect simple shard only (dim=0, add_dist=True, min_local_shape=1)
            for node in gm.graph.nodes:
                if is_linear_op(node):
                    expected_transformations.append(
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=SplitDimension.COLUMN,  # Simple shard uses dim=0
                            config=config,
                            dist_op="all_gather",
                            min_local_shape=1,
                            layer_type=LayerType.MLP,
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
                        FP8WeightShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            config=config,
                            dist_op=dist_op,
                            min_local_shape=1,
                        )
                    )
        elif model_cls == NemotronHMamba2Mixer:
            for node in gm.graph.nodes:
                if is_linear_op(node):
                    # in_proj should be sharded column-wise
                    # out_proj should be sharded row-wise with all_reduce
                    if "out_proj" in node.args[1].name:
                        dim = SplitDimension.ROW
                        dist_op = "all_reduce"
                        fused_weight_dims = None
                    else:
                        dim = SplitDimension.COLUMN
                        dist_op = None
                        fused_weight_dims = (
                            num_features,
                            num_features,
                            16 * num_heads,
                            16 * num_heads,
                            num_heads,
                        )
                    expected_transformations.append(
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            config=config,
                            dist_op=dist_op,
                            min_local_shape=1,
                            layer_type=LayerType.SSM,
                            fused_weight_dims=fused_weight_dims,
                        )
                    )
                elif is_op(node, torch.ops.auto_deploy.torch_causal_conv1d):
                    expected_transformations.append(
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=SplitDimension.COLUMN,
                            config=config,
                            dist_op=None,
                            min_local_shape=1,
                            layer_type=LayerType.SSM,
                            fused_weight_dims=(num_features, 16 * num_heads, 16 * num_heads),
                        )
                    )
                elif is_op(node, torch.ops.auto_deploy.torch_ssm):
                    expected_transformations.append(
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=SplitDimension.COLUMN,
                            config=config,
                            dist_op=None,
                            min_local_shape=1,
                            layer_type=LayerType.SSM,
                            fused_weight_dims=None,
                        )
                    )
                elif len(node.args) > 1 and (
                    "norm_weight" in node.args[0].name or "a_log" in node.args[0].name
                ):
                    expected_transformations.append(
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=SplitDimension.COLUMN,
                            config=config,
                            dist_op=None,
                            min_local_shape=1,
                            layer_type=LayerType.SSM,
                            fused_weight_dims=None,
                        )
                    )
        elif model_cls == MLA_Block:
            for node in gm.graph.nodes:
                if is_linear_op(node):
                    # kv_a_proj_with_mqa: gather (no sharding)
                    # q_b_proj/kv_b_proj: column-wise
                    # o_proj: row-wise with all_reduce
                    min_local_shape = 2
                    if "o_proj" in node.args[1].name:
                        dim = SplitDimension.ROW
                        dist_op = "all_reduce"
                    elif (
                        "kv_a_proj_with_mqa" in node.args[1].name or "q_a_proj" in node.args[1].name
                    ):
                        # This is simple-shard gather
                        dim = SplitDimension.COLUMN
                        dist_op = "all_gather"
                        min_local_shape = 1
                    else:
                        dim = SplitDimension.COLUMN
                        dist_op = None
                    expected_transformations.append(
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            config=config,
                            dist_op=dist_op,
                            min_local_shape=min_local_shape,
                            layer_type=LayerType.MLA,
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
    detected_transformations = (
        optimizer.shared_config.sharding_transform_container.weight_sharding_transforms
    )

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
        (NemotronHMamba2Mixer, "torch_dist_all_reduce"),
        (MLA_Block, "torch_dist_all_reduce"),
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
        job=partial(_run_sharding_execution_job, model_cls, dist_op_expected, bias, from_config),
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
        (NemotronHMamba2Mixer, "torch_dist_all_reduce"),
        (MLA_Block, "torch_dist_all_reduce"),
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
