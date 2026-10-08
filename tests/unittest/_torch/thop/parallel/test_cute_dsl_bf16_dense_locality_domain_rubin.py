# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from _torch.thop.parallel._cute_dsl_bf16_rubin_test_utils import (
    RUBIN_CUTE_DSL_MARKS,
    bf16_matmul_atol,
    make_bf16_gemm_runner,
    reset_bf16_gemm_state,
    run_locality_domain_composite,
    select_bf16_tactic,
    skip_if_no_locality_domain,
)

from tensorrt_llm._torch.autotuner import AutoTuner

pytestmark = RUBIN_CUTE_DSL_MARKS


@pytest.mark.parametrize(
    ("kernel_variant", "mnk"),
    [("base", (128, 128, 128)), ("preferred_cluster", (256, 256, 128))],
)
def test_cute_dsl_bf16_gemm_locality_domain_rubin(kernel_variant, mnk):
    skip_if_no_locality_domain()
    torch.manual_seed(123)
    runner = make_bf16_gemm_runner()

    m, n, k = mnk
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight_0 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    weight_1 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    tactics = runner.get_valid_tactics([act, weight_0, output], None)
    tactic = select_bf16_tactic(tactics, kernel_variant, split_k_slices=1)

    expected_0 = act.float() @ weight_0.t().float()
    expected_1 = act.float() @ weight_1.t().float()
    runner([act, weight_0, output], tactic=tactic)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected_0, rtol=1e-2, atol=bf16_matmul_atol(k))

    wide_output = torch.empty(m, n * 2, dtype=torch.bfloat16, device="cuda")
    run_locality_domain_composite(
        "cute_dsl_bf16_gemm_locality_domain_inplace_rubin",
        (act, weight_0, weight_1, wide_output),
        (expected_0, expected_1),
        partition_dim=1,
        kernel_variant=kernel_variant,
        split_k_slices=1,
    )


def test_cute_dsl_bf16_gemm_locality_domain_production_shapes_rubin():
    """Exercise the exact per-partition shapes used by DeepSeek BF16 projections."""
    skip_if_no_locality_domain()
    shapes = (
        ("fused_a", 1, 1056, 7168),
        ("q_b", 1, 12288, 1536),
        ("kv_b", 1, 16384, 512),
        ("o_proj", 1, 3584, 16384),
        ("gate_up", 1, 18432, 7168),
        ("down", 1, 3584, 18432),
    )

    for workload, m, n, k in shapes:
        torch.manual_seed(2112)
        act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        weight_0 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        weight_1 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        output = torch.empty(m, n * 2, dtype=torch.bfloat16, device="cuda")
        expected_0 = act.float() @ weight_0.t().float()
        expected_1 = act.float() @ weight_1.t().float()

        try:
            run_locality_domain_composite(
                "cute_dsl_bf16_gemm_locality_domain_inplace_rubin",
                (act, weight_0, weight_1, output),
                (expected_0, expected_1),
                partition_dim=1,
                kernel_variant="base",
                split_k_slices=1,
                capture_graph=workload == "fused_a",
            )
        except AssertionError as error:
            raise AssertionError(f"{workload} shape {(m, n, k)} failed") from error


def test_gated_mlp_bf16_locality_domain_end_to_end_rubin():
    import torch.nn.functional as F

    from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    skip_if_no_locality_domain()
    torch.manual_seed(2115)
    AutoTuner.get().clear_cache()

    hidden_size, intermediate_size = 128, 128
    input = torch.randn(8, hidden_size, dtype=torch.bfloat16, device="cuda")
    gate_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device="cuda")
    up_weight = torch.randn_like(gate_weight)
    down_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device="cuda")
    expected = F.linear(
        F.silu(F.linear(input, gate_weight)) * F.linear(input, up_weight),
        down_weight,
    )

    mlp = GatedMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bias=False,
        dtype=torch.bfloat16,
        config=ModelConfig(
            use_cute_dsl_bf16_gemm=True,
            locality_domain_policy=LocalityDomainPolicy(enabled=True),
        ),
        enable_locality_domain_bf16_linear=True,
    ).cuda()
    mlp.gate_up_proj.load_weights([{"weight": gate_weight}, {"weight": up_weight}])
    mlp.down_proj.load_weights([{"weight": down_weight}])
    mlp.gate_up_proj.post_load_weights()
    mlp.down_proj.post_load_weights()

    assert mlp.gate_up_proj.partition_plan.enabled
    assert mlp.down_proj.partition_plan.enabled
    assert mlp.gate_up_proj.weight.numel() == 0
    assert mlp.down_proj.weight.numel() == 0
    gate_up_shards = mlp.gate_up_proj._locality_domain_weight_shards
    assert gate_up_shards is not None
    torch.testing.assert_close(gate_up_shards[0]["weight"], gate_weight)
    torch.testing.assert_close(gate_up_shards[1]["weight"], up_weight)

    with torch.inference_mode():
        output = mlp(input)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2.0)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Cross-device locality domain dispatch requires at least two visible CUDA devices",
)
def test_cute_dsl_bf16_gemm_locality_domain_uses_input_device_rubin():
    """The custom op must not select locality domain streams from the caller's current device."""
    skip_if_no_locality_domain()
    original_device = torch.cuda.current_device()
    input_device = 1 if original_device == 0 else 0
    torch.manual_seed(107)
    reset_bf16_gemm_state()

    with torch.cuda.device(input_device):
        act = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
        weight_0 = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        weight_1 = torch.randn_like(weight_0)
        output = torch.empty(1, 256, dtype=torch.bfloat16, device="cuda")
        expected_0 = act.float() @ weight_0.t().float()
        expected_1 = act.float() @ weight_1.t().float()

    assert torch.cuda.current_device() == original_device
    torch.ops.trtllm.cute_dsl_bf16_gemm_locality_domain_inplace_rubin(
        act, weight_0, weight_1, output
    )
    torch.cuda.synchronize(input_device)

    assert torch.cuda.current_device() == original_device
    atol = bf16_matmul_atol(act.shape[-1])
    torch.testing.assert_close(output[:, :128].float(), expected_0, rtol=1e-2, atol=atol)
    torch.testing.assert_close(output[:, 128:].float(), expected_1, rtol=1e-2, atol=atol)


def test_bf16_linear_locality_domain_end_to_end_rubin():
    import torch.nn.functional as F

    from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
    from tensorrt_llm._torch.modules.linear import Linear

    skip_if_no_locality_domain()
    torch.manual_seed(2027)
    reset_bf16_gemm_state()

    batch_size, seq_len, in_features, out_features = 2, 8, 128, 256
    input = torch.randn(
        batch_size,
        seq_len,
        in_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")
    expected = F.linear(input, weight, bias)
    atol = bf16_matmul_atol(in_features)

    linear = Linear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        dtype=torch.bfloat16,
        use_cute_dsl_bf16_gemm=True,
        enable_locality_domain_bf16_linear=True,
        locality_domain_policy=LocalityDomainPolicy(enabled=True),
    ).cuda()
    linear.load_weights([{"weight": weight, "bias": bias}])
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    ModelLoader._walk_transform(linear)
    shards = linear._locality_domain_weight_shards
    ModelLoader._walk_cache_state(linear)
    linear.post_load_weights()
    assert linear._locality_domain_weight_shards is shards

    assert linear.partition_plan.enabled
    assert linear.partition_plan.op_kind == "bf16_linear"
    assert linear._locality_domain_weight_shards is not None
    assert len(linear._locality_domain_weight_shards) == 2
    assert linear.weight.numel() == 0
    localized_weight = torch.cat(
        [shard["weight"] for shard in linear._locality_domain_weight_shards], dim=0
    )
    torch.testing.assert_close(localized_weight, weight)

    with torch.inference_mode():
        output = linear(input)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=atol)

    first_generation_shards = linear._locality_domain_weight_shards
    linear.pre_reload_weights()
    assert linear._locality_domain_weight_shards is None
    assert tuple(linear.weight.shape) == tuple(weight.shape)

    reloaded_weight = torch.randn_like(weight)
    reloaded_bias = torch.randn_like(bias)
    reloaded_expected = F.linear(input, reloaded_weight, reloaded_bias)
    linear.load_weights([{"weight": reloaded_weight, "bias": reloaded_bias}])
    linear.post_load_weights()

    assert linear._locality_domain_weight_shards is not None
    assert linear._locality_domain_weight_shards is not first_generation_shards
    assert len(linear._locality_domain_weight_shards) == 2
    assert linear.weight.numel() == 0
    reloaded_localized_weight = torch.cat(
        [shard["weight"] for shard in linear._locality_domain_weight_shards], dim=0
    )
    torch.testing.assert_close(reloaded_localized_weight, reloaded_weight)

    with torch.inference_mode():
        reloaded_output = linear(input)
    torch.cuda.synchronize()
    torch.testing.assert_close(reloaded_output, reloaded_expected, rtol=1e-2, atol=atol)


def test_cute_dsl_bf16_split_k_locality_domain_rubin():
    """Split-K runs one tactic concurrently across both real locality domain partitions."""
    skip_if_no_locality_domain()

    torch.manual_seed(99)
    reset_bf16_gemm_state()

    split_k_slices = 4
    m, n, k = 256, 256, 8192
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight_0 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    weight_1 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    expected_0 = act.float() @ weight_0.t().float()
    expected_1 = act.float() @ weight_1.t().float()

    wide_output = torch.empty(m, n * 2, dtype=torch.bfloat16, device="cuda")
    run_locality_domain_composite(
        "cute_dsl_bf16_gemm_locality_domain_inplace_rubin",
        (act, weight_0, weight_1, wide_output),
        (expected_0, expected_1),
        partition_dim=1,
        kernel_variant="base",
        split_k_slices=split_k_slices,
        capture_graph=True,
        # TMA reduce-add rounds each partial sum and accumulation to BF16,
        # so account for all slices as well as the reduction length.
        atol=bf16_matmul_atol(k) * split_k_slices,
    )
