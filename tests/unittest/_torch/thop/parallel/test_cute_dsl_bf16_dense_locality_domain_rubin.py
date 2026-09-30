# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from _torch.thop.parallel._cute_dsl_bf16_rubin_test_utils import (
    RUBIN_CUTE_DSL_MARKS,
    bf16_matmul_atol,
    make_bf16_bmm_runner,
    make_bf16_gemm_runner,
    reset_bf16_gemm_state,
    run_locality_domain_composite,
    select_bf16_tactic,
    select_captured_locality_domain_tactic,
    skip_if_no_locality_domain,
)

from tensorrt_llm._torch.autotuner import AutoTuner, autotune

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


def test_deepseek_gate_bf16_locality_domain_end_to_end_rubin():
    from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate

    skip_if_no_locality_domain()
    torch.manual_seed(2114)
    reset_bf16_gemm_state()

    hidden_states = torch.randn(1, 1, 7168, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(256, 7168, dtype=torch.bfloat16, device="cuda")
    correction_bias = torch.randn(256, dtype=torch.float32, device="cuda")
    expected = hidden_states.float() @ weight.t().float()
    atol = bf16_matmul_atol(hidden_states.shape[-1])

    gate = DeepseekV3Gate(
        hidden_size=7168,
        num_experts=256,
        top_k=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        dtype=torch.bfloat16,
        use_cute_dsl_bf16_gemm=True,
        locality_domain_policy=LocalityDomainPolicy(enabled=True),
    ).cuda()
    gate.load_weights([{"weight": weight, "e_score_correction_bias": correction_bias}])
    gate.post_load_weights()

    assert gate.partition_plan.enabled
    assert gate.partition_plan.op_kind == "bf16_linear"
    assert gate._locality_domain_weight_shards is not None
    assert gate.weight.numel() == 0
    torch.testing.assert_close(gate.e_score_correction_bias, correction_bias)

    tuner = AutoTuner.get()
    with tuner.capture() as tactics_capture, torch.inference_mode():
        gate(hidden_states)
    concurrent_runner, tactic, _ = select_captured_locality_domain_tactic(
        tactics_capture,
        "cute_dsl_bf16_gemm_locality_domain_inplace_rubin",
        "base",
        split_k_slices=1,
    )

    graph = torch.cuda.CUDAGraph()
    with tuner.replay(((concurrent_runner, tactic),)), torch.inference_mode():
        gate(hidden_states)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            output = gate(hidden_states)
    with torch.inference_mode():
        output.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()

    assert output.dtype == torch.float32
    assert output.shape == (1, 1, 256)
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=atol)

    first_generation_shards = gate._locality_domain_weight_shards
    gate.pre_reload_weights()
    assert gate._locality_domain_weight_shards is None
    assert gate.weight.shape == weight.shape

    reloaded_weight = torch.randn_like(weight)
    reloaded_bias = torch.randn_like(correction_bias)
    reloaded_expected = hidden_states.float() @ reloaded_weight.t().float()
    gate.load_weights([{"weight": reloaded_weight, "e_score_correction_bias": reloaded_bias}])
    gate.post_load_weights()

    assert gate._locality_domain_weight_shards is not None
    assert gate._locality_domain_weight_shards is not first_generation_shards
    assert gate.weight.numel() == 0
    torch.testing.assert_close(gate.e_score_correction_bias, reloaded_bias)
    with tuner.replay(((concurrent_runner, tactic),)), torch.inference_mode():
        reloaded_output = gate(hidden_states)
    torch.cuda.synchronize()
    torch.testing.assert_close(reloaded_output, reloaded_expected, rtol=1e-2, atol=atol)


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


@pytest.mark.parametrize(
    ("kernel_variant", "bm_nk"),
    [("base", (2, 64, 128, 128)), ("preferred_cluster", (2, 256, 256, 128))],
)
def test_cute_dsl_bf16_bmm_locality_domain_rubin(kernel_variant, bm_nk):
    skip_if_no_locality_domain()
    torch.manual_seed(456)
    runner = make_bf16_bmm_runner()

    batch_size, m, n, k = bm_nk
    act = torch.randn(batch_size, m, k, dtype=torch.bfloat16, device="cuda")
    weight_0 = torch.randn(batch_size, n, k, dtype=torch.bfloat16, device="cuda")
    weight_1 = torch.randn(batch_size, n, k, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(batch_size, m, n, dtype=torch.bfloat16, device="cuda")
    tactics = runner.get_valid_tactics([act, weight_0, output], None)
    tactic = select_bf16_tactic(tactics, kernel_variant)

    expected_0 = torch.bmm(act.float(), weight_0.transpose(1, 2).float())
    expected_1 = torch.bmm(act.float(), weight_1.transpose(1, 2).float())
    runner([act, weight_0, output], tactic=tactic)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected_0, rtol=1e-2, atol=bf16_matmul_atol(k))

    wide_output = torch.empty(batch_size, m, n * 2, dtype=torch.bfloat16, device="cuda")
    run_locality_domain_composite(
        "cute_dsl_bf16_bmm_locality_domain_inplace_rubin",
        (act, weight_0, weight_1, wide_output),
        (expected_0, expected_1),
        partition_dim=2,
        kernel_variant=kernel_variant,
    )


@pytest.mark.parametrize(
    ("projection", "k", "n", "output_padding"),
    [
        pytest.param("k_b_proj", 128, 256, 64, id="k-projection"),
        pytest.param("v_b_proj", 512, 64, 0, id="v-projection"),
    ],
)
def test_cute_dsl_bf16_bmm_locality_domain_mla_strided_cuda_graph_rubin(
    projection, k, n, output_padding
):
    """Match MLA absorption's transposed A and strided output views at decode M=1."""
    del projection
    skip_if_no_locality_domain()
    torch.manual_seed(512)
    batch_size, m = 128, 1
    # MLA starts from [tokens, heads, K] and transposes tokens/heads. The
    # k-projection input is also sliced out of a wider q-head tensor.
    input_padding = 64 if k == 128 else 0
    input_storage = torch.randn(
        m,
        batch_size,
        k + input_padding,
        dtype=torch.bfloat16,
        device="cuda",
    )
    act = input_storage[..., :k].transpose(0, 1)
    assert act.stride(0) == k + input_padding
    assert act.stride(-1) == 1

    weight_0 = torch.randn(batch_size, n, k, dtype=torch.bfloat16, device="cuda")
    weight_1 = torch.randn_like(weight_0)
    full_n = n * 2
    output_storage = torch.empty(
        m,
        batch_size,
        full_n + output_padding,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = output_storage[..., :full_n].transpose(0, 1)
    assert output.stride(0) == full_n + output_padding
    expected_0 = torch.bmm(act.float(), weight_0.transpose(1, 2).float())
    expected_1 = torch.bmm(act.float(), weight_1.transpose(1, 2).float())

    tactics = run_locality_domain_composite(
        "cute_dsl_bf16_bmm_locality_domain_inplace_rubin",
        (act, weight_0, weight_1, output),
        (expected_0, expected_1),
        partition_dim=2,
        kernel_variant="base",
    )
    assert tactics
    assert all(tactic[0] == "base" and tactic[3][0] == 1 for tactic in tactics)


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


def test_cute_dsl_bf16_bmm_autotune_all_tactics_rubin():
    torch.manual_seed(789)
    runner = make_bf16_bmm_runner()

    batch_size, m, n, k = 1, 256, 256, 128
    act = torch.randn(batch_size, m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(batch_size, n, k, dtype=torch.bfloat16, device="cuda")
    expected = torch.bmm(act.float(), weight.transpose(1, 2).float())
    atol = bf16_matmul_atol(k)

    output = torch.empty(batch_size, m, n, dtype=torch.bfloat16, device="cuda")
    with autotune(skip_dynamic_tuning_buckets=True):
        torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(act, weight, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=atol)

    expected_tactics = runner.get_valid_tactics([act, weight, output], None)
    assert any(t[0] == "base" and t[1] is True for t in expected_tactics)
    assert any(t[0] == "preferred_cluster" and t[1] is True for t in expected_tactics)

    tuner = AutoTuner.get()
    with tuner.capture() as all_tactics:
        captured_output = torch.empty_like(output)
        torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(act, weight, captured_output)
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_output.float(), expected, rtol=1e-2, atol=atol)

    tested_tactics = []
    for ((captured_runner, tactic),) in all_tactics:
        replay_output = torch.empty_like(output)
        tested_tactics.append(tactic)
        with tuner.replay(((captured_runner, tactic),)):
            torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(act, weight, replay_output)
        torch.cuda.synchronize()
        torch.testing.assert_close(replay_output.float(), expected, rtol=1e-2, atol=atol)

    assert tested_tactics == expected_tactics


@pytest.mark.parametrize("mnk", [(16462, 2112, 7168), (16384, 24576, 1536)])
def test_cute_dsl_bf16_gemm_preferred_cluster_full_coverage_rubin(mnk):
    """Multi-wave full-output-coverage regression for the preferred-cluster GEMM.

    The Rubin preferred-cluster mega-kernel runs a flexible cluster launch:
    the hardware downgrades some preferred (4,2)=8-CTA clusters to the
    fallback (2,1) shape (always happens on 216-SM parts due to GPC
    fragmentation). Both cluster populations MUST schedule over the single
    preferred tile partition; if the fallback-shaped clusters use a different
    partition, whole cluster work-items are never visited and the
    corresponding output tiles keep stale memory -- a silent accuracy bug.

    Small single-wave shapes (one persistent wave covers every tile) never
    expose this. Here M is large enough to force many waves; we NaN-poison the
    output so any unvisited tile shows up as NaN rows, independent of error
    tolerance, and require every preferred_cluster tactic to write the full
    output and match the reference.
    """
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    m, n, k = mnk
    torch.manual_seed(41)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
    c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    runner = cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner(use_tvm_ffi=True)
    tactics = runner.get_valid_tactics([a, w, c], None)
    pc = [t for t in tactics if isinstance(t, tuple) and t[0] == "preferred_cluster"]
    assert pc, "no preferred_cluster tactic offered at this multi-wave shape"

    for tactic in pc:
        c.fill_(float("nan"))
        runner([a, w, c], tactic=tactic)
        torch.cuda.synchronize()
        nan_rows = int(torch.isnan(c).any(dim=1).sum().item())
        assert nan_rows == 0, f"tactic {tactic} left {nan_rows} output rows unwritten"
        worst = 0.0
        for lo in range(0, m, 4096):
            hi = min(m, lo + 4096)
            ref = a[lo:hi].float() @ w.float().t()
            worst = max(worst, (c[lo:hi].float() - ref).abs().max().item())
        assert worst < 0.5, f"tactic {tactic} wrong output: max_abs={worst:.3f}"


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
