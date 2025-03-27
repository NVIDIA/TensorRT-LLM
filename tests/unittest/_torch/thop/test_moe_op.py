import pytest
import torch
from _torch.helpers import (calc_diff, ceil_div, per_block_cast_to_fp8,
                            reference_block_scale_moe_torch,
                            reference_moe_torch)
from utils.util import getSMVersion

from tensorrt_llm._torch.modules.fused_moe import (
    FusedMoEQuantScalesFP8BlockScales, RenormalizeMoeRoutingMethod)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_op_profile(dtype):
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = 3
    TOP_K = 2
    TP_SIZE = 1
    TP_RANK = 0
    EP_SIZE = 1
    EP_RANK = 0
    torch.manual_seed(0)
    w2_weight = torch.randn((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                            dtype=dtype).cuda()

    use_fp8_block_scaling = False
    profiler = torch.classes.trtllm.FusedMoeProfiler.get_instance(
        dtype, dtype, dtype, use_fp8_block_scaling)

    # profile
    profiler.run_profile(
        w2_weight,
        TOP_K,
        TP_SIZE,
        TP_RANK,
        EP_SIZE,
        EP_RANK,
        [2, 4, 8]  # num_tokens_buckets
    )

    # after profile, check beyond bucket range
    bucket_1_profile_ids = profiler.get_profile_ids(1, w2_weight, TOP_K,
                                                    NUM_EXPERTS)
    bucket_2_profile_ids = profiler.get_profile_ids(2, w2_weight, TOP_K,
                                                    NUM_EXPERTS)
    assert bucket_1_profile_ids == bucket_2_profile_ids
    assert len(bucket_1_profile_ids) == 2

    bucket_8_profile_ids = profiler.get_profile_ids(8, w2_weight, TOP_K,
                                                    NUM_EXPERTS)
    bucket_16_profile_ids = profiler.get_profile_ids(16, w2_weight, TOP_K,
                                                     NUM_EXPERTS)
    assert bucket_8_profile_ids == bucket_16_profile_ids
    assert len(bucket_8_profile_ids) == 2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_op_run(dtype):
    SEQ_LEN = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = 3
    TOP_K = 2
    TP_SIZE = 1
    TP_RANK = 0
    EP_SIZE = 1
    EP_RANK = 0
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()
    routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
    selected_experts, final_scales = routing_method.apply(router_logits)

    w3_w1_stacked_weight = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=dtype).cuda()
    w2_weight = torch.empty((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                            dtype=dtype).cuda()
    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
        w2 = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda()
        w3 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
        weights[f"{expert_id}.w1.weight"] = w1
        weights[f"{expert_id}.w2.weight"] = w2
        weights[f"{expert_id}.w3.weight"] = w3
        w3_w1_stacked_weight.data[expert_id].copy_(torch.cat([w3, w1], dim=-2))
        w2_weight.data[expert_id].copy_(w2)

    # run no profile
    with torch.inference_mode():
        output_no_profile = torch.ops.trtllm.fused_moe(
            x,
            selected_experts,
            final_scales,
            w3_w1_stacked_weight,
            w2_weight,
            dtype,
            quant_scales=None,
            tp_size=TP_SIZE,
            tp_rank=TP_RANK,
            ep_size=EP_SIZE,
            ep_rank=EP_RANK,
            profile_ids=None,
        )

    # run with profile
    use_fp8_block_scaling = False
    profiler = torch.classes.trtllm.FusedMoeProfiler.get_instance(
        dtype, dtype, dtype, use_fp8_block_scaling)
    profiler.run_profile(
        w2_weight,
        TOP_K,
        TP_SIZE,
        TP_RANK,
        EP_SIZE,
        EP_RANK,
        [2, 4, 8]  # num_tokens_buckets
    )
    profile_ids = profiler.get_profile_ids(SEQ_LEN, w2_weight, TOP_K,
                                           NUM_EXPERTS)
    assert len(profile_ids) == 2
    with torch.inference_mode():
        output_with_profile = torch.ops.trtllm.fused_moe(
            x,
            selected_experts,
            final_scales,
            w3_w1_stacked_weight,
            w2_weight,
            dtype,
            quant_scales=None,
            tp_size=TP_SIZE,
            tp_rank=TP_RANK,
            ep_size=EP_SIZE,
            ep_rank=EP_RANK,
            profile_ids=profile_ids,
        )

    # torch run
    with torch.inference_mode():
        ref_output = reference_moe_torch(x, selected_experts, final_scales,
                                         NUM_EXPERTS, weights)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output_no_profile,
                               output_with_profile,
                               rtol=0.1,
                               atol=0.1)
    torch.testing.assert_close(output_no_profile,
                               ref_output,
                               rtol=0.2,
                               atol=0.5)
    torch.testing.assert_close(output_with_profile,
                               ref_output,
                               rtol=0.2,
                               atol=0.5)

    diff = calc_diff(output_no_profile, ref_output)
    assert diff < 0.001, f'{SEQ_LEN=}, {HIDDEN_SIZE=}, {INTERMEDIATE_SIZE=},  {NUM_EXPERTS=}, {diff:.5f}'


@pytest.mark.skipif(
    getSMVersion() != 90,
    reason="Op only supported on Hopper",
)
def test_moe_op_fp8_block_scale_run(shape_m=8,
                                    shape_k=256,
                                    shape_n=256,
                                    num_experts=3,
                                    top_k=2):
    act_dtype = torch.bfloat16
    weight_dtype = torch.float8_e4m3fn
    scale_dtype = torch.float32
    SEQ_LEN = shape_m
    HIDDEN_SIZE = shape_k
    INTERMEDIATE_SIZE = shape_n
    NUM_EXPERTS = num_experts
    TOP_K = top_k
    TP_SIZE = 1
    TP_RANK = 0
    EP_SIZE = 1
    EP_RANK = 0
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.rand((SEQ_LEN, HIDDEN_SIZE), dtype=act_dtype).cuda() - 0.5
    router_logits = torch.rand(
        (SEQ_LEN, NUM_EXPERTS), dtype=act_dtype).cuda() - 0.5
    routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
    selected_experts, final_scales = routing_method.apply(router_logits)

    w3_w1_stacked_weight = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE),
        dtype=weight_dtype).cuda()
    w3_w1_weight_scaling_factor = torch.empty(
        (NUM_EXPERTS, ceil_div(INTERMEDIATE_SIZE, 128) * 2,
         ceil_div(HIDDEN_SIZE, 128)),
        dtype=scale_dtype).cuda()

    w2_weight = torch.empty((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                            dtype=weight_dtype).cuda()
    w2_weight_scaling_factor = torch.empty(
        (NUM_EXPERTS, ceil_div(HIDDEN_SIZE,
                               128), ceil_div(INTERMEDIATE_SIZE, 128)),
        dtype=scale_dtype).cuda()
    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1 = torch.rand(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=act_dtype).cuda() - 0.5
        w2 = torch.rand(
            (HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=act_dtype).cuda() - 0.5
        w3 = torch.rand(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=act_dtype).cuda() - 0.5
        weights[f"{expert_id}.w1.weight"] = w1
        weights[f"{expert_id}.w2.weight"] = w2
        weights[f"{expert_id}.w3.weight"] = w3

        w1_fp8, w1_scale = per_block_cast_to_fp8(w1)
        w2_fp8, w2_scale = per_block_cast_to_fp8(w2)
        w3_fp8, w3_scale = per_block_cast_to_fp8(w3)

        w3_w1_stacked_weight.data[expert_id].copy_(
            torch.cat([w3_fp8, w1_fp8], dim=-2))
        w3_w1_weight_scaling_factor.data[expert_id].copy_(
            torch.cat([w3_scale, w1_scale], dim=-2))

        w2_weight.data[expert_id].copy_(w2_fp8)
        w2_weight_scaling_factor.data[expert_id].copy_(w2_scale)

    quant_scales = FusedMoEQuantScalesFP8BlockScales(
        fc_weight_scales=w3_w1_weight_scaling_factor,
        proj_weight_scales=w2_weight_scaling_factor,
    )

    # run no profile
    with torch.inference_mode():
        output_no_profile = torch.ops.trtllm.fused_moe(
            x,
            selected_experts,
            final_scales,
            w3_w1_stacked_weight,
            w2_weight,
            act_dtype,
            quant_scales=quant_scales,
            tp_size=TP_SIZE,
            tp_rank=TP_RANK,
            ep_size=EP_SIZE,
            ep_rank=EP_RANK,
            profile_ids=None,
            use_fp8_block_scaling=True)

    # torch run
    with torch.inference_mode():
        ref_block_scale_output = reference_block_scale_moe_torch(
            x, selected_experts, final_scales, NUM_EXPERTS, weights)

    # compare
    torch.cuda.synchronize()

    diff = calc_diff(output_no_profile, ref_block_scale_output)
    assert diff < 0.001, f'{SEQ_LEN=}, {HIDDEN_SIZE=}, {INTERMEDIATE_SIZE=},  {NUM_EXPERTS=}, {diff:.5f}'
