import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helpers import reference_moe_torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_op_profile(dtype):
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = 3
    TOP_K = 2
    TP_SIZE = 1
    TP_RANK = 0
    torch.manual_seed(0)
    w2_weight = torch.randn((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                            dtype=dtype).cuda()

    profiler = torch.classes.trtllm.FusedMoeProfiler.get_instance(
        dtype, dtype, dtype)

    # profile
    profiler.run_profile(
        w2_weight,
        TOP_K,
        TP_SIZE,
        TP_RANK,
        [2, 4, 8]  # num_tokens_buckets
    )

    # after profile, check beyond bucket range
    bucket_1_profile_ids = profiler.get_profile_ids(1, w2_weight, TOP_K)
    bucket_2_profile_ids = profiler.get_profile_ids(2, w2_weight, TOP_K)
    assert bucket_1_profile_ids == bucket_2_profile_ids
    assert len(bucket_1_profile_ids) == 2

    bucket_8_profile_ids = profiler.get_profile_ids(8, w2_weight, TOP_K)
    bucket_16_profile_ids = profiler.get_profile_ids(16, w2_weight, TOP_K)
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
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

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
            router_logits.float(),
            w3_w1_stacked_weight,
            w2_weight,
            dtype,
            TOP_K,
            quant_scales=None,
            tp_size=TP_SIZE,
            tp_rank=TP_RANK,
            profile_ids=None,
        )

    # run with profile
    profiler = torch.classes.trtllm.FusedMoeProfiler.get_instance(
        dtype, dtype, dtype)
    profiler.run_profile(
        w2_weight,
        TOP_K,
        TP_SIZE,
        TP_RANK,
        [2, 4, 8]  # num_tokens_buckets
    )
    profile_ids = profiler.get_profile_ids(SEQ_LEN, w2_weight, TOP_K)
    assert len(profile_ids) == 2
    with torch.inference_mode():
        output_with_profile = torch.ops.trtllm.fused_moe(
            x,
            router_logits.float(),
            w3_w1_stacked_weight,
            w2_weight,
            dtype,
            TOP_K,
            quant_scales=None,
            tp_size=TP_SIZE,
            tp_rank=TP_RANK,
            profile_ids=profile_ids,
        )

    # torch run
    with torch.inference_mode():
        ref_output = reference_moe_torch(x, router_logits, TOP_K, weights)

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
