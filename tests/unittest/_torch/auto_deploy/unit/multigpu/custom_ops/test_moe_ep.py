from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from _dist_test_utils import get_device_counts
from _torch.helpers import reference_moe_torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.distributed.common import spawn_multiprocess_job


def _run_moe_ep_test(num_experts: int, topk: int, rank: int, world_size: int):
    SEQ_LEN = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = num_experts
    TOP_K = topk
    dtype = torch.bfloat16

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda") * 0.5

    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    final_scales, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
    final_scales = final_scales / final_scales.sum(dim=-1, keepdim=True)
    final_scales = final_scales.to(x.dtype)

    fused_w3_w1_stacked_weight = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=dtype, device="cuda"
    )
    fused_w2_weight = torch.empty(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype, device="cuda"
    )
    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype, device="cuda") * 0.5
        w2 = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype, device="cuda") * 0.5
        w3 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype, device="cuda") * 0.5
        weights[f"{expert_id}.w1.weight"] = w1
        weights[f"{expert_id}.w2.weight"] = w2
        weights[f"{expert_id}.w3.weight"] = w3

        fused_w3_w1_stacked_weight[expert_id].copy_(torch.cat([w3, w1], dim=-2))
        fused_w2_weight[expert_id].copy_(w2)

    # Shard the fused weights along the expert dimension (dim=0)
    # For num_experts % world_size != 0 case,
    # assign the last (num_experts % world_size) experts to the last rank
    def get_partition(t: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
        n = t.shape[0]
        chunk_size = n // world_size
        if rank == world_size - 1:
            # Last rank gets all remaining rows
            return t[rank * chunk_size :]
        else:
            return t[rank * chunk_size : (rank + 1) * chunk_size]

    sharded_w3_w1 = get_partition(fused_w3_w1_stacked_weight, world_size, rank)
    sharded_w2 = get_partition(fused_w2_weight, world_size, rank)

    # Explicitly mapping selected_experts and final_scales to local version
    experts_per_rank = NUM_EXPERTS // world_size
    low = experts_per_rank * rank
    selected_experts_local = selected_experts - low

    if rank == world_size - 1:
        # For num_experts % world_size != 0 case,
        # assign the last (num_experts % world_size) experts to the last rank
        rank_mask = (selected_experts // experts_per_rank) >= rank
    else:
        rank_mask = (selected_experts // experts_per_rank) == rank

    final_scales_local = final_scales * rank_mask

    output_trt = torch.ops.auto_deploy.trtllm_moe_fused(
        x,
        selected_experts_local,
        final_scales_local,
        sharded_w3_w1,
        sharded_w2,
    )

    # Sum the partial output from every rank.
    dist.all_reduce(output_trt, op=dist.ReduceOp.SUM)

    ref_output = reference_moe_torch(x, selected_experts, final_scales, NUM_EXPERTS, weights)

    torch.cuda.synchronize()
    torch.testing.assert_close(output_trt, ref_output, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("num_experts", [10])
@pytest.mark.parametrize("topk", [3, 10])
@pytest.mark.parametrize("device_count", get_device_counts())
def test_moe_ep(device_count, num_experts, topk):
    spawn_multiprocess_job(job=partial(_run_moe_ep_test, num_experts, topk), size=device_count)
