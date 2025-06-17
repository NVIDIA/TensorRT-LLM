import torch

from tensorrt_llm._torch.modules.fused_moe import DefaultMoeRoutingMethod
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping


def renorm_permutate_op():
    router_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], device='cuda:0', dtype=torch.float32)
    topk = 2
    output = torch.ops.trtllm.renorm_permute_op(router_logits, topk)
    print(f"limin: output = {output}")

def test_moe_finalize_scale_op(top_k=2, SEQ_LEN=4, HIDDEN_SIZE=16, NUM_EXPERTS=6, dtype=torch.bfloat16):
    mapping = Mapping()
    mapping.rank = mpi_rank()
    torch.cuda.set_device(mapping.rank)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    tp_size = 1
    tp_rank = 0
    ep_size = 1
    ep_rank = 0

    permuted_data_tensor = torch.tensor([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]],
        device='cuda:0', dtype=torch.bfloat16)
    src_to_dest_map_tensor = torch.tensor([3, 2, 7, 1, 0, 5, 4, 6], device='cuda:0', dtype=torch.int32)

    unpermuted_token_selected_experts_tensor = torch.tensor([3, 0, 1, 4, 5, 3, 0, 4], device='cuda:0', dtype=torch.int32)
    expert_first_token_offset_tensor = torch.tensor([0, 2, 3, 3, 5, 7, 8], device='cuda:0', dtype=torch.int32)
    token_final_scales = torch.tensor([0.3074, 0.2681, 0.3343, 0.2403, 0.3420, 0.2402, 0.3572, 0.2782], device='cuda:0', dtype=torch.float32).reshape(SEQ_LEN, top_k)

    output = torch.ops.trtllm.moe_finalize_scale_op(
        permuted_data_tensor,
        None,
        token_final_scales,
        src_to_dest_map_tensor,
        unpermuted_token_selected_experts_tensor,
        expert_first_token_offset_tensor,
        SEQ_LEN,
        HIDDEN_SIZE,
        top_k,
        NUM_EXPERTS,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank
    )
    print(f"limin: output = {output}")

def test_moe_permutate_op(top_k=2, SEQ_LEN=4, HIDDEN_SIZE=16, NUM_EXPERTS=6, dtype=torch.bfloat16):
    # mapping = mapping or Mapping()
    mapping = Mapping()
    mapping.rank = mpi_rank()
    torch.cuda.set_device(mapping.rank)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    input = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    for i in range(SEQ_LEN):
        input[i, :] = i
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()
    routing_method = DefaultMoeRoutingMethod(top_k=top_k)
    # [4, 2]
    token_selected_experts, token_final_scales = routing_method.apply(
            router_logits)
    print(f"limin: input = {input}")
    print(f"limin: router_logits = {router_logits}")
    print(f"limin: token_selected_experts = {token_selected_experts}")
    print(f"limin: token_final_scales = {token_final_scales}")
    tp_size = 1
    tp_rank = 0
    ep_size = 1
    ep_rank = 0
    # 这代表什么含义？
    cluster_size = 1
    cluster_rank = 0
    min_latency_mode = False
    use_fp8_block_scaling = True

    input_sf = None
    fc1_expert_weights = None
    fc2_expert_weights = None
    quant_scales = None
    (
        unpermuted_token_selected_experts_tensor, 
        unpermuted_source_token_ids_tensor, 
        permuted_source_token_ids_tensor, 
        permuted_token_selected_experts_tensor, 
        permuted_data_tensor, 
        expert_first_token_offset_tensor, 
        permuted_token_final_scales_tensor, 
        src_to_dest_map_tensor
    ) = torch.ops.trtllm.moe_permute_op(
        input, 
        token_selected_experts, 
        token_final_scales, 
        fc1_expert_weights, 
        fc2_expert_weights, 
        quant_scales, 
        input_sf, 
        NUM_EXPERTS, # num_experts_on_rank
        tp_size, 
        tp_rank,
        ep_size, 
        ep_rank, 
        cluster_size, 
        cluster_rank, 
        min_latency_mode, 
        use_fp8_block_scaling
    )
    # std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_moe_expand_op(torch::Tensor const& input,
    # torch::optional<torch::Tensor> token_final_scales, torch::Tensor const& permuted_source_token_ids,
    # int64_t const num_rows, torch::Tensor & expert_first_token_offset_tensor, int64_t const hidden_size,
    # int64_t const experts_per_token, int64_t const num_experts_per_node, int64_t const tp_size, int64_t const tp_rank,
    # int64_t const ep_size, int64_t const ep_rank, bool use_fp8_block_scaling)
    (
        new_permuted_data_tensor, 
        new_permuted_token_final_scales_tensor, 
        new_src_to_dest_map_tensor
    ) = torch.ops.trtllm.moe_expand_op(
        input,
        token_final_scales,
        permuted_source_token_ids_tensor,
        SEQ_LEN,
        expert_first_token_offset_tensor,
        HIDDEN_SIZE,
        top_k,
        NUM_EXPERTS,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
        use_fp8_block_scaling
    )
    print(f"limin: unpermuted_token_selected_experts_tensor = {unpermuted_token_selected_experts_tensor.shape}, {unpermuted_token_selected_experts_tensor}")
    print(f"limin: unpermuted_source_token_ids_tensor = {unpermuted_source_token_ids_tensor.shape}, {unpermuted_source_token_ids_tensor}")
    print(f"limin: permuted_source_token_ids_tensor = {permuted_source_token_ids_tensor.shape}, {permuted_source_token_ids_tensor}")
    print(f"limin: permuted_token_selected_experts_tensor = {permuted_token_selected_experts_tensor.shape}, {permuted_token_selected_experts_tensor}")
    print(f"limin: permuted_data_tensor = {permuted_data_tensor.shape}, {permuted_data_tensor}")
    print(f"limin: expert_first_token_offset_tensor = {expert_first_token_offset_tensor.shape}, {expert_first_token_offset_tensor}")
    print(f"limin: permuted_token_final_scales_tensor = {permuted_token_final_scales_tensor.shape}, {permuted_token_final_scales_tensor}")
    print(f"limin: src_to_dest_map_tensor = {src_to_dest_map_tensor.shape}, {src_to_dest_map_tensor}")
    torch.testing.assert_close(new_permuted_data_tensor, permuted_data_tensor)
    torch.testing.assert_close(new_permuted_token_final_scales_tensor, permuted_token_final_scales_tensor)
    torch.testing.assert_close(new_src_to_dest_map_tensor, src_to_dest_map_tensor)
    print("PASSED")
    # run_moe_finalize_scale_op(torch::Tensor const& gemm2_output, torch::Tensor const& fc2_expert_biases,
    # torch::Tensor const& unpermuted_final_scales, torch::Tensor const& expanded_source_row_to_expanded_dest_row,
    # torch::Tensor const& expert_for_source_row,
    # /*torch::Tensor const& num_valid_tokens_ptr,*/ torch::Tensor const& expert_first_token_offset_tensor,
    # int64_t const num_rows, int64_t const hidden_size, int64_t const experts_per_token, int64_t const num_experts_per_node, 
    # int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank)
    output = torch.ops.trtllm.moe_finalize_scale_op(
       permuted_data_tensor,
       None,
       token_final_scales,
       src_to_dest_map_tensor,
       unpermuted_token_selected_experts_tensor,
       expert_first_token_offset_tensor,
       SEQ_LEN,
       HIDDEN_SIZE,
       top_k,
       NUM_EXPERTS,
       tp_size,
       tp_rank,
       ep_size,
       ep_rank
    )
    print(f"limin: output = {output}")

if __name__ == "__main__":
    test_moe_permutate_op(top_k=6, SEQ_LEN=8192, HIDDEN_SIZE=16, NUM_EXPERTS=72, dtype=torch.bfloat16)
    test_moe_finalize_scale_op()
    # renorm_permutate_op()
