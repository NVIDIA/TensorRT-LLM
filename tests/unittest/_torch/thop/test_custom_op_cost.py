import torch
import tensorrt_llm
from time import time
# from utils.util import getSMVersion
import nvtx

def test_custom_op_cost(m=1024, n=1024, k=1024, func=torch.ops.trtllm.custom_op_cost_test):
    input = torch.randn(m, k, device='cuda')
    weight = torch.randn(n, k, device='cuda')
    input_sf = torch.randn(m, k, device='cuda')
    weight_sf = torch.randn(n, k, device='cuda')
    for i in range(3):
        output = func(input, weight, input_sf, weight_sf)

    all_time = 0
    for i in range(10):
        start = time()
        output = func(input, weight, input_sf, weight_sf)
        end = time()
        print(f"time: {(end - start) * 1000000} us")
        all_time += (end - start) * 1000000
    # print(f"output: {output}")
    print(f"average time: {all_time / 10} us")

def test_custom_op_cost_16_input_tensors(m=1024, n=1024, k=1024, func=torch.ops.trtllm.custom_op_cost_test_16_input_tensors):
    input = torch.randn(m, k, device='cuda')
    for i in range(3):
        output = func(input, input, input, input, input, input, input, input, input, input, input, input, input, input, input, input)

    all_time = 0
    for i in range(10):
        start = time()
        output = func(input, input, input, input, input, input, input, input, input, input, input, input, input, input, input, input)
        end = time()
        print(f"time: {(end - start) * 1000000} us")
        all_time += (end - start) * 1000000
    # print(f"output: {output}")
    print(f"average time: {all_time / 10} us")


def test_permute_op_cost(num_tokens=512, hidden_size=1024, k=6):
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')
    token_selected_experts = torch.randint(0, 72, (num_tokens, k), dtype=torch.int32, device='cuda')
    token_final_scales = torch.randn(num_tokens, k, dtype=torch.float32, device='cuda')
    x_sf = None

    for i in range(3):
        output = torch.ops.trtllm.moe_permute_op(
            x,
            token_selected_experts,
            token_final_scales,
            None,  # w3_w1_weight.view(weight_dtype),
            None,  # w2_weight.view(weight_dtype),
            None,  # quant_scales,
            input_sf=x_sf,
            num_experts_on_rank=72,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            cluster_size=1,
            cluster_rank=0,
            min_latency_mode=False,
            use_fp8_block_scaling=True,
        )
    all_time = 0
    for i in range(10):
        start = time()
        with nvtx.annotate("moe_permute_op", color="red"):
            output = torch.ops.trtllm.moe_permute_op(
                x,
                token_selected_experts,
                token_final_scales,
                None,  # w3_w1_weight.view(weight_dtype),
                None,  # w2_weight.view(weight_dtype),
                None,  # quant_scales,
                input_sf=x_sf,
                num_experts_on_rank=72,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                cluster_size=1,
                cluster_rank=0,
                min_latency_mode=False,
                use_fp8_block_scaling=True,
            )
        end = time()
        print(f"time: {(end - start) * 1000000} us")
        all_time += (end - start) * 1000000
    print(f"average time: {all_time / 10} us")

def test_torch_linear_cost(m=1024, n=1024, k=1024):
    input = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')
    for i in range(3):
        output = torch.nn.functional.linear(input, weight)
    all_time = 0
    for i in range(10):
        start = time()
        output = torch.nn.functional.linear(input, weight)
        end = time()
        print(f"torch linear bf16 host time: {(end - start) * 1000000} us")
        all_time += (end - start) * 1000000
    print(f"torch linear bf16 host average time: {all_time / 10} us")

if __name__ == "__main__":
    # test_custom_op_cost(func=torch.ops.trtllm.custom_op_cost_test)
    # test_custom_op_cost(func=torch.ops.trtllm.custom_op_python_cost_test)
    # test_torch_linear_cost()
    test_permute_op_cost()
    # test_custom_op_cost_16_input_tensors()