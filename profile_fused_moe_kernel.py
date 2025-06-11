import time
import torch
import numpy as np
from typing import List, Tuple, Dict

from tensorrt_llm._torch.modules.fused_moe import DefaultMoeRoutingMethod
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping


def setup_fused_moe_kernel_inputs(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int, 
    num_experts: int,
    num_activated_experts: int,
    top_k: int,
    dtype: torch.dtype = torch.bfloat16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
    """Setup inputs for the torch.ops.trtllm.fused_moe kernel."""
    
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Input tensor
    x = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()
    
    # Create routing method and apply it to get expert selections and scales
    routing_method = DefaultMoeRoutingMethod(top_k=top_k)
    router_logits = create_routing_logits(num_tokens, num_experts, num_activated_experts, top_k, dtype)
    
    token_selected_experts, token_final_scales = routing_method.apply(router_logits)
    
    # For the kernel, token_selected_slots is the same as token_selected_experts in simple case
    token_selected_slots = token_selected_experts
    
    # Create expert weights 
    # w3_w1_weight: concatenated w3 and w1 weights [num_experts, 2*intermediate_size, hidden_size]
    w3_w1_weight = torch.randn((num_experts, 2 * intermediate_size, hidden_size), dtype=dtype).cuda()
    
    # w2_weight: down projection weights [num_experts, hidden_size, intermediate_size]  
    w2_weight = torch.randn((num_experts, hidden_size, intermediate_size), dtype=dtype).cuda()
    
    # Quantization scales (empty for non-quantized)
    quant_scales = []
    
    return x, token_selected_slots, token_final_scales, w3_w1_weight, w2_weight, quant_scales


def create_routing_logits(seq_len: int, num_experts: int, num_activated_experts: int, top_k: int, dtype: torch.dtype) -> torch.Tensor:
    """Create router logits that activate specified number of experts."""
    torch.manual_seed(42)
    
    # Create logits with strong bias towards first num_activated_experts
    router_logits = torch.full((seq_len, num_experts), -10.0, dtype=dtype).cuda()
    
    # Make sure we activate the right number of experts for top_k routing
    # For top_k > 1, we need to ensure multiple experts per token get high scores
    for i in range(seq_len):
        # Distribute tokens across activated experts in round-robin fashion
        base_expert = (i * top_k) % num_activated_experts
        for k in range(top_k):
            expert_idx = (base_expert + k) % num_activated_experts
            router_logits[i, expert_idx] = 10.0 - k * 0.1  # Slight preference for first expert
        
    return router_logits


def profile_fused_moe_kernel(
    test_name: str,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_activated_experts: int,
    top_k: int,
    num_warmup: int = 10,
    num_iterations: int = 100
) -> float:
    """Profile the torch.ops.trtllm.fused_moe kernel directly."""
    
    print(f"\nProfiling: {test_name}")
    print(f"  Tokens: {num_tokens}, Hidden: {hidden_size}, Intermediate: {intermediate_size}")
    print(f"  Experts: {num_experts}, Activated: {num_activated_experts}, TopK: {top_k}")
    
    # Setup kernel inputs
    x, token_selected_slots, token_final_scales, w3_w1_weight, w2_weight, quant_scales = setup_fused_moe_kernel_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_activated_experts=num_activated_experts,
        top_k=top_k,
        dtype=torch.bfloat16
    )
    
    output_dtype = torch.bfloat16
    
    # Kernel parameters
    tp_size = 1
    tp_rank = 0
    ep_size = 1
    ep_rank = 0
    cluster_size = 1
    cluster_rank = 0
    use_deepseek_fp8_block_scale = False
    use_w4a8_group_scaling = False
    min_latency_mode = False
    tune_max_num_tokens = 8192
    
    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = torch.ops.trtllm.fused_moe(
                x,
                token_selected_slots,
                token_final_scales,
                w3_w1_weight,
                w2_weight,
                output_dtype,
                quant_scales=quant_scales,
                input_sf=None,
                tp_size=tp_size,
                tp_rank=tp_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
                cluster_size=cluster_size,
                cluster_rank=cluster_rank,
                use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
                use_w4a8_group_scaling=use_w4a8_group_scaling,
                min_latency_mode=min_latency_mode,
                tune_max_num_tokens=tune_max_num_tokens,
            )
    
    torch.cuda.synchronize()
    
    # Timing using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.inference_mode():
        start_event.record()
        for _ in range(num_iterations):
            _ = torch.ops.trtllm.fused_moe(
                x,
                token_selected_slots,
                token_final_scales,
                w3_w1_weight,
                w2_weight,
                output_dtype,
                quant_scales=quant_scales,
                input_sf=None,
                tp_size=tp_size,
                tp_rank=tp_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
                cluster_size=cluster_size,
                cluster_rank=cluster_rank,
                use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
                use_w4a8_group_scaling=use_w4a8_group_scaling,
                min_latency_mode=min_latency_mode,
                tune_max_num_tokens=tune_max_num_tokens,
            )
        end_event.record()
    
    end_event.synchronize()
    total_time = start_event.elapsed_time(end_event)  # Total time in ms
    avg_time = total_time / num_iterations
    std_time = 0.0  # We can't calculate std deviation with this approach
    
    print(f"  Average time: {avg_time:.4f} ms (±{std_time:.4f} ms)")
    
    return avg_time


def main():
    """Main profiling function for the fused_moe kernel."""
    
    print("torch.ops.trtllm.fused_moe Kernel Profiling - Qwen3 235B Configuration")
    print("=" * 80)
    
    test_configs = []
    
    # Define base configurations for different TP/EP combinations
    tp_ep_configs = [
        # TP1_EP8: Our intermediate_size=1536, their FC1 shows 3072x4096 (w1+w3 concat), FC2 shows 4096x1536
        {'tp': 1, 'ep': 8, 'hidden_size': 4096, 'intermediate_size': 1536, 'num_experts': 16, 'top_k': 1},
        # TP2_EP4: Our intermediate_size=768, their FC1 shows 1536x4096 (w1+w3 concat), FC2 shows 4096x768
        {'tp': 2, 'ep': 4, 'hidden_size': 4096, 'intermediate_size': 768, 'num_experts': 32, 'top_k': 2},
        # TP4_EP2: Our intermediate_size=384, their FC1 shows 768x4096 (w1+w3 concat), FC2 shows 4096x384 
        {'tp': 4, 'ep': 2, 'hidden_size': 4096, 'intermediate_size': 384, 'num_experts': 64, 'top_k': 4},
        # TP8_EP1: Our intermediate_size=192, their FC1 shows 384x4096 (w1+w3 concat), FC2 shows 4096x192
        {'tp': 8, 'ep': 1, 'hidden_size': 4096, 'intermediate_size': 192, 'num_experts': 128, 'top_k': 8},
    ]
    
    # Token configurations with corresponding activated experts
    token_configs = [
        {'tokens': 4, 'activated_ratio': 0.25},    # 4/16, 8/32, 16/64, 32/128
        {'tokens': 16, 'activated_ratio': 1.0},    # 16/16, 32/32, 64/64, 128/128  
        {'tokens': 32, 'activated_ratio': 1.0},    # All experts activated
        {'tokens': 64, 'activated_ratio': 1.0},    # All experts activated
        {'tokens': 1024, 'activated_ratio': 1.0},  # All experts activated
        {'tokens': 2048, 'activated_ratio': 1.0},  # All experts activated
    ]
    
    # Generate all test configurations
    for tp_ep in tp_ep_configs:
        for token_config in token_configs:
            num_tokens = token_config['tokens']
            
            # Calculate activated experts based on ratio
            if token_config['activated_ratio'] == 0.25:
                num_activated_experts = tp_ep['num_experts'] // 4
            else:
                num_activated_experts = tp_ep['num_experts']
            
            # Single kernel test configuration
            test_configs.append({
                'test_name': f'Kernel_Qwen3_235B_TP{tp_ep["tp"]}_EP{tp_ep["ep"]}_MoE_tokens{num_tokens}',
                'num_tokens': num_tokens,
                'hidden_size': tp_ep['hidden_size'],
                'intermediate_size': tp_ep['intermediate_size'],
                'num_experts': tp_ep['num_experts'],
                'num_activated_experts': num_activated_experts,
                'top_k': tp_ep['top_k'],
                'benchmark_fc1_dims': f"{tp_ep['intermediate_size']*2}x{tp_ep['hidden_size']}",
                'benchmark_fc2_dims': f"{tp_ep['hidden_size']}x{tp_ep['intermediate_size']}"
            })
    
    # Run profiling for each configuration
    results = []
    for config in test_configs:
        # Filter out display-only fields for the function call
        profile_config = {k: v for k, v in config.items() 
                         if k not in ['benchmark_fc1_dims', 'benchmark_fc2_dims']}
        avg_time = profile_fused_moe_kernel(**profile_config)
        results.append({
            'testName': config['test_name'],
            'numTokens': config['num_tokens'],
            'hidden_size': config['hidden_size'],
            'intermediate_size': config['intermediate_size'],
            'numExperts': config['num_experts'],
            'numActivatedExperts': config['num_activated_experts'],
            'topK': config['top_k'],
            'time_ms': avg_time,
            'benchmark_fc1_dims': config['benchmark_fc1_dims'],
            'benchmark_fc2_dims': config['benchmark_fc2_dims']
        })
    
    # Print results table
    print("\n" + "=" * 100)
    print("FUSED MOE KERNEL PROFILING RESULTS")
    print("=" * 100)
    print("NOTE: This directly profiles torch.ops.trtllm.fused_moe kernel (FC1 + FC2 fused)")
    print("-" * 100)
    print(f"{'Tokens':<8} {'Test Name':<45} {'Hidden':<8} {'Intermediate':<12} {'Experts':<8} {'Activated':<10} {'TopK':<6} {'Time(ms)':<10} {'Maps to Benchmark':<25}")
    print("-" * 100)
    
    for result in results:
        benchmark_note = f"FC1({result['benchmark_fc1_dims']}) + FC2({result['benchmark_fc2_dims']})"
        print(f"{result['numTokens']:<8} {result['testName']:<45} {result['hidden_size']:<8} {result['intermediate_size']:<12} {result['numExperts']:<8} {result['numActivatedExperts']:<10} {result['topK']:<6} {result['time_ms']:<10.4f} {benchmark_note:<25}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This profiling script requires CUDA.")
        exit(1)
        
    main() 