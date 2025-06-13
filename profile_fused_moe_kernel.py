import time
import torch
import numpy as np
import csv
import argparse
from typing import List, Tuple, Dict

from tensorrt_llm._torch.modules.fused_moe import DefaultMoeRoutingMethod
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping


import torch.cuda.nvtx as nvtx



def setup_fused_moe_kernel_inputs(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int, 
    num_experts: int,
    top_k: int,
    tp_size: int,
    ep_size: int,
    dtype: torch.dtype = torch.bfloat16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List, int, Mapping]:
    """Setup inputs for the torch.ops.trtllm.fused_moe kernel."""
    
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Create proper mapping for TP/EP configuration
    mapping = Mapping(
        world_size=tp_size * ep_size,
        rank=0,  # We're simulating rank 0
        tp_size=tp_size * ep_size,
        pp_size=1,
        moe_tp_size=tp_size,
        moe_ep_size=ep_size
    )
    
    # Calculate dimensions per rank
    # For TP: hidden dimensions are divided by tp_size
    # For EP: experts are divided by ep_size
    hidden_size_per_rank = hidden_size // tp_size
    intermediate_size_per_rank = intermediate_size // tp_size
    num_experts_per_rank = num_experts // ep_size
    
    # Input tensor (tokens stay the same, hidden_size might be partitioned for TP)
    x = torch.randn((num_tokens, hidden_size_per_rank), dtype=dtype).cuda()
    
    # Create routing method and apply it to get expert selections and scales
    routing_method = DefaultMoeRoutingMethod(top_k=top_k)
    router_logits, actual_activated_experts = create_routing_logits(num_tokens, num_experts, top_k, dtype)
    
    token_selected_experts, token_final_scales = routing_method.apply(router_logits)
    
    # For the kernel, token_selected_slots is the same as token_selected_experts in simple case
    token_selected_slots = token_selected_experts
    
    # Create expert weights for this rank
    # w3_w1_weight: concatenated w3 and w1 weights [num_experts_per_rank, 2*intermediate_size_per_rank, hidden_size_per_rank]
    w3_w1_weight = torch.randn((num_experts_per_rank, 2 * intermediate_size_per_rank, hidden_size_per_rank), dtype=dtype).cuda()
    
    # w2_weight: down projection weights [num_experts_per_rank, hidden_size_per_rank, intermediate_size_per_rank]  
    w2_weight = torch.randn((num_experts_per_rank, hidden_size_per_rank, intermediate_size_per_rank), dtype=dtype).cuda()
    
    # Quantization scales (empty for non-quantized)
    quant_scales = []
    
    return x, token_selected_slots, token_final_scales, w3_w1_weight, w2_weight, quant_scales, actual_activated_experts, mapping


def create_routing_logits(seq_len: int, num_experts: int, top_k: int, dtype: torch.dtype) -> Tuple[torch.Tensor, int]:
    """Create router logits that activate as many experts as possible.
    
    Returns:
        router_logits: Tensor with high logits for selected experts
        actual_activated_experts: Actual number of experts that will be activated
    """
    torch.manual_seed(42)
    
    # Calculate maximum number of experts we can activate
    # Limited by: total selections available and total experts
    total_selections = seq_len * top_k
    actual_activated_experts = min(total_selections, num_experts)
    
    # Create logits with strong bias towards first actual_activated_experts
    router_logits = torch.full((seq_len, num_experts), -10.0, dtype=dtype).cuda()
    
    # Round-robin assignment: for each token, assign top_k experts in round-robin fashion
    expert_counter = 0
    for token_idx in range(seq_len):
        for k in range(top_k):
            expert_idx = expert_counter % actual_activated_experts
            # Slight preference for earlier selections within top_k for each token
            router_logits[token_idx, expert_idx] = 10.0 - k * 0.1
            expert_counter += 1
        
    return router_logits, actual_activated_experts



@torch.inference_mode()
def profile_fused_moe_kernel(
    test_name: str,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    tp_size: int,
    ep_size: int,
    nsys_mode: bool = False,
    num_warmup: int = 10,
    num_iterations: int = 100
) -> float:
    """Profile the torch.ops.trtllm.fused_moe kernel directly."""
    
    print(f"\nProfiling: {test_name}")
    print(f"  Tokens: {num_tokens}, Hidden: {hidden_size}, Intermediate: {intermediate_size}")
    
    # Setup kernel inputs
    x, token_selected_slots, token_final_scales, w3_w1_weight, w2_weight, quant_scales, actual_activated_experts, mapping = setup_fused_moe_kernel_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        tp_size=tp_size,
        ep_size=ep_size,
        dtype=torch.bfloat16
    )
    
    print(f"  Experts: {num_experts}, Activated: {actual_activated_experts}, TopK: {top_k}")
    
    # Adjust parameters for nsys mode
    if nsys_mode:
        num_warmup = 0
        num_iterations = 1
        print(f"  NSYS Mode: No warmup, 1 iteration")
    
    output_dtype = torch.bfloat16
    
    # Kernel parameters from mapping
    tp_rank = mapping.rank % tp_size
    ep_rank = mapping.rank // tp_size
    # cluster_size = tp_size * ep_size
    # cluster_rank = mapping.rank
    use_deepseek_fp8_block_scale = False
    use_w4a8_group_scaling = False
    min_latency_mode = False
    tune_max_num_tokens = 8192
    
    # Warmup
    if num_warmup > 0:
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
                # cluster_size=cluster_size,
                # cluster_rank=cluster_rank,
                use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
                use_w4a8_group_scaling=use_w4a8_group_scaling,
                min_latency_mode=min_latency_mode,
                tune_max_num_tokens=tune_max_num_tokens,
            )
        
        torch.cuda.synchronize()
    
    # Timing using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    if nsys_mode:
        nvtx_range = nvtx.range_start(test_name)
    
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
            # cluster_size=cluster_size,
            # cluster_rank=cluster_rank,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4a8_group_scaling=use_w4a8_group_scaling,
            min_latency_mode=min_latency_mode,
            tune_max_num_tokens=tune_max_num_tokens,
        )
    end_event.record()
    
    if nsys_mode:
        nvtx.range_end(nvtx_range)
    
    end_event.synchronize()
    total_time = start_event.elapsed_time(end_event)  # Total time in ms
    avg_time = total_time / num_iterations
    std_time = 0.0  # We can't calculate std deviation with this approach
    
    print(f"  Average time: {avg_time:.4f} ms (±{std_time:.4f} ms)")
    
    return avg_time


def main():
    """Main profiling function for the fused_moe kernel."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Profile fused MoE kernel',
        epilog='''
Example usage:
  Regular profiling:
    python profile_fused_moe_kernel.py
    
  NSYS profiling:
    nsys profile -o moe_cutlass_kernels -t nvtx,cuda -c cudaProfilerApi -f true python profile_fused_moe_kernel.py --nsys
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--nsys', action='store_true', 
                       help='Enable nsys profiling mode (no warmup, 1 iteration, NVTX markers, CUDA profiler start/stop)')
    args = parser.parse_args()
    
    nsys_mode = args.nsys
    
    print("torch.ops.trtllm.fused_moe Kernel Profiling - Qwen3 235B Configuration")
    print("=" * 80)
    if nsys_mode:
        print("NSYS PROFILING MODE ENABLED")
        print("- No warmup iterations")
        print("- Single iteration per test")
        print("- NVTX markers for timeline analysis")
        print("- CUDA profiler start/stop for precise kernel capture")
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
    
    # Token configurations
    token_configs = [4, 16, 32, 64, 1024, 2048]
    
    # Generate all test configurations
    for tp_ep in tp_ep_configs:
        for num_tokens in token_configs:
            
            # Single kernel test configuration
            test_configs.append({
                'test_name': f'Kernel_Qwen3_235B_TP{tp_ep["tp"]}_EP{tp_ep["ep"]}_MoE_tokens{num_tokens}',
                'num_tokens': num_tokens,
                'hidden_size': tp_ep['hidden_size'],
                'intermediate_size': tp_ep['intermediate_size'],
                'num_experts': tp_ep['num_experts'],
                'top_k': tp_ep['top_k'],
                'tp_size': tp_ep['tp'],
                'ep_size': tp_ep['ep'],
                'benchmark_fc1_dims': f"{tp_ep['intermediate_size']*2}x{tp_ep['hidden_size']}",
                'benchmark_fc2_dims': f"{tp_ep['hidden_size']}x{tp_ep['intermediate_size']}"
            })
    
    # Run profiling for each configuration
    results = []
    if nsys_mode:  
        torch.cuda.profiler.start()
    for config in test_configs:
        # Filter out display-only fields for the function call
        profile_config = {k: v for k, v in config.items() 
                         if k not in ['benchmark_fc1_dims', 'benchmark_fc2_dims']}
        profile_config['nsys_mode'] = nsys_mode
        avg_time = profile_fused_moe_kernel(**profile_config)
        results.append({
            'testName': config['test_name'],
            'numTokens': config['num_tokens'],
            'hidden_size': config['hidden_size'],
            'intermediate_size': config['intermediate_size'],
            'numExperts': config['num_experts'],
            'topK': config['top_k'],
            'time_ms': avg_time,
            'benchmark_fc1_dims': config['benchmark_fc1_dims'],
            'benchmark_fc2_dims': config['benchmark_fc2_dims']
        })
    if nsys_mode:
        torch.cuda.profiler.stop()
    
    # Sort results to match benchmark order: tokens -> TP config
    results.sort(key=lambda x: (x['numTokens'], x['topK'], x['testName']))
    
    # Export to CSV in benchmark format
    csv_filename = "fused_moe_kernel_profiling_results_nsys.csv" if nsys_mode else "fused_moe_kernel_profiling_results.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header matching their benchmark format  
        writer.writerow(['numTokens', 'testName', 'm', 'k', 'numExperts', 'numActivatedExperts', 'topK', 'time_ms'])
        
        # Write data rows - each of our tests represents both FC1 and FC2 from their benchmark
        for result in results:
            # Since our kernel fuses FC1+FC2, we create one entry that represents the combined operation
            # Use FC1 dimensions (m = 2*intermediate_size) to match their benchmark format
            fc1_m = result['intermediate_size'] * 2  # w1+w3 concatenated
            fc1_k = result['hidden_size']
            
            # Calculate actual activated experts (will be determined at runtime)
            actual_activated_experts = min(result['numTokens'] * result['topK'], result['numExperts'])
            
            writer.writerow([
                result['numTokens'],
                result['testName'],
                fc1_m,  # Using FC1 m dimension for consistency with benchmark
                fc1_k,  # Using FC1 k dimension
                result['numExperts'],
                actual_activated_experts,  # Calculate based on tokens*topK
                result['topK'],
                f"{result['time_ms']:.4f}"
            ])
    
    print(f"\nResults exported to: {csv_filename}")
    
    # Print results table
    print("\n" + "=" * 100)
    print("FUSED MOE KERNEL PROFILING RESULTS")
    print("=" * 100)
    print("NOTE: This directly profiles torch.ops.trtllm.fused_moe kernel (FC1 + FC2 fused)")
    print("-" * 100)
    print(f"{'Tokens':<8} {'Test Name':<45} {'Hidden':<8} {'Intermediate':<12} {'Experts':<8} {'TopK':<6} {'Time(ms)':<10} {'Maps to Benchmark':<25}")
    print("-" * 100)
    
    for result in results:
        benchmark_note = f"FC1({result['benchmark_fc1_dims']}) + FC2({result['benchmark_fc2_dims']})"
        print(f"{result['numTokens']:<8} {result['testName']:<45} {result['hidden_size']:<8} {result['intermediate_size']:<12} {result['numExperts']:<8} {result['topK']:<6} {result['time_ms']:<10.4f} {benchmark_note:<25}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This profiling script requires CUDA.")
        exit(1)
        
    main() 