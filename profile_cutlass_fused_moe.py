import time
import torch
import numpy as np
from typing import List, Tuple, Dict

from tensorrt_llm._torch.modules.fused_moe import CutlassFusedMoE, DefaultMoeRoutingMethod
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping


def setup_cutlass_fused_moe(
    hidden_size: int,
    intermediate_size: int, 
    num_experts: int,
    top_k: int,
    dtype: torch.dtype = torch.bfloat16
) -> Tuple[CutlassFusedMoE, Dict]:
    """Setup CutlassFusedMoE with specified dimensions and return weights."""
    
    routing_method = DefaultMoeRoutingMethod(top_k=top_k)
    mapping = Mapping()
    mapping.rank = 0
    
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Create weights for all experts
    weights = {}
    for expert_id in range(num_experts):
        # FC1: w1 (gate) and w3 (up) projections
        w1_weight = torch.randn((intermediate_size, hidden_size), dtype=dtype).cuda()
        w3_weight = torch.randn((intermediate_size, hidden_size), dtype=dtype).cuda()
        # FC2: w2 (down) projection  
        w2_weight = torch.randn((hidden_size, intermediate_size), dtype=dtype).cuda()
        
        weights[f"{expert_id}.w1.weight"] = w1_weight
        weights[f"{expert_id}.w2.weight"] = w2_weight
        weights[f"{expert_id}.w3.weight"] = w3_weight
    
    # Create fused MoE module
    fused_moe = CutlassFusedMoE(
        num_experts=num_experts,
        routing_method=routing_method,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        reduce_results=True,
        model_config=ModelConfig(mapping=mapping),
    )
    fused_moe.load_weights([weights])
    fused_moe.cuda()
    
    return fused_moe, weights


def create_routing_logits(seq_len: int, num_experts: int, num_activated_experts: int, dtype: torch.dtype) -> torch.Tensor:
    """Create router logits that activate specified number of experts."""
    torch.manual_seed(42)
    
    # Create logits with strong bias towards first num_activated_experts
    router_logits = torch.full((seq_len, num_experts), -10.0, dtype=dtype).cuda()
    
    # Make first num_activated_experts have high logits in round-robin fashion
    for i in range(seq_len):
        expert_idx = i % num_activated_experts
        router_logits[i, expert_idx] = 10.0
        
    return router_logits


def profile_moe_configuration(
    test_name: str,
    num_tokens: int,
    hidden_size: int,  # k dimension
    intermediate_size: int,  # m dimension for FC1
    num_experts: int,
    num_activated_experts: int,
    top_k: int,
    num_warmup: int = 10,
    num_iterations: int = 100
) -> float:
    """Profile a specific MoE configuration and return average time in ms."""
    
    print(f"\nProfiling: {test_name}")
    print(f"  Tokens: {num_tokens}, Hidden: {hidden_size}, Intermediate: {intermediate_size}")
    print(f"  Experts: {num_experts}, Activated: {num_activated_experts}, TopK: {top_k}")
    
    # Setup MoE module
    fused_moe, _ = setup_cutlass_fused_moe(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        dtype=torch.bfloat16
    )
    
    # Create input tensors
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16).cuda()
    router_logits = create_routing_logits(num_tokens, num_experts, num_activated_experts, torch.bfloat16)
    
    # Warmup
    for _ in range(num_warmup):
        with torch.inference_mode():
            _ = fused_moe.forward(x, router_logits)
    
    torch.cuda.synchronize()
    
    # Timing
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            _ = fused_moe.forward(x, router_logits)
            
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Average time: {avg_time:.4f} ms (±{std_time:.4f} ms)")
    
    return avg_time


def main():
    """Main profiling function that replicates the benchmark setup."""
    
    print("CutlassFusedMoE Profiling - Qwen3 235B Configuration")
    print("=" * 60)
    
    # Test configurations based on the benchmark
    # Qwen3 235B parameters: hidden_size=7168, intermediate_size=18432 for full model
    # But the benchmark shows smaller dimensions, so using those
    
    test_configs = [
        # FC1 configurations (corresponds to w1/w3 projections)
        {
            'test_name': 'Qwen3_235B_TP1_EP8_MoE_FC1_tokens4',
            'num_tokens': 4,
            'hidden_size': 4096,  # k dimension 
            'intermediate_size': 3072,  # m dimension
            'num_experts': 16,
            'num_activated_experts': 4,
            'top_k': 1
        },
        # FC2 configurations (corresponds to w2 projection)  
        {
            'test_name': 'Qwen3_235B_TP1_EP8_MoE_FC2_tokens4',
            'num_tokens': 4,
            'hidden_size': 1536,  # k dimension
            'intermediate_size': 4096,  # m dimension  
            'num_experts': 16,
            'num_activated_experts': 4,
            'top_k': 1
        },
        # More token configurations
        {
            'test_name': 'Qwen3_235B_TP1_EP8_MoE_FC1_tokens16',
            'num_tokens': 16,
            'hidden_size': 4096,
            'intermediate_size': 3072,
            'num_experts': 16,
            'num_activated_experts': 8,
            'top_k': 1
        },
        {
            'test_name': 'Qwen3_235B_TP1_EP8_MoE_FC2_tokens16', 
            'num_tokens': 16,
            'hidden_size': 1536,
            'intermediate_size': 4096,
            'num_experts': 16,
            'num_activated_experts': 8,
            'top_k': 1
        },
        {
            'test_name': 'Qwen3_235B_TP1_EP8_MoE_FC1_tokens64',
            'num_tokens': 64,
            'hidden_size': 4096,
            'intermediate_size': 3072,
            'num_experts': 16,
            'num_activated_experts': 12,
            'top_k': 1
        },
        {
            'test_name': 'Qwen3_235B_TP1_EP8_MoE_FC2_tokens64',
            'num_tokens': 64,
            'hidden_size': 1536,
            'intermediate_size': 4096,
            'num_experts': 16,
            'num_activated_experts': 12,
            'top_k': 1
        },
        {
            'test_name': 'Qwen3_235B_TP1_EP8_MoE_FC1_tokens256',
            'num_tokens': 256,
            'hidden_size': 4096,
            'intermediate_size': 3072,
            'num_experts': 16,
            'num_activated_experts': 16,
            'top_k': 1
        },
        {
            'test_name': 'Qwen3_235B_TP1_EP8_MoE_FC2_tokens256',
            'num_tokens': 256,
            'hidden_size': 1536,
            'intermediate_size': 4096,
            'num_experts': 16,
            'num_activated_experts': 16,
            'top_k': 1
        },
        {
            'test_name': 'Qwen3_235B_TP1_EP8_MoE_FC1_tokens1024',
            'num_tokens': 1024,
            'hidden_size': 4096,
            'intermediate_size': 3072,
            'num_experts': 16,
            'num_activated_experts': 16,
            'top_k': 1
        },
        {
            'test_name': 'Qwen3_235B_TP1_EP8_MoE_FC2_tokens1024',
            'num_tokens': 1024,
            'hidden_size': 1536,
            'intermediate_size': 4096,
            'num_experts': 16,
            'num_activated_experts': 16,
            'top_k': 1
        }
    ]
    
    # Run profiling for each configuration
    results = []
    for config in test_configs:
        avg_time = profile_moe_configuration(**config)
        results.append({
            'testName': config['test_name'],
            'numTokens': config['num_tokens'],
            'm': config['intermediate_size'],
            'k': config['hidden_size'],
            'numExperts': config['num_experts'],
            'numActivatedExperts': config['num_activated_experts'],
            'topK': config['top_k'],
            'time_ms': avg_time
        })
    
    # Print results table
    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)
    print(f"{'numTokens':<10} {'testName':<40} {'m':<6} {'k':<6} {'numExperts':<12} {'numActivatedExperts':<20} {'topK':<6} {'time_ms':<10}")
    print("-" * 120)
    
    for result in results:
        print(f"{result['numTokens']:<10} {result['testName']:<40} {result['m']:<6} {result['k']:<6} {result['numExperts']:<12} {result['numActivatedExperts']:<20} {result['topK']:<6} {result['time_ms']:<10.4f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This profiling script requires CUDA.")
        exit(1)
        
    main() 