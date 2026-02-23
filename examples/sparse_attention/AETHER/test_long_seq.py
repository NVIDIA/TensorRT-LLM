#!/usr/bin/env python3
"""Test AETHER with longer sequences to demonstrate sparsity."""

import torch
import torch.nn.functional as F
import sys
import os

kernel_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "tensorrt_llm", "_torch", "kernels"
)
sys.path.insert(0, kernel_path)

from aether_sparse import aether_sparse_attention, AetherConfig

print("=" * 60)
print("AETHER LONG SEQUENCE TEST")
print("=" * 60)

device = "cuda"
dtype = torch.float16

# Test with various sequence lengths
for S in [512, 1024, 2048, 4096]:
    B, H, D = 1, 32, 128
    q_len = 1
    
    q = torch.randn(B, H, q_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    # Configure for moderate sparsity
    config = AetherConfig(
        block_size=128,
        threshold=0.3,  # More aggressive for sparsity
        local_window=4,
    )
    
    # Run AETHER
    output = aether_sparse_attention(q, k, v, config=config, is_causal=True)
    
    # Run dense
    dense_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Compare
    cos_sim = F.cosine_similarity(
        output.flatten().float(), 
        dense_output.flatten().float(), 
        dim=0
    ).item()
    
    # Benchmark
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    for _ in range(100):
        _ = aether_sparse_attention(q, k, v, config=config, is_causal=True)
    torch.cuda.synchronize()
    aether_ms = (time.perf_counter() - start) / 100 * 1000
    
    start = time.perf_counter()
    for _ in range(100):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    dense_ms = (time.perf_counter() - start) / 100 * 1000
    
    speedup = dense_ms / aether_ms if aether_ms > 0 else 0
    
    print(f"\nS={S:4d}: CosSim={cos_sim:.4f} | AETHER={aether_ms:.3f}ms | Dense={dense_ms:.3f}ms | Speedup={speedup:.2f}x")

print("\n" + "=" * 60)
print("LONG SEQUENCE TESTS COMPLETE")
print("=" * 60)
