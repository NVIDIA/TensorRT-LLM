#!/usr/bin/env python3
"""Debug script to understand AETHER mask behavior."""

import torch
import torch.nn.functional as F
import sys
import os

kernel_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "tensorrt_llm", "_torch", "kernels"
)
sys.path.insert(0, kernel_path)

from aether_sparse import (
    aether_sparse_attention, AetherConfig, precompute_block_metadata, compute_block_scores
)

# Test configuration
B, H, S, D = 1, 8, 512, 64
q_len = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Device: {device}")

# Create test tensors
torch.manual_seed(42)
q = torch.randn(B, H, q_len, D, device=device, dtype=dtype)
k = torch.randn(B, H, S, D, device=device, dtype=dtype)
v = torch.randn(B, H, S, D, device=device, dtype=dtype)

# Compute metadata
means, radii, variances = precompute_block_metadata(k, block_size=64)
num_blocks = means.shape[2]
print(f"Number of blocks: {num_blocks}")

# Compute block scores and mask
config = AetherConfig(threshold=0.05, local_window=8)
block_mask = compute_block_scores(
    q, means, radii, variances,
    threshold=config.threshold,
    use_variance=config.use_variance,
)

print(f"Block mask shape: {block_mask.shape}")
print(f"Block mask dtype: {block_mask.dtype}")

# Check how many blocks are kept
blocks_kept = block_mask.float().sum(dim=-1).mean().item()
total_blocks = block_mask.shape[-1]
sparsity = 1.0 - (blocks_kept / total_blocks)
print(f"Blocks kept (avg): {blocks_kept:.1f} / {total_blocks}")
print(f"Sparsity: {sparsity:.2%}")

# Check attention output
output = aether_sparse_attention(q, k, v, config=config, is_causal=True)
dense_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

cos_sim = F.cosine_similarity(output.flatten().float(), dense_output.flatten().float(), dim=0).item()
print(f"Cosine similarity: {cos_sim:.4f}")

# If sparsity is too high, that's the problem
if sparsity > 0.5:
    print("\n[DIAGNOSIS] High sparsity is causing quality loss!")
    print("Solution: Lower threshold or increase local_window")
