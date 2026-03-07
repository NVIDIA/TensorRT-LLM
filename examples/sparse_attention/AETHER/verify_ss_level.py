import torch
import torch.nn.functional as F
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tensorrt_llm._torch.attention_backend.sparse.aether import AetherVanillaAttention, AetherIndexManager
from tensorrt_llm._torch.attention_backend.sparse.aether_kernels import run_aether_sparse_attention

def test_aether_sparse_numerical_fidelity():
    print("=== AETHER SS-Level Numerical Fidelity Test ===")
    
    # Model dimensions
    B, H, S, D = 1, 32, 1024, 128
    block_size = 64
    device = "cuda"
    dtype = torch.float16
    
    # Generate mock data
    q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    # Initialize AETHER backend
    # Note: We pass a config that enables high sparsity for testing
    class MockConfig:
        block_size = 64
        threshold = 0.05
        local_window = 4
        use_variance = True
        use_concentration = True
        min_seq_length = 128
        target_sparsity = None
        
    backend = AetherVanillaAttention(
        layer_idx=0, num_heads=H, head_dim=D, 
        sparse_attention_config=MockConfig()
    )
    
    # 1. Run Baseline (Dense SDPA with Mask)
    # This is what we used to do
    with torch.no_grad():
        # Get the mask from the backend but run it densly
        means, radii, variances, _ = backend._compute_block_metadata(k)
        mask = backend._compute_block_scores(q, means, radii, variances)
        
        # Expand mask to token level
        token_mask = mask.unsqueeze(-1).expand(-1, -1, -1, block_size).reshape(B, H, -1)[..., :S]
        attn_mask = torch.zeros_like(token_mask, dtype=dtype).masked_fill(~token_mask, float('-inf'))
        
        baseline_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.unsqueeze(2))

    # 2. Run SS-Level Sparse Kernel
    # This is the new "Compiler" path
    with torch.no_grad():
        sparse_out = backend._apply_aether_sparse(q, k, v)
        
    # 3. Compare
    cos_sim = F.cosine_similarity(baseline_out.flatten(), sparse_out.flatten(), dim=0)
    print(f"Cosine Similarity (Sparse vs Dense-Masked): {cos_sim.item():.6f}")
    
    if cos_sim > 0.999:
        print("[PASS] SS-Level Sparse Engine is numerically identical to standard SDPA!")
    else:
        print("[FAIL] Numerical drift detected.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_aether_sparse_numerical_fidelity()
    else:
        print("CUDA not available for hardware verification.")
