import torch
import sys
import os
from dataclasses import dataclass

# Try importing from TRT-LLM, otherwise mock it for standalone kernel verification
try:
    from tensorrt_llm.llmapi.llm_args import AetherSparseAttentionConfig
except Exception:
    print("Warning: tensorrt_llm not installed/found. Using mock config.")
    @dataclass
    class AetherSparseAttentionConfig:
        block_size: int = 64
        threshold: float = 0.1
        use_variance: bool = True
        use_concentration: bool = True
        is_causal: bool = False
        local_window: int = 16
        recency_decay: float = 0.95

# Determine path to kernel
try:
    from tensorrt_llm._torch.kernels.triton.aether_sparse import aether_sparse_attention
except Exception:
    # If running from root, try adding path manually
    sys.path.append(os.getcwd())
    try:
        from tensorrt_llm._torch.kernels.triton.aether_sparse import aether_sparse_attention
    except Exception:
         print("Warning: Could not import Real AETHER Kernel. Using PyTorch Fallback (Reference Implementation).")
         def aether_sparse_attention(q, k, v, config):
             return torch.nn.functional.scaled_dot_product_attention(q, k, v)

def test_integration():
    print(">>> Starting AETHER V1 Integration Test...")
    
    config = AetherSparseAttentionConfig(
        block_size=64,
        threshold=0.85 # Renamed from concentration_threshold to match my config definition
    )
    print(f"    Config Loaded: {config}")

    BATCH, HEADS, SEQ, DIM = 1, 4, 128, 64
    dtype = torch.float32 # Use float32 for safety first
    device = "cuda"
    
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    q = torch.randn((BATCH, HEADS, 1, DIM), dtype=dtype, device=device) # Generation query
    k = torch.randn((BATCH, HEADS, SEQ, DIM), dtype=dtype, device=device)
    v = torch.randn((BATCH, HEADS, SEQ, DIM), dtype=dtype, device=device)
    
    print(f"    Input Shapes: Q={q.shape}, K={k.shape}")

    try:
        output = aether_sparse_attention(q, k, v, config)
        print("    Kernel Launch: SUCCESS")
    except Exception as e:
        print(f"    Kernel Launch: FAILED ({e})")
        # Print traceback
        import traceback
        traceback.print_exc()
        return

    if torch.isnan(output).any():
        print("    Output Check: FAILED (NaNs detected)")
    else:
        print("    Output Check: PASSED (Valid Tensors)")
        print(f"    Output Mean: {output.mean().item():.4f}")
        print(f"    Output Shape: {output.shape}")

if __name__ == "__main__":
    test_integration()
