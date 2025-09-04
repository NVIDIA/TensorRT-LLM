import torch
from safetensors.torch import load_file

def compare_safetensors(file1, file2):
    # Load safetensors dicts
    tensors1 = load_file(file1)
    tensors2 = load_file(file2)

    for key in tensors1.keys():
        if key not in tensors2:
            print(f"Key {key} not found in second file!")
            continue
        
        t1 = tensors1[key]
        t2 = tensors2[key]

        # Verify shapes
        if t1.shape != t2.shape:
            print(f"Shape mismatch for {key}: {t1.shape} vs {t2.shape}")
            continue

        # Compute differences
        print("t1: ", t1)
        print("t2: ", t2)
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"[{key}] shape: {t1.shape}, max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e}")

if __name__ == "__main__":
    # Block 257 comparison
    file_257_a = "prefill_helix_safekeep/request_2048/block_id_257_rank_0.safetensors"
    file_257_b = "prefill_helix_ref_safekeep/request_2048/block_id_257_rank_0.safetensors"
    
    print("Comparing Block 257...")
    compare_safetensors(file_257_a, file_257_b)

    # Block 258 comparison
    file_258_a = "prefill_helix_safekeep/request_2048/block_id_258_rank_0.safetensors"
    file_258_b = "prefill_helix_ref_safekeep/request_2048/block_id_258_rank_0.safetensors"
    
    print("\nComparing Block 258...")
    compare_safetensors(file_258_a, file_258_b)

