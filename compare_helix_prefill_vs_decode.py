import torch
from safetensors.torch import load_file

def compare_tensors(file_a, file_b):
    # Load tensors from safetensor files
    tensors_a = load_file(file_a)
    tensors_b = load_file(file_b)

    # Ensure both files contain the same tensor keys
    assert tensors_a.keys() == tensors_b.keys(), f"Tensor keys mismatch between {file_a} and {file_b}"

    for k in tensors_a.keys():
        t1 = tensors_a[k]
        t2 = tensors_b[k]

        print("t1: ", t1)
        print("t2: ", t2)

        # Check shape equality
        assert t1.shape == t2.shape, f"Shape mismatch for tensor {k}: {t1.shape} vs {t2.shape}"

        # Compute differences
        diff = t1 - t2
        max_diff = diff.abs().max().item()
        mean_diff = diff.abs().mean().item()

        print(f"Tensor: {k}")
        print(f"  Shape       : {t1.shape}")
        print(f"  Max diff    : {max_diff}")
        print(f"  Mean diff   : {mean_diff}\n")


if __name__ == "__main__":
    # File pairs to compare
    pairs = [
        (
            "prefill_helix_safekeep/request_2048/block_id_257_rank_0.safetensors",
            "decode_helix_rank_0/request_2048/block_id_0_rank_0.safetensors"
        ),
        (
            "prefill_helix_safekeep/request_2048/block_id_258_rank_0.safetensors",
            "decode_helix_rank_1/request_2048/block_id_0_rank_1.safetensors"
        ),
    ]

    for f1, f2 in pairs:
        print(f"Comparing {f1} vs {f2}...\n")
        compare_tensors(f1, f2)

