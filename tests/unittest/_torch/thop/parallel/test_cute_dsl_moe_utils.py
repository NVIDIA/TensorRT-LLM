import pytest
import torch

import tensorrt_llm  # noqa


@pytest.mark.parametrize("tile_size", [64, 128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_moe_permute(dtype: str, num_tokens: int, top_k: int, tile_size: int):
    dtype = getattr(torch, dtype)
    hidden_size = 4096
    num_experts = 256
    x = torch.randint(0,
                      100, (num_tokens, hidden_size),
                      dtype=dtype,
                      device="cuda")

    num_permuted_tokens = num_tokens * top_k + (tile_size - 1) * num_experts
    permuted_idx_to_expanded_idx = torch.randint(0,
                                                 num_tokens * top_k,
                                                 (num_permuted_tokens, ),
                                                 dtype=torch.int32,
                                                 device="cuda")
    num_non_exiting_tiles_val = (num_tokens * top_k) // tile_size
    num_non_exiting_tiles = torch.tensor([num_non_exiting_tiles_val],
                                         dtype=torch.int32,
                                         device="cuda")
    permuted_x, *_ = torch.ops.trtllm.moe_permute(x,
                                                  permuted_idx_to_expanded_idx,
                                                  num_non_exiting_tiles,
                                                  tile_size, top_k)

    for i in range(num_permuted_tokens):
        if i >= num_non_exiting_tiles_val * tile_size:
            break
        expanded_idx = permuted_idx_to_expanded_idx[i].item()
        if expanded_idx < 0:
            continue
        token_idx = expanded_idx // top_k
        torch.testing.assert_close(permuted_x[i], x[token_idx])
