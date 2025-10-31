import pytest
import torch

import tensorrt_llm  # noqa


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float8", "float4"])
def test_moe_permute(dtype: str, num_tokens: int, top_k: int, tile_size: int):
    hidden_size = 4096
    num_experts = 256
    x = torch.randint(-100,
                      100, (num_tokens, hidden_size),
                      dtype=torch.int32,
                      device="cuda")
    x_sf = None
    if dtype == "float4":
        x = x[:, :hidden_size // 2].to(torch.int8).view(torch.float4_e2m1fn_x2)
        x_sf = torch.randint(-100,
                             100, (num_tokens, hidden_size // 16),
                             dtype=torch.int32,
                             device="cuda").to(torch.float8_e4m3fn).view(
                                 torch.uint8)
    elif dtype == "float8":
        x = x.to(torch.float8_e4m3fn)
    else:
        x = x.to(getattr(torch, dtype))

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
    permuted_x, permuted_sf = torch.ops.trtllm.moe_permute(
        x, x_sf, permuted_idx_to_expanded_idx, num_non_exiting_tiles, tile_size,
        top_k)
    if dtype == "float4":
        assert permuted_sf is not None
    else:
        assert permuted_sf is None

    for i in range(num_permuted_tokens):
        if i >= num_non_exiting_tiles_val * tile_size:
            break
        expanded_idx = permuted_idx_to_expanded_idx[i].item()
        if expanded_idx < 0:
            continue
        token_idx = expanded_idx // top_k
        if dtype == "float4":
            torch.testing.assert_close(permuted_x[i].view(torch.uint8),
                                       x[token_idx].view(torch.uint8))
            torch.testing.assert_close(permuted_sf[i], x_sf[token_idx])
        else:
            torch.testing.assert_close(permuted_x[i], x[token_idx])
