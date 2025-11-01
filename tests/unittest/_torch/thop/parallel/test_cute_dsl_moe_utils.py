import pytest
import torch

from tensorrt_llm._torch.utils import unswizzle_sf


def get_max_num_tiles(num_tokens: int, top_k: int, tile_size: int,
                      num_local_experts: int) -> int:
    num_expanded_tokens = num_tokens * top_k
    if num_expanded_tokens <= num_local_experts:
        return num_expanded_tokens
    return (num_expanded_tokens +
            (tile_size - 1) * num_local_experts) // tile_size


def get_max_num_permuted_tokens(num_tokens: int, top_k: int, tile_size: int,
                                num_local_experts: int) -> int:
    return get_max_num_tiles(num_tokens, top_k, tile_size,
                             num_local_experts) * tile_size


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float8", "float4"])
def test_moe_permute(dtype: str, num_tokens: int, top_k: int, tile_size: int):
    sf_vec_size = 16
    hidden_size = 4096
    num_local_experts = 256 // 32
    x = torch.randint(-100,
                      100, (num_tokens, hidden_size),
                      dtype=torch.int32,
                      device="cuda")
    x_sf = None
    if dtype == "float4":
        x = x[:, :hidden_size // 2].to(torch.int8).view(torch.float4_e2m1fn_x2)
        x_sf = torch.randint(-100,
                             100, (num_tokens, hidden_size // sf_vec_size),
                             dtype=torch.int32,
                             device="cuda").to(torch.float8_e4m3fn).view(
                                 torch.uint8)
    elif dtype == "float8":
        x = x.to(torch.float8_e4m3fn)
    else:
        x = x.to(getattr(torch, dtype))

    num_permuted_tokens = get_max_num_permuted_tokens(num_tokens, top_k,
                                                      tile_size,
                                                      num_local_experts)
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
        permuted_sf = unswizzle_sf(permuted_sf, num_permuted_tokens,
                                   hidden_size, sf_vec_size)
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


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_moe_unpermute(dtype: str, num_tokens: int, top_k: int, tile_size: int):
    hidden_size = 4096
    num_local_experts = 256 // 32
    num_permuted_tokens = get_max_num_permuted_tokens(num_tokens, top_k,
                                                      tile_size,
                                                      num_local_experts)
    permuted_x = torch.randint(-100 // top_k,
                               100 // top_k, (num_permuted_tokens, hidden_size),
                               dtype=torch.int32,
                               device="cuda").to(getattr(torch, dtype))

    expanded_idx_to_permuted_idx = torch.randint(0,
                                                 num_permuted_tokens,
                                                 (num_tokens, top_k),
                                                 dtype=torch.int32,
                                                 device="cuda")
    x = torch.ops.trtllm.moe_unpermute(permuted_x, expanded_idx_to_permuted_idx)
    torch.testing.assert_close(
        x, permuted_x[expanded_idx_to_permuted_idx].sum(dim=1))
