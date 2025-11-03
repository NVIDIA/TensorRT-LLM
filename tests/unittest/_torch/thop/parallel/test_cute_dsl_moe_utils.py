import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.routing import RoutingMethodType
from tensorrt_llm._torch.utils import unswizzle_sf


def get_max_num_tiles(num_tokens: int, top_k: int, tile_size: int, num_local_experts: int) -> int:
    num_expanded_tokens = num_tokens * top_k
    if num_expanded_tokens <= num_local_experts:
        return num_expanded_tokens
    return (num_expanded_tokens + (tile_size - 1) * num_local_experts) // tile_size


def get_max_num_permuted_tokens(
    num_tokens: int, top_k: int, tile_size: int, num_local_experts: int
) -> int:
    return get_max_num_tiles(num_tokens, top_k, tile_size, num_local_experts) * tile_size


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_moe_sort(num_tokens: int, top_k: int, tile_size: int):
    num_experts = 256
    num_local_experts = 256 // 32

    routing_logits = torch.randn(num_tokens, num_experts, device="cuda")
    token_final_scales, token_selected_experts = routing_logits.topk(top_k, dim=-1)
    token_selected_experts = token_selected_experts.to(torch.int32)
    token_final_scales = token_final_scales.softmax(dim=-1).to(torch.bfloat16)

    (
        tile_idx_to_batch_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = torch.ops.trtllm.moe_sort(
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        n_group=1,
        topk_group=1,
        local_expert_offset=0,
        local_num_experts=num_local_experts,
        routed_scaling_factor=1.0,
        tile_tokens_dim=tile_size,
        routing_method_type=RoutingMethodType.DeepSeekV3,
    )

    num_tokens_per_expert = torch.bincount(token_selected_experts.flatten(), minlength=num_experts)
    num_tokens_per_expert = num_tokens_per_expert[:num_local_experts]
    num_tiles_per_expert = (num_tokens_per_expert + tile_size - 1) // tile_size
    num_tokens_per_expert = num_tokens_per_expert.cpu()
    num_tiles_per_expert = num_tiles_per_expert.cpu()

    max_num_tiles = get_max_num_tiles(num_tokens, top_k, tile_size, num_local_experts)
    num_valid_tiles = num_tiles_per_expert.sum().item()
    max_num_permuted_tokens = get_max_num_permuted_tokens(
        num_tokens, top_k, tile_size, num_local_experts
    )
    num_valid_permuted_tokens = num_tiles_per_expert.sum().item() * tile_size
    assert num_valid_tiles <= max_num_tiles
    assert num_valid_permuted_tokens <= max_num_permuted_tokens

    tile_idx_to_batch_idx = tile_idx_to_batch_idx.cpu()
    tile_idx_to_mn_limit = tile_idx_to_mn_limit.cpu()
    assert tile_idx_to_batch_idx.size() == (max_num_tiles,)
    assert tile_idx_to_mn_limit.size() == (max_num_tiles,)
    tile_idx = 0
    for expert_idx in range(num_local_experts):
        num_remaining_tokens = num_tokens_per_expert[expert_idx].item()
        for i in range(num_tiles_per_expert[expert_idx].item()):
            mn_limit = tile_idx * tile_size
            if i < num_tiles_per_expert[expert_idx].item() - 1:
                assert num_remaining_tokens > tile_size
                num_remaining_tokens -= tile_size
                mn_limit += tile_size
            else:
                assert 0 < num_remaining_tokens <= tile_size
                mn_limit += num_remaining_tokens
            assert tile_idx_to_batch_idx[tile_idx].item() == expert_idx
            assert tile_idx_to_mn_limit[tile_idx].item() == mn_limit
            tile_idx += 1

    token_selected_experts = token_selected_experts.cpu()
    expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.cpu()
    permuted_idx_to_expanded_idx = permuted_idx_to_expanded_idx.cpu()
    assert expanded_idx_to_permuted_idx.size() == (num_tokens, top_k)
    assert permuted_idx_to_expanded_idx.size() == (max_num_permuted_tokens,)
    for i in range(num_tokens):
        for k in range(top_k):
            expert_idx = token_selected_experts[i, k].item()
            expanded_idx = i * top_k + k
            permuted_idx = expanded_idx_to_permuted_idx[i, k].item()
            if expert_idx >= num_local_experts:
                assert permuted_idx == -1
            else:
                assert permuted_idx >= 0
                assert permuted_idx_to_expanded_idx[permuted_idx].item() == expanded_idx
                tile_idx = permuted_idx // tile_size
                assert tile_idx_to_batch_idx[tile_idx].item() == expert_idx

    for i in range(num_valid_permuted_tokens):
        tile_idx = i // tile_size
        if i < tile_idx_to_mn_limit[tile_idx].item():
            expanded_idx = permuted_idx_to_expanded_idx[i].item()
            token_idx = expanded_idx // top_k
            topk_idx = expanded_idx % top_k
            assert expanded_idx_to_permuted_idx[token_idx, topk_idx].item() == i

    assert total_num_padded_tokens.size() == (1,)
    assert total_num_padded_tokens[0].item() == num_valid_permuted_tokens
    assert num_non_exiting_tiles.size() == (1,)
    assert num_non_exiting_tiles[0].item() == num_valid_tiles


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float8", "float4"])
def test_moe_permute(dtype: str, num_tokens: int, top_k: int, tile_size: int):
    sf_vec_size = 16
    hidden_size = 4096
    num_local_experts = 256 // 32
    x = torch.randint(-100, 100, (num_tokens, hidden_size), dtype=torch.int32, device="cuda")
    x_sf = None
    if dtype == "float4":
        x = x[:, : hidden_size // 2].to(torch.int8).view(torch.float4_e2m1fn_x2)
        x_sf = (
            torch.randint(
                -100,
                100,
                (num_tokens, hidden_size // sf_vec_size),
                dtype=torch.int32,
                device="cuda",
            )
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
    elif dtype == "float8":
        x = x.to(torch.float8_e4m3fn)
    else:
        x = x.to(getattr(torch, dtype))

    max_num_tiles = get_max_num_tiles(num_tokens, top_k, tile_size, num_local_experts)
    max_num_permuted_tokens = get_max_num_permuted_tokens(
        num_tokens, top_k, tile_size, num_local_experts
    )
    tile_idx_to_mn_limit = (
        torch.arange(1, max_num_tiles + 1, dtype=torch.int32, device="cuda") * tile_size
    )
    permuted_idx_to_expanded_idx = torch.randint(
        0, num_tokens * top_k, (max_num_permuted_tokens,), dtype=torch.int32, device="cuda"
    )
    num_non_exiting_tiles_val = (num_tokens * top_k) // tile_size
    num_non_exiting_tiles = torch.tensor(
        [num_non_exiting_tiles_val], dtype=torch.int32, device="cuda"
    )
    permuted_x, permuted_sf = torch.ops.trtllm.moe_permute(
        x,
        x_sf,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        tile_size,
        top_k,
    )
    if dtype == "float4":
        assert permuted_sf is not None
        permuted_sf = unswizzle_sf(permuted_sf, max_num_permuted_tokens, hidden_size, sf_vec_size)
    else:
        assert permuted_sf is None

    for i in range(max_num_permuted_tokens):
        if i >= num_non_exiting_tiles_val * tile_size:
            break
        expanded_idx = permuted_idx_to_expanded_idx[i].item()
        if expanded_idx < 0:
            continue
        token_idx = expanded_idx // top_k
        if dtype == "float4":
            torch.testing.assert_close(
                permuted_x[i].view(torch.uint8), x[token_idx].view(torch.uint8)
            )
            torch.testing.assert_close(permuted_sf[i], x_sf[token_idx])
        else:
            torch.testing.assert_close(permuted_x[i], x[token_idx])


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_moe_unpermute(dtype: str, num_tokens: int, top_k: int, tile_size: int):
    dtype = getattr(torch, dtype)
    hidden_size = 4096
    num_local_experts = 256 // 32
    num_permuted_tokens = get_max_num_permuted_tokens(
        num_tokens, top_k, tile_size, num_local_experts
    )
    permuted_x = torch.randint(
        -100, 100, (num_permuted_tokens, hidden_size), dtype=torch.int32, device="cuda"
    ).to(dtype)

    expanded_idx_to_permuted_idx = torch.randint(
        0, num_permuted_tokens, (num_tokens, top_k), dtype=torch.int32, device="cuda"
    )
    topk_scales = torch.randn(num_tokens, top_k, dtype=torch.float32, device="cuda").softmax(dim=-1)
    x = torch.ops.trtllm.moe_unpermute(permuted_x, expanded_idx_to_permuted_idx, topk_scales)

    x_ref = (
        (permuted_x[expanded_idx_to_permuted_idx] * topk_scales.unsqueeze(-1)).sum(dim=1).to(dtype)
    )
    torch.testing.assert_close(x, x_ref)
