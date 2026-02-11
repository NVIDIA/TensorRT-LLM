import pytest
import torch
from utils.util import check_accuracy

from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import GroupedGemmInputsHelper
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import cute_dsl_nvfp4_grouped_gemm_ref
from tensorrt_llm._torch.modules.fused_moe.quantization import interleave_linear_and_gate
from tensorrt_llm._torch.utils import swizzle_sf, unswizzle_sf
from tensorrt_llm._utils import get_sm_version


def swiglu_ref(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return x * torch.nn.functional.silu(gate)


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 6, 8])
def test_grouped_gemm_inputs_helper(top_k: int, ep_size: int, tile_size: int):
    num_experts = 256
    num_local_experts = num_experts // ep_size

    helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
    max_num_tokens = 8192
    num_tokens_list = list(range(1, max_num_tokens + 1))
    max_num_permuted_tokens_list = [helper.get_max_num_permuted_tokens(x) for x in num_tokens_list]
    num_inferred_tokens_list = [helper.infer_num_tokens(x) for x in max_num_permuted_tokens_list]

    for i in range(max_num_tokens):
        assert num_inferred_tokens_list[i] >= num_tokens_list[i]
        assert num_inferred_tokens_list[i] < num_tokens_list[i] + tile_size
        if i > 0:
            assert num_inferred_tokens_list[i] >= num_inferred_tokens_list[i - 1]

    buckets = helper.gen_tuning_buckets(max_num_permuted_tokens_list[-1])
    assert set([helper.map_to_tuning_buckets(x) for x in max_num_permuted_tokens_list]) == set(
        buckets
    )


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_moe_sort(num_tokens: int, top_k: int, ep_size: int, tile_size: int):
    num_experts = 256
    num_local_experts = num_experts // ep_size

    routing_logits = torch.randn(num_tokens, num_experts, device="cuda")
    token_final_scales, token_selected_experts = routing_logits.topk(top_k, dim=-1)
    token_selected_experts = token_selected_experts.to(torch.int32)
    token_final_scales = token_final_scales.softmax(dim=-1).to(torch.bfloat16)

    (
        tile_idx_to_group_idx,
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
        local_expert_offset=0,
        local_num_experts=num_local_experts,
        tile_tokens_dim=tile_size,
    )

    num_tokens_per_expert = torch.bincount(token_selected_experts.flatten(), minlength=num_experts)
    num_tokens_per_expert = num_tokens_per_expert[:num_local_experts]
    num_tiles_per_expert = (num_tokens_per_expert + tile_size - 1) // tile_size
    num_tokens_per_expert = num_tokens_per_expert.cpu()
    num_tiles_per_expert = num_tiles_per_expert.cpu()

    helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
    max_num_tiles = helper.get_max_num_tiles(num_tokens)
    max_num_permuted_tokens = helper.get_max_num_permuted_tokens(num_tokens)
    num_valid_tiles = num_tiles_per_expert.sum().item()
    num_valid_permuted_tokens = num_valid_tiles * tile_size
    assert 0 <= num_valid_tiles <= max_num_tiles
    assert 0 <= num_valid_permuted_tokens <= max_num_permuted_tokens

    tile_idx_to_group_idx = tile_idx_to_group_idx.cpu()
    tile_idx_to_mn_limit = tile_idx_to_mn_limit.cpu()
    assert tile_idx_to_group_idx.size() == (max_num_tiles,)
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
            assert tile_idx_to_group_idx[tile_idx].item() == expert_idx
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
                assert tile_idx_to_group_idx[tile_idx].item() == expert_idx

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
    num_experts = 256
    num_local_experts = num_experts // 32
    x = torch.randint(-100, 100, (num_tokens, hidden_size), dtype=torch.int32, device="cuda")
    x_sf = None
    if dtype == "float4":
        x = x[:, : hidden_size // 2].to(torch.int8).view(torch.float4_e2m1fn_x2)
        x_sf = torch.randint(
            -100, 100, (num_tokens, hidden_size // sf_vec_size), dtype=torch.int32, device="cuda"
        )
        x_sf = x_sf.to(torch.float8_e4m3fn).view(torch.uint8)
    elif dtype == "float8":
        x = x.to(torch.float8_e4m3fn)
    else:
        x = x.to(getattr(torch, dtype))

    helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
    max_num_tiles = helper.get_max_num_tiles(num_tokens)
    max_num_permuted_tokens = helper.get_max_num_permuted_tokens(num_tokens)
    tile_idx_to_mn_limit = (
        torch.arange(1, max_num_tiles + 1, dtype=torch.int32, device="cuda") * tile_size
    )
    permuted_idx_to_expanded_idx = torch.randint(
        0, num_tokens * top_k, (max_num_permuted_tokens,), dtype=torch.int32, device="cuda"
    )
    num_non_exiting_tiles_val = (num_tokens * top_k + tile_size - 1) // tile_size
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
    num_experts = 256
    num_local_experts = num_experts // 32
    helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
    max_num_permuted_tokens = helper.get_max_num_permuted_tokens(num_tokens)
    permuted_x = torch.randint(
        -100, 100, (max_num_permuted_tokens, hidden_size), dtype=torch.int32, device="cuda"
    ).to(dtype)

    expanded_idx_to_permuted_idx = torch.randint(
        0, max_num_permuted_tokens, (num_tokens, top_k), dtype=torch.int32, device="cuda"
    )
    topk_scales = torch.randn(num_tokens, top_k, dtype=torch.float32, device="cuda").softmax(dim=-1)
    x = torch.ops.trtllm.moe_unpermute(permuted_x, expanded_idx_to_permuted_idx, topk_scales)

    x_ref = (
        (permuted_x[expanded_idx_to_permuted_idx] * topk_scales.unsqueeze(-1)).sum(dim=1).to(dtype)
    )
    torch.testing.assert_close(x, x_ref)


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_moe_output_memset_inplace(
    dtype: str, num_tokens: int, top_k: int, ep_size: int, tile_size: int
):
    dtype = getattr(torch, dtype)
    hidden_size = 4096
    num_experts = 256
    num_local_experts = num_experts // ep_size
    enable_alltoall = True

    routing_logits = torch.randn(num_tokens, num_experts, device="cuda")
    token_final_scales, token_selected_experts = routing_logits.topk(top_k, dim=-1)
    token_selected_experts = token_selected_experts.to(torch.int32)
    token_final_scales = token_final_scales.softmax(dim=-1).to(torch.float32)

    (
        tile_idx_to_group_idx,
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
        local_expert_offset=0,
        local_num_experts=num_local_experts,
        tile_tokens_dim=tile_size,
    )

    x = torch.ones(num_tokens, hidden_size, dtype=dtype, device="cuda")
    torch.ops.trtllm.moe_output_memset_inplace(
        x,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        tile_size,
        top_k,
        ep_size,
        enable_alltoall=enable_alltoall,
    )
    x_ref = torch.zeros_like(x)
    if enable_alltoall and ep_size > top_k:
        x_ref[(expanded_idx_to_permuted_idx < 0).all(dim=-1)] = 1
    torch.testing.assert_close(x, x_ref)


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_moe_swiglu(dtype: str, num_tokens: int, top_k: int, tile_size: int):
    dtype = getattr(torch, dtype)
    interm_size = 4096
    num_experts = 256
    num_local_experts = num_experts // 32
    helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
    max_num_tiles = helper.get_max_num_tiles(num_tokens)
    max_num_permuted_tokens = helper.get_max_num_permuted_tokens(num_tokens)

    x = torch.randint(
        -100, 100, (max_num_permuted_tokens, interm_size * 2), dtype=torch.int32, device="cuda"
    ).to(dtype)
    tile_idx_to_mn_limit = (
        torch.arange(1, max_num_tiles + 1, dtype=torch.int32, device="cuda") * tile_size
    )
    num_non_exiting_tiles_val = (num_tokens * top_k + tile_size - 1) // tile_size
    num_non_exiting_tiles = torch.tensor(
        [num_non_exiting_tiles_val], dtype=torch.int32, device="cuda"
    )
    num_permuted_tokens = num_non_exiting_tiles_val * tile_size

    y = torch.ops.trtllm.moe_swiglu(x, tile_idx_to_mn_limit, num_non_exiting_tiles, tile_size)
    y_ref = swiglu_ref(x)
    torch.testing.assert_close(y[:num_permuted_tokens], y_ref[:num_permuted_tokens])


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="This test is only supported on SM 100 and SM 103 GPUs",
)
@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_moe_swiglu_nvfp4_quantize(dtype: str, num_tokens: int, top_k: int, tile_size: int):
    dtype = getattr(torch, dtype)
    sf_vec_size = 16
    interm_size = 4096
    num_experts = 256
    num_local_experts = num_experts // 32
    helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
    max_num_tiles = helper.get_max_num_tiles(num_tokens)
    max_num_permuted_tokens = helper.get_max_num_permuted_tokens(num_tokens)

    x = torch.randint(
        -100, 100, (max_num_permuted_tokens, interm_size * 2), dtype=torch.int32, device="cuda"
    ).to(dtype)
    tile_idx_to_mn_limit = (
        torch.arange(1, max_num_tiles + 1, dtype=torch.int32, device="cuda") * tile_size
    )
    num_non_exiting_tiles_val = (num_tokens * top_k + tile_size - 1) // tile_size
    num_non_exiting_tiles = torch.tensor(
        [num_non_exiting_tiles_val], dtype=torch.int32, device="cuda"
    )
    num_permuted_tokens = num_non_exiting_tiles_val * tile_size

    global_sf = swiglu_ref(x).abs().max().float() / (448 * 6)
    global_sf = 1 / global_sf
    y, y_sf = torch.ops.trtllm.moe_swiglu_nvfp4_quantize(
        x, global_sf, tile_idx_to_mn_limit, num_non_exiting_tiles, tile_size
    )
    y_ref, y_sf_ref = torch.ops.trtllm.fp4_quantize(swiglu_ref(x), global_sf, sf_vec_size, False)
    match_ratio = (
        y[:num_permuted_tokens].view(torch.uint8) == y_ref[:num_permuted_tokens]
    ).sum().item() / y[:num_permuted_tokens].numel()
    assert match_ratio > 0.999

    num_sf_elements = num_permuted_tokens * interm_size // sf_vec_size
    match_ratio = (
        y_sf[:num_sf_elements] == y_sf_ref[:num_sf_elements]
    ).sum().item() / num_sf_elements
    assert match_ratio > 0.999


@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_moe_gelu(dtype: str, num_tokens: int, top_k: int, tile_size: int):
    dtype = getattr(torch, dtype)
    interm_size = 4096
    num_experts = 256
    num_local_experts = num_experts // 32
    helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
    max_num_tiles = helper.get_max_num_tiles(num_tokens)
    max_num_permuted_tokens = helper.get_max_num_permuted_tokens(num_tokens)

    x = torch.randint(
        -100, 100, (max_num_permuted_tokens, interm_size), dtype=torch.int32, device="cuda"
    ).to(dtype)
    tile_idx_to_mn_limit = (
        torch.arange(1, max_num_tiles + 1, dtype=torch.int32, device="cuda") * tile_size
    )
    num_non_exiting_tiles_val = (num_tokens * top_k + tile_size - 1) // tile_size
    num_non_exiting_tiles = torch.tensor(
        [num_non_exiting_tiles_val], dtype=torch.int32, device="cuda"
    )
    num_permuted_tokens = num_non_exiting_tiles_val * tile_size

    y = torch.ops.trtllm.moe_gelu(x, tile_idx_to_mn_limit, num_non_exiting_tiles, tile_size)
    y_ref = torch.nn.functional.gelu(x)
    torch.testing.assert_close(y[:num_permuted_tokens], y_ref[:num_permuted_tokens])


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="This test is only supported on SM 100 and SM 103 GPUs",
)
@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_nvfp4_grouped_gemm_blackwell(num_tokens: int, top_k: int, ep_size: int, tile_size: int):
    sf_vec_size = 16
    hidden_size = 4096
    interm_size = 8192
    num_experts = 256
    num_local_experts = num_experts // ep_size

    helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
    max_num_tiles = helper.get_max_num_tiles(num_tokens)
    max_num_permuted_tokens = helper.get_max_num_permuted_tokens(num_tokens)
    routing_logits = torch.randn(num_tokens, num_experts, device="cuda")
    _, token_selected_experts = routing_logits.topk(top_k, dim=-1)
    token_selected_experts = token_selected_experts.to(torch.int32)
    num_tokens_per_expert = torch.bincount(token_selected_experts.flatten(), minlength=num_experts)
    num_tokens_per_expert = num_tokens_per_expert[:num_local_experts]
    # Ensure at least one valid token
    if num_tokens_per_expert.sum().item() == 0:
        num_tokens_per_expert[0] = 1
    num_tiles_per_expert = (num_tokens_per_expert + tile_size - 1) // tile_size
    num_tokens_per_expert = num_tokens_per_expert.cpu()
    num_tiles_per_expert = num_tiles_per_expert.cpu()
    num_valid_tiles = num_tiles_per_expert.sum().item()
    num_valid_permuted_tokens = num_valid_tiles * tile_size
    assert 0 <= num_valid_tiles <= max_num_tiles
    assert 0 <= num_valid_permuted_tokens <= max_num_permuted_tokens

    num_non_exiting_tiles = torch.tensor([num_valid_tiles], dtype=torch.int32, device="cuda")
    tile_idx_to_group_idx = torch.empty(max_num_tiles, dtype=torch.int32)
    # Note: Fill -2e9 for invalid tiles.
    tile_idx_to_group_idx.fill_(-2e9)
    tile_idx = 0
    for expert_idx in range(num_local_experts):
        for i in range(num_tiles_per_expert[expert_idx].item()):
            tile_idx_to_group_idx[tile_idx] = expert_idx
            tile_idx += 1
    tile_idx_to_group_idx = tile_idx_to_group_idx.cuda()

    a = torch.randint(
        -5, 5, (max_num_permuted_tokens, hidden_size), dtype=torch.int32, device="cuda"
    ).to(torch.bfloat16)
    b = torch.randint(
        -5,
        5,
        (num_local_experts, interm_size, hidden_size),
        dtype=torch.int32,
        device="cuda",
    ).to(torch.bfloat16)

    a_global_sf = a.abs().max().float() / (448 * 6)
    b_global_sf = b.abs().amax(dim=(1, 2)).float() / (448 * 6)
    a, a_sf = torch.ops.trtllm.fp4_quantize(a, 1 / a_global_sf, sf_vec_size, False)
    a = a.view(torch.float4_e2m1fn_x2)
    b, b_sf = torch.ops.trtllm.fp4_quantize(b, 1 / b_global_sf, sf_vec_size, False)
    b = b.view(torch.float4_e2m1fn_x2)
    b_sf = b_sf.view(num_local_experts, interm_size, hidden_size // sf_vec_size)
    alpha = a_global_sf * b_global_sf

    c = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
        tile_idx_to_group_idx,
        num_non_exiting_tiles,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        output_dtype=torch.bfloat16,
        scaling_vector_size=sf_vec_size,
    )
    c_ref = cute_dsl_nvfp4_grouped_gemm_ref(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
        tile_idx_to_group_idx,
        num_non_exiting_tiles,
        tile_size=tile_size,
        output_dtype=torch.bfloat16,
        scaling_vector_size=sf_vec_size,
    )
    torch.testing.assert_close(c[:num_valid_permuted_tokens], c_ref[:num_valid_permuted_tokens])


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="This test is only supported on SM 100 and SM 103 GPUs",
)
@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_nvfp4_grouped_gemm_finalize_blackwell(
    num_tokens: int, top_k: int, ep_size: int, tile_size: int
):
    sf_vec_size = 16
    hidden_size = 4096
    interm_size = 8192
    num_experts = 256
    num_local_experts = num_experts // ep_size

    routing_logits = torch.randn(num_tokens, num_experts, device="cuda")
    token_final_scales, token_selected_experts = routing_logits.topk(top_k, dim=-1)
    token_selected_experts = token_selected_experts.to(torch.int32)
    token_final_scales = token_final_scales.softmax(dim=-1).to(torch.float32)

    (
        tile_idx_to_group_idx,
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
        local_expert_offset=0,
        local_num_experts=num_local_experts,
        tile_tokens_dim=tile_size,
    )

    max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0)
    a = torch.randint(
        -5, 5, (max_num_permuted_tokens, hidden_size), dtype=torch.int32, device="cuda"
    ).to(torch.bfloat16)
    b = torch.randint(
        -5,
        5,
        (num_local_experts, interm_size, hidden_size),
        dtype=torch.int32,
        device="cuda",
    ).to(torch.bfloat16)

    a_global_sf = a.abs().max().float() / (448 * 6)
    b_global_sf = b.abs().amax(dim=(1, 2)).float() / (448 * 6)
    a, a_sf = torch.ops.trtllm.fp4_quantize(a, 1 / a_global_sf, sf_vec_size, False)
    a = a.view(torch.float4_e2m1fn_x2)
    b, b_sf = torch.ops.trtllm.fp4_quantize(b, 1 / b_global_sf, sf_vec_size, False)
    b = b.view(torch.float4_e2m1fn_x2)
    b_sf = b_sf.view(num_local_experts, interm_size, hidden_size // sf_vec_size)
    alpha = a_global_sf * b_global_sf

    c = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_blackwell(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        output_dtype=torch.bfloat16,
        scaling_vector_size=sf_vec_size,
    )

    c_ref = cute_dsl_nvfp4_grouped_gemm_ref(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
        tile_idx_to_group_idx,
        num_non_exiting_tiles,
        tile_size=tile_size,
        output_dtype=torch.bfloat16,
        scaling_vector_size=sf_vec_size,
    )
    c_ref = torch.ops.trtllm.moe_unpermute(
        permuted_input=c_ref,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        topk_scales=token_final_scales,
    )
    match_ratio = torch.isclose(c, c_ref, rtol=1.6e-2, atol=1e-5).sum().item() / c.numel()
    assert match_ratio > 0.99


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="This test is only supported on SM 100 and SM 103 GPUs",
)
@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_nvfp4_grouped_gemm_swiglu_blackwell(
    num_tokens: int, top_k: int, ep_size: int, tile_size: int
):
    sf_vec_size = 16
    hidden_size = 4096
    interm_size = 8192
    num_experts = 256
    num_local_experts = num_experts // ep_size

    helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
    max_num_tiles = helper.get_max_num_tiles(num_tokens)
    max_num_permuted_tokens = helper.get_max_num_permuted_tokens(num_tokens)
    routing_logits = torch.randn(num_tokens, num_experts, device="cuda")
    _, token_selected_experts = routing_logits.topk(top_k, dim=-1)
    token_selected_experts = token_selected_experts.to(torch.int32)
    num_tokens_per_expert = torch.bincount(token_selected_experts.flatten(), minlength=num_experts)
    num_tokens_per_expert = num_tokens_per_expert[:num_local_experts]
    # Ensure at least one valid token
    if num_tokens_per_expert.sum().item() == 0:
        num_tokens_per_expert[0] = 1
    num_tiles_per_expert = (num_tokens_per_expert + tile_size - 1) // tile_size
    num_tokens_per_expert = num_tokens_per_expert.cpu()
    num_tiles_per_expert = num_tiles_per_expert.cpu()
    num_valid_tiles = num_tiles_per_expert.sum().item()
    num_valid_permuted_tokens = num_valid_tiles * tile_size
    assert 0 <= num_valid_tiles <= max_num_tiles
    assert 0 <= num_valid_permuted_tokens <= max_num_permuted_tokens

    num_non_exiting_tiles = torch.tensor([num_valid_tiles], dtype=torch.int32, device="cuda")
    tile_idx_to_group_idx = torch.empty(max_num_tiles, dtype=torch.int32)
    # Note: Fill -2e9 for invalid tiles.
    tile_idx_to_group_idx.fill_(-2e9)
    tile_idx = 0
    for expert_idx in range(num_local_experts):
        for i in range(num_tiles_per_expert[expert_idx].item()):
            tile_idx_to_group_idx[tile_idx] = expert_idx
            tile_idx += 1
    tile_idx_to_group_idx = tile_idx_to_group_idx.cuda()

    a = torch.randint(
        -5, 5, (max_num_permuted_tokens, hidden_size), dtype=torch.int32, device="cuda"
    ).to(torch.bfloat16)
    b = torch.randint(
        -5,
        5,
        (num_local_experts, interm_size * 2, hidden_size),
        dtype=torch.int32,
        device="cuda",
    ).to(torch.bfloat16)

    a_global_sf = a.abs().max().float() / (448 * 6)
    b_global_sf = b.abs().amax(dim=(1, 2)).float() / (448 * 6)
    a, a_sf = torch.ops.trtllm.fp4_quantize(a, 1 / a_global_sf, sf_vec_size, False)
    a = a.view(torch.float4_e2m1fn_x2)
    b, b_sf = torch.ops.trtllm.fp4_quantize(b, 1 / b_global_sf, sf_vec_size, False)
    b = b.view(torch.float4_e2m1fn_x2)
    b_sf = b_sf.view(num_local_experts, interm_size * 2, hidden_size // sf_vec_size)
    alpha = a_global_sf * b_global_sf

    b_interleaved = interleave_linear_and_gate(b.view(torch.uint8), group_size=64, dim=1).view(
        torch.float4_e2m1fn_x2
    )
    b_sf_unswizzled = unswizzle_sf(b_sf, interm_size * 2, hidden_size).view(
        num_local_experts, interm_size * 2, hidden_size // sf_vec_size
    )
    b_sf_unswizzled_interleaved = interleave_linear_and_gate(b_sf_unswizzled, group_size=64, dim=1)
    b_sf_interleaved = swizzle_sf(b_sf_unswizzled_interleaved, interm_size * 2, hidden_size).view(
        num_local_experts, interm_size * 2, hidden_size // sf_vec_size
    )

    c_ref = cute_dsl_nvfp4_grouped_gemm_ref(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
        tile_idx_to_group_idx,
        num_non_exiting_tiles,
        tile_size=tile_size,
        output_dtype=torch.bfloat16,
        scaling_vector_size=sf_vec_size,
    )
    c_ref = swiglu_ref(c_ref)
    global_sf = c_ref[:num_valid_permuted_tokens].abs().max().float() / (448 * 6)
    c_ref, c_sf_ref = torch.ops.trtllm.fp4_quantize(c_ref, 1 / global_sf, sf_vec_size, False)

    c, c_sf = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_swiglu_blackwell(
        a,
        b_interleaved,
        a_sf,
        b_sf_interleaved,
        alpha,
        tile_idx_to_group_idx,
        num_non_exiting_tiles,
        1 / global_sf,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
    )

    match_ratio = (
        c[:num_valid_permuted_tokens].view(torch.uint8) == c_ref[:num_valid_permuted_tokens]
    ).sum().item() / c[:num_valid_permuted_tokens].numel()
    assert match_ratio > 0.95

    num_sf_elements = num_valid_permuted_tokens * interm_size // sf_vec_size
    match_ratio = (
        c_sf[:num_sf_elements] == c_sf_ref[:num_sf_elements]
    ).sum().item() / num_sf_elements
    assert match_ratio > 0.95


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="This test is only supported on SM 100 and SM 103 GPUs",
)
@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_nvfp4_gather_grouped_gemm_swiglu_blackwell(
    num_tokens: int, top_k: int, ep_size: int, tile_size: int
):
    """Test gather-based grouped GEMM with SwiGLU fusion.

    This test validates the gather kernel which:
    1. Uses LDGSTS for A/SFA loading with permuted_idx_to_expanded_idx
    2. Performs GEMM with interleaved weights
    3. Applies SwiGLU activation fusion
    4. Quantizes output to FP4 with scale factor generation
    """
    sf_vec_size = 16
    hidden_size = 4096
    interm_size = 8192
    num_experts = 256
    num_local_experts = num_experts // ep_size

    # Generate routing information
    routing_logits = torch.randn(num_tokens, num_experts, device="cuda")
    token_final_scales, token_selected_experts = routing_logits.topk(top_k, dim=-1)
    token_selected_experts = token_selected_experts.to(torch.int32)
    token_final_scales = token_final_scales.softmax(dim=-1).to(torch.float32)
    # Ensure at least one valid token
    token_selected_experts[0] = 0

    (
        tile_idx_to_group_idx,
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
        local_expert_offset=0,
        local_num_experts=num_local_experts,
        tile_tokens_dim=tile_size,
    )

    max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0)
    num_valid_permuted_tokens = total_num_padded_tokens.item()

    # Create input tensors (original size, not permuted)
    a = torch.randint(-5, 5, (num_tokens, hidden_size), dtype=torch.int32, device="cuda").to(
        torch.bfloat16
    )
    b = torch.randint(
        -5,
        5,
        (num_local_experts, interm_size * 2, hidden_size),
        dtype=torch.int32,
        device="cuda",
    ).to(torch.bfloat16)

    # Quantize inputs to FP4
    a_global_sf = a.abs().max().float() / (448 * 6)
    b_global_sf = b.abs().amax(dim=(1, 2)).float() / (448 * 6)
    a, a_sf = torch.ops.trtllm.fp4_quantize(a, 1 / a_global_sf, sf_vec_size, False)
    a = a.view(torch.float4_e2m1fn_x2)
    a_sf_unswizzled = unswizzle_sf(a_sf, (num_tokens + 127) // 128 * 128, hidden_size)[:num_tokens]
    b, b_sf = torch.ops.trtllm.fp4_quantize(b, 1 / b_global_sf, sf_vec_size, False)
    b = b.view(torch.float4_e2m1fn_x2)
    b_sf = b_sf.view(num_local_experts, interm_size * 2, hidden_size // sf_vec_size)
    alpha = a_global_sf * b_global_sf

    # Interleave weights for SwiGLU
    b_interleaved = interleave_linear_and_gate(b.view(torch.uint8), group_size=64, dim=1).view(
        torch.float4_e2m1fn_x2
    )
    b_sf_unswizzled = unswizzle_sf(b_sf, interm_size * 2, hidden_size).view(
        num_local_experts, interm_size * 2, hidden_size // sf_vec_size
    )
    b_sf_unswizzled_interleaved = interleave_linear_and_gate(b_sf_unswizzled, group_size=64, dim=1)
    b_sf_interleaved = swizzle_sf(b_sf_unswizzled_interleaved, interm_size * 2, hidden_size).view(
        num_local_experts, interm_size * 2, hidden_size // sf_vec_size
    )

    # Compute reference: manually gather, compute GEMM, apply SwiGLU, then quantize
    permuted_idx_to_expanded_idx_list = permuted_idx_to_expanded_idx.cpu().tolist()
    tile_idx_to_mn_limit_list = tile_idx_to_mn_limit.cpu().tolist()

    a_gathered = torch.empty(max_num_permuted_tokens, hidden_size // 2, dtype=a.dtype)
    a_sf_gathered = torch.empty(
        max_num_permuted_tokens, hidden_size // sf_vec_size, dtype=a_sf.dtype
    )
    for i in range(num_valid_permuted_tokens):
        if i >= tile_idx_to_mn_limit_list[i // tile_size]:
            continue
        expanded_idx = permuted_idx_to_expanded_idx_list[i]
        token_id = expanded_idx // top_k
        a_gathered[i] = a[token_id]
        a_sf_gathered[i] = a_sf_unswizzled[token_id]
    a_gathered = a_gathered.to(a.device)
    a_sf_gathered = a_sf_gathered.to(a.device)

    # Swizzle a_sf_gathered for reference GEMM
    a_sf_gathered_swizzled = swizzle_sf(
        a_sf_gathered.view(max_num_permuted_tokens, hidden_size // sf_vec_size),
        max_num_permuted_tokens,
        hidden_size,
    )

    c_ref = cute_dsl_nvfp4_grouped_gemm_ref(
        a_gathered,
        b,
        a_sf_gathered_swizzled,
        b_sf,
        alpha,
        tile_idx_to_group_idx,
        num_non_exiting_tiles,
        tile_size=tile_size,
        output_dtype=torch.bfloat16,
        scaling_vector_size=sf_vec_size,
    )
    c_ref = swiglu_ref(c_ref)
    global_sf = c_ref[:num_valid_permuted_tokens].abs().max().float() / (448 * 6)
    c_ref, c_sf_ref = torch.ops.trtllm.fp4_quantize(c_ref, 1 / global_sf, sf_vec_size, False)

    # Call gather kernel
    c, c_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell(
        a,
        b_interleaved,
        a_sf_unswizzled,
        b_sf_interleaved,
        alpha,
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        torch.tensor([1 / global_sf], dtype=torch.float32, device="cuda"),
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
    )

    # Verify output (only compare valid tokens, skip padding tokens where permuted_idx_to_expanded_idx == -1)
    # Create mask for valid tokens
    valid_token_mask = torch.zeros(num_valid_permuted_tokens, dtype=torch.bool, device="cuda")
    for i in range(num_valid_permuted_tokens):
        if i >= tile_idx_to_mn_limit_list[i // tile_size]:
            continue
        valid_token_mask[i] = True

    num_valid_tokens = valid_token_mask.sum().item()
    if num_valid_tokens > 0:
        # Compare output values only for valid tokens
        c_valid = c[:num_valid_permuted_tokens].view(torch.uint8)[valid_token_mask]
        c_ref_valid = c_ref[:num_valid_permuted_tokens][valid_token_mask]
        check_accuracy(c_valid, c_ref_valid, atol=1e-4, rtol=1e-4, percent=0.95)

        c_sf_unswizzled = unswizzle_sf(c_sf, max_num_permuted_tokens, interm_size, sf_vec_size)
        c_sf_ref_unswizzled = unswizzle_sf(
            c_sf_ref, max_num_permuted_tokens, interm_size, sf_vec_size
        )

        # Compare scale factors only for valid tokens
        c_sf_valid = []
        c_sf_ref_valid = []
        for i in range(num_valid_permuted_tokens):
            if i >= tile_idx_to_mn_limit_list[i // tile_size]:
                continue
            c_sf_valid.append(c_sf_unswizzled[i])
            c_sf_ref_valid.append(c_sf_ref_unswizzled[i])

        c_sf_valid = torch.cat(c_sf_valid)
        c_sf_ref_valid = torch.cat(c_sf_ref_valid)
        check_accuracy(c_sf_valid, c_sf_ref_valid, atol=1e-4, rtol=1e-4, percent=0.95)
