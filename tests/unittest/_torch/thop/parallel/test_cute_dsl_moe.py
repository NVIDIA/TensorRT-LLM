# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from utils.util import check_accuracy

from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile, TunableRunner
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
    GroupedGemmInputsHelper,
    _get_sm107_nvfp4_default_mma_config,
)
from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)
from tensorrt_llm._torch.locality_domain_utils import (
    end_for_all_locality_domain,
    get_locality_domain_stream,
    is_locality_domain_enabled,
    locality_domain_device,
    start_for_all_locality_domain,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl import (
    CuteDslFusedMoE,
    _LocalityDomainConcurrentTunableRunner,
    _runner_tactics_match_tile_size,
    cute_dsl_nvfp4_grouped_gemm_ref,
)
from tensorrt_llm._torch.moe.fused_moe.quantization import interleave_linear_and_gate
from tensorrt_llm._torch.utils import (
    ActivationType,
    Fp4QuantizedTensor,
    is_gated_activation,
    relu2,
    swizzle_sf,
    unswizzle_sf,
)
from tensorrt_llm._utils import get_sm_version


def swiglu_ref(x: torch.Tensor, swiglu_limit: float = float("inf")) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    if swiglu_limit != float("inf"):
        gate = gate.clamp(max=swiglu_limit)
        x = x.clamp(min=-swiglu_limit, max=swiglu_limit)
    return x * torch.nn.functional.silu(gate)


def apply_activation_ref(
    x: torch.Tensor, activation_type: ActivationType, swiglu_limit: float = float("inf")
) -> torch.Tensor:
    if activation_type == ActivationType.Swiglu:
        return swiglu_ref(x, swiglu_limit)
    if activation_type == ActivationType.Relu2:
        return relu2(x)
    raise ValueError(f"Unsupported activation_type: {activation_type}")


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
@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available")
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
@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available")
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
@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available")
@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_nvfp4_grouped_gemm_swiglu_blackwell(
    num_tokens: int, top_k: int, ep_size: int, tile_size: int
):
    sf_vec_size = 16
    swiglu_limit = 1.0
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
    c_ref = swiglu_ref(c_ref, swiglu_limit)
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
        swiglu_limit_scalar=swiglu_limit,
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
@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available")
@pytest.mark.parametrize(
    "activation_type",
    [ActivationType.Swiglu, ActivationType.Relu2],
    ids=["swiglu", "relu2"],
)
@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
    num_tokens: int,
    top_k: int,
    ep_size: int,
    tile_size: int,
    activation_type: ActivationType,
):
    """Test gather-based grouped GEMM with fused activation.

    This test validates the gather kernel which:
    1. Uses LDGSTS for A/SFA loading with permuted_idx_to_expanded_idx
    2. Performs GEMM with (interleaved for gated) weights
    3. Applies the fused activation (SwiGLU for gated, Relu2 for non-gated)
    4. Quantizes output to FP4 with scale factor generation
    """
    is_gated = is_gated_activation(activation_type)
    swiglu_limit = 1.0 if is_gated else float("inf")
    weight_n_multiplier = 2 if is_gated else 1
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
        (num_local_experts, interm_size * weight_n_multiplier, hidden_size),
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
    b_sf = b_sf.view(
        num_local_experts, interm_size * weight_n_multiplier, hidden_size // sf_vec_size
    )
    alpha = a_global_sf * b_global_sf

    # Interleave weights for gated activations (SwiGLU); non-gated uses plain weights.
    if is_gated:
        b_kernel = interleave_linear_and_gate(b.view(torch.uint8), group_size=64, dim=1).view(
            torch.float4_e2m1fn_x2
        )
        b_sf_unswizzled = unswizzle_sf(b_sf, interm_size * weight_n_multiplier, hidden_size).view(
            num_local_experts, interm_size * weight_n_multiplier, hidden_size // sf_vec_size
        )
        b_sf_unswizzled_interleaved = interleave_linear_and_gate(
            b_sf_unswizzled, group_size=64, dim=1
        )
        b_sf_kernel = swizzle_sf(
            b_sf_unswizzled_interleaved, interm_size * weight_n_multiplier, hidden_size
        ).view(num_local_experts, interm_size * weight_n_multiplier, hidden_size // sf_vec_size)
    else:
        b_kernel = b
        b_sf_kernel = b_sf.view(
            num_local_experts, interm_size * weight_n_multiplier, hidden_size // sf_vec_size
        )

    # Compute reference: manually gather, compute GEMM, apply fused activation, then quantize
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
    c_ref = apply_activation_ref(c_ref, activation_type, swiglu_limit)
    global_sf = c_ref[:num_valid_permuted_tokens].abs().max().float() / (448 * 6)
    c_ref, c_sf_ref = torch.ops.trtllm.fp4_quantize(c_ref, 1 / global_sf, sf_vec_size, False)

    # Call gather kernel (single-B)
    c, c_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
        a,
        b_kernel,
        a_sf_unswizzled,
        b_sf_kernel,
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
        activation_type=activation_type,
        swiglu_limit_scalar=swiglu_limit,
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


# ============================================================================
# Rubin (SM107) Tests
# ============================================================================


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on SM 107 (Rubin) GPUs",
)
@pytest.mark.parametrize(
    "activation_type",
    [ActivationType.Swiglu, ActivationType.Relu2],
    ids=["swiglu", "relu2"],
)
@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_nvfp4_gather_grouped_gemm_act_fusion_rubin(
    num_tokens: int,
    top_k: int,
    ep_size: int,
    tile_size: int,
    activation_type: ActivationType,
):
    """Test gather-based grouped GEMM with fused activation on Rubin (SM107).

    This test validates the gather kernel which:
    1. Uses gather for A/SFA loading with permuted_idx_to_expanded_idx
    2. Performs GEMM with (interleaved for gated) weights
    3. Applies the fused activation (SwiGLU for gated, Relu2 for non-gated)
    4. Quantizes output to FP4 with scale factor generation
    """
    is_gated = is_gated_activation(activation_type)
    weight_n_multiplier = 2 if is_gated else 1
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
        (num_local_experts, interm_size * weight_n_multiplier, hidden_size),
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
    weight_n = interm_size * weight_n_multiplier
    b_sf = b_sf.view(num_local_experts, weight_n, hidden_size // sf_vec_size)
    alpha = a_global_sf * b_global_sf

    b_kernel = b
    b_sf_kernel = b_sf
    if is_gated:
        b_kernel = interleave_linear_and_gate(b.view(torch.uint8), group_size=64, dim=1).view(
            torch.float4_e2m1fn_x2
        )
        b_sf_unswizzled = unswizzle_sf(b_sf, weight_n, hidden_size).view(
            num_local_experts, weight_n, hidden_size // sf_vec_size
        )
        b_sf_unswizzled = interleave_linear_and_gate(b_sf_unswizzled, group_size=64, dim=1)
        b_sf_kernel = swizzle_sf(b_sf_unswizzled, weight_n, hidden_size).view(
            num_local_experts, weight_n, hidden_size // sf_vec_size
        )

    # Compute reference: manually gather, compute GEMM, apply activation, then quantize
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
    c_ref = apply_activation_ref(c_ref, activation_type)
    global_sf = c_ref[:num_valid_permuted_tokens].abs().max().float() / (448 * 6)
    c_ref, c_sf_ref = torch.ops.trtllm.fp4_quantize(c_ref, 1 / global_sf, sf_vec_size, False)

    # Call Rubin gather kernel
    c, c_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin(
        a,
        b_kernel,
        a_sf_unswizzled,
        b_sf_kernel,
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
        output_tensor=None,
        output_sf_tensor=None,
        scaling_vector_size=sf_vec_size,
        activation_type=activation_type,
    )

    # Verify output (only compare valid tokens, skip padding)
    valid_token_mask = torch.zeros(num_valid_permuted_tokens, dtype=torch.bool, device="cuda")
    for i in range(num_valid_permuted_tokens):
        if i >= tile_idx_to_mn_limit_list[i // tile_size]:
            continue
        valid_token_mask[i] = True

    num_valid_tokens = valid_token_mask.sum().item()
    if num_valid_tokens > 0:
        c_valid = c[:num_valid_permuted_tokens].view(torch.uint8)[valid_token_mask]
        c_ref_valid = c_ref[:num_valid_permuted_tokens][valid_token_mask]
        check_accuracy(c_valid, c_ref_valid, atol=1e-4, rtol=1e-4, percent=0.95)

        c_sf_unswizzled = unswizzle_sf(c_sf, max_num_permuted_tokens, interm_size, sf_vec_size)
        c_sf_ref_unswizzled = unswizzle_sf(
            c_sf_ref, max_num_permuted_tokens, interm_size, sf_vec_size
        )

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


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on SM 107 (Rubin) GPUs",
)
@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, hidden_size, interm_size, tile_size",
    [
        # DeepSeek V3 Lite-like: small tokens with tile_size=128 and 256
        (16, 72, 6, 2560, 1536, 128),
        (16, 72, 6, 2560, 1536, 256),
        (8, 72, 6, 2560, 1536, 128),
        (8, 72, 6, 2560, 1536, 256),
        # Qwen3-30B-A3B-like: hidden_size=2048 (triggered Fix 7)
        (8, 128, 8, 2048, 1536, 128),
        (8, 128, 8, 2048, 1536, 256),
        # Very small: 1 token, tile_size=256 (triggered Fix 9)
        (1, 72, 6, 2560, 1536, 128),
        (1, 72, 6, 2560, 1536, 256),
        # Small tokens, high padding ratio
        (4, 72, 6, 2560, 1536, 256),
        (2, 128, 8, 2048, 1536, 256),
    ],
)
def test_nvfp4_gather_grouped_gemm_swiglu_rubin_small_tokens(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    interm_size: int,
    tile_size: int,
):
    """Test FC1 gather+SwiGLU kernel on Rubin with small num_tokens (high padding ratio).

    Covers configurations that triggered Fix 9 (pad_val crash with tile_size=256)
    and Qwen3-30B-A3B shapes (hidden_size=2048, Fix 7). Uses real moe_sort routing
    metadata (not synthetic).
    """
    sf_vec_size = 16
    num_local_experts = num_experts  # ep_size=1

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
    n_tiles = num_non_exiting_tiles.item()

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

    a_global_sf = a.abs().max().float() / (448 * 6)
    b_global_sf = b.abs().amax(dim=(1, 2)).float() / (448 * 6)
    a, a_sf = torch.ops.trtllm.fp4_quantize(a, 1 / a_global_sf, sf_vec_size, False)
    a = a.view(torch.float4_e2m1fn_x2)
    a_sf_unswizzled = unswizzle_sf(a_sf, (num_tokens + 127) // 128 * 128, hidden_size)[:num_tokens]
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

    # Compute reference: gather A using permuted_idx, then grouped GEMM + SwiGLU
    # Use uint8 + view because torch.zeros doesn't support Float4_e2m1fn_x2
    a_gathered = torch.zeros(
        max_num_permuted_tokens, hidden_size // 2, dtype=torch.uint8, device=a.device
    ).view(torch.float4_e2m1fn_x2)
    a_sf_gathered = torch.zeros(
        max_num_permuted_tokens, hidden_size // sf_vec_size, dtype=a_sf.dtype, device=a_sf.device
    )
    num_valid_permuted_tokens = n_tiles * tile_size
    for i in range(min(num_valid_permuted_tokens, max_num_permuted_tokens)):
        expanded_idx = permuted_idx_to_expanded_idx[i].item()
        if expanded_idx > 0 or i == 0:
            token_id = expanded_idx // top_k
            if token_id < num_tokens:
                a_gathered[i] = a[token_id]
                a_sf_gathered[i] = a_sf_unswizzled[token_id]

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
    if global_sf == 0:
        global_sf = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    c_ref, c_sf_ref = torch.ops.trtllm.fp4_quantize(c_ref, 1 / global_sf, sf_vec_size, False)

    c, c_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin(
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
        output_tensor=None,
        output_sf_tensor=None,
        scaling_vector_size=sf_vec_size,
        activation_type=ActivationType.Swiglu,
    )

    # Verify output for valid (non-padding) tokens
    valid_token_mask = permuted_idx_to_expanded_idx[:num_valid_permuted_tokens] != 0
    valid_token_mask[0] = True  # index 0 is always valid
    num_valid_tokens = valid_token_mask.sum().item()
    if num_valid_tokens > 0:
        c_valid = c[:num_valid_permuted_tokens].view(torch.uint8)[valid_token_mask]
        c_ref_valid = c_ref[:num_valid_permuted_tokens][valid_token_mask]
        check_accuracy(c_valid, c_ref_valid, atol=1e-4, rtol=1e-4, percent=0.95)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on SM 107 (Rubin) GPUs",
)
@pytest.mark.parametrize("tile_size", [128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_nvfp4_grouped_gemm_finalize_rubin(
    num_tokens: int, top_k: int, ep_size: int, tile_size: int
):
    """Test grouped GEMM with finalize fusion on Rubin (SM107).

    Same test logic as test_nvfp4_grouped_gemm_finalize_blackwell
    but calls the Rubin-specific custom op.
    """
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

    # Call Rubin finalize kernel
    c = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_rubin(
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

    # Compute reference for test_nvfp4_grouped_gemm_finalize_rubin
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
    get_sm_version() != 107,
    reason="This test is only supported on SM 107 (Rubin) GPUs",
)
@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, hidden_size, interm_size, tile_size",
    [
        # DeepSeek V3 Lite-like: 16 tokens, 72 experts, top_k=6, tile_size=128
        (16, 72, 6, 2560, 1536, 128),
        # DeepSeek V3 Lite-like: 16 tokens, 72 experts, top_k=6, tile_size=256 (CRASHES in e2e)
        (16, 72, 6, 2560, 1536, 256),
        # DeepSeek V3 Lite-like: 8 tokens
        (8, 72, 6, 2560, 1536, 128),
        (8, 72, 6, 2560, 1536, 256),
        # Qwen3-30B-A3B-like: 8 tokens, 128 experts, top_k=8
        (8, 128, 8, 2048, 1536, 128),
        (8, 128, 8, 2048, 1536, 256),
        # Very small: 1 token
        (1, 72, 6, 2560, 1536, 128),
        (1, 72, 6, 2560, 1536, 256),
        # Small tokens, high padding ratio
        (4, 72, 6, 2560, 1536, 256),
        (2, 128, 8, 2048, 1536, 256),
    ],
)
def test_nvfp4_grouped_gemm_finalize_rubin_small_tokens(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    interm_size: int,
    tile_size: int,
):
    """Test FC2 finalize kernel on Rubin with small num_tokens (high padding ratio).

    This reproduces the crashing e2e configuration where num_tokens is small
    relative to num_experts, causing most tile rows to be padding. Uses real
    moe_sort routing metadata (not synthetic).
    """
    sf_vec_size = 16
    num_local_experts = num_experts  # ep_size=1

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
    n_tiles = num_non_exiting_tiles.item()
    print(
        f"\n[test_small_tokens] num_tokens={num_tokens}, num_experts={num_experts}, "
        f"top_k={top_k}, tile_size={tile_size}, "
        f"total_padded={max_num_permuted_tokens}, num_tiles={n_tiles}"
    )

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

    # Call Rubin finalize kernel with real moe_sort routing data
    c = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_rubin(
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

    # Compute reference for test_nvfp4_grouped_gemm_finalize_rubin_small_tokens
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
    get_sm_version() != 107,
    reason="This test is only supported on SM 107 (Rubin) GPUs",
)
@pytest.mark.parametrize("tile_size", [64, 128, 256])
@pytest.mark.parametrize("ep_size", [1, 8, 32])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
def test_bf16_gather_grouped_gemm_swiglu_rubin(
    num_tokens: int,
    top_k: int,
    ep_size: int,
    tile_size: int,
):
    """Test BF16 gather-based grouped GEMM with SwiGLU fusion on Rubin (SM107).

    Uses torch.ops.trtllm.cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin.
    No scale factors or quantization — direct BF16 inputs/outputs.
    """
    hidden_size = 4096
    interm_size = 8192
    num_experts = 256
    num_local_experts = num_experts // ep_size
    interleave_granularity = 32

    # Generate routing information
    routing_logits = torch.randn(num_tokens, num_experts, device="cuda")
    token_final_scales, token_selected_experts = routing_logits.topk(top_k, dim=-1)
    token_selected_experts = token_selected_experts.to(torch.int32)
    token_final_scales = token_final_scales.softmax(dim=-1).to(torch.float32)
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

    # Create BF16 input tensors
    a = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        num_local_experts, interm_size * 2, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    alpha = torch.ones(num_local_experts, dtype=torch.float32, device="cuda")

    # Interleave weights for SwiGLU: [up_0:64, gate_64:128, ...]
    b_interleaved = interleave_linear_and_gate(
        b.view(torch.uint8), group_size=interleave_granularity, dim=1
    ).view(torch.bfloat16)

    # Compute reference: gather A, GEMM per group, SwiGLU
    permuted_idx_list = permuted_idx_to_expanded_idx.cpu().tolist()
    tile_group_list = tile_idx_to_group_idx.cpu().tolist()
    tile_mn_limit_list = tile_idx_to_mn_limit.cpu().tolist()

    c_ref = torch.zeros(max_num_permuted_tokens, interm_size, dtype=torch.float32, device="cuda")
    for tile_idx in range(num_non_exiting_tiles.item()):
        group_idx = tile_group_list[tile_idx]
        mn_limit = tile_mn_limit_list[tile_idx]
        start = tile_idx * tile_size
        end = min(start + tile_size, mn_limit)

        for i in range(start, end):
            token_id = permuted_idx_list[i] // top_k
            a_row = a[token_id].float()
            gemm_row = a_row @ b_interleaved[group_idx].float().T * alpha[group_idx].item()
            # SwiGLU on interleaved result
            out_row = torch.zeros(interm_size, dtype=torch.float32, device="cuda")
            for n_block in range(0, interm_size * 2, 2 * interleave_granularity):
                up = gemm_row[n_block : n_block + interleave_granularity]
                gate = gemm_row[
                    n_block + interleave_granularity : n_block + 2 * interleave_granularity
                ]
                out_start = n_block // 2
                out_row[out_start : out_start + interleave_granularity] = up * (
                    gate * torch.sigmoid(gate)
                )
            c_ref[i] = out_row

    # Build valid mask for accuracy checking
    valid_mask = torch.zeros(num_valid_permuted_tokens, dtype=torch.bool, device="cuda")
    for i in range(num_valid_permuted_tokens):
        if i < tile_mn_limit_list[i // tile_size]:
            valid_mask[i] = True
    c_ref_valid = c_ref[:num_valid_permuted_tokens][valid_mask]

    # Even-tile padding for Rubin cluster sync
    kernel_nnet = ((num_non_exiting_tiles + 1) // 2) * 2

    # Test all valid autotuner candidate tactics via direct runner call
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        Sm107ContiguousGatherGroupedGemmSwigluFusionRunner,
    )

    runner = Sm107ContiguousGatherGroupedGemmSwigluFusionRunner(
        num_experts, top_k, num_local_experts, 0, tile_size
    )
    inputs = [
        a,
        b_interleaved,
        alpha,
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        kernel_nnet,
    ]
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) > 0, f"No valid tactics for tile_size={tile_size}"

    failed = []
    for tactic in tactics:
        mma_tiler, _, cluster, _ = tactic
        label = f"mma={mma_tiler[:2]} cluster={cluster}"
        with torch.inference_mode():
            c = runner.forward(inputs, tactic=tactic)
        c_valid = c[:num_valid_permuted_tokens].float()[valid_mask]
        if c_ref_valid.numel() > 0:
            match = (
                torch.isclose(c_valid, c_ref_valid, rtol=1e-2, atol=0.5).sum().item()
                / c_ref_valid.numel()
            )
            if match < 0.95:
                failed.append(f"{label}: match={match:.4f}")
    assert not failed, (
        f"tile_size={tile_size}: {len(failed)}/{len(tactics)} tactics failed:\n  "
        + "\n  ".join(failed)
    )


def _skip_if_no_locality_domain():
    is_locality_domain_enabled.cache_clear()
    if not is_locality_domain_enabled():
        pytest.skip("locality domain localization is not enabled/supported on this system")


def _setup_locality_domain_routing(num_tokens, num_experts, num_local_experts, top_k, tile_size):
    torch.manual_seed(42)
    routing_logits = torch.randn(num_tokens, num_experts, device="cuda")
    token_final_scales, token_selected_experts = routing_logits.topk(top_k, dim=-1)
    token_selected_experts = token_selected_experts.to(torch.int32)
    token_final_scales = token_final_scales.softmax(dim=-1).to(torch.float32)
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

    return (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
        token_final_scales,
    )


def _valid_permuted_token_mask(tile_idx_to_mn_limit, num_valid, tile_size):
    tile_idx_to_mn_limit_list = tile_idx_to_mn_limit.cpu().tolist()
    valid_mask = torch.zeros(num_valid, dtype=torch.bool, device=tile_idx_to_mn_limit.device)
    for row_idx in range(num_valid):
        if row_idx < tile_idx_to_mn_limit_list[row_idx // tile_size]:
            valid_mask[row_idx] = True
    return valid_mask


def _create_quantized_locality_domain_inputs(num_tokens, hidden_size, sf_vec_size=16, seed=42):
    torch.manual_seed(seed)
    a = torch.randint(-5, 5, (num_tokens, hidden_size), dtype=torch.int32, device="cuda").to(
        torch.bfloat16
    )
    a_global_sf = a.abs().max().float() / (448 * 6)
    a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a, 1 / a_global_sf, sf_vec_size, False)
    a_fp4 = a_fp4.view(torch.float4_e2m1fn_x2)
    a_sf_unswizzled = unswizzle_sf(a_sf, (num_tokens + 127) // 128 * 128, hidden_size)[:num_tokens]
    return a_fp4, a_sf_unswizzled, a_global_sf


def _create_quantized_locality_domain_weights(
    num_local_experts, interm_size, hidden_size, sf_vec_size=16, seed=123
):
    torch.manual_seed(seed)
    weight = torch.randint(
        -5, 5, (num_local_experts, interm_size * 2, hidden_size), dtype=torch.int32, device="cuda"
    ).to(torch.bfloat16)
    weight_global_sf = weight.abs().amax(dim=(1, 2)).float() / (448 * 6)
    weight_fp4, weight_sf = torch.ops.trtllm.fp4_quantize(
        weight, 1 / weight_global_sf, sf_vec_size, False
    )
    weight_fp4 = weight_fp4.view(torch.float4_e2m1fn_x2)
    weight_sf = weight_sf.view(num_local_experts, interm_size * 2, hidden_size // sf_vec_size)

    weight_interleaved = interleave_linear_and_gate(
        weight_fp4.view(torch.uint8), group_size=64, dim=1
    ).view(torch.float4_e2m1fn_x2)
    weight_sf_unswizzled = unswizzle_sf(weight_sf, interm_size * 2, hidden_size).view(
        num_local_experts, interm_size * 2, hidden_size // sf_vec_size
    )
    weight_sf_unswizzled_interleaved = interleave_linear_and_gate(
        weight_sf_unswizzled, group_size=64, dim=1
    )
    weight_sf_interleaved = swizzle_sf(
        weight_sf_unswizzled_interleaved, interm_size * 2, hidden_size
    ).view(num_local_experts, interm_size * 2, hidden_size // sf_vec_size)

    return weight_interleaved, weight_sf_interleaved, weight_global_sf


def test_moe_output_memset_aux_stream_guard_requires_full_aux_state():
    from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
    from tensorrt_llm._torch.utils import AuxStreamType, EventType

    backend = object.__new__(CuteDslFusedMoE)
    backend.event_dict = {EventType.Main: object()}
    backend.aux_stream_dict = {}

    assert not backend._has_moe_output_memset_aux_stream()

    backend.event_dict[EventType.MoeOutputMemset] = object()
    backend.aux_stream_dict[AuxStreamType.MoeOutputMemset] = object()

    assert backend._has_moe_output_memset_aux_stream()


def test_runner_tactics_match_tile_size_unwraps_locality_domain_runner():
    class OuterRunner:
        pass

    class FakeOpRunner(TunableRunner):
        def get_valid_tactics(self, inputs, profile, **kwargs):
            return []

        def forward(self, inputs, tactic):
            return None

    op_runner = FakeOpRunner()
    runner = _LocalityDomainConcurrentTunableRunner(
        op_runner,
        SimpleNamespace(num_partitions=2),
        2,
        lambda *_: None,
    )
    outer_runner = OuterRunner()
    matching_tactic = ((128, 128, 64), (128, 128, 16), (1, 1), False)
    mismatched_tactic = ((64, 128, 64), (64, 128, 16), (1, 1), False)
    assert _runner_tactics_match_tile_size(
        [(outer_runner, 128), (runner, matching_tactic)],
        OuterRunner,
        (FakeOpRunner,),
    )
    assert not _runner_tactics_match_tile_size(
        [(outer_runner, 128), (runner, mismatched_tactic)],
        OuterRunner,
        (FakeOpRunner,),
    )


def test_locality_domain_concurrent_tunable_runner_delegates_and_launches_all_partitions():
    class FakeOpRunner(TunableRunner):
        def __init__(self):
            self.tactics_call = None

        def unique_id(self):
            return ("fake-op", 7)

        def get_valid_tactics(self, inputs, profile, **kwargs):
            self.tactics_call = (id(inputs), profile, kwargs)
            return [-1, ("fast", 128)]

        def forward(self, /, inputs, *, tactic=-1, **kwargs):
            raise AssertionError("the wrapper must call launch_fn, not the op runner")

    class FakeRuntime:
        def __init__(self, topology):
            self.topology = topology
            self.events = []

        def topology_identity(self):
            return self.topology

        def fork(self):
            self.events.append(("fork",))

        @contextmanager
        def partition_context(self, partition_id):
            self.events.append(("enter", partition_id))
            yield
            self.events.append(("exit", partition_id))

        def join(self):
            self.events.append(("join",))

    op_runner = FakeOpRunner()
    runtime = FakeRuntime(((100, 212), (100, 212), (12, 212)))
    inputs = [torch.empty(1)]

    def launch(partition_id, launch_inputs, tactic):
        runtime.events.append(("launch", partition_id, id(launch_inputs), tactic))

    runner = _LocalityDomainConcurrentTunableRunner(op_runner, runtime, 3, launch)
    profile = OptimizationProfile()

    assert runner.get_valid_tactics(inputs, profile, marker="x") == [-1, ("fast", 128)]
    assert op_runner.tactics_call == (id(inputs), profile, {"marker": "x"})
    assert runner.unique_id()[0] == op_runner.unique_id()
    assert runner.op_runner is op_runner

    other_runtime = FakeRuntime(((106, 212), (106, 212), (0, 212)))
    other_runner = _LocalityDomainConcurrentTunableRunner(op_runner, other_runtime, 3, launch)
    assert runner.unique_id() != other_runner.unique_id()

    tactic = ("chosen", 256)
    assert runner(inputs, tactic=tactic) is None
    assert runtime.events == [
        ("fork",),
        ("enter", 0),
        ("launch", 0, id(inputs), tactic),
        ("exit", 0),
        ("enter", 1),
        ("launch", 1, id(inputs), tactic),
        ("exit", 1),
        ("enter", 2),
        ("launch", 2, id(inputs), tactic),
        ("exit", 2),
        ("join",),
    ]


@pytest.mark.parametrize(
    "runner_name,num_inputs,output_idx,tile_sizes",
    [
        ("CuteDslFusedMoENvfp4Runner", 5, 4, [128, 256, 512]),
        ("CuteDslFusedMoEBF16Runner", 4, 3, [64, 128, 256]),
    ],
)
def test_locality_domain_outer_preparation_disables_memset_overlap(
    monkeypatch,
    runner_name: str,
    num_inputs: int,
    output_idx: int,
    tile_sizes: list[int],
):
    from tensorrt_llm._torch.moe.fused_moe import fused_moe_cute_dsl

    monkeypatch.setattr(fused_moe_cute_dsl, "get_sm_version", lambda: 107)
    preparation_calls = []

    def forward_impl(*args, **kwargs):
        preparation_calls.append((args, kwargs))
        return args[output_idx]

    runner_cls = getattr(fused_moe_cute_dsl, runner_name)
    runner = runner_cls(
        forward_impl,
        num_experts=1,
        top_k=1,
        num_local_experts=1,
        local_expert_offset=0,
        workload_identity=("locality_domain",),
    )
    inputs = [torch.empty(0) for _ in range(num_inputs)]

    result = runner(inputs, tactic=-1, do_preparation=True)

    assert result is inputs[output_idx]
    assert [kwargs["tile_size"] for _, kwargs in preparation_calls] == tile_sizes
    for args, kwargs in preparation_calls:
        assert args == tuple(inputs)
        assert kwargs["enable_alltoall"] is False
        assert kwargs["overlap_moe_output_memset"] is False


def _cute_dsl_eligibility(**deployment_kwargs):
    """``can_implement`` verdict for NVFP4 on an SM107 machine with Rubin DSL."""
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
        MoEDeployment,
        MoEEnvironment,
        MoEProblem,
    )
    from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep

    problem = MoEProblem(quant="NVFP4", dtype_act=torch.bfloat16)
    deployment = MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=8,
        env=MoEEnvironment(
            sm=107,
            available_deps=(MoEDep.CUTEDSL_RUBIN.value, MoEDep.LOCALITY_DOMAIN.value),
        ),
        **deployment_kwargs,
    )
    return CuteDslFusedMoE.can_implement(problem, deployment)


def test_cute_dsl_locality_domain_rejects_eplb():
    """Localized shards cannot follow EPLB migration, so selection declines."""
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERejectReason

    verdict = _cute_dsl_eligibility(locality_domain_requested=True, eplb_enabled=True)
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.EPLB_UNSUPPORTED
    assert _cute_dsl_eligibility(locality_domain_requested=True).eligible
    assert _cute_dsl_eligibility(eplb_enabled=True).eligible


def test_cute_dsl_sm107_requires_fused_finalize():
    """SM107 has no unfused FC2, so disabling finalize fusion declines."""
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERejectReason

    verdict = _cute_dsl_eligibility(fused_finalize_enabled=False)
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.FINALIZE_FUSION_REQUIRED


def test_cute_dsl_unquantized_rejects_non_swiglu():
    """The BF16 FC1 op fuses SwiGLU by name; NVFP4 serves both activations."""
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
        MoEDeployment,
        MoEEnvironment,
        MoEProblem,
        MoERejectReason,
    )
    from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep

    env = MoEEnvironment(sm=107, available_deps=(MoEDep.CUTEDSL_RUBIN.value,))
    deployment = MoEDeployment(
        ep_size=1, tp_size=1, parallel_size=1, use_dp=False, num_slots=8, env=env
    )

    def verdict(quant):
        return CuteDslFusedMoE.can_implement(
            MoEProblem(quant=quant, dtype_act=torch.bfloat16, activation="Relu2"), deployment
        )

    assert not verdict(None).eligible
    assert verdict(None).reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED
    assert verdict("NVFP4").eligible


def test_cute_dsl_locality_domain_disables_dwdp():
    """DWDP rebinds parameters, so it stays off rather than raising."""
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

    moe = SimpleNamespace(
        backend=SimpleNamespace(
            capabilities=SimpleNamespace(supports_dwdp=True),
            uses_locality_domain=True,
        )
    )
    assert ConfigurableMoE._should_enable_dwdp(moe) is False


@pytest.mark.parametrize("quantized", [True, False], ids=["nvfp4", "bf16"])
def test_cute_dsl_moe_zero_tokens_short_circuits_autotune(quantized: bool):
    hidden_size = 8
    top_k = 2
    token_selected_experts = torch.empty((0, top_k), dtype=torch.int32)
    token_final_scales = torch.empty((0, top_k), dtype=torch.float32)
    moe_output = torch.empty((0, hidden_size), dtype=torch.bfloat16)

    if quantized:
        backend = SimpleNamespace(
            has_nvfp4=True,
            has_deepseek_fp8_block_scales=False,
            has_any_quant=True,
            activation_type=ActivationType.Swiglu,
            _locality_domain_runtime=object(),
            hidden_size=hidden_size,
            scaling_vector_size=4,
        )
        x, x_sf = CuteDslFusedMoE.quantize_input(
            backend,
            Fp4QuantizedTensor(
                fp4_tensor=torch.empty((0, hidden_size // 2), dtype=torch.uint8),
                scaling_factor=torch.empty(0, dtype=torch.uint8),
                is_sf_swizzled=False,
            ),
        )
        assert x_sf.shape == (0, hidden_size // backend.scaling_vector_size)
        output = CuteDslFusedMoE.run_moe_nvfp4(
            backend,
            x,
            token_selected_experts,
            token_final_scales,
            x_sf=x_sf,
            moe_output=moe_output,
            weight_view=object(),
        )
    else:
        backend = SimpleNamespace(
            has_any_quant=False,
            hidden_size=hidden_size,
            use_fused_finalize=True,
        )
        output = CuteDslFusedMoE.run_moe_bf16(
            backend,
            torch.empty((0, hidden_size), dtype=torch.bfloat16),
            token_selected_experts,
            token_final_scales,
            moe_output=moe_output,
        )

    assert output is moe_output


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
def test_cute_dsl_nvfp4_quantize_empty_input_rubin():
    hidden_size = 32
    scaling_vector_size = 16
    backend = SimpleNamespace(
        has_nvfp4=True,
        has_deepseek_fp8_block_scales=False,
        has_any_quant=True,
        fc31_input_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        hidden_size=hidden_size,
        scaling_vector_size=scaling_vector_size,
    )

    x, x_sf = CuteDslFusedMoE.quantize_input(
        backend,
        torch.empty((0, hidden_size), dtype=torch.bfloat16, device="cuda"),
    )

    assert x.shape == (0, hidden_size // 2)
    assert x_sf.shape == (0, hidden_size // scaling_vector_size)


def test_sm107_nvfp4_tile512_fallback_uses_two_cta_cluster():
    mma_tiler, mma_inst_shape, cluster_shape_mn = _get_sm107_nvfp4_default_mma_config(512)

    assert mma_tiler == (512, 128, 256)
    assert mma_inst_shape == (256, 128, 128)
    assert cluster_shape_mn == (2, 1)


def _make_sm107_nvfp4_finalize_inputs(n: int = 384) -> list[torch.Tensor]:
    m = 128
    packed_k = 32
    scale_k = packed_k * 2 // 16
    num_tokens = 2
    return [
        torch.empty((m, packed_k), dtype=torch.float4_e2m1fn_x2),
        torch.empty((1, n, packed_k), dtype=torch.float4_e2m1fn_x2),
        torch.empty(m * scale_k, dtype=torch.uint8),
        torch.empty((1, n, scale_k), dtype=torch.uint8),
        torch.empty(1, dtype=torch.float32),
        torch.empty((num_tokens, n), dtype=torch.bfloat16),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty(m, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty((num_tokens, 1), dtype=torch.float32),
    ]


def _make_sm107_nvfp4_finalize_runner(monkeypatch):
    monkeypatch.setattr(cute_dsl_custom_ops, "get_sm_version", lambda: 107)
    runner_cls = cute_dsl_custom_ops.Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner
    return runner_cls(
        num_experts=1,
        top_k=1,
        num_local_experts=1,
        local_expert_offset=0,
        tile_size=128,
        output_dtype=torch.bfloat16,
    )


@pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="This test requires the public Rubin CuTe DSL package",
)
def test_sm107_nvfp4_finalize_get_valid_tactics_filters_n_tiling(monkeypatch):
    runner = _make_sm107_nvfp4_finalize_runner(monkeypatch)
    monkeypatch.setattr(runner.kernel_class, "can_implement", lambda **_: True)

    tactics = runner.get_valid_tactics(_make_sm107_nvfp4_finalize_inputs(), OptimizationProfile())

    assert tactics
    assert all(384 % (tactic[0][1] * tactic[2][1]) == 0 for tactic in tactics)


@pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="This test requires the public Rubin CuTe DSL package",
)
def test_sm107_nvfp4_finalize_forward_rejects_incompatible_n_tiling(monkeypatch):
    runner = _make_sm107_nvfp4_finalize_runner(monkeypatch)
    incompatible_tactic = ((128, 128, 256), (128, 128, 128), (1, 2), False)

    with pytest.raises(ValueError, match="incompatible with N=384"):
        runner(_make_sm107_nvfp4_finalize_inputs(), tactic=incompatible_tactic)


@pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="This test requires the public Rubin CuTe DSL package",
)
def test_sm107_nvfp4_finalize_accepts_two_cta_n_tiling():
    runner_cls = cute_dsl_custom_ops.Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner

    assert runner_cls._is_n_tiling_compatible(n=256, mma_n=128, cluster_n=2)


def _get_registered_rubin_moe_op(name: str):
    try:
        return getattr(torch.ops.trtllm, name)
    except AttributeError:
        pytest.skip("public Rubin CuTe DSL MoE ops are not registered")


def _assert_rubin_moe_op_schema(
    op_name: str,
    argument_names: tuple[str, ...],
    default_values: dict[str, bool | int | None],
    mutated_arguments: set[str],
    return_types: tuple[str, ...],
) -> None:
    schema = _get_registered_rubin_moe_op(op_name).default._schema

    assert tuple(argument.name for argument in schema.arguments) == argument_names
    assert {
        argument.name: argument.default_value
        for argument in schema.arguments
        if argument.has_default_value()
    } == default_values
    assert {
        argument.name
        for argument in schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    } == mutated_arguments
    assert tuple(str(result.type) for result in schema.returns) == return_types


@pytest.mark.parametrize(
    "op_name,argument_names,default_values,mutated_arguments,return_types",
    [
        pytest.param(
            "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin",
            (
                "input",
                "weight",
                "input_scale",
                "weight_scale",
                "alpha",
                "tile_idx_to_group_idx",
                "tile_idx_to_mn_limit",
                "permuted_idx_to_expanded_idx",
                "num_non_exiting_tiles",
                "global_sf",
                "num_experts",
                "top_k",
                "num_local_experts",
                "local_expert_offset",
                "tile_size",
                "output_tensor",
                "output_sf_tensor",
                "scaling_vector_size",
                "partition_id",
                "activation_type",
                "precomputed_tactic",
            ),
            {
                "scaling_vector_size": 16,
                "partition_id": -1,
                "activation_type": int(ActivationType.Swiglu),
                "precomputed_tactic": None,
            },
            {"output_tensor", "output_sf_tensor"},
            ("Optional[Tensor]", "Optional[Tensor]"),
            id="nvfp4_fc1",
        ),
        pytest.param(
            "cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin",
            (
                "input",
                "weight",
                "input_scale",
                "weight_scale",
                "alpha",
                "output",
                "tile_idx_to_group_idx",
                "tile_idx_to_mn_limit",
                "permuted_idx_to_expanded_idx",
                "num_non_exiting_tiles",
                "token_final_scales",
                "num_experts",
                "top_k",
                "num_local_experts",
                "local_expert_offset",
                "tile_size",
                "output_dtype",
                "scaling_vector_size",
                "precomputed_tactic",
            ),
            {"scaling_vector_size": 16, "precomputed_tactic": None},
            {"output"},
            (),
            id="nvfp4_fc2",
        ),
        pytest.param(
            "cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin",
            (
                "input",
                "weight",
                "alpha",
                "tile_idx_to_group_idx",
                "tile_idx_to_mn_limit",
                "permuted_idx_to_expanded_idx",
                "num_non_exiting_tiles",
                "num_experts",
                "top_k",
                "num_local_experts",
                "local_expert_offset",
                "tile_size",
                "output_tensor",
                "partition_id",
                "precomputed_tactic",
            ),
            {"precomputed_tactic": None},
            {"output_tensor"},
            ("Optional[Tensor]",),
            id="bf16_fc1",
        ),
        pytest.param(
            "cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin",
            (
                "input",
                "weight",
                "output",
                "tile_idx_to_group_idx",
                "tile_idx_to_mn_limit",
                "permuted_idx_to_expanded_idx",
                "num_non_exiting_tiles",
                "token_final_scales",
                "num_experts",
                "top_k",
                "num_local_experts",
                "local_expert_offset",
                "tile_size",
                "output_dtype",
                "precomputed_tactic",
            ),
            {"precomputed_tactic": None},
            {"output"},
            (),
            id="bf16_fc2",
        ),
    ],
)
def test_rubin_moe_leaf_schema(
    op_name: str,
    argument_names: tuple[str, ...],
    default_values: dict[str, bool | int | None],
    mutated_arguments: set[str],
    return_types: tuple[str, ...],
) -> None:
    _assert_rubin_moe_op_schema(
        op_name,
        argument_names,
        default_values,
        mutated_arguments,
        return_types,
    )


def _make_fake_rubin_moe_tensors(quantized: bool) -> SimpleNamespace:
    dtype = torch.uint8 if quantized else torch.bfloat16
    fc1_n = 32 if quantized else 8
    fc2_n = 32 if quantized else 4
    output_n = 32 if quantized else 8
    return SimpleNamespace(
        input=torch.empty((2, 4), dtype=dtype, device="cuda"),
        fc1_weight=torch.empty((1, fc1_n, 4), dtype=dtype, device="cuda"),
        fc2_weight=torch.empty((1, fc2_n, 4), dtype=dtype, device="cuda"),
        input_scale=torch.empty(2, dtype=torch.uint8, device="cuda"),
        fc1_weight_scale=torch.empty((1, fc1_n, 1), dtype=torch.uint8, device="cuda"),
        fc2_weight_scale=torch.empty((1, fc2_n, 1), dtype=torch.uint8, device="cuda"),
        alpha=torch.empty(1, dtype=torch.float32, device="cuda"),
        tile_idx_to_group_idx=torch.empty(1, dtype=torch.int32, device="cuda"),
        tile_idx_to_mn_limit=torch.empty(1, dtype=torch.int32, device="cuda"),
        expanded_idx_to_permuted_idx=torch.empty((2, 1), dtype=torch.int32, device="cuda"),
        permuted_idx_to_expanded_idx=torch.empty(2, dtype=torch.int32, device="cuda"),
        num_non_exiting_tiles=torch.empty(1, dtype=torch.int32, device="cuda"),
        global_sf=torch.empty(1, dtype=torch.float32, device="cuda"),
        token_final_scales=torch.empty((2, 1), dtype=torch.float32, device="cuda"),
        fc1_output=torch.empty((2, 8), dtype=dtype, device="cuda"),
        fc1_output_sf=torch.empty(2, dtype=torch.uint8, device="cuda"),
        output=torch.empty((2, output_n), dtype=torch.bfloat16, device="cuda"),
    )


def test_rubin_moe_precomputed_tactic_fake_signatures():
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        pytest.skip("FakeTensorMode is not available")

    fc1_op = _get_registered_rubin_moe_op("cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin")
    fc2_op = _get_registered_rubin_moe_op("cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin")

    with FakeTensorMode():
        tensors = _make_fake_rubin_moe_tensors(quantized=True)

        output, output_sf = fc1_op(
            input=tensors.input,
            weight=tensors.fc1_weight,
            input_scale=tensors.input_scale,
            weight_scale=tensors.fc1_weight_scale,
            alpha=tensors.alpha,
            tile_idx_to_group_idx=tensors.tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tensors.tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=tensors.permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=tensors.num_non_exiting_tiles,
            global_sf=tensors.global_sf,
            num_experts=1,
            top_k=1,
            num_local_experts=1,
            local_expert_offset=0,
            tile_size=128,
            output_tensor=None,
            output_sf_tensor=None,
            scaling_vector_size=16,
            partition_id=-1,
            activation_type=int(ActivationType.Swiglu),
            precomputed_tactic=repr(-1),
        )
        assert output.shape == (2, 8)
        assert output_sf.shape == (2,)

        result = fc2_op(
            input=tensors.input,
            weight=tensors.fc2_weight,
            input_scale=tensors.input_scale,
            weight_scale=tensors.fc2_weight_scale,
            alpha=tensors.alpha,
            output=tensors.output,
            tile_idx_to_group_idx=tensors.tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tensors.tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=tensors.permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=tensors.num_non_exiting_tiles,
            token_final_scales=tensors.token_final_scales,
            num_experts=1,
            top_k=1,
            num_local_experts=1,
            local_expert_offset=0,
            tile_size=128,
            output_dtype=torch.bfloat16,
            scaling_vector_size=16,
            precomputed_tactic=repr(-1),
        )
        assert result is None


def test_rubin_bf16_moe_precomputed_tactic_fake_signatures():
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        pytest.skip("FakeTensorMode is not available")

    fc1_op = _get_registered_rubin_moe_op("cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin")
    fc2_op = _get_registered_rubin_moe_op("cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin")

    with FakeTensorMode():
        tensors = _make_fake_rubin_moe_tensors(quantized=False)

        result = fc1_op(
            input=tensors.input,
            weight=tensors.fc1_weight,
            alpha=tensors.alpha,
            tile_idx_to_group_idx=tensors.tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tensors.tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=tensors.permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=tensors.num_non_exiting_tiles,
            num_experts=1,
            top_k=1,
            num_local_experts=1,
            local_expert_offset=0,
            tile_size=128,
            output_tensor=tensors.fc1_output,
            partition_id=0,
            precomputed_tactic=repr(-1),
        )
        assert result is None

        result = fc2_op(
            input=tensors.input,
            weight=tensors.fc2_weight,
            output=tensors.output,
            tile_idx_to_group_idx=tensors.tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tensors.tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=tensors.permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=tensors.num_non_exiting_tiles,
            token_final_scales=tensors.token_final_scales,
            num_experts=1,
            top_k=1,
            num_local_experts=1,
            local_expert_offset=0,
            tile_size=128,
            output_dtype=torch.bfloat16,
            precomputed_tactic=repr(-1),
        )
        assert result is None


@pytest.mark.parametrize(
    "op_name,argument_names,default_values,mutated_arguments",
    [
        pytest.param(
            "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_locality_domain_inplace_rubin",
            (
                "input",
                "weight_0",
                "weight_1",
                "input_scale",
                "weight_scale_0",
                "weight_scale_1",
                "alpha",
                "tile_idx_to_group_idx",
                "tile_idx_to_mn_limit",
                "permuted_idx_to_expanded_idx",
                "num_non_exiting_tiles",
                "global_sf",
                "num_experts",
                "top_k",
                "num_local_experts",
                "local_expert_offset",
                "tile_size",
                "output_tensor",
                "output_sf_tensor",
                "scaling_vector_size",
                "activation_type",
            ),
            {
                "scaling_vector_size": 16,
                "activation_type": int(ActivationType.Swiglu),
            },
            {"output_tensor", "output_sf_tensor"},
            id="nvfp4_fc1",
        ),
        pytest.param(
            "cute_dsl_nvfp4_grouped_gemm_finalize_locality_domain_inplace_rubin",
            (
                "input",
                "weight_0",
                "weight_1",
                "input_scale",
                "weight_scale_0",
                "weight_scale_1",
                "alpha",
                "output",
                "tile_idx_to_group_idx",
                "tile_idx_to_mn_limit",
                "expanded_idx_to_permuted_idx",
                "permuted_idx_to_expanded_idx",
                "num_non_exiting_tiles",
                "token_final_scales",
                "num_experts",
                "top_k",
                "num_local_experts",
                "local_expert_offset",
                "tile_size",
                "output_dtype",
                "ep_size",
                "enable_alltoall",
                "scaling_vector_size",
            ),
            {"enable_alltoall": False, "scaling_vector_size": 16},
            {"output"},
            id="nvfp4_fc2",
        ),
        pytest.param(
            "cute_dsl_bf16_gather_grouped_gemm_swiglu_locality_domain_inplace_rubin",
            (
                "input",
                "weight_0",
                "weight_1",
                "alpha",
                "tile_idx_to_group_idx",
                "tile_idx_to_mn_limit",
                "permuted_idx_to_expanded_idx",
                "num_non_exiting_tiles",
                "num_experts",
                "top_k",
                "num_local_experts",
                "local_expert_offset",
                "tile_size",
                "output_tensor",
            ),
            {},
            {"output_tensor"},
            id="bf16_fc1",
        ),
        pytest.param(
            "cute_dsl_bf16_grouped_gemm_finalize_locality_domain_inplace_rubin",
            (
                "input",
                "weight_0",
                "weight_1",
                "output",
                "tile_idx_to_group_idx",
                "tile_idx_to_mn_limit",
                "expanded_idx_to_permuted_idx",
                "permuted_idx_to_expanded_idx",
                "num_non_exiting_tiles",
                "token_final_scales",
                "num_experts",
                "top_k",
                "num_local_experts",
                "local_expert_offset",
                "tile_size",
                "output_dtype",
                "ep_size",
                "enable_alltoall",
            ),
            {"enable_alltoall": False},
            {"output"},
            id="bf16_fc2",
        ),
    ],
)
def test_rubin_moe_locality_domain_composite_schema(
    op_name: str,
    argument_names: tuple[str, ...],
    default_values: dict[str, bool | int | None],
    mutated_arguments: set[str],
) -> None:
    _assert_rubin_moe_op_schema(
        op_name,
        argument_names,
        default_values,
        mutated_arguments,
        (),
    )


def test_rubin_nvfp4_moe_locality_domain_composite_fake_signatures():
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        pytest.skip("FakeTensorMode is not available")

    fc1_op = _get_registered_rubin_moe_op(
        "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_locality_domain_inplace_rubin"
    )
    fc2_op = _get_registered_rubin_moe_op(
        "cute_dsl_nvfp4_grouped_gemm_finalize_locality_domain_inplace_rubin"
    )

    with FakeTensorMode():
        tensors = _make_fake_rubin_moe_tensors(quantized=True)

        result = fc1_op(
            input=tensors.input,
            weight_0=tensors.fc1_weight,
            weight_1=tensors.fc1_weight,
            input_scale=tensors.input_scale,
            weight_scale_0=tensors.fc1_weight_scale,
            weight_scale_1=tensors.fc1_weight_scale,
            alpha=tensors.alpha,
            tile_idx_to_group_idx=tensors.tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tensors.tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=tensors.permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=tensors.num_non_exiting_tiles,
            global_sf=tensors.global_sf,
            num_experts=1,
            top_k=1,
            num_local_experts=1,
            local_expert_offset=0,
            tile_size=128,
            output_tensor=tensors.fc1_output,
            output_sf_tensor=tensors.fc1_output_sf,
            scaling_vector_size=16,
            activation_type=int(ActivationType.Swiglu),
        )
        assert result is None

        result = fc2_op(
            input=tensors.input,
            weight_0=tensors.fc2_weight,
            weight_1=tensors.fc2_weight,
            input_scale=tensors.input_scale,
            weight_scale_0=tensors.fc2_weight_scale,
            weight_scale_1=tensors.fc2_weight_scale,
            alpha=tensors.alpha,
            output=tensors.output,
            tile_idx_to_group_idx=tensors.tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tensors.tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx=tensors.expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx=tensors.permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=tensors.num_non_exiting_tiles,
            token_final_scales=tensors.token_final_scales,
            num_experts=1,
            top_k=1,
            num_local_experts=1,
            local_expert_offset=0,
            tile_size=128,
            output_dtype=torch.bfloat16,
            ep_size=1,
            enable_alltoall=False,
            scaling_vector_size=16,
        )
        assert result is None


def test_rubin_bf16_moe_locality_domain_composite_fake_signatures():
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        pytest.skip("FakeTensorMode is not available")

    fc1_op = _get_registered_rubin_moe_op(
        "cute_dsl_bf16_gather_grouped_gemm_swiglu_locality_domain_inplace_rubin"
    )
    fc2_op = _get_registered_rubin_moe_op(
        "cute_dsl_bf16_grouped_gemm_finalize_locality_domain_inplace_rubin"
    )

    with FakeTensorMode():
        tensors = _make_fake_rubin_moe_tensors(quantized=False)

        result = fc1_op(
            input=tensors.input,
            weight_0=tensors.fc1_weight,
            weight_1=tensors.fc1_weight,
            alpha=tensors.alpha,
            tile_idx_to_group_idx=tensors.tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tensors.tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=tensors.permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=tensors.num_non_exiting_tiles,
            num_experts=1,
            top_k=1,
            num_local_experts=1,
            local_expert_offset=0,
            tile_size=128,
            output_tensor=tensors.fc1_output,
        )
        assert result is None

        result = fc2_op(
            input=tensors.input,
            weight_0=tensors.fc2_weight,
            weight_1=tensors.fc2_weight,
            output=tensors.output,
            tile_idx_to_group_idx=tensors.tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tensors.tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx=tensors.expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx=tensors.permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=tensors.num_non_exiting_tiles,
            token_final_scales=tensors.token_final_scales,
            num_experts=1,
            top_k=1,
            num_local_experts=1,
            local_expert_offset=0,
            tile_size=128,
            output_dtype=torch.bfloat16,
            ep_size=1,
            enable_alltoall=False,
        )
        assert result is None


@pytest.mark.parametrize(
    "composite_name,leaf_name,runner_name,quantized,is_fc1",
    [
        (
            "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_locality_domain_inplace_rubin",
            "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin",
            "Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner",
            True,
            True,
        ),
        (
            "cute_dsl_nvfp4_grouped_gemm_finalize_locality_domain_inplace_rubin",
            "cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin",
            "Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner",
            True,
            False,
        ),
        (
            "cute_dsl_bf16_gather_grouped_gemm_swiglu_locality_domain_inplace_rubin",
            "cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin",
            "Sm107ContiguousGatherGroupedGemmSwigluFusionRunner",
            False,
            True,
        ),
        (
            "cute_dsl_bf16_grouped_gemm_finalize_locality_domain_inplace_rubin",
            "cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin",
            "Sm107ContiguousGroupedGemmFinalizeFusionRunner",
            False,
            False,
        ),
    ],
)
def test_rubin_moe_locality_domain_composite_owns_concurrent_tuning(
    monkeypatch,
    composite_name: str,
    leaf_name: str,
    runner_name: str,
    quantized: bool,
    is_fc1: bool,
):
    composite_op = getattr(cute_dsl_custom_ops, composite_name, None)
    if composite_op is None:
        pytest.skip("public Rubin CuTe DSL MoE composite op is not registered")

    tuning_config = object()
    chosen_tactic = ("chosen", 128)
    runner_instances = []
    runtime_instances = []
    tune_calls = []
    concurrent_calls = []
    leaf_calls = []
    memset_calls = []
    execution_order = []
    tuner_state = SimpleNamespace(is_tuning_mode=not is_fc1)

    class FakeOpRunner:
        def __init__(self, *args, **kwargs):
            self.init_call = (args, kwargs)
            self.tuning_config_calls = []
            runner_instances.append(self)

        def get_tuning_config(self, *args, **kwargs):
            self.tuning_config_calls.append((args, kwargs))
            return tuning_config

    class FakeRuntime:
        def __init__(self, num_partitions: int):
            self.num_partitions = num_partitions
            runtime_instances.append(self)

    class FakeConcurrentRunner:
        def __init__(self, launch_partition):
            self.launch_partition = launch_partition

        def __call__(self, inputs, *, tactic):
            execution_order.append("launch")
            concurrent_calls.append((inputs, tactic))
            for partition_id in range(2):
                self.launch_partition(partition_id, inputs, tactic)

    def fake_tune_locality_domain_concurrent(
        op_name,
        op_runner,
        runtime,
        num_partitions,
        launch_partition,
        inputs,
        actual_tuning_config,
    ):
        execution_order.append("tune")
        tune_calls.append(
            (
                op_name,
                op_runner,
                runtime,
                num_partitions,
                launch_partition,
                inputs,
                actual_tuning_config,
            )
        )
        return FakeConcurrentRunner(launch_partition), chosen_tactic

    def fake_leaf_op(*args, **kwargs):
        leaf_calls.append((args, kwargs))

    def fake_moe_output_memset(*args, **kwargs):
        execution_order.append("reset")
        memset_calls.append((args, kwargs))

    monkeypatch.setattr(cute_dsl_custom_ops, runner_name, FakeOpRunner)
    monkeypatch.setattr(cute_dsl_custom_ops, "LocalityDomainRuntime", FakeRuntime)
    monkeypatch.setattr(
        cute_dsl_custom_ops,
        "tune_locality_domain_concurrent",
        fake_tune_locality_domain_concurrent,
    )
    monkeypatch.setattr(cute_dsl_custom_ops, "get_sm_version", lambda: 107)
    monkeypatch.setattr(
        AutoTuner,
        "get",
        staticmethod(lambda: tuner_state),
    )
    monkeypatch.setattr(torch.ops.trtllm, leaf_name, fake_leaf_op)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "moe_output_memset_inplace",
        fake_moe_output_memset,
    )

    dtype = torch.uint8 if quantized else torch.bfloat16
    input_tensor = torch.empty((2, 4), dtype=dtype)
    weight_0 = torch.empty((1, 8, 4), dtype=dtype)
    weight_1 = torch.empty_like(weight_0)
    alpha = torch.empty((1,), dtype=torch.float32)
    tile_idx_to_group_idx = torch.empty((1,), dtype=torch.int32)
    tile_idx_to_mn_limit = torch.empty((1,), dtype=torch.int32)
    expanded_idx_to_permuted_idx = torch.empty((2, 1), dtype=torch.int32)
    permuted_idx_to_expanded_idx = torch.empty((2,), dtype=torch.int32)
    num_non_exiting_tiles = torch.empty((1,), dtype=torch.int32)
    token_final_scales = torch.empty((2, 1), dtype=torch.float32)
    output = torch.empty((2, 16), dtype=torch.bfloat16)

    common_kwargs = {
        "input": input_tensor,
        "weight_0": weight_0,
        "weight_1": weight_1,
        "tile_idx_to_group_idx": tile_idx_to_group_idx,
        "tile_idx_to_mn_limit": tile_idx_to_mn_limit,
        "permuted_idx_to_expanded_idx": permuted_idx_to_expanded_idx,
        "num_non_exiting_tiles": num_non_exiting_tiles,
        "num_experts": 1,
        "top_k": 1,
        "num_local_experts": 1,
        "local_expert_offset": 0,
        "tile_size": 128,
    }
    if is_fc1:
        call_kwargs = {
            **common_kwargs,
            "alpha": alpha,
            "output_tensor": output if not quantized else torch.empty((2, 8), dtype=torch.uint8),
        }
        if quantized:
            weight_scale_0 = torch.empty((1, 8, 1), dtype=torch.uint8)
            weight_scale_1 = torch.empty_like(weight_scale_0)
            call_kwargs.update(
                {
                    "input_scale": torch.empty((2,), dtype=torch.uint8),
                    "weight_scale_0": weight_scale_0,
                    "weight_scale_1": weight_scale_1,
                    "global_sf": torch.empty((1,), dtype=torch.float32),
                    "output_sf_tensor": torch.empty((2,), dtype=torch.uint8),
                    "scaling_vector_size": 16,
                    "activation_type": int(ActivationType.Swiglu),
                }
            )
    else:
        call_kwargs = {
            **common_kwargs,
            "output": output,
            "expanded_idx_to_permuted_idx": expanded_idx_to_permuted_idx,
            "token_final_scales": token_final_scales,
            "output_dtype": torch.bfloat16,
            "ep_size": 4,
            "enable_alltoall": True,
        }
        if quantized:
            weight_scale_0 = torch.empty((1, 8, 1), dtype=torch.uint8)
            weight_scale_1 = torch.empty_like(weight_scale_0)
            call_kwargs.update(
                {
                    "input_scale": torch.empty((2,), dtype=torch.uint8),
                    "weight_scale_0": weight_scale_0,
                    "weight_scale_1": weight_scale_1,
                    "alpha": alpha,
                    "scaling_vector_size": 16,
                }
            )

    # Exercise the registered Python implementation directly; dispatcher and
    # fake contracts are covered separately by the schema/fake tests above.
    assert composite_op._init_fn(**call_kwargs) is None

    assert len(runner_instances) == 1
    assert len(runtime_instances) == 1
    assert runtime_instances[0].num_partitions == 2
    assert len(tune_calls) == 1
    (
        tune_key,
        tuned_op_runner,
        tuned_runtime,
        num_partitions,
        _,
        tuned_inputs,
        actual_tuning_config,
    ) = tune_calls[0]
    assert tune_key == f"trtllm::{leaf_name}"
    assert tuned_op_runner is runner_instances[0]
    assert tuned_runtime is runtime_instances[0]
    assert num_partitions == 2
    assert actual_tuning_config is tuning_config
    assert len(runner_instances[0].tuning_config_calls) == 1
    runner_args, runner_kwargs = runner_instances[0].init_call
    if quantized and is_fc1:
        assert runner_args == (1, 1, 1, 0, 128, 16)
        assert runner_kwargs == {"activation_type": ActivationType.Swiglu}
    elif quantized:
        assert runner_args == (1, 1, 1, 0, 128, torch.bfloat16, 16)
        assert not runner_kwargs
    elif is_fc1:
        assert runner_args == (1, 1, 1, 0, 128)
        assert runner_kwargs == {"input_dtype": torch.bfloat16}
        assert runner_instances[0].tuning_config_calls == [((), {"has_output_tensor": True})]
    else:
        assert runner_args == (1, 1, 1, 0, 128, torch.bfloat16)
        assert runner_kwargs == {"input_dtype": torch.bfloat16}
    if quantized or not is_fc1:
        assert runner_instances[0].tuning_config_calls == [((), {})]
    assert len(concurrent_calls) == 1
    assert concurrent_calls[0][0] is tuned_inputs
    assert concurrent_calls[0][1] == chosen_tactic

    assert len(leaf_calls) == 2
    assert leaf_calls[0][1]["weight"] is weight_0
    assert leaf_calls[1][1]["weight"] is weight_1
    for _, leaf_kwargs in leaf_calls:
        assert leaf_kwargs["input"] is input_tensor
        assert leaf_kwargs["tile_idx_to_group_idx"] is tile_idx_to_group_idx
        assert leaf_kwargs["tile_idx_to_mn_limit"] is tile_idx_to_mn_limit
        assert leaf_kwargs["permuted_idx_to_expanded_idx"] is permuted_idx_to_expanded_idx
        assert leaf_kwargs["num_non_exiting_tiles"] is num_non_exiting_tiles
        assert leaf_kwargs["num_experts"] == 1
        assert leaf_kwargs["top_k"] == 1
        assert leaf_kwargs["num_local_experts"] == 1
        assert leaf_kwargs["local_expert_offset"] == 0
        assert leaf_kwargs["tile_size"] == 128
        assert leaf_kwargs["precomputed_tactic"] == repr(chosen_tactic)
    if is_fc1:
        assert [kwargs["partition_id"] for _, kwargs in leaf_calls] == [0, 1]
        assert all(kwargs["alpha"] is alpha for _, kwargs in leaf_calls)
        assert all(
            kwargs["output_tensor"] is call_kwargs["output_tensor"] for _, kwargs in leaf_calls
        )
        assert not memset_calls
        assert execution_order == ["tune", "launch"]
    else:
        assert all("partition_id" not in kwargs for _, kwargs in leaf_calls)
        assert all(kwargs["output"] is output for _, kwargs in leaf_calls)
        assert all(kwargs["token_final_scales"] is token_final_scales for _, kwargs in leaf_calls)
        assert all(kwargs["output_dtype"] == torch.bfloat16 for _, kwargs in leaf_calls)
        assert execution_order == ["tune", "reset", "launch"]
        assert len(memset_calls) == 1
        reset_args, reset_kwargs = memset_calls[0]
        assert not reset_args
        assert reset_kwargs == {
            "input": output,
            "tile_idx_to_mn_limit": tile_idx_to_mn_limit,
            "expanded_idx_to_permuted_idx": expanded_idx_to_permuted_idx,
            "permuted_idx_to_expanded_idx": permuted_idx_to_expanded_idx,
            "num_non_exiting_tiles": num_non_exiting_tiles,
            "tile_tokens_dim": 128,
            "top_k": 1,
            "ep_size": 4,
            "enable_alltoall": True,
        }
        execution_order.clear()
        memset_calls.clear()
        tuner_state.is_tuning_mode = False
        assert composite_op._init_fn(**call_kwargs) is None
        assert execution_order == ["tune", "launch"]
        assert not memset_calls
    if quantized:
        assert leaf_calls[0][1]["weight_scale"] is weight_scale_0
        assert leaf_calls[1][1]["weight_scale"] is weight_scale_1
        assert all(kwargs["input_scale"] is call_kwargs["input_scale"] for _, kwargs in leaf_calls)
        assert all(kwargs["alpha"] is alpha for _, kwargs in leaf_calls)
        assert all(kwargs["scaling_vector_size"] == 16 for _, kwargs in leaf_calls)
        if is_fc1:
            assert all(kwargs["global_sf"] is call_kwargs["global_sf"] for _, kwargs in leaf_calls)
            assert all(
                kwargs["output_sf_tensor"] is call_kwargs["output_sf_tensor"]
                for _, kwargs in leaf_calls
            )
            assert all(
                kwargs["activation_type"] == int(ActivationType.Swiglu) for _, kwargs in leaf_calls
            )


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize("tile_size", [128])
@pytest.mark.parametrize("ep_size", [1, 8])
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("num_tokens", [128, 515])
def test_nvfp4_gather_grouped_gemm_swiglu_locality_domain_rubin(
    num_tokens: int, top_k: int, ep_size: int, tile_size: int
):
    _skip_if_no_locality_domain()

    sf_vec_size = 16
    hidden_size = 2048
    interm_size = 1536
    num_experts = 256
    num_local_experts = num_experts // ep_size

    (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
        _,
    ) = _setup_locality_domain_routing(num_tokens, num_experts, num_local_experts, top_k, tile_size)
    max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0)

    a_fp4, a_sf_unswizzled, a_global_sf = _create_quantized_locality_domain_inputs(
        num_tokens, hidden_size, sf_vec_size
    )
    weight, weight_sf, weight_global_sf = _create_quantized_locality_domain_weights(
        num_local_experts, interm_size, hidden_size, sf_vec_size
    )
    alpha = a_global_sf * weight_global_sf
    global_sf = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    c_ref, c_sf_ref = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin(
        a_fp4,
        weight,
        a_sf_unswizzled,
        weight_sf,
        alpha,
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        global_sf,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        output_tensor=None,
        output_sf_tensor=None,
        partition_id=-1,
        activation_type=ActivationType.Swiglu,
    )

    half_weight_n = weight.size(1) // 2
    c_locality_domain = torch.empty(
        max_num_permuted_tokens, interm_size // 2, dtype=a_fp4.dtype, device=a_fp4.device
    )
    c_sf_locality_domain = torch.empty(
        max_num_permuted_tokens * interm_size // sf_vec_size, dtype=torch.uint8, device=a_fp4.device
    )

    start_for_all_locality_domain()
    try:
        for locality_domain_id in range(2):
            with locality_domain_device(locality_domain_id):
                with torch.cuda.stream(get_locality_domain_stream(locality_domain_id)):
                    weight_shard = weight.view(torch.uint8)[
                        :,
                        locality_domain_id * half_weight_n : (locality_domain_id + 1)
                        * half_weight_n,
                    ]
                    weight_shard = weight_shard.contiguous().view(torch.float4_e2m1fn_x2)
                    weight_sf_shard = weight_sf[
                        :,
                        locality_domain_id * half_weight_n : (locality_domain_id + 1)
                        * half_weight_n,
                    ].contiguous()
                    torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin(
                        a_fp4,
                        weight_shard,
                        a_sf_unswizzled,
                        weight_sf_shard,
                        alpha,
                        tile_idx_to_group_idx,
                        tile_idx_to_mn_limit,
                        permuted_idx_to_expanded_idx,
                        num_non_exiting_tiles,
                        global_sf,
                        num_experts=num_experts,
                        top_k=top_k,
                        num_local_experts=num_local_experts,
                        local_expert_offset=0,
                        tile_size=tile_size,
                        scaling_vector_size=sf_vec_size,
                        output_tensor=c_locality_domain,
                        output_sf_tensor=c_sf_locality_domain,
                        partition_id=locality_domain_id,
                        activation_type=ActivationType.Swiglu,
                    )
    finally:
        end_for_all_locality_domain()
    torch.cuda.synchronize()

    num_valid = total_num_padded_tokens.item()
    valid_mask = _valid_permuted_token_mask(tile_idx_to_mn_limit, num_valid, tile_size)
    c_valid = c_locality_domain.view(torch.uint8)[:num_valid][valid_mask]
    c_ref_valid = c_ref.view(torch.uint8)[:num_valid][valid_mask]
    assert c_valid.any()
    torch.testing.assert_close(c_valid, c_ref_valid)

    c_sf_locality_domain = unswizzle_sf(
        c_sf_locality_domain, max_num_permuted_tokens, interm_size, sf_vec_size
    )
    c_sf_ref = unswizzle_sf(c_sf_ref, max_num_permuted_tokens, interm_size, sf_vec_size)
    torch.testing.assert_close(
        c_sf_locality_domain[:num_valid][valid_mask], c_sf_ref[:num_valid][valid_mask]
    )


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize("tile_size", [128])
@pytest.mark.parametrize("ep_size", [1, 8])
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("num_tokens", [128, 515])
def test_nvfp4_grouped_gemm_finalize_locality_domain_rubin(
    num_tokens: int, top_k: int, ep_size: int, tile_size: int
):
    _skip_if_no_locality_domain()

    sf_vec_size = 16
    hidden_size = 2048
    interm_size = 1536
    num_experts = 256
    num_local_experts = num_experts // ep_size

    (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _,
        num_non_exiting_tiles,
        token_final_scales,
    ) = _setup_locality_domain_routing(num_tokens, num_experts, num_local_experts, top_k, tile_size)
    max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0)

    torch.manual_seed(99)
    fc2_input_bf16 = (
        torch.randn(max_num_permuted_tokens, interm_size, dtype=torch.bfloat16, device="cuda")
        * 0.05
    )
    fc2_global_sf = fc2_input_bf16.abs().max().float() / (448 * 6)
    fc2_input, fc2_input_sf = torch.ops.trtllm.fp4_quantize(
        fc2_input_bf16, 1 / fc2_global_sf, sf_vec_size, False
    )
    fc2_input = fc2_input.view(torch.float4_e2m1fn_x2)

    fc2_weight_bf16 = (
        torch.randn(
            num_local_experts, hidden_size, interm_size, dtype=torch.bfloat16, device="cuda"
        )
        * 0.05
    )
    fc2_weight_global_sf = fc2_weight_bf16.abs().amax(dim=(1, 2)).float() / (448 * 6)
    fc2_weight, fc2_weight_sf = torch.ops.trtllm.fp4_quantize(
        fc2_weight_bf16, 1 / fc2_weight_global_sf, sf_vec_size, False
    )
    fc2_weight = fc2_weight.view(torch.float4_e2m1fn_x2)
    fc2_weight_sf = fc2_weight_sf.view(num_local_experts, hidden_size, interm_size // sf_vec_size)
    fc2_alpha = fc2_global_sf * fc2_weight_global_sf

    output_ref = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_rubin(
        input=fc2_input,
        weight=fc2_weight,
        input_scale=fc2_input_sf.view(torch.uint8),
        weight_scale=fc2_weight_sf.view(torch.uint8),
        alpha=fc2_alpha,
        tile_idx_to_group_idx=tile_idx_to_group_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        output_dtype=torch.bfloat16,
    )

    output = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    half_hidden = hidden_size // 2
    start_for_all_locality_domain()
    try:
        for locality_domain_id in range(2):
            with locality_domain_device(locality_domain_id):
                with torch.cuda.stream(get_locality_domain_stream(locality_domain_id)):
                    weight_shard = fc2_weight.view(torch.uint8)[
                        :, locality_domain_id * half_hidden : (locality_domain_id + 1) * half_hidden
                    ]
                    weight_shard = weight_shard.contiguous().view(torch.float4_e2m1fn_x2)
                    weight_sf_shard = fc2_weight_sf[
                        :, locality_domain_id * half_hidden : (locality_domain_id + 1) * half_hidden
                    ]
                    torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin(
                        input=fc2_input,
                        weight=weight_shard,
                        input_scale=fc2_input_sf.view(torch.uint8),
                        weight_scale=weight_sf_shard.contiguous().view(torch.uint8),
                        alpha=fc2_alpha,
                        output=output,
                        tile_idx_to_group_idx=tile_idx_to_group_idx,
                        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                        num_non_exiting_tiles=num_non_exiting_tiles,
                        token_final_scales=token_final_scales,
                        num_experts=num_experts,
                        top_k=top_k,
                        num_local_experts=num_local_experts,
                        local_expert_offset=0,
                        tile_size=tile_size,
                        output_dtype=torch.bfloat16,
                    )
    finally:
        end_for_all_locality_domain()
    torch.cuda.synchronize()

    assert output[:, :half_hidden].any()
    assert output[:, half_hidden:].any()
    torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=0.15)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize("tile_size", [128])
@pytest.mark.parametrize("ep_size", [1, 8])
@pytest.mark.parametrize("top_k", [1, 2])
def test_bf16_gather_grouped_gemm_swiglu_locality_domain_rubin(
    top_k: int, ep_size: int, tile_size: int
):
    _skip_if_no_locality_domain()

    num_tokens = 128
    hidden_size = 2048
    interm_size = 1536
    num_experts = 256
    num_local_experts = num_experts // ep_size

    (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
        _,
    ) = _setup_locality_domain_routing(num_tokens, num_experts, num_local_experts, top_k, tile_size)
    max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0)

    torch.manual_seed(7)
    input_tensor = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.05
    weight = (
        torch.randn(
            num_local_experts, interm_size * 2, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        * 0.05
    )
    weight = interleave_linear_and_gate(weight, group_size=32, dim=1)
    alpha = torch.ones(num_local_experts, dtype=torch.float32, device="cuda")

    output_ref = torch.ops.trtllm.cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin(
        input=input_tensor,
        weight=weight,
        alpha=alpha,
        tile_idx_to_group_idx=tile_idx_to_group_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        output_tensor=None,
        partition_id=-1,
    )

    output = torch.empty(max_num_permuted_tokens, interm_size, dtype=torch.bfloat16, device="cuda")
    half_weight_n = weight.size(1) // 2
    start_for_all_locality_domain()
    try:
        for locality_domain_id in range(2):
            with locality_domain_device(locality_domain_id):
                with torch.cuda.stream(get_locality_domain_stream(locality_domain_id)):
                    torch.ops.trtllm.cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin(
                        input=input_tensor,
                        weight=weight[
                            :,
                            locality_domain_id * half_weight_n : (locality_domain_id + 1)
                            * half_weight_n,
                        ].contiguous(),
                        alpha=alpha,
                        tile_idx_to_group_idx=tile_idx_to_group_idx,
                        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                        num_non_exiting_tiles=num_non_exiting_tiles,
                        num_experts=num_experts,
                        top_k=top_k,
                        num_local_experts=num_local_experts,
                        local_expert_offset=0,
                        tile_size=tile_size,
                        output_tensor=output,
                        partition_id=locality_domain_id,
                    )
    finally:
        end_for_all_locality_domain()
    torch.cuda.synchronize()

    num_valid = total_num_padded_tokens.item()
    valid_mask = _valid_permuted_token_mask(tile_idx_to_mn_limit, num_valid, tile_size)
    torch.testing.assert_close(
        output[:num_valid][valid_mask], output_ref[:num_valid][valid_mask], rtol=1e-2, atol=0.15
    )


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize("tile_size", [128])
@pytest.mark.parametrize("ep_size", [1, 8])
@pytest.mark.parametrize("top_k", [1, 2])
def test_bf16_grouped_gemm_finalize_locality_domain_rubin(top_k: int, ep_size: int, tile_size: int):
    _skip_if_no_locality_domain()

    num_tokens = 128
    hidden_size = 2048
    interm_size = 1536
    num_experts = 256
    num_local_experts = num_experts // ep_size

    (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _,
        num_non_exiting_tiles,
        token_final_scales,
    ) = _setup_locality_domain_routing(num_tokens, num_experts, num_local_experts, top_k, tile_size)
    max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0)

    torch.manual_seed(11)
    input_tensor = (
        torch.randn(max_num_permuted_tokens, interm_size, dtype=torch.bfloat16, device="cuda")
        * 0.05
    )
    weight = (
        torch.randn(
            num_local_experts, hidden_size, interm_size, dtype=torch.bfloat16, device="cuda"
        )
        * 0.05
    )

    output_ref = torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_finalize_rubin(
        input=input_tensor,
        weight=weight,
        tile_idx_to_group_idx=tile_idx_to_group_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        output_dtype=torch.bfloat16,
    )

    output = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    half_hidden = hidden_size // 2
    start_for_all_locality_domain()
    try:
        for locality_domain_id in range(2):
            with locality_domain_device(locality_domain_id):
                with torch.cuda.stream(get_locality_domain_stream(locality_domain_id)):
                    torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin(
                        input=input_tensor,
                        weight=weight[
                            :,
                            locality_domain_id * half_hidden : (locality_domain_id + 1)
                            * half_hidden,
                        ].contiguous(),
                        output=output,
                        tile_idx_to_group_idx=tile_idx_to_group_idx,
                        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                        num_non_exiting_tiles=num_non_exiting_tiles,
                        token_final_scales=token_final_scales,
                        num_experts=num_experts,
                        top_k=top_k,
                        num_local_experts=num_local_experts,
                        local_expert_offset=0,
                        tile_size=tile_size,
                        output_dtype=torch.bfloat16,
                    )
    finally:
        end_for_all_locality_domain()
    torch.cuda.synchronize()

    assert output[:, :half_hidden].any()
    assert output[:, half_hidden:].any()
    torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=0.15)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize("num_tokens", [128])
@pytest.mark.parametrize("top_k", [1, 2])
def test_moe_module_locality_domain_correctness_rubin(num_tokens: int, top_k: int):
    _skip_if_no_locality_domain()

    from _torch.moe.quantize_utils import get_test_quant_params
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
    from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend
    from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
    from tensorrt_llm._utils import mpi_rank
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    hidden_size = 2048
    intermediate_size = 1536
    num_experts = 256
    dtype = torch.bfloat16

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
        input_tensor = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo.NVFP4, input_tensor, "CUTEDSL"
        )
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        pretrained_config = PretrainedConfig()
        pretrained_config.num_experts = num_experts
        pretrained_config.hidden_size = hidden_size
        pretrained_config.intermediate_size = intermediate_size
        pretrained_config.torch_dtype = dtype

        def create_backend(enable_locality_domains: bool):
            model_config = ModelConfig(
                pretrained_config=pretrained_config,
                quant_config=quant_config,
                mapping=mapping,
                moe_backend="CUTEDSL",
                locality_domain_policy=LocalityDomainPolicy(enabled=enable_locality_domains),
            )
            backend = create_moe_backend(
                moe_cls=CuteDslFusedMoE,
                routing_method=routing_method,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                reduce_results=True,
                model_config=model_config,
                init_load_balancer=False,
            )
            backend.load_weights([weights])
            source_storage_ptrs = {}
            if enable_locality_domains and top_k == 2:
                source_storage_ptrs = {
                    "w3_w1_weight": backend.w3_w1_weight.untyped_storage().data_ptr(),
                    "w2_weight": backend.w2_weight.untyped_storage().data_ptr(),
                    "fc1_weight_block": backend.quant_scales.fc1_weight_block.untyped_storage().data_ptr(),
                    "fc2_weight_block": backend.quant_scales.fc2_weight_block.untyped_storage().data_ptr(),
                }
            backend.post_load_weights()
            backend.cuda()
            return backend, source_storage_ptrs

        base_backend, _ = create_backend(False)
        locality_domain_backend, source_storage_ptrs = create_backend(True)

        if top_k == 2:
            assert locality_domain_backend._locality_domain_runtime is not None
            assert locality_domain_backend._locality_domain_weight_shards is not None
            assert hasattr(locality_domain_backend, "_cached_reserved_moe_output_memset_stream")
            for shard in locality_domain_backend._locality_domain_weight_shards:
                for name, source_storage_ptr in source_storage_ptrs.items():
                    assert shard[name].untyped_storage().data_ptr() != source_storage_ptr
            for param_name in (
                "w3_w1_weight",
                "w2_weight",
                "w3_w1_weight_scale",
                "w2_weight_scale",
            ):
                assert getattr(locality_domain_backend, param_name).numel() == 0
            assert locality_domain_backend.quant_scales.fc1_weight_block.numel() == 0
            assert locality_domain_backend.quant_scales.fc2_weight_block.numel() == 0

        with torch.inference_mode():
            base_output = base_backend.forward_chunk(input_tensor, router_logits)
            locality_domain_output = locality_domain_backend.forward_chunk(
                input_tensor, router_logits
            )

        torch.cuda.synchronize()
        torch.testing.assert_close(base_output, locality_domain_output, rtol=1e-2, atol=0.15)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
def test_moe_module_bf16_locality_domain_lifecycle_and_forward_chunk_rubin():
    _skip_if_no_locality_domain()

    from _torch.moe.quantize_utils import get_test_quant_params
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
    from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend
    from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
    from tensorrt_llm._utils import mpi_rank
    from tensorrt_llm.mapping import Mapping

    hidden_size = 2048
    intermediate_size = 1536
    num_experts = 256
    num_tokens = 128
    top_k = 2
    dtype = torch.bfloat16

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
        input_tensor = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            None, input_tensor, "CUTEDSL"
        )
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        pretrained_config = PretrainedConfig()
        pretrained_config.num_experts = num_experts
        pretrained_config.hidden_size = hidden_size
        pretrained_config.intermediate_size = intermediate_size
        pretrained_config.torch_dtype = dtype

        def create_backend(enable_locality_domains: bool):
            model_config = ModelConfig(
                pretrained_config=pretrained_config,
                quant_config=quant_config,
                mapping=mapping,
                moe_backend="CUTEDSL",
                locality_domain_policy=LocalityDomainPolicy(enabled=enable_locality_domains),
            )
            backend = create_moe_backend(
                moe_cls=CuteDslFusedMoE,
                routing_method=routing_method,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                reduce_results=True,
                model_config=model_config,
                init_load_balancer=False,
            )
            backend.load_weights([weights])
            full_w3_w1 = None
            full_w2 = None
            source_storage_ptrs = ()
            if enable_locality_domains:
                full_w3_w1 = backend.w3_w1_weight.data.clone()
                full_w2 = backend.w2_weight.data.clone()
                source_storage_ptrs = (
                    backend.w3_w1_weight.untyped_storage().data_ptr(),
                    backend.w2_weight.untyped_storage().data_ptr(),
                )
            backend.post_load_weights()
            backend.cuda()
            return backend, full_w3_w1, full_w2, source_storage_ptrs

        base_backend, _, _, _ = create_backend(False)
        locality_domain_backend, full_w3_w1, full_w2, source_storage_ptrs = create_backend(True)

        assert locality_domain_backend._locality_domain_runtime is not None
        assert locality_domain_backend._locality_domain_weight_shards is not None
        assert hasattr(locality_domain_backend, "_cached_reserved_moe_output_memset_stream")

        shards = locality_domain_backend._locality_domain_weight_shards
        for shard in shards:
            assert shard["w3_w1_weight"].untyped_storage().data_ptr() != source_storage_ptrs[0]
            assert shard["w2_weight"].untyped_storage().data_ptr() != source_storage_ptrs[1]
        assert torch.equal(torch.cat([s["w3_w1_weight"] for s in shards], dim=1), full_w3_w1)
        assert torch.equal(torch.cat([s["w2_weight"] for s in shards], dim=1), full_w2)
        assert locality_domain_backend.w3_w1_weight.numel() == 0
        assert locality_domain_backend.w2_weight.numel() == 0

        # Keep the production-shape lifecycle and public forward_chunk
        # integration here. Broad accuracy, autotune, capture, and outer-tile
        # replay are covered by the unified backend matrix.
        with torch.inference_mode():
            base_output = base_backend.forward_chunk(input_tensor, router_logits)
            locality_domain_output = locality_domain_backend.forward_chunk(
                input_tensor, router_logits
            )

        torch.cuda.synchronize()
        torch.testing.assert_close(base_output, locality_domain_output, rtol=1e-2, atol=0.15)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on SM 107 (Rubin) GPUs",
)
@pytest.mark.parametrize("tile_size", [64, 128, 256])
@pytest.mark.parametrize("ep_size", [1, 8])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [128, 515, 1024])
def test_bf16_grouped_gemm_finalize_rubin(
    num_tokens: int, top_k: int, ep_size: int, tile_size: int
):
    """Test BF16 grouped GEMM with finalize fusion on Rubin (SM107).

    Uses torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_finalize_rubin.
    No scale factors or quantization — direct BF16 inputs/outputs.
    """
    hidden_size = 4096
    interm_size = 8192
    num_experts = 256
    num_local_experts = num_experts // ep_size

    # Generate routing information
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

    # Create BF16 input tensors (FC2: interm_size -> hidden_size)
    a = torch.randn(max_num_permuted_tokens, interm_size, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        num_local_experts, hidden_size, interm_size, dtype=torch.bfloat16, device="cuda"
    )
    # Compute reference: per-group GEMM + scatter-add finalize
    tile_group_list = tile_idx_to_group_idx.cpu().tolist()
    tile_mn_limit_list = tile_idx_to_mn_limit.cpu().tolist()
    permuted_idx_list = permuted_idx_to_expanded_idx.cpu().tolist()

    c_permuted = torch.zeros(
        max_num_permuted_tokens, hidden_size, dtype=torch.float32, device="cuda"
    )
    for tile_idx in range(num_non_exiting_tiles.item()):
        group_idx = tile_group_list[tile_idx]
        mn_limit = tile_mn_limit_list[tile_idx]
        start = tile_idx * tile_size
        end = min(start + tile_size, mn_limit)

        for i in range(start, end):
            a_row = a[i].float()
            gemm_row = a_row @ b[group_idx].float().T
            c_permuted[i] = gemm_row

    # Scatter-add with token_final_scales
    c_ref = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    for tile_idx in range(num_non_exiting_tiles.item()):
        mn_limit = tile_mn_limit_list[tile_idx]
        start = tile_idx * tile_size
        end = min(start + tile_size, mn_limit)
        for i in range(start, end):
            expanded_idx = permuted_idx_list[i]
            token_idx = expanded_idx // top_k
            topk_idx = expanded_idx % top_k
            scale = token_final_scales[token_idx, topk_idx].item()
            c_ref[token_idx] += (c_permuted[i] * scale).to(torch.bfloat16)

    # Even-tile padding for Rubin cluster sync
    kernel_nnet = ((num_non_exiting_tiles + 1) // 2) * 2

    # Test all valid autotuner candidate tactics via direct runner call
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        Sm107ContiguousGroupedGemmFinalizeFusionRunner,
    )

    runner = Sm107ContiguousGroupedGemmFinalizeFusionRunner(
        num_experts, top_k, num_local_experts, 0, tile_size, torch.bfloat16
    )

    tactics = runner.get_valid_tactics(
        [
            a,
            b,
            torch.zeros_like(c_ref),
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            token_final_scales,
        ],
        None,
    )
    assert len(tactics) > 0, f"No valid tactics for tile_size={tile_size}"

    failed = []
    for tactic in tactics:
        mma_tiler, _, cluster, _ = tactic
        label = f"mma={mma_tiler[:2]} cluster={cluster}"
        output = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
        inputs = [
            a,
            b,
            output,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            kernel_nnet,
            token_final_scales,
        ]
        with torch.inference_mode():
            c = runner.forward(inputs, tactic=tactic)
        match = torch.isclose(c, c_ref, rtol=1.6e-2, atol=1e-1).sum().item() / c_ref.numel()
        if match < 0.95:
            failed.append(f"{label}: match={match:.4f}")
    assert not failed, (
        f"tile_size={tile_size}: {len(failed)}/{len(tactics)} tactics failed:\n  "
        + "\n  ".join(failed)
    )
