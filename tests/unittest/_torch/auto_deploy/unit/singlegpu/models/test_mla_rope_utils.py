# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.models.custom.mla_rope_utils import (
    _rope_deinterleave_load_hook,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_rope_deinterleave_load_hook_reorders_q_and_kv_weights(dtype):
    num_heads = 2
    qk_nope_head_dim = 2
    qk_rope_head_dim = 4
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q_lora_rank = 3
    kv_lora_rank = 2

    q_weight = torch.arange(num_heads * qk_head_dim * q_lora_rank, dtype=torch.float32).reshape(
        num_heads * qk_head_dim, q_lora_rank
    )
    kv_weight = torch.arange(
        (kv_lora_rank + qk_rope_head_dim) * q_lora_rank, dtype=torch.float32
    ).reshape(kv_lora_rank + qk_rope_head_dim, q_lora_rank)
    kv_bias = torch.arange(kv_lora_rank + qk_rope_head_dim, dtype=torch.float32)

    state_dict = {
        "model.layers.0.self_attn.q_b_proj.weight": q_weight.to(dtype),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": kv_weight.to(dtype),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.bias": kv_bias.to(dtype),
    }

    _rope_deinterleave_load_hook(
        state_dict,
        prefix="",
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        num_layers=1,
    )

    perm = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    q_weight_expected = q_weight.to(dtype).float().view(num_heads, qk_head_dim, q_lora_rank)
    q_weight_expected = torch.cat(
        [
            q_weight_expected[:, :qk_nope_head_dim, :],
            q_weight_expected[:, qk_nope_head_dim:, :].index_select(1, perm),
        ],
        dim=1,
    ).reshape(-1, q_lora_rank)
    kv_weight_expected = torch.cat(
        [
            kv_weight.to(dtype).float()[:kv_lora_rank, :],
            kv_weight.to(dtype).float()[kv_lora_rank:, :].index_select(0, perm),
        ],
        dim=0,
    )
    kv_bias_expected = torch.cat(
        [
            kv_bias.to(dtype).float()[:kv_lora_rank],
            kv_bias.to(dtype).float()[kv_lora_rank:].index_select(0, perm),
        ]
    )

    torch.testing.assert_close(
        state_dict["model.layers.0.self_attn.q_b_proj.weight"].float(), q_weight_expected
    )
    torch.testing.assert_close(
        state_dict["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"].float(),
        kv_weight_expected,
    )
    torch.testing.assert_close(
        state_dict["model.layers.0.self_attn.kv_a_proj_with_mqa.bias"].float(),
        kv_bias_expected,
    )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="requires float8 support")
def test_rope_deinterleave_load_hook_preserves_fp8_bytes():
    source = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32).to(torch.float8_e4m3fn)
    state_dict = {"model.layers.0.self_attn.kv_a_proj_with_mqa.bias": source.clone()}

    _rope_deinterleave_load_hook(
        state_dict,
        prefix="",
        qk_rope_head_dim=4,
        qk_nope_head_dim=0,
        num_heads=1,
        kv_lora_rank=0,
        num_layers=1,
    )

    expected = (
        source.view(torch.uint8).index_select(0, torch.tensor([0, 2, 1, 3])).view(source.dtype)
    )
    assert torch.equal(
        state_dict["model.layers.0.self_attn.kv_a_proj_with_mqa.bias"].view(torch.uint8),
        expected.view(torch.uint8),
    )
