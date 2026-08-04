# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MoE LoRA layout helpers.

These tests are CPU-only and exercise the synthetic-adapter generator that the
end-to-end multi-LoRA tests will use.
"""

import torch

from tensorrt_llm._torch.peft.lora.moe_layout import (
    MOE_LORA_MODULES,
    make_per_expert_lora,
    reference_moe_lora_delta,
)


def test_module_list_complete():
    # Sanity check: the canonical module names match the validator's set.
    assert set(MOE_LORA_MODULES) == {"moe_h_to_4h", "moe_4h_to_h", "moe_gate"}


def test_make_per_expert_lora_shapes_per_expert():
    adapter = make_per_expert_lora(num_experts=4, rank=8, in_dim=64, out_dim=128, seed=0)
    assert adapter["A"].shape == (4, 8, 64)
    assert adapter["B"].shape == (4, 128, 8)
    # Expert slices must be independent (per-expert noise).
    assert not torch.allclose(adapter["A"][0], adapter["A"][1])
    assert not torch.allclose(adapter["B"][0], adapter["B"][1])


def test_make_per_expert_lora_seed_reproducible():
    a1 = make_per_expert_lora(2, 4, 8, 8, seed=42)
    a2 = make_per_expert_lora(2, 4, 8, 8, seed=42)
    torch.testing.assert_close(a1["A"], a2["A"])
    torch.testing.assert_close(a1["B"], a2["B"])


def test_reference_moe_lora_delta_per_expert():
    torch.manual_seed(0)
    num_experts = 3
    rank = 4
    in_dim = 16
    out_dim = 8
    num_tokens = 5

    adapter = make_per_expert_lora(
        num_experts=num_experts,
        rank=rank,
        in_dim=in_dim,
        out_dim=out_dim,
        dtype=torch.float32,
        seed=7,
    )
    x = torch.randn(num_tokens, in_dim, dtype=torch.float32)
    token_to_expert = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)

    delta = reference_moe_lora_delta(adapter["A"], adapter["B"], x, token_to_expert)
    assert delta.shape == (num_tokens, out_dim)
    # Spot-check token 0: should equal B[0] @ A[0] @ x[0].
    expected_t0 = adapter["B"][0] @ (adapter["A"][0] @ x[0])
    torch.testing.assert_close(delta[0], expected_t0)

    # Scale factor propagates linearly.
    delta_scaled = reference_moe_lora_delta(
        adapter["A"], adapter["B"], x, token_to_expert, scale=2.0
    )
    torch.testing.assert_close(delta_scaled[0], 2.0 * expected_t0)
