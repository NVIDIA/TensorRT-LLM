# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MoE LoRA layout helpers.

These tests are CPU-only and exercise the synthetic-adapter generator that the
end-to-end multi-LoRA tests will use.
"""

import pytest
import torch

from tensorrt_llm._torch.peft.lora.moe_layout import (
    DEFAULT_SHARED_SIDE, MOE_LORA_MODULES, make_per_expert_lora,
    reference_moe_lora_delta)


def test_module_list_complete():
    # Sanity check: the canonical module names match the validator's set.
    assert set(MOE_LORA_MODULES) == {"moe_h_to_4h", "moe_4h_to_h", "moe_gate"}


def test_default_shared_side_assignment():
    # Up-projections share A (residual-stream side); down-projection shares B.
    assert DEFAULT_SHARED_SIDE["moe_h_to_4h"] == "A"
    assert DEFAULT_SHARED_SIDE["moe_gate"] == "A"
    assert DEFAULT_SHARED_SIDE["moe_4h_to_h"] == "B"


def test_make_per_expert_lora_shapes_per_expert():
    adapter = make_per_expert_lora(num_experts=4,
                                   rank=8,
                                   in_dim=64,
                                   out_dim=128,
                                   shared_side=None,
                                   seed=0)
    assert adapter["A"].shape == (4, 8, 64)
    assert adapter["B"].shape == (4, 128, 8)


def test_make_per_expert_lora_shared_a_replicates():
    adapter = make_per_expert_lora(num_experts=3,
                                   rank=4,
                                   in_dim=32,
                                   out_dim=16,
                                   shared_side="A",
                                   seed=1)
    a, b = adapter["A"], adapter["B"]
    assert a.shape == (3, 4, 32)
    assert b.shape == (3, 16, 4)
    # All expert slices of A must be identical (shared).
    for e in range(1, 3):
        torch.testing.assert_close(a[e], a[0])
    # B slices must NOT all be identical (per-expert noise).
    assert not torch.allclose(b[0], b[1])


def test_make_per_expert_lora_shared_b_replicates():
    adapter = make_per_expert_lora(num_experts=2,
                                   rank=6,
                                   in_dim=16,
                                   out_dim=8,
                                   shared_side="B",
                                   seed=2)
    a, b = adapter["A"], adapter["B"]
    assert a.shape == (2, 6, 16)
    assert b.shape == (2, 8, 6)
    torch.testing.assert_close(b[0], b[1])
    assert not torch.allclose(a[0], a[1])


def test_make_per_expert_lora_seed_reproducible():
    a1 = make_per_expert_lora(2, 4, 8, 8, seed=42)
    a2 = make_per_expert_lora(2, 4, 8, 8, seed=42)
    torch.testing.assert_close(a1["A"], a2["A"])
    torch.testing.assert_close(a1["B"], a2["B"])


def test_make_per_expert_lora_invalid_shared_side():
    with pytest.raises(ValueError, match="shared_side"):
        make_per_expert_lora(2, 4, 8, 8, shared_side="C")


def test_reference_moe_lora_delta_per_expert():
    torch.manual_seed(0)
    num_experts = 3
    rank = 4
    in_dim = 16
    out_dim = 8
    num_tokens = 5

    adapter = make_per_expert_lora(num_experts=num_experts,
                                   rank=rank,
                                   in_dim=in_dim,
                                   out_dim=out_dim,
                                   dtype=torch.float32,
                                   shared_side=None,
                                   seed=7)
    x = torch.randn(num_tokens, in_dim, dtype=torch.float32)
    token_to_expert = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)

    delta = reference_moe_lora_delta(adapter["A"], adapter["B"], x,
                                     token_to_expert)
    assert delta.shape == (num_tokens, out_dim)
    # Spot-check token 0: should equal B[0] @ A[0] @ x[0].
    expected_t0 = adapter["B"][0] @ (adapter["A"][0] @ x[0])
    torch.testing.assert_close(delta[0], expected_t0)


def test_reference_moe_lora_delta_shared_outer_matches_per_expert_when_b_shared():
    """When B is shared, every expert produces the same delta direction modulo
    the per-expert A matrix. The reference helper must still pick the right
    expert per token."""
    torch.manual_seed(0)
    num_experts = 2
    rank = 4
    in_dim = 8
    out_dim = 6

    adapter = make_per_expert_lora(num_experts=num_experts,
                                   rank=rank,
                                   in_dim=in_dim,
                                   out_dim=out_dim,
                                   dtype=torch.float32,
                                   shared_side="B",
                                   seed=11)
    x = torch.randn(3, in_dim, dtype=torch.float32)
    token_to_expert = torch.tensor([0, 1, 0], dtype=torch.int32)

    delta = reference_moe_lora_delta(adapter["A"], adapter["B"], x,
                                     token_to_expert, scale=2.0)
    assert delta.shape == (3, out_dim)
    # The B used for all tokens is the same matrix (shared); we just check the
    # scale propagated.
    expected_t0 = 2.0 * (adapter["B"][0] @ (adapter["A"][0] @ x[0]))
    torch.testing.assert_close(delta[0], expected_t0)
