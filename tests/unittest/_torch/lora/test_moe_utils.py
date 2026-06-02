# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for routed-expert MoE LoRA utilities.

Covers validation, shared-outer kernel-flag translation/merge, and the
synthetic-adapter tooling, without any GPU or model-engine dependencies.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.peft.lora.moe_utils import (
    check_moe_lora_supported,
    has_moe_lora_targets,
    make_per_expert_lora,
    reference_moe_lora_delta,
)
from tensorrt_llm.lora_helper import (
    MOE_MODULE_SHARED_FLAG,
    all_false_moe_shared_flags,
    merge_moe_shared_flags_for_batch,
    moe_shared_sides_to_kernel_flags,
)


class _NotCutlassMoE:
    """Stand-in resolved backend class that is not the Cutlass kernel."""


def test_moe_lora_module_names():
    assert set(MOE_MODULE_SHARED_FLAG) == {"moe_h_to_4h", "moe_gate", "moe_4h_to_h"}


# ---------- validation ----------


class _FakeLoraConfig:
    """Minimal stand-in for `LoraConfig` for tests that don't need pydantic."""

    def __init__(self, lora_target_modules):
        self.lora_target_modules = lora_target_modules


class _FakeQuantMode:
    def __init__(self, any_quant: bool):
        self._any = any_quant

    def has_any_quant(self, exclude_kv_cache: bool = False):
        return self._any


class _FakeQuantConfig:
    def __init__(self, any_quant: bool):
        self.quant_mode = _FakeQuantMode(any_quant)


def test_has_moe_lora_targets_none():
    assert has_moe_lora_targets(None) is False


def test_has_moe_lora_targets_empty():
    assert has_moe_lora_targets(_FakeLoraConfig([])) is False


def test_has_moe_lora_targets_attn_only():
    cfg = _FakeLoraConfig(["attn_q", "attn_k", "attn_v"])
    assert has_moe_lora_targets(cfg) is False


@pytest.mark.parametrize("name", sorted(MOE_MODULE_SHARED_FLAG))
def test_has_moe_lora_targets_each_module(name):
    cfg = _FakeLoraConfig(["attn_q", name])
    assert has_moe_lora_targets(cfg) is True


def test_check_no_lora_is_noop():
    # No LoRA at all; validator must not raise regardless of backend/quant.
    check_moe_lora_supported(
        moe_cls=_NotCutlassMoE,
        requested_moe_backend="WIDEEP",
        lora_config=None,
        quant_config=_FakeQuantConfig(True),
    )


def test_check_no_moe_lora_is_noop():
    # LoRA on attention only; validator must not reject any backend / quant.
    cfg = _FakeLoraConfig(["attn_q", "attn_k"])
    check_moe_lora_supported(
        moe_cls=_NotCutlassMoE,
        requested_moe_backend="TRTLLM",
        lora_config=cfg,
        quant_config=_FakeQuantConfig(True),
    )


def test_check_moe_lora_cutlass_unquantized_ok():
    cfg = _FakeLoraConfig(["moe_h_to_4h", "moe_4h_to_h", "moe_gate"])
    # No quant config -> definitely unquantized.
    check_moe_lora_supported(
        moe_cls=CutlassFusedMoE,
        requested_moe_backend="CUTLASS",
        lora_config=cfg,
        quant_config=None,
    )


def test_check_moe_lora_cutlass_unquantized_quant_mode_ok():
    # An explicit QuantConfig that reports no active quant should pass.
    cfg = _FakeLoraConfig(["moe_4h_to_h", "moe_gate"])
    check_moe_lora_supported(
        moe_cls=CutlassFusedMoE,
        requested_moe_backend="cutlass",
        lora_config=cfg,
        quant_config=_FakeQuantConfig(False),
    )


@pytest.mark.parametrize(
    "backend",
    [
        "WIDEEP",
        "TRITON",
        "DEEPGEMM",
        "VANILLA",
        "DENSEGEMM",
        "CUTEDSL",
        "TRTLLM",
        "MEGAMOE_DEEPGEMM",
    ],
)
def test_check_moe_lora_rejects_non_cutlass_backend(backend):
    # Any resolved class other than the exact CutlassFusedMoE kernel is rejected;
    # the requested backend name is surfaced in the error message.
    cfg = _FakeLoraConfig(["moe_gate", "moe_4h_to_h"])
    with pytest.raises(ValueError, match="Cutlass MoE backend"):
        check_moe_lora_supported(
            moe_cls=_NotCutlassMoE,
            requested_moe_backend=backend,
            lora_config=cfg,
            quant_config=None,
        )


def test_check_moe_lora_rejects_quantized_base():
    cfg = _FakeLoraConfig(["moe_gate", "moe_4h_to_h"])
    with pytest.raises(ValueError, match="unquantized fp16/bf16 base weights"):
        check_moe_lora_supported(
            moe_cls=CutlassFusedMoE,
            requested_moe_backend="CUTLASS",
            lora_config=cfg,
            quant_config=_FakeQuantConfig(True),
        )


def test_check_moe_lora_layer_idx_in_message():
    cfg = _FakeLoraConfig(["moe_gate", "moe_4h_to_h"])
    with pytest.raises(ValueError, match=r"\[layer_idx=7\]"):
        check_moe_lora_supported(
            moe_cls=_NotCutlassMoE,
            requested_moe_backend="TRTLLM",
            lora_config=cfg,
            quant_config=None,
            layer_idx=7,
        )


# ---------- moe_shared_sides_to_kernel_flags ----------


def test_all_false_moe_shared_flags_has_three_canonical_keys():
    flags = all_false_moe_shared_flags()
    assert set(flags) == {"fc1_shared_a", "gated_shared_a", "fc2_shared_b"}
    assert not any(flags.values())


def test_moe_shared_sides_to_kernel_flags_empty():
    assert moe_shared_sides_to_kernel_flags({}) == all_false_moe_shared_flags()


def test_moe_shared_sides_to_kernel_flags_canonical():
    shared_sides = {
        "moe_h_to_4h": (True, False),
        "moe_gate": (True, False),
        "moe_4h_to_h": (False, True),
    }
    assert moe_shared_sides_to_kernel_flags(shared_sides) == {
        "fc1_shared_a": True,
        "gated_shared_a": True,
        "fc2_shared_b": True,
    }


def test_moe_shared_sides_to_kernel_flags_ignores_unknown_module():
    assert (
        moe_shared_sides_to_kernel_flags({"attn_q": (True, True)}) == all_false_moe_shared_flags()
    )


def test_moe_shared_sides_to_kernel_flags_ignores_non_canonical_side():
    # moe_gate shares its residual side via A; a detected B-share is ignored.
    assert (
        moe_shared_sides_to_kernel_flags({"moe_gate": (False, True)})
        == all_false_moe_shared_flags()
    )


# ---------- merge_moe_shared_flags_for_batch ----------


def test_merge_returns_none_for_no_uids():
    assert merge_moe_shared_flags_for_batch([], lambda uid: all_false_moe_shared_flags()) is None


def test_merge_returns_none_when_all_flags_false():
    flags = all_false_moe_shared_flags()
    assert merge_moe_shared_flags_for_batch(["a"], lambda uid: flags) is None


def test_merge_single_uid_with_shared_flags():
    expected = all_false_moe_shared_flags()
    expected["gated_shared_a"] = True
    assert merge_moe_shared_flags_for_batch(["uid-1"], lambda uid: expected) == expected


def test_merge_multiple_uids_matching_flags():
    expected = all_false_moe_shared_flags()
    expected["fc2_shared_b"] = True
    assert merge_moe_shared_flags_for_batch(["a", "b"], lambda uid: expected) == expected


def test_merge_intersects_when_one_uid_shares_and_another_does_not():
    # uid 'a' shares fc2_shared_b, uid 'b' shares nothing: the batch falls back
    # to the per-expert read everywhere (intersection is all-False -> None).
    flags_a = all_false_moe_shared_flags()
    flags_a["fc2_shared_b"] = True
    flags_b = all_false_moe_shared_flags()
    result = merge_moe_shared_flags_for_batch(
        ["a", "b"],
        lambda uid: flags_a if uid == "a" else flags_b,
    )
    assert result is None


def test_merge_drops_sides_not_shared_by_all():
    # 'a' shares fc1_shared_a + fc2_shared_b; 'b' shares only fc1_shared_a.
    # Only the common side (fc1_shared_a) survives the intersection.
    flags_a = all_false_moe_shared_flags()
    flags_a["fc1_shared_a"] = True
    flags_a["fc2_shared_b"] = True
    flags_b = all_false_moe_shared_flags()
    flags_b["fc1_shared_a"] = True
    expected = all_false_moe_shared_flags()
    expected["fc1_shared_a"] = True
    result = merge_moe_shared_flags_for_batch(
        ["a", "b"],
        lambda uid: flags_a if uid == "a" else flags_b,
    )
    assert result == expected


# ---------- synthetic adapter tooling ----------


def test_make_per_expert_lora_shapes_per_expert():
    adapter = make_per_expert_lora(
        num_experts=4, rank=8, in_dim=64, out_dim=128, shared_side=None, seed=0
    )
    assert adapter["A"].shape == (4, 8, 64)
    assert adapter["B"].shape == (4, 128, 8)


def test_make_per_expert_lora_shared_a_replicates():
    adapter = make_per_expert_lora(
        num_experts=3, rank=4, in_dim=32, out_dim=16, shared_side="A", seed=1
    )
    a, b = adapter["A"], adapter["B"]
    assert a.shape == (3, 4, 32)
    assert b.shape == (3, 16, 4)
    # All expert slices of A must be identical (shared).
    for e in range(1, 3):
        torch.testing.assert_close(a[e], a[0])
    # B slices must NOT all be identical (per-expert noise).
    assert not torch.allclose(b[0], b[1])


def test_make_per_expert_lora_shared_b_replicates():
    adapter = make_per_expert_lora(
        num_experts=2, rank=6, in_dim=16, out_dim=8, shared_side="B", seed=2
    )
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

    adapter = make_per_expert_lora(
        num_experts=num_experts,
        rank=rank,
        in_dim=in_dim,
        out_dim=out_dim,
        dtype=torch.float32,
        shared_side=None,
        seed=7,
    )
    x = torch.randn(num_tokens, in_dim, dtype=torch.float32)
    token_to_expert = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)

    delta = reference_moe_lora_delta(adapter["A"], adapter["B"], x, token_to_expert)
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

    adapter = make_per_expert_lora(
        num_experts=num_experts,
        rank=rank,
        in_dim=in_dim,
        out_dim=out_dim,
        dtype=torch.float32,
        shared_side="B",
        seed=11,
    )
    x = torch.randn(3, in_dim, dtype=torch.float32)
    token_to_expert = torch.tensor([0, 1, 0], dtype=torch.int32)

    delta = reference_moe_lora_delta(adapter["A"], adapter["B"], x, token_to_expert, scale=2.0)
    assert delta.shape == (3, out_dim)
    # The B used for all tokens is the same matrix (shared); we just check the
    # scale propagated.
    expected_t0 = 2.0 * (adapter["B"][0] @ (adapter["A"][0] @ x[0]))
    torch.testing.assert_close(delta[0], expected_t0)
