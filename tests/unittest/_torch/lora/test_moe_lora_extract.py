# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for CutlassFusedMoE._extract_moe_lora_tensors.

This function maps the per-layer lora_params dict (keyed by LoraModuleType)
onto the fc1, gated, and fc2 kwargs the fused MoE op expects. The op-level
tests pass fc1 and gated tensors directly and do not exercise this mapping, so
it is tested here in isolation.

The mapping is correctness-critical: SwiGLU is asymmetric (SiLU gate side vs
linear up side), so swapping moe_h_to_4h and moe_gate between the fc1 and gated
slots silently produces wrong outputs. The canonical convention (moe_h_to_4h is
w1 gate/SiLU, moe_gate is w3 up/linear) and the kernel both require:
moe_h_to_4h to fc1, moe_gate to gated, moe_4h_to_h to fc2.
"""

import pytest
import torch

# These imports are pure-Python; skip cleanly if the package layout changes.
fused_moe_cutlass = pytest.importorskip("tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass")
lora_layer = pytest.importorskip("tensorrt_llm._torch.peft.lora.layer")

CutlassFusedMoE = fused_moe_cutlass.CutlassFusedMoE
LoraModuleType = lora_layer.LoraModuleType


class _ExtractStub:
    """Minimal stand-in for a CutlassFusedMoE instance.

    _extract_moe_lora_tensors only reads self.layer_idx and the class attribute
    self._MOE_LORA_MODULE_NAMES, so we can drive it via the unbound method
    without constructing real weights or a GPU layer.
    """

    _MOE_LORA_MODULE_NAMES = CutlassFusedMoE._MOE_LORA_MODULE_NAMES

    def __init__(self, layer_idx=0):
        self.layer_idx = layer_idx


def _extract(layer_idx, lora_params):
    return CutlassFusedMoE._extract_moe_lora_tensors(_ExtractStub(layer_idx), lora_params)


def _module_entry(rank: int, a_ptr: int, b_ptr: int, num_seqs: int = 1):
    """Build a single module's lora_params entry with distinctive values."""
    return {
        "adapter_size": torch.full((num_seqs,), rank, dtype=torch.int32),
        "weight_pointers": torch.tensor([[a_ptr, b_ptr, 0]] * num_seqs, dtype=torch.int64),
    }


def _make_lora_params(layer_idx, modules, num_seqs=1):
    """Assemble a lora_params dict mirroring the runtime layout.

    modules maps a LoraModuleType to (rank, a_ptr, b_ptr).
    """
    layer_dict = {
        int(mod): _module_entry(rank, a_ptr, b_ptr, num_seqs)
        for mod, (rank, a_ptr, b_ptr) in modules.items()
    }
    return {
        "num_seqs": num_seqs,
        "host_request_types": torch.zeros(num_seqs, dtype=torch.int32),
        "prompt_lens_cpu": torch.ones(num_seqs, dtype=torch.int32),
        layer_idx: layer_dict,
    }


# ---------------------------------------------------------------------------
# Routing correctness (the swap regression).
# ---------------------------------------------------------------------------


def test_extract_routes_modules_to_correct_kernel_slots():
    """moe_h_to_4h -> fc1, moe_gate -> gated, moe_4h_to_h -> fc2.

    Each module is given a distinctive (rank, A_ptr, B_ptr) triple so a swap
    between any two slots is caught.
    """
    layer_idx = 0
    modules = {
        # (rank, A_ptr, B_ptr)
        LoraModuleType.MOE_H_TO_4H: (4, 0x1110, 0x1111),  # -> fc1
        LoraModuleType.MOE_GATE: (8, 0x3330, 0x3331),  # -> gated
        LoraModuleType.MOE_4H_TO_H: (16, 0x2220, 0x2221),  # -> fc2
    }
    out = _extract(layer_idx, _make_lora_params(layer_idx, modules))
    assert out is not None

    # fc1 must carry moe_h_to_4h (gate / SiLU side).
    assert int(out["fc1_lora_ranks"][0]) == 4
    assert out["fc1_lora_weight_ptrs"][0].tolist() == [0x1110, 0x1111, 0]

    # gated must carry moe_gate (up / linear side).
    assert int(out["gated_lora_ranks"][0]) == 8
    assert out["gated_lora_weight_ptrs"][0].tolist() == [0x3330, 0x3331, 0]

    # fc2 must carry moe_4h_to_h (down projection).
    assert int(out["fc2_lora_ranks"][0]) == 16
    assert out["fc2_lora_weight_ptrs"][0].tolist() == [0x2220, 0x2221, 0]

    # lora_max_low_rank is the max rank across the active modules.
    assert out["lora_max_low_rank"] == 16


def test_extract_does_not_swap_fc1_and_gated():
    """Guard that moe_h_to_4h does not land in the gated slot and moe_gate does
    not land in the fc1 slot."""
    layer_idx = 2
    modules = {
        LoraModuleType.MOE_H_TO_4H: (5, 0xAAA0, 0xAAA1),
        LoraModuleType.MOE_GATE: (6, 0xBBB0, 0xBBB1),
        LoraModuleType.MOE_4H_TO_H: (7, 0xCCC0, 0xCCC1),
    }
    out = _extract(layer_idx, _make_lora_params(layer_idx, modules))
    # A swap would route moe_h_to_4h pointers into the gated slot.
    assert out["gated_lora_weight_ptrs"][0].tolist() != [0xAAA0, 0xAAA1, 0]
    assert out["fc1_lora_weight_ptrs"][0].tolist() != [0xBBB0, 0xBBB1, 0]


# ---------------------------------------------------------------------------
# None / no-op cases.
# ---------------------------------------------------------------------------


def test_extract_none_params_returns_none():
    assert _extract(0, None) is None
    assert _extract(0, {}) is None


def test_extract_no_layer_entry_returns_none():
    # lora_params exists but has nothing for this layer_idx.
    params = _make_lora_params(0, {LoraModuleType.MOE_H_TO_4H: (4, 0x10, 0x11)})
    assert _extract(layer_idx=99, lora_params=params) is None


def test_extract_attention_only_returns_none():
    # A layer entry that only has (hypothetical) attention modules -> no MoE.
    layer_idx = 0
    params = {
        "num_seqs": 1,
        "host_request_types": torch.zeros(1, dtype=torch.int32),
        "prompt_lens_cpu": torch.ones(1, dtype=torch.int32),
        layer_idx: {
            int(LoraModuleType.ATTENTION_Q): _module_entry(4, 0x10, 0x11),
        },
    }
    assert _extract(layer_idx, params) is None


# ---------------------------------------------------------------------------
# Mandatory-module enforcement.
# ---------------------------------------------------------------------------


def test_extract_requires_fc1_module():
    """fc1 slot (moe_h_to_4h) is mandatory: providing only gated + fc2 raises."""
    layer_idx = 0
    modules = {
        LoraModuleType.MOE_GATE: (8, 0x30, 0x31),
        LoraModuleType.MOE_4H_TO_H: (16, 0x20, 0x21),
    }
    with pytest.raises(ValueError, match="moe_h_to_4h"):
        _extract(layer_idx, _make_lora_params(layer_idx, modules))


def test_extract_requires_fc2_module():
    """fc2 slot (moe_4h_to_h) is mandatory: providing only fc1 + gated raises."""
    layer_idx = 0
    modules = {
        LoraModuleType.MOE_H_TO_4H: (4, 0x10, 0x11),
        LoraModuleType.MOE_GATE: (8, 0x30, 0x31),
    }
    with pytest.raises(ValueError, match="moe_4h_to_h"):
        _extract(layer_idx, _make_lora_params(layer_idx, modules))


def test_extract_gated_optional_when_fc1_and_fc2_present():
    """Without moe_gate, gated_* must be None but fc1/fc2 still populated."""
    layer_idx = 0
    modules = {
        LoraModuleType.MOE_H_TO_4H: (4, 0x10, 0x11),
        LoraModuleType.MOE_4H_TO_H: (16, 0x20, 0x21),
    }
    out = _extract(layer_idx, _make_lora_params(layer_idx, modules))
    assert out is not None
    assert out["gated_lora_ranks"] is None
    assert out["gated_lora_weight_ptrs"] is None
    assert int(out["fc1_lora_ranks"][0]) == 4
    assert int(out["fc2_lora_ranks"][0]) == 16
