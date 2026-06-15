# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the routed-expert MoE LoRA validator."""

import pytest

from tensorrt_llm._torch.peft.lora.validation import (
    MOE_LORA_MODULE_NAMES,
    check_moe_lora_supported,
    has_moe_lora_targets,
)


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


@pytest.mark.parametrize("name", sorted(MOE_LORA_MODULE_NAMES))
def test_has_moe_lora_targets_each_module(name):
    cfg = _FakeLoraConfig(["attn_q", name])
    assert has_moe_lora_targets(cfg) is True


def test_check_no_lora_is_noop():
    # No LoRA at all; validator must not raise regardless of backend/quant.
    check_moe_lora_supported(
        moe_backend_name="WIDEEP",
        lora_config=None,
        quant_config=_FakeQuantConfig(True),
    )


def test_check_no_moe_lora_is_noop():
    # LoRA on attention only; validator must not reject any backend / quant.
    cfg = _FakeLoraConfig(["attn_q", "attn_k"])
    check_moe_lora_supported(
        moe_backend_name="TRTLLM",
        lora_config=cfg,
        quant_config=_FakeQuantConfig(True),
    )


def test_check_moe_lora_cutlass_unquantized_ok():
    cfg = _FakeLoraConfig(["moe_h_to_4h", "moe_4h_to_h", "moe_gate"])
    # No quant config -> definitely unquantized.
    check_moe_lora_supported(
        moe_backend_name="CUTLASS",
        lora_config=cfg,
        quant_config=None,
    )


def test_check_moe_lora_cutlass_unquantized_quant_mode_ok():
    # An explicit QuantConfig that reports no active quant should pass.
    cfg = _FakeLoraConfig(["moe_4h_to_h", "moe_gate"])
    check_moe_lora_supported(
        moe_backend_name="cutlass",  # case-insensitive
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
    cfg = _FakeLoraConfig(["moe_gate", "moe_4h_to_h"])
    with pytest.raises(ValueError, match="moe_backend='CUTLASS'"):
        check_moe_lora_supported(
            moe_backend_name=backend,
            lora_config=cfg,
            quant_config=None,
        )


def test_check_moe_lora_rejects_quantized_base():
    cfg = _FakeLoraConfig(["moe_gate", "moe_4h_to_h"])
    with pytest.raises(ValueError, match="unquantized fp16/bf16 base weights"):
        check_moe_lora_supported(
            moe_backend_name="CUTLASS",
            lora_config=cfg,
            quant_config=_FakeQuantConfig(True),
        )


def test_check_moe_lora_layer_idx_in_message():
    cfg = _FakeLoraConfig(["moe_gate", "moe_4h_to_h"])
    with pytest.raises(ValueError, match=r"\[layer_idx=7\]"):
        check_moe_lora_supported(
            moe_backend_name="TRTLLM",
            lora_config=cfg,
            quant_config=None,
            layer_idx=7,
        )
