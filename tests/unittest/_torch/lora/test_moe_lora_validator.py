# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the routed-expert MoE LoRA validator."""

import pytest

from tensorrt_llm._torch.peft.lora.validation import (
    MOE_LORA_MODULE_NAMES,
    check_moe_lora_supported,
    has_moe_lora_targets,
)
from tensorrt_llm.quantization.mode import QuantMode


class _FakeLoraConfig:
    """Minimal stand-in for `LoraConfig` for tests that don't need pydantic."""

    def __init__(self, lora_target_modules):
        self.lora_target_modules = lora_target_modules


class _FakeQuantConfig:
    """Minimal stand-in for `QuantConfig` carrying a real `QuantMode`."""

    def __init__(self, quant_mode: QuantMode):
        self.quant_mode = quant_mode


_NO_QUANT = QuantMode(0)
_FP8_QDQ = QuantMode.from_description(use_fp8_qdq=True)
_FP8_BLOCK_SCALE = QuantMode.from_description(use_fp8_block_scales=True)
_NVFP4 = QuantMode.from_description(use_nvfp4=True)


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
        quant_config=_FakeQuantConfig(_FP8_BLOCK_SCALE),
    )


def test_check_no_moe_lora_is_noop():
    # LoRA on attention only; validator must not reject any backend / quant.
    cfg = _FakeLoraConfig(["attn_q", "attn_k"])
    check_moe_lora_supported(
        moe_backend_name="TRTLLM",
        lora_config=cfg,
        quant_config=_FakeQuantConfig(_FP8_BLOCK_SCALE),
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
        quant_config=_FakeQuantConfig(_NO_QUANT),
    )


def test_check_moe_lora_cutlass_fp8_qdq_ok():
    # Per-tensor FP8 (qdq) base weights are supported on Cutlass.
    cfg = _FakeLoraConfig(["moe_h_to_4h", "moe_4h_to_h", "moe_gate"])
    check_moe_lora_supported(
        moe_backend_name="CUTLASS",
        lora_config=cfg,
        quant_config=_FakeQuantConfig(_FP8_QDQ),
    )


def test_check_moe_lora_rejects_fp8_block_scale():
    # Only per-tensor FP8 (qdq) is supported; FP8 block-scale has no LoRA path
    # and must be rejected.
    cfg = _FakeLoraConfig(["moe_h_to_4h", "moe_4h_to_h", "moe_gate"])
    with pytest.raises(ValueError, match="FP8 block-scale"):
        check_moe_lora_supported(
            moe_backend_name="CUTLASS",
            lora_config=cfg,
            quant_config=_FakeQuantConfig(_FP8_BLOCK_SCALE),
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


def test_check_moe_lora_rejects_nvfp4_base():
    # FP4 (and other non per-tensor-FP8 quant) has no LoRA path and must be
    # rejected.
    cfg = _FakeLoraConfig(["moe_gate", "moe_4h_to_h"])
    with pytest.raises(ValueError, match="per-tensor FP8"):
        check_moe_lora_supported(
            moe_backend_name="CUTLASS",
            lora_config=cfg,
            quant_config=_FakeQuantConfig(_NVFP4),
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
