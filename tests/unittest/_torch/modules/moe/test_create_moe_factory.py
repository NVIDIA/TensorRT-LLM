# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``get_moe_cls`` factory branches, with focus on
``MEGAMOE_AUTO``: threshold-based selection between MegaMoEDeepGemm and
TRTLLMGenFusedMoE plus capability-gate fallback to CutlassFusedMoE.
"""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.create_moe import get_moe_cls
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.modules.fused_moe.mega_moe import MegaMoEDeepGemm
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _make_model_config(
    *,
    moe_backend: str,
    max_num_tokens: int,
    megamoe_max_tokens_per_rank: int = 1024,
    quant_algo: Optional[QuantAlgo] = QuantAlgo.W4A8_MXFP4_MXFP8,
):
    """Lightweight ModelConfig stand-in. ``get_moe_cls`` only reads a small
    subset of attributes; SimpleNamespace avoids the full ModelConfig dataclass
    contract while still exercising the factory."""
    pretrained = SimpleNamespace(
        hidden_size=7168,
        intermediate_size=3072,
        moe_intermediate_size=3072,
        torch_dtype=torch.bfloat16,
    )
    quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo is not None else QuantConfig()
    return SimpleNamespace(
        moe_backend=moe_backend,
        max_num_tokens=max_num_tokens,
        megamoe_max_tokens_per_rank=megamoe_max_tokens_per_rank,
        quant_config=quant_config,
        pretrained_config=pretrained,
    )


def _patch_mega_capable(monkeypatch, ok: bool, reason: str = "skipped"):
    monkeypatch.setattr(
        MegaMoEDeepGemm,
        "can_implement",
        classmethod(lambda cls, *a, **kw: (ok, None if ok else reason)),
    )


# ---------------------------------------------------------------------------
# MEGAMOE_AUTO — threshold-driven branch selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_num_tokens, expected_cls",
    [
        (256, MegaMoEDeepGemm),  # well below threshold
        (1024, MegaMoEDeepGemm),  # threshold boundary (inclusive)
        (1025, TRTLLMGenFusedMoE),  # just above threshold
        (8192, TRTLLMGenFusedMoE),  # typical CTX prefill shape
    ],
)
def test_megamoe_auto_threshold(monkeypatch, max_num_tokens, expected_cls):
    _patch_mega_capable(monkeypatch, ok=True)
    cfg = _make_model_config(
        moe_backend="MEGAMOE_AUTO",
        max_num_tokens=max_num_tokens,
    )
    assert get_moe_cls(cfg) is expected_cls


@pytest.mark.parametrize("max_num_tokens", [256, 2048, 4096])
def test_megamoe_auto_custom_threshold(monkeypatch, max_num_tokens):
    """User-provided ``megamoe_max_tokens_per_rank`` should override the 1024
    default. With threshold 4096, all three shapes prefer MegaMoE."""
    _patch_mega_capable(monkeypatch, ok=True)
    cfg = _make_model_config(
        moe_backend="MEGAMOE_AUTO",
        max_num_tokens=max_num_tokens,
        megamoe_max_tokens_per_rank=4096,
    )
    assert get_moe_cls(cfg) is MegaMoEDeepGemm


# ---------------------------------------------------------------------------
# MEGAMOE_AUTO — capability-gate fallback chains
# ---------------------------------------------------------------------------


def test_megamoe_auto_falls_through_when_mega_gate_fails(monkeypatch):
    """Below-threshold path: if MegaMoE rejects (e.g. wrong SM), fall through
    to TRTLLMGen rather than silently returning MegaMoE."""
    _patch_mega_capable(monkeypatch, ok=False, reason="non-SM100")
    cfg = _make_model_config(moe_backend="MEGAMOE_AUTO", max_num_tokens=512)
    assert get_moe_cls(cfg) is TRTLLMGenFusedMoE


def test_megamoe_auto_cutlass_when_both_gates_fail(monkeypatch):
    """If both MegaMoE and TRTLLMGen reject, final fallback is Cutlass.
    MegaMoE rejected via patched ``can_implement``; TRTLLMGen rejected via
    a quant_config it does not support."""
    _patch_mega_capable(monkeypatch, ok=False, reason="non-SM100")
    cfg = _make_model_config(
        moe_backend="MEGAMOE_AUTO",
        max_num_tokens=512,
        quant_algo=None,  # leaves QuantConfig() default; no FP8/NVFP4 mode
    )
    assert get_moe_cls(cfg) is CutlassFusedMoE


def test_megamoe_auto_above_threshold_uses_trtllm_when_quant_supported(monkeypatch):
    """Above-threshold path picks TRTLLMGen directly without consulting the
    MegaMoE gate."""
    # Stub MegaMoE.can_implement to True so we know the result is not from
    # accidentally taking the below-threshold branch.
    _patch_mega_capable(monkeypatch, ok=True)
    cfg = _make_model_config(moe_backend="MEGAMOE_AUTO", max_num_tokens=8192)
    assert get_moe_cls(cfg) is TRTLLMGenFusedMoE


def test_megamoe_auto_above_threshold_falls_back_to_cutlass_when_quant_unsupported(monkeypatch):
    """Above-threshold + TRTLLMGen-unsupported quant -> Cutlass."""
    _patch_mega_capable(monkeypatch, ok=True)
    cfg = _make_model_config(
        moe_backend="MEGAMOE_AUTO",
        max_num_tokens=8192,
        quant_algo=None,
    )
    assert get_moe_cls(cfg) is CutlassFusedMoE


# ---------------------------------------------------------------------------
# MEGAMOE_AUTO — boundary: missing / zero max_num_tokens
# ---------------------------------------------------------------------------


def test_megamoe_auto_zero_max_num_tokens_picks_trtllm(monkeypatch):
    """``max_num_tokens=0`` (unset / placeholder) treats the workload as
    'too big to know', falling back to TRTLLMGen rather than MegaMoE."""
    _patch_mega_capable(monkeypatch, ok=True)
    cfg = _make_model_config(moe_backend="MEGAMOE_AUTO", max_num_tokens=0)
    assert get_moe_cls(cfg) is TRTLLMGenFusedMoE
