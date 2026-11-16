# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Isolated MXFP8-QK/FP8-V attention fidelity and LTX-2.5 modality scope."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.rope import LTXRopeType
from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import TransformerConfig
from tensorrt_llm._torch.visual_gen.models.ltx2_5.transformer_ltx2_5 import LTX25TransformerBlock
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("sequence_length", [40, 128, 704])
def test_quantized_attention_matches_dense_backend_with_calibrated_error(
    sequence_length, monkeypatch
):
    """T2: same real video-self-attention shapes/QKV, dense FlashInfer reference."""
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Attention fidelity thresholds are calibrated only on B200 (SM100)")
    torch.manual_seed(42)
    dense = FlashInferAttention(num_heads=32, head_dim=128, dtype=torch.bfloat16)
    quant = FlashInferAttention(
        num_heads=32,
        head_dim=128,
        dtype=torch.bfloat16,
        quant_attention_config=QuantAttentionConfig(qk_dtype="mxfp8", v_dtype="fp8"),
    )
    calls = []
    original = quant._run_blockscaled_sm10x

    def active(*args, **kwargs):
        calls.append(kwargs["qk_mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(quant, "_run_blockscaled_sm10x", active)
    q, k, v = [
        torch.randn(1, sequence_length, 32, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    with torch.inference_mode():
        expected, actual = dense.forward(q, k, v), quant.forward(q, k, v)
    assert calls == ["mxfp8"]
    a, b = actual.float().flatten(), expected.float().flatten()
    delta = a - b
    metrics = {
        "normalized_rmse": (delta.norm() / b.norm()).item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "max_abs": delta.abs().max().item(),
        "p99_abs": torch.quantile(delta.abs(), 0.99).item(),
    }
    print(sequence_length, metrics)
    assert torch.isfinite(actual).all(), metrics
    # B200 operator calibration at these shapes: relative L2 <=0.05257, cosine >=0.99861.
    assert metrics["normalized_rmse"] <= 0.08 and metrics["cosine"] >= 0.997, metrics


@pytest.mark.parametrize("qk_dtype", ["mxfp8", "nvfp4"])
def test_only_video_self_attention_is_quantized(qk_dtype):
    config = DiffusionModelConfig(
        attention_metadata_state={},
        attention=AttentionConfig(
            backend="FLASHINFER",
            quant_attention_config=QuantAttentionConfig(qk_dtype=qk_dtype, v_dtype="fp8"),
        ),
    )
    block = LTX25TransformerBlock(
        idx=0,
        config=config,
        rope_type=LTXRopeType.INTERLEAVED,
        video=TransformerConfig(
            dim=128, heads=1, d_head=128, context_dim=128, apply_gated_attention=True
        ),
        audio=TransformerConfig(
            dim=128, heads=2, d_head=64, context_dim=128, apply_gated_attention=True
        ),
    )
    assert block.attn1.attn.quant_attention_config.qk_dtype == qk_dtype
    for name in (
        "attn2",
        "audio_attn1",
        "audio_attn2",
        "audio_to_video_attn",
        "video_to_audio_attn",
    ):
        assert getattr(block, name).attn.quant_attention_config is None, name
    assert config.attention.quant_attention_config is not None
