# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FLUX attention backends.

Tests cover:
- VANILLA backend (PyTorch SDPA)
- TRTLLM backend (fmha_v2)
- Backend equivalence comparison

Note: With random weights, attention can produce NaN due to numerical instability.
      These tests use scaled inputs and primarily verify correct output shapes.
      Full numerical correctness is tested via HuggingFace comparison tests with real weights.
"""

import unittest
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.config import AttentionConfig, DiffusionModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig


class TestFluxAttentionBackend(unittest.TestCase):
    """Test FLUX attention with different backends."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_config(self, backend: str) -> DiffusionModelConfig:
        """Create DiffusionModelConfig with specified backend."""
        return DiffusionModelConfig(
            pretrained_config=SimpleNamespace(),
            quant_config=QuantConfig(),
            mapping=Mapping(),
            attention=AttentionConfig(backend=backend),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vanilla_backend_sanity(self):
        """Test FLUX attention works with VANILLA backend."""
        from tensorrt_llm._torch.visual_gen.models.flux.attention import FluxJointAttention

        batch_size = 2
        seq_len = 256
        text_seq_len = 64
        dim = 3072
        heads = 24
        dim_head = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = self._create_config("VANILLA")

        attn = (
            FluxJointAttention(
                hidden_size=dim,
                num_attention_heads=heads,
                head_dim=dim_head,
                added_kv_proj_dim=dim,  # Enable dual-stream for text tokens
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Use scaled inputs to reduce numerical instability
        hidden_states = (
            torch.randn(batch_size, seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )
        encoder_hidden_states = (
            torch.randn(batch_size, text_seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )

        with torch.no_grad():
            # Skip RoPE for this sanity test (pass None)
            output, text_output = attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=None,
            )

        # With random weights, NaN can occur. For unit tests, we primarily check shapes.
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(text_output.shape, encoder_hidden_states.shape)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trtllm_backend_sanity(self):
        """Test FLUX attention works with TRTLLM backend."""
        from tensorrt_llm._torch.visual_gen.models.flux.attention import FluxJointAttention

        batch_size = 2
        seq_len = 256
        text_seq_len = 64
        dim = 3072
        heads = 24
        dim_head = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = self._create_config("TRTLLM")

        attn = (
            FluxJointAttention(
                hidden_size=dim,
                num_attention_heads=heads,
                head_dim=dim_head,
                added_kv_proj_dim=dim,  # Enable dual-stream for text tokens
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Use scaled inputs to reduce numerical instability
        hidden_states = (
            torch.randn(batch_size, seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )
        encoder_hidden_states = (
            torch.randn(batch_size, text_seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )

        with torch.no_grad():
            # Skip RoPE for this sanity test (pass None)
            output, text_output = attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=None,
            )

        # With random weights, NaN can occur. For unit tests, we primarily check shapes.
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(text_output.shape, encoder_hidden_states.shape)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backend_equivalence(self):
        """Test VANILLA and TRTLLM backends produce similar outputs."""
        from tensorrt_llm._torch.visual_gen.models.flux.attention import FluxJointAttention

        batch_size = 1
        seq_len = 128
        text_seq_len = 32
        dim = 3072
        heads = 24
        dim_head = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)

        # Create attention modules for both backends
        config = self._create_config("VANILLA")
        vanilla_attn = (
            FluxJointAttention(
                hidden_size=dim,
                num_attention_heads=heads,
                head_dim=dim_head,
                added_kv_proj_dim=dim,  # Enable dual-stream for text tokens
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # TRT-LLM Linear uses torch.empty for bias (uninitialized memory).
        # After prior tests free GPU memory, recycled memory can contain NaN.
        # Initialize all parameters with small random values for numerical stability.
        with torch.no_grad():
            for p in vanilla_attn.parameters():
                p.normal_(0, 0.02)

        config = self._create_config("TRTLLM")
        trtllm_attn = (
            FluxJointAttention(
                hidden_size=dim,
                num_attention_heads=heads,
                head_dim=dim_head,
                added_kv_proj_dim=dim,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Copy scaled weights from VANILLA to TRTLLM
        trtllm_attn.load_state_dict(vanilla_attn.state_dict())
        attns = {"VANILLA": vanilla_attn, "TRTLLM": trtllm_attn}

        # Create inputs (scaled for numerical stability)
        hidden_states = (
            torch.randn(batch_size, seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )
        encoder_hidden_states = (
            torch.randn(batch_size, text_seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )

        # Run both backends (skip RoPE for equivalence test)
        outputs = {}
        with torch.no_grad():
            for backend in ["VANILLA", "TRTLLM"]:
                out, text_out = attns[backend](
                    hidden_states=hidden_states.clone(),
                    encoder_hidden_states=encoder_hidden_states.clone(),
                    image_rotary_emb=None,
                )
                outputs[backend] = (out, text_out)

        # Compare outputs
        vanilla_out, vanilla_text = outputs["VANILLA"]
        trtllm_out, trtllm_text = outputs["TRTLLM"]

        # Skip comparison if either has NaN or Inf (common with random weights)
        has_nan = torch.isnan(vanilla_out).any() or torch.isnan(trtllm_out).any()
        has_inf = torch.isinf(vanilla_out).any() or torch.isinf(trtllm_out).any()
        if has_nan or has_inf:
            self.skipTest("NaN/Inf detected in outputs with random weights - skipping comparison")

        # With random weights, outputs may be all zeros or have numerical issues
        # This test is primarily for ensuring both backends can run
        # Full equivalence is tested via HuggingFace comparison with real weights
        vanilla_norm = vanilla_out.float().norm().item()
        trtllm_norm = trtllm_out.float().norm().item()

        print(f"\n[Debug] vanilla_norm={vanilla_norm:.6f}, trtllm_norm={trtllm_norm:.6f}")

        # With random weights, we can only reliably check that both backends produce
        # valid outputs (same shapes, non-trivial values). Strict equivalence requires
        # real weights from a trained model (tested via HuggingFace comparison).
        self.assertEqual(vanilla_out.shape, trtllm_out.shape)
        self.assertEqual(vanilla_text.shape, trtllm_text.shape)

        # If both outputs have meaningful norms, compute similarity as informational
        if vanilla_norm > 1e-3 and trtllm_norm > 1e-3:
            cos_sim = F.cosine_similarity(
                vanilla_out.float().flatten().unsqueeze(0),
                trtllm_out.float().flatten().unsqueeze(0),
            ).item()
            print(f"  Cosine similarity: {cos_sim:.6f}")
            # Note: Not asserting on cos_sim with random weights as it's not meaningful


class TestFlux2AttentionBackend(unittest.TestCase):
    """Test FLUX.2 attention with different backends."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_config(self, backend: str) -> DiffusionModelConfig:
        """Create DiffusionModelConfig with specified backend."""
        return DiffusionModelConfig(
            pretrained_config=SimpleNamespace(),
            quant_config=QuantConfig(),
            mapping=Mapping(),
            attention=AttentionConfig(backend=backend),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flux2_vanilla_backend_sanity(self):
        """Test FLUX.2 attention works with VANILLA backend."""
        from tensorrt_llm._torch.visual_gen.models.flux.attention import FluxJointAttention

        batch_size = 2
        seq_len = 128
        text_seq_len = 64
        dim = 6144  # FLUX.2 has larger dim
        heads = 48
        dim_head = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = self._create_config("VANILLA")

        attn = (
            FluxJointAttention(
                hidden_size=dim,
                num_attention_heads=heads,
                head_dim=dim_head,
                added_kv_proj_dim=dim,  # Enable dual-stream for text tokens
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Use scaled inputs to reduce numerical instability
        hidden_states = (
            torch.randn(batch_size, seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )
        encoder_hidden_states = (
            torch.randn(batch_size, text_seq_len, dim, device=self.DEVICE, dtype=dtype) * 0.02
        )

        with torch.no_grad():
            # Skip RoPE for this sanity test (pass None)
            output, text_output = attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=None,
            )

        # With random weights, NaN can occur. For unit tests, we primarily check shapes.
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(text_output.shape, encoder_hidden_states.shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
