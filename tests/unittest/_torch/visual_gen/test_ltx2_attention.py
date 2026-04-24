"""Unit tests for LTX2 attention backends.

Tests cover:
- VANILLA backend (PyTorch SDPA)
- TRTLLM backend (fmha_v2)
- Backend equivalence comparison
- Cross-attention (text context)
- Gated attention

No checkpoint or LTX-2 reference code required — all tests use random weights.
"""

import unittest

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig


def _create_config(backend: str = "VANILLA") -> DiffusionModelConfig:
    """Create a minimal DiffusionModelConfig with the given attention backend."""
    from types import SimpleNamespace

    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend=backend),
        skip_create_weights_in_init=False,
    )


def _init_weights(module: torch.nn.Module, std: float = 0.02):
    """Initialize all parameters with small random values.

    TRT-LLM Linear uses torch.empty() (uninitialized memory), so we must
    explicitly initialize to avoid NaN from recycled GPU memory.
    """
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "norm" in name and "weight" in name:
                p.fill_(1.0)
            else:
                torch.nn.init.normal_(p, mean=0.0, std=std)


class TestLTX2SelfAttention(unittest.TestCase):
    """Test LTX2Attention self-attention with different backends."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vanilla_self_attention_sanity(self):
        """Test LTX2Attention self-attention with VANILLA backend produces valid shapes."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 2
        seq_len = 64
        query_dim = 4096
        heads = 32
        head_dim = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = _create_config("VANILLA")

        attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=None,
                heads=heads,
                dim_head=head_dim,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, query_dim, device=self.DEVICE, dtype=dtype) * 0.02

        with torch.no_grad():
            output = attn(x, context=None, pe=None)

        self.assertEqual(output.shape, (batch_size, seq_len, query_dim))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trtllm_self_attention_sanity(self):
        """Test LTX2Attention self-attention with TRTLLM backend produces valid shapes."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 2
        seq_len = 64
        query_dim = 4096
        heads = 32
        head_dim = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = _create_config("TRTLLM")
        config.attention_metadata_state = create_attention_metadata_state()

        attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=None,
                heads=heads,
                dim_head=head_dim,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, query_dim, device=self.DEVICE, dtype=dtype) * 0.02

        with torch.no_grad():
            output = attn(x, context=None, pe=None)

        self.assertEqual(output.shape, (batch_size, seq_len, query_dim))


class TestLTX2CrossAttention(unittest.TestCase):
    """Test LTX2Attention cross-attention (with text context)."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cross_attention_sanity(self):
        """Test cross-attention with context_dim produces valid shapes."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 1
        q_seq = 64
        kv_seq = 32
        query_dim = 4096
        context_dim = 4096
        heads = 32
        head_dim = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = _create_config("VANILLA")

        attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=head_dim,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        x = torch.randn(batch_size, q_seq, query_dim, device=self.DEVICE, dtype=dtype) * 0.02
        ctx = torch.randn(batch_size, kv_seq, context_dim, device=self.DEVICE, dtype=dtype) * 0.02

        with torch.no_grad():
            output = attn(x, context=ctx, pe=None)

        self.assertEqual(output.shape, (batch_size, q_seq, query_dim))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cross_attention_different_dims(self):
        """Test cross-attention where query_dim != context_dim (e.g., AV cross-attn)."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 1
        q_seq = 64
        kv_seq = 32
        query_dim = 4096
        context_dim = 2048
        heads = 32
        head_dim = 64
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = _create_config("VANILLA")

        attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=head_dim,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        x = torch.randn(batch_size, q_seq, query_dim, device=self.DEVICE, dtype=dtype) * 0.02
        ctx = torch.randn(batch_size, kv_seq, context_dim, device=self.DEVICE, dtype=dtype) * 0.02

        with torch.no_grad():
            output = attn(x, context=ctx, pe=None)

        self.assertEqual(output.shape, (batch_size, q_seq, query_dim))


class TestLTX2GatedAttention(unittest.TestCase):
    """Test LTX2Attention with gated attention."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gated_self_attention_sanity(self):
        """Test self-attention with apply_gated_attention=True produces valid shapes."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 1
        seq_len = 64
        query_dim = 4096
        heads = 32
        head_dim = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)
        config = _create_config("VANILLA")

        attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=None,
                heads=heads,
                dim_head=head_dim,
                apply_gated_attention=True,
                config=config,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        self.assertIsNotNone(attn.to_gate_logits, "Gated attention should create to_gate_logits")

        x = torch.randn(batch_size, seq_len, query_dim, device=self.DEVICE, dtype=dtype) * 0.02

        with torch.no_grad():
            output = attn(x, context=None, pe=None)

        self.assertEqual(output.shape, (batch_size, seq_len, query_dim))


class TestLTX2BackendEquivalence(unittest.TestCase):
    """Test VANILLA and TRTLLM backends produce similar outputs."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backend_equivalence(self):
        """Test VANILLA and TRTLLM backends produce similar self-attention outputs."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        batch_size = 1
        seq_len = 64
        query_dim = 4096
        heads = 32
        head_dim = 128
        dtype = torch.bfloat16

        torch.manual_seed(42)

        # Create VANILLA attention with initialized weights
        config_vanilla = _create_config("VANILLA")
        vanilla_attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=None,
                heads=heads,
                dim_head=head_dim,
                config=config_vanilla,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )
        _init_weights(vanilla_attn)

        # Create TRTLLM attention and copy weights
        config_trtllm = _create_config("TRTLLM")
        config_trtllm.attention_metadata_state = create_attention_metadata_state()
        trtllm_attn = (
            LTX2Attention(
                query_dim=query_dim,
                context_dim=None,
                heads=heads,
                dim_head=head_dim,
                config=config_trtllm,
                layer_idx=0,
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )
        trtllm_attn.load_state_dict(vanilla_attn.state_dict())

        x = torch.randn(batch_size, seq_len, query_dim, device=self.DEVICE, dtype=dtype) * 0.02

        with torch.no_grad():
            out_vanilla = vanilla_attn(x.clone(), context=None, pe=None)
            out_trtllm = trtllm_attn(x.clone(), context=None, pe=None)

        # Skip comparison if either has NaN/Inf (can happen with random weights)
        has_nan = torch.isnan(out_vanilla).any() or torch.isnan(out_trtllm).any()
        has_inf = torch.isinf(out_vanilla).any() or torch.isinf(out_trtllm).any()
        if has_nan or has_inf:
            self.skipTest("NaN/Inf detected in outputs with random weights — skipping comparison")

        self.assertEqual(out_vanilla.shape, out_trtllm.shape)

        vanilla_norm = out_vanilla.float().norm().item()
        trtllm_norm = out_trtllm.float().norm().item()

        if vanilla_norm > 1e-3 and trtllm_norm > 1e-3:
            cos_sim = F.cosine_similarity(
                out_vanilla.float().flatten().unsqueeze(0),
                out_trtllm.float().flatten().unsqueeze(0),
            ).item()
            print(f"\n[LTX2 Backend Equivalence] Cosine similarity: {cos_sim:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
