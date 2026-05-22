"""Tests for fused QKV support in diffusion models.

Tests:
1. Model structure with fuse_qkv=True (default) vs fuse_qkv=False
2. Weight loading works for fused QKV layers
"""

import unittest
from types import SimpleNamespace
from typing import Dict

import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


def _create_test_config(hidden_size: int = 64) -> DiffusionModelConfig:
    """Create a test DiffusionModelConfig."""
    num_heads = hidden_size // 8  # e.g., 64 // 8 = 8 heads
    head_dim = 8
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            num_layers=2,
            ffn_dim=256,
            out_channels=16,
            patch_size=[1, 2, 2],
            in_channels=16,
            text_dim=64,
            freq_dim=32,
        ),
    )


class TestFusedQKVWeightLoading(unittest.TestCase):
    """Test weight loading for fused QKV layers."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.hidden_size = 64

    def _create_mock_checkpoint_weights(self) -> Dict[str, torch.Tensor]:
        """Create mock checkpoint weights with separate to_q, to_k, to_v."""
        dtype = torch.bfloat16  # Match model dtype
        weights = {}
        for block_idx in range(2):
            for attn_name in ["attn1", "attn2"]:
                prefix = f"blocks.{block_idx}.{attn_name}"
                # Separate QKV weights (as in checkpoint)
                weights[f"{prefix}.to_q.weight"] = torch.randn(
                    self.hidden_size, self.hidden_size, dtype=dtype
                )
                weights[f"{prefix}.to_q.bias"] = torch.randn(self.hidden_size, dtype=dtype)
                weights[f"{prefix}.to_k.weight"] = torch.randn(
                    self.hidden_size, self.hidden_size, dtype=dtype
                )
                weights[f"{prefix}.to_k.bias"] = torch.randn(self.hidden_size, dtype=dtype)
                weights[f"{prefix}.to_v.weight"] = torch.randn(
                    self.hidden_size, self.hidden_size, dtype=dtype
                )
                weights[f"{prefix}.to_v.bias"] = torch.randn(self.hidden_size, dtype=dtype)
                # Output projection
                weights[f"{prefix}.to_out.0.weight"] = torch.randn(
                    self.hidden_size, self.hidden_size, dtype=dtype
                )
                weights[f"{prefix}.to_out.0.bias"] = torch.randn(self.hidden_size, dtype=dtype)

            # FFN weights
            ffn_dim = 256
            prefix = f"blocks.{block_idx}.ffn"
            weights[f"{prefix}.net.0.proj.weight"] = torch.randn(
                ffn_dim, self.hidden_size, dtype=dtype
            )
            weights[f"{prefix}.net.0.proj.bias"] = torch.randn(ffn_dim, dtype=dtype)
            weights[f"{prefix}.net.2.weight"] = torch.randn(self.hidden_size, ffn_dim, dtype=dtype)
            weights[f"{prefix}.net.2.bias"] = torch.randn(self.hidden_size, dtype=dtype)

        # proj_out
        weights["proj_out.weight"] = torch.randn(64, self.hidden_size, dtype=dtype)
        weights["proj_out.bias"] = torch.randn(64, dtype=dtype)

        return weights

    def test_load_weights_fused(self):
        """Test loading weights with fused QKV (default for self-attention)."""
        from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

        config = _create_test_config(self.hidden_size)

        # Create model - self-attention (attn1) uses fused QKV by default
        model = WanTransformer3DModel(model_config=config)
        weights = self._create_mock_checkpoint_weights()

        # Load weights (model handles fused QKV internally via DynamicLinearWeightLoader)
        model.load_weights(weights)

        # Verify fused weights were loaded correctly for self-attention
        attn1 = model.blocks[0].attn1
        qkv_weight = attn1.qkv_proj.weight.data

        # Expected: concatenation of to_q, to_k, to_v weights
        expected_weight = torch.cat(
            [
                weights["blocks.0.attn1.to_q.weight"],
                weights["blocks.0.attn1.to_k.weight"],
                weights["blocks.0.attn1.to_v.weight"],
            ],
            dim=0,
        )

        self.assertEqual(qkv_weight.shape, expected_weight.shape)
        self.assertTrue(torch.allclose(qkv_weight, expected_weight))

        # Also verify cross-attention (attn2) uses separate Q/K/V
        attn2 = model.blocks[0].attn2
        self.assertTrue(hasattr(attn2, "to_q"), "Cross-attention should have separate to_q")
        self.assertTrue(
            torch.allclose(attn2.to_q.weight.data, weights["blocks.0.attn2.to_q.weight"])
        )


if __name__ == "__main__":
    unittest.main()
