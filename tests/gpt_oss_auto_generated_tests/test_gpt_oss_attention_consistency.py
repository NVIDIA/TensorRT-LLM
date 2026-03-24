# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Consistency test for GptOssAttention module.

Verifies:
1. The module instantiates without error with correct parameters.
2. The sinks parameter has the correct shape (TP-sliced num_heads).
3. The sliding window flag is correctly set per layer type.
4. The attention mask used for sliding vs full layers is correct.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock

import torch

# We need to set up minimal mocks before importing the modeling code,
# since it depends on TRT-LLM internals that require GPU/runtime setup.


def _make_mock_config(layer_idx=0):
    """Create a mock PretrainedConfig matching gpt-oss-20b."""
    config = MagicMock()
    config.hidden_size = 2880
    config.num_attention_heads = 64
    config.num_key_value_heads = 8
    config.head_dim = 64
    config.max_position_embeddings = 131072
    config.attention_bias = True
    config.torch_dtype = torch.bfloat16
    config.rms_norm_eps = 1e-5
    config.sliding_window = 128
    config.rope_theta = 150000
    config.rope_scaling = {
        "rope_type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "truncate": False,
    }
    config.layer_types = [
        "sliding_attention" if i % 2 == 0 else "full_attention"
        for i in range(24)
    ]
    config.model_type = "gpt_oss"
    # Extra attributes that RopeParams.from_config may access
    config.rotary_pct = None
    config.partial_rotary_factor = None
    config.rotary_dim = None
    config.rotary_emb_base = None
    config.qk_rope_head_dim = None
    config.rope_parameters = None
    config.max_seq_len = None
    return config


def _make_mock_model_config(pretrained_config):
    """Create a mock ModelConfig."""
    model_config = MagicMock()
    model_config.pretrained_config = pretrained_config

    # Mapping mock (TP=1 for basic test)
    mapping = MagicMock()
    mapping.tp_size = 1
    mapping.pp_size = 1
    mapping.cp_size = 1
    mapping.tp_rank = 0
    mapping.rank = 0
    mapping.gpus_per_node = 8
    mapping.enable_attention_dp = False
    mapping.has_cp_helix.return_value = False
    mapping.cp_config = {"cp_type": "HELIX"}
    model_config.mapping = mapping
    model_config.allreduce_strategy = None
    model_config.extra_attrs = {}
    model_config.moe_backend = "cutlass"

    # Quant config
    model_config.quant_config = MagicMock()
    model_config.quant_config.layer_quant_details = {}
    model_config.quant_config.quant_algo = None
    model_config.quant_config.kv_cache_quant_algo = None

    return model_config


class TestGptOssAttentionInstantiation(unittest.TestCase):
    """Test that GptOssAttention can be instantiated and has correct attributes."""

    @classmethod
    def setUpClass(cls):
        """Import the module -- this tests that the class definition is valid."""
        try:
            from tensorrt_llm._torch.models.modeling_gpt_oss import \
                GptOssAttention
            cls.GptOssAttention = GptOssAttention
            cls.import_error = None
        except Exception as e:
            cls.GptOssAttention = None
            cls.import_error = e

    def test_import_succeeds(self):
        """GptOssAttention class should be importable."""
        if self.import_error is not None:
            self.fail(
                f"Failed to import GptOssAttention: {self.import_error}")

    def test_instantiate_sliding_layer(self):
        """Instantiate GptOssAttention for a sliding_attention layer (layer 0)."""
        if self.GptOssAttention is None:
            self.skipTest("Import failed")

        layer_idx = 0  # sliding_attention
        config = _make_mock_config(layer_idx)
        model_config = _make_mock_model_config(config)

        try:
            attn = self.GptOssAttention(model_config, layer_idx=layer_idx)
        except Exception as e:
            self.fail(
                f"Failed to instantiate GptOssAttention for sliding layer: {e}"
            )

        # Check sliding window attributes
        self.assertTrue(attn._is_sliding,
                        "Layer 0 should be a sliding_attention layer")
        self.assertEqual(attn._window_size, 128,
                         "Sliding window size should be 128")

    def test_instantiate_full_layer(self):
        """Instantiate GptOssAttention for a full_attention layer (layer 1)."""
        if self.GptOssAttention is None:
            self.skipTest("Import failed")

        layer_idx = 1  # full_attention
        config = _make_mock_config(layer_idx)
        model_config = _make_mock_model_config(config)

        try:
            attn = self.GptOssAttention(model_config, layer_idx=layer_idx)
        except Exception as e:
            self.fail(
                f"Failed to instantiate GptOssAttention for full layer: {e}")

        # Check full attention attributes
        self.assertFalse(attn._is_sliding,
                         "Layer 1 should be a full_attention layer")
        self.assertIsNone(attn._window_size,
                          "Full attention layer should have no window size")

    def test_sinks_parameter_shape(self):
        """Sinks parameter should have shape [num_heads // tp_size]."""
        if self.GptOssAttention is None:
            self.skipTest("Import failed")

        layer_idx = 0
        config = _make_mock_config(layer_idx)
        model_config = _make_mock_model_config(config)

        attn = self.GptOssAttention(model_config, layer_idx=layer_idx)

        # With TP=1, num_heads should be 64
        self.assertTrue(hasattr(attn, 'sinks'),
                        "GptOssAttention should have a 'sinks' parameter")
        self.assertIsInstance(attn.sinks, torch.nn.Parameter,
                              "sinks should be an nn.Parameter")
        expected_num_heads = 64  # num_attention_heads with TP=1
        self.assertEqual(
            attn.sinks.shape, (expected_num_heads,),
            f"sinks shape should be ({expected_num_heads},), "
            f"got {attn.sinks.shape}")
        self.assertEqual(attn.sinks.dtype, torch.bfloat16,
                         "sinks dtype should be bfloat16")

    def test_sinks_parameter_shape_tp2(self):
        """With TP=2, sinks should have shape [num_heads // 2]."""
        if self.GptOssAttention is None:
            self.skipTest("Import failed")

        layer_idx = 0
        config = _make_mock_config(layer_idx)
        model_config = _make_mock_model_config(config)
        model_config.mapping.tp_size = 2
        model_config.mapping.tp_rank = 0

        attn = self.GptOssAttention(model_config, layer_idx=layer_idx)

        # With TP=2, num_heads should be 32
        expected_num_heads = 32
        self.assertEqual(
            attn.sinks.shape, (expected_num_heads,),
            f"sinks shape with TP=2 should be ({expected_num_heads},), "
            f"got {attn.sinks.shape}")

    def test_all_layer_types_correct(self):
        """Verify sliding/full classification for all 24 layers."""
        if self.GptOssAttention is None:
            self.skipTest("Import failed")

        config = _make_mock_config()
        model_config = _make_mock_model_config(config)

        for layer_idx in range(24):
            attn = self.GptOssAttention(model_config, layer_idx=layer_idx)
            expected_sliding = (layer_idx % 2 == 0)
            self.assertEqual(
                attn._is_sliding, expected_sliding,
                f"Layer {layer_idx}: _is_sliding should be {expected_sliding}")
            if expected_sliding:
                self.assertEqual(
                    attn._window_size, 128,
                    f"Layer {layer_idx}: window size should be 128")
            else:
                self.assertIsNone(
                    attn._window_size,
                    f"Layer {layer_idx}: window size should be None")

    def test_attention_base_class_parameters(self):
        """Verify that Attention base class received the correct parameters."""
        if self.GptOssAttention is None:
            self.skipTest("Import failed")

        layer_idx = 0
        config = _make_mock_config(layer_idx)
        model_config = _make_mock_model_config(config)

        attn = self.GptOssAttention(model_config, layer_idx=layer_idx)

        # Check parameters that Attention.__init__ sets
        # Note: num_heads is TP-divided, so with TP=1 it stays 64
        self.assertEqual(attn.hidden_size, 2880)
        self.assertEqual(attn.head_dim, 64)
        self.assertEqual(attn.max_position_embeddings, 131072)
        # dense_bias should be True (set from config.attention_bias)
        self.assertTrue(attn.dense_bias,
                        "dense_bias should be True")

    def test_sliding_window_mask_enum_exists(self):
        """Check that the PredefinedAttentionMask.SLIDING_WINDOW_CAUSAL
        referenced in the forward method actually exists."""
        if self.GptOssAttention is None:
            self.skipTest("Import failed")

        from tensorrt_llm._torch.attention_backend.interface import \
            PredefinedAttentionMask

        has_sliding = hasattr(PredefinedAttentionMask,
                              'SLIDING_WINDOW_CAUSAL')
        self.assertTrue(
            has_sliding,
            "PredefinedAttentionMask.SLIDING_WINDOW_CAUSAL does not exist! "
            "The GptOssAttention.forward() method references it but the enum "
            "only has members: "
            f"{[m.name for m in PredefinedAttentionMask]}. "
            "This will cause an AttributeError at runtime. "
            "The sliding window should be handled via attention_window_size "
            "parameter with CAUSAL mask (see Gemma3 pattern).")


class TestDeinterleaveGateUp(unittest.TestCase):
    """Test the _deinterleave_gate_up static method."""

    @classmethod
    def setUpClass(cls):
        try:
            from tensorrt_llm._torch.models.modeling_gpt_oss import \
                GptOssForCausalLM
            cls.cls_ = GptOssForCausalLM
            cls.import_error = None
        except Exception as e:
            cls.cls_ = None
            cls.import_error = e

    def test_deinterleave_1d(self):
        """Test de-interleaving on a 1D bias tensor."""
        if self.cls_ is None:
            self.skipTest(f"Import failed: {self.import_error}")

        # Simulate interleaved bias [g0, u0, g1, u1, g2, u2]
        tensor = torch.tensor([10, 20, 11, 21, 12, 22], dtype=torch.float)
        result = self.cls_._deinterleave_gate_up(tensor, dim=0)
        expected = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.float)
        self.assertTrue(
            torch.equal(result, expected),
            f"Deinterleave mismatch: got {result}, expected {expected}")

    def test_deinterleave_2d(self):
        """Test de-interleaving on a 2D tensor along dim=1."""
        if self.cls_ is None:
            self.skipTest(f"Import failed: {self.import_error}")

        # shape [2, 6] with interleaved gate/up on dim=1
        tensor = torch.tensor(
            [[10, 20, 11, 21, 12, 22], [30, 40, 31, 41, 32, 42]],
            dtype=torch.float)
        result = self.cls_._deinterleave_gate_up(tensor, dim=1)
        expected = torch.tensor(
            [[10, 11, 12, 20, 21, 22], [30, 31, 32, 40, 41, 42]],
            dtype=torch.float)
        self.assertTrue(
            torch.equal(result, expected),
            f"Deinterleave mismatch: got {result}, expected {expected}")


class TestTransformWeightsSinksTPSlice(unittest.TestCase):
    """Test that _transform_weights correctly TP-slices attention sinks."""

    @classmethod
    def setUpClass(cls):
        try:
            from tensorrt_llm._torch.models.modeling_gpt_oss import \
                GptOssForCausalLM
            cls.cls_ = GptOssForCausalLM
            cls.import_error = None
        except Exception as e:
            cls.cls_ = None
            cls.import_error = e

    def test_sinks_tp_slice(self):
        """Sinks should be TP-sliced along the heads dimension."""
        if self.cls_ is None:
            self.skipTest(f"Import failed: {self.import_error}")

        # Create a minimal mock instance with the _transform_weights method
        instance = object.__new__(self.cls_)
        instance.model_config = MagicMock()
        instance.model_config.mapping.tp_size = 2
        instance.model_config.mapping.tp_rank = 1

        sinks = torch.arange(64, dtype=torch.bfloat16)
        weights = {"model.layers.0.self_attn.sinks": sinks}

        transformed = instance._transform_weights(weights)
        result = transformed["model.layers.0.self_attn.sinks"]

        # TP rank 1 should get the second half: [32..63]
        expected = torch.arange(32, 64, dtype=torch.bfloat16)
        self.assertTrue(
            torch.equal(result, expected),
            f"TP-sliced sinks mismatch: got {result[:5]}..., "
            f"expected {expected[:5]}...")


if __name__ == "__main__":
    unittest.main()
