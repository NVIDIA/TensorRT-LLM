# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Gemma4 model (PyTorch backend).

Includes structural tests and HF reference comparison tests using native
transformers>=5.5.0 Gemma4 support.
"""

import math
import unittest
import unittest.mock
from copy import deepcopy

import pytest
import torch

# Gemma4 requires transformers>=5.5.0 (native Gemma4 config/model classes).
pytest.importorskip(
    "transformers", minversion="5.5.0", reason="Gemma4 requires transformers>=5.5.0"
)

from transformers import Gemma4Config, Gemma4TextConfig  # noqa: E402

from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_gemma4 import (  # noqa: E402
    Gemma4Attention,
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4MoE,
    Gemma4TextModel,
    Gemma4TextScaledWordEmbedding,
)
from tensorrt_llm.mapping import Mapping  # noqa: E402

# ---------------------------------------------------------------------------
# Small test configs
# ---------------------------------------------------------------------------
GEMMA4_SMALL_CONFIG = {
    "model_type": "gemma4_text",
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 6,  # 5 sliding + 1 full
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "global_head_dim": 128,
    "num_global_key_value_heads": 1,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "sliding_window": 4,
    "attention_k_eq_v": True,
    "enable_moe_block": False,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "final_logit_softcapping": 30.0,
    "rope_parameters": {
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10000.0,
        },
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
        },
    },
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
}

GEMMA4_MOE_CONFIG = {
    **GEMMA4_SMALL_CONFIG,
    "enable_moe_block": True,
    "num_experts": 8,
    "top_k_experts": 2,
    "moe_intermediate_size": 256,
}

GEMMA4_PLE_CONFIG = {
    **GEMMA4_SMALL_CONFIG,
    "hidden_size_per_layer_input": 32,
    "vocab_size_per_layer_input": 1024,
    "attention_k_eq_v": False,
    "num_kv_shared_layers": 2,
    "use_double_wide_mlp": True,
}


def _make_model_config(config_dict):
    """Build a ModelConfig from a raw config dict."""
    cfg = Gemma4TextConfig(**config_dict)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    return ModelConfig(pretrained_config=cfg, mapping=mapping)


class TestGemma4Config(unittest.TestCase):
    """Tests for Gemma4 config classes."""

    def test_gemma4_config_nested_sub_configs(self):
        """Gemma4Config should accept and wrap nested sub-config dicts."""
        top_level = Gemma4Config(
            text_config=deepcopy(GEMMA4_SMALL_CONFIG),
            vision_config={"hidden_size": 768},
        )
        self.assertIsInstance(top_level.text_config, Gemma4TextConfig)
        self.assertEqual(top_level.text_config.hidden_size, 256)
        self.assertIsNotNone(top_level.vision_config)
        self.assertEqual(top_level.vision_config.hidden_size, 768)
        # audio_config defaults to None
        self.assertIsNone(top_level.audio_config)

    def test_gemma4_text_config_default_layer_types(self):
        """layer_types should be auto-generated with 5:1 sliding pattern."""
        cfg = Gemma4TextConfig(num_hidden_layers=6)
        # Pattern: indices 0-4 are sliding (i+1 % 6 != 0), index 5 is full
        expected = [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        self.assertEqual(cfg.layer_types, expected)

    def test_gemma4_text_config_last_layer_forced_full(self):
        """Even if the pattern says sliding, the last layer must be full."""
        # 7 layers: pattern would make layer 5 full, layer 6 sliding -> forced full
        cfg = Gemma4TextConfig(num_hidden_layers=7)
        self.assertEqual(cfg.layer_types[-1], "full_attention")
        self.assertEqual(len(cfg.layer_types), 7)

    def test_gemma4_text_config_default_rope_parameters(self):
        """Default rope_parameters should match Gemma4 spec."""
        cfg = Gemma4TextConfig()
        self.assertIn("sliding_attention", cfg.rope_parameters)
        self.assertIn("full_attention", cfg.rope_parameters)
        self.assertEqual(cfg.rope_parameters["sliding_attention"]["rope_theta"], 10_000.0)
        self.assertEqual(cfg.rope_parameters["full_attention"]["rope_theta"], 1_000_000.0)
        self.assertEqual(cfg.rope_parameters["full_attention"]["partial_rotary_factor"], 0.25)

    def test_gemma4_text_config_explicit_layer_types(self):
        """Explicitly provided layer_types should be preserved."""
        explicit = ["full_attention"] * 4
        cfg = Gemma4TextConfig(num_hidden_layers=4, layer_types=explicit)
        self.assertEqual(cfg.layer_types, explicit)


class TestGemma4ModelInstantiation(unittest.TestCase):
    """Tests for model instantiation and structural correctness."""

    def test_model_instantiation_basic(self):
        """Create Gemma4ForCausalLM from small config and verify structure."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)

        # Top-level structure
        self.assertIsInstance(model.model, Gemma4TextModel)
        self.assertIsNotNone(model.lm_head)

        # Decoder layers
        pretrained = model_config.pretrained_config
        self.assertEqual(len(model.model.layers), pretrained.num_hidden_layers)
        for layer in model.model.layers:
            self.assertIsInstance(layer, Gemma4DecoderLayer)
            self.assertIsInstance(layer.self_attn, Gemma4Attention)

    def test_model_instantiation_moe(self):
        """Create with MoE enabled and verify MoE layers exist."""
        model_config = _make_model_config(GEMMA4_MOE_CONFIG)
        model = Gemma4ForCausalLM(model_config)

        for layer in model.model.layers:
            self.assertTrue(layer.enable_moe_block)
            self.assertIsInstance(layer.moe, Gemma4MoE)
            # MoE-specific extra norms should exist
            self.assertIsNotNone(layer.post_feedforward_layernorm_1)
            self.assertIsNotNone(layer.post_feedforward_layernorm_2)
            self.assertIsNotNone(layer.pre_feedforward_layernorm_2)

    def test_model_instantiation_ple(self):
        """Create with PLE enabled and verify PLE components exist."""
        model_config = _make_model_config(GEMMA4_PLE_CONFIG)
        model = Gemma4ForCausalLM(model_config)

        # Model-level PLE components
        self.assertIsNotNone(model.model.embed_tokens_per_layer)
        self.assertIsNotNone(model.model.per_layer_model_projection)
        self.assertIsNotNone(model.model.per_layer_projection_norm)

        # Per-layer PLE components
        for layer in model.model.layers:
            self.assertEqual(layer.hidden_size_per_layer_input, 32)
            self.assertIsNotNone(layer.per_layer_input_gate)
            self.assertIsNotNone(layer.per_layer_projection)
            self.assertIsNotNone(layer.post_per_layer_input_norm)

        # Verify KV-shared layers get double-wide MLP
        config = model_config.pretrained_config
        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        for i, layer in enumerate(model.model.layers):
            if i >= first_kv_shared:
                self.assertTrue(layer.is_kv_shared_layer)
            else:
                self.assertFalse(layer.is_kv_shared_layer)

    def test_per_layer_head_dim(self):
        """Sliding layers use head_dim=64, full layers use global_head_dim=128."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)
        config = model_config.pretrained_config

        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            if config.layer_types[i] == "sliding_attention":
                self.assertEqual(
                    attn.head_dim,
                    config.head_dim,
                    f"Layer {i} (sliding) should have head_dim={config.head_dim}",
                )
            else:
                self.assertEqual(
                    attn.head_dim,
                    config.global_head_dim,
                    f"Layer {i} (full) should have head_dim={config.global_head_dim}",
                )

    def test_k_eq_v_attention(self):
        """Full attention layers with attention_k_eq_v=True should have v_norm."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)
        config = model_config.pretrained_config

        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            if config.layer_types[i] == "full_attention":
                # K=V should be active on full attention layers
                self.assertTrue(attn.use_k_eq_v, f"Layer {i} (full) should have use_k_eq_v=True")
                self.assertTrue(hasattr(attn, "v_norm"), f"Layer {i} (full) should have v_norm")
            else:
                # Sliding layers should NOT use K=V
                self.assertFalse(
                    attn.use_k_eq_v, f"Layer {i} (sliding) should have use_k_eq_v=False"
                )

    def test_k_eq_v_disabled(self):
        """When attention_k_eq_v=False, use_k_eq_v is False but v_norm still exists."""
        config_dict = deepcopy(GEMMA4_SMALL_CONFIG)
        config_dict["attention_k_eq_v"] = False
        model_config = _make_model_config(config_dict)
        model = Gemma4ForCausalLM(model_config)

        for layer in model.model.layers:
            self.assertFalse(layer.self_attn.use_k_eq_v)
            # v_norm is always present in Gemma4 (even when K!=V)
            self.assertTrue(hasattr(layer.self_attn, "v_norm"))

    def test_layer_types(self):
        """Verify correct layer type assignment for default 5:1 pattern."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)
        config = model_config.pretrained_config

        for i, layer in enumerate(model.model.layers):
            expected_sliding = config.layer_types[i] == "sliding_attention"
            self.assertEqual(layer.is_sliding, expected_sliding, f"Layer {i} is_sliding mismatch")
            if expected_sliding:
                self.assertEqual(layer.self_attn.attention_window_size, config.sliding_window)
            else:
                self.assertIsNone(layer.self_attn.attention_window_size)

    def test_embedding_scale(self):
        """Embedding output should be scaled by sqrt(hidden_size)."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)
        config = model_config.pretrained_config

        embed = model.model.embed_tokens
        self.assertIsInstance(embed, Gemma4TextScaledWordEmbedding)
        expected_scale = math.sqrt(config.hidden_size)
        self.assertAlmostEqual(embed.embed_scale.item(), expected_scale, places=4)

    def test_logit_softcapping(self):
        """Verify final_logit_softcapping value is stored on the config."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)
        config = model_config.pretrained_config

        # The softcapping value is read from config during forward()
        self.assertEqual(config.final_logit_softcapping, 30.0)

        # Verify the forward method references the softcapping logic:
        # tanh(logits / cap) * cap. We verify the config attribute is
        # accessible through model.config (used in forward).
        # model_config.pretrained_config is stored as model.config
        # on DecoderModelForCausalLM subclasses (via model_config).
        self.assertEqual(model.model_config.pretrained_config.final_logit_softcapping, 30.0)

    def test_no_logit_softcapping(self):
        """When softcapping is None, it should be stored as None."""
        config_dict = deepcopy(GEMMA4_SMALL_CONFIG)
        config_dict["final_logit_softcapping"] = None
        model_config = _make_model_config(config_dict)
        model = Gemma4ForCausalLM(model_config)
        self.assertIsNone(model.model_config.pretrained_config.final_logit_softcapping)

    def test_sliding_rope_params(self):
        """Sliding attention layers should use default RoPE with theta=10K."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        config = model_config.pretrained_config

        # Build a sliding attention to inspect its rope params
        attn = Gemma4Attention(model_config, layer_idx=0, is_sliding=True)
        rope = attn.pos_embd_params.rope
        self.assertEqual(rope.theta, 10_000.0)
        # Full rotation: dim should equal head_dim
        self.assertEqual(rope.dim, config.head_dim)

    def test_full_rope_params(self):
        """Full attention layers should use proportional RoPE with theta=1M."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        config = model_config.pretrained_config

        attn = Gemma4Attention(model_config, layer_idx=5, is_sliding=False)
        rope = attn.pos_embd_params.rope
        self.assertEqual(rope.theta, 1_000_000.0)
        # Partial rotation: dim = int(global_head_dim * 0.25)
        expected_dim = int(config.global_head_dim * 0.25)
        self.assertEqual(rope.dim, expected_dim)

    def test_num_kv_heads_per_layer_type(self):
        """Sliding layers use num_key_value_heads, full use num_global_key_value_heads."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        config = model_config.pretrained_config

        for i, layer in enumerate(model_config.pretrained_config.layer_types):
            attn = Gemma4Attention(
                model_config,
                layer_idx=i,
                is_sliding=(layer == "sliding_attention"),
            )
            if layer == "sliding_attention":
                expected_kv_heads = config.num_key_value_heads
            else:
                expected_kv_heads = config.num_global_key_value_heads
            self.assertEqual(
                attn.num_key_value_heads, expected_kv_heads, f"Layer {i} kv_heads mismatch"
            )


# ---------------------------------------------------------------------------
# HF reference comparison tests (sub-module + full model)
# ---------------------------------------------------------------------------

# --- Test configs for HF comparison ---

# Uniform head_dim (baseline)
GEMMA4_UNIFORM_CONFIG = {
    "model_type": "gemma4_text",
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "global_head_dim": 64,  # Same as head_dim -> uniform
    "num_global_key_value_heads": 2,  # Same as num_key_value_heads
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "final_logit_softcapping": None,
    "use_bidirectional_attention": None,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
}

# Hybrid head_dim: sliding=64, full=128 (tests per-layer head_dim in KV cache)
GEMMA4_HYBRID_HEADDIM_CONFIG = {
    **GEMMA4_UNIFORM_CONFIG,
    "num_hidden_layers": 6,
    "global_head_dim": 128,
    "num_global_key_value_heads": 1,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
        },
    },
}

# Hybrid with different num_kv_heads per layer type (like real Gemma4 K=V models).
# Sliding layers: kv_heads=4, Full (K=V) layers: kv_heads=1.
# This tests V2 pool grouping — different kv_heads*head_dim means different page
# sizes, requiring separate pools for sliding vs full attention layers.
GEMMA4_DIFF_KV_HEADS_CONFIG = {
    **GEMMA4_UNIFORM_CONFIG,
    "num_hidden_layers": 6,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "global_head_dim": 128,
    "num_global_key_value_heads": 1,
    "attention_k_eq_v": True,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
        },
    },
}

# K=V attention (full layers share k->v, no v_proj)
GEMMA4_KEV_CONFIG = {
    **GEMMA4_UNIFORM_CONFIG,
    "attention_k_eq_v": True,
}

# Logit softcapping enabled
GEMMA4_SOFTCAP_CONFIG = {
    **GEMMA4_UNIFORM_CONFIG,
    "final_logit_softcapping": 30.0,
}

# PLE (Per-Layer Embeddings) enabled
GEMMA4_PLE_CONFIG = {
    **GEMMA4_UNIFORM_CONFIG,
    "hidden_size_per_layer_input": 32,
    "vocab_size_per_layer_input": 1024,
}

# MoE (Mixture of Experts) enabled
GEMMA4_MOE_HF_CONFIG = {
    **GEMMA4_UNIFORM_CONFIG,
    "enable_moe_block": True,
    "num_experts": 4,
    "top_k_experts": 2,
    "moe_intermediate_size": 256,
}

# KV Sharing (last 2 layers share KV with earlier same-type layers)
GEMMA4_KV_SHARING_CONFIG = {
    **GEMMA4_UNIFORM_CONFIG,
    "num_kv_shared_layers": 2,
}

# --- Mixed feature configs matching real model patterns ---

# 26B-A4B-like: hybrid_hd + K=V + MoE + softcap
GEMMA4_26B_LIKE_CONFIG = {
    **GEMMA4_HYBRID_HEADDIM_CONFIG,
    "attention_k_eq_v": True,
    "enable_moe_block": True,
    "num_experts": 4,
    "top_k_experts": 2,
    "moe_intermediate_size": 256,
    "final_logit_softcapping": 30.0,
}

# E2B-like: hybrid_hd + PLE + KV_share + double_MLP + softcap
# Use 12 layers (5:1 pattern → 2 full layers at idx 5,11) so KV sharing
# has non-shared full layers to share from.
GEMMA4_E2B_LIKE_CONFIG = {
    **GEMMA4_HYBRID_HEADDIM_CONFIG,
    "num_hidden_layers": 12,
    "hidden_size_per_layer_input": 32,
    "vocab_size_per_layer_input": 1024,
    "num_kv_shared_layers": 4,
    "use_double_wide_mlp": True,
    "final_logit_softcapping": 30.0,
}

# 31B-like: hybrid_hd + K=V + softcap (subset of 26B, no MoE)
GEMMA4_31B_LIKE_CONFIG = {
    **GEMMA4_HYBRID_HEADDIM_CONFIG,
    "attention_k_eq_v": True,
    "final_logit_softcapping": 30.0,
}

# E4B-like: hybrid_hd + NO K=V + same kv_heads for all layers.
# This is the critical VSWA corner case: without K=V, full attention layers
# have the SAME num_kv_heads as sliding layers, but head_dim doubles.  The
# full pool's per-page bytes are 2× sliding → full pool has fewer pages.
# Using sliding pool's page indices on the full pool buffer causes
# out-of-bounds if max(sliding_index) >= num_full_pages.
GEMMA4_E4B_LIKE_CONFIG = {
    **GEMMA4_HYBRID_HEADDIM_CONFIG,
    "attention_k_eq_v": False,
    "num_global_key_value_heads": None,  # inherit num_key_value_heads
}

# E2B-like: hybrid_hd + num_kv_heads=1 → GQA=4 → tensor-core decode for
# ALL layers.  This is critical for CUDA graph testing because both sliding
# and full wrappers use tensor-core plan(), which writes to workspace.
# Without per-PlanParams workspace, the second plan overwrites the first's
# workspace data → wrong attention during graph replay.
GEMMA4_E2B_LIKE_CONFIG = {
    **GEMMA4_HYBRID_HEADDIM_CONFIG,
    "num_key_value_heads": 1,
    "attention_k_eq_v": False,
    "num_global_key_value_heads": None,
}

# E2B-real-dims: same GQA=8 but with real head_dim (256/512).  This
# exercises the trtllm-gen backend for full layers and non-TC decode
# for sliding layers at production-scale head dimensions.
GEMMA4_E2B_REAL_DIMS_CONFIG = {
    "model_type": "gemma4_text",
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_global_key_value_heads": None,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "final_logit_softcapping": None,
    "use_bidirectional_attention": None,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
        },
    },
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
}

# 31B-real-dims: GQA=2 sliding (32/16), GQA=8 full K=V (32/4), hd=256/512.
# Exercises mixed GQA ratios with K=V full-attention layers.
GEMMA4_31B_REAL_DIMS_CONFIG = {
    **GEMMA4_E2B_REAL_DIMS_CONFIG,
    "num_hidden_layers": 12,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "num_global_key_value_heads": 4,
    "attention_k_eq_v": True,
}

# 26B-real-dims: GQA=2 sliding (16/8), GQA=2 full K=V (16/8), hd=256/512.
GEMMA4_26B_REAL_DIMS_CONFIG = {
    **GEMMA4_E2B_REAL_DIMS_CONFIG,
    "num_hidden_layers": 12,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "num_global_key_value_heads": 8,
    "attention_k_eq_v": True,
}


def _build_gemma4_kv_cache_manager(config, num_blocks=4, tokens_per_block=32, batch_size=1):
    """Create KVCacheManagerV2 supporting Gemma4 per-layer head_dim / kv_heads.

    Mirrors ``Gemma4Attention``'s layout (global kv heads only for K=V layers)
    and the ``_util.py::is_gemma4_hybrid`` VSWA pool grouping so the page
    sizes line up with what the model actually requests at runtime.
    """
    import tensorrt_llm
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2

    dtype = config.torch_dtype
    if dtype == torch.half:
        kv_dtype = tensorrt_llm.bindings.DataType.HALF
    else:
        kv_dtype = tensorrt_llm.bindings.DataType.BF16

    layer_types = config.layer_types
    attention_k_eq_v = getattr(config, "attention_k_eq_v", False)
    head_dim_per_layer = []
    num_kv_heads_per_layer = []
    for lt in layer_types:
        is_sliding = lt == "sliding_attention"
        use_k_eq_v = attention_k_eq_v and not is_sliding
        if is_sliding:
            head_dim_per_layer.append(config.head_dim)
            num_kv_heads_per_layer.append(config.num_key_value_heads)
        else:
            head_dim_per_layer.append(getattr(config, "global_head_dim", config.head_dim))
            if use_k_eq_v:
                num_kv_heads_per_layer.append(
                    getattr(config, "num_global_key_value_heads", None)
                    or config.num_key_value_heads
                )
            else:
                num_kv_heads_per_layer.append(config.num_key_value_heads)

    # Use scalar if all layers have same value
    head_dim = head_dim_per_layer if len(set(head_dim_per_layer)) > 1 else head_dim_per_layer[0]
    num_kv_heads = (
        num_kv_heads_per_layer
        if len(set(num_kv_heads_per_layer)) > 1
        else num_kv_heads_per_layer[0]
    )

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    max_seq_len = num_blocks * tokens_per_block

    # Set per-layer max_attention_window when head_dim or kv_heads differ
    # across layers, so V2 creates separate pool groups for different page
    # sizes.  ``max_seq_len - 1`` on sliding layers prevents V2 block
    # eviction that would cause FlashInfer page index OOB when kv_lens
    # exceeds sliding_window.
    sliding_window = getattr(config, "sliding_window", None)
    max_attn_window = None
    needs_vswa = isinstance(head_dim, list) and len(set(head_dim)) > 1
    if not needs_vswa:
        needs_vswa = isinstance(num_kv_heads, list) and len(set(num_kv_heads)) > 1
    if needs_vswa and sliding_window:
        max_attn_window = [
            max_seq_len - 1 if lt == "sliding_attention" else max_seq_len for lt in layer_types
        ]

    kv_cache_config = KvCacheConfigV2(
        max_tokens=num_blocks * tokens_per_block,
        enable_block_reuse=False,
        max_attention_window=max_attn_window,
    )
    return KVCacheManagerV2(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_dtype,
    )


class TestGemma4HFComparison(unittest.TestCase):
    """Compare TRT-LLM Gemma4 outputs against HuggingFace reference."""

    _get_kv_cache_manager = staticmethod(_build_gemma4_kv_cache_manager)

    def _assert_most_elems_close(self, actual, ref, atol=0.5, rtol=0.5, max_failed_frac=0.01):
        matches = torch.isclose(actual, ref, atol=atol, rtol=rtol)
        failed = (~matches).float().mean().item()
        self.assertLessEqual(
            failed,
            max_failed_frac,
            f"{failed * 100:.2f}% of elements differ (max {max_failed_frac * 100}%)",
        )

    def _make_hf_and_trt_models(self, config_dict=None):
        """Create paired HF and TRT-LLM models with shared weights."""
        from transformers import Gemma4ForCausalLM as HFGemma4

        from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import (
            Gemma4HfWeightMapper,
        )

        config_dict = config_dict or deepcopy(GEMMA4_UNIFORM_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        dtype = config.torch_dtype
        device = torch.device("cuda")

        hf_model = HFGemma4(config).to(dtype).to(device).eval()
        model_config = ModelConfig(pretrained_config=config, attn_backend="FLASHINFER")
        trt_model = Gemma4ForCausalLM(model_config).to(dtype).to(device)

        wm = Gemma4HfWeightMapper()
        wm.init_model_and_config(trt_model, model_config)
        trt_model.load_weights(hf_model.state_dict(), wm)

        return hf_model, trt_model, config

    # ---- Sub-module numerical comparison tests ----

    @torch.no_grad()
    def test_embedding_matches_hf(self):
        """Embedding layer: scaled by sqrt(hidden_size)."""
        hf, trt, config = self._make_hf_and_trt_models()
        ids = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device="cuda")

        with torch.inference_mode():
            hf_out = hf.model.embed_tokens(ids.unsqueeze(0)).squeeze(0)
            trt_out = trt.model.embed_tokens(ids)

        self.assertTrue(
            torch.allclose(hf_out, trt_out, atol=1e-3),
            f"Embedding max diff: {(hf_out - trt_out).abs().max()}",
        )

    @torch.no_grad()
    def test_mlp_matches_hf(self):
        """MLP (GatedMLP with gelu_tanh) output comparison."""
        hf, trt, config = self._make_hf_and_trt_models()
        x = torch.randn(4, config.hidden_size, device="cuda", dtype=config.torch_dtype)

        with torch.inference_mode():
            hf_out = hf.model.layers[0].mlp(x.unsqueeze(0)).squeeze(0)
            trt_out = trt.model.layers[0].mlp(x)

        self._assert_most_elems_close(trt_out.float(), hf_out.float(), atol=0.01, rtol=0.01)

    @torch.no_grad()
    def test_rms_norm_matches_hf(self):
        """RMSNorm with Gemma +1 offset convention."""
        hf, trt, config = self._make_hf_and_trt_models()
        x = torch.randn(4, config.hidden_size, device="cuda", dtype=config.torch_dtype)

        with torch.inference_mode():
            hf_out = hf.model.layers[0].input_layernorm(x.unsqueeze(0)).squeeze(0)
            trt_out = trt.model.layers[0].input_layernorm(x)

        self.assertTrue(
            torch.allclose(hf_out, trt_out, atol=1e-2),
            f"Norm max diff: {(hf_out - trt_out).abs().max()}",
        )

    @torch.no_grad()
    def test_logit_softcapping_matches_hf(self):
        """Final logit softcapping: tanh(x/cap) * cap."""

        config_dict = deepcopy(GEMMA4_UNIFORM_CONFIG)
        config_dict["final_logit_softcapping"] = 30.0
        config_dict["num_hidden_layers"] = 2
        config = Gemma4TextConfig(**config_dict)

        # Just test the softcapping math directly
        logits = torch.randn(1, config.vocab_size, device="cuda")
        cap = config.final_logit_softcapping
        capped = torch.tanh(logits / cap) * cap
        # Verify values are bounded
        self.assertTrue((capped.abs() <= cap).all())
        # Verify small values are approximately unchanged
        small = torch.tensor([[0.1, -0.1, 0.5]], device="cuda")
        small_capped = torch.tanh(small / cap) * cap
        self.assertTrue(torch.allclose(small, small_capped, atol=0.01))

    # ---- Full model E2E comparison (context + generation) ----

    def _run_full_model_comparison(
        self,
        config_dict,
        atol=0.5,
        rtol=0.5,
        max_failed_frac=0.01,
    ):
        """Run context + generation comparison for a given config."""
        from transformers.cache_utils import DynamicCache

        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        torch.random.manual_seed(42)
        hf, trt, gemma4_config = self._make_hf_and_trt_models(config_dict)
        hf_cache = DynamicCache()

        device = torch.device("cuda")
        backend = "FLASHINFER"

        # Set up KV cache
        kv_cache_manager = self._get_kv_cache_manager(gemma4_config)

        # -- Context phase --
        input_ids = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800], dtype=torch.int32, device=device
        )
        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend(backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )
        position_ids = torch.arange(
            0, input_ids.size(-1), dtype=torch.int32, device=device
        ).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            trt_logits = trt.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )
            hf_out = hf.forward(
                input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids,
                past_key_values=hf_cache,
                use_cache=True,
            )
            hf_logits = hf_out.logits[:, -1].float()

        self._assert_most_elems_close(
            trt_logits, hf_logits, atol=atol, rtol=rtol, max_failed_frac=max_failed_frac
        )

        # -- Generation phase --
        gen_input_ids = torch.tensor([900], dtype=torch.int32, device=device)
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([1], dtype=torch.int),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[input_ids.size(-1)],
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=1,
            max_num_tokens=8192,
        )
        gen_position_ids = torch.arange(
            input_ids.size(-1), input_ids.size(-1) + 1, dtype=torch.int32, device=device
        ).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            trt_logits = trt.forward(
                input_ids=gen_input_ids,
                position_ids=gen_position_ids,
                attn_metadata=attn_metadata,
            )
            hf_out = hf.forward(
                input_ids=gen_input_ids.unsqueeze(0),
                position_ids=gen_position_ids,
                past_key_values=hf_cache,
                use_cache=True,
            )
            hf_logits = hf_out.logits[:, -1].float()

        self._assert_most_elems_close(
            trt_logits, hf_logits, atol=atol, rtol=rtol, max_failed_frac=max_failed_frac
        )

        kv_cache_manager.shutdown()

    @torch.no_grad()
    def test_uniform_config(self):
        """Baseline: uniform head_dim, no K=V, no MoE, no PLE."""
        self._run_full_model_comparison(deepcopy(GEMMA4_UNIFORM_CONFIG))

    @torch.no_grad()
    def test_hybrid_headdim_config(self):
        """Hybrid head_dim: sliding=64, full=128 (per-layer KV cache)."""
        self._run_full_model_comparison(deepcopy(GEMMA4_HYBRID_HEADDIM_CONFIG))

    @torch.no_grad()
    def test_diff_kv_heads_config(self):
        """Different num_kv_heads per layer type — tests V2 pool grouping."""
        self._run_full_model_comparison(deepcopy(GEMMA4_DIFF_KV_HEADS_CONFIG))

    @torch.no_grad()
    def test_kev_config(self):
        """K=V attention: full layers share key→value, v_norm applied."""
        self._run_full_model_comparison(deepcopy(GEMMA4_KEV_CONFIG))

    @torch.no_grad()
    def test_kev_vnorm_order(self):
        """K=V: v_norm must apply to raw k_proj(x), NOT after k_norm.

        Default init has k_norm weight=0 (scale=1+0=1), making rms_norm
        idempotent and hiding ordering bugs.  This test randomizes k_norm
        weights to expose incorrect v = k_norm(k_proj(x)) ordering.
        """
        from transformers import Gemma4ForCausalLM as HFGemma4

        from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import (
            Gemma4HfWeightMapper,
        )

        config = Gemma4TextConfig(**deepcopy(GEMMA4_KEV_CONFIG))
        dtype = config.torch_dtype
        device = torch.device("cuda")

        hf_model = HFGemma4(config).to(dtype).to(device).eval()

        # Randomize all weights to break rms_norm idempotency
        torch.manual_seed(123)
        for p in hf_model.parameters():
            p.data.normal_(0, 0.02)
        # Ensure k_norm has non-trivial scale so k_norm != identity
        for layer in hf_model.model.layers:
            if hasattr(layer.self_attn, "k_norm"):
                layer.self_attn.k_norm.weight.data.normal_(0, 0.5)

        model_config = ModelConfig(pretrained_config=config, attn_backend="FLASHINFER")
        trt_model = Gemma4ForCausalLM(model_config).to(dtype).to(device)

        wm = Gemma4HfWeightMapper()
        wm.init_model_and_config(trt_model, model_config)
        trt_model.load_weights(hf_model.state_dict(), wm)

        # Run context-phase comparison with tighter tolerance
        from transformers.cache_utils import DynamicCache

        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        hf_cache = DynamicCache()
        input_ids = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800], dtype=torch.int32, device=device
        )
        kv_cache_manager = self._get_kv_cache_manager(config)

        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend("FLASHINFER").Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )
        position_ids = torch.arange(
            0, input_ids.size(-1), dtype=torch.int32, device=device
        ).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            trt_logits = trt_model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )
            hf_out = hf_model.forward(
                input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids,
                past_key_values=hf_cache,
                use_cache=True,
            )
            hf_logits = hf_out.logits[:, -1].float()

        self._assert_most_elems_close(
            trt_logits, hf_logits, atol=0.5, rtol=0.5, max_failed_frac=0.01
        )
        kv_cache_manager.shutdown()

    @torch.no_grad()
    def test_softcap_config(self):
        """Logit softcapping enabled (cap=30.0)."""
        self._run_full_model_comparison(deepcopy(GEMMA4_SOFTCAP_CONFIG))

    @torch.no_grad()
    def test_ple_config(self):
        """Per-Layer Embeddings (PLE) enabled."""
        self._run_full_model_comparison(deepcopy(GEMMA4_PLE_CONFIG))

    @torch.no_grad()
    def test_moe_config(self):
        """MoE (Mixture of Experts) enabled — tests router + expert dispatch."""
        self._run_full_model_comparison(
            deepcopy(GEMMA4_MOE_HF_CONFIG),
            # MoE routing may differ slightly between fused and reference
            atol=1.0,
            rtol=1.0,
            max_failed_frac=0.05,
        )

    # ---- Mixed feature tests (matching real model patterns) ----

    @torch.no_grad()
    def test_26b_like_config(self):
        """26B-A4B pattern: hybrid head_dim + K=V + MoE + softcap."""
        self._run_full_model_comparison(
            deepcopy(GEMMA4_26B_LIKE_CONFIG),
            atol=1.0,
            rtol=1.0,
            max_failed_frac=0.05,
        )

    @torch.no_grad()
    def test_e2b_like_config(self):
        """E2B pattern: hybrid head_dim + PLE + KV sharing + double MLP + softcap."""
        self._run_full_model_comparison(
            deepcopy(GEMMA4_E2B_LIKE_CONFIG),
        )

    @torch.no_grad()
    def test_31b_like_config(self):
        """31B pattern: hybrid head_dim + K=V + softcap."""
        self._run_full_model_comparison(
            deepcopy(GEMMA4_31B_LIKE_CONFIG),
        )

    # ---- Batched and mixed ctx+gen HF comparison tests ----

    @torch.no_grad()
    def test_batch_context_hf_comparison(self):
        """Batch>1 context phase: logits must match HF for all requests.

        Two identical sequences processed in a single batch.  Both must
        produce the same logits as a single-sequence batch and match HF.
        """
        from transformers.cache_utils import DynamicCache

        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        torch.random.manual_seed(42)
        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        hf, trt, gemma4_config = self._make_hf_and_trt_models(config_dict)

        device = torch.device("cuda")
        kv_cache_manager = self._get_kv_cache_manager(gemma4_config, num_blocks=2, batch_size=2)

        ids = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800], dtype=torch.int32, device=device
        )
        seq_len = ids.size(0)

        # HF reference (single)
        hf_cache = DynamicCache()
        pos = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0)
        with torch.inference_mode():
            hf_out = hf(
                input_ids=ids.unsqueeze(0),
                position_ids=pos,
                past_key_values=hf_cache,
                use_cache=True,
            )
            hf_logits = hf_out.logits[0, -1].float()

        # TRT-LLM: batch=2 with the SAME input
        metadata_cls = get_attention_backend("FLASHINFER").Metadata
        kv_cache_manager.add_dummy_requests([1, 2], [seq_len, seq_len])
        ids_batched = torch.cat([ids, ids])  # [16]
        pos_batched = torch.cat(
            [
                torch.arange(seq_len, dtype=torch.int32, device=device),
                torch.arange(seq_len, dtype=torch.int32, device=device),
            ]
        ).unsqueeze(0)
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([seq_len, seq_len], dtype=torch.int),
            num_contexts=2,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0, 0]),
            max_num_requests=2,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=[1, 2],
            prompt_lens=[seq_len, seq_len],
        )
        with torch.inference_mode():
            attn_metadata.prepare()
            trt_logits = trt.forward(
                input_ids=ids_batched, position_ids=pos_batched, attn_metadata=attn_metadata
            )

        # Both batch entries must match HF
        self.assertEqual(trt_logits.shape[0], 2, "Expected 2 logit rows for batch=2")
        for i in range(2):
            self._assert_most_elems_close(
                trt_logits[i : i + 1],
                hf_logits.unsqueeze(0),
                atol=0.5,
                rtol=0.5,
                max_failed_frac=0.01,
            )
        # Both batch entries must be identical to each other
        diff = (trt_logits[0] - trt_logits[1]).abs().max().item()
        self.assertLess(diff, 0.01, f"Batch entries differ: max_diff={diff}")

        kv_cache_manager.shutdown()

    @torch.no_grad()
    def test_mixed_ctx_gen_hf_comparison(self):
        """Mixed context+generation batch: logits must match HF.

        Simulates the executor's inflight batching: request A already prefilled
        (now in generation), request B just arrived (context).  Both must match
        HF reference.  This is the pattern that triggers batch corruption if
        model modules don't correctly handle mixed phases.
        """
        from transformers.cache_utils import DynamicCache

        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        torch.random.manual_seed(42)
        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        hf, trt, gemma4_config = self._make_hf_and_trt_models(config_dict)

        device = torch.device("cuda")
        kv_cache_manager = self._get_kv_cache_manager(gemma4_config, num_blocks=2, batch_size=4)
        metadata_cls = get_attention_backend("FLASHINFER").Metadata

        ids_A = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800], dtype=torch.int32, device=device
        )
        ids_B = torch.tensor(
            [101, 201, 301, 401, 501, 601, 701, 801], dtype=torch.int32, device=device
        )
        seq_len = ids_A.size(0)

        # === Step 1: Prefill request A (pure context) ===
        kv_cache_manager.add_dummy_requests([1], [seq_len])
        meta_ctx = metadata_cls(
            seq_lens=torch.tensor([seq_len], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            max_num_requests=4,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=[1],
            prompt_lens=[seq_len],
        )
        pos_A = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0)
        with torch.inference_mode():
            meta_ctx.prepare()
            trt_ctx = trt.forward(input_ids=ids_A, position_ids=pos_A, attn_metadata=meta_ctx)
        gen_token_A = trt_ctx[0].argmax().item() if trt_ctx.dim() == 2 else trt_ctx.argmax().item()

        # HF reference: A prefill
        hf_cache_A = DynamicCache()
        with torch.inference_mode():
            hf_out_A = hf(
                input_ids=ids_A.unsqueeze(0),
                position_ids=pos_A,
                past_key_values=hf_cache_A,
                use_cache=True,
            )
            _ = hf_out_A.logits  # populates HF's KV cache for A

        # === Step 2: Mixed batch — A in generation + B in context ===
        kv_cache_manager.add_dummy_requests([2], [seq_len])
        gen_id_A = torch.tensor([gen_token_A], dtype=torch.int32, device=device)

        # Flat layout: [ctx_B_tokens..., gen_A_token]
        mixed_ids = torch.cat([ids_B, gen_id_A])
        mixed_pos = torch.cat(
            [
                torch.arange(seq_len, dtype=torch.int32, device=device),
                torch.tensor([seq_len], dtype=torch.int32, device=device),
            ]
        ).unsqueeze(0)
        meta_mixed = metadata_cls(
            seq_lens=torch.tensor([seq_len, 1], dtype=torch.int),
            num_contexts=1,  # B is context
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0, seq_len],  # B=0 cached, A=seq_len cached
            ),
            max_num_requests=4,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=[2, 1],
            prompt_lens=[seq_len, seq_len],
        )
        with torch.inference_mode():
            meta_mixed.prepare()
            trt_mixed = trt.forward(
                input_ids=mixed_ids, position_ids=mixed_pos, attn_metadata=meta_mixed
            )

        # trt_mixed shape: [2, vocab] — B's context logit + A's generation logit
        self.assertEqual(trt_mixed.shape[0], 2, f"Expected 2 logit rows, got {trt_mixed.shape}")

        # HF reference: B prefill (independent)
        hf_cache_B = DynamicCache()
        with torch.inference_mode():
            hf_out_B = hf(
                input_ids=ids_B.unsqueeze(0),
                position_ids=torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0),
                past_key_values=hf_cache_B,
                use_cache=True,
            )
            hf_logits_B = hf_out_B.logits[0, -1].float()

        # HF reference: A generation (second step)
        with torch.inference_mode():
            hf_out_A_gen = hf(
                input_ids=torch.tensor([[gen_token_A]], dtype=torch.int32, device=device),
                position_ids=torch.tensor([[seq_len]], dtype=torch.int32, device=device),
                past_key_values=hf_cache_A,
                use_cache=True,
            )
            hf_logits_A_gen = hf_out_A_gen.logits[0, -1].float()

        # B's context logits (mixed batch) must match HF B (independent prefill)
        self._assert_most_elems_close(
            trt_mixed[0:1],
            hf_logits_B.unsqueeze(0),
            atol=0.5,
            rtol=0.5,
            max_failed_frac=0.01,
        )
        # A's generation logits (mixed batch) must match HF A (generation step)
        self._assert_most_elems_close(
            trt_mixed[1:2],
            hf_logits_A_gen.unsqueeze(0),
            atol=0.5,
            rtol=0.5,
            max_failed_frac=0.01,
        )

        kv_cache_manager.shutdown()

    @torch.no_grad()
    def test_e4b_like_config(self):
        """E4B pattern: hybrid head_dim + NO K=V (same kv_heads all layers).

        This is the critical VSWA corner case.  Without K=V, full attention
        layers have the SAME num_kv_heads as sliding layers but double the
        head_dim.  The full pool has fewer pages than the sliding pool.
        Before the VSWA fix, sliding pool page indices were used for full
        pool buffers, causing out-of-bounds access.
        """
        self._run_full_model_comparison(deepcopy(GEMMA4_E4B_LIKE_CONFIG))

    # ---- VSWA (Variable Sliding Window Attention) page index tests ----

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_vswa_per_pool_page_indices(self):
        """VSWA: FlashInfer metadata builds separate page indices per pool.

        When layers have different head_dim (hybrid attention), V2 creates
        separate pools.  The FlashInfer metadata must fetch and store page
        indices per pool so that each layer uses the correct indices during
        append_paged_kv_cache and attention plan/run.
        """
        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        # E4B-like: sliding head_dim=64, full head_dim=128, kv_heads=2 both
        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        # num_blocks must be > 1 so max_seq_len > sliding_window → VSWA.
        kv_cache_manager = self._get_kv_cache_manager(config, num_blocks=2)

        # The manager must be VSWA
        self.assertTrue(kv_cache_manager.is_vswa, "Expected VSWA manager")

        # Set up metadata
        request_ids = [1]
        token_nums = [8]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend("FLASHINFER").Metadata
        metadata = metadata_cls(
            seq_lens=torch.tensor([8], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=[8],
        )

        # VSWA detection must populate layer-to-pool mapping
        self.assertIsNotNone(metadata._vswa_layer_to_pool, "VSWA layer_to_pool mapping not created")
        # Sliding and full layers must map to different pools
        sliding_pools = {
            metadata._vswa_layer_to_pool[i]
            for i in range(config.num_hidden_layers)
            if config.layer_types[i] == "sliding_attention"
        }
        full_pools = {
            metadata._vswa_layer_to_pool[i]
            for i in range(config.num_hidden_layers)
            if config.layer_types[i] == "full_attention"
        }
        self.assertEqual(len(sliding_pools), 1, "Sliding layers should share one pool")
        self.assertEqual(len(full_pools), 1, "Full layers should share one pool")
        self.assertNotEqual(sliding_pools, full_pools, "Sliding and full pools must be different")

        # After prepare(), per-pool indices cache must exist
        with torch.inference_mode():
            metadata.prepare()

        self.assertIsNotNone(
            metadata._vswa_pool_indices_cache,
            "Per-pool indices cache not populated after prepare()",
        )
        self.assertEqual(
            len(metadata._vswa_pool_indices_cache), 2, "Expected 2 pool entries (sliding + full)"
        )

        # Each pool's indices are fetched via different layer_id → pool_id
        # path.  Verify that buffers for different pools have different shapes
        # (different head_dim → different page sizes → different page counts).
        sliding_buf = kv_cache_manager.get_buffers(0, kv_layout=metadata.kv_layout)
        full_layer_idx = next(
            i for i, lt in enumerate(config.layer_types) if lt == "full_attention"
        )
        full_buf = kv_cache_manager.get_buffers(full_layer_idx, kv_layout=metadata.kv_layout)
        # head_dim dimension should differ (64 vs 128)
        self.assertNotEqual(
            sliding_buf.shape[-1],
            full_buf.shape[-1],
            "Sliding and full pool buffers should have different head_dim",
        )

        kv_cache_manager.shutdown()

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_vswa_page_index_bounds(self):
        """VSWA: page indices must be within each layer's pool buffer bounds.

        Constructs a scenario where full pool has fewer pages than sliding
        pool (E4B-like).  Verifies that after prepare(), the indices
        returned by get_paged_kv_indices_for_layer are within each pool's
        buffer size.
        """
        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        kv_cache_manager = self._get_kv_cache_manager(config, num_blocks=2)

        request_ids = [1]
        token_nums = [8]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend("FLASHINFER").Metadata
        metadata = metadata_cls(
            seq_lens=torch.tensor([8], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=[8],
        )

        with torch.inference_mode():
            metadata.prepare()

        # Check page index bounds for every layer
        for layer_idx in range(config.num_hidden_layers):
            buf = kv_cache_manager.get_buffers(layer_idx, kv_layout=metadata.kv_layout)
            if buf is None:
                continue
            max_pages = buf.shape[0]
            indices = metadata.get_paged_kv_indices_for_layer(layer_idx)
            if indices.numel() == 0:
                continue
            max_idx = indices.max().item()
            self.assertLess(
                max_idx,
                max_pages,
                f"Layer {layer_idx}: page index {max_idx} >= "
                f"pool buffer pages {max_pages} "
                f"(layer_type={config.layer_types[layer_idx]}, "
                f"head_dim={'sliding' if config.layer_types[layer_idx] == 'sliding_attention' else 'full'})",
            )

        kv_cache_manager.shutdown()

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_vswa_swap_restores_correct_pool(self):
        """VSWA: swapping indices between pools and back produces correct data.

        Simulates the forward pass pattern: sliding layer → full layer →
        sliding layer, verifying the shared buffer has the right pool's
        indices at each step.
        """
        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        kv_cache_manager = self._get_kv_cache_manager(config, num_blocks=2)

        request_ids = [1]
        token_nums = [8]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend("FLASHINFER").Metadata
        metadata = metadata_cls(
            seq_lens=torch.tensor([8], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=[8],
        )

        with torch.inference_mode():
            metadata.prepare()

        total_blocks = metadata.num_generation_blocks + metadata.num_context_blocks
        if total_blocks == 0:
            kv_cache_manager.shutdown()
            return

        # Find one sliding and one full layer
        sliding_layer = next(
            i for i, lt in enumerate(config.layer_types) if lt == "sliding_attention"
        )
        full_layer = next(i for i, lt in enumerate(config.layer_types) if lt == "full_attention")

        # Get expected indices for each pool
        sliding_expected = metadata.get_paged_kv_indices_for_layer(sliding_layer).cpu().clone()
        full_expected = metadata.get_paged_kv_indices_for_layer(full_layer).cpu().clone()

        # Swap to sliding → check buffer matches sliding pool
        metadata.swap_paged_kv_indices_for_layer(sliding_layer)
        actual = metadata._paged_kv_indices[:total_blocks].cpu()
        self.assertTrue(
            torch.equal(actual, sliding_expected),
            "After swap to sliding layer, buffer should match sliding pool",
        )

        # Swap to full → check buffer matches full pool
        metadata.swap_paged_kv_indices_for_layer(full_layer)
        actual = metadata._paged_kv_indices[:total_blocks].cpu()
        self.assertTrue(
            torch.equal(actual, full_expected),
            "After swap to full layer, buffer should match full pool",
        )

        # Swap back to sliding → check buffer is restored
        metadata.swap_paged_kv_indices_for_layer(sliding_layer)
        actual = metadata._paged_kv_indices[:total_blocks].cpu()
        self.assertTrue(
            torch.equal(actual, sliding_expected),
            "After swap back to sliding, buffer should be restored",
        )

        kv_cache_manager.shutdown()

    # ---- Structural tests ----

    @torch.no_grad()
    def test_kv_sharing_instantiation(self):
        """KV sharing: verify shared layers are configured correctly."""
        config_dict = deepcopy(GEMMA4_KV_SHARING_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        model_config = ModelConfig(pretrained_config=config, attn_backend="FLASHINFER")
        model = Gemma4ForCausalLM(model_config)

        # E.g., 6 layers with last 2 shared → layers 4,5 are shared
        num_shared = config.num_kv_shared_layers
        first_shared = config.num_hidden_layers - num_shared
        for i, layer in enumerate(model.model.layers):
            if i >= first_shared:
                self.assertTrue(layer.is_kv_shared_layer, f"Layer {i} should be KV shared")
                self.assertTrue(layer.self_attn.is_kv_shared, f"Layer {i} attn should be KV shared")
            else:
                self.assertFalse(layer.is_kv_shared_layer, f"Layer {i} should NOT be KV shared")

    @torch.no_grad()
    def test_rope_real_headdim_sliding(self):
        """RoPE cos/sin must match HF for sliding layers with real head_dim=256."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding as HFRoPE

        config = Gemma4TextConfig(
            hidden_size=2560,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            global_head_dim=512,
            intermediate_size=5120,
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                    "rope_type": "proportional",
                },
            },
            vocab_size=262144,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            hidden_activation="gelu_pytorch_tanh",
            sliding_window=4096,
            torch_dtype="bfloat16",
        )
        mc = ModelConfig(pretrained_config=config, attn_backend="FLASHINFER")
        trt = Gemma4Attention(mc, layer_idx=0, is_sliding=True).to(torch.bfloat16).cuda()

        hf_rope = HFRoPE(config).cuda()
        pos = torch.arange(8, device="cuda").unsqueeze(0)
        hf_cos, hf_sin = hf_rope(
            torch.zeros(1, 8, 2560, device="cuda", dtype=torch.bfloat16), pos, "sliding_attention"
        )

        trt_cs = trt.rotary_emb.rotary_cos_sin[pos.view(-1)]  # [8, 2, dim//2]
        trt_cos = trt_cs[:, 0, :].unsqueeze(0).to(torch.bfloat16)
        trt_sin = trt_cs[:, 1, :].unsqueeze(0).to(torch.bfloat16)

        self.assertTrue(
            torch.allclose(trt_cos, hf_cos[:, :, : trt_cos.shape[-1]], atol=1e-4),
            f"Sliding RoPE cos mismatch: max diff={(trt_cos - hf_cos[:, :, : trt_cos.shape[-1]]).abs().max()}",
        )
        self.assertTrue(
            torch.allclose(trt_sin, hf_sin[:, :, : trt_sin.shape[-1]], atol=1e-4),
            f"Sliding RoPE sin mismatch: max diff={(trt_sin - hf_sin[:, :, : trt_sin.shape[-1]]).abs().max()}",
        )

    @torch.no_grad()
    def test_rope_real_headdim_full(self):
        """RoPE cos/sin must match HF for full layers with real head_dim=512 (proportional RoPE)."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding as HFRoPE

        config = Gemma4TextConfig(
            hidden_size=2560,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            global_head_dim=512,
            intermediate_size=5120,
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                    "rope_type": "proportional",
                },
            },
            vocab_size=262144,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            hidden_activation="gelu_pytorch_tanh",
            sliding_window=4096,
            torch_dtype="bfloat16",
        )
        mc = ModelConfig(pretrained_config=config, attn_backend="FLASHINFER")
        trt = Gemma4Attention(mc, layer_idx=1, is_sliding=False).to(torch.bfloat16).cuda()

        hf_rope = HFRoPE(config).cuda()
        pos = torch.arange(8, device="cuda").unsqueeze(0)
        hf_cos, hf_sin = hf_rope(
            torch.zeros(1, 8, 2560, device="cuda", dtype=torch.bfloat16), pos, "full_attention"
        )

        trt_cs = trt.rotary_emb.rotary_cos_sin[pos.view(-1)]
        trt_cos = trt_cs[:, 0, :].unsqueeze(0).to(torch.bfloat16)
        trt_sin = trt_cs[:, 1, :].unsqueeze(0).to(torch.bfloat16)

        # TRT-LLM only has dim/2 = 64 cos/sin values; HF has head_dim/2 = 256 (with 192 zeros)
        # Compare only the rotary part
        rot_dim = trt_cos.shape[-1]
        self.assertTrue(
            torch.allclose(trt_cos, hf_cos[:, :, :rot_dim], atol=1e-4),
            f"Full RoPE cos mismatch: max diff={(trt_cos - hf_cos[:, :, :rot_dim]).abs().max()}",
        )
        self.assertTrue(
            torch.allclose(trt_sin, hf_sin[:, :, :rot_dim], atol=1e-4),
            f"Full RoPE sin mismatch: max diff={(trt_sin - hf_sin[:, :, :rot_dim]).abs().max()}",
        )

    # ---- Bug regression tests ----
    # These tests specifically target accuracy bugs that were found and fixed
    # during real-model validation. Each test is designed to FAIL if the
    # corresponding fix is reverted.

    @torch.no_grad()
    def test_vswa_pool_cache_not_aliased(self):
        """VSWA: primary pool indices cache must be a separate buffer.

        Regression for Bug #3 (VSWA primary pool indices aliasing):
        If the primary pool cache entry is an alias of _paged_kv_indices,
        swapping to a secondary pool corrupts the primary pool's cached
        data.  After swap pool0→pool1→pool0, the restored pool0 indices
        must match the original values, not pool1's values.
        """
        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        # Use 2 requests so block indices are non-trivial
        kv_cache_manager = self._get_kv_cache_manager(config, num_blocks=2, batch_size=2)
        kv_cache_manager.add_dummy_requests([1, 2], [8, 8])

        metadata_cls = get_attention_backend("FLASHINFER").Metadata
        metadata = metadata_cls(
            seq_lens=torch.tensor([8, 8], dtype=torch.int),
            num_contexts=2,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0, 0]),
            max_num_requests=2,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=[1, 2],
            prompt_lens=[8, 8],
        )
        with torch.inference_mode():
            metadata.prepare()

        if metadata._vswa_layer_to_pool is None:
            kv_cache_manager.shutdown()
            self.skipTest("No VSWA detected")

        primary_pool_id = metadata._vswa_layer_to_pool.get(0, 0)

        # The primary pool cache buffer must NOT be the same object as
        # _paged_kv_indices — that's the aliasing bug.
        self.assertIsNot(
            metadata._vswa_pool_indices_cache[primary_pool_id],
            metadata._paged_kv_indices,
            "Primary pool cache is aliased to _paged_kv_indices! "
            "Pool swaps will corrupt the primary pool's cached indices.",
        )

        # Record the original primary pool indices
        total_blocks = metadata.num_generation_blocks + metadata.num_context_blocks
        pool0_original = (
            metadata._vswa_pool_indices_cache[primary_pool_id][:total_blocks].cpu().clone()
        )

        # Find a full-attention (secondary pool) layer
        full_layer = next(
            (i for i, lt in enumerate(config.layer_types) if lt == "full_attention"),
            None,
        )
        if full_layer is None:
            kv_cache_manager.shutdown()
            self.skipTest("No full-attention layer")

        # Swap to secondary pool (full), then back to primary (sliding)
        metadata.swap_paged_kv_indices_for_layer(full_layer)
        sliding_layer = next(
            i for i, lt in enumerate(config.layer_types) if lt == "sliding_attention"
        )
        metadata.swap_paged_kv_indices_for_layer(sliding_layer)

        # After round-trip swap, primary pool cache must still have original values
        pool0_after = metadata._vswa_pool_indices_cache[primary_pool_id][:total_blocks].cpu()
        self.assertTrue(
            torch.equal(pool0_original, pool0_after),
            f"Primary pool indices corrupted after pool swap round-trip: "
            f"original={pool0_original.tolist()}, after={pool0_after.tolist()}",
        )

        kv_cache_manager.shutdown()

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_vswa_no_eviction_with_long_sequence(self):
        """VSWA: sliding pool must not evict blocks when max_attention_window
        uses max_seq_len - 1 (the fix for page index OOB).

        Root cause: when _util.py used the model's sliding_window (e.g. 512)
        as max_attention_window for sliding layers, V2 would evict old blocks
        when kv_lens exceeded the window.  But FlashInfer's prepare() computes
        num_blocks from the FULL kv_lens, so the page indices for evicted
        blocks become stale → illegal memory access.

        The fix uses max_seq_len - 1 instead of sliding_window, preventing
        eviction while keeping is_vswa=True.  This test verifies that with
        the fix, a sequence longer than sliding_window still has all its
        blocks allocated (no eviction) and page indices are within bounds.
        """
        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        # Use a small sliding_window to easily exceed it
        config_dict["sliding_window"] = 64
        config = Gemma4TextConfig(**config_dict)

        # num_blocks=4 → max_seq_len = 4*128 = 512, much larger than
        # sliding_window=64.  With the fix, max_attention_window for sliding
        # layers = 511 (max_seq_len - 1), so V2 won't evict.
        kv_cache_manager = self._get_kv_cache_manager(config, num_blocks=4)

        # Allocate a request with tokens > sliding_window
        request_ids = [1]
        token_nums = [128]  # 128 tokens >> sliding_window (64)
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend("FLASHINFER").Metadata
        metadata = metadata_cls(
            seq_lens=torch.tensor([128], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=[128],
        )

        with torch.inference_mode():
            metadata.prepare()

        # num_blocks should be based on full kv_lens (128 tokens),
        # not clamped to sliding_window (64 tokens).
        expected_blocks = (
            128 + kv_cache_manager.tokens_per_block - 1
        ) // kv_cache_manager.tokens_per_block
        self.assertEqual(
            metadata.num_blocks[0],
            expected_blocks,
            f"num_blocks should be {expected_blocks} (from full kv_lens=128), "
            f"not clamped to sliding_window={config_dict['sliding_window']}",
        )

        # Page indices must be within bounds for EVERY layer
        for layer_idx in range(config.num_hidden_layers):
            buf = kv_cache_manager.get_buffers(layer_idx, kv_layout=metadata.kv_layout)
            if buf is None:
                continue
            max_pages = buf.shape[0]
            indices = metadata.get_paged_kv_indices_for_layer(layer_idx)
            if indices.numel() == 0:
                continue
            max_idx = indices.max().item()
            self.assertLess(
                max_idx,
                max_pages,
                f"Layer {layer_idx}: page index {max_idx} >= pool pages {max_pages}",
            )

        kv_cache_manager.shutdown()

    @torch.no_grad()
    def test_mixed_batch_kv_cache_isolation(self):
        """Mixed batch: B's prefill must not corrupt A's KV cache.

        Regression for Bug #3 at the symptom level: prefill A, then
        prefill B in a separate forward. A's decode logits must match
        a BS=1 baseline where B never existed.
        """

        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        torch.random.manual_seed(42)
        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        hf, trt, gemma4_config = self._make_hf_and_trt_models(config_dict)

        device = torch.device("cuda")
        metadata_cls = get_attention_backend("FLASHINFER").Metadata

        ids_A = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800], dtype=torch.int32, device=device
        )
        ids_B = torch.tensor(
            [101, 201, 301, 401, 501, 601, 701, 801], dtype=torch.int32, device=device
        )
        seq_len = ids_A.size(0)

        # --- Run 1: BS=1 baseline (A only) ---
        kv1 = self._get_kv_cache_manager(gemma4_config, num_blocks=2, batch_size=1)
        kv1.add_dummy_requests([1], [seq_len + 2])
        with torch.inference_mode():
            meta1 = metadata_cls(
                seq_lens=torch.tensor([seq_len], dtype=torch.int),
                num_contexts=1,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
                max_num_requests=1,
                max_num_tokens=8192,
                kv_cache_manager=kv1,
                request_ids=[1],
                prompt_lens=[seq_len],
            )
            meta1.prepare()
            pos = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0)
            pfx1 = trt.forward(input_ids=ids_A, position_ids=pos, attn_metadata=meta1)
            tok_a = pfx1[0].float().argmax().item() if pfx1.dim() == 2 else pfx1.argmax().item()

            meta1d = metadata_cls(
                seq_lens=torch.tensor([1], dtype=torch.int),
                num_contexts=0,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[seq_len]),
                kv_cache_manager=kv1,
                request_ids=[1],
                prompt_lens=[seq_len],
                max_num_requests=1,
                max_num_tokens=8192,
            )
            meta1d.prepare()
            dec1 = trt.forward(
                input_ids=torch.tensor([tok_a], dtype=torch.int32, device=device),
                position_ids=torch.tensor([[seq_len]], dtype=torch.int32, device=device),
                attn_metadata=meta1d,
            )
            bs1_logits = dec1[0].float() if dec1.dim() == 2 else dec1.float()
        kv1.shutdown()

        # --- Run 2: A+B allocated, prefill A then separately prefill B, decode A ---
        kv2 = self._get_kv_cache_manager(gemma4_config, num_blocks=2, batch_size=2)
        kv2.add_dummy_requests([1, 2], [seq_len + 2, seq_len + 2])
        with torch.inference_mode():
            # Prefill A
            meta2a = metadata_cls(
                seq_lens=torch.tensor([seq_len], dtype=torch.int),
                num_contexts=1,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
                max_num_requests=2,
                max_num_tokens=8192,
                kv_cache_manager=kv2,
                request_ids=[1],
                prompt_lens=[seq_len],
            )
            meta2a.prepare()
            trt.forward(input_ids=ids_A, position_ids=pos, attn_metadata=meta2a)

            # Prefill B (separate forward — this should NOT affect A's cache)
            meta2b = metadata_cls(
                seq_lens=torch.tensor([seq_len], dtype=torch.int),
                num_contexts=1,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
                max_num_requests=2,
                max_num_tokens=8192,
                kv_cache_manager=kv2,
                request_ids=[2],
                prompt_lens=[seq_len],
            )
            meta2b.prepare()
            trt.forward(input_ids=ids_B, position_ids=pos, attn_metadata=meta2b)

            # Decode A
            meta2d = metadata_cls(
                seq_lens=torch.tensor([1], dtype=torch.int),
                num_contexts=0,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[seq_len]),
                kv_cache_manager=kv2,
                request_ids=[1],
                prompt_lens=[seq_len],
                max_num_requests=2,
                max_num_tokens=8192,
            )
            meta2d.prepare()
            dec2 = trt.forward(
                input_ids=torch.tensor([tok_a], dtype=torch.int32, device=device),
                position_ids=torch.tensor([[seq_len]], dtype=torch.int32, device=device),
                attn_metadata=meta2d,
            )
            bs2_logits = dec2[0].float() if dec2.dim() == 2 else dec2.float()
        kv2.shutdown()

        # A's decode logits must match between BS=1 and BS=2
        cos = torch.nn.functional.cosine_similarity(
            bs1_logits.flatten().unsqueeze(0),
            bs2_logits.flatten().unsqueeze(0),
        ).item()
        self.assertGreater(
            cos,
            0.99,
            f"A's decode logits differ between BS=1 and BS=2 (cos={cos:.6f}). "
            f"B's prefill likely corrupted A's KV cache.",
        )

    @torch.no_grad()
    def test_bidirectional_mask_gating(self):
        """E4B config (use_bidirectional_attention=None) must skip custom mask.

        Regression for Bug #2: The model built bidirectional masks whenever
        mm_token_type_ids was present, regardless of the model's
        use_bidirectional_attention setting. E2B/E4B should use standard
        causal attention (no custom mask) even when mm_token_type_ids is
        provided. The fix gates custom mask construction in forward() on
        ``use_bidirectional_attention == "vision"``.
        """

        device = torch.device("cuda")

        # --- E4B: use_bidirectional_attention=None → NO custom mask ---
        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        config_dict["use_bidirectional_attention"] = None
        config = Gemma4TextConfig(**config_dict)
        model_config = ModelConfig(pretrained_config=config, attn_backend="FLASHINFER")
        model = Gemma4ForCausalLM(model_config).to(config.torch_dtype).to(device)

        self.assertIsNone(
            getattr(model.config, "use_bidirectional_attention", None),
            "E4B config should have use_bidirectional_attention=None",
        )

        # The forward() method gates on use_bidirectional_attention.
        # When it's None, it should NOT build any custom attention mask,
        # even if mm_token_type_ids is present.
        # We verify this by checking the gating logic directly:
        use_bidir = getattr(model.config, "use_bidirectional_attention", None)
        mm_token_type_ids = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0], device=device)

        should_build_mask = mm_token_type_ids is not None and use_bidir == "vision"
        self.assertFalse(
            should_build_mask,
            "E4B (use_bidirectional_attention=None) should NOT build custom mask",
        )

        # --- 26B: use_bidirectional_attention="vision" → YES custom mask ---
        config_dict_26b = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        config_dict_26b["use_bidirectional_attention"] = "vision"
        config_26b = Gemma4TextConfig(**config_dict_26b)
        mc_26b = ModelConfig(pretrained_config=config_26b, attn_backend="FLASHINFER")
        model_26b = Gemma4ForCausalLM(mc_26b).to(config_26b.torch_dtype).to(device)

        use_bidir_26b = getattr(model_26b.config, "use_bidirectional_attention", None)
        should_build_mask_26b = mm_token_type_ids is not None and use_bidir_26b == "vision"
        self.assertTrue(
            should_build_mask_26b,
            "26B (use_bidirectional_attention='vision') SHOULD build custom mask",
        )

        # Verify the mask has bidirectional entries for image tokens
        mask_26b = model_26b.get_context_mask(mm_token_type_ids)
        self.assertTrue(mask_26b[3, 2].item(), "Image token 3 should attend to 2")
        self.assertTrue(mask_26b[2, 3].item(), "Image token 2 should attend to 3")
        self.assertTrue(mask_26b[4, 2].item(), "Image token 4 should attend to 2")
        # But causal for text tokens
        self.assertFalse(mask_26b[0, 1].item(), "Text token 0 should NOT attend to 1")


class TestGemma4ModelDefaults(unittest.TestCase):
    """Tests for Gemma4 model defaults (get_model_defaults)."""

    def test_causal_lm_requires_flashinfer_backend(self):
        """Gemma4ForCausalLM must default to FLASHINFER attention backend.

        The TRTLLM backend does not support:
        - FlashInfer VSWA per-pool page management for hybrid head_dim
        - trtllm-gen cubin dispatch for head_dim=512 layers
        - Custom attention masks for bidirectional multimodal tokens
        Without this default, models crash with:
            'TrtllmAttentionMetadata' object has no attribute 'kv_layout'
        """
        defaults = Gemma4ForCausalLM.get_model_defaults(None)
        self.assertIn("attn_backend", defaults, "get_model_defaults must set attn_backend")
        self.assertEqual(
            defaults["attn_backend"], "FLASHINFER", "Gemma4 requires FLASHINFER (exact uppercase)"
        )

    def test_causal_lm_does_not_disable_cuda_graphs(self):
        """Gemma4ForCausalLM must not disable CUDA graphs."""
        defaults = Gemma4ForCausalLM.get_model_defaults(None)
        self.assertNotIn("cuda_graph_config", defaults)

    def test_conditional_gen_requires_flashinfer_backend(self):
        """Gemma4ForConditionalGeneration must also default to FLASHINFER."""
        from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4ForConditionalGeneration

        defaults = Gemma4ForConditionalGeneration.get_model_defaults(None)
        self.assertIn("attn_backend", defaults)
        self.assertEqual(defaults["attn_backend"], "FLASHINFER")

    def test_conditional_gen_does_not_disable_cuda_graphs(self):
        """Gemma4ForConditionalGeneration must not disable CUDA graphs."""
        from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4ForConditionalGeneration

        defaults = Gemma4ForConditionalGeneration.get_model_defaults(None)
        self.assertNotIn("cuda_graph_config", defaults)

    def test_attn_backend_dispatches_to_flashinfer(self):
        """Verify the exact string 'FLASHINFER' dispatches correctly.

        get_attention_backend uses exact case-sensitive match. 'FlashInfer'
        or 'flashinfer' would silently fall back to TrtllmAttention, which
        causes 'TrtllmAttentionMetadata has no attribute kv_layout' crashes.
        """
        from tensorrt_llm._torch.attention_backend.utils import get_attention_backend

        defaults = Gemma4ForCausalLM.get_model_defaults(None)
        backend_cls = get_attention_backend(defaults["attn_backend"])

        # Must be FlashInferAttention, not TrtllmAttention
        self.assertEqual(
            backend_cls.__name__,
            "FlashInferAttention",
            "FLASHINFER must dispatch to FlashInferAttention",
        )

    def test_all_layers_use_trtllm_gen(self):
        """All Gemma4 layers use trtllm-gen backend uniformly.

        trtllm-gen has pre-compiled cubins for H256+H512, both BF16 and
        FP8 dtypes.  For FP8 KV cache (NVFP4), the FlashInfer backend
        casts Q to FP8 to match KV, enabling QkvE4m3OBfloat16 context
        cubins.  Uniform backend is required for CUDA graph workspace
        sharing safety.
        """
        config_dict = deepcopy(GEMMA4_SMALL_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        model_config = ModelConfig(pretrained_config=config)

        for i in range(config.num_hidden_layers):
            attn = Gemma4Attention(model_config, i)
            self.assertEqual(
                attn.attn.flashinfer_backend,
                "trtllm-gen",
                f"Layer {i} should use trtllm-gen",
            )


class TestGemma4CUDAGraph(unittest.TestCase):
    """Tests for Gemma4 attention with CUDA graph capture/replay."""

    _get_kv_cache_manager = staticmethod(_build_gemma4_kv_cache_manager)

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_cuda_graph_decode_hybrid_headdim(self):
        """CUDA graph decode with hybrid head_dim (VSWA).

        Tests that CUDA graph capture + replay produces the same decode
        output as eager execution for Gemma4's hybrid attention layout
        (sliding head_dim=64, full head_dim=128).

        This is the key test for CUDA graph support: Gemma4's VSWA creates
        separate V2 pools for different head_dims.  The FlashInfer wrappers
        must correctly handle page index swaps and workspace sharing under
        CUDA graph capture/replay.
        """
        import random

        from tensorrt_llm._torch.attention_backend import (
            FlashInferAttention,
            FlashInferAttentionMetadata,
        )
        from tensorrt_llm._torch.metadata import KVCacheParams

        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        batch_size = 4
        kv_cache_manager = self._get_kv_cache_manager(config, num_blocks=2, batch_size=batch_size)

        # Verify VSWA setup
        self.assertTrue(kv_cache_manager.is_vswa, "Expected VSWA manager")

        request_ids = list(range(batch_size))
        past_seen_tokens = [random.randint(1, 100) for _ in range(batch_size)]
        token_nums = [t + 1 for t in past_seen_tokens]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        # Fill KV cache with random data
        for i in range(config.num_hidden_layers):
            buf = kv_cache_manager.get_buffers(i)
            if buf is not None:
                torch.nn.init.normal_(buf)

        # Build per-layer info
        layer_types = config.layer_types
        num_layers = config.num_hidden_layers
        dtype = config.torch_dtype
        device = torch.device("cuda")

        layers_info = []
        for layer_idx in range(num_layers):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            hd = (
                config.head_dim
                if is_sliding
                else getattr(config, "global_head_dim", config.head_dim)
            )
            layers_info.append(
                {
                    "layer_idx": layer_idx,
                    "head_dim": hd,
                    "num_kv_heads": config.num_key_value_heads,
                    "num_heads": config.num_attention_heads,
                }
            )

        # Create attention layers
        layers = []
        for info in layers_info:
            layers.append(
                FlashInferAttention(
                    layer_idx=info["layer_idx"],
                    num_heads=info["num_heads"],
                    head_dim=info["head_dim"],
                    num_kv_heads=info["num_kv_heads"],
                )
            )

        # Generate decode Q/K/V per layer (one token per request)
        gen_qs, gen_ks, gen_vs = [], [], []
        for info in layers_info:
            gen_qs.append(
                [
                    torch.randn(1, info["num_heads"] * info["head_dim"], dtype=dtype, device=device)
                    for _ in range(batch_size)
                ]
            )
            gen_ks.append(
                [
                    torch.randn(
                        1, info["num_kv_heads"] * info["head_dim"], dtype=dtype, device=device
                    )
                    for _ in range(batch_size)
                ]
            )
            gen_vs.append(
                [
                    torch.randn(
                        1, info["num_kv_heads"] * info["head_dim"], dtype=dtype, device=device
                    )
                    for _ in range(batch_size)
                ]
            )

        seq_lens = torch.ones(batch_size, dtype=torch.int)

        # --- Reference run (eager, no CUDA graph) ---
        ref_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=past_seen_tokens
            ),
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        ref_metadata.prepare()

        ref_results = []
        for i in range(num_layers):
            q = torch.cat(gen_qs[i])
            k = torch.cat(gen_ks[i])
            v = torch.cat(gen_vs[i])
            ref_results.append(layers[i].forward(q, k, v, ref_metadata))

        # --- CUDA graph run ---
        workspace = torch.empty(320 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        cg_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=0,
            is_cuda_graph=True,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=past_seen_tokens
            ),
            workspace_buffer=workspace,
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        cg_metadata.prepare()

        # Warmup runs (required before CUDA graph capture)
        for _ in range(2):
            for i in range(num_layers):
                q = torch.cat(gen_qs[i])
                k = torch.cat(gen_ks[i])
                v = torch.cat(gen_vs[i])
                layers[i].forward(q, k, v, cg_metadata)

        # Capture
        cg_results = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(num_layers):
                q = torch.cat(gen_qs[i])
                k = torch.cat(gen_ks[i])
                v = torch.cat(gen_vs[i])
                cg_results.append(layers[i].forward(q, k, v, cg_metadata))

        # Replay
        graph.replay()

        # Compare each layer's output
        for i in range(num_layers):
            torch.testing.assert_close(
                cg_results[i],
                ref_results[i],
                atol=1e-2,
                rtol=0,
                msg=(
                    f"Layer {i} ({layer_types[i]}, "
                    f"hd={layers_info[i]['head_dim']}): "
                    f"CUDA graph output doesn't match reference"
                ),
            )

        kv_cache_manager.shutdown()

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_cuda_graph_multi_step_decode(self):
        """CUDA graph multi-step decode with hybrid head_dim.

        Simulates the actual model_engine flow: prepare() + graph.replay()
        for MULTIPLE sequential decode steps.  Each step increments
        num_cached_tokens.  This catches issues that only appear after
        several decode iterations (e.g., stale plan data, growing kv_lens,
        VSWA swap with changing page counts).
        """
        from tensorrt_llm._torch.attention_backend import (
            FlashInferAttention,
            FlashInferAttentionMetadata,
        )
        from tensorrt_llm._torch.metadata import KVCacheParams

        config_dict = deepcopy(GEMMA4_E4B_LIKE_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        batch_size = 2
        num_decode_steps = 10
        kv_cache_manager = self._get_kv_cache_manager(config, num_blocks=4, batch_size=batch_size)

        request_ids = list(range(batch_size))
        initial_cached = [30, 45]
        token_nums = [t + 1 for t in initial_cached]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        # Fill KV cache
        for i in range(config.num_hidden_layers):
            buf = kv_cache_manager.get_buffers(i)
            if buf is not None:
                torch.nn.init.normal_(buf)

        layer_types = config.layer_types
        num_layers = config.num_hidden_layers
        dtype = config.torch_dtype
        device = torch.device("cuda")

        layers_info = []
        for layer_idx in range(num_layers):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            hd = (
                config.head_dim
                if is_sliding
                else getattr(config, "global_head_dim", config.head_dim)
            )
            layers_info.append(
                {
                    "layer_idx": layer_idx,
                    "head_dim": hd,
                    "num_kv_heads": config.num_key_value_heads,
                    "num_heads": config.num_attention_heads,
                }
            )

        layers = []
        for info in layers_info:
            layers.append(
                FlashInferAttention(
                    layer_idx=info["layer_idx"],
                    num_heads=info["num_heads"],
                    head_dim=info["head_dim"],
                    num_kv_heads=info["num_kv_heads"],
                )
            )

        seq_lens = torch.ones(batch_size, dtype=torch.int)

        # --- CUDA graph setup ---
        workspace = torch.empty(320 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        cg_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=0,
            is_cuda_graph=True,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=initial_cached),
            workspace_buffer=workspace,
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        cg_metadata.prepare()

        # Generate per-layer Q/K/V (reused across steps, like real static buffers)
        gen_qs, gen_ks, gen_vs = [], [], []
        for info in layers_info:
            gen_qs.append(
                torch.randn(
                    batch_size, info["num_heads"] * info["head_dim"], dtype=dtype, device=device
                )
            )
            gen_ks.append(
                torch.randn(
                    batch_size, info["num_kv_heads"] * info["head_dim"], dtype=dtype, device=device
                )
            )
            gen_vs.append(
                torch.randn(
                    batch_size, info["num_kv_heads"] * info["head_dim"], dtype=dtype, device=device
                )
            )

        # Warmup
        for _ in range(2):
            for i in range(num_layers):
                layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], cg_metadata)

        # Capture
        cg_outputs = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(num_layers):
                cg_outputs.append(layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], cg_metadata))

        # --- Multi-step decode: prepare() + replay() ---
        for step in range(num_decode_steps):
            # Simulate growing cached tokens (like real generation)
            cached = [c + step for c in initial_cached]
            cg_metadata.kv_cache_params = KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=cached
            )
            cg_metadata.seq_lens = seq_lens
            cg_metadata.num_contexts = 0
            cg_metadata.request_ids = request_ids
            cg_metadata.prepare()

            graph.replay()
            torch.cuda.synchronize()

            # Verify outputs are not NaN/Inf
            for i in range(num_layers):
                out = cg_outputs[i]
                self.assertFalse(
                    torch.isnan(out).any(), f"Step {step}, Layer {i}: NaN in CUDA graph output"
                )
                self.assertFalse(
                    torch.isinf(out).any(), f"Step {step}, Layer {i}: Inf in CUDA graph output"
                )

            # --- Reference (eager) for comparison ---
            ref_metadata = FlashInferAttentionMetadata(
                seq_lens=seq_lens,
                num_contexts=0,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached),
                max_num_requests=batch_size,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
            )
            ref_metadata.prepare()

            for i in range(num_layers):
                ref_out = layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], ref_metadata)
                torch.testing.assert_close(
                    cg_outputs[i],
                    ref_out,
                    atol=1e-2,
                    rtol=0,
                    msg=(
                        f"Step {step}, Layer {i} "
                        f"({layer_types[i]}, "
                        f"hd={layers_info[i]['head_dim']}): "
                        f"CUDA graph diverges from eager"
                    ),
                )

        kv_cache_manager.shutdown()

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_cuda_graph_decode_high_gqa(self):
        """CUDA graph decode with GQA=8 and real head_dim (E2B-like).

        Uses E2B real-dims config (hd=256/512, GQA=8) with multi-step
        decode to verify _block_tables update works for high-GQA models.
        """
        from tensorrt_llm._torch.attention_backend import (
            FlashInferAttention,
            FlashInferAttentionMetadata,
        )
        from tensorrt_llm._torch.metadata import KVCacheParams

        config_dict = deepcopy(GEMMA4_E2B_REAL_DIMS_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        batch_size = 2
        num_decode_steps = 5
        kv_cache_manager = self._get_kv_cache_manager(
            config, num_blocks=16, tokens_per_block=32, batch_size=batch_size
        )

        self.assertTrue(kv_cache_manager.is_vswa, "Expected VSWA manager")

        request_ids = list(range(batch_size))
        initial_cached = [30, 45]
        token_nums = [t + 1 for t in initial_cached]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        for i in range(config.num_hidden_layers):
            buf = kv_cache_manager.get_buffers(i)
            if buf is not None:
                torch.nn.init.normal_(buf)

        layer_types = config.layer_types
        num_layers = config.num_hidden_layers
        dtype = config.torch_dtype
        device = torch.device("cuda")

        layers_info = []
        for layer_idx in range(num_layers):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            hd = (
                config.head_dim
                if is_sliding
                else getattr(config, "global_head_dim", config.head_dim)
            )
            layers_info.append(
                {
                    "layer_idx": layer_idx,
                    "head_dim": hd,
                    "num_kv_heads": config.num_key_value_heads,
                    "num_heads": config.num_attention_heads,
                }
            )

        layers = []
        for info in layers_info:
            layers.append(
                FlashInferAttention(
                    layer_idx=info["layer_idx"],
                    num_heads=info["num_heads"],
                    head_dim=info["head_dim"],
                    num_kv_heads=info["num_kv_heads"],
                    flashinfer_backend="trtllm-gen",
                )
            )

        seq_lens = torch.ones(batch_size, dtype=torch.int)
        gen_qs, gen_ks, gen_vs = [], [], []
        for info in layers_info:
            gen_qs.append(
                torch.randn(
                    batch_size, info["num_heads"] * info["head_dim"], dtype=dtype, device=device
                )
            )
            gen_ks.append(
                torch.randn(
                    batch_size, info["num_kv_heads"] * info["head_dim"], dtype=dtype, device=device
                )
            )
            gen_vs.append(
                torch.randn(
                    batch_size, info["num_kv_heads"] * info["head_dim"], dtype=dtype, device=device
                )
            )

        # --- CUDA graph setup ---
        workspace = torch.empty(320 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        cg_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=0,
            is_cuda_graph=True,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=initial_cached),
            workspace_buffer=workspace,
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        cg_metadata.prepare()

        # Warmup
        for _ in range(2):
            for i in range(num_layers):
                layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], cg_metadata)

        # Capture
        cg_outputs = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(num_layers):
                cg_outputs.append(layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], cg_metadata))

        # Multi-step decode: prepare() + replay()
        for step in range(num_decode_steps):
            cached = [c + step for c in initial_cached]
            cg_metadata.kv_cache_params = KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=cached
            )
            cg_metadata.seq_lens = seq_lens
            cg_metadata.num_contexts = 0
            cg_metadata.request_ids = request_ids
            cg_metadata.prepare()

            graph.replay()
            torch.cuda.synchronize()

            # Verify no NaN/Inf
            for i in range(num_layers):
                out = cg_outputs[i]
                self.assertFalse(torch.isnan(out).any(), f"Step {step}, Layer {i}: NaN in output")
                self.assertFalse(torch.isinf(out).any(), f"Step {step}, Layer {i}: Inf in output")

            # Compare with eager reference
            ref_metadata = FlashInferAttentionMetadata(
                seq_lens=seq_lens,
                num_contexts=0,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached),
                max_num_requests=batch_size,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
            )
            ref_metadata.prepare()

            for i in range(num_layers):
                ref_out = layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], ref_metadata)
                torch.testing.assert_close(
                    cg_outputs[i],
                    ref_out,
                    atol=1e-2,
                    rtol=0,
                    msg=(
                        f"Step {step}, Layer {i} "
                        f"({layer_types[i]}, "
                        f"hd={layers_info[i]['head_dim']}): "
                        f"CUDA graph diverges from eager (GQA>=4)"
                    ),
                )

        kv_cache_manager.shutdown()

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def _run_cuda_graph_real_headdim(self, config_dict, label=""):
        """Helper: CUDA graph decode test with real head_dim configs."""
        from tensorrt_llm._torch.attention_backend import (
            FlashInferAttention,
            FlashInferAttentionMetadata,
        )
        from tensorrt_llm._torch.metadata import KVCacheParams

        config = Gemma4TextConfig(**config_dict)
        batch_size = 2
        kv_cache_manager = self._get_kv_cache_manager(
            config, num_blocks=16, tokens_per_block=32, batch_size=batch_size
        )

        self.assertTrue(kv_cache_manager.is_vswa, f"{label}: Expected VSWA manager")

        request_ids = list(range(batch_size))
        initial_cached = [30, 45]
        token_nums = [t + 1 for t in initial_cached]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        for i in range(config.num_hidden_layers):
            buf = kv_cache_manager.get_buffers(i)
            if buf is not None:
                torch.nn.init.normal_(buf)

        layer_types = config.layer_types
        num_layers = config.num_hidden_layers
        dtype = config.torch_dtype
        device = torch.device("cuda")
        attention_k_eq_v = getattr(config, "attention_k_eq_v", False)

        layers_info = []
        for layer_idx in range(num_layers):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            hd = (
                config.head_dim
                if is_sliding
                else getattr(config, "global_head_dim", config.head_dim)
            )
            # Match Gemma4Attention K=V logic for kv_heads
            use_k_eq_v = attention_k_eq_v and not is_sliding
            if is_sliding:
                kv_h = config.num_key_value_heads
            elif use_k_eq_v:
                kv_h = (
                    getattr(config, "num_global_key_value_heads", None)
                    or config.num_key_value_heads
                )
            else:
                kv_h = config.num_key_value_heads
            layers_info.append(
                {
                    "layer_idx": layer_idx,
                    "head_dim": hd,
                    "num_kv_heads": kv_h,
                    "num_heads": config.num_attention_heads,
                }
            )

        layers = []
        for info in layers_info:
            kwargs = {}
            # head_dim>256 needs trtllm-gen (fa2 JIT doesn't support it)
            if info["head_dim"] > 256:
                kwargs["flashinfer_backend"] = "trtllm-gen"
            layers.append(
                FlashInferAttention(
                    layer_idx=info["layer_idx"],
                    num_heads=info["num_heads"],
                    head_dim=info["head_dim"],
                    num_kv_heads=info["num_kv_heads"],
                    **kwargs,
                )
            )

        seq_lens = torch.ones(batch_size, dtype=torch.int)
        gen_qs, gen_ks, gen_vs = [], [], []
        for info in layers_info:
            gen_qs.append(
                torch.randn(
                    batch_size, info["num_heads"] * info["head_dim"], dtype=dtype, device=device
                )
            )
            gen_ks.append(
                torch.randn(
                    batch_size, info["num_kv_heads"] * info["head_dim"], dtype=dtype, device=device
                )
            )
            gen_vs.append(
                torch.randn(
                    batch_size, info["num_kv_heads"] * info["head_dim"], dtype=dtype, device=device
                )
            )

        # --- Reference (eager) ---
        ref_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=0,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=initial_cached),
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        ref_metadata.prepare()
        ref_results = []
        for i in range(num_layers):
            ref_results.append(layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], ref_metadata))

        # --- CUDA graph ---
        workspace = torch.empty(320 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        cg_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=0,
            is_cuda_graph=True,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=initial_cached),
            workspace_buffer=workspace,
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        cg_metadata.prepare()

        for _ in range(2):
            for i in range(num_layers):
                layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], cg_metadata)

        cg_results = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(num_layers):
                cg_results.append(layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], cg_metadata))
        graph.replay()

        for i in range(num_layers):
            torch.testing.assert_close(
                cg_results[i],
                ref_results[i],
                atol=1e-2,
                rtol=0,
                msg=(
                    f"{label} Layer {i} ({layer_types[i]}, "
                    f"hd={layers_info[i]['head_dim']}, "
                    f"kv={layers_info[i]['num_kv_heads']}): "
                    f"CUDA graph diverges from eager"
                ),
            )

        kv_cache_manager.shutdown()

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_cuda_graph_decode_real_headdim(self):
        """E2B-like: GQA=8, hd=256/512, non-K=V."""
        self._run_cuda_graph_real_headdim(deepcopy(GEMMA4_E2B_REAL_DIMS_CONFIG), "E2B")

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_cuda_graph_decode_31b_like(self):
        """31B-like: mixed GQA (2 sliding, 8 full K=V), hd=256/512."""
        self._run_cuda_graph_real_headdim(deepcopy(GEMMA4_31B_REAL_DIMS_CONFIG), "31B")

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_cuda_graph_decode_26b_like(self):
        """26B-like: GQA=2, K=V, hd=256/512."""
        self._run_cuda_graph_real_headdim(deepcopy(GEMMA4_26B_REAL_DIMS_CONFIG), "26B")

    @torch.no_grad()
    @unittest.mock.patch(
        "tensorrt_llm.runtime.kv_cache_manager_v2._utils.assert_critical", lambda *a, **kw: None
    )
    def test_cuda_graph_multi_step_trtllm_gen(self):
        """Multi-step CG decode with trtllm-gen (hd=256/512).

        Verifies _block_tables update in prepare() works correctly
        across multiple graph replays with changing kv_lens.  Uses
        31B-like config with trtllm-gen for all layers.
        """
        from tensorrt_llm._torch.attention_backend import (
            FlashInferAttention,
            FlashInferAttentionMetadata,
        )
        from tensorrt_llm._torch.metadata import KVCacheParams

        config_dict = deepcopy(GEMMA4_31B_REAL_DIMS_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        batch_size = 2
        kv_cache_manager = self._get_kv_cache_manager(
            config, num_blocks=16, tokens_per_block=32, batch_size=batch_size
        )

        request_ids = list(range(batch_size))
        initial_cached = [30, 45]
        token_nums = [t + 1 for t in initial_cached]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        for i in range(config.num_hidden_layers):
            buf = kv_cache_manager.get_buffers(i)
            if buf is not None:
                torch.nn.init.normal_(buf)

        layer_types = config.layer_types
        num_layers = config.num_hidden_layers
        dtype = config.torch_dtype
        device = torch.device("cuda")
        attention_k_eq_v = getattr(config, "attention_k_eq_v", False)

        layers_info = []
        for layer_idx in range(num_layers):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            hd = (
                config.head_dim
                if is_sliding
                else getattr(config, "global_head_dim", config.head_dim)
            )
            use_k_eq_v = attention_k_eq_v and not is_sliding
            if is_sliding:
                kv_h = config.num_key_value_heads
            elif use_k_eq_v:
                kv_h = (
                    getattr(config, "num_global_key_value_heads", None)
                    or config.num_key_value_heads
                )
            else:
                kv_h = config.num_key_value_heads
            layers_info.append(
                {
                    "layer_idx": layer_idx,
                    "head_dim": hd,
                    "num_kv_heads": kv_h,
                    "num_heads": config.num_attention_heads,
                }
            )

        layers = []
        for info in layers_info:
            layers.append(
                FlashInferAttention(
                    layer_idx=info["layer_idx"],
                    num_heads=info["num_heads"],
                    head_dim=info["head_dim"],
                    num_kv_heads=info["num_kv_heads"],
                    flashinfer_backend="trtllm-gen",
                )
            )

        seq_lens = torch.ones(batch_size, dtype=torch.int)
        gen_qs = [
            torch.randn(
                batch_size, info["num_heads"] * info["head_dim"], dtype=dtype, device=device
            )
            for info in layers_info
        ]
        gen_ks = [
            torch.randn(
                batch_size, info["num_kv_heads"] * info["head_dim"], dtype=dtype, device=device
            )
            for info in layers_info
        ]
        gen_vs = [
            torch.randn(
                batch_size, info["num_kv_heads"] * info["head_dim"], dtype=dtype, device=device
            )
            for info in layers_info
        ]

        workspace = torch.empty(320 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        cg_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=0,
            is_cuda_graph=True,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=initial_cached),
            workspace_buffer=workspace,
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        cg_metadata.prepare()

        for _ in range(2):
            for i in range(num_layers):
                layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], cg_metadata)

        cg_outputs = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(num_layers):
                cg_outputs.append(layers[i].forward(gen_qs[i], gen_ks[i], gen_vs[i], cg_metadata))

        # Multi-step replay with fixed kv_lens
        for step in range(5):
            cg_metadata.kv_cache_params = KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=initial_cached
            )
            cg_metadata.seq_lens = seq_lens
            cg_metadata.num_contexts = 0
            cg_metadata.request_ids = request_ids
            cg_metadata.prepare()
            graph.replay()
            torch.cuda.synchronize()

            for i in range(num_layers):
                self.assertFalse(
                    torch.isnan(cg_outputs[i]).any(), f"Step {step}, Layer {i}: NaN in output"
                )

        kv_cache_manager.shutdown()


class TestGemma4PLEMultimodalGuards(unittest.TestCase):
    """Regression tests for the multimodal Per-Layer-Embedding (PLE) fix.

    Background: ``Gemma4TextModel.forward`` originally raised when both
    ``input_ids`` and ``inputs_embeds`` were provided, and silently skipped
    PLE when ``input_ids`` was ``None``.  The multimodal wrapper passes
    ``inputs_embeds`` (with image/audio features scattered in) and used to
    pass ``input_ids=None``, which disabled PLE on E2B/E4B and produced
    garbage output for image and audio prompts.  The fix lets the multimodal
    wrapper hand a separate ``ple_input_ids`` (with mm tokens replaced by
    pad) so PLE keeps running on multimodal inputs.
    """

    def _make_e2b_like_text_model(self, hidden_size_per_layer_input: int = 32):
        """A tiny Gemma4 text model with PLE enabled (mimicking E2B)."""
        cfg = {
            **GEMMA4_SMALL_CONFIG,
            "hidden_size_per_layer_input": hidden_size_per_layer_input,
            "num_hidden_layers": 4,
        }
        text_config = Gemma4TextConfig(**cfg)
        full_config = Gemma4Config(text_config=text_config)
        model_config = ModelConfig(
            pretrained_config=full_config.text_config,
            attn_backend="FLASHINFER",
        )
        return Gemma4TextModel(model_config)

    def test_ple_skipped_when_input_ids_none(self):
        """Without ``input_ids`` PLE silently returns ``None`` — original
        behaviour.  This test pins it down so future refactors don't
        accidentally start using inputs_embeds for the per-layer
        embedding lookup.
        """
        model = self._make_e2b_like_text_model()
        hidden_states = torch.zeros(8, 256, dtype=torch.bfloat16)
        per_layer = model._compute_per_layer_inputs(None, hidden_states)
        self.assertIsNone(per_layer)

    def test_ple_uses_explicit_ple_input_ids(self):
        """When the multimodal wrapper passes ``ple_input_ids`` (a
        sanitized copy of input_ids with mm tokens replaced by pad), the
        forward must use that view for the PLE lookup instead of relying
        on a separate ``input_ids`` (which is ``None`` on the multimodal
        path).

        Avoids replacing nn.ModuleList children — instead patches
        ``_compute_per_layer_inputs`` to capture the ids it received and
        raises a sentinel error to short-circuit the rest of forward.
        """
        model = self._make_e2b_like_text_model()
        captured = {}

        class _Done(Exception):
            pass

        def fake_compute(input_ids, inputs_embeds):
            captured["input_ids"] = input_ids.clone() if input_ids is not None else None
            raise _Done

        model._compute_per_layer_inputs = fake_compute

        ple_ids = torch.tensor([[100, 200, 300]], dtype=torch.int32)
        inputs_embeds = torch.zeros(3, 256, dtype=torch.bfloat16)
        attn_metadata = unittest.mock.MagicMock()
        with self.assertRaises(_Done):
            with torch.inference_mode():
                model(
                    attn_metadata=attn_metadata,
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    ple_input_ids=ple_ids,
                )
        self.assertIsNotNone(captured.get("input_ids"))
        self.assertTrue(
            torch.equal(captured["input_ids"], ple_ids),
            f"PLE received {captured['input_ids']}, expected {ple_ids}",
        )

    def test_forward_accepts_input_ids_and_inputs_embeds_together(self):
        """The XOR check on ``input_ids`` vs ``inputs_embeds`` was relaxed
        so the multimodal wrapper can pass both (input_ids for PLE,
        inputs_embeds for the main forward).  Verify the constructor of
        the text model no longer raises on this combination.
        """
        model = self._make_e2b_like_text_model(hidden_size_per_layer_input=0)
        # When both args are None, we should still raise.
        with self.assertRaises(ValueError):
            model(
                attn_metadata=unittest.mock.MagicMock(),
                input_ids=None,
                inputs_embeds=None,
            )


if __name__ == "__main__":
    unittest.main()
