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

import unittest

import torch
from transformers import Gemma2Config
from transformers import Gemma2ForCausalLM as HFGemma2ForCausalLM

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.gemma2_weight_mapper import \
    Gemma2HfWeightMapper
from tensorrt_llm._torch.models.modeling_gemma2 import Gemma2ForCausalLM
from tensorrt_llm.mapping import Mapping

# Small Gemma2 config suitable for unit testing (no GPU weights needed)
GEMMA2_SMALL_CONFIG = {
    "architectures": ["Gemma2ForCausalLM"],
    "attn_logit_softcapping": 50.0,
    "final_logit_softcapping": 30.0,
    "head_dim": 64,
    "hidden_size": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "intermediate_size": 512,
    "max_position_embeddings": 128,
    "model_type": "gemma2",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "query_pre_attn_scalar": 64,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {"rope_theta": 10000.0, "rope_type": "default"},
    "sliding_window": 32,
    "layer_types": [
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
    ],
    "torch_dtype": "bfloat16",
    "vocab_size": 1024,
    "tie_word_embeddings": True,
}


def _build_hf_config(config_dict):
    config = Gemma2Config(**{
        k: v
        for k, v in config_dict.items()
        if k not in ("architectures", "torch_dtype")
    })
    config.torch_dtype = torch.bfloat16
    return config


class TestGemma2ForCausalLM(unittest.TestCase):

    def _build_model(self, config_dict: dict):
        config = _build_hf_config(config_dict)
        mapping = Mapping(world_size=1, tp_size=1)
        model_config = ModelConfig(pretrained_config=config, mapping=mapping)
        return Gemma2ForCausalLM(model_config)

    def test_model_instantiation(self):
        model = self._build_model(GEMMA2_SMALL_CONFIG)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, Gemma2ForCausalLM)

    def test_alternating_attention_window(self):
        """layer_types: [sliding, full, sliding, full]"""
        model = self._build_model(GEMMA2_SMALL_CONFIG)
        layers = model.model.layers
        self.assertIsNotNone(layers[0].self_attn.attention_window_size)
        self.assertIsNone(layers[1].self_attn.attention_window_size)
        self.assertIsNotNone(layers[2].self_attn.attention_window_size)
        self.assertIsNone(layers[3].self_attn.attention_window_size)

    def test_softcap_stored(self):
        """attn_logit_softcapping and final_logit_softcapping are stored."""
        model = self._build_model(GEMMA2_SMALL_CONFIG)
        for layer in model.model.layers:
            self.assertEqual(layer.self_attn.logits_soft_cap, 50.0)
        self.assertEqual(model._final_logit_softcap, 30.0)

    def test_weight_loading_from_hf(self):
        """Weights load correctly and 1P RMSNorm (+1) correction is applied."""
        hf_config = _build_hf_config(GEMMA2_SMALL_CONFIG)
        hf_model = HFGemma2ForCausalLM(hf_config).eval()
        trtllm_model = self._build_model(GEMMA2_SMALL_CONFIG)
        mapper = Gemma2HfWeightMapper(trtllm_model, "HF")

        weights = dict(hf_model.state_dict())
        trtllm_model.load_weights(weights, mapper)

        # Verify 1P RMSNorm: TRT-LLM weight = HF weight + 1.0
        hf_norm_w = weights["model.layers.0.input_layernorm.weight"]
        trt_norm_w = trtllm_model.model.layers[
            0].input_layernorm.weight.data.float()
        self.assertTrue(
            torch.allclose(trt_norm_w, hf_norm_w.float() + 1.0),
            "1P RMSNorm +1 correction not applied correctly",
        )

        # Verify a non-norm weight (embed_tokens) is copied as-is
        hf_embed = weights["model.embed_tokens.weight"]
        trt_embed = trtllm_model.model.embed_tokens.weight.data
        self.assertTrue(
            torch.allclose(trt_embed.float(), hf_embed.float()),
            "embed_tokens weight mismatch",
        )

    def test_rope_params_default_type(self):
        """RopeParams.from_config must not raise for rope_type='default'."""
        from tensorrt_llm._torch.attention_backend.interface import RopeParams
        config = Gemma2Config()
        # Explicitly set to verify the "default" alias branch is exercised.
        config.rope_scaling = {"rope_theta": 10000.0, "rope_type": "default"}
        rope_params = RopeParams.from_config(config)
        self.assertIsNotNone(rope_params)

    def test_embed_scale_is_float(self):
        """embed_scale must be a Python float to avoid CPU/GPU device mismatch."""
        model = self._build_model(GEMMA2_SMALL_CONFIG)
        self.assertIsInstance(model.model.embed_tokens.embed_scale, float)


class TestGemma2AttentionSliding(unittest.TestCase):

    def test_sliding_layer_has_window(self):
        config = _build_hf_config(GEMMA2_SMALL_CONFIG)
        mapping = Mapping(world_size=1, tp_size=1)
        model_config = ModelConfig(pretrained_config=config, mapping=mapping)

        from tensorrt_llm._torch.models.modeling_gemma2 import Gemma2Attention
        attn_sliding = Gemma2Attention(model_config, layer_idx=0,
                                       is_sliding=True)
        attn_full = Gemma2Attention(model_config, layer_idx=1, is_sliding=False)

        self.assertEqual(attn_sliding.attention_window_size,
                         config.sliding_window)
        self.assertIsNone(attn_full.attention_window_size)


class TestPaliGemmaSmoke(unittest.TestCase):
    """Narrow smoke tests for PaliGemmaForConditionalGeneration."""

    def _build_paligemma_model(self):
        from transformers import PaliGemmaConfig, SiglipVisionConfig
        from tensorrt_llm._torch.models.modeling_paligemma import \
            PaliGemmaForConditionalGeneration

        vision_config = SiglipVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=56,
            patch_size=14,
            vision_use_head=False,
        )
        text_config = _build_hf_config(GEMMA2_SMALL_CONFIG)
        text_config.model_type = "gemma2"

        config = PaliGemmaConfig(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=text_config.hidden_size,
        )
        config.torch_dtype = torch.bfloat16

        mapping = Mapping(world_size=1, tp_size=1)
        model_config = ModelConfig(pretrained_config=config, mapping=mapping)
        return PaliGemmaForConditionalGeneration(model_config)

    def test_paligemma_instantiation(self):
        """Model creates with expected sub-components."""
        model = self._build_paligemma_model()
        self.assertTrue(hasattr(model, "llm"))
        self.assertTrue(hasattr(model, "siglip_tower"))
        self.assertTrue(hasattr(model, "mm_projector"))

    def test_image_token_ids_is_buffer(self):
        """image_token_ids must be a registered buffer, not a plain tensor."""
        from tensorrt_llm._torch.models.modeling_paligemma import \
            PaliGemmaForConditionalGeneration
        model = self._build_paligemma_model()
        buffer_names = [name for name, _ in model.named_buffers()]
        self.assertIn("image_token_ids", buffer_names,
                      "image_token_ids must be a registered buffer")

    def test_mm_projector_shape(self):
        """Projector linear weight has expected shape."""
        model = self._build_paligemma_model()
        w = model.mm_projector.linear.weight
        vision_hidden = model.siglip_tower.config.hidden_size
        text_hidden = model.llm.config.hidden_size
        self.assertEqual(w.shape, (text_hidden, vision_hidden))


if __name__ == "__main__":
    unittest.main()
