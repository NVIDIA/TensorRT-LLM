# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Routing-predicate tests for the Kimi K3 checkpoint family.

``is_kimi_k3_multimodal_config`` decides, for every ``model_type: kimi_k3``
checkpoint, whether TRT-LLM keeps the composite VLM config (and routes to
``KimiK3ForConditionalGeneration``) or flattens to the text-only
``KimiLinearConfig`` path. These cases pin that contract so a field rename in
a released ``config.json`` fails loudly here instead of silently rerouting
the checkpoint.
"""

import unittest

from tensorrt_llm._torch.pyexecutor.config_utils import is_kimi_k3_multimodal_config


def _composite_config():
    """Minimal shape of the released K3 VL checkpoint config."""
    return {
        "model_type": "kimi_k3",
        "text_config": {"model_type": "kimi_linear", "hidden_size": 7168},
        "vision_config": {"vt_num_attention_heads": 12, "vt_hidden_size": 1024},
    }


class TestIsKimiK3MultimodalConfig(unittest.TestCase):
    def test_composite_vlm_config_routes_multimodal(self):
        self.assertTrue(is_kimi_k3_multimodal_config(_composite_config()))

    def test_language_model_only_opts_out(self):
        cfg = _composite_config()
        cfg["language_model_only"] = True
        self.assertFalse(is_kimi_k3_multimodal_config(cfg))

    def test_language_model_only_false_stays_multimodal(self):
        cfg = _composite_config()
        cfg["language_model_only"] = False
        self.assertTrue(is_kimi_k3_multimodal_config(cfg))

    def test_missing_vision_config_is_text_only(self):
        cfg = _composite_config()
        del cfg["vision_config"]
        self.assertFalse(is_kimi_k3_multimodal_config(cfg))

    def test_missing_text_config_is_not_multimodal(self):
        cfg = _composite_config()
        del cfg["text_config"]
        self.assertFalse(is_kimi_k3_multimodal_config(cfg))

    def test_empty_dict_subconfigs_are_text_only(self):
        cfg = _composite_config()
        cfg["vision_config"] = {}
        self.assertFalse(is_kimi_k3_multimodal_config(cfg))

        cfg = _composite_config()
        cfg["text_config"] = {}
        self.assertFalse(is_kimi_k3_multimodal_config(cfg))

    def test_non_dict_subconfigs_are_text_only(self):
        cfg = _composite_config()
        cfg["vision_config"] = "not-a-dict"
        self.assertFalse(is_kimi_k3_multimodal_config(cfg))

    def test_text_only_kimi_linear_checkpoint(self):
        self.assertFalse(
            is_kimi_k3_multimodal_config({"model_type": "kimi_linear", "hidden_size": 7168})
        )

    def test_other_model_type_with_subconfigs(self):
        cfg = _composite_config()
        cfg["model_type"] = "qwen3_5"
        self.assertFalse(is_kimi_k3_multimodal_config(cfg))


if __name__ == "__main__":
    unittest.main()
