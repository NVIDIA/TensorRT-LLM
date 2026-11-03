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

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import PretrainedConfig

from tensorrt_llm._torch.models.modeling_kimi_k25 import _vision_requires_replication
from tensorrt_llm._torch.pyexecutor.config_utils import (
    is_kimi_k3_multimodal_config,
    load_pretrained_config,
)


def _composite_config():
    """Minimal shape of the released K3 VL checkpoint config."""
    return {
        "model_type": "kimi_k3",
        "text_config": {"model_type": "kimi_linear", "hidden_size": 7168},
        "vision_config": {"vt_num_attention_heads": 12, "vt_hidden_size": 1024},
    }


def _k31_composite_config():
    """Minimal shape of the K3.1 VL checkpoint config: K3 plus a DSA indexer."""
    cfg = _composite_config()
    cfg["model_type"] = "kimi_k31"
    cfg["text_config"]["dsa_config"] = {
        "dsa_layers": [4, 8],
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_num_heads": 32,
        "index_rope_dim": 0,
        "index_k_norm_type": "layernorm",
        "indexer_use_fp8": False,
        "index_topk_freq": 1,
        "index_skip_topk_offset": 1,
    }
    return cfg


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

    def test_k31_composite_vlm_config_routes_multimodal(self):
        self.assertTrue(is_kimi_k3_multimodal_config(_k31_composite_config()))


def _vision_model_config(tp_size, enable_attention_dp, cp_size=1):
    """Minimal stand-in exposing the mapping fields the predicate reads."""
    return SimpleNamespace(
        mapping=SimpleNamespace(
            tp_size=tp_size, enable_attention_dp=enable_attention_dp, cp_size=cp_size
        )
    )


class TestVisionRequiresReplication(unittest.TestCase):
    """Pin the (mapping, num_heads) contract of _vision_requires_replication.

    The predicate gates whether the MoonViT tower TP-shards or runs replicated
    (tp=1); a silent change to the divisibility rule or the attention-DP
    short-circuit would otherwise only fail at multi-GPU runtime.
    """

    def test_k3_12_heads_under_tp16_requires_replication(self):
        self.assertTrue(_vision_requires_replication(_vision_model_config(16, False), num_heads=12))

    def test_k25_16_heads_under_tp8_shards(self):
        self.assertFalse(_vision_requires_replication(_vision_model_config(8, False), num_heads=16))

    def test_attention_dp_always_replicates(self):
        self.assertTrue(_vision_requires_replication(_vision_model_config(16, True), num_heads=16))

    def test_helix_cp_requires_replication(self):
        # Helix carries its parallelism in cp with tp_size=1; the tower has no
        # context-parallel form, so any cp > 1 must replicate even when
        # num_heads % tp_size == 0.
        self.assertTrue(
            _vision_requires_replication(_vision_model_config(1, False, cp_size=8), num_heads=12)
        )


def _load_config_from_dict(cfg: dict[str, Any]) -> PretrainedConfig:
    """Round-trip a raw config dict through load_pretrained_config."""
    with tempfile.TemporaryDirectory() as model_dir:
        (Path(model_dir) / "config.json").write_text(json.dumps(cfg))
        return load_pretrained_config(model_dir)


class TestKimiK3LoaderRouting(unittest.TestCase):
    """End-to-end config.json -> load_pretrained_config routing.

    Complements the pure-predicate tests above: these exercise the loader
    branch itself — composite KimiK3Config construction, the architecture
    assignment, and the text-only flatten — so a change to the branch (not
    just the predicate) fails in unit CI.
    """

    def test_composite_checkpoint_loads_as_vlm(self) -> None:
        config = _load_config_from_dict(_composite_config())
        self.assertEqual(config.architectures, ["KimiK3ForConditionalGeneration"])
        self.assertEqual(config.model_type, "kimi_k3")
        self.assertEqual(config.text_config.model_type, "kimi_linear")
        self.assertIsNotNone(config.vision_config)
        self.assertEqual(config.vision_config.vt_num_attention_heads, 12)

    def test_language_model_only_checkpoint_flattens_to_text(self) -> None:
        cfg = _composite_config()
        cfg["language_model_only"] = True
        config = _load_config_from_dict(cfg)
        self.assertEqual(config.architectures, ["KimiLinearForCausalLM"])
        self.assertEqual(config.model_type, "kimi_linear")

    def test_text_only_kimi_linear_checkpoint_flattens(self) -> None:
        config = _load_config_from_dict({"model_type": "kimi_linear", "hidden_size": 7168})
        self.assertEqual(config.architectures, ["KimiLinearForCausalLM"])
        self.assertEqual(config.model_type, "kimi_linear")
        self.assertEqual(config.hidden_size, 7168)
        self.assertIsNone(config.dsa_config)
        self.assertFalse(config.is_dsa_layer(3))

    def test_k31_composite_checkpoint_loads_as_vlm(self) -> None:
        config = _load_config_from_dict(_k31_composite_config())
        self.assertEqual(config.architectures, ["KimiK31ForConditionalGeneration"])
        self.assertEqual(config.model_type, "kimi_k31")
        text_config = config.text_config
        self.assertEqual(text_config.model_type, "kimi_linear")
        # dsa_layers is 1-indexed; is_dsa_layer takes 0-indexed layers.
        self.assertTrue(text_config.is_dsa_layer(3))
        self.assertTrue(text_config.is_dsa_layer(7))
        self.assertFalse(text_config.is_dsa_layer(4))
        self.assertEqual(text_config.index_n_heads, 32)
        self.assertEqual(text_config.index_head_dim, 128)
        self.assertEqual(text_config.index_topk, 2048)
        self.assertEqual(text_config.index_rope_dim, 0)
        self.assertEqual(text_config.index_topk_freq, 1)
        # The indexer key LayerNorm uses the model's norm epsilon.
        self.assertEqual(text_config.index_k_norm_eps, text_config.rms_norm_eps)

    def test_k31_rejects_rmsnorm_index_key_norm(self) -> None:
        cfg = _k31_composite_config()
        cfg["text_config"]["dsa_config"]["index_k_norm_type"] = "rmsnorm"
        with self.assertRaisesRegex(NotImplementedError, "index_k_norm_type"):
            _load_config_from_dict(cfg)

    def test_k31_language_model_only_checkpoint_flattens_to_text(self) -> None:
        cfg = _k31_composite_config()
        cfg["language_model_only"] = True
        config = _load_config_from_dict(cfg)
        self.assertEqual(config.architectures, ["KimiLinearForCausalLM"])
        self.assertEqual(config.model_type, "kimi_linear")
        self.assertEqual(config.index_topk, 2048)

    def test_only_k31_keeps_kda_gate_params_in_fp32(self) -> None:
        """Kimi K3 rounds the KDA A_log / dt_bias to bf16; Kimi K3.1 keeps fp32."""
        k3 = _load_config_from_dict(_composite_config())
        self.assertFalse(k3.text_config.kda_fp32_gate_params)
        k3_text = _composite_config()
        k3_text["language_model_only"] = True
        self.assertFalse(_load_config_from_dict(k3_text).kda_fp32_gate_params)
        self.assertFalse(_load_config_from_dict({"model_type": "kimi_linear"}).kda_fp32_gate_params)

        k31 = _load_config_from_dict(_k31_composite_config())
        self.assertTrue(k31.text_config.kda_fp32_gate_params)
        k31_text = _k31_composite_config()
        k31_text["language_model_only"] = True
        self.assertTrue(_load_config_from_dict(k31_text).kda_fp32_gate_params)


class TestKimiK31SparseAttentionConfig(unittest.TestCase):
    """The DSA config built for K3.1 keeps every user-set field."""

    def test_user_sparse_fields_survive_the_rebuild(self) -> None:
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

        requested = DeepSeekSparseAttentionConfig(
            enable_heuristic_topk=True,
            use_self_sampling_topk=True,
            gvr_deterministic_topk=True,
            seq_len_threshold=4096,
        )
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "config.json").write_text(json.dumps(_k31_composite_config()))
            model_config = ModelConfig.from_pretrained(
                model_dir, trust_remote_code=True, sparse_attention_config=requested
            )
        sparse = model_config.sparse_attention_config
        self.assertTrue(sparse.gvr_deterministic_topk)
        self.assertEqual(sparse.seq_len_threshold, 4096)
        self.assertTrue(sparse.enable_heuristic_topk)
        # Checkpoint-derived geometry is still filled in.
        self.assertEqual(sparse.index_n_heads, 32)
        self.assertEqual(sparse.index_topk, 2048)

    def _from_checkpoint(self, **kwargs):
        from tensorrt_llm._torch.model_config import ModelConfig

        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "config.json").write_text(json.dumps(_k31_composite_config()))
            return ModelConfig.from_pretrained(model_dir, trust_remote_code=True, **kwargs)

    def test_fp8_index_scores_are_the_default(self) -> None:
        # The K3.1 reference scores the indexer in FP8 like DeepSeek-V3.2; the
        # checkpoint's indexer_use_fp8 flag does not change that.
        sparse = self._from_checkpoint().sparse_attention_config
        self.assertEqual(sparse.indexer_k_dtype, "fp8")
        self.assertTrue(sparse.to_sparse_params().use_cute_dsl_paged_mqa_logits)

    def test_bf16_index_scores_are_opt_in(self) -> None:
        from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

        requested = DeepSeekSparseAttentionConfig(indexer_k_dtype="bf16")
        sparse = self._from_checkpoint(sparse_attention_config=requested).sparse_attention_config
        self.assertEqual(sparse.indexer_k_dtype, "bf16")
        params = sparse.to_sparse_params()
        self.assertEqual(params.indexer_k_dtype, "bf16")
        # The CuTe DSL paged MQA logits kernels read FP8/FP4 keys only.
        self.assertFalse(params.use_cute_dsl_paged_mqa_logits)
        self.assertFalse(sparse.to_sparse_metadata_params().use_cute_dsl_paged_mqa_logits)


if __name__ == "__main__":
    unittest.main()
