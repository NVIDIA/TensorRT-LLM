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
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _torch.attention.multi_gpu.helix_test_utils import CACHE_TYPE_SELFKONLY, setup_kv_and_metadata
from transformers import PretrainedConfig

from tensorrt_llm._torch.configs.kimi_k3 import KimiK3Config
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_k3_vl import (
    KimiK3ForConditionalGeneration,
    KimiK3VisionModel,
)
from tensorrt_llm._torch.models.modeling_kimi_k25 import _vision_requires_replication
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
from tensorrt_llm._torch.pyexecutor.config_utils import (
    is_kimi_k3_multimodal_config,
    load_pretrained_config,
)
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.inputs.multimodal import MultimodalParams


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_kimi_k3_forward_with_attached_embeddings():
    text_config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        attn_res_block_size=1,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=128,
        torch_dtype=torch.bfloat16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        linear_attn_config={
            "kda_layers": [],
            "full_attn_layers": [1],
            "num_heads": 32,
        },
    )
    config = KimiK3Config(
        architectures=["KimiK3ForConditionalGeneration"],
        text_config=text_config,
        vision_config={
            "vt_hidden_size": 128,
            "vt_num_hidden_layers": 1,
            "vt_num_attention_heads": 1,
            "qkv_hidden_size": 128,
            "vt_intermediate_size": 256,
            "text_hidden_size": 256,
            "patch_size": 2,
            "init_pos_emb_height": 2,
            "init_pos_emb_width": 2,
            "init_pos_emb_time": 1,
            "media_placeholder_token_id": 127,
            "pad_token_id": 0,
        },
        media_placeholder_token_id=127,
        pad_token_id=0,
        torch_dtype=torch.bfloat16,
    )
    model_config = ModelConfig(
        pretrained_config=config,
        attn_backend="TRTLLM",
        max_num_tokens=4,
        max_seq_len=128,
    )
    model_cls = get_registered_model_class("KimiK3ForConditionalGeneration")
    assert model_cls is KimiK3ForConditionalGeneration
    with torch.device("cuda"):
        model = model_cls(model_config).eval()
    assert isinstance(model.mm_encoder, KimiK3VisionModel)
    assert isinstance(model.llm, KimiLinearForCausalLM)
    assert len(model.llm.model.layers) == 1

    generator = torch.Generator(device="cpu").manual_seed(147)
    for parameter in model.parameters():
        if parameter.ndim == 1:
            parameter.fill_(1.0)
        else:
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.05
            )

    # Native MLA requires a cache pool even when recomputing the entire context.
    scenario = SimpleNamespace(
        ctx_len=4,
        batch=1,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=torch.bfloat16,
    )
    manager, metadata = setup_kv_and_metadata(
        scenario,
        model_config.mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=text_config.kv_lora_rank + text_config.qk_rope_head_dim,
    )
    try:
        extra_attrs = dict(model_config.extra_attrs)
        extra_attrs["attention_metadata"] = weakref.ref(metadata)
        input_ids = torch.tensor([3, 4, 5, 6], dtype=torch.int32, device="cuda")
        media_ids = input_ids.clone()
        media_ids[1] = 127
        params = MultimodalParams(
            multimodal_data={
                "multimodal_embedding": model.llm.model.embed_tokens(input_ids[1:2]).clone()
            }
        )

        with model_extra_attrs(extra_attrs):
            metadata.prepare()
            expected = model(metadata, input_ids=input_ids, return_context_logits=True)
            changed = model(metadata, input_ids=media_ids, return_context_logits=True)
            actual = model(
                metadata,
                input_ids=media_ids,
                return_context_logits=True,
                multimodal_params=[params],
                mm_token_indices=torch.tensor([1], device="cuda"),
                text_token_indices=torch.tensor([0, 2, 3], device="cuda"),
            )

        for logits in (expected, changed, actual):
            assert logits.shape == (4, text_config.vocab_size)
            assert logits.is_cuda and logits.dtype == torch.float32
            assert torch.isfinite(logits).all()
        assert not torch.allclose(changed[-1], expected[-1], atol=0.005, rtol=0.03)
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.03)
    finally:
        manager.shutdown()


if __name__ == "__main__":
    unittest.main()
