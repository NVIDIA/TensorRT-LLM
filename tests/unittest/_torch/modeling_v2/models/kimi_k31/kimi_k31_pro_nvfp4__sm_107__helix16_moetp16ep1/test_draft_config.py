# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``draft_config_for`` reads the served DFlash2 drafter's own ``config.json``, not a stand-in.

The checkpoint's config is copied below byte for byte (Kimi-K3.1-DFlash2-Preview, revision
d0d8bc2, sha256 b59ceb23...537a). A hand-built config skips the transformers loader, so it once
hid a failure there: a bare ``PretrainedConfig.from_dict`` standardizes this YaRN RoPE before
``max_position_embeddings`` is set and raises. No GPU: the stubs carry only what
``draft_config_for`` reads from the target's ``ModelConfig``.
"""

import hashlib
import json
from importlib import import_module
from types import SimpleNamespace

import pytest

_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
)

_CONFIG_JSON = """{
  "architectures": [
    "DFlash2DraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "base_model": "moonshotai/Kimi-K3.1-Pro",
  "base_model_revision": "535833b1b6099ea85659e047561ed0f22ab38db1",
  "block_size": 5,
  "bos_token_id": 163584,
  "dflash_config": {
    "attention_mode": "mla",
    "aux_hidden_stream": "attn_res",
    "block_size": 5,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 163824,
    "selector_rank": 256,
    "selector_top_k": 32,
    "target_layer_ids": [
      7,
      23,
      51,
      67,
      83,
      91
    ]
  },
  "draft_vocab_size": 163840,
  "dtype": "bfloat16",
  "eos_token_id": 163586,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 7168,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "is_causal": true,
  "kv_lora_rank": 512,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "max_position_embeddings": 1048576,
  "max_window_layers": 5,
  "mla_use_nope": false,
  "model_type": "qwen3",
  "num_attention_heads": 96,
  "num_hidden_layers": 5,
  "num_key_value_heads": 96,
  "num_target_layers": 93,
  "pad_token_id": 163839,
  "q_lora_rank": 1536,
  "qk_head_dim": 192,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-05,
  "rope_interleave": true,
  "rope_parameters": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 32.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 50000.0,
    "rope_type": "yarn"
  },
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 32.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 32768,
    "rope_type": "yarn"
  },
  "rope_theta": 50000.0,
  "sliding_window": 1024,
  "tie_word_embeddings": false,
  "transformers_version": "5.19.0",
  "use_cache": true,
  "use_sliding_window": true,
  "v_head_dim": 128,
  "vocab_size": 163840
}"""
_CONFIG_SHA256 = "b59ceb231cbec61f287968de7ecdb8520c786f0a813088b7636753fd423e537a"


class _Mode:
    def use_one_engine(self) -> bool:
        return True

    def is_dflash(self) -> bool:
        return True


def _target_config(drafter_dir) -> SimpleNamespace:
    tp_mapping = SimpleNamespace(name="tp16")
    return SimpleNamespace(
        spec_config=SimpleNamespace(spec_dec_mode=_Mode(), speculative_model=str(drafter_dir)),
        mapping=SimpleNamespace(repurpose_helix_cp_to_tp=lambda: tp_mapping),
        quant_config=None,
        quant_config_dict={"target": "only"},
        pretrained_config=None,
        _frozen=True,
    )


@pytest.fixture
def drafter_dir(tmp_path):
    (tmp_path / "config.json").write_text(_CONFIG_JSON)
    return tmp_path


def test_the_copy_is_the_checkpoints_config() -> None:
    assert hashlib.sha256(_CONFIG_JSON.encode()).hexdigest() == _CONFIG_SHA256


def test_draft_config_reads_the_served_drafter_config(drafter_dir) -> None:
    modeling = import_module(f"{_PKG}.modeling")
    weights = import_module(f"{_PKG}.weights")
    target = _target_config(drafter_dir)
    draft = modeling.draft_config_for(target)
    cfg = draft.pretrained_config
    for key, want in weights.DRAFT_SHAPE.items():
        assert getattr(cfg, key) == want, key
    source = json.loads(_CONFIG_JSON)["dflash_config"]
    assert cfg.dflash_config == source
    assert cfg.dflash_config["block_size"] == 5
    assert cfg.dflash_config["mask_token_id"] == 163824
    assert cfg.dflash_config["target_layer_ids"] == [7, 23, 51, 67, 83, 91]
    assert cfg.layer_types == ["sliding_attention"] * 5
    assert cfg.max_position_embeddings == 1048576
    # What the target keeps of its own config, and what it replaces.
    assert draft.mapping.name == "tp16"
    assert draft.spec_config is None and draft.quant_config_dict is None
    assert draft.quant_config.kv_cache_quant_algo == modeling.QuantAlgo.FP8
    assert target.pretrained_config is None and target.spec_config is not None


def test_draft_rope_is_the_checkpoints_yarn(drafter_dir) -> None:
    modeling = import_module(f"{_PKG}.modeling")
    cfg = modeling.draft_config_for(_target_config(drafter_dir)).pretrained_config
    assert modeling._draft_rope(cfg) == {
        "base": 50000.0,
        "scaling_factor": 32.0,
        "original_max_position_embeddings": 32768,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
